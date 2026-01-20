# models/pipeline.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler
from .audio_encoder import AudioEncoder
from .storyboard_unet import StoryboardUNet

class AudioToStoryboardPipeline(nn.Module):
    """
    Audio-to-Storyboard Generation Pipeline
    """
    
    def __init__(
        self,
        pretrained_model: str = "runwayml/stable-diffusion-v1-5",
        audio_encoder_config: dict = None,
        freeze_unet: bool = True
    ):
        super().__init__()
        
        if audio_encoder_config is None:
            audio_encoder_config = {
                'mel_channels': 128,
                'hidden_dim': 512,
                'output_dim': 768,
                'num_layers': 6,
                'num_heads': 8,
                'output_seq_len': 77  # CLIP과 동일
            }
        
        # 1. Audio Encoder
        self.audio_encoder = AudioEncoder(**audio_encoder_config)
        
        # 2. U-Net
        self.storyboard_unet = StoryboardUNet(
            pretrained_model=pretrained_model,
            freeze_unet=freeze_unet
        )
        
        # 3. Scheduler - DDIM 사용 (SD v1.5와 호환)
        self.scheduler = DDIMScheduler.from_pretrained(
            pretrained_model,
            subfolder="scheduler"
        )
        
        # 4. Learnable null embedding (CFG용)
        self.null_audio_embed = nn.Parameter(torch.randn(1, 77, 768) * 0.02)
        self.null_text_embed = nn.Parameter(torch.randn(1, 77, 768) * 0.02)
        
        # 5. VAE (추론 시 lazy loading)
        self.vae = None
        self.pretrained_model = pretrained_model
        
        # 6. Latent scaling factor
        self.vae_scale_factor = 0.18215
    
    
    def forward(
        self,
        mel: torch.Tensor = None,
        latent: torch.Tensor = None,
        text_embed: torch.Tensor = None,
        mel_mask: torch.Tensor = None,
        conditioning_mode: str = "audio"  # "audio", "text", "both"
    ) -> dict:
        """
        학습용 Forward pass
        
        Args:
            conditioning_mode: "audio", "text", or "both"
        """
        # Determine batch size and device from available inputs
        if mel is not None:
            B = mel.shape[0]
            device = mel.device
        elif text_embed is not None:
            B = text_embed.shape[0]
            device = text_embed.device
        else:
            raise ValueError("At least one of mel or text_embed must be provided")
        
        # 1. Audio encoding (if needed)
        audio_embeds = None
        if conditioning_mode in ["audio", "both"]:
            if mel is None:
                raise ValueError(f"mel required for '{conditioning_mode}' mode")
            audio_embeds = self.audio_encoder(mel, mel_mask)
        
        # 2. Storyboard latent 준비
        storyboard_latent = self._merge_latents(latent)
        
        # 3. Random timestep
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (B,), device=device, dtype=torch.long
        )
        
        # 4. Noise 추가
        noise = torch.randn_like(storyboard_latent)
        noisy_latent = self.scheduler.add_noise(storyboard_latent, noise, timesteps)
        
        # 5. Noise 예측
        noise_pred = self.storyboard_unet(
            sample=noisy_latent,
            timestep=timesteps,
            audio_embeds=audio_embeds,
            text_embeds=text_embed,
            conditioning_mode=conditioning_mode
        )
        
        # 6. Loss
        loss = F.mse_loss(noise_pred, noise)
        
        return {
            'loss': loss,
            'noise_pred': noise_pred,
            'noise': noise,
            'timesteps': timesteps
        }
    
    def _merge_latents(self, latent: torch.Tensor) -> torch.Tensor:
        """[B, 4, 4, 64, 64] → [B, 4, 128, 128]"""
        top = torch.cat([latent[:, 0], latent[:, 1]], dim=-1)
        bottom = torch.cat([latent[:, 2], latent[:, 3]], dim=-1)
        return torch.cat([top, bottom], dim=-2)
    
    def _split_latents(self, storyboard: torch.Tensor) -> torch.Tensor:
        """[B, 4, 128, 128] → [B, 4, 4, 64, 64]"""
        top, bottom = storyboard.chunk(2, dim=-2)
        frame_0, frame_1 = top.chunk(2, dim=-1)
        frame_2, frame_3 = bottom.chunk(2, dim=-1)
        return torch.stack([frame_0, frame_1, frame_2, frame_3], dim=1)
    
    @torch.no_grad()
    def generate(
        self,
        mel: torch.Tensor = None,
        text_embed: torch.Tensor = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: torch.Generator = None,
        conditioning_mode: str = "audio"  # 추가
    ) -> torch.Tensor:
        """추론: Audio/Text → Storyboard"""
        
        # Determine batch size and device
        if mel is not None:
            device = mel.device
            B = mel.shape[0]
        elif text_embed is not None:
            device = text_embed.device
            B = text_embed.shape[0]
        else:
            raise ValueError("At least one of mel or text_embed must be provided")
        
        # VAE lazy loading
        if self.vae is None:
            self.vae = AutoencoderKL.from_pretrained(
                self.pretrained_model, subfolder="vae"
            ).to(device)
            self.vae.eval()
        
        # 1. Audio encoding (if needed)
        audio_embeds = None
        if conditioning_mode in ["audio", "both"]:
            if mel is None:
                raise ValueError(f"mel required for '{conditioning_mode}' mode")
            audio_embeds = self.audio_encoder(mel)
        
        # 2. 초기 noise
        latent_shape = (B, 4, 128, 128)
        latents = torch.randn(latent_shape, device=device, generator=generator)
        
        # 3. Scheduler 설정
        self.scheduler.set_timesteps(num_inference_steps)
        latents = latents * self.scheduler.init_noise_sigma
        
        # 4. Null embedding for CFG
        null_embed = self.null_audio_embed.expand(B, -1, -1).to(device)
        
        # 5. Denoising loop
        for t in self.scheduler.timesteps:
            t_tensor = t.to(device)
            
            if guidance_scale > 1.0:
                latent_input = torch.cat([latents] * 2)
                
                # Prepare conditional/unconditional inputs based on mode
                if conditioning_mode == "audio":
                    cond_embed = audio_embeds
                    uncond_embed = null_embed
                    audio_input = torch.cat([uncond_embed, cond_embed])
                    text_input = None
                elif conditioning_mode == "text":
                    cond_embed = text_embed
                    uncond_embed = self.null_text_embed.expand(B, -1, -1).to(device)
                    audio_input = None
                    text_input = torch.cat([uncond_embed, cond_embed])
                else:  # both
                    audio_input = torch.cat([null_embed, audio_embeds])
                    null_text = self.null_text_embed.expand(B, -1, -1).to(device)
                    text_input = torch.cat([null_text, text_embed])
                
                noise_pred = self.storyboard_unet(
                    latent_input,
                    t_tensor.expand(B * 2),
                    audio_input,
                    text_input,
                    conditioning_mode=conditioning_mode
                )
                
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = self.storyboard_unet(
                    latents,
                    t_tensor.expand(B),
                    audio_embeds,
                    text_embed,
                    conditioning_mode=conditioning_mode
                )
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # 6. VAE decode (동일)
        latents = latents / self.vae_scale_factor
        frame_latents = self._split_latents(latents)
        
        images = []
        for i in range(4):
            frame = self.vae.decode(frame_latents[:, i]).sample
            frame = (frame / 2 + 0.5).clamp(0, 1)
            images.append(frame)
        
        return torch.stack(images, dim=1)