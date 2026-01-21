# models/pipeline.py
"""
Audio-to-Storyboard Generation Pipeline
with Consistent Self-Attention for character consistency

핵심 변경: 4프레임을 [B*4, ...] Flat Batch로 처리
- 기존: 2x2 그리드로 병합 [B, 4, 128, 128]
- 변경: Flat batch [B*4, 4, 64, 64]로 처리
- Consistent Self-Attention이 배치 내에서 자동으로 프레임 간 참조
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler
from .audio_encoder import AudioEncoder
from .storyboard_unet import StoryboardUNet


class AudioToStoryboardPipeline(nn.Module):
    """
    Audio-to-Storyboard Generation Pipeline
    
    4프레임을 동시에 생성하며, Consistent Self-Attention으로
    캐릭터 일관성을 자동으로 유지
    """
    
    def __init__(
        self,
        pretrained_model: str = "runwayml/stable-diffusion-v1-5",
        audio_encoder_config: dict = None,
        freeze_unet: bool = True,
        use_consistent_attention: bool = True,
        num_frames: int = 4,
        attention_mode: str = "first",  # "first" or "mutual"
    ):
        super().__init__()
        
        if audio_encoder_config is None:
            audio_encoder_config = {
                'mel_channels': 128,
                'hidden_dim': 512,
                'output_dim': 768,
                'num_layers': 6,
                'num_heads': 8,
                'output_seq_len': 77
            }
        
        self.num_frames = num_frames
        self.use_consistent_attention = use_consistent_attention
        
        # 1. Audio Encoder
        self.audio_encoder = AudioEncoder(**audio_encoder_config)
        
        # 2. U-Net with Consistent Self-Attention
        self.storyboard_unet = StoryboardUNet(
            pretrained_model=pretrained_model,
            freeze_unet=freeze_unet,
            use_consistent_attention=use_consistent_attention,
            num_frames=num_frames,
            attention_mode=attention_mode,
        )
        
        # 3. Scheduler
        self.scheduler = DDIMScheduler.from_pretrained(
            pretrained_model,
            subfolder="scheduler"
        )
        
        # 4. Learnable null embeddings (CFG용)
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
        conditioning_mode: str = "audio"
    ) -> dict:
        """
        학습용 Forward pass
        
        Args:
            mel: [B, 128, T] - Mel-spectrogram (하나의 오디오가 4프레임에 공유)
            latent: [B, 4, 4, 64, 64] - 4프레임의 VAE latent
            text_embed: [B, 77, 768] - Text embedding (4프레임에 공유)
            mel_mask: [B, T] - Mel padding mask
            conditioning_mode: "audio", "text", or "both"
        
        Returns:
            dict with 'loss', 'noise_pred', 'noise', 'timesteps'
        """
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
            # [B, 77, 768]
            audio_embeds = self.audio_encoder(mel, mel_mask)
        
        # 2. Latent 준비: [B, 4, 4, 64, 64] → [B*4, 4, 64, 64]
        # 각 프레임을 별도 배치로 펼침
        flat_latent = latent.view(B * self.num_frames, 4, 64, 64)
        
        # 3. Embedding 확장: [B, 77, 768] → [B*4, 77, 768]
        # 같은 conditioning을 4프레임에 공유
        if audio_embeds is not None:
            audio_embeds = audio_embeds.unsqueeze(1).expand(-1, self.num_frames, -1, -1)
            audio_embeds = audio_embeds.reshape(B * self.num_frames, 77, 768)
        
        if text_embed is not None:
            text_embed = text_embed.unsqueeze(1).expand(-1, self.num_frames, -1, -1)
            text_embed = text_embed.reshape(B * self.num_frames, 77, 768)
        
        # 4. Random timestep (모든 프레임에 동일한 timestep 사용)
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (B,), device=device, dtype=torch.long
        )
        # [B] → [B*4]
        timesteps = timesteps.unsqueeze(1).expand(-1, self.num_frames).reshape(-1)
        
        # 5. Noise 추가
        noise = torch.randn_like(flat_latent)
        noisy_latent = self.scheduler.add_noise(flat_latent, noise, timesteps)
        
        # 6. Noise 예측 (Consistent Self-Attention이 자동 적용)
        noise_pred = self.storyboard_unet(
            sample=noisy_latent,
            timestep=timesteps,
            audio_embeds=audio_embeds,
            text_embeds=text_embed,
            conditioning_mode=conditioning_mode
        )
        
        # 7. Loss
        loss = F.mse_loss(noise_pred, noise)
        
        return {
            'loss': loss,
            'noise_pred': noise_pred,
            'noise': noise,
            'timesteps': timesteps
        }
    
    @torch.no_grad()
    def generate(
        self,
        mel: torch.Tensor = None,
        text_embed: torch.Tensor = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: torch.Generator = None,
        conditioning_mode: str = "audio",
        use_consistent_attention: bool = None,
    ) -> torch.Tensor:
        """
        추론: Audio/Text → 4-frame Storyboard
        
        Args:
            mel: [B, 128, T] - Mel-spectrogram
            text_embed: [B, 77, 768] - Text embedding
            num_inference_steps: Denoising steps
            guidance_scale: CFG scale
            generator: Random generator
            conditioning_mode: "audio", "text", or "both"
            use_consistent_attention: Override default (None = use default)
        
        Returns:
            images: [B, 4, 3, 512, 512] - 4 frames of generated images
        """
        
        # Override consistent attention if specified
        if use_consistent_attention is not None:
            if use_consistent_attention and not self.storyboard_unet.use_consistent_attention:
                self.storyboard_unet.enable_consistent_attention()
            elif not use_consistent_attention and self.storyboard_unet.use_consistent_attention:
                self.storyboard_unet.disable_consistent_attention()
        
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
        
        # 1. Audio encoding
        audio_embeds = None
        if conditioning_mode in ["audio", "both"]:
            if mel is None:
                raise ValueError(f"mel required for '{conditioning_mode}' mode")
            audio_embeds = self.audio_encoder(mel)
            # [B, 77, 768] → [B*4, 77, 768]
            audio_embeds = audio_embeds.unsqueeze(1).expand(-1, self.num_frames, -1, -1)
            audio_embeds = audio_embeds.reshape(B * self.num_frames, 77, 768)
        
        # Text embedding 확장
        if text_embed is not None:
            text_embed = text_embed.unsqueeze(1).expand(-1, self.num_frames, -1, -1)
            text_embed = text_embed.reshape(B * self.num_frames, 77, 768)
        
        # 2. 초기 noise: [B*4, 4, 64, 64]
        latent_shape = (B * self.num_frames, 4, 64, 64)
        latents = torch.randn(latent_shape, device=device, generator=generator)
        
        # 3. Scheduler 설정
        self.scheduler.set_timesteps(num_inference_steps)
        latents = latents * self.scheduler.init_noise_sigma
        
        # 4. Null embeddings 준비
        null_audio = self.null_audio_embed.expand(B * self.num_frames, -1, -1).to(device)
        null_text = self.null_text_embed.expand(B * self.num_frames, -1, -1).to(device)
        
        # 5. Denoising loop
        for t in self.scheduler.timesteps:
            t_tensor = t.to(device)
            # [B*4]
            timestep_batch = t_tensor.expand(B * self.num_frames)
            
            if guidance_scale > 1.0:
                # CFG: [B*4*2, ...]
                latent_input = torch.cat([latents] * 2)
                timestep_input = torch.cat([timestep_batch] * 2)
                
                # Prepare conditional/unconditional inputs
                if conditioning_mode == "audio":
                    audio_input = torch.cat([null_audio, audio_embeds])
                    text_input = None
                elif conditioning_mode == "text":
                    audio_input = None
                    text_input = torch.cat([null_text, text_embed])
                else:  # both
                    audio_input = torch.cat([null_audio, audio_embeds])
                    text_input = torch.cat([null_text, text_embed])
                
                noise_pred = self.storyboard_unet(
                    latent_input,
                    timestep_input,
                    audio_input,
                    text_input,
                    conditioning_mode=conditioning_mode
                )
                
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = self.storyboard_unet(
                    latents,
                    timestep_batch,
                    audio_embeds,
                    text_embed,
                    conditioning_mode=conditioning_mode
                )
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # 6. VAE decode: [B*4, 4, 64, 64] → [B*4, 3, 512, 512]
        latents = latents / self.vae_scale_factor
        
        images = []
        for i in range(B * self.num_frames):
            frame = self.vae.decode(latents[i:i+1]).sample
            frame = (frame / 2 + 0.5).clamp(0, 1)
            images.append(frame)
        
        # [B*4, 3, 512, 512] → [B, 4, 3, 512, 512]
        images = torch.cat(images, dim=0)
        images = images.view(B, self.num_frames, 3, 512, 512)
        
        return images
    
    @torch.no_grad()
    def generate_single_frame(
        self,
        mel: torch.Tensor = None,
        text_embed: torch.Tensor = None,
        frame_idx: int = 0,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: torch.Generator = None,
        conditioning_mode: str = "audio",
    ) -> torch.Tensor:
        """
        단일 프레임 생성 (테스트/디버깅용)
        
        Returns:
            image: [B, 3, 512, 512]
        """
        if mel is not None:
            device = mel.device
            B = mel.shape[0]
        elif text_embed is not None:
            device = text_embed.device
            B = text_embed.shape[0]
        else:
            raise ValueError("At least one of mel or text_embed must be provided")
        
        if self.vae is None:
            self.vae = AutoencoderKL.from_pretrained(
                self.pretrained_model, subfolder="vae"
            ).to(device)
            self.vae.eval()
        
        # Disable consistent attention for single frame
        self.storyboard_unet.disable_consistent_attention()
        
        # Audio encoding
        audio_embeds = None
        if conditioning_mode in ["audio", "both"]:
            audio_embeds = self.audio_encoder(mel)
        
        # Initial noise
        latents = torch.randn((B, 4, 64, 64), device=device, generator=generator)
        
        self.scheduler.set_timesteps(num_inference_steps)
        latents = latents * self.scheduler.init_noise_sigma
        
        null_audio = self.null_audio_embed.expand(B, -1, -1).to(device)
        null_text = self.null_text_embed.expand(B, -1, -1).to(device)
        
        for t in self.scheduler.timesteps:
            t_tensor = t.to(device).expand(B)
            
            if guidance_scale > 1.0:
                latent_input = torch.cat([latents] * 2)
                timestep_input = torch.cat([t_tensor] * 2)
                
                if conditioning_mode == "audio":
                    audio_input = torch.cat([null_audio, audio_embeds])
                    text_input = None
                elif conditioning_mode == "text":
                    audio_input = None
                    text_input = torch.cat([null_text, text_embed])
                else:
                    audio_input = torch.cat([null_audio, audio_embeds])
                    text_input = torch.cat([null_text, text_embed])
                
                noise_pred = self.storyboard_unet(
                    latent_input,
                    timestep_input,
                    audio_input,
                    text_input,
                    conditioning_mode=conditioning_mode
                )
                
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = self.storyboard_unet(
                    latents,
                    t_tensor,
                    audio_embeds,
                    text_embed,
                    conditioning_mode=conditioning_mode
                )
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # VAE decode
        latents = latents / self.vae_scale_factor
        images = self.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        
        # Re-enable consistent attention
        self.storyboard_unet.enable_consistent_attention()
        
        return images
