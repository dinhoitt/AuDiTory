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
    with Consistent Self-Attention for character consistency
    """
    
    def __init__(
        self,
        pretrained_model: str = "runwayml/stable-diffusion-v1-5",
        audio_encoder_config: dict = None,
        freeze_unet: bool = True,
        use_consistent_attention: bool = True,
        num_frames: int = 4,
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
        
        Note: 학습 시에는 Consistent Self-Attention의 효과가 자동으로 적용됩니다.
        4개의 프레임이 배치로 들어오면, self-attention이 모든 프레임 간
        features를 공유하게 됩니다.
        
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
        conditioning_mode: str = "audio",
        use_consistent_attention: bool = None,  # Override default setting
    ) -> torch.Tensor:
        """
        추론: Audio/Text → Storyboard with Consistent Self-Attention
        
        Consistent Self-Attention은 denoising 과정에서 자동으로 적용됩니다.
        """
        
        # Override consistent attention setting if specified
        if use_consistent_attention is not None:
            _original_setting = self.use_consistent_attention
            if use_consistent_attention and not self.use_consistent_attention:
                self.storyboard_unet.enable_consistent_attention()
            elif not use_consistent_attention and self.use_consistent_attention:
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
        
        # 5. Reset consistent attention for new generation
        if self.use_consistent_attention:
            self.storyboard_unet.reset_attention_bank()
        
        # 6. Denoising loop with Consistent Self-Attention
        for step_idx, t in enumerate(self.scheduler.timesteps):
            t_tensor = t.to(device)
            
            # Update attention step (for consistent attention)
            if self.use_consistent_attention and self.storyboard_unet.ca_manager:
                self.storyboard_unet.ca_manager.cur_step = step_idx
                self.storyboard_unet.ca_manager._update_processors()
            
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
        
        # 7. VAE decode
        latents = latents / self.vae_scale_factor
        frame_latents = self._split_latents(latents)
        
        images = []
        for i in range(4):
            frame = self.vae.decode(frame_latents[:, i]).sample
            frame = (frame / 2 + 0.5).clamp(0, 1)
            images.append(frame)
        
        # Restore original setting if overridden
        if use_consistent_attention is not None:
            if _original_setting and not use_consistent_attention:
                self.storyboard_unet.enable_consistent_attention()
            elif not _original_setting and use_consistent_attention:
                self.storyboard_unet.disable_consistent_attention()
        
        return torch.stack(images, dim=1)
    
    @torch.no_grad()
    def generate_with_reference(
        self,
        mel: torch.Tensor = None,
        text_embed: torch.Tensor = None,
        reference_latents: torch.Tensor = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: torch.Generator = None,
        conditioning_mode: str = "audio",
        reference_strength: float = 0.5,
    ) -> torch.Tensor:
        """
        Generate with reference frames for stronger consistency
        
        Two-pass generation:
        1. First pass (write mode): Generate reference frames and store features
        2. Second pass (read mode): Generate remaining frames using stored features
        
        Args:
            reference_latents: Optional pre-computed reference latents
            reference_strength: Strength of reference features (0~1)
        """
        
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
        
        # Audio encoding
        audio_embeds = None
        if conditioning_mode in ["audio", "both"]:
            if mel is None:
                raise ValueError(f"mel required for '{conditioning_mode}' mode")
            audio_embeds = self.audio_encoder(mel)
        
        # Initial noise
        latent_shape = (B, 4, 128, 128)
        latents = torch.randn(latent_shape, device=device, generator=generator)
        
        # Scheduler setup
        self.scheduler.set_timesteps(num_inference_steps)
        latents = latents * self.scheduler.init_noise_sigma
        
        # Null embedding for CFG
        null_embed = self.null_audio_embed.expand(B, -1, -1).to(device)
        
        # Reset consistent attention
        if self.use_consistent_attention:
            self.storyboard_unet.reset_attention_bank()
            # Set to write mode for first pass
            self.storyboard_unet.set_attention_mode(write=True)
        
        # Determine encoder hidden states
        if conditioning_mode == "audio":
            encoder_hidden_states = audio_embeds
            null_states = null_embed
        elif conditioning_mode == "text":
            encoder_hidden_states = text_embed
            null_states = self.null_text_embed.expand(B, -1, -1).to(device)
        else:  # both
            encoder_hidden_states = (audio_embeds + text_embed) / 2
            null_states = (null_embed + self.null_text_embed.expand(B, -1, -1).to(device)) / 2
        
        # === First Pass: Write mode (store features) ===
        print("🎬 First pass: Storing reference features...")
        for step_idx, t in enumerate(self.scheduler.timesteps):
            t_tensor = t.to(device)
            
            if self.use_consistent_attention and self.storyboard_unet.ca_manager:
                self.storyboard_unet.ca_manager.cur_step = step_idx
                self.storyboard_unet.ca_manager.write_mode = True
                self.storyboard_unet.ca_manager._update_processors()
            
            if guidance_scale > 1.0:
                latent_input = torch.cat([latents] * 2)
                hidden_input = torch.cat([null_states, encoder_hidden_states])
                
                noise_pred = self.storyboard_unet.forward_with_consistent_attention(
                    latent_input,
                    t_tensor.expand(B * 2),
                    hidden_input,
                    write_mode=True,
                    cur_step=step_idx,
                )
                
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = self.storyboard_unet.forward_with_consistent_attention(
                    latents,
                    t_tensor.expand(B),
                    encoder_hidden_states,
                    write_mode=True,
                    cur_step=step_idx,
                )
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Store first pass results
        first_pass_latents = latents.clone()
        
        # === Second Pass: Read mode (use stored features) ===
        print("🎬 Second pass: Using stored features for consistency...")
        
        # Re-initialize latents with some noise for variation
        latents = torch.randn(latent_shape, device=device, generator=generator)
        latents = latents * self.scheduler.init_noise_sigma
        
        # Blend with first pass for continuity
        blend_factor = reference_strength
        
        self.scheduler.set_timesteps(num_inference_steps)
        
        for step_idx, t in enumerate(self.scheduler.timesteps):
            t_tensor = t.to(device)
            
            if self.use_consistent_attention and self.storyboard_unet.ca_manager:
                self.storyboard_unet.ca_manager.cur_step = step_idx
                self.storyboard_unet.ca_manager.write_mode = False  # Read mode
                self.storyboard_unet.ca_manager._update_processors()
            
            if guidance_scale > 1.0:
                latent_input = torch.cat([latents] * 2)
                hidden_input = torch.cat([null_states, encoder_hidden_states])
                
                noise_pred = self.storyboard_unet.forward_with_consistent_attention(
                    latent_input,
                    t_tensor.expand(B * 2),
                    hidden_input,
                    write_mode=False,
                    cur_step=step_idx,
                )
                
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = self.storyboard_unet.forward_with_consistent_attention(
                    latents,
                    t_tensor.expand(B),
                    encoder_hidden_states,
                    write_mode=False,
                    cur_step=step_idx,
                )
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # VAE decode
        latents = latents / self.vae_scale_factor
        frame_latents = self._split_latents(latents)
        
        images = []
        for i in range(4):
            frame = self.vae.decode(frame_latents[:, i]).sample
            frame = (frame / 2 + 0.5).clamp(0, 1)
            images.append(frame)
        
        return torch.stack(images, dim=1)
