import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, AutoencoderKL
from .audio_encoder import AudioEncoder
from .storyboard_unet import StoryboardUNet

class AudioToStoryboardPipeline(nn.Module):
    """
    Audio-to-Storyboard Generation Pipeline
    
    기능:
    1. Audio Encoder로 오디오 특징 추출
    2. U-Net을 통해 노이즈 예측 (학습)
    3. DDPMScheduler를 이용한 노이즈 추가/제거 (학습/추론)
    """
    
    def __init__(
        self,
        pretrained_model: str = "runwayml/stable-diffusion-v1-5",
        audio_encoder_config: dict = None,
        freeze_unet: bool = True
    ):
        super().__init__()
        
        # 1. Audio Encoder Config 설정
        if audio_encoder_config is None:
            audio_encoder_config = {
                'mel_channels': 128,
                'hidden_dim': 512,
                'output_dim': 768,
                'num_layers': 6,
                'num_heads': 8,
                'num_segments': 4
            }
        
        # 2. 모델 초기화
        self.audio_encoder = AudioEncoder(**audio_encoder_config)
        self.storyboard_unet = StoryboardUNet(pretrained_model, freeze_unet)
        
        # 3. Noise Scheduler (학습 및 추론용)
        self.scheduler = DDPMScheduler.from_pretrained(
            pretrained_model, subfolder="scheduler"
        )
        
        # 4. VAE (추론 시에만 로드, 학습 시에는 전처리된 Latent 사용)
        self.vae = None 
        self.pretrained_model = pretrained_model
        
    def forward(
        self, 
        mel: torch.Tensor,          # [B, 128, T]
        latent: torch.Tensor,       # [B, 4, 4, 64, 64] (Clean Latent)
        text_embed: torch.Tensor,   # [B, 77, 768]
        mel_mask: torch.Tensor = None
    ):
        """
        [학습 모드] Forward Pass & Loss Calculation
        """
        device = latent.device
        B, NumFrames, C, H, W = latent.shape # NumFrames=4
        
        # 1. Audio Encoding
        # [B, 128, T] -> [B, 4, 768]
        audio_emb = self.audio_encoder(mel, mel_mask)
        
        # 2. Inputs Flattening (Batch * 4)
        # 스토리보드 4컷을 개별 이미지처럼 처리하기 위해 Batch 차원으로 펼칩니다.
        # Latent: [B, 4, 4, 64, 64] -> [B*4, 4, 64, 64]
        latents_flat = latent.view(-1, C, H, W)
        
        # Conditions 확장 (Repeat)
        # Audio: [B, 4, 768] -> [B*4, 4, 768] (모든 컷이 전체 오디오 맥락을 봄)
        audio_emb_flat = audio_emb.repeat_interleave(NumFrames, dim=0)
        
        # Text: [B, 77, 768] -> [B*4, 77, 768]
        text_embed_flat = text_embed.repeat_interleave(NumFrames, dim=0)
        
        # 3. Noise Injection (Diffusion Process)
        noise = torch.randn_like(latents_flat)
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, 
            (latents_flat.shape[0],), device=device
        ).long()
        
        noisy_latents = self.scheduler.add_noise(latents_flat, noise, timesteps)
        
        # 4. Predict Noise (U-Net)
        # storyboard_unet 내부에서 Audio와 Text가 Concat되어 처리됨
        noise_pred = self.storyboard_unet(
            sample=noisy_latents,
            timestep=timesteps,
            audio_embeds=audio_emb_flat,
            text_embeds=text_embed_flat
        )
        
        # 5. Loss Calculation (MSE)
        loss = F.mse_loss(noise_pred, noise)
        
        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        mel: torch.Tensor,
        text_embed: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: torch.Generator = None
    ):
        """
        [추론 모드] Audio -> Images Generation
        """
        device = mel.device
        B = mel.shape[0]
        NumFrames = 4
        
        # VAE 로드 (필요 시)
        if self.vae is None:
            self.vae = AutoencoderKL.from_pretrained(
                self.pretrained_model, subfolder="vae"
            ).to(device)
            self.vae.eval()
            
        # 1. Audio Encoding
        audio_emb = self.audio_encoder(mel) # [B, 4, 768]
        
        # 2. Conditions 준비 (Flatten)
        # [B*4, 4, 768]
        audio_emb_flat = audio_emb.repeat_interleave(NumFrames, dim=0)
        # [B*4, 77, 768]
        text_embed_flat = text_embed.repeat_interleave(NumFrames, dim=0)
        
        # 3. 초기 Noise 생성
        # [B*4, 4, 64, 64]
        latents = torch.randn(
            (B * NumFrames, 4, 64, 64), 
            device=device, 
            generator=generator
        )
        
        # Scheduler 초기화
        self.scheduler.set_timesteps(num_inference_steps)
        latents = latents * self.scheduler.init_noise_sigma
        
        # 4. Denoising Loop
        for t in self.scheduler.timesteps:
            # CFG를 위한 입력 복제 (Unconditional + Conditional)
            # 여기서는 간단하게 구현 (필요시 Uncond embedding 추가 필요)
            # 현재는 guidance 없이 conditional만 수행하는 구조로 단순화
            
            # U-Net Forward
            noise_pred = self.storyboard_unet(
                sample=latents,
                timestep=t,
                audio_embeds=audio_emb_flat,
                text_embeds=text_embed_flat
            )
            
            # Scheduler Step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
        # 5. VAE Decode
        # Scaling Factor 보정
        latents = latents / 0.18215
        images = self.vae.decode(latents).sample
        
        # [B*4, 3, 512, 512] -> [0, 1] 범위로 변환
        images = (images / 2 + 0.5).clamp(0, 1)
        
        # [B, 4, 3, 512, 512] 형태로 복원
        images = images.view(B, NumFrames, 3, 512, 512)
        
        return images