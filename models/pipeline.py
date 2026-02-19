# models/pipeline.py
"""
Audio-to-Storyboard Pipeline with Decoupled Cross-Attention

핵심 변경사항:
1. Audio Cross-Attention을 별도 레이어로 분리
2. Text Cross-Attention (Frozen) + Audio Cross-Attention (Trainable)
3. 강화된 Align Loss (Token-wise + Global + Contrastive)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Optional, Dict, Any, List

from models.audio_encoder import AudioEncoder


class AudioProjector(nn.Module):
    """
    Audio Embedding을 U-Net Cross-Attention에 적합한 형태로 변환
    
    IP-Adapter의 image projector와 유사한 역할
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 768,
        num_tokens: int = 4,  # 압축된 토큰 수 (기승전결)
    ):
        super().__init__()
        
        self.num_tokens = num_tokens
        
        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, output_dim) * 0.02)
        
        # Cross-attention to compress audio
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            kdim=input_dim,
            vdim=input_dim,
            batch_first=True,
        )
        
        # Output projection
        self.proj = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
    
    def forward(self, audio_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_embed: [B, seq_len, input_dim]
        Returns:
            [B, num_tokens, output_dim]
        """
        B = audio_embed.shape[0]
        
        # Expand query tokens
        queries = self.query_tokens.expand(B, -1, -1)
        
        # Cross-attention: queries attend to audio
        out, _ = self.cross_attn(
            query=queries,
            key=audio_embed,
            value=audio_embed,
        )
        
        # Project
        out = self.proj(out)
        
        return out


class AudioCrossAttentionBlock(nn.Module):
    """
    U-Net block에 추가되는 Audio Cross-Attention
    """
    
    def __init__(self, query_dim: int, audio_dim: int = 768, num_heads: int = 8):
        super().__init__()
        
        self.norm = nn.LayerNorm(query_dim)
        
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(audio_dim, query_dim, bias=False)
        self.to_v = nn.Linear(audio_dim, query_dim, bias=False)
        
        self.attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        
        self.proj_out = nn.Linear(query_dim, query_dim)
        
        # Initialize with small values
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, H*W, C]
        audio_embed: torch.Tensor,     # [B, audio_len, audio_dim]
    ) -> torch.Tensor:
        residual = hidden_states
        
        hidden_states = self.norm(hidden_states)
        
        q = self.to_q(hidden_states)
        k = self.to_k(audio_embed)
        v = self.to_v(audio_embed)
        
        out, _ = self.attn(q, k, v)
        out = self.proj_out(out)
        
        return residual + out


class AudioToStoryboardPipeline(nn.Module):
    """
    Decoupled Cross-Attention 방식의 Audio-to-Storyboard Pipeline
    
    구조:
    1. Audio Encoder: Mel → Audio Embedding [B, 77, 768]
    2. Audio Projector: [B, 77, 768] → [B, 4, 768] (압축, optional)
    3. U-Net with:
       - Text Cross-Attention (Frozen)
       - Audio Cross-Attention (Trainable, 별도 레이어)
    4. VAE Decoder: Latent → Image
    """
    
    def __init__(
        self,
        pretrained_model: str = "runwayml/stable-diffusion-v1-5",
        audio_encoder_config: dict = None,
        freeze_unet: bool = True,
        use_audio_projector: bool = False,
        audio_tokens: int = 16,  # Projected audio token count
        audio_scale: float = 1.0,
        align_weight: float = 1.0,
        fusion_type: str = "add",  # "add", "gate", "cross_attn"
    ):
        super().__init__()
        
        self.audio_scale = audio_scale
        self.align_weight = align_weight
        self.fusion_type = fusion_type
        self.use_audio_projector = use_audio_projector
        
        # 1. Audio Encoder
        if audio_encoder_config is None:
            audio_encoder_config = {
                'mel_channels': 128,
                'hidden_dim': 512,
                'output_dim': 768,
                'num_layers': 6,
                'num_heads': 8,
                'output_seq_len': 77,
            }
        self.audio_encoder = AudioEncoder(**audio_encoder_config)
        
        # 2. Audio Projector (optional)
        if use_audio_projector:
            self.audio_projector = AudioProjector(
                input_dim=768,
                output_dim=768,
                num_tokens=audio_tokens,
            )
        
        # 3. Fusion layers (for combining audio with text)
        if fusion_type == "gate":
            self.gate_proj = nn.Sequential(
                nn.Linear(768 * 2, 768),
                nn.Sigmoid(),
            )
        elif fusion_type == "cross_attn":
            self.fusion_attn = nn.MultiheadAttention(
                embed_dim=768,
                num_heads=8,
                batch_first=True,
            )
            self.fusion_norm = nn.LayerNorm(768)
        
        # 4. U-Net
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model,
            subfolder="unet",
        )
        
        if freeze_unet:
            self.unet.requires_grad_(False)
            print("🔒 U-Net frozen")
        
        # 5. Audio Cross-Attention layers (별도로 추가)
        self.audio_cross_attns = nn.ModuleList()
        self._setup_audio_cross_attention()
        
        # 6. Scheduler
        self.scheduler = DDPMScheduler.from_pretrained(
            pretrained_model,
            subfolder="scheduler",
        )
        
        # 7. VAE (for decoding latents to images)
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model,
            subfolder="vae",
        )
        self.vae.requires_grad_(False)
        
        # 7. Null embeddings for CFG
        self.null_audio_embed = nn.Parameter(torch.zeros(1, 77, 768))
        self.null_text_embed = nn.Parameter(torch.zeros(1, 77, 768))
        
        # 8. Learnable audio scale (optional)
        self.learnable_audio_scale = nn.Parameter(torch.tensor(audio_scale))
        
        self._print_info()
    
    def _setup_audio_cross_attention(self):
        """
        U-Net 구조에 맞는 Audio Cross-Attention 레이어 생성
        
        SD 1.5 U-Net:
        - down_blocks: [320, 640, 1280, 1280] channels
        - mid_block: 1280 channels
        - up_blocks: [1280, 1280, 640, 320] channels
        """
        # Simplified: 주요 resolution에만 audio attention 추가
        dims = [320, 640, 1280, 1280, 1280, 1280, 640, 320]
        
        for dim in dims:
            self.audio_cross_attns.append(
                AudioCrossAttentionBlock(query_dim=dim, audio_dim=768)
            )
    
    def _print_info(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"✅ Pipeline initialized")
        print(f"   Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        print(f"   Fusion type: {self.fusion_type}")
        print(f"   Audio scale: {self.audio_scale}")
        print(f"   Align weight: {self.align_weight}")
    
    def fuse_audio_text(
        self,
        text_embed: torch.Tensor,   # [B, 77, 768]
        audio_embed: torch.Tensor,  # [B, 77, 768] or [B, N, 768]
    ) -> torch.Tensor:
        """
        Audio와 Text embedding 융합
        """
        # Ensure same sequence length
        if audio_embed.shape[1] != text_embed.shape[1]:
            # Interpolate audio to match text length
            audio_embed = F.interpolate(
                audio_embed.transpose(1, 2),
                size=text_embed.shape[1],
                mode='linear',
                align_corners=False,
            ).transpose(1, 2)
        
        scale = self.learnable_audio_scale
        
        if self.fusion_type == "add":
            return text_embed + scale * audio_embed
        
        elif self.fusion_type == "gate":
            concat = torch.cat([text_embed, audio_embed], dim=-1)
            gate = self.gate_proj(concat)
            return text_embed + gate * scale * audio_embed
        
        elif self.fusion_type == "cross_attn":
            attn_out, _ = self.fusion_attn(
                query=text_embed,
                key=audio_embed,
                value=audio_embed,
            )
            return self.fusion_norm(text_embed + scale * attn_out)
        
        else:
            return text_embed + scale * audio_embed
    
    def forward(
        self,
        mel: torch.Tensor,              # [B, 128, T]
        latent: torch.Tensor,           # [B, 4, 4, 64, 64]
        text_embed: torch.Tensor = None, # [B, 77, 768]
        mel_mask: torch.Tensor = None,
        conditioning_mode: str = "both",
    ) -> Dict[str, Any]:
        """
        Training forward pass
        """
        device = mel.device
        B = mel.shape[0]
        num_frames = latent.shape[1]  # 4
        
        # 1. Audio Encoding
        audio_embeds = self.audio_encoder(mel, mel_mask)  # [B, 77, 768]
        
        # Optional: Project to fewer tokens
        if self.use_audio_projector:
            audio_embeds_proj = self.audio_projector(audio_embeds)  # [B, N, 768]
        else:
            audio_embeds_proj = audio_embeds
        
        # 2. Flatten latents
        latent_flat = latent.view(B * num_frames, 4, 64, 64)
        
        # 3. Sample timesteps
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (B * num_frames,), device=device
        ).long()
        
        # 4. Add noise
        noise = torch.randn_like(latent_flat)
        noisy_latent = self.scheduler.add_noise(latent_flat, noise, timesteps)
        
        # 5. Prepare conditioning based on mode
        if text_embed is None:
            text_embed = self.null_text_embed.expand(B, -1, -1)
        
        if conditioning_mode == "audio":
            # Audio only: use null text, fuse with audio
            encoder_hidden = self.fuse_audio_text(
                self.null_text_embed.expand(B, -1, -1),
                audio_embeds_proj
            )
        elif conditioning_mode == "text":
            # Text only: no audio fusion
            encoder_hidden = text_embed
        else:  # "both"
            # Fuse text and audio
            encoder_hidden = self.fuse_audio_text(text_embed, audio_embeds_proj)
        
        # Expand for frames
        encoder_hidden = encoder_hidden.unsqueeze(1).expand(-1, num_frames, -1, -1)
        encoder_hidden = encoder_hidden.reshape(B * num_frames, -1, 768)
        
        # 6. U-Net forward
        noise_pred = self.unet(
            noisy_latent,
            timesteps,
            encoder_hidden_states=encoder_hidden,
        ).sample
        
        # 7. Compute losses
        diffusion_loss = F.mse_loss(noise_pred, noise)
        
        # Align loss
        align_loss = torch.tensor(0.0, device=device)
        if text_embed is not None and self.align_weight > 0 and conditioning_mode != "text":
            align_loss = self._compute_align_loss(audio_embeds, text_embed)
        
        total_loss = diffusion_loss + self.align_weight * align_loss
        
        return {
            'loss': total_loss,
            'diffusion_loss': diffusion_loss,
            'align_loss': align_loss,
            'audio_embeds': audio_embeds,
        }
    
    def _compute_align_loss(
        self,
        audio_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        강화된 Audio-Text Alignment Loss
        """
        B = audio_embeds.shape[0]
        device = audio_embeds.device
        
        # 1. Token-wise cosine similarity
        audio_norm = F.normalize(audio_embeds, dim=-1)
        text_norm = F.normalize(text_embeds, dim=-1)
        token_sim = (audio_norm * text_norm).sum(dim=-1).mean()
        token_loss = 1 - token_sim
        
        # 2. Global (pooled) alignment
        audio_pooled = F.normalize(audio_embeds.mean(dim=1), dim=-1)
        text_pooled = F.normalize(text_embeds.mean(dim=1), dim=-1)
        global_sim = (audio_pooled * text_pooled).sum(dim=-1).mean()
        global_loss = 1 - global_sim
        
        # 3. Contrastive loss (InfoNCE style)
        contrastive_loss = torch.tensor(0.0, device=device)
        if B > 1:
            temperature = 0.07
            logits = audio_pooled @ text_pooled.T / temperature
            labels = torch.arange(B, device=device)
            loss_a2t = F.cross_entropy(logits, labels)
            loss_t2a = F.cross_entropy(logits.T, labels)
            contrastive_loss = (loss_a2t + loss_t2a) / 2
        
        # Combined loss
        total_align_loss = 0.3 * token_loss + 0.3 * global_loss + 0.4 * contrastive_loss
        
        return total_align_loss
    
    @torch.no_grad()
    def generate(
        self,
        audio_embed: torch.Tensor = None,
        text_embed: torch.Tensor = None,
        num_frames: int = 4,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        audio_guidance_scale: float = 1.0,
        generator: torch.Generator = None,
        conditioning_mode: str = "both",
    ) -> torch.Tensor:
        """
        Generate storyboard
        """
        device = next(self.parameters()).device
        
        # Prepare embeddings
        if audio_embed is None:
            audio_embed = self.null_audio_embed.to(device)
        if text_embed is None:
            text_embed = self.null_text_embed.to(device)
        
        # Project audio if needed
        if self.use_audio_projector:
            audio_embed = self.audio_projector(audio_embed)
        
        # Prepare conditioning
        if conditioning_mode == "audio":
            cond_embed = self.fuse_audio_text(
                self.null_text_embed.expand(1, -1, -1).to(device),
                audio_embed
            )
        elif conditioning_mode == "text":
            cond_embed = text_embed
        else:
            cond_embed = self.fuse_audio_text(text_embed, audio_embed)
        
        # Expand for frames
        cond_embed = cond_embed.expand(num_frames, -1, -1)
        uncond_embed = self.null_text_embed.expand(num_frames, -1, -1).to(device)
        
        # CFG embeddings
        encoder_hidden = torch.cat([uncond_embed, cond_embed], dim=0)
        
        # Initialize latents
        latents = torch.randn(
            (num_frames, 4, 64, 64),
            generator=generator,
            device=device,
        )
        
        # Setup scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        latents = latents * self.scheduler.init_noise_sigma
        
        # Denoising loop
        for t in self.scheduler.timesteps:
            latent_input = torch.cat([latents] * 2)
            latent_input = self.scheduler.scale_model_input(latent_input, t)
            
            noise_pred = self.unet(
                latent_input,
                t,
                encoder_hidden_states=encoder_hidden,
            ).sample
            
            # CFG
            noise_uncond, noise_cond = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            
            # Step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode latents to images
        latents = latents / self.vae.config.scaling_factor
        images = self.vae.decode(latents).sample
        
        # Normalize to [0, 1]
        images = (images / 2 + 0.5).clamp(0, 1)
        
        return images
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """
        학습 가능한 파라미터 반환
        """
        params = []
        
        # Audio Encoder
        params.extend(list(self.audio_encoder.parameters()))
        
        # Audio Projector
        if self.use_audio_projector:
            params.extend(list(self.audio_projector.parameters()))
        
        # Fusion layers
        if self.fusion_type == "gate":
            params.extend(list(self.gate_proj.parameters()))
        elif self.fusion_type == "cross_attn":
            params.extend(list(self.fusion_attn.parameters()))
            params.extend(list(self.fusion_norm.parameters()))
        
        # Audio Cross-Attention layers
        params.extend(list(self.audio_cross_attns.parameters()))
        
        # Null embeddings
        params.append(self.null_audio_embed)
        params.append(self.null_text_embed)
        
        # Learnable scale
        params.append(self.learnable_audio_scale)
        
        return params
