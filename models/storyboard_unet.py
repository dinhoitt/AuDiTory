# models/storyboard_unet.py
"""
Storyboard UNet with Consistent Self-Attention
배치 내 상호 참조 방식으로 캐릭터 일관성 유지
"""

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from .consistent_attention import (
    set_consistent_attention,
    remove_consistent_attention,
)


class StoryboardUNet(nn.Module):
    """
    Audio/Text conditioning을 받는 UNet
    + Consistent Self-Attention for character consistency across frames
    
    입력: [B * num_frames, C, H, W] 형태의 Flat Batch
    내부에서 Consistent Self-Attention이 자동으로 프레임 간 참조 수행
    
    Conditioning modes:
    - "audio": Audio embedding만 사용
    - "text": Text embedding만 사용 (ablation)
    - "both": Audio + Text fusion
    """
    
    def __init__(
        self,
        pretrained_model: str = "runwayml/stable-diffusion-v1-5",
        freeze_unet: bool = True,
        use_consistent_attention: bool = True,
        num_frames: int = 4,
        attention_mode: str = "first",  # "first" or "mutual"
    ):
        super().__init__()
        
        self.num_frames = num_frames
        self.use_consistent_attention = use_consistent_attention
        
        # 1. Load UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model,
            subfolder="unet"
        )
        
        self.cross_attention_dim = self.unet.config.cross_attention_dim
        
        # 2. Freeze UNet
        if freeze_unet:
            self.unet.requires_grad_(False)
            print("🔒 U-Net frozen")
        else:
            print("🔓 U-Net unfrozen")
        
        # 3. Inject Consistent Self-Attention (StoryDiffusion Core)
        if use_consistent_attention:
            set_consistent_attention(
                self.unet,
                num_frames=num_frames,
                attention_mode=attention_mode,
            )
    
    def enable_consistent_attention(self, attention_mode: str = "first"):
        """Consistent Attention 활성화"""
        if not self.use_consistent_attention:
            set_consistent_attention(
                self.unet,
                num_frames=self.num_frames,
                attention_mode=attention_mode,
            )
            self.use_consistent_attention = True
    
    def disable_consistent_attention(self):
        """Consistent Attention 비활성화 (Ablation용)"""
        if self.use_consistent_attention:
            remove_consistent_attention(self.unet)
            self.use_consistent_attention = False
    
    def forward(
        self,
        sample: torch.Tensor,        # [B * num_frames, 4, H, W] - Flat Batch
        timestep: torch.Tensor,      # [B * num_frames]
        audio_embeds: torch.Tensor = None,  # [B * num_frames, 77, 768]
        text_embeds: torch.Tensor = None,   # [B * num_frames, 77, 768]
        conditioning_mode: str = "audio"    # "audio", "text", "both"
    ) -> torch.Tensor:
        """
        Args:
            sample: Noisy latent [B * num_frames, 4, H, W]
            timestep: Diffusion timestep [B * num_frames]
            audio_embeds: Audio encoder output [B * num_frames, 77, 768]
            text_embeds: CLIP text embedding [B * num_frames, 77, 768]
            conditioning_mode: "audio", "text", or "both"
        
        Returns:
            noise_pred: [B * num_frames, 4, H, W]
        
        Note:
            Consistent Self-Attention이 내부적으로 작동하여
            배치 안에서 첫 번째 프레임(index 0, num_frames, 2*num_frames, ...)의
            정보를 나머지 프레임들이 참조하게 됨
        """
        
        # Conditioning Fusion
        if conditioning_mode == "audio":
            if audio_embeds is None:
                raise ValueError("audio_embeds required for 'audio' mode")
            encoder_hidden_states = audio_embeds
            
        elif conditioning_mode == "text":
            if text_embeds is None:
                raise ValueError("text_embeds required for 'text' mode")
            encoder_hidden_states = text_embeds
            
        elif conditioning_mode == "both":
            if audio_embeds is None or text_embeds is None:
                raise ValueError("Both audio_embeds and text_embeds required for 'both' mode")
            encoder_hidden_states = (audio_embeds + text_embeds) / 2
            
        else:
            raise ValueError(f"Unknown conditioning_mode: {conditioning_mode}")
        
        # UNet Forward
        # 내부적으로 ConsistentSelfAttentionProcessor가 작동하여
        # 배치 안에서 프레임 간 정보 공유
        noise_pred = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states
        ).sample
        
        return noise_pred
