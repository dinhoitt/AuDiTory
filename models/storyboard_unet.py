# models/storyboard_unet.py
import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

class StoryboardUNet(nn.Module):
    """
    Audio embedding이 Text embedding을 대체하는 구조
    
    핵심: Audio Encoder가 [B, 77, 768] 출력 → 기존 text와 동일한 형태
    → Frozen UNet이 해석 가능
    """
    
    def __init__(
        self,
        pretrained_model: str = "runwayml/stable-diffusion-v1-5",
        freeze_unet: bool = True
    ):
        super().__init__()
        
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model,
            subfolder="unet"
        )
        
        self.cross_attention_dim = self.unet.config.cross_attention_dim
        
        if freeze_unet:
            self.unet.requires_grad_(False)
            print("🔒 U-Net frozen")
        else:
            print("🔓 U-Net unfrozen")
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        audio_embeds: torch.Tensor,      # [B, 77, 768] - text와 같은 shape
        text_embeds: torch.Tensor = None  # [B, 77, 768] - optional (fusion용)
    ) -> torch.Tensor:
        """
        Args:
            sample: Noisy latent [B, 4, H, W]
            timestep: Diffusion timestep [B]
            audio_embeds: Audio encoder output [B, 77, 768]
            text_embeds: CLIP text embedding [B, 77, 768] (optional)
        """
        
        if text_embeds is not None:
            # Audio + Text fusion (element-wise addition or learned gate)
            # 단순 평균 또는 가중합
            encoder_hidden_states = (audio_embeds + text_embeds) / 2
        else:
            # Audio only
            encoder_hidden_states = audio_embeds
        
        noise_pred = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states
        ).sample
        
        return noise_pred