# models/storyboard_unet.py
import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

class StoryboardUNet(nn.Module):
    """
    Audio embedding이 Text embedding을 대체하는 구조
    
    Conditioning modes:
    - "audio": Audio embedding만 사용
    - "text": Text embedding만 사용 (ablation)
    - "both": Audio + Text fusion
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
        audio_embeds: torch.Tensor = None,  # [B, 77, 768] - optional
        text_embeds: torch.Tensor = None,   # [B, 77, 768] - optional
        conditioning_mode: str = "audio"    # "audio", "text", "both"
    ) -> torch.Tensor:
        """
        Args:
            sample: Noisy latent [B, 4, H, W]
            timestep: Diffusion timestep [B]
            audio_embeds: Audio encoder output [B, 77, 768]
            text_embeds: CLIP text embedding [B, 77, 768]
            conditioning_mode: "audio", "text", or "both"
        """
        
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
        
        noise_pred = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states
        ).sample
        
        return noise_pred