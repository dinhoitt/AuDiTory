# models/storyboard_unet.py
import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from .consistent_attention import (
    ConsistentAttentionManager,
    set_consistent_attention_processor,
    clear_attention_bank,
)


class StoryboardUNet(nn.Module):
    """
    Audio embedding이 Text embedding을 대체하는 구조
    + Consistent Self-Attention for character consistency across frames
    
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
    ):
        super().__init__()
        
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model,
            subfolder="unet"
        )
        
        self.cross_attention_dim = self.unet.config.cross_attention_dim
        self.use_consistent_attention = use_consistent_attention
        self.num_frames = num_frames
        
        # Consistent Attention Manager
        self.ca_manager = None
        if use_consistent_attention:
            self._setup_consistent_attention()
        
        if freeze_unet:
            self.unet.requires_grad_(False)
            print("🔒 U-Net frozen")
        else:
            print("🔓 U-Net unfrozen")
    
    def _setup_consistent_attention(self):
        """Setup Consistent Self-Attention processors"""
        device = next(self.unet.parameters()).device
        dtype = next(self.unet.parameters()).dtype
        
        self.ca_manager = ConsistentAttentionManager(
            unet=self.unet,
            num_frames=self.num_frames,
            enabled=True,
            device=device,
            dtype=dtype,
        )
        print(f"🎯 Consistent Self-Attention enabled ({self.ca_manager.processor_count} processors)")
    
    def enable_consistent_attention(self):
        """Enable consistent attention (if not already)"""
        if self.ca_manager is None:
            self._setup_consistent_attention()
        else:
            self.ca_manager.enable()
    
    def disable_consistent_attention(self):
        """Disable consistent attention"""
        if self.ca_manager is not None:
            self.ca_manager.disable()
    
    def reset_attention_bank(self):
        """Reset attention feature bank for new generation"""
        if self.ca_manager is not None:
            self.ca_manager.reset()
    
    def set_attention_mode(self, write: bool):
        """
        Set attention mode
        Args:
            write: True = store features, False = use stored features
        """
        if self.ca_manager is not None:
            self.ca_manager.write_mode = write
    
    def step_attention(self):
        """Advance attention step counter"""
        if self.ca_manager is not None:
            self.ca_manager.step()
    
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
    
    def forward_with_consistent_attention(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        write_mode: bool = True,
        cur_step: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass with explicit consistent attention control
        
        Args:
            sample: Noisy latent [B, 4, H, W]
            timestep: Diffusion timestep [B]
            encoder_hidden_states: Conditioning embedding [B, 77, 768]
            write_mode: True = store features, False = use stored features
            cur_step: Current denoising step
        """
        # Update attention manager state
        if self.ca_manager is not None:
            self.ca_manager.write_mode = write_mode
            self.ca_manager.cur_step = cur_step
            self.ca_manager._update_processors()
        
        noise_pred = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states
        ).sample
        
        return noise_pred
