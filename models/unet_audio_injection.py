# models/unet_audio_injection.py
"""
U-Net with Audio Cross-Attention Injection

실제로 U-Net의 Cross-Attention에 Audio를 주입하는 구현.
두 가지 방식 제공:
1. Hook-based: 기존 U-Net을 수정하지 않고 hook으로 주입
2. Wrapper-based: U-Net forward를 래핑하여 수정
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from typing import Optional, Dict, Callable
from models.audio_attention import AudioCrossAttention


class AudioInjectionUNet(nn.Module):
    """
    Audio Cross-Attention을 U-Net에 주입하는 Wrapper
    
    IP-Adapter 방식: 기존 Text Cross-Attention 출력에 Audio Cross-Attention 출력을 더함
    """
    
    def __init__(
        self,
        pretrained_model: str = "runwayml/stable-diffusion-v1-5",
        audio_dim: int = 768,
        audio_scale: float = 1.0,
        freeze_unet: bool = True,
    ):
        super().__init__()
        
        self.audio_scale = audio_scale
        
        # Load U-Net
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model,
            subfolder="unet",
        )
        
        if freeze_unet:
            self.unet.requires_grad_(False)
            print("🔒 U-Net frozen")
        
        # Channel dimensions for SD 1.5
        # down_blocks: [320, 640, 1280, 1280]
        # mid_block: 1280
        # up_blocks: [1280, 1280, 640, 320]
        
        # Create Audio Cross-Attention layers for each Transformer block
        self.audio_attns = nn.ModuleDict()
        self._create_audio_attention_layers(audio_dim)
        
        # Register hooks
        self._hooks = []
        self._audio_embed = None
        self._register_hooks()
        
        print(f"✅ Created {len(self.audio_attns)} Audio Cross-Attention layers")
    
    def _create_audio_attention_layers(self, audio_dim: int):
        """
        U-Net의 각 Transformer block에 대응하는 Audio Cross-Attention 생성
        """
        # SD 1.5 U-Net structure:
        # down_blocks[0]: BasicTransformerBlock with dim=320 (2 blocks)
        # down_blocks[1]: BasicTransformerBlock with dim=640 (2 blocks)
        # down_blocks[2]: BasicTransformerBlock with dim=1280 (2 blocks)
        # mid_block: BasicTransformerBlock with dim=1280 (1 block)
        # up_blocks[0]: BasicTransformerBlock with dim=1280 (3 blocks)
        # up_blocks[1]: BasicTransformerBlock with dim=1280 (3 blocks)
        # up_blocks[2]: BasicTransformerBlock with dim=640 (3 blocks)
        # up_blocks[3]: BasicTransformerBlock with dim=320 (3 blocks)
        
        layers_config = [
            # Down blocks
            ('down_0_0', 320), ('down_0_1', 320),
            ('down_1_0', 640), ('down_1_1', 640),
            ('down_2_0', 1280), ('down_2_1', 1280),
            # Mid block
            ('mid_0', 1280),
            # Up blocks
            ('up_0_0', 1280), ('up_0_1', 1280), ('up_0_2', 1280),
            ('up_1_0', 1280), ('up_1_1', 1280), ('up_1_2', 1280),
            ('up_2_0', 640), ('up_2_1', 640), ('up_2_2', 640),
            ('up_3_0', 320), ('up_3_1', 320), ('up_3_2', 320),
        ]
        
        for name, dim in layers_config:
            self.audio_attns[name] = AudioCrossAttention(
                query_dim=dim,
                audio_dim=audio_dim,
                heads=8,
            )
    
    def _register_hooks(self):
        """
        U-Net의 Cross-Attention 출력에 hook 등록
        """
        self._hook_outputs = {}
        
        def make_hook(layer_name: str, audio_attn: AudioCrossAttention):
            def hook(module, input, output):
                if self._audio_embed is not None:
                    # output: [B, H*W, C]
                    hidden_states = output
                    
                    # Apply audio cross-attention
                    audio_out = audio_attn(
                        hidden_states=hidden_states,
                        audio_embed=self._audio_embed,
                    )
                    
                    # Add to output
                    return hidden_states + self.audio_scale * audio_out
                return output
            return hook
        
        # Find and hook cross-attention (attn2) layers
        layer_idx = 0
        layer_names = list(self.audio_attns.keys())
        
        for name, module in self.unet.named_modules():
            # Look for cross-attention layers (attn2)
            if 'attn2' in name and hasattr(module, 'to_out'):
                if layer_idx < len(layer_names):
                    layer_name = layer_names[layer_idx]
                    audio_attn = self.audio_attns[layer_name]
                    
                    # Register forward hook on the output projection
                    hook = module.to_out.register_forward_hook(
                        make_hook(layer_name, audio_attn)
                    )
                    self._hooks.append(hook)
                    layer_idx += 1
        
        print(f"   Registered {len(self._hooks)} hooks")
    
    def set_audio_embed(self, audio_embed: torch.Tensor):
        """
        Forward 전에 audio embedding 설정
        """
        self._audio_embed = audio_embed
    
    def clear_audio_embed(self):
        """
        Forward 후 audio embedding 클리어
        """
        self._audio_embed = None
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with audio injection
        """
        # Set audio embedding for hooks
        if audio_hidden_states is not None:
            self.set_audio_embed(audio_hidden_states)
        
        # Standard U-Net forward (hooks will inject audio)
        output = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs,
        )
        
        # Clear audio embedding
        self.clear_audio_embed()
        
        return output


class SimpleAudioInjectionUNet(nn.Module):
    """
    간단한 Audio 주입 방식
    
    U-Net forward를 직접 수정하지 않고,
    Text embedding과 Audio embedding을 결합하여 conditioning으로 사용
    """
    
    def __init__(
        self,
        pretrained_model: str = "runwayml/stable-diffusion-v1-5",
        audio_dim: int = 768,
        text_dim: int = 768,
        fusion_type: str = "add",  # "add", "concat", "gate"
        audio_scale: float = 1.0,
        freeze_unet: bool = True,
    ):
        super().__init__()
        
        self.fusion_type = fusion_type
        self.audio_scale = audio_scale
        
        # Load U-Net
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model,
            subfolder="unet",
        )
        
        if freeze_unet:
            self.unet.requires_grad_(False)
            print("🔒 U-Net frozen")
        
        # Fusion layers
        if fusion_type == "concat":
            # Project concatenated embedding back to original dim
            self.fusion_proj = nn.Linear(text_dim + audio_dim, text_dim)
        elif fusion_type == "gate":
            # Gating mechanism
            self.gate = nn.Sequential(
                nn.Linear(text_dim + audio_dim, text_dim),
                nn.Sigmoid(),
            )
            self.audio_proj = nn.Linear(audio_dim, text_dim)
        elif fusion_type == "cross_attn":
            # Cross-attention fusion
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=text_dim,
                num_heads=8,
                batch_first=True,
            )
            self.norm = nn.LayerNorm(text_dim)
    
    def fuse_embeddings(
        self,
        text_embed: torch.Tensor,  # [B, 77, 768]
        audio_embed: torch.Tensor,  # [B, 77, 768]
    ) -> torch.Tensor:
        """
        Text와 Audio embedding 융합
        """
        if self.fusion_type == "add":
            # Simple addition
            return text_embed + self.audio_scale * audio_embed
        
        elif self.fusion_type == "concat":
            # Concatenate and project
            concat = torch.cat([text_embed, audio_embed], dim=-1)  # [B, 77, 1536]
            return self.fusion_proj(concat)  # [B, 77, 768]
        
        elif self.fusion_type == "gate":
            # Gated fusion
            concat = torch.cat([text_embed, audio_embed], dim=-1)
            gate = self.gate(concat)  # [B, 77, 768]
            audio_proj = self.audio_proj(audio_embed)  # [B, 77, 768]
            return text_embed + gate * audio_proj
        
        elif self.fusion_type == "cross_attn":
            # Cross-attention: text attends to audio
            attn_out, _ = self.cross_attn(
                query=text_embed,
                key=audio_embed,
                value=audio_embed,
            )
            return self.norm(text_embed + self.audio_scale * attn_out)
        
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor = None,
        **kwargs,
    ):
        """
        Forward with fused conditioning
        """
        # Fuse text and audio embeddings
        if audio_hidden_states is not None:
            fused_embed = self.fuse_embeddings(encoder_hidden_states, audio_hidden_states)
        else:
            fused_embed = encoder_hidden_states
        
        # Standard U-Net forward with fused embedding
        output = self.unet(
            sample,
            timestep,
            encoder_hidden_states=fused_embed,
            **kwargs,
        )
        
        return output


class DecoupledCrossAttentionUNet(nn.Module):
    """
    Decoupled Cross-Attention U-Net (IP-Adapter 방식)
    
    U-Net을 수정하여 Text와 Audio에 대한 별도의 Cross-Attention 경로를 가짐
    """
    
    def __init__(
        self,
        pretrained_model: str = "runwayml/stable-diffusion-v1-5",
        audio_dim: int = 768,
        audio_scale: float = 1.0,
        freeze_unet: bool = True,
    ):
        super().__init__()
        
        self.audio_scale = nn.Parameter(torch.tensor(audio_scale))
        
        # Load U-Net
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model,
            subfolder="unet",
        )
        
        if freeze_unet:
            self.unet.requires_grad_(False)
            print("🔒 U-Net frozen")
        
        # Create parallel Audio Cross-Attention layers
        # These mirror the structure of U-Net's cross-attention layers
        self.audio_cross_attn = nn.ModuleDict()
        self._setup_audio_layers(audio_dim)
        
        # Storage for intermediate states
        self._intermediate_states = {}
    
    def _setup_audio_layers(self, audio_dim: int):
        """
        U-Net 구조를 분석하여 대응하는 Audio Cross-Attention 생성
        """
        for name, module in self.unet.named_modules():
            # Find cross-attention layers (attn2 in transformer blocks)
            if hasattr(module, 'to_k') and 'attn2' in name:
                # Get dimensions
                query_dim = module.to_q.in_features
                
                # Clean name
                clean_name = name.replace('.', '_')
                
                # Create corresponding audio attention
                self.audio_cross_attn[clean_name] = nn.ModuleDict({
                    'to_k': nn.Linear(audio_dim, module.to_k.out_features, bias=False),
                    'to_v': nn.Linear(audio_dim, module.to_v.out_features, bias=False),
                    'scale': nn.Parameter(torch.ones(1) * 0.1),
                })
        
        print(f"   Created {len(self.audio_cross_attn)} audio attention layers")
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor = None,
        return_dict: bool = True,
    ):
        """
        Forward with decoupled attention
        
        현재는 SimpleAudioInjectionUNet의 add 방식 사용 (안정성)
        추후 full decoupled 구현 가능
        """
        # For now, use simple addition fusion
        if audio_hidden_states is not None:
            # Ensure same sequence length
            if audio_hidden_states.shape[1] != encoder_hidden_states.shape[1]:
                # Interpolate or truncate
                audio_hidden_states = F.interpolate(
                    audio_hidden_states.transpose(1, 2),
                    size=encoder_hidden_states.shape[1],
                    mode='linear',
                    align_corners=False,
                ).transpose(1, 2)
            
            fused = encoder_hidden_states + self.audio_scale * audio_hidden_states
        else:
            fused = encoder_hidden_states
        
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states=fused,
            return_dict=return_dict,
        )
