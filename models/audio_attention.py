# models/audio_attention.py
"""
Decoupled Audio Cross-Attention Module

IP-Adapter/SonicDiffusion 방식을 참고하여 구현.
기존 UNet의 Cross-Attention과 별도로 Audio 전용 Cross-Attention을 추가.

핵심:
- 기존 Text Cross-Attention은 Frozen
- Audio Cross-Attention만 학습
- 두 출력을 가중합하여 최종 출력 생성
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AudioCrossAttention(nn.Module):
    """
    Audio 전용 Cross-Attention Layer
    
    U-Net의 hidden state를 Query로, Audio embedding을 Key/Value로 사용
    """
    
    def __init__(
        self,
        query_dim: int,           # U-Net hidden state dimension
        audio_dim: int = 768,     # Audio embedding dimension
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        
        # Query projection (from U-Net hidden states)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        
        # Key, Value projection (from Audio embedding)
        self.to_k = nn.Linear(audio_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(audio_dim, inner_dim, bias=False)
        
        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization for stability
        self.norm = nn.LayerNorm(query_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        # 작은 값으로 초기화하여 학습 초기에 영향 최소화
        for module in [self.to_q, self.to_k, self.to_v]:
            nn.init.xavier_uniform_(module.weight, gain=0.01)
        nn.init.xavier_uniform_(self.to_out[0].weight, gain=0.01)
        nn.init.zeros_(self.to_out[0].bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,      # [B, H*W, query_dim] from U-Net
        audio_embed: torch.Tensor,         # [B, seq_len, audio_dim]
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: U-Net의 중간 feature [B, spatial_tokens, query_dim]
            audio_embed: Audio embedding [B, audio_tokens, audio_dim]
            attention_mask: Optional mask for audio tokens
        
        Returns:
            Attention output [B, spatial_tokens, query_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Normalize input
        hidden_states_norm = self.norm(hidden_states)
        
        # Project to Q, K, V
        q = self.to_q(hidden_states_norm)  # [B, H*W, inner_dim]
        k = self.to_k(audio_embed)          # [B, audio_len, inner_dim]
        v = self.to_v(audio_embed)          # [B, audio_len, inner_dim]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        # Now: [B, heads, seq_len, dim_head]
        
        # Attention scores
        scale = self.dim_head ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        # [B, heads, H*W, audio_len]
        
        # Apply mask if provided
        if attention_mask is not None:
            # attention_mask: [B, audio_len], True = masked
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, audio_len]
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        
        # Softmax
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0)  # NaN 방지
        
        # Apply attention to values
        out = torch.matmul(attn_probs, v)  # [B, heads, H*W, dim_head]
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        # [B, H*W, inner_dim]
        
        # Output projection
        out = self.to_out(out)  # [B, H*W, query_dim]
        
        return out


class AudioAttentionProcessor:
    """
    Attention Processor that adds Audio Cross-Attention to existing Text Cross-Attention
    
    사용법:
    1. U-Net의 각 Cross-Attention layer에 대해 이 processor를 설정
    2. Forward 시 text attention + audio attention 결합
    """
    
    def __init__(
        self,
        query_dim: int,
        audio_dim: int = 768,
        audio_scale: float = 1.0,
        heads: int = 8,
    ):
        self.audio_cross_attn = AudioCrossAttention(
            query_dim=query_dim,
            audio_dim=audio_dim,
            heads=heads,
        )
        self.audio_scale = audio_scale
    
    def __call__(
        self,
        attn,                        # Original attention module
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,  # Text embedding
        audio_hidden_states: torch.Tensor = None,    # Audio embedding
        attention_mask: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        # 1. Original Text Cross-Attention (기존 동작 유지)
        text_out = attn(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # 2. Audio Cross-Attention (새로 추가)
        if audio_hidden_states is not None:
            audio_out = self.audio_cross_attn(
                hidden_states=hidden_states,
                audio_embed=audio_hidden_states,
            )
            # 결합
            out = text_out + self.audio_scale * audio_out
        else:
            out = text_out
        
        return out


class AudioIPAdapter(nn.Module):
    """
    IP-Adapter 스타일의 Audio Adapter
    
    U-Net의 여러 layer에 Audio Cross-Attention을 추가
    """
    
    def __init__(
        self,
        unet_channels: list = [320, 640, 1280, 1280],  # U-Net channel dimensions
        audio_dim: int = 768,
        num_audio_tokens: int = 77,  # Audio embedding sequence length
        audio_scale: float = 1.0,
        num_heads: int = 8,
    ):
        super().__init__()
        
        self.audio_scale = audio_scale
        self.num_audio_tokens = num_audio_tokens
        
        # 각 U-Net block에 대한 Audio Cross-Attention
        # down_blocks: 3개, mid_block: 1개, up_blocks: 3개
        self.down_audio_attns = nn.ModuleList([
            AudioCrossAttention(query_dim=ch, audio_dim=audio_dim, heads=num_heads)
            for ch in unet_channels[:3]  # down_blocks
        ])
        
        self.mid_audio_attn = AudioCrossAttention(
            query_dim=unet_channels[-1],  # mid_block
            audio_dim=audio_dim,
            heads=num_heads
        )
        
        self.up_audio_attns = nn.ModuleList([
            AudioCrossAttention(query_dim=ch, audio_dim=audio_dim, heads=num_heads)
            for ch in reversed(unet_channels[:3])  # up_blocks (reversed)
        ])
        
        # Audio embedding projection (optional: 더 나은 표현 학습)
        self.audio_proj = nn.Sequential(
            nn.LayerNorm(audio_dim),
            nn.Linear(audio_dim, audio_dim),
            nn.GELU(),
            nn.Linear(audio_dim, audio_dim),
        )
    
    def forward(
        self,
        audio_embed: torch.Tensor,
        block_type: str,  # "down", "mid", "up"
        block_idx: int,   # block index
        hidden_states: torch.Tensor,  # U-Net hidden states
    ) -> torch.Tensor:
        """
        Args:
            audio_embed: [B, seq_len, audio_dim]
            block_type: "down", "mid", or "up"
            block_idx: index of the block
            hidden_states: [B, H*W, C] from U-Net
        
        Returns:
            Audio attention output to be added to text attention output
        """
        # Project audio embedding
        audio_embed = self.audio_proj(audio_embed)
        
        # Select appropriate attention layer
        if block_type == "down":
            attn_layer = self.down_audio_attns[block_idx]
        elif block_type == "mid":
            attn_layer = self.mid_audio_attn
        elif block_type == "up":
            attn_layer = self.up_audio_attns[block_idx]
        else:
            raise ValueError(f"Unknown block_type: {block_type}")
        
        # Apply audio cross-attention
        audio_out = attn_layer(hidden_states, audio_embed)
        
        return self.audio_scale * audio_out


class TemporalAudioIPAdapter(nn.Module):
    """
    Temporal-aware Audio IP-Adapter for Storyboard Generation
    
    4개의 kishōtenketsu segment를 고려한 Audio Adapter
    각 프레임이 해당하는 시간 구간의 audio에 더 집중하도록 설계
    """
    
    def __init__(
        self,
        unet_channels: list = [320, 640, 1280, 1280],
        audio_dim: int = 768,
        num_frames: int = 4,
        num_audio_tokens: int = 77,
        audio_scale: float = 1.0,
        num_heads: int = 8,
    ):
        super().__init__()
        
        self.num_frames = num_frames
        self.audio_scale = audio_scale
        self.tokens_per_segment = num_audio_tokens // num_frames  # 19
        
        # 공유 Audio Cross-Attention layers
        self.audio_attns = nn.ModuleDict({
            'down_0': AudioCrossAttention(unet_channels[0], audio_dim, num_heads),
            'down_1': AudioCrossAttention(unet_channels[1], audio_dim, num_heads),
            'down_2': AudioCrossAttention(unet_channels[2], audio_dim, num_heads),
            'mid': AudioCrossAttention(unet_channels[3], audio_dim, num_heads),
            'up_0': AudioCrossAttention(unet_channels[2], audio_dim, num_heads),
            'up_1': AudioCrossAttention(unet_channels[1], audio_dim, num_heads),
            'up_2': AudioCrossAttention(unet_channels[0], audio_dim, num_heads),
        })
        
        # Frame-specific projection (각 프레임마다 다른 projection 가능)
        self.frame_proj = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(audio_dim),
                nn.Linear(audio_dim, audio_dim),
            )
            for _ in range(num_frames)
        ])
    
    def get_frame_audio_embed(
        self,
        audio_embed: torch.Tensor,  # [B, 77, 768]
        frame_idx: int,
    ) -> torch.Tensor:
        """
        특정 프레임에 해당하는 audio segment 추출
        
        Audio: [====Seg0====|====Seg1====|====Seg2====|====Seg3====]
                  Frame0       Frame1       Frame2       Frame3
        """
        start = frame_idx * self.tokens_per_segment
        end = start + self.tokens_per_segment
        if frame_idx == self.num_frames - 1:
            end = audio_embed.shape[1]  # 마지막 프레임은 나머지 포함
        
        segment = audio_embed[:, start:end, :]  # [B, ~19, 768]
        
        # Frame-specific projection
        segment = self.frame_proj[frame_idx](segment)
        
        return segment
    
    def forward(
        self,
        audio_embed: torch.Tensor,   # [B, 77, 768]
        block_name: str,             # e.g., "down_0", "mid", "up_2"
        hidden_states: torch.Tensor, # [B*num_frames, H*W, C]
        frame_idx: int = None,       # Optional: specific frame
    ) -> torch.Tensor:
        """
        Temporal-aware audio attention
        
        각 프레임이 해당 시간 구간의 audio에 집중
        """
        B_total = hidden_states.shape[0]
        B = B_total // self.num_frames
        
        attn_layer = self.audio_attns[block_name]
        
        outputs = []
        for f in range(self.num_frames):
            # 해당 프레임의 hidden states
            h = hidden_states[f * B:(f + 1) * B]  # [B, H*W, C]
            
            # 해당 프레임의 audio segment
            audio_seg = self.get_frame_audio_embed(audio_embed, f)  # [B, ~19, 768]
            
            # Audio attention
            out = attn_layer(h, audio_seg)  # [B, H*W, C]
            outputs.append(out)
        
        # Concatenate back
        output = torch.cat(outputs, dim=0)  # [B*num_frames, H*W, C]
        
        return self.audio_scale * output
