# models/consistent_attention.py
"""
Consistent Self-Attention for Storyboard Generation
Based on StoryDiffusion: https://github.com/HVision-NKU/StoryDiffusion

배치 내 상호 참조(Batch Mutual Attention) 방식:
- 4장의 프레임이 동시에 생성될 때, 첫 번째 프레임을 참조하여 일관성 유지
- Write/Read 모드 없이 자동으로 작동
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConsistentSelfAttentionProcessor(nn.Module):
    """
    StoryDiffusion Style: Batch-wise Consistent Self-Attention
    
    배치 내의 다른 프레임(이미지)들을 서로 참조하게 함.
    모든 프레임은 자기 자신 + 첫 번째 프레임(Reference)을 참조.
    
    Input shape: [B * num_frames, SeqLen, Dim]
    """
    
    def __init__(
        self,
        num_frames: int = 4,
        attention_mode: str = "first",  # "first" (첫 장 참조) or "mutual" (서로 참조)
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.attention_mode = attention_mode
        self.device = device
        self.dtype = dtype

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            attn: Attention module from diffusers
            hidden_states: [B * num_frames, SeqLen, Dim]
            encoder_hidden_states: Not used for self-attention
            attention_mask: Optional attention mask
            temb: Optional timestep embedding
        """
        residual = hidden_states
        
        # Spatial norm if exists
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        
        input_ndim = hidden_states.ndim
        
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)
        
        # hidden_states: [B * num_frames, SeqLen, Dim]
        batch_size_total, sequence_length, dim = hidden_states.shape
        batch_size = batch_size_total // self.num_frames
        
        # Group norm if exists
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        # 1. Q, K, V Projection
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        
        # Reshape for multi-head attention
        # [B*N, Seq, Dim] -> [B*N, Heads, Seq, HeadDim]
        query = query.view(batch_size_total, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size_total, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size_total, -1, attn.heads, head_dim).transpose(1, 2)
        
        # =========================================================================
        # 🔥 핵심: StoryDiffusion Logic (Consistent Self-Attention)
        # =========================================================================
        # 전략: "모든 프레임(N개)은 자기 자신 + 첫 번째 프레임(Ref)을 본다"
        # Key/Value를 [B, N, Heads, Seq, Dim] 형태로 분리하여 조작
        
        # [B*N, Heads, Seq, Dim] -> [B, N, Heads, Seq, Dim]
        key_reshaped = key.view(batch_size, self.num_frames, attn.heads, -1, head_dim)
        value_reshaped = value.view(batch_size, self.num_frames, attn.heads, -1, head_dim)
        
        if self.attention_mode == "first":
            # Reference Key/Value (첫 번째 프레임) 추출: [B, 1, Heads, Seq, Dim]
            key_ref = key_reshaped[:, 0:1]
            value_ref = value_reshaped[:, 0:1]
            
            # 모든 프레임에 대해 브로드캐스트: [B, N, Heads, Seq, Dim]
            key_ref = key_ref.expand(-1, self.num_frames, -1, -1, -1)
            value_ref = value_ref.expand(-1, self.num_frames, -1, -1, -1)
            
        elif self.attention_mode == "mutual":
            # 모든 프레임의 평균을 참조
            key_ref = key_reshaped.mean(dim=1, keepdim=True).expand(-1, self.num_frames, -1, -1, -1)
            value_ref = value_reshaped.mean(dim=1, keepdim=True).expand(-1, self.num_frames, -1, -1, -1)
        
        # 다시 병합: [B, N, Heads, Seq, Dim] -> [B*N, Heads, Seq, Dim]
        key_ref = key_ref.reshape(batch_size_total, attn.heads, -1, head_dim)
        value_ref = value_ref.reshape(batch_size_total, attn.heads, -1, head_dim)
        
        # Original Key/Value와 Reference Key/Value 결합 (Concatenate)
        # 결과: 각 토큰은 "자기 자신의 정보" + "첫 컷의 정보"를 동시에 봄
        # Key shape: [B*N, Heads, Seq + Seq, Dim]
        key_combined = torch.cat([key_ref, key], dim=2)
        value_combined = torch.cat([value_ref, value], dim=2)
        
        # =========================================================================
        
        # 3. Scaled Dot-Product Attention
        hidden_states = F.scaled_dot_product_attention(
            query, key_combined, value_combined,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False
        )
        
        # Reshape back
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size_total, -1, inner_dim
        )
        hidden_states = hidden_states.to(query.dtype)
        
        # Linear projection
        hidden_states = attn.to_out[0](hidden_states)
        # Dropout
        hidden_states = attn.to_out[1](hidden_states)
        
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
        
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        
        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states


class StandardAttentionProcessor(nn.Module):
    """Standard attention processor (no consistent attention)"""
    
    def __init__(self):
        super().__init__()
    
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        
        residual = hidden_states
        
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        
        input_ndim = hidden_states.ndim
        
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)
        
        batch_size, sequence_length, _ = hidden_states.shape
        
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )
        
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        query = attn.to_q(hidden_states)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        hidden_states = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False
        )
        
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)
        
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
        
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        
        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states


def set_consistent_attention(
    unet,
    num_frames: int = 4,
    attention_mode: str = "first",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> int:
    """
    UNet의 Self-Attention 레이어에 Consistent Self-Attention 적용
    
    Args:
        unet: UNet2DConditionModel
        num_frames: 스토리보드 프레임 수
        attention_mode: "first" (첫 장 참조) or "mutual" (상호 참조)
        device: Device
        dtype: Data type
    
    Returns:
        적용된 프로세서 수
    """
    attn_procs = {}
    consistent_count = 0
    
    for name in unet.attn_processors.keys():
        # Self-attention (attn1) vs Cross-attention (attn2) 구분
        cross_attention_dim = (
            None if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )
        
        if cross_attention_dim is None:
            # Self-attention 레이어
            # Up-blocks에만 적용 (논문에서 권장하는 방식)
            if "up_blocks" in name:
                attn_procs[name] = ConsistentSelfAttentionProcessor(
                    num_frames=num_frames,
                    attention_mode=attention_mode,
                    device=device,
                    dtype=dtype,
                )
                consistent_count += 1
            else:
                attn_procs[name] = StandardAttentionProcessor()
        else:
            # Cross-attention 레이어 - 표준 프로세서 사용
            attn_procs[name] = StandardAttentionProcessor()
    
    unet.set_attn_processor(attn_procs)
    print(f"✅ Consistent Self-Attention 적용 완료 (up_blocks, {consistent_count}개)")
    
    return consistent_count


def remove_consistent_attention(unet):
    """Consistent Attention을 제거하고 표준 Attention으로 복원"""
    attn_procs = {}
    
    for name in unet.attn_processors.keys():
        attn_procs[name] = StandardAttentionProcessor()
    
    unet.set_attn_processor(attn_procs)
    print("✅ Standard Attention으로 복원 완료")
