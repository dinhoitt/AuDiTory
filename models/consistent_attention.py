# models/consistent_attention.py
"""
Consistent Self-Attention for Storyboard Generation
Based on StoryDiffusion: https://github.com/HVision-NKU/StoryDiffusion

Adapted for SD v1.5 and 4-frame storyboard generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import copy


class ConsistentSelfAttentionProcessor(nn.Module):
    """
    Consistent Self-Attention Processor for maintaining character consistency
    across multiple frames in storyboard generation.
    
    Key mechanism:
    1. During 'write' phase: Store self-attention features from reference frames
    2. During 'read' phase: Concatenate stored features for cross-frame attention
    """
    
    def __init__(
        self,
        hidden_size: int = None,
        cross_attention_dim: int = None,
        num_frames: int = 4,  # 스토리보드의 프레임 수
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "ConsistentSelfAttentionProcessor requires PyTorch 2.0+, "
                "please upgrade PyTorch."
            )
        
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_frames = num_frames
        
        # Feature bank for storing attention features
        self.feature_bank: Dict[int, torch.Tensor] = {}
        
    def clear_bank(self):
        """Clear the feature bank"""
        self.feature_bank = {}
    
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        write_mode: bool = True,
        cur_step: int = 0,
        sa_strength: float = 0.5,
    ) -> torch.Tensor:
        """
        Args:
            attn: Attention module
            hidden_states: [B*num_frames, seq_len, hidden_dim]
            encoder_hidden_states: Not used for self-attention
            attention_mask: Optional attention mask
            temb: Optional timestep embedding
            write_mode: If True, store features; if False, use stored features
            cur_step: Current denoising step
            sa_strength: Strength of consistent self-attention (0~1)
        """
        
        residual = hidden_states
        
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        
        input_ndim = hidden_states.ndim
        
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)
        
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        
        # Group norm
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        # Store or retrieve features based on mode
        if write_mode:
            # Write mode: Store features for this step
            self.feature_bank[cur_step] = hidden_states.detach().clone()
            key_value_states = hidden_states
        else:
            # Read mode: Concatenate stored features with current features
            if cur_step in self.feature_bank:
                stored_features = self.feature_bank[cur_step].to(hidden_states.device)
                # Concatenate stored features with current features
                key_value_states = torch.cat([stored_features, hidden_states], dim=1)
            else:
                key_value_states = hidden_states
        
        # Compute Q, K, V
        query = attn.to_q(hidden_states)
        key = attn.to_k(key_value_states)
        value = attn.to_v(key_value_states)
        
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        
        # Reshape for multi-head attention
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
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


def set_consistent_attention_processor(
    unet,
    num_frames: int = 4,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> int:
    """
    Set Consistent Self-Attention processors on UNet's self-attention layers.
    
    Args:
        unet: UNet2DConditionModel
        num_frames: Number of frames in storyboard
        device: Device for attention processor
        dtype: Data type
    
    Returns:
        Number of consistent attention processors set
    """
    attn_procs = {}
    consistent_count = 0
    
    for name in unet.attn_processors.keys():
        # Check if this is a self-attention layer (attn1)
        cross_attention_dim = (
            None if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )
        
        if cross_attention_dim is None:
            # Self-attention layer - apply consistent attention to up_blocks only
            # (following StoryDiffusion's approach)
            if name.startswith("up_blocks"):
                attn_procs[name] = ConsistentSelfAttentionProcessor(
                    num_frames=num_frames,
                    device=device,
                    dtype=dtype
                )
                consistent_count += 1
            else:
                attn_procs[name] = StandardAttentionProcessor()
        else:
            # Cross-attention layer - use standard processor
            attn_procs[name] = StandardAttentionProcessor()
    
    unet.set_attn_processor(attn_procs)
    print(f"✅ Set {consistent_count} Consistent Self-Attention processors (up_blocks)")
    
    return consistent_count


def clear_attention_bank(unet):
    """Clear all feature banks in consistent attention processors"""
    for name, processor in unet.attn_processors.items():
        if isinstance(processor, ConsistentSelfAttentionProcessor):
            processor.clear_bank()


def set_attention_write_mode(unet, write_mode: bool):
    """Set write/read mode for all consistent attention processors"""
    for name, processor in unet.attn_processors.items():
        if isinstance(processor, ConsistentSelfAttentionProcessor):
            processor.write_mode = write_mode


class ConsistentAttentionManager:
    """
    Manager class for Consistent Self-Attention during training and inference.
    
    Usage:
        manager = ConsistentAttentionManager(unet, num_frames=4)
        
        # For each batch:
        manager.reset()
        
        # Generate reference frames (write mode)
        with manager.write_mode():
            ref_output = model(ref_frames)
        
        # Generate remaining frames (read mode)
        with manager.read_mode():
            output = model(frames)
    """
    
    def __init__(
        self,
        unet,
        num_frames: int = 4,
        enabled: bool = True,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.unet = unet
        self.num_frames = num_frames
        self.enabled = enabled
        self.device = device
        self.dtype = dtype
        self.cur_step = 0
        self._write_mode = True
        self.processor_count = 0
        
        if enabled:
            self.processor_count = set_consistent_attention_processor(
                unet, num_frames, device, dtype
            )
    
    def reset(self):
        """Reset for new generation"""
        self.cur_step = 0
        if self.enabled:
            clear_attention_bank(self.unet)
    
    def step(self):
        """Advance to next denoising step"""
        self.cur_step += 1
    
    @property
    def write_mode(self):
        return self._write_mode
    
    @write_mode.setter
    def write_mode(self, value: bool):
        self._write_mode = value
        self._update_processors()
    
    def _update_processors(self):
        """Update all processors with current state"""
        for name, processor in self.unet.attn_processors.items():
            if isinstance(processor, ConsistentSelfAttentionProcessor):
                processor.write_mode = self._write_mode
                processor.cur_step = self.cur_step
    
    def set_write(self):
        """Set to write mode (store features)"""
        self.write_mode = True
    
    def set_read(self):
        """Set to read mode (use stored features)"""
        self.write_mode = False
    
    def disable(self):
        """Temporarily disable consistent attention"""
        self.enabled = False
    
    def enable(self):
        """Re-enable consistent attention"""
        self.enabled = True
