# models/audio_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Transformer용 Positional Encoding"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class AudioEncoder(nn.Module):
    """
    Mel-spectrogram → Audio Embedding (Temporal-Aware)
    
    핵심: 77개 query를 4개 kishōtenketsu 구간으로 분할하여
    시간 순서가 명시적으로 보존되도록 설계
    
    Input:  [B, 128, T] - Mel-spectrogram
    Output: [B, 77, 768] - Text embedding과 같은 shape
    """
    
    def __init__(
        self,
        mel_channels: int = 128,
        hidden_dim: int = 512,
        output_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        output_seq_len: int = 77,
        num_segments: int = 4,  # kishōtenketsu
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_seq_len = output_seq_len
        self.num_segments = num_segments
        
        # 각 segment당 query 수 계산
        self.queries_per_segment = output_seq_len // num_segments  # 77//4 = 19
        self.remainder = output_seq_len % num_segments  # 77%4 = 1 (마지막에 추가)
        
        # 1. CNN Frontend
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=(2, 2), padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            
            nn.Conv2d(256, hidden_dim, kernel_size=3, stride=(2, 2), padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        
        # 2. Projection
        self.proj = nn.Linear(hidden_dim * 8, hidden_dim)
        
        # 3. Positional Encoding (for audio features)
        self.audio_pos_encoder = PositionalEncoding(hidden_dim)
        
        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 5. Segment-specific query tokens (4개 세그먼트 × 각 19~20개)
        self.segment_queries = nn.ParameterList([
            nn.Parameter(torch.randn(1, self._get_segment_size(i), hidden_dim) * 0.02)
            for i in range(num_segments)
        ])
        
        # 6. Segment positional embedding (기/승/전/결 구분)
        self.segment_embed = nn.Parameter(torch.randn(1, num_segments, hidden_dim) * 0.02)
        
        # 7. Query positional encoding (77개 query 내 순서)
        self.query_pos_encoder = PositionalEncoding(hidden_dim, max_len=output_seq_len)
        
        # 8. Segment-wise Cross-attention
        self.segment_cross_attn = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_segments)
        ])
        
        # 9. Global refinement
        self.global_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 10. Output projection (출력 안정화를 위한 LayerNorm 추가)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self._init_weights()
    
    def _get_segment_size(self, segment_idx: int) -> int:
        """각 segment의 query 수 계산"""
        base_size = self.queries_per_segment
        if segment_idx == self.num_segments - 1:
            return base_size + self.remainder
        return base_size
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, mel: torch.Tensor, mel_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            mel: [B, 128, T] - Mel-spectrogram
            mel_mask: [B, T] - True = padding position
        Returns:
            [B, 77, 768] - Temporally-aligned embedding
        """
        B, n_mels, T = mel.shape
        
        # [B, 128, T] → [B, 1, 128, T]
        x = mel.unsqueeze(1)
        
        # CNN: [B, 1, 128, T] → [B, hidden_dim, 8, T']
        x = self.cnn(x)
        B, C, H, T_new = x.shape
        
        # Reshape: [B, T', hidden_dim]
        x = x.permute(0, 3, 1, 2).reshape(B, T_new, C * H)
        x = self.proj(x)
        
        # Audio positional encoding
        x = self.audio_pos_encoder(x)
        
        # Transformer mask
        transformer_mask = None
        if mel_mask is not None:
            transformer_mask = self._downsample_mask(mel_mask, T_new)
            # 🔥 전체가 패딩인 경우 방지: 최소 1개는 유효하게
            transformer_mask = self._ensure_valid_mask(transformer_mask)
        
        # Transformer self-attention
        x = self.transformer(x, src_key_padding_mask=transformer_mask)
        
        # ========== Temporal-Aware Cross-Attention ==========
        audio_segments = self._split_audio_segments(x, transformer_mask)
        
        segment_outputs = []
        for i in range(self.num_segments):
            audio_seg, seg_mask = audio_segments[i]
            
            # Segment query + segment embedding
            query = self.segment_queries[i].expand(B, -1, -1)
            query = query + self.segment_embed[:, i:i+1, :]
            
            # 🔥 안전한 Cross-Attention (패딩 전체 방지)
            attn_out = self._safe_cross_attention(
                query=query,
                key=audio_seg,
                value=audio_seg,
                key_padding_mask=seg_mask,
                attn_module=self.segment_cross_attn[i],
                segment_idx=i
            )
            segment_outputs.append(attn_out)
        
        # Concatenate: [B, 77, hidden_dim]
        combined = torch.cat(segment_outputs, dim=1)
        
        # Query positional encoding
        combined = self.query_pos_encoder(combined)
        
        # Global refinement
        refined, _ = self.global_attn(
            query=combined,
            key=combined,
            value=combined
        )
        combined = combined + refined
        
        # Output projection
        output = self.output_proj(combined)
        
        # 🔥 최종 안전 장치: NaN을 0으로 대체
        if torch.isnan(output).any():
            output = torch.nan_to_num(output, nan=0.0)
        
        return output
    
    def _safe_cross_attention(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        key_padding_mask: torch.Tensor,
        attn_module: nn.MultiheadAttention,
        segment_idx: int
    ) -> torch.Tensor:
        """
        NaN-safe Cross-Attention
        
        패딩이 전부인 경우를 처리하여 NaN 방지
        """
        B = query.shape[0]
        
        if key_padding_mask is not None:
            # 각 배치에서 모든 위치가 패딩인지 확인
            all_masked = key_padding_mask.all(dim=1)  # [B]
            
            if all_masked.any():
                # 전체 패딩인 배치 처리
                safe_mask = key_padding_mask.clone()
                
                # 전체 True인 행은 첫 번째 위치를 False로 (최소 1개는 attend)
                for b in range(B):
                    if all_masked[b]:
                        safe_mask[b, 0] = False
                
                key_padding_mask = safe_mask
        
        # Cross-Attention 실행
        attn_out, _ = attn_module(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask
        )
        
        # NaN 체크 및 대체
        if torch.isnan(attn_out).any():
            # NaN 발생 시 query를 그대로 반환 (fallback)
            attn_out = torch.nan_to_num(attn_out, nan=0.0)
            # 또는 query를 그대로 사용: attn_out = query
        
        return attn_out
    
    def _ensure_valid_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        마스크가 전부 True인 경우 방지
        최소 1개 위치는 False (유효)로 보장
        """
        if mask is None:
            return None
        
        B, T = mask.shape
        all_masked = mask.all(dim=1)  # [B]
        
        if all_masked.any():
            mask = mask.clone()
            for b in range(B):
                if all_masked[b]:
                    # 첫 번째 위치를 유효하게
                    mask[b, 0] = False
        
        return mask
    
    def _split_audio_segments(self, x, mask=None):
        """
        Audio features를 4개 시간 구간으로 분할
        
        Args:
            x: [B, T', hidden_dim]
            mask: [B, T'] or None
        
        Returns:
            List of (audio_segment, segment_mask) tuples
        """
        B, T, C = x.shape
        
        # T가 너무 작으면 패딩
        if T < self.num_segments:
            pad_len = self.num_segments - T
            pad = torch.zeros(B, pad_len, C, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
            if mask is not None:
                mask_pad = torch.ones(B, pad_len, device=mask.device, dtype=mask.dtype)
                mask = torch.cat([mask, mask_pad], dim=1)
            T = self.num_segments
        
        segment_size = T // self.num_segments
        
        segments = []
        for i in range(self.num_segments):
            start = i * segment_size
            end = start + segment_size if i < self.num_segments - 1 else T
            
            audio_seg = x[:, start:end, :]
            
            seg_mask = None
            if mask is not None:
                seg_mask = mask[:, start:end]
                # 🔥 segment mask도 안전하게
                seg_mask = self._ensure_valid_mask(seg_mask)
            
            segments.append((audio_seg, seg_mask))
        
        return segments
    
    def _downsample_mask(self, mask: torch.Tensor, target_len: int) -> torch.Tensor:
        """Mel mask를 CNN 출력 길이에 맞게 다운샘플링"""
        B, T = mask.shape
        
        # bool → float for pooling
        mask_float = mask.float().unsqueeze(1)
        downsampled = F.adaptive_max_pool1d(mask_float, target_len)
        
        return downsampled.squeeze(1).bool()
    
    def get_segment_embeddings(self, mel: torch.Tensor, mel_mask: torch.Tensor = None) -> torch.Tensor:
        """
        S-IMSM 계산용: 4개 segment별 embedding 반환
        
        Returns:
            [B, 4, hidden_dim]
        """
        B, n_mels, T = mel.shape
        
        # Forward pass (output_proj 전까지)
        x = mel.unsqueeze(1)
        x = self.cnn(x)
        B, C, H, T_new = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T_new, C * H)
        x = self.proj(x)
        x = self.audio_pos_encoder(x)
        
        transformer_mask = None
        if mel_mask is not None:
            transformer_mask = self._downsample_mask(mel_mask, T_new)
            transformer_mask = self._ensure_valid_mask(transformer_mask)
        
        x = self.transformer(x, src_key_padding_mask=transformer_mask)
        
        audio_segments = self._split_audio_segments(x, transformer_mask)
        
        segment_embeds = []
        for i in range(self.num_segments):
            audio_seg, seg_mask = audio_segments[i]
            query = self.segment_queries[i].expand(B, -1, -1)
            query = query + self.segment_embed[:, i:i+1, :]
            
            attn_out = self._safe_cross_attention(
                query=query,
                key=audio_seg,
                value=audio_seg,
                key_padding_mask=seg_mask,
                attn_module=self.segment_cross_attn[i],
                segment_idx=i
            )
            
            # 각 segment의 평균
            seg_embed = attn_out.mean(dim=1)  # [B, hidden_dim]
            segment_embeds.append(seg_embed)
        
        return torch.stack(segment_embeds, dim=1)  # [B, 4, hidden_dim]
