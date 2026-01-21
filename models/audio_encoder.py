# models/audio_encoder.py
import torch
import torch.nn as nn
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
        # 각 segment별로 다른 learnable queries
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
        
        # 9. Global refinement (optional: 전체 context 통합)
        self.global_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 10. Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self._init_weights()
    
    def _get_segment_size(self, segment_idx: int) -> int:
        """각 segment의 query 수 계산"""
        base_size = self.queries_per_segment
        # 마지막 segment에 나머지 추가 (77 = 19*4 + 1)
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
    
    def forward(
        self, 
        mel: torch.Tensor, 
        mel_mask: torch.Tensor = None
    ) -> torch.Tensor:
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
        
        # Transformer self-attention
        x = self.transformer(x, src_key_padding_mask=transformer_mask)
        
        # ========== Temporal-Aware Cross-Attention ==========
        # Audio를 4개 시간 구간으로 분할
        audio_segments = self._split_audio_segments(x, transformer_mask)
        
        # 각 segment별로 cross-attention 수행
        segment_outputs = []
        for i in range(self.num_segments):
            audio_seg, seg_mask = audio_segments[i]
            
            # Segment query + segment embedding
            query = self.segment_queries[i].expand(B, -1, -1)  # [B, 19, hidden_dim]
            query = query + self.segment_embed[:, i:i+1, :]  # segment 정체성 추가
            
            # Cross-attention: query가 해당 시간 구간의 audio만 attend
            attn_out, _ = self.segment_cross_attn[i](
                query=query,
                key=audio_seg,
                value=audio_seg,
                key_padding_mask=seg_mask
            )
            segment_outputs.append(attn_out)
        
        # Concatenate: [B, 19, h] × 4 → [B, 77, hidden_dim]
        # (마지막 segment는 20개일 수 있음)
        combined = torch.cat(segment_outputs, dim=1)  # [B, 77, hidden_dim]
        
        # Query positional encoding 추가 (77개 내 순서)
        combined = self.query_pos_encoder(combined)
        
        # Global refinement: 전체 context 고려 (optional)
        refined, _ = self.global_attn(
            query=combined,
            key=combined,
            value=combined
        )
        combined = combined + refined  # Residual connection
        
        # Output projection
        output = self.output_proj(combined)  # [B, 77, 768]
        
        return output
    
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
        
        # [Safety Check] T가 너무 작으면 강제로 늘림 (Pad)
        if T < self.num_segments:
            pad_len = self.num_segments - T
            pad = torch.zeros(B, pad_len, C, device=x.device)
            x = torch.cat([x, pad], dim=1)
            if mask is not None:
                mask_pad = torch.ones(B, pad_len, device=mask.device).bool() # 패딩 마스크
                mask = torch.cat([mask, mask_pad], dim=1)
            T = self.num_segments # T 업데이트

        segment_size = T // self.num_segments
        
        segments = []
        for i in range(self.num_segments):
            start = i * segment_size
            # 마지막 segment는 나머지 전부 포함
            end = start + segment_size if i < self.num_segments - 1 else T
            
            audio_seg = x[:, start:end, :]  # [B, seg_len, hidden_dim]
            
            seg_mask = None
            if mask is not None:
                seg_mask = mask[:, start:end]  # [B, seg_len]
            
            segments.append((audio_seg, seg_mask))
        
        return segments
    
    def _downsample_mask(self, mask: torch.Tensor, target_len: int) -> torch.Tensor:
        """Mel mask를 CNN 출력 길이에 맞게 다운샘플링"""
        B, T = mask.shape
        mask_float = mask.float().unsqueeze(1)
        downsampled = torch.nn.functional.adaptive_max_pool1d(mask_float, target_len)
        return downsampled.squeeze(1).bool()