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
    Mel-spectrogram → Audio Embedding that replaces text embedding
    
    핵심 변경: Audio를 Text embedding 공간(77 tokens)에 매핑
    → Frozen UNet이 해석 가능한 형태로 변환
    
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
        output_seq_len: int = 77,  # CLIP text와 동일한 길이
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_seq_len = output_seq_len
        
        # 1. CNN Frontend
        # [B, 1, 128, T] → [B, hidden_dim, 8, T/8]
        self.cnn = nn.Sequential(
            # stride=(2,1): freq 1/2, time 유지
            nn.Conv2d(1, 64, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            
            # stride=(2,2): freq 1/2, time 1/2
            nn.Conv2d(64, 128, kernel_size=3, stride=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            
            # stride=(2,2): freq 1/2, time 1/2
            nn.Conv2d(128, 256, kernel_size=3, stride=(2, 2), padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            
            # stride=(2,2): freq 1/2, time 1/2
            nn.Conv2d(256, hidden_dim, kernel_size=3, stride=(2, 2), padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        # 최종: freq = 128/16 = 8, time = T/8
        
        # 2. Projection
        self.proj = nn.Linear(hidden_dim * 8, hidden_dim)
        
        # 3. Positional Encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
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
        
        # 5. Learnable query tokens (77개) - Text embedding 공간으로 매핑
        self.query_tokens = nn.Parameter(torch.randn(1, output_seq_len, hidden_dim) * 0.02)
        
        # 6. Cross-attention: query_tokens가 audio features를 attend
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 7. Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self._init_weights()
    
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
            mel_mask: [B, T] - True = padding position (for transformer)
        Returns:
            [B, 77, 768] - Text embedding과 같은 shape
        """
        B, n_mels, T = mel.shape
        
        # [B, 128, T] → [B, 1, 128, T]
        x = mel.unsqueeze(1)
        
        # CNN: [B, 1, 128, T] → [B, hidden_dim, 8, T']
        x = self.cnn(x)
        B, C, H, T_new = x.shape
        
        # Reshape: [B, hidden_dim, 8, T'] → [B, T', hidden_dim]
        x = x.permute(0, 3, 1, 2).reshape(B, T_new, C * H)
        x = self.proj(x)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer mask 생성 (mel_mask를 CNN 출력 크기에 맞게 조정)
        transformer_mask = None
        if mel_mask is not None:
            # mel_mask: [B, T] → [B, T'] (CNN downsampling 반영)
            transformer_mask = self._downsample_mask(mel_mask, T_new)
        
        # Transformer (self-attention with mask)
        x = self.transformer(x, src_key_padding_mask=transformer_mask)
        
        # Cross-attention: learnable queries → audio features
        query = self.query_tokens.expand(B, -1, -1)  # [B, 77, hidden_dim]
        
        # key_padding_mask for cross-attention
        attn_output, _ = self.cross_attn(
            query=query,
            key=x,
            value=x,
            key_padding_mask=transformer_mask
        )
        
        # Output projection
        output = self.output_proj(attn_output)  # [B, 77, 768]
        
        return output
    
    def _downsample_mask(self, mask: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Mel mask를 CNN 출력 길이에 맞게 다운샘플링
        [B, T] → [B, T']
        """
        B, T = mask.shape
        
        # Max pooling으로 다운샘플링 (패딩이 하나라도 있으면 패딩으로 처리)
        mask_float = mask.float().unsqueeze(1)  # [B, 1, T]
        downsampled = torch.nn.functional.adaptive_max_pool1d(mask_float, target_len)
        
        return downsampled.squeeze(1).bool()  # [B, T']