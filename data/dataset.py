# data/dataset.py
import torch
from torch.utils.data import Dataset
import json
import os

class PreprocessedStoryboardDataset(Dataset):
    """
    전처리된 Features를 로드하는 Dataset
    """
    
    def __init__(
        self, 
        features_dir: str,
        split: str = "train",
        max_mel_length: int = 2048,
        features_json: str = "dataset_features.json"
    ):
        self.features_dir = features_dir
        self.max_mel_length = max_mel_length
        
        json_path = os.path.join(features_dir, features_json)
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.samples = data['data'][split]
        self.mel_config = data.get('mel_config', {})
        
        print(f"📂 Loaded {split} split: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # torch.load - weights_only 제거, map_location 추가
        latent = torch.load(sample['latent_path'], map_location='cpu')
        mel = torch.load(sample['mel_path'], map_location='cpu')
        text_embed = torch.load(sample['text_path'], map_location='cpu')
        
        # 원본 mel 길이 저장 (mask 생성용)
        original_mel_len = mel.shape[1]
        
        # Mel 패딩 및 마스크 생성
        mel, mel_mask = self._pad_mel(mel, original_mel_len)
        
        return {
            'latent': latent,
            'mel': mel,
            'mel_mask': mel_mask,
            'text_embed': text_embed,
            'id': sample['id'],
            'mel_len': original_mel_len
        }
    
    def _pad_mel(self, mel, original_len):
        """Mel-spectrogram 패딩 및 마스크 생성"""
        _, T = mel.shape
        
        # 마스크: True = 패딩 위치 (Transformer src_key_padding_mask 규칙)
        mel_mask = torch.ones(self.max_mel_length, dtype=torch.bool)
        
        if T > self.max_mel_length:
            mel = mel[:, :self.max_mel_length]
            mel_mask[:] = False  # 전부 유효
        else:
            pad = torch.zeros(mel.shape[0], self.max_mel_length - T)
            mel = torch.cat([mel, pad], dim=1)
            mel_mask[:T] = False  # 유효한 부분
            mel_mask[T:] = True   # 패딩 부분
        
        return mel, mel_mask


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    return {
        'latent': torch.stack([b['latent'] for b in batch]),
        'mel': torch.stack([b['mel'] for b in batch]),
        'mel_mask': torch.stack([b['mel_mask'] for b in batch]),
        'text_embed': torch.stack([b['text_embed'] for b in batch]),
        'id': [b['id'] for b in batch],
        'mel_len': torch.tensor([b['mel_len'] for b in batch], dtype=torch.long)
    }