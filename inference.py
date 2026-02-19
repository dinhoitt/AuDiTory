# inference.py
# Audio-to-Storyboard Inference Script

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
import yaml
from PIL import Image

# Audio processing
import librosa

# CLIP (for text conditioning)
try:
    from transformers import CLIPTextModel, CLIPTokenizer
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("⚠️ transformers not installed. Text conditioning will not be available.")

from models.pipeline import AudioToStoryboardPipeline


# ============================================
# Mel-Spectrogram 설정 (전처리와 동일)
# ============================================
MEL_CONFIG = {
    'sr': 24000,
    'n_mels': 128,
    'hop_length': 512,
    'n_fft': 2048,
    'fmin': 0,
    'fmax': 12000
}


def load_checkpoint(checkpoint_path: str, model: AudioToStoryboardPipeline, device: str = 'cuda'):
    """
    체크포인트 로드
    
    Args:
        checkpoint_path: 체크포인트 파일 경로
        model: AudioToStoryboardPipeline 인스턴스
        device: 디바이스
    
    Returns:
        로드된 모델
    """
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Audio Encoder state 로드
    model.audio_encoder.load_state_dict(checkpoint['audio_encoder_state_dict'])
    
    # Null embeddings 로드 (device 명시)
    model.null_audio_embed.data = checkpoint['null_audio_embed'].to(device)
    if 'null_text_embed' in checkpoint:
        model.null_text_embed.data = checkpoint['null_text_embed'].to(device)
    
    # UNet state 로드 (fine-tuning된 경우)
    if 'unet_state_dict' in checkpoint:
        model.storyboard_unet.unet.load_state_dict(checkpoint['unet_state_dict'])
        print("   ✓ UNet state loaded (fine-tuned)")
    
    epoch = checkpoint.get('epoch', 'unknown')
    loss = checkpoint.get('loss', float('inf'))
    print(f"✅ Checkpoint loaded! Epoch: {epoch}, Loss: {loss:.4f}")
    
    return model


def load_audio_to_mel(
    audio_path: str, 
    max_length: int = 2048, 
    device: str = 'cuda'
) -> tuple:
    """
    오디오 파일을 mel-spectrogram으로 변환 (전처리와 동일한 방식)
    
    Args:
        audio_path: 오디오 파일 경로
        max_length: 최대 시간 프레임 수
        device: 디바이스
    
    Returns:
        mel: [1, 128, max_length] 텐서
        mel_mask: [1, max_length] bool 텐서 (True = 패딩 위치)
    """
    print(f"🎵 Loading audio: {audio_path}")
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # 오디오 로드 (전처리와 동일한 sample rate)
    y, sr = librosa.load(audio_path, sr=MEL_CONFIG['sr'])
    
    # 최소 길이 보장 (1초) - 전처리와 동일
    min_samples = MEL_CONFIG['sr']
    if len(y) < min_samples:
        y = np.pad(y, (0, min_samples - len(y)))
    
    # Mel-spectrogram 계산 (전처리와 동일)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=MEL_CONFIG['n_mels'],
        hop_length=MEL_CONFIG['hop_length'],
        n_fft=MEL_CONFIG['n_fft'],
        fmin=MEL_CONFIG['fmin'],
        fmax=MEL_CONFIG['fmax']
    )
    
    # Log scale 변환 (전처리와 동일 - 정규화 없음!)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # 원본 길이 저장
    original_len = mel_db.shape[1]
    
    # Padding/Truncation (Dataset과 동일한 방식)
    if original_len > max_length:
        mel_db = mel_db[:, :max_length]
        original_len = max_length
    else:
        pad_len = max_length - original_len
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_len)), mode='constant', constant_values=0)
    
    # Mask 생성 (True = 패딩 위치, Dataset과 동일)
    mel_mask = np.zeros(max_length, dtype=bool)
    mel_mask[original_len:] = True
    
    # Tensor 변환
    mel = torch.from_numpy(mel_db).float().unsqueeze(0).to(device)      # [1, 128, max_length]
    mel_mask = torch.from_numpy(mel_mask).unsqueeze(0).to(device)       # [1, max_length]
    
    duration = len(y) / sr
    print(f"   ✓ Mel shape: {mel.shape}, Duration: {duration:.2f}s, Original frames: {original_len}")
    
    return mel, mel_mask


def load_text_to_embed(
    text: str, 
    pretrained_model: str = "runwayml/stable-diffusion-v1-5",
    device: str = 'cuda'
) -> torch.Tensor:
    """
    텍스트를 CLIP embedding으로 변환 (전처리와 동일한 방식)
    
    Args:
        text: 텍스트 문자열
        pretrained_model: SD 모델 경로 (tokenizer, text_encoder 사용)
        device: 디바이스
    
    Returns:
        text_embed: [1, 77, 768] 텐서
    """
    if not CLIP_AVAILABLE:
        raise RuntimeError("transformers package required for text conditioning. "
                          "Install with: pip install transformers")
    
    print(f"📝 Encoding text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    
    # SD의 tokenizer와 text_encoder 사용 (전처리와 동일)
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model, subfolder="text_encoder").to(device)
    text_encoder.eval()
    
    # 빈 텍스트 처리 (전처리와 동일)
    if not text:
        text = ""
    
    # 토큰화 및 인코딩 (전처리와 동일)
    inputs = tokenizer(
        text,
        padding="max_length",
        max_length=tokenizer.model_max_length,  # 77
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        text_embed = text_encoder(**inputs).last_hidden_state  # [1, 77, 768]
    
    print(f"   ✓ Text embed shape: {text_embed.shape}")
    
    return text_embed


def save_storyboard(
    images: torch.Tensor, 
    output_dir: str, 
    prefix: str = "storyboard"
) -> list:
    """
    생성된 스토리보드 이미지 저장
    
    Args:
        images: [B, 4, 3, H, W] 텐서 (값 범위: 0~1)
        output_dir: 출력 디렉토리
        prefix: 파일명 prefix
    
    Returns:
        저장된 파일 경로 리스트
    """
    os.makedirs(output_dir, exist_ok=True)
    
    B = images.shape[0]
    saved_files = []
    
    for batch_idx in range(B):
        batch_images = images[batch_idx]  # [4, 3, H, W]
        
        # 개별 프레임 저장
        frame_files = []
        for frame_idx in range(4):
            img = batch_images[frame_idx]  # [3, H, W]
            img = img.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            
            pil_img = Image.fromarray(img)
            filename = f"{prefix}_{batch_idx:03d}_frame{frame_idx}.png"
            filepath = os.path.join(output_dir, filename)
            pil_img.save(filepath)
            frame_files.append(filepath)
        
        # 4프레임 그리드 저장 (2x2 배열)
        H, W = batch_images.shape[2], batch_images.shape[3]
        grid_img = Image.new('RGB', (W * 2, H * 2))
        
        for frame_idx in range(4):
            img = batch_images[frame_idx]
            img = img.permute(1, 2, 0).cpu().numpy()
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img)
            
            # 2x2 그리드 위치 계산
            row, col = frame_idx // 2, frame_idx % 2
            grid_img.paste(pil_img, (col * W, row * H))
        
        grid_filename = f"{prefix}_{batch_idx:03d}_grid.png"
        grid_filepath = os.path.join(output_dir, grid_filename)
        grid_img.save(grid_filepath)
        
        # 가로 배열 그리드도 저장
        horizontal_img = Image.new('RGB', (W * 4, H))
        for frame_idx in range(4):
            img = batch_images[frame_idx]
            img = img.permute(1, 2, 0).cpu().numpy()
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img)
            horizontal_img.paste(pil_img, (frame_idx * W, 0))
        
        horizontal_filename = f"{prefix}_{batch_idx:03d}_horizontal.png"
        horizontal_filepath = os.path.join(output_dir, horizontal_filename)
        horizontal_img.save(horizontal_filepath)
        
        print(f"💾 Saved: {grid_filename} (2x2), {horizontal_filename} (1x4)")
        saved_files.extend(frame_files + [grid_filepath, horizontal_filepath])
    
    return saved_files


def inference(args):
    """추론 실행"""
    
    # Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Config 로드
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    pretrained_model = config['model']['pretrained_model']
    
    # 모델 초기화
    print("\n🔧 Initializing model...")
    model = AudioToStoryboardPipeline(
        pretrained_model=pretrained_model,
        audio_encoder_config=config['model']['audio_encoder'],
        freeze_unet=config['model'].get('freeze_unet', True),
        align_weight=config['model'].get('align_weight', 0.1),
    ).to(device)
    
    # 체크포인트 로드
    model = load_checkpoint(args.checkpoint, model, device)
    model.eval()
    
    # ============================================
    # 입력 준비
    # ============================================
    mel = None
    mel_mask = None
    text_embed = None
    
    # Audio 로드
    if args.audio:
        mel, mel_mask = load_audio_to_mel(
            args.audio,
            max_length=config['data'].get('max_mel_length', 2048),
            device=device
        )
    
    # Text 로드
    if args.text:
        text_embed = load_text_to_embed(
            args.text,
            pretrained_model=pretrained_model,
            device=device
        )
    
    # Conditioning mode 결정
    if mel is not None and text_embed is not None:
        conditioning_mode = "both"
    elif mel is not None:
        conditioning_mode = "audio"
    elif text_embed is not None:
        conditioning_mode = "text"
    else:
        raise ValueError("At least one of --audio or --text must be provided")
    
    print(f"\n🎯 Conditioning mode: {conditioning_mode}")
    
    # ============================================
    # 진단: Audio/Text Embedding 통계 확인
    # ============================================
    print("\n" + "=" * 60)
    print("🔍 Embedding 진단")
    print("=" * 60)
    
    with torch.no_grad():
        if mel is not None:
            # Audio Encoder 출력 확인
            audio_embeds = model.audio_encoder(mel, mel_mask)
            print(f"\n📊 [Audio Encoder 출력]")
            print(f"   Shape: {audio_embeds.shape}")
            print(f"   Mean:  {audio_embeds.mean().item():.4f}")
            print(f"   Std:   {audio_embeds.std().item():.4f}")
            print(f"   Max:   {audio_embeds.max().item():.4f}")
            print(f"   Min:   {audio_embeds.min().item():.4f}")
            
            # 문제 진단
            if torch.isnan(audio_embeds).any():
                print("   ⚠️ WARNING: NaN detected in audio embedding!")
            if torch.isinf(audio_embeds).any():
                print("   ⚠️ WARNING: Inf detected in audio embedding!")
            if audio_embeds.max().abs() > 50:
                print("   ⚠️ WARNING: 값 폭발 가능성! (Max > 50)")
            if audio_embeds.std() < 0.1:
                print("   ⚠️ WARNING: 출력 붕괴 가능성! (Std < 0.1)")
        
        if text_embed is not None:
            print(f"\n📊 [Text Embedding 참고]")
            print(f"   Shape: {text_embed.shape}")
            print(f"   Mean:  {text_embed.mean().item():.4f}")
            print(f"   Std:   {text_embed.std().item():.4f}")
            print(f"   Max:   {text_embed.max().item():.4f}")
            print(f"   Min:   {text_embed.min().item():.4f}")
        
        # 비교 분석
        if mel is not None and text_embed is not None:
            print(f"\n📊 [비교 분석]")
            audio_mean = audio_embeds.mean().item()
            audio_std = audio_embeds.std().item()
            text_mean = text_embed.mean().item()
            text_std = text_embed.std().item()
            
            mean_diff = abs(audio_mean - text_mean)
            std_ratio = audio_std / (text_std + 1e-8)
            
            print(f"   Mean 차이: {mean_diff:.4f}")
            print(f"   Std 비율 (Audio/Text): {std_ratio:.4f}")
            
            if mean_diff > 1.0:
                print("   ⚠️ Mean 차이가 큼 - 분포 불일치 가능성")
            if std_ratio > 3.0 or std_ratio < 0.3:
                print("   ⚠️ Std 비율 이상 - 분포 불일치 가능성")
    
    print("=" * 60)
    
    # ============================================
    # 생성
    # ============================================
    print(f"\n🎨 Generating storyboard...")
    print(f"   Steps: {args.steps}")
    print(f"   Guidance scale: {args.guidance}")
    
    # Generator 설정 (재현성)
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
    
    # Consistent attention 설정 (기본값: True)
    use_consistent = not getattr(args, 'no_consistent_attention', False)
    print(f"   Consistent attention: {'enabled' if use_consistent else 'disabled'}")
    
    with torch.no_grad():
        images = model.generate(
            audio_embed=audio_embeds if mel is not None else None,
            text_embed=text_embed,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            conditioning_mode=conditioning_mode,
            generator=generator,
        )
    
    # Add batch dimension if needed: [4, 3, H, W] -> [1, 4, 3, H, W]
    if images.dim() == 4:
        images = images.unsqueeze(0)
    
    print(f"✅ Generated images shape: {images.shape}")  # [B, 4, 3, H, W]
    
    # ============================================
    # 저장
    # ============================================
    saved_files = save_storyboard(images, args.output, prefix=args.prefix)
    
    print(f"\n🎉 Done! Results saved to: {args.output}")
    print(f"   Total files: {len(saved_files)}")
    
    return saved_files


def main():
    parser = argparse.ArgumentParser(
        description='Audio-to-Storyboard Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Audio only
  python inference.py --checkpoint best_model.pt --audio music.wav
  
  # Text only
  python inference.py --checkpoint best_model.pt --text "A hero's journey"
  
  # Both audio and text
  python inference.py --checkpoint best_model.pt --audio music.wav --text "Epic adventure"
  
  # With custom settings
  python inference.py --checkpoint best_model.pt --audio music.wav \\
      --steps 100 --guidance 9.0 --seed 42 --output ./results
        """
    )
    
    # Required
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')
    
    # Input (at least one required)
    parser.add_argument('--audio', type=str, default=None,
                        help='Path to audio file (wav, mp3, flac, etc.)')
    parser.add_argument('--text', type=str, default=None,
                        help='Text prompt for conditioning')
    
    # Config
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='Path to config file (default: configs/train_config.yaml)')
    
    # Generation settings
    parser.add_argument('--steps', type=int, default=50,
                        help='Number of denoising steps (default: 50)')
    parser.add_argument('--guidance', type=float, default=7.5,
                        help='CFG guidance scale (default: 7.5)')
    parser.add_argument('--no_consistent_attention', action='store_true',
                        help='Disable consistent self-attention (enabled by default)')
    
    # Output
    parser.add_argument('--output', type=str, default='./outputs',
                        help='Output directory (default: ./outputs)')
    parser.add_argument('--prefix', type=str, default='storyboard',
                        help='Output filename prefix (default: storyboard)')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validation
    if args.audio is None and args.text is None:
        parser.error("At least one of --audio or --text must be provided")
    
    # Seed 설정
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"🎲 Random seed set: {args.seed}")
    
    # 정보 출력
    print("=" * 60)
    print("🎬 Audio-to-Storyboard Inference")
    print("=" * 60)
    print(f"📄 Config:     {args.config}")
    print(f"📦 Checkpoint: {args.checkpoint}")
    if args.audio:
        print(f"🎵 Audio:      {args.audio}")
    if args.text:
        print(f"📝 Text:       {args.text}")
    print(f"🎨 Steps:      {args.steps}")
    print(f"🎯 Guidance:   {args.guidance}")
    print(f"💾 Output:     {args.output}")
    if args.seed is not None:
        print(f"🎲 Seed:       {args.seed}")
    print("=" * 60)
    
    # 추론 실행
    inference(args)


if __name__ == "__main__":
    main()