# train.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import yaml
import os
import glob
import re
import argparse
from tqdm import tqdm
import wandb
import math

from data.dataset import PreprocessedStoryboardDataset, collate_fn
from models.pipeline import AudioToStoryboardPipeline


def scan_checkpoint(checkpoint_dir, prefix):
    """체크포인트 디렉토리에서 최신 체크포인트 찾기"""
    pattern = os.path.join(checkpoint_dir, prefix + '*')
    checkpoints = glob.glob(pattern)
    if len(checkpoints) == 0:
        return None
    def extract_number(path):
        numbers = re.findall(r'\d+', os.path.basename(path))
        return int(numbers[-1]) if numbers else 0
    checkpoints.sort(key=extract_number)
    return checkpoints[-1]


def load_checkpoint(checkpoint_path, model, optimizer=None, device='cuda'):
    """체크포인트 로드 및 상태 복원"""
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.audio_encoder.load_state_dict(checkpoint['audio_encoder_state_dict'])
    model.null_audio_embed.data = checkpoint['null_audio_embed'].to(device)
    model.null_text_embed.data = checkpoint['null_text_embed'].to(device)
    
    if 'unet_state_dict' in checkpoint:
        model.storyboard_unet.unet.load_state_dict(checkpoint['unet_state_dict'])
        print("   UNet state loaded")
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("   Optimizer state loaded")
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    global_step = checkpoint.get('global_step', 0)
    
    print(f"✅ Checkpoint loaded! Epoch: {epoch}, Loss: {loss:.4f}")
    return epoch, loss, global_step


def get_scheduler_with_warmup(optimizer, warmup_steps, total_steps):
    """Warmup + Cosine Annealing Scheduler"""
    def lr_lambda(current_step):
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)


def normalize_mel(mel):
    """
    Mel-spectrogram 정규화
    전처리에서 power_to_db(ref=np.max) 적용 → 대략 [-80, 0] 범위
    이를 [-1, 1] 범위로 정규화
    """
    # 값 범위 확인
    mel_min = mel.min()
    mel_max = mel.max()
    
    # dB scale로 보이면 정규화 (min이 -50 이하)
    if mel_min < -50:
        # [-80, 0] → [0, 1] → [-1, 1]
        mel = (mel + 80) / 80  # [0, 1]
        mel = mel * 2 - 1       # [-1, 1]
    
    # 극단값 클램핑
    mel = torch.clamp(mel, -5, 5)
    
    return mel


def check_tensor(tensor, name, step=None):
    """텐서의 NaN/Inf 체크 및 통계 출력"""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    if has_nan or has_inf:
        step_str = f" at step {step}" if step else ""
        print(f"⚠️ [{name}]{step_str}: NaN={has_nan}, Inf={has_inf}")
        print(f"   Shape: {tensor.shape}")
        print(f"   Stats: min={tensor.min():.4f}, max={tensor.max():.4f}, mean={tensor.mean():.4f}")
        return False
    return True


def diagnose_nan_location(model, mel, latent, text_embed, mel_mask, device):
    """NaN 발생 위치를 진단하는 함수"""
    print("\n" + "=" * 60)
    print("🔍 NaN 발생 위치 진단 시작")
    print("=" * 60)
    
    with torch.no_grad():
        # 1. 입력 데이터 체크
        print("\n[1] 입력 데이터 체크:")
        print(f"   mel: min={mel.min():.4f}, max={mel.max():.4f}, mean={mel.mean():.4f}")
        print(f"   latent: min={latent.min():.4f}, max={latent.max():.4f}, mean={latent.mean():.4f}")
        print(f"   text_embed: min={text_embed.min():.4f}, max={text_embed.max():.4f}, mean={text_embed.mean():.4f}")
        
        # 2. Mel 정규화 후 체크
        mel_norm = normalize_mel(mel.clone())
        print(f"\n[2] Mel 정규화 후:")
        print(f"   mel_norm: min={mel_norm.min():.4f}, max={mel_norm.max():.4f}, mean={mel_norm.mean():.4f}")
        
        # 3. Audio Encoder 단계별 체크
        print("\n[3] Audio Encoder 단계별 체크:")
        
        audio_encoder = model.audio_encoder
        B, n_mels, T = mel_norm.shape
        
        # CNN
        x = mel_norm.unsqueeze(1)  # [B, 1, 128, T]
        
        # Input BatchNorm (있다면)
        if hasattr(audio_encoder, 'input_norm'):
            x = audio_encoder.input_norm(x)
            print(f"   After input_norm: min={x.min():.4f}, max={x.max():.4f}")
        
        # CNN layers
        x = audio_encoder.cnn(x)
        print(f"   After CNN: min={x.min():.4f}, max={x.max():.4f}, shape={x.shape}")
        
        if torch.isnan(x).any():
            print("   ❌ NaN detected after CNN!")
            return "CNN"
        
        # Reshape & Project
        B, C, H, T_new = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T_new, C * H)
        x = audio_encoder.proj(x)
        print(f"   After proj: min={x.min():.4f}, max={x.max():.4f}, shape={x.shape}")
        
        if torch.isnan(x).any():
            print("   ❌ NaN detected after projection!")
            return "Projection"
        
        # Positional encoding
        x = audio_encoder.audio_pos_encoder(x)
        print(f"   After pos_enc: min={x.min():.4f}, max={x.max():.4f}")
        
        if torch.isnan(x).any():
            print("   ❌ NaN detected after positional encoding!")
            return "PositionalEncoding"
        
        # Transformer
        transformer_mask = None
        if mel_mask is not None:
            transformer_mask = audio_encoder._downsample_mask(mel_mask, T_new)
        
        x = audio_encoder.transformer(x, src_key_padding_mask=transformer_mask)
        print(f"   After transformer: min={x.min():.4f}, max={x.max():.4f}")
        
        if torch.isnan(x).any():
            print("   ❌ NaN detected after Transformer!")
            return "Transformer"
        
        # Segment Cross-Attention
        audio_segments = audio_encoder._split_audio_segments(x, transformer_mask)
        
        segment_outputs = []
        for i in range(audio_encoder.num_segments):
            audio_seg, seg_mask = audio_segments[i]
            query = audio_encoder.segment_queries[i].expand(B, -1, -1)
            query = query + audio_encoder.segment_embed[:, i:i+1, :]
            
            attn_out, _ = audio_encoder.segment_cross_attn[i](
                query=query,
                key=audio_seg,
                value=audio_seg,
                key_padding_mask=seg_mask
            )
            
            if torch.isnan(attn_out).any():
                print(f"   ❌ NaN detected in segment {i} cross-attention!")
                return f"CrossAttention_Segment{i}"
            
            segment_outputs.append(attn_out)
            print(f"   Segment {i}: min={attn_out.min():.4f}, max={attn_out.max():.4f}")
        
        # Combine & Global attention
        combined = torch.cat(segment_outputs, dim=1)
        combined = audio_encoder.query_pos_encoder(combined)
        
        refined, _ = audio_encoder.global_attn(combined, combined, combined)
        combined = combined + refined
        print(f"   After global_attn: min={combined.min():.4f}, max={combined.max():.4f}")
        
        if torch.isnan(combined).any():
            print("   ❌ NaN detected after global attention!")
            return "GlobalAttention"
        
        # Output projection
        output = audio_encoder.output_proj(combined)
        print(f"   After output_proj: min={output.min():.4f}, max={output.max():.4f}")
        
        if torch.isnan(output).any():
            print("   ❌ NaN detected in final output!")
            return "OutputProjection"
        
        print("\n✅ Audio Encoder 전체 정상!")
        print("=" * 60 + "\n")
        return None


def train(args):
    # Config 로드
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    # Dataset
    train_dataset = PreprocessedStoryboardDataset(
        features_dir=config['data']['features_dir'],
        split='train',
        max_mel_length=config['data']['max_mel_length']
    )
    
    val_dataset = PreprocessedStoryboardDataset(
        features_dir=config['data']['features_dir'],
        split='val',
        max_mel_length=config['data']['max_mel_length']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )
    
    # Model - Pipeline에 정의된 파라미터만 전달
    model = AudioToStoryboardPipeline(
        pretrained_model=config['model']['pretrained_model'],
        audio_encoder_config=config['model']['audio_encoder'],
        freeze_unet=config['model'].get('freeze_unet', True),
        align_weight=config['model'].get('align_weight', 1.0),
    ).to(device)
    
    # Trainable parameters
    trainable_params = list(model.audio_encoder.parameters())
    trainable_params += [model.null_audio_embed]
    trainable_params += [model.null_text_embed]
    
    if not config['model'].get('freeze_unet', True):
        trainable_params += list(model.storyboard_unet.unet.parameters())
    
    optimizer = AdamW(
        trainable_params,
        lr=config['training']['learning_rate'],
        weight_decay=0.01
    )
    
    # Gradient accumulation
    grad_accum_steps = config['training']['gradient_accumulation_steps']
    steps_per_epoch = len(train_loader) // grad_accum_steps
    total_steps = steps_per_epoch * config['training']['num_epochs']
    
    # Mixed precision
    use_amp = config.get('mixed_precision', 'fp32') == 'fp16'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    # Output dir
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # ============================================
    # 체크포인트 재개
    # ============================================
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    
    if args.resume:
        checkpoint_path = args.resume if os.path.isfile(args.resume) else os.path.join(config['output_dir'], args.resume)
        if os.path.exists(checkpoint_path):
            start_epoch, _, global_step = load_checkpoint(checkpoint_path, model, optimizer, device)
            start_epoch += 1
            if global_step == 0:
                global_step = start_epoch * steps_per_epoch
            print(f"🔄 Resuming from epoch {start_epoch}, step {global_step}")
        else:
            print(f"⚠️ Checkpoint not found: {checkpoint_path}")
    
    elif args.auto_resume:
        cp_epoch = scan_checkpoint(config['output_dir'], 'checkpoint_epoch_')
        cp_step = scan_checkpoint(config['output_dir'], 'checkpoint_step_')
        
        checkpoint_path = None
        if cp_epoch and cp_step:
            checkpoint_path = cp_epoch if os.path.getmtime(cp_epoch) > os.path.getmtime(cp_step) else cp_step
        elif cp_epoch:
            checkpoint_path = cp_epoch
        elif cp_step:
            checkpoint_path = cp_step
        
        if checkpoint_path:
            start_epoch, _, global_step = load_checkpoint(checkpoint_path, model, optimizer, device)
            start_epoch += 1
            if global_step == 0:
                global_step = start_epoch * steps_per_epoch
            print(f"🔄 Auto-resuming from epoch {start_epoch}, step {global_step}")
        else:
            print("📂 No checkpoint found. Starting from scratch...")
    
    # Scheduler
    scheduler = get_scheduler_with_warmup(
        optimizer,
        warmup_steps=config['training'].get('warmup_steps', 0),
        total_steps=total_steps
    )
    for _ in range(global_step):
        scheduler.step()
    
    # Wandb
    wandb.init(project="audio-to-storyboard", config=config, resume="allow")
    
    # Conditioning mode
    conditioning_mode = config['training'].get('conditioning_mode', 'both')
    print(f"🎯 Conditioning mode: {conditioning_mode}")
    print(f"📊 Align Weight: {model.align_weight}")
    
    # 파라미터 스냅샷
    import copy
    initial_params = {
        name: param.clone().detach().cpu() 
        for name, param in model.audio_encoder.named_parameters()
    }
    initial_null_audio = model.null_audio_embed.clone().detach().cpu()
    initial_null_text = model.null_text_embed.clone().detach().cpu()
    print("📸 Initial parameter snapshot saved")
    
    # ============================================
    # 첫 번째 배치로 NaN 진단 (디버그 모드)
    # ============================================
    if args.diagnose:
        print("\n🔍 Diagnose mode enabled - checking first batch...")
        first_batch = next(iter(train_loader))
        mel = first_batch['mel'].to(device)
        latent = first_batch['latent'].to(device)
        text_embed = first_batch['text_embed'].to(device)
        mel_mask = first_batch['mel_mask'].to(device)
        
        nan_location = diagnose_nan_location(model, mel, latent, text_embed, mel_mask, device)
        
        if nan_location:
            print(f"\n❌ NaN 발생 위치: {nan_location}")
            print("   해당 부분을 수정 후 다시 시도하세요.")
            return
        else:
            print("\n✅ 진단 완료 - NaN 없음. 학습을 시작합니다.")
    
    # Training loop
    print(f"\n🚀 Training: Epoch {start_epoch + 1} ~ {config['training']['num_epochs']}")
    print(f"   Steps per epoch: {steps_per_epoch}, Starting step: {global_step}")
    
    nan_count = 0
    max_nan_count = 50
    diagnosed_nan = False  # 첫 NaN 발생 시 한 번만 진단
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        model.train()
        epoch_loss = 0.0
        valid_batches = 0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            mel = batch['mel'].to(device)
            latent = batch['latent'].to(device)
            text_embed = batch['text_embed'].to(device)
            mel_mask = batch['mel_mask'].to(device)
            
            # ============================================
            # 입력 데이터 검증 및 정규화
            # ============================================
            if torch.isnan(mel).any() or torch.isinf(mel).any():
                print(f"⚠️ NaN/Inf in mel at batch {batch_idx}, skipping...")
                continue
            if torch.isnan(latent).any() or torch.isinf(latent).any():
                print(f"⚠️ NaN/Inf in latent at batch {batch_idx}, skipping...")
                continue
            if torch.isnan(text_embed).any() or torch.isinf(text_embed).any():
                print(f"⚠️ NaN/Inf in text_embed at batch {batch_idx}, skipping...")
                continue
            
            # 🔥 Mel 정규화 적용
            mel = normalize_mel(mel)
            
            # Forward
            with torch.amp.autocast('cuda', enabled=use_amp):
                output = model(
                    mel=mel,
                    latent=latent,
                    text_embed=text_embed,
                    mel_mask=mel_mask,
                    conditioning_mode=conditioning_mode
                )
                loss = output['loss'] / grad_accum_steps
            
            # ============================================
            # NaN Loss 처리
            # ============================================
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                print(f"⚠️ NaN loss at step {global_step} (count: {nan_count}/{max_nan_count})")
                
                # 첫 NaN 발생 시 진단 실행
                if not diagnosed_nan:
                    diagnosed_nan = True
                    nan_location = diagnose_nan_location(model, mel, latent, text_embed, mel_mask, device)
                    if nan_location:
                        print(f"   NaN 발생 위치: {nan_location}")
                
                optimizer.zero_grad()
                
                if nan_count >= max_nan_count:
                    print("❌ Too many NaN losses, stopping!")
                    save_checkpoint(
                        model, optimizer, epoch, float('inf'),
                        os.path.join(config['output_dir'], f'checkpoint_nan_stop_{global_step}.pt'),
                        config['model'].get('freeze_unet', True), global_step
                    )
                    wandb.finish()
                    return
                continue
            else:
                nan_count = 0
            
            # Backward
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            epoch_loss += output['loss'].item()
            valid_batches += 1
            
            # Gradient accumulation step
            if (batch_idx + 1) % grad_accum_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, 0.5)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 0.5)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Logging
                if global_step % 10 == 0:
                    log_dict = {
                        'train/loss': output['loss'].item(),
                        'train/lr': scheduler.get_last_lr()[0],
                        'train/epoch': epoch
                    }
                    if 'diffusion_loss' in output:
                        diff_loss = output['diffusion_loss']
                        log_dict['train/diffusion_loss'] = diff_loss.item() if torch.is_tensor(diff_loss) else diff_loss
                    if 'align_loss' in output:
                        align_loss = output['align_loss']
                        log_dict['train/align_loss'] = align_loss.item() if torch.is_tensor(align_loss) else align_loss
                    wandb.log(log_dict, step=global_step)
                
                # Checkpoint
                if global_step % args.checkpoint_interval == 0 and global_step > 0:
                    save_checkpoint(
                        model, optimizer, epoch, output['loss'].item(),
                        os.path.join(config['output_dir'], f'checkpoint_step_{global_step}.pt'),
                        config['model'].get('freeze_unet', True), global_step
                    )
                    print(f"💾 Checkpoint saved at step {global_step}")
            
            # Progress bar
            postfix = {'loss': f"{output['loss'].item():.4f}", 'lr': f"{scheduler.get_last_lr()[0]:.2e}"}
            if 'align_loss' in output:
                align_val = output['align_loss']
                align_val = align_val.item() if torch.is_tensor(align_val) else align_val
                postfix['align'] = f"{align_val:.4f}"
            pbar.set_postfix(postfix)
        
        # Epoch end
        avg_train_loss = epoch_loss / max(valid_batches, 1)
        print(f"📊 Epoch {epoch+1} - Avg Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        if (epoch + 1) % config['training']['eval_every'] == 0:
            val_loss = validate(model, val_loader, device, scaler, conditioning_mode)
            print(f"📊 Epoch {epoch+1} - Val Loss: {val_loss:.4f}")
            
            wandb.log({'val/loss': val_loss, 'val/epoch': epoch}, step=global_step)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, epoch, val_loss,
                    os.path.join(config['output_dir'], 'best_model.pt'),
                    config['model'].get('freeze_unet', True), global_step
                )
                print(f"💾 Best model saved!")
        
        # Periodic save
        if (epoch + 1) % config['training']['save_every'] == 0:
            save_checkpoint(
                model, optimizer, epoch, avg_train_loss,
                os.path.join(config['output_dir'], f'checkpoint_epoch_{epoch+1}.pt'),
                config['model'].get('freeze_unet', True), global_step
            )
    
    # ============================================
    # 파라미터 변화 분석
    # ============================================
    print("\n" + "=" * 60)
    print("📊 Audio Encoder 파라미터 변화 분석")
    print("=" * 60)
    
    total_change = 0
    total_params = 0
    layer_changes = []
    
    for name, param in model.audio_encoder.named_parameters():
        if name in initial_params:
            change = (param.detach().cpu() - initial_params[name]).abs()
            mean_change = change.mean().item()
            max_change = change.max().item()
            layer_changes.append((name, mean_change, max_change))
            total_change += change.sum().item()
            total_params += param.numel()
    
    layer_changes.sort(key=lambda x: x[1], reverse=True)
    print("\n🔝 Top 5 변화 레이어:")
    for name, mean_c, max_c in layer_changes[:5]:
        print(f"   {name}: mean={mean_c:.6f}, max={max_c:.6f}")
    
    null_audio_change = (model.null_audio_embed.detach().cpu() - initial_null_audio).abs().mean().item()
    null_text_change = (model.null_text_embed.detach().cpu() - initial_null_text).abs().mean().item()
    print(f"\n📍 Null Audio Embed 변화: {null_audio_change:.6f}")
    print(f"📍 Null Text Embed 변화: {null_text_change:.6f}")
    
    avg_change = total_change / max(total_params, 1)
    print(f"\n📈 전체 평균 파라미터 변화: {avg_change:.8f}")
    
    if avg_change < 1e-8:
        print("⚠️ 경고: 파라미터가 거의 변하지 않았습니다!")
    elif avg_change < 1e-5:
        print("✅ 파라미터가 조금 변했습니다. 학습이 천천히 진행 중입니다.")
    else:
        print("✅ 파라미터가 충분히 변했습니다. 학습이 잘 진행되었습니다!")
    
    print("=" * 60)
    print("\n✅ Training complete!")
    wandb.finish()


def save_checkpoint(model, optimizer, epoch, loss, path, freeze_unet, global_step=0):
    """체크포인트 저장"""
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'audio_encoder_state_dict': model.audio_encoder.state_dict(),
        'null_audio_embed': model.null_audio_embed.data.cpu(),
        'null_text_embed': model.null_text_embed.data.cpu(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    
    if not freeze_unet:
        checkpoint['unet_state_dict'] = model.storyboard_unet.unet.state_dict()
    
    torch.save(checkpoint, path)


@torch.no_grad()
def validate(model, val_loader, device, scaler=None, conditioning_mode="both"):
    model.eval()
    total_loss = 0.0
    valid_batches = 0
    use_amp = scaler is not None
    
    for batch in tqdm(val_loader, desc="Validation"):
        mel = batch['mel'].to(device)
        latent = batch['latent'].to(device)
        text_embed = batch['text_embed'].to(device)
        mel_mask = batch['mel_mask'].to(device)
        
        if torch.isnan(mel).any() or torch.isnan(latent).any() or torch.isnan(text_embed).any():
            continue
        
        # Mel 정규화
        mel = normalize_mel(mel)
        
        with torch.amp.autocast('cuda', enabled=use_amp):
            output = model(
                mel=mel,
                latent=latent,
                text_embed=text_embed,
                mel_mask=mel_mask,
                conditioning_mode=conditioning_mode
            )
        
        loss = output['loss']
        
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        
        total_loss += loss.item()
        valid_batches += 1
    
    if valid_batches == 0:
        print("⚠️ All validation batches had NaN loss!")
        return float('inf')
    
    skipped = len(val_loader) - valid_batches
    if skipped > 0:
        print(f"⚠️ Skipped {skipped}/{len(val_loader)} validation batches due to NaN/Inf")
    
    return total_loss / valid_batches


def main():
    print("🎬 Audio-to-Storyboard Training")
    print("=" * 50)
    
    parser = argparse.ArgumentParser(description='Audio-to-Storyboard Training')
    
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file to resume from')
    parser.add_argument('--auto_resume', action='store_true',
                        help='Automatically find and resume from latest checkpoint')
    parser.add_argument('--checkpoint_interval', type=int, default=50,
                        help='Save checkpoint every N steps (default: 50)')
    parser.add_argument('--diagnose', action='store_true',
                        help='Run NaN diagnosis on first batch before training')
    
    args = parser.parse_args()
    
    print(f"📄 Config: {args.config}")
    if args.resume:
        print(f"🔄 Resume from: {args.resume}")
    elif args.auto_resume:
        print(f"🔄 Auto-resume enabled")
    print(f"💾 Checkpoint interval: {args.checkpoint_interval} steps")
    if args.diagnose:
        print(f"🔍 Diagnose mode: enabled")
    print("=" * 50)
    
    train(args)


if __name__ == "__main__":
    main()