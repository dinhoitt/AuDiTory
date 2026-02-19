# train.py (Decoupled Cross-Attention version)
"""
Audio-to-Storyboard Training with Decoupled Cross-Attention

주요 변경사항:
1. 새로운 Pipeline 구조 (Audio-Text fusion)
2. 강화된 Align Loss
3. Progressive training 지원
"""

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
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        # Legacy format
        model.audio_encoder.load_state_dict(checkpoint['audio_encoder_state_dict'])
        if 'null_audio_embed' in checkpoint:
            model.null_audio_embed.data = checkpoint['null_audio_embed'].to(device)
        if 'null_text_embed' in checkpoint:
            model.null_text_embed.data = checkpoint['null_text_embed'].to(device)
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            print("   ⚠️ Could not load optimizer state (parameter mismatch)")
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    global_step = checkpoint.get('global_step', 0)
    
    print(f"✅ Checkpoint loaded! Epoch: {epoch}, Loss: {loss:.4f}")
    return epoch, loss, global_step


def get_scheduler_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def normalize_mel(mel):
    """Mel 정규화: [-80, 0] dB → [-1, 1]"""
    if mel.min() < -50:
        mel = (mel + 80) / 80
        mel = mel * 2 - 1
    mel = torch.clamp(mel, -5, 5)
    return mel


def train(args):
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
    
    # Model with new configuration
    model = AudioToStoryboardPipeline(
        pretrained_model=config['model']['pretrained_model'],
        audio_encoder_config=config['model']['audio_encoder'],
        freeze_unet=config['model'].get('freeze_unet', True),
        use_audio_projector=config['model'].get('use_audio_projector', False),
        audio_tokens=config['model'].get('audio_tokens', 16),
        audio_scale=config['model'].get('audio_scale', 1.0),
        align_weight=config['model'].get('align_weight', 1.0),
        fusion_type=config['model'].get('fusion_type', 'add'),
    ).to(device)
    
    # Get trainable parameters
    trainable_params = model.get_trainable_parameters()
    
    print(f"📊 Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
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
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Resume
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    
    if args.resume:
        checkpoint_path = args.resume if os.path.isfile(args.resume) else os.path.join(config['output_dir'], args.resume)
        if os.path.exists(checkpoint_path):
            start_epoch, _, global_step = load_checkpoint(checkpoint_path, model, optimizer, device)
            start_epoch += 1
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
    
    # Scheduler
    scheduler = get_scheduler_with_warmup(
        optimizer,
        warmup_steps=config['training'].get('warmup_steps', 0),
        total_steps=total_steps
    )
    for _ in range(global_step):
        scheduler.step()
    
    # Wandb with resume support
    run_id_file = os.path.join(config['output_dir'], 'wandb_run_id.txt')
    if os.path.exists(run_id_file):
        with open(run_id_file, 'r') as f:
            run_id = f.read().strip()
        print(f"📊 Wandb: Resuming run '{run_id}'")
    else:
        run_id = wandb.util.generate_id()
        with open(run_id_file, 'w') as f:
            f.write(run_id)
        print(f"📊 Wandb: New run '{run_id}'")
    
    wandb.init(
        project="audio-to-storyboard",
        id=run_id,
        name=f"decoupled_{run_id[:8]}",
        config=config,
        resume="allow"
    )
    
    conditioning_mode = config['training'].get('conditioning_mode', 'both')
    print(f"🎯 Conditioning mode: {conditioning_mode}")
    print(f"📊 Fusion type: {config['model'].get('fusion_type', 'add')}")
    print(f"📊 Align Weight: {model.align_weight}")
    print(f"📊 Audio Scale: {model.audio_scale}")
    
    # Training loop
    print(f"\n🚀 Training: Epoch {start_epoch + 1} ~ {config['training']['num_epochs']}")
    
    nan_count = 0
    max_nan_count = 50
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        model.train()
        epoch_loss = 0.0
        epoch_align_loss = 0.0
        valid_batches = 0
        optimizer.zero_grad()
        
        # Fixed align_weight (no progressive training)
        current_align_weight = model.align_weight
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            mel = batch['mel'].to(device)
            latent = batch['latent'].to(device)
            text_embed = batch['text_embed'].to(device)
            mel_mask = batch['mel_mask'].to(device)
            
            # Input validation
            if torch.isnan(mel).any() or torch.isinf(mel).any():
                continue
            if torch.isnan(latent).any() or torch.isinf(latent).any():
                continue
            if torch.isnan(text_embed).any() or torch.isinf(text_embed).any():
                continue
            
            # Normalize mel
            mel = normalize_mel(mel)
            
            # Forward
            with torch.amp.autocast('cuda', enabled=use_amp):
                # Temporarily adjust align_weight for progressive training
                original_align_weight = model.align_weight
                model.align_weight = current_align_weight
                
                output = model(
                    mel=mel,
                    latent=latent,
                    text_embed=text_embed,
                    mel_mask=mel_mask,
                    conditioning_mode=conditioning_mode
                )
                
                model.align_weight = original_align_weight
                
                loss = output['loss'] / grad_accum_steps
            
            # NaN check
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                optimizer.zero_grad()
                if nan_count >= max_nan_count:
                    print("❌ Too many NaN losses, stopping!")
                    save_checkpoint(model, optimizer, epoch, float('inf'),
                                  os.path.join(config['output_dir'], f'checkpoint_nan_stop.pt'),
                                  global_step)
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
            if 'align_loss' in output:
                epoch_align_loss += output['align_loss'].item() if torch.is_tensor(output['align_loss']) else output['align_loss']
            valid_batches += 1
            
            # Gradient step
            if (batch_idx + 1) % grad_accum_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Logging
                if global_step % 10 == 0:
                    log_dict = {
                        'train/loss': output['loss'].item(),
                        'train/diffusion_loss': output['diffusion_loss'].item() if torch.is_tensor(output['diffusion_loss']) else output['diffusion_loss'],
                        'train/align_loss': output['align_loss'].item() if torch.is_tensor(output['align_loss']) else output['align_loss'],
                        'train/lr': scheduler.get_last_lr()[0],
                        'train/epoch': epoch,
                        'train/audio_scale': model.learnable_audio_scale.item(),
                        'train/current_align_weight': current_align_weight,
                    }
                    wandb.log(log_dict, step=global_step)
                
                # Checkpoint
                if global_step % args.checkpoint_interval == 0:
                    save_checkpoint(model, optimizer, epoch, output['loss'].item(),
                                  os.path.join(config['output_dir'], f'checkpoint_step_{global_step}.pt'),
                                  global_step)
                    print(f"💾 Checkpoint saved at step {global_step}")
            
            # Progress bar
            postfix = {
                'loss': f"{output['loss'].item():.4f}",
                'align': f"{output['align_loss'].item() if torch.is_tensor(output['align_loss']) else output['align_loss']:.4f}",
                'scale': f"{model.learnable_audio_scale.item():.3f}",
            }
            pbar.set_postfix(postfix)
        
        # Epoch end
        avg_loss = epoch_loss / max(valid_batches, 1)
        avg_align = epoch_align_loss / max(valid_batches, 1)
        print(f"📊 Epoch {epoch+1} - Loss: {avg_loss:.4f}, Align: {avg_align:.4f}")
        
        # Validation
        if (epoch + 1) % config['training']['eval_every'] == 0:
            val_loss = validate(model, val_loader, device, scaler, conditioning_mode)
            print(f"📊 Epoch {epoch+1} - Val Loss: {val_loss:.4f}")
            
            wandb.log({'val/loss': val_loss}, step=global_step)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch, val_loss,
                              os.path.join(config['output_dir'], 'best_model.pt'),
                              global_step)
                print("💾 Best model saved!")
        
        # Periodic save
        if (epoch + 1) % config['training']['save_every'] == 0:
            save_checkpoint(model, optimizer, epoch, avg_loss,
                          os.path.join(config['output_dir'], f'checkpoint_epoch_{epoch+1}.pt'),
                          global_step)
    
    print("\n✅ Training complete!")
    wandb.finish()


def save_checkpoint(model, optimizer, epoch, loss, path, global_step=0):
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': {
            'audio_encoder': model.audio_encoder.state_dict(),
            'null_audio_embed': model.null_audio_embed.data.cpu(),
            'null_text_embed': model.null_text_embed.data.cpu(),
            'learnable_audio_scale': model.learnable_audio_scale.data.cpu(),
        },
        'audio_encoder_state_dict': model.audio_encoder.state_dict(),
        'null_audio_embed': model.null_audio_embed.data.cpu(),
        'null_text_embed': model.null_text_embed.data.cpu(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    
    # Save additional modules if they exist
    if hasattr(model, 'audio_projector') and model.use_audio_projector:
        checkpoint['model_state_dict']['audio_projector'] = model.audio_projector.state_dict()
    if hasattr(model, 'gate_proj'):
        checkpoint['model_state_dict']['gate_proj'] = model.gate_proj.state_dict()
    if hasattr(model, 'fusion_attn'):
        checkpoint['model_state_dict']['fusion_attn'] = model.fusion_attn.state_dict()
    if hasattr(model, 'audio_cross_attns'):
        checkpoint['model_state_dict']['audio_cross_attns'] = model.audio_cross_attns.state_dict()
    
    torch.save(checkpoint, path)


@torch.no_grad()
def validate(model, val_loader, device, scaler=None, conditioning_mode="both"):
    model.eval()
    total_loss = 0.0
    valid_batches = 0
    
    for batch in tqdm(val_loader, desc="Validation"):
        mel = batch['mel'].to(device)
        latent = batch['latent'].to(device)
        text_embed = batch['text_embed'].to(device)
        mel_mask = batch['mel_mask'].to(device)
        
        if torch.isnan(mel).any() or torch.isnan(latent).any():
            continue
        
        mel = normalize_mel(mel)
        
        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            output = model(
                mel=mel,
                latent=latent,
                text_embed=text_embed,
                mel_mask=mel_mask,
                conditioning_mode=conditioning_mode
            )
        
        if torch.isnan(output['loss']) or torch.isinf(output['loss']):
            continue
        
        total_loss += output['loss'].item()
        valid_batches += 1
    
    if valid_batches == 0:
        return float('inf')
    
    return total_loss / valid_batches


def main():
    print("🎬 Audio-to-Storyboard Training (Decoupled Cross-Attention)")
    print("=" * 60)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_config.yaml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--checkpoint_interval', type=int, default=50)
    
    args = parser.parse_args()
    
    print(f"📄 Config: {args.config}")
    print("=" * 60)
    
    train(args)


if __name__ == "__main__":
    main()