# train.py
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import yaml
import os
from tqdm import tqdm
import wandb
import math

from data.dataset import PreprocessedStoryboardDataset, collate_fn
from models.pipeline import AudioToStoryboardPipeline


def get_scheduler_with_warmup(optimizer, warmup_steps, total_steps):
    """Warmup + Cosine Annealing Scheduler"""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine annealing
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)


def train(config_path: str):
    # Config 로드
    with open(config_path, 'r') as f:
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
    
    # Model
    model = AudioToStoryboardPipeline(
        pretrained_model=config['model']['pretrained_model'],
        audio_encoder_config=config['model']['audio_encoder'],
        freeze_unet=config['model']['freeze_unet']
    ).to(device)
    
    # Trainable parameters
    trainable_params = list(model.audio_encoder.parameters())
    trainable_params += [model.null_audio_embed]  # CFG용 null embedding
    trainable_params += [model.null_text_embed]
    
    if not config['model']['freeze_unet']:
        trainable_params += list(model.storyboard_unet.unet.parameters())
    
    optimizer = AdamW(
        trainable_params,
        lr=config['training']['learning_rate'],
        weight_decay=0.01
    )
    
    # Gradient accumulation 반영한 step 계산
    grad_accum_steps = config['training']['gradient_accumulation_steps']
    steps_per_epoch = len(train_loader) // grad_accum_steps
    total_steps = steps_per_epoch * config['training']['num_epochs']
    
    # Scheduler with warmup
    scheduler = get_scheduler_with_warmup(
        optimizer,
        warmup_steps=config['training']['warmup_steps'],
        total_steps=total_steps
    )
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if config['mixed_precision'] == 'fp16' else None
    
    # Wandb
    wandb.init(project="audio-to-storyboard", config=config)
    
    # Output dir
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Conditioning mode
    conditioning_mode = config['training'].get('conditioning_mode', 'both')
    print(f"🎯 Conditioning mode: {conditioning_mode}")
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(config['training']['num_epochs']):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()  # Epoch 시작 시 gradient 초기화
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            mel = batch['mel'].to(device)
            latent = batch['latent'].to(device)
            text_embed = batch['text_embed'].to(device)
            mel_mask = batch['mel_mask'].to(device)
            
            # Forward
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                output = model(
                    mel=mel,
                    latent=latent,
                    text_embed=text_embed,
                    mel_mask=mel_mask,
                    conditioning_mode=conditioning_mode
                )
                # Gradient accumulation을 위해 loss 나누기
                loss = output['loss'] / grad_accum_steps
            
            # Backward (accumulate)
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            epoch_loss += output['loss'].item()  # 원본 loss 기록
            
            # Gradient accumulation step
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
                if global_step % 50 == 0:
                    wandb.log({
                        'train/loss': output['loss'].item(),
                        'train/lr': scheduler.get_last_lr()[0],
                        'train/epoch': epoch
                    }, step=global_step)
            
            pbar.set_postfix({
                'loss': f"{output['loss'].item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
        
        # Epoch end
        avg_train_loss = epoch_loss / len(train_loader)
        print(f"📊 Epoch {epoch+1} - Avg Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        if (epoch + 1) % config['training']['eval_every'] == 0:
            val_loss = validate(model, val_loader, device, scaler, conditioning_mode)
            print(f"📊 Epoch {epoch+1} - Val Loss: {val_loss:.4f}")
            
            wandb.log({
                'val/loss': val_loss,
                'val/epoch': epoch
            }, step=global_step)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, epoch, val_loss,
                    os.path.join(config['output_dir'], 'best_model.pt'),
                    config['model']['freeze_unet']
                )
                print(f"💾 Best model saved!")
        
        # Periodic save
        if (epoch + 1) % config['training']['save_every'] == 0:
            save_checkpoint(
                model, optimizer, epoch, avg_train_loss,
                os.path.join(config['output_dir'], f'checkpoint_epoch_{epoch+1}.pt'),
                config['model']['freeze_unet']
            )
    
    print("✅ Training complete!")
    wandb.finish()


def save_checkpoint(model, optimizer, epoch, loss, path, freeze_unet):
    """체크포인트 저장 (UNet fine-tune 시 UNet도 저장)"""
    checkpoint = {
        'epoch': epoch,
        'audio_encoder_state_dict': model.audio_encoder.state_dict(),
        'null_audio_embed': model.null_audio_embed.data,
        'null_text_embed': model.null_text_embed.data,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    
    # UNet도 학습했다면 저장
    if not freeze_unet:
        checkpoint['unet_state_dict'] = model.storyboard_unet.unet.state_dict()
    
    torch.save(checkpoint, path)


@torch.no_grad()
def validate(model, val_loader, device, scaler=None, conditioning_mode="both"):
    model.eval()
    total_loss = 0.0
    
    for batch in tqdm(val_loader, desc="Validation"):
        mel = batch['mel'].to(device)
        latent = batch['latent'].to(device)
        text_embed = batch['text_embed'].to(device)
        mel_mask = batch['mel_mask'].to(device)
        
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            output = model(
                mel=mel,
                latent=latent,
                text_embed=text_embed,
                mel_mask=mel_mask,
                conditioning_mode=conditioning_mode
            )
        
        total_loss += output['loss'].item()
    
    return total_loss / len(val_loader)


if __name__ == "__main__":
    train("configs/train_config.yaml")