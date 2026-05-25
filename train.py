"""Training script for EfficientNet AI image detector."""
import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset import get_transforms, RobustImageFolder
from src.engine import evaluate, train_one_epoch
from src.model import build_model, freeze_backbone, unfreeze_all
from src.utils import (
    build_optimizer,
    build_scheduler,
    save_checkpoint,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train EfficientNet AI Detector')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    parser.add_argument('--device', type=str, default=None, help='Override device')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--limit-batches', type=int, default=None, help='Limit batches per epoch for smoke test')
    return parser.parse_args()


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def resolve_paths(config, config_path):
    """Resolve relative data paths against the config file's directory."""
    config_dir = Path(config_path).parent
    for key in ('train_dir', 'val_dir', 'test_dir'):
        if key in config['data']:
            p = Path(config['data'][key])
            if not p.is_absolute():
                config['data'][key] = str(config_dir.parent / p)
    return config


def main():
    args = parse_args()
    config = load_config(args.config)
    config = resolve_paths(config, args.config)

    # Overrides from CLI
    if args.epochs:
        config['train']['epochs'] = args.epochs
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.lr:
        config['optimizer']['lr'] = args.lr
    if args.device:
        config['device'] = args.device

    # Device setup
    device_str = config['device']
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    print(f"Using device: {device}")

    # Seed
    set_seed(config['train']['seed'])

    # Transforms
    train_transform = get_transforms(
        img_size=config['data']['img_size'],
        is_train=True,
        aug_config=config.get('augmentation', {}),
    )
    val_transform = get_transforms(
        img_size=config['data']['img_size'],
        is_train=False,
    )

    # Datasets (cache disabled to avoid memory issues with large images)
    train_dataset = RobustImageFolder(config['data']['train_dir'], transform=train_transform, cache_in_memory=False)
    val_dataset = RobustImageFolder(config['data']['val_dir'], transform=val_transform, cache_in_memory=False)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    if train_dataset.corrupted:
        print(f"Corrupted train images skipped: {len(train_dataset.corrupted)}")
    if val_dataset.corrupted:
        print(f"Corrupted val images skipped: {len(val_dataset.corrupted)}")

    # DataLoaders
    num_workers = config['data']['num_workers']
    pin_memory = config['data']['pin_memory']
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size_eval'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    if args.limit_batches:
        config['data']['num_workers'] = 0
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=config['data']['pin_memory'],
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['data']['batch_size_eval'],
            shuffle=False,
            num_workers=0,
            pin_memory=config['data']['pin_memory'],
        )
        print(f"Smoke test mode: limiting to {args.limit_batches} batches, num_workers=0")

    # Model
    model = build_model(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        dropout=config['model'].get('dropout', 0.0),
    )
    model = model.to(device)

    # Loss
    label_smoothing = config['train'].get('label_smoothing', 0.0)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Training run directory
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{config['train']['run_name_prefix']}_{ts}"
    save_dir = Path(config['train']['save_dir']) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = save_dir / 'tb'
    writer = SummaryWriter(tb_dir)
    print(f"Run directory: {save_dir}")

    # Save config
    with open(save_dir / 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f)

    epochs = config['train']['epochs']
    freeze_epochs = config['train'].get('freeze_backbone_epochs', 1)
    best_metric = 0.0
    patience_counter = 0
    early_stop_patience = config['early_stop']['patience']
    early_stop_metric = config['early_stop']['metric']
    early_stop_mode = config['early_stop']['mode']

    start_epoch = 1

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        # 2-stage training: freeze backbone for first N epochs
        if epoch <= freeze_epochs:
            freeze_backbone(model)
            optimizer = build_optimizer(model, config, is_backbone_frozen=True)
            scheduler = build_scheduler(optimizer, config, epochs)
            print(f"[Epoch {epoch}] Backbone frozen, training head only")
        else:
            if epoch == freeze_epochs + 1:
                unfreeze_all(model)
                optimizer = build_optimizer(model, config, is_backbone_frozen=False)
                scheduler = build_scheduler(optimizer, config, epochs)
                print(f"[Epoch {epoch}] Unfreezing backbone, fine-tuning all layers")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            grad_clip=config['train'].get('grad_clip', 1.0),
            use_amp=(device.type == 'cuda'),
            limit_batches=args.limit_batches,
        )

        # Val
        val_metrics, _, _, _ = evaluate(
            model, val_loader, criterion, device,
            use_amp=(device.type == 'cuda'),
            limit_batches=args.limit_batches,
        )

        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - epoch_start

        # Log to TensorBoard
        for k, v in train_metrics.items():
            writer.add_scalar(f'train/{k}', v, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f'val/{k}', v, epoch)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('train/epoch_time', epoch_time, epoch)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} val_auroc={val_metrics['auroc']:.4f} | "
            f"time={epoch_time:.1f}s"
        )

        # Checkpoint
        current_metric = val_metrics.get(early_stop_metric.replace('val_', ''), 0.0)
        is_best = False
        if early_stop_mode == 'max':
            is_best = current_metric > best_metric
        else:
            is_best = current_metric < best_metric

        if is_best:
            best_metric = current_metric
            patience_counter = 0
        else:
            patience_counter += 1

        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_metric': best_metric,
            'config': config,
        }
        save_checkpoint(state, is_best, save_dir, filename='last.pt')

        if patience_counter >= early_stop_patience:
            print(f"Early stopping triggered after {epoch} epochs (no improvement for {patience_counter} epochs)")
            break

    writer.close()
    print(f"Training complete. Best {early_stop_metric}: {best_metric:.4f}")
    print(f"Checkpoints saved to: {save_dir}")
    print(f"TensorBoard: tensorboard --logdir {tb_dir.parent}")


if __name__ == '__main__':
    main()
