"""Training script for Xception AI image detector."""
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import timm
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset import RobustImageFolder, get_transforms
from src.engine import evaluate, train_one_epoch
from src.model import freeze_backbone, unfreeze_all
from src.utils import build_optimizer, build_scheduler, save_checkpoint, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Train Xception AI Detector')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--train-dir', type=str, default='dataset/train', help='Train folder')
    parser.add_argument('--val-dir', type=str, default='dataset/val', help='Validation folder')
    parser.add_argument('--test-dir', type=str, default='dataset/test', help='Optional test folder')
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Train batch size, RTX 3050 4GB should start with 8')
    parser.add_argument('--batch-size-eval', type=int, default=16, help='Validation/test batch size')
    parser.add_argument('--img-size', type=int, default=299, help='Xception input size')
    parser.add_argument('--lr', type=float, default=None, help='Override head learning rate')
    parser.add_argument('--lr-backbone', type=float, default=None, help='Override backbone learning rate')
    parser.add_argument('--device', type=str, default='auto', help='auto, cuda, or cpu')
    parser.add_argument('--num-workers', type=int, default=2, help='DataLoader workers')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use ImageNet pretrained weights')
    parser.add_argument('--no-pretrained', action='store_false', dest='pretrained', help='Disable pretrained weights')
    parser.add_argument('--limit-batches', type=int, default=None, help='Limit batches per epoch for smoke test')
    return parser.parse_args()


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def resolve_path(path):
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(Path.cwd() / p)


def build_xception(num_classes=2, pretrained=True, dropout=0.2):
    return timm.create_model(
        'xception',
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
    )


def write_metrics_header(metrics_path):
    with open(metrics_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'lr', 'epoch_time',
            'train_loss', 'train_accuracy', 'train_precision', 'train_recall', 'train_f1', 'train_auroc',
            'val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1', 'val_auroc',
        ])


def append_metrics(metrics_path, epoch, lr, epoch_time, train_metrics, val_metrics):
    with open(metrics_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch, lr, epoch_time,
            train_metrics['loss'], train_metrics['accuracy'], train_metrics['precision'], train_metrics['recall'], train_metrics['f1'], train_metrics['auroc'],
            val_metrics['loss'], val_metrics['accuracy'], val_metrics['precision'], val_metrics['recall'], val_metrics['f1'], val_metrics['auroc'],
        ])


def main():
    args = parse_args()
    config = load_config(args.config)

    config['data']['train_dir'] = resolve_path(args.train_dir)
    config['data']['val_dir'] = resolve_path(args.val_dir)
    config['data']['test_dir'] = resolve_path(args.test_dir)
    config['data']['img_size'] = args.img_size
    config['data']['batch_size'] = args.batch_size
    config['data']['batch_size_eval'] = args.batch_size_eval
    config['data']['num_workers'] = args.num_workers
    config['data']['pin_memory'] = True
    config['model']['name'] = 'xception'
    config['model']['pretrained'] = args.pretrained
    config['train']['run_name_prefix'] = 'xception'

    if args.epochs:
        config['train']['epochs'] = args.epochs
    if args.lr:
        config['optimizer']['lr'] = args.lr
    if args.lr_backbone:
        config['optimizer']['lr_backbone'] = args.lr_backbone
    if args.device:
        config['device'] = args.device

    if config['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['device'])

    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        print('CUDA is not available. Check your NVIDIA driver and CUDA-enabled PyTorch installation if you want GPU training.')

    set_seed(config['train']['seed'])

    train_transform = get_transforms(
        img_size=config['data']['img_size'],
        is_train=True,
        aug_config=config.get('augmentation', {}),
    )
    val_transform = get_transforms(img_size=config['data']['img_size'], is_train=False)

    train_dataset = RobustImageFolder(config['data']['train_dir'], transform=train_transform, cache_in_memory=False)
    val_dataset = RobustImageFolder(config['data']['val_dir'], transform=val_transform, cache_in_memory=False)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError('Dataset is empty. Expected folders like dataset/train/<class_name> and dataset/val/<class_name>.')

    num_workers = 0 if args.limit_batches else config['data']['num_workers']
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size_eval'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(num_workers > 0),
    )

    model = build_xception(
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        dropout=config['model'].get('dropout', 0.2),
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config['train'].get('label_smoothing', 0.0))

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{config['train']['run_name_prefix']}_{ts}"
    save_dir = Path(config['train']['save_dir']) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = save_dir / 'tb'
    writer = SummaryWriter(tb_dir)
    metrics_path = save_dir / 'metrics.csv'
    write_metrics_header(metrics_path)

    with open(save_dir / 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f)

    print(f"Run directory: {save_dir}")

    epochs = config['train']['epochs']
    freeze_epochs = config['train'].get('freeze_backbone_epochs', 1)
    best_metric = 0.0 if config['early_stop']['mode'] == 'max' else float('inf')
    patience_counter = 0
    early_stop_patience = config['early_stop']['patience']
    early_stop_metric = config['early_stop']['metric'].replace('val_', '')
    early_stop_mode = config['early_stop']['mode']

    optimizer = None
    scheduler = None
    history_rows = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        if epoch <= freeze_epochs:
            freeze_backbone(model)
            optimizer = build_optimizer(model, config, is_backbone_frozen=True)
            scheduler = build_scheduler(optimizer, config, epochs)
            print(f"[Epoch {epoch}] Backbone frozen, training classifier only")
        elif epoch == freeze_epochs + 1:
            unfreeze_all(model)
            optimizer = build_optimizer(model, config, is_backbone_frozen=False)
            scheduler = build_scheduler(optimizer, config, epochs)
            print(f"[Epoch {epoch}] Unfreezing backbone, fine-tuning all layers")

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            grad_clip=config['train'].get('grad_clip', 1.0),
            use_amp=(device.type == 'cuda'),
            limit_batches=args.limit_batches,
        )
        val_metrics, _, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device,
            use_amp=(device.type == 'cuda'),
            limit_batches=args.limit_batches,
        )

        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]['lr']

        for k, v in train_metrics.items():
            writer.add_scalar(f'train/{k}', v, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f'val/{k}', v, epoch)
        writer.add_scalar('train/lr', lr, epoch)
        writer.add_scalar('train/epoch_time', epoch_time, epoch)

        append_metrics(metrics_path, epoch, lr, epoch_time, train_metrics, val_metrics)
        history_rows.append({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_accuracy': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'val_auroc': val_metrics['auroc'],
        })

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} val_auroc={val_metrics['auroc']:.4f} | "
            f"time={epoch_time:.1f}s"
        )
        print("\nAccuracy table")
        print("-" * 82)
        print(f"{'Epoch':>5} | {'Train Acc':>10} | {'Val Acc':>10} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val AUROC':>10}")
        print("-" * 82)
        for row in history_rows:
            print(
                f"{row['epoch']:>5} | "
                f"{row['train_accuracy'] * 100:>9.2f}% | "
                f"{row['val_accuracy'] * 100:>9.2f}% | "
                f"{row['train_loss']:>10.4f} | "
                f"{row['val_loss']:>10.4f} | "
                f"{row['val_auroc']:>10.4f}"
            )
        print("-" * 82)

        current_metric = val_metrics.get(early_stop_metric, 0.0)
        is_best = current_metric > best_metric if early_stop_mode == 'max' else current_metric < best_metric

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
            'classes': train_dataset.classes,
        }
        save_checkpoint(state, is_best, save_dir, filename='last.pt')

        if patience_counter >= early_stop_patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break

    writer.close()
    print("\nFinal accuracy table")
    print("-" * 82)
    print(f"{'Epoch':>5} | {'Train Acc':>10} | {'Val Acc':>10} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val AUROC':>10}")
    print("-" * 82)
    for row in history_rows:
        print(
            f"{row['epoch']:>5} | "
            f"{row['train_accuracy'] * 100:>9.2f}% | "
            f"{row['val_accuracy'] * 100:>9.2f}% | "
            f"{row['train_loss']:>10.4f} | "
            f"{row['val_loss']:>10.4f} | "
            f"{row['val_auroc']:>10.4f}"
        )
    print("-" * 82)
    print(f"Training complete. Best val_{early_stop_metric}: {best_metric:.4f}")
    print(f"Metrics CSV: {metrics_path}")
    print(f"Checkpoints saved to: {save_dir}")
    print(f"TensorBoard: tensorboard --logdir {tb_dir.parent}")


if __name__ == '__main__':
    main()
