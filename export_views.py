"""Export all evaluation visualizations as standalone PNG files (no TensorBoard needed)."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from src.dataset import RobustImageFolder, get_transforms
from src.engine import evaluate
from src.model import build_model
from src.utils import load_checkpoint
from src.viz import (
    plot_calibration,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_prob_distribution,
    plot_roc_curve,
    plot_threshold_sweep,
    plot_worst_predictions,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Export all visualizations as PNG files')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint .pt')
    parser.add_argument('--config', type=str, default=None, help='Path to config yaml (optional, defaults to one beside checkpoint)')
    parser.add_argument('--device', type=str, default='auto', help='Device')
    parser.add_argument('--out-dir', type=str, default=None, help='Output directory (default: <checkpoint_dir>/views)')
    return parser.parse_args()


def resolve_paths(config, config_path):
    config_dir = Path(config_path).parent
    for key in ('train_dir', 'val_dir', 'test_dir'):
        if key in config['data']:
            p = Path(config['data'][key])
            if not p.is_absolute():
                config['data'][key] = str(config_dir.parent / p)
    return config


def save_fig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def save_grid(grid_tensor, path):
    """Save a CHW or NCHW tensor as image."""
    if grid_tensor.dim() == 4:
        grid_tensor = make_grid(grid_tensor, nrow=4, normalize=True, scale_each=True)
    img = grid_tensor.detach().cpu()
    if img.max() > 1.5:
        img = img / 255.0
    save_image(img, path)
    print(f"  Saved: {path}")


def plot_training_curves(tb_dir, out_dir):
    """Read scalars from TensorBoard event files and save as PNG."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("  Skipping training curves (tensorboard not installed)")
        return

    ea = EventAccumulator(str(tb_dir))
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    if not tags:
        print("  No scalar data found in TensorBoard logs")
        return

    # Group related tags
    groups = {
        'loss': [t for t in tags if 'loss' in t.lower()],
        'accuracy': [t for t in tags if 'acc' in t.lower()],
        'auroc': [t for t in tags if 'auroc' in t.lower() or 'auc' in t.lower()],
        'lr': [t for t in tags if 'lr' in t.lower() or 'learning_rate' in t.lower()],
    }

    for group_name, group_tags in groups.items():
        if not group_tags:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for tag in group_tags:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            ax.plot(steps, values, label=tag, marker='o', markersize=3)
        ax.set_xlabel('Step')
        ax.set_ylabel(group_name)
        ax.set_title(f'Training {group_name.capitalize()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        save_fig(fig, out_dir / f'training_{group_name}.png')


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint)

    # Load config
    if args.config:
        config_path = args.config
    else:
        config_path = ckpt_path.parent / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config = resolve_paths(config, str(config_path))

    # Device
    device_str = args.device if args.device != 'auto' else config.get('device', 'auto')
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    print(f"Using device: {device}")

    # Output directory
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent / 'views'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # Build & load model
    model = build_model(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=False,
        dropout=config['model'].get('dropout', 0.0),
    )
    model, _, _, _, _ = load_checkpoint(str(ckpt_path), model, device=device)
    model = model.to(device)

    # Test dataset
    test_transform = get_transforms(img_size=config['data']['img_size'], is_train=False)
    test_dataset = RobustImageFolder(config['data']['test_dir'], transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size_eval'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
    )
    print(f"Test samples: {len(test_dataset)}")

    # Run evaluation
    criterion = torch.nn.CrossEntropyLoss()
    metrics, all_probs, all_targets, _ = evaluate(
        model, test_loader, criterion, device,
        use_amp=(device.type == 'cuda'),
        phase='Export',
    )
    y_pred = (all_probs[:, 1] >= 0.5).astype(int)

    print("\nTest metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save metrics to txt
    with open(out_dir / 'metrics.txt', 'w', encoding='utf-8') as f:
        f.write("Test set metrics\n")
        f.write("=" * 40 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    print(f"  Saved: {out_dir / 'metrics.txt'}")

    print("\nGenerating plots...")

    # 1. ROC curve
    fig, _ = plot_roc_curve(all_targets, all_probs)
    save_fig(fig, out_dir / '01_roc_curve.png')

    # 2. PR curve
    fig, _ = plot_pr_curve(all_targets, all_probs)
    save_fig(fig, out_dir / '02_pr_curve.png')

    # 3. Confusion matrix
    fig = plot_confusion_matrix(all_targets, y_pred)
    save_fig(fig, out_dir / '03_confusion_matrix.png')

    # 4. Calibration
    fig = plot_calibration(all_targets, all_probs)
    save_fig(fig, out_dir / '04_calibration.png')

    # 5. Threshold sweep
    fig, best_th = plot_threshold_sweep(all_targets, all_probs)
    save_fig(fig, out_dir / '05_threshold_sweep.png')
    print(f"  Best F1 threshold: {best_th:.3f}")

    # 6. Probability distribution
    fig = plot_prob_distribution(all_targets, all_probs)
    save_fig(fig, out_dir / '06_prob_distribution.png')

    # 7. Worst predictions
    grids = plot_worst_predictions(test_dataset, all_targets, all_probs, y_pred, k=16)
    for name, grid in grids.items():
        save_grid(grid, out_dir / f'07_worst_{name}.png')

    # 8. Training curves from TensorBoard logs
    tb_dir = ckpt_path.parent / 'tb'
    if tb_dir.exists():
        print("\nGenerating training curves from TB logs...")
        plot_training_curves(tb_dir, out_dir)

    print(f"\nAll views exported to: {out_dir}")


if __name__ == '__main__':
    main()
