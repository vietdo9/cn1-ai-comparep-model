"""Standalone visualization script that generates all TensorBoard views from a checkpoint."""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset import RobustImageFolder, get_transforms
from src.engine import evaluate
from src.model import build_model
from src.utils import load_checkpoint
from src.viz import (
    fig_to_tensor,
    plot_calibration,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_prob_distribution,
    plot_roc_curve,
    plot_threshold_sweep,
    plot_worst_predictions,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate TensorBoard visualizations')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint .pt')
    parser.add_argument('--config', type=str, default=None, help='Path to config yaml')
    parser.add_argument('--device', type=str, default='auto', help='Device')
    parser.add_argument('--num-embed', type=int, default=2000, help='Number of samples for embedding projector')
    return parser.parse_args()


@torch.no_grad()
def extract_embeddings(model, dataloader, device, max_samples=2000):
    """Extract feature embeddings before classifier."""
    model.eval()
    features = []
    labels = []
    images = []
    count = 0
    for imgs, targets in dataloader:
        imgs = imgs.to(device, non_blocking=True)
        # Hook into global pool output for EfficientNet
        feats = None

        def hook(module, input, output):
            nonlocal feats
            feats = output

        handle = None
        for name, module in model.named_modules():
            if name.endswith('global_pool') or name == 'global_pool':
                handle = module.register_forward_hook(hook)
                break
        if handle is None:
            # fallback: just use model output logits
            out = model(imgs)
            feats = out
        else:
            model(imgs)
            handle.remove()

        if feats.dim() > 2:
            feats = feats.view(feats.size(0), -1)

        features.append(feats.cpu().numpy())
        labels.append(targets.numpy())
        images.append(imgs.cpu())

        count += imgs.size(0)
        if count >= max_samples:
            break

    features = np.concatenate(features)[:max_samples]
    labels = np.concatenate(labels)[:max_samples]
    images = torch.cat(images)[:max_samples]
    return features, labels, images


def resolve_paths(config, config_path):
    from pathlib import Path
    config_dir = Path(config_path).parent
    for key in ('train_dir', 'val_dir', 'test_dir'):
        if key in config['data']:
            p = Path(config['data'][key])
            if not p.is_absolute():
                config['data'][key] = str(config_dir.parent / p)
    return config


def main():
    args = parse_args()

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        config = resolve_paths(config, args.config)
    else:
        config = checkpoint.get('config', {})

    device_str = args.device if args.device != 'auto' else config.get('device', 'auto')
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    print(f"Using device: {device}")

    model = build_model(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=False,
        dropout=config['model'].get('dropout', 0.0),
    )
    model, _, _, _, _ = load_checkpoint(args.checkpoint, model, device=device)
    model = model.to(device)

    test_transform = get_transforms(
        img_size=config['data']['img_size'],
        is_train=False,
    )
    test_dataset = RobustImageFolder(config['data']['test_dir'], transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size_eval'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
    )

    criterion = torch.nn.CrossEntropyLoss()
    metrics, all_probs, all_targets, _ = evaluate(
        model, test_loader, criterion, device,
        use_amp=(device.type == 'cuda'),
        phase='TestViz',
    )
    y_pred = (all_probs[:, 1] >= 0.5).astype(int)

    print("Test metrics:", metrics)

    save_dir = Path(args.checkpoint).parent
    tb_dir = save_dir / 'tb'
    writer = SummaryWriter(tb_dir)

    # ROC
    roc_fig, roc_auc = plot_roc_curve(all_targets, all_probs)
    writer.add_image('test/roc_curve', fig_to_tensor(roc_fig), 0)
    writer.add_scalar('test/roc_auc', roc_auc, 0)

    # PR
    pr_fig, pr_auc = plot_pr_curve(all_targets, all_probs)
    writer.add_image('test/pr_curve', fig_to_tensor(pr_fig), 0)
    writer.add_scalar('test/pr_auc', pr_auc, 0)

    # Calibration
    cal_fig = plot_calibration(all_targets, all_probs)
    writer.add_image('test/calibration', fig_to_tensor(cal_fig), 0)

    # Confusion matrix
    cm_fig = plot_confusion_matrix(all_targets, y_pred)
    writer.add_image('test/confusion_matrix', fig_to_tensor(cm_fig), 0)

    # Threshold sweep
    th_fig, best_th = plot_threshold_sweep(all_targets, all_probs)
    writer.add_image('test/threshold_sweep', fig_to_tensor(th_fig), 0)
    writer.add_scalar('test/best_f1_threshold', best_th, 0)

    # Probability distribution
    prob_fig = plot_prob_distribution(all_targets, all_probs)
    writer.add_image('test/prob_distribution', fig_to_tensor(prob_fig), 0)

    # PR curve native
    writer.add_pr_curve('test/pr_curve_native', all_targets, all_probs[:, 1], 0)

    # Worst predictions grids
    grids = plot_worst_predictions(test_dataset, all_targets, all_probs, y_pred, k=16)
    for name, grid in grids.items():
        writer.add_image(f'test/worst_{name}', grid, 0)

    # Embeddings projector
    embed_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size_eval'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
    )
    feats, labels, imgs = extract_embeddings(model, embed_loader, device, max_samples=args.num_embed)
    class_names = ['fake', 'real']
    metadata = [class_names[l] for l in labels]
    # Make thumbnails 64x64 for projector
    thumbs = torch.nn.functional.interpolate(imgs, size=(64, 64), mode='bilinear', align_corners=False)
    writer.add_embedding(feats, metadata=metadata, label_img=thumbs, tag='embeddings/test')

    writer.close()
    print(f"All visualizations logged to: {tb_dir}")
    print(f"Run: tensorboard --logdir {tb_dir.parent}")


if __name__ == '__main__':
    main()
