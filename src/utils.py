"""Utilities: seeding, metrics, checkpointing, optimizer/scheduler."""
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Compute and store the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_metrics(y_true, y_prob):
    """Compute classification metrics."""
    y_pred = (y_prob[:, 1] >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auroc = roc_auc_score(y_true, y_prob[:, 1])
    except ValueError:
        auroc = 0.0
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc,
    }


def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pt'):
    """Save model checkpoint."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    filepath = save_dir / filename
    torch.save(state, filepath)
    if is_best:
        best_path = save_dir / 'best.pt'
        torch.save(state, best_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cpu'):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    best_metric = checkpoint.get('best_metric', 0.0)
    return model, optimizer, scheduler, epoch, best_metric


def build_optimizer(model, config, is_backbone_frozen=False):
    """Build optimizer with different LRs for backbone and head."""
    lr = config['optimizer']['lr']
    lr_backbone = config['optimizer'].get('lr_backbone', lr)
    weight_decay = config['optimizer']['weight_decay']

    if is_backbone_frozen:
        # Only head parameters need gradients
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        backbone_params, head_params = get_backbone_head_params(model)
        optimizer = AdamW(
            [
                {'params': backbone_params, 'lr': lr_backbone},
                {'params': head_params, 'lr': lr},
            ],
            weight_decay=weight_decay,
        )
    return optimizer


def build_scheduler(optimizer, config, epochs):
    """Build learning rate scheduler."""
    scheduler_name = config['scheduler']['name'].lower()
    if scheduler_name == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=config['scheduler'].get('min_lr', 1e-6),
        )
    else:
        scheduler = None
    return scheduler


def get_backbone_head_params(model):
    """Separate backbone and head parameters."""
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if any(k in name for k in ('classifier', 'head', 'fc')):
            head_params.append(param)
        else:
            backbone_params.append(param)
    return backbone_params, head_params
