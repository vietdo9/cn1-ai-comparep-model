"""Training and evaluation loops with AMP."""
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from .utils import AverageMeter, compute_metrics


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, grad_clip=1.0, use_amp=True, limit_batches=None):
    model.train()
    losses = AverageMeter()
    all_probs = []
    all_targets = []

    scaler = GradScaler('cuda') if use_amp and device.type == 'cuda' else None
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}")

    for batch_idx, (images, targets) in enumerate(pbar):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_size = images.size(0)
        losses.update(loss.item(), batch_size)

        probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
        all_probs.append(probs)
        all_targets.append(targets.cpu().numpy())

        pbar.set_postfix({'loss': f"{losses.avg:.4f}"})

    all_probs = torch.cat([torch.from_numpy(p) for p in all_probs]).numpy()
    all_targets = torch.cat([torch.from_numpy(t) for t in all_targets]).numpy()
    metrics = compute_metrics(all_targets, all_probs)
    metrics['loss'] = losses.avg
    return metrics


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, use_amp=True, phase='Val', limit_batches=None):
    model.eval()
    losses = AverageMeter()
    all_probs = []
    all_targets = []
    all_paths = [] if hasattr(dataloader.dataset, 'samples') else None

    pbar = tqdm(dataloader, desc=f"{phase}")

    for batch_idx, batch in enumerate(pbar):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        if len(batch) == 3:
            images, targets, paths = batch
        else:
            images, targets = batch
            paths = None

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if use_amp and device.type == 'cuda':
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, targets)
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)

        batch_size = images.size(0)
        losses.update(loss.item(), batch_size)

        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(targets.cpu().numpy())

        if paths is not None and all_paths is not None:
            all_paths.extend(paths)

        pbar.set_postfix({'loss': f"{losses.avg:.4f}"})

    all_probs = torch.cat([torch.from_numpy(p) for p in all_probs]).numpy()
    all_targets = torch.cat([torch.from_numpy(t) for t in all_targets]).numpy()
    metrics = compute_metrics(all_targets, all_probs)
    metrics['loss'] = losses.avg
    return metrics, all_probs, all_targets, all_paths
