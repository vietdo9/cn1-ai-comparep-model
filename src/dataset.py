"""Dataset utilities with torchvision transforms and robust image loading."""
import os
import random
from pathlib import Path

import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Suppress DecompressionBombWarning for large images
from torch.utils.data import Dataset
from torchvision import transforms


class RobustImageFolder(Dataset):
    """ImageFolder that skips corrupted images and logs them.
    Supports in-memory caching to reduce disk I/O bottleneck."""

    def __init__(self, root, transform=None, cache_in_memory=False):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.corrupted = []
        self._cache = {} if cache_in_memory else None

        for cls_name in self.classes:
            cls_dir = self.root / cls_name
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
                    self.samples.append((str(img_path), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path, max_size=2048):
        """Load image from disk and safely resize if too large."""
        with Image.open(path) as img:
            img = img.convert('RGB')
            # Resize very large images before any processing to save memory
            if img.width > max_size or img.height > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            return img.copy()

    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            # Use cache if enabled
            if self._cache is not None:
                if index not in self._cache:
                    self._cache[index] = self._load_image(path)
                img = self._cache[index].copy()
            else:
                img = self._load_image(path)

            if self.transform:
                img = self.transform(img)
            return img, target
        except Exception as e:
            if path not in self.corrupted:
                self.corrupted.append(path)
            # Return a dummy image on failure so DataLoader doesn't crash
            dummy = Image.new('RGB', (224, 224), (128, 128, 128))
            if self.transform:
                dummy = self.transform(dummy)
            return dummy, target

    def get_cache_info(self):
        """Return cache statistics."""
        if self._cache is None:
            return "Caching disabled"
        cached = len(self._cache)
        total = len(self.samples)
        return f"Cached {cached}/{total} images ({cached/total*100:.1f}%)"



def get_transforms(img_size=224, is_train=True, aug_config=None):
    """Build torchvision transforms."""
    if aug_config is None:
        aug_config = {}

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_train and aug_config.get('use_train_aug', True):
        crop_scale = aug_config.get('random_crop_scale', [0.8, 1.0])
        hflip = aug_config.get('horizontal_flip_prob', 0.5)
        cj = aug_config.get('color_jitter', {})
        transform_list = [
            transforms.RandomResizedCrop(img_size, scale=tuple(crop_scale)),
            transforms.RandomHorizontalFlip(p=hflip),
        ]
        if cj:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=cj.get('brightness', 0.2),
                    contrast=cj.get('contrast', 0.2),
                    saturation=cj.get('saturation', 0.1),
                    hue=cj.get('hue', 0.05),
                )
            )
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
        return transforms.Compose(transform_list)
    else:
        return transforms.Compose([
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
