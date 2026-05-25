"""Inference script for single image or folder."""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from src.dataset import get_transforms
from src.model import build_model
from src.utils import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Predict AI-generated image')
    parser.add_argument('--image', type=str, default=None, help='Path to single image')
    parser.add_argument('--folder', type=str, default=None, help='Path to folder of images')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint .pt')
    parser.add_argument('--config', type=str, default=None, help='Path to config yaml')
    parser.add_argument('--device', type=str, default='auto', help='Device')
    parser.add_argument('--threshold', type=float, default=0.5, help='Decision threshold')
    return parser.parse_args()


def predict_image(model, transform, image_path, device, threshold=0.5):
    img = Image.open(image_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred = 1 if probs[1] >= threshold else 0
    label = 'real' if pred == 1 else 'fake'
    return {
        'path': str(image_path),
        'prob_fake': float(probs[0]),
        'prob_real': float(probs[1]),
        'prediction': label,
        'confidence': float(max(probs)),
    }


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
    if not args.image and not args.folder:
        raise ValueError("Provide either --image or --folder")

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
    model.eval()

    transform = get_transforms(
        img_size=config['data']['img_size'],
        is_train=False,
    )

    results = []
    if args.image:
        results.append(predict_image(model, transform, args.image, device, args.threshold))
    elif args.folder:
        folder = Path(args.folder)
        for img_path in folder.iterdir():
            if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
                try:
                    results.append(predict_image(model, transform, img_path, device, args.threshold))
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

    for r in results:
        print(f"\n{r['path']}")
        print(f"  Prediction: {r['prediction']} (confidence={r['confidence']:.4f})")
        print(f"  Prob fake:  {r['prob_fake']:.4f}")
        print(f"  Prob real:  {r['prob_real']:.4f}")


if __name__ == '__main__':
    main()
