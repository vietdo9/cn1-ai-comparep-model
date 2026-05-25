"""Visualization helpers for TensorBoard and static plots."""
import io
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image as PILImage
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from torchvision.utils import make_grid


def fig_to_tensor(fig):
    """Convert a matplotlib figure to a CHW tensor."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img = PILImage.open(buf).convert('RGB')
    arr = np.array(img).transpose(2, 0, 1) / 255.0
    plt.close(fig)
    return torch.from_numpy(arr).float()


def plot_roc_curve(y_true, y_prob, num_classes=2):
    """Plot ROC curve."""
    fig, ax = plt.subplots(figsize=(6, 6))
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    return fig, roc_auc


def plot_pr_curve(y_true, y_prob, num_classes=2):
    """Plot Precision-Recall curve."""
    fig, ax = plt.subplots(figsize=(6, 6))
    precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
    pr_auc = auc(recall, precision)
    ax.plot(recall, precision, lw=2, label=f"PR (AUC = {pr_auc:.4f})")
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    return fig, pr_auc


def plot_calibration(y_true, y_prob, n_bins=10):
    """Plot reliability diagram (calibration)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    prob_true, prob_pred = calibration_curve(y_true, y_prob[:, 1], n_bins=n_bins)
    ax.plot(prob_pred, prob_true, 's-', label='Model')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title('Calibration Plot')
    ax.legend(loc='lower right')
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names=('fake', 'real')):
    """Plot confusion matrix (normalized and raw)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    for ax, mat, title in zip(axes, [cm, cm_norm], ['Raw', 'Normalized']):
        im = ax.imshow(mat, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f'Confusion Matrix ({title})')
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                text = f"{mat[i, j]:.2f}" if title == 'Normalized' else str(mat[i, j])
                ax.text(j, i, text, ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax)
    return fig


def plot_threshold_sweep(y_true, y_prob):
    """Plot F1, precision, recall vs threshold."""
    from sklearn.metrics import f1_score, precision_score, recall_score
    thresholds = np.linspace(0.01, 0.99, 100)
    f1s, precisions, recalls = [], [], []
    for th in thresholds:
        y_pred = (y_prob[:, 1] >= th).astype(int)
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, f1s, label='F1')
    ax.plot(thresholds, precisions, label='Precision')
    ax.plot(thresholds, recalls, label='Recall')
    best_idx = int(np.argmax(f1s))
    ax.axvline(thresholds[best_idx], color='r', linestyle='--', label=f'Best F1 th={thresholds[best_idx]:.2f}')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Threshold Sweep')
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    return fig, thresholds[best_idx]


def plot_prob_distribution(y_true, y_prob):
    """Histogram of predicted probabilities for each class."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_prob[y_true == 0, 1], bins=50, alpha=0.6, label='fake (class 0)', color='red', range=(0, 1))
    ax.hist(y_prob[y_true == 1, 1], bins=50, alpha=0.6, label='real (class 1)', color='green', range=(0, 1))
    ax.set_xlabel('Predicted probability for real (class 1)')
    ax.set_ylabel('Count')
    ax.set_title('Probability Distribution')
    ax.legend()
    return fig


def plot_worst_predictions(dataset, y_true, y_prob, y_pred, class_names=('fake', 'real'), k=16):
    """Return grids of worst FP and FN."""
    prob_real = y_prob[:, 1]

    # False positives: real (1) predicted as fake (0), highest prob_fake
    fp_mask = (y_true == 1) & (y_pred == 0)
    fp_indices = np.where(fp_mask)[0]
    fp_sorted = fp_indices[np.argsort(-(1 - prob_real[fp_indices]))][:k]

    # False negatives: fake (0) predicted as real (1), highest prob_real
    fn_mask = (y_true == 0) & (y_pred == 1)
    fn_indices = np.where(fn_mask)[0]
    fn_sorted = fn_indices[np.argsort(-prob_real[fn_indices])][:k]

    grids = {}
    for name, idxs in [('fp', fp_sorted), ('fn', fn_sorted)]:
        if len(idxs) == 0:
            continue
        imgs = []
        for idx in idxs:
            img, _ = dataset[idx]
            imgs.append(img)
        grid = make_grid(torch.stack(imgs), nrow=4, normalize=True, value_range=(0, 1))
        grids[name] = grid
    return grids
