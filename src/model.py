"""Model builder using timm."""
import timm
import torch
import torch.nn as nn


def build_model(model_name='tf_efficientnet_b0.ns_jft_in1k', num_classes=2, pretrained=True, dropout=0.2):
    """Create a timm model with a custom classification head."""
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
    )
    return model


def freeze_backbone(model):
    """Freeze all parameters except the classifier head."""
    for name, param in model.named_parameters():
        if 'classifier' not in name and 'head' not in name and 'fc' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True


def unfreeze_all(model):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True


def get_backbone_params(model):
    """Return parameters that belong to the backbone (non-head)."""
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'classifier' in name or 'head' in name or 'fc' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    return backbone_params, head_params
