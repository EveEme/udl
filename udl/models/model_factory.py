import timm
import torch
from torch import nn

from .resnet_cifar import (
    resnet_c_preact_26,
    wide_resnet_c_preact_26_3,
    wide_resnet_c_preact_26_5,
)

LOCAL_MODEL_NAME_TO_MODEL = {
    "wide_resnet_c_preact_26_3": wide_resnet_c_preact_26_3,
    "wide_resnet_c_preact_26_5": wide_resnet_c_preact_26_5,
    "resnet_c_preact_26": resnet_c_preact_26,
}


def create_model(
    num_classes: int,
    device: torch.device,
    model_name: str,
) -> nn.Module:
    """Create model based on dataset.

    Args:
        num_classes: Number of classes in dataset.
        device: Device to place model on.
        model_name: Model name.

    Returns:
        Initialized model on specified device.

    Note:
        If `model_name` starts with 'timm:', a timm model will be created
        (e.g., 'timm:vit_little_patch16_reg4_gap_256.sbb_in1k').
    """
    # Optional override via timm (e.g., 'timm:vit_little_patch16_reg4_gap_256.sbb_in1k')
    if model_name.startswith("timm:"):
        timm_model_name_stripped = model_name.split(":", 1)[1]
        model = timm.create_model(
            timm_model_name_stripped, pretrained=True, num_classes=num_classes
        )
        return model.to(device)

    model = LOCAL_MODEL_NAME_TO_MODEL[model_name](num_classes=num_classes)

    return model.to(device)
