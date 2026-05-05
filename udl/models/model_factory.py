import torch
from torch import nn

from .resnet_cifar import (
    resnet_c_preact_26,
)

LOCAL_MODEL_NAME_TO_MODEL = {
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
    """
    model = LOCAL_MODEL_NAME_TO_MODEL[model_name](num_classes=num_classes)

    return model.to(device)
