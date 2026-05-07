"""Model utilities."""

import logging
from typing import Any

import torch

from untangle.models import (
    resnet_c_preact_26,
)
from untangle.wrappers import (
    AdaptedLaplaceWrapper,
    CEBaselineWrapper,
    DualLaplaceWrapper,
)

logger = logging.getLogger(__name__)

UNTANGLE_STR_TO_MODEL_CLASS = {
    "resnet_c_preact_26": resnet_c_preact_26,
}


def wrap_model(
    model: torch.nn.Module,
    model_wrapper_name: str,
    reset_classifier: bool,
    weight_paths: list[str],
    num_mc_samples: int,
    num_mc_samples_cv: int,
    checkpoint_path: str,
) -> torch.nn.Module:
    """Wraps the given model with Laplace wrapper.

    Args:
        model: The model to be wrapped.
        model_wrapper_name: Name of the wrapper to use.
        reset_classifier: Whether to reset the classifier.
        weight_paths: Paths to model weights.
        num_mc_samples: Number of Monte Carlo samples.
        num_mc_samples_cv: Number of Monte Carlo samples for cross-validation.
        checkpoint_path: Path to the checkpoint.

    Returns:
        The wrapped Laplace model.

    Raises:
        ValueError: If an unsupported wrapper name is provided.
    """
    if reset_classifier:
        model.reset_classifier(model.num_classes)

    if model_wrapper_name == "ce-baseline":
        wrapped_model = CEBaselineWrapper(model=model)
    elif model_wrapper_name == "adapted-laplace":
        wrapped_model = AdaptedLaplaceWrapper(
            model=model,
            num_mc_samples=num_mc_samples,
            num_mc_samples_cv=num_mc_samples_cv,
            weight_path=weight_paths[0],
        )
    elif model_wrapper_name == "dual-laplace":
        wrapped_model = DualLaplaceWrapper(
            model=model,
            num_mc_samples=num_mc_samples,
            num_mc_samples_cv=num_mc_samples_cv,
            weight_path=weight_paths[0],
        )
    else:
        msg = (
            f'Model wrapper "{model_wrapper_name}" is currently not supported. '
            f'Use "ce-baseline", "adapted-laplace", or "dual-laplace".'
        )
        raise ValueError(msg)

    if checkpoint_path:
        load_checkpoint(wrapped_model, checkpoint_path)

    num_params = sum(param.numel() for param in wrapped_model.parameters())

    logger.info(
        f"Wrapper {model_wrapper_name} created, total param count: {num_params}."
    )
    logger.info(str(wrapped_model))

    return wrapped_model


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
) -> dict[str, Any]:
    """Loads a checkpoint into the given model.

    Args:
        model: The model to load the checkpoint into.
        checkpoint_path: Path to the checkpoint file.

    Returns:
        A dictionary containing information about incompatible keys.

    Raises:
        FileNotFoundError: If no checkpoint is found at the specified path.
    """
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint["state_dict"]
        logger.info(f"Loaded state_dict from checkpoint '{checkpoint_path}'.")
    else:
        msg = f"No checkpoint found at '{checkpoint_path}'"
        raise FileNotFoundError(msg)

    incompatible_keys = model.load_state_dict(state_dict, strict=True)

    return incompatible_keys
