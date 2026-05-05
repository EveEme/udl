"""Linearized model definition."""

from copy import deepcopy

import torch
from torch import Tensor, nn
from torch.func import functional_call, jvp


class LinearizedModel(nn.Module):
    """Creates a linearized version of a nn.Module."""

    def __init__(self, model: nn.Module) -> None:
        """Initializes the linearized model.

        Args:
            model: The model to linearize around its current parameters.
        """
        super().__init__()

        model = deepcopy(model)
        model.eval()

        # Store the original model and its parameters
        self.original_model = model
        self.original_buffers = dict(model.named_buffers())
        self.original_params = {
            k: v for k, v in model.named_parameters() if v.requires_grad
        }

        # Initialize trained_map_params to original_params
        # (will be updated if MAP training occurs)
        self.trained_map_params = {
            k: v.clone() for k, v in self.original_params.items()
        }

        # Create current parameters (theta) initialized to theta_*
        self.current_params = {}
        for name, param in self.original_params.items():
            param_name = f"param_{name.replace('.', '_')}"

            # Initialize to original parameters
            current_param = nn.Parameter(param.clone(), requires_grad=True)
            self.register_parameter(param_name, current_param)
            self.current_params[name] = current_param

        # Freeze original params
        for p in model.parameters():
            p.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        """Computes the first-order Taylor approximation.

        Args:
            x: The input to the linearized model.

        Returns:
            f(x, theta) + J(theta) delta.
        """
        # Compute deltas: delta = theta - theta_*
        current_deltas = {
            name: self.current_params[name] - self.original_params[name]
            for name in self.original_params
        }

        out, dp = jvp(
            lambda params: functional_call(
                self.original_model, (params, self.original_buffers), (x,)
            ),
            (self.original_params,),
            (current_deltas,),
        )
        return out + dp

    @torch.no_grad()
    def set_params(self, params_dict: dict[str, Tensor]) -> None:
        """Set parameters to theta.

        Args:
            params_dict: Dictionary mapping parameter names to parameter tensors.
        """
        for name, param in params_dict.items():
            self.current_params[name].data = param

    @torch.no_grad()
    def set_deltas(self, deltas_dict: dict[str, Tensor]) -> None:
        """Set parameters via deltas: theta = theta_* + delta.

        Args:
            deltas_dict: Dictionary mapping parameter names to delta tensors.
        """
        for name, delta in deltas_dict.items():
            self.current_params[name].data = self.original_params[name] + delta

    @torch.no_grad()
    def save_current_as_trained_map(self) -> None:
        """Save current parameters as the trained MAP parameters.

        This should be called after MAP training to update the reference point
        for subsequent unlearning operations.
        """
        for name, param in self.current_params.items():
            self.trained_map_params[name] = param.clone()

    @torch.no_grad()
    def reset_to_map(self) -> None:
        """Reset current parameters to the trained MAP parameters.

        This resets to either the original pretrained parameters (if no MAP training
        was performed) or the trained MAP parameters (if MAP training was performed).
        """
        for name, param in self.trained_map_params.items():
            self.current_params[name].data = param

    @staticmethod
    def train(*args, **kwargs) -> None:
        """No-op to maintain compatibility with training loops."""
        del args, kwargs

    @staticmethod
    def eval(*args, **kwargs) -> None:
        """No-op to maintain compatibility with training loops."""
        del args, kwargs


def apply_params_to_nonlinear_model(
    nonlinear_model: nn.Module,
    params_dict: dict[str, Tensor],
) -> nn.Module:
    """Apply parameters directly to a nonlinear model and return the modified model.

    Args:
        nonlinear_model: The original nonlinear model.
        params_dict: Dictionary mapping parameter names to parameter tensors.

    Returns:
        Copy of the model with parameters applied directly.
    """
    model_copy = deepcopy(nonlinear_model)

    param_dict = {k: v for k, v in model_copy.named_parameters() if v.requires_grad}

    with torch.no_grad():
        for name, param in params_dict.items():
            if name in param_dict:
                param_dict[name].data = param

    return model_copy
