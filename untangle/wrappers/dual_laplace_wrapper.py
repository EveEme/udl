"""Dual Laplace Wrapper Class."""

import re
from copy import deepcopy
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from bul.utils.filter import enable_grads_by_regex
from untangle.utils.loader import PrefetchLoader
from untangle.wrappers.adapted_laplace_wrapper import AdaptedLaplaceWrapper
from untangle.wrappers.model_wrapper import DistributionalWrapper


class DualLaplaceWrapper(DistributionalWrapper):
    """Wrapper that creates two separate Laplace Approximation.

    Args:
        model: The neural network model to be wrapped.
        num_mc_samples: Number of Monte Carlo samples for prediction.
        num_mc_samples_cv: Number of Monte Carlo samples for cross-validation.
        weight_path: Path to the model weights.
    """

    def __init__(
        self,
        model: nn.Module,
        num_mc_samples: int,
        num_mc_samples_cv: int,
        weight_path: Path,
    ) -> None:
        super().__init__(model)

        self.epistemic_wrapper = AdaptedLaplaceWrapper(
            model=deepcopy(model),
            num_mc_samples=num_mc_samples,
            num_mc_samples_cv=num_mc_samples_cv,
            weight_path=weight_path,
        )

        self.aleatoric_wrapper = AdaptedLaplaceWrapper(
            model=deepcopy(model),
            num_mc_samples=num_mc_samples,
            num_mc_samples_cv=num_mc_samples_cv,
            weight_path=weight_path,
        )

    def _get_matching_parameters(self, regex: str, subset_name: str) -> list[str]:
        pattern = re.compile(regex)
        matches = [
            name for name, _ in self.model.named_parameters() if pattern.match(name)
        ]

        if not matches:
            msg = f"{subset_name} regex {regex!r} did not match any model parameters."
            raise ValueError(msg)

        return matches

    def _validate_subsets(
        self,
        epistemic_params_regex: str | None,
        aleatoric_params_regex: str | None,
    ) -> None:
        if epistemic_params_regex is None:
            msg = "epistemic_params_regex must be provided for dual-laplace."
            raise ValueError(msg)

        if aleatoric_params_regex is None:
            msg = "aleatoric_params_regex must be provided for dual-laplace."
            raise ValueError(msg)

        epistemic_matches = self._get_matching_parameters(
            epistemic_params_regex, "Epistemic"
        )
        aleatoric_matches = self._get_matching_parameters(
            aleatoric_params_regex, "Aleatoric"
        )

        overlapping_parameters = sorted(
            set(epistemic_matches).intersection(aleatoric_matches)
        )

        if overlapping_parameters:
            joined_parameters = ", ".join(overlapping_parameters)
            msg = (
                "epistemic_params_regex and aleatoric_params_regex must select "
                f"disjoint parameter subsets. Overlap: {joined_parameters}"
            )
            raise ValueError(msg)

    def perform_dual_laplace_approximation(
        self,
        train_loader: DataLoader | PrefetchLoader,
        val_loader: DataLoader | PrefetchLoader,
        epistemic_params_regex: str | None,
        aleatoric_params_regex: str | None,
        approx_method: str,
        rank: int | None = None,
    ) -> None:
        """Performs two Laplace Approximations.

        Args:
            train_loader: DataLoader or PrefetchLoader for the training data.
            val_loader: DataLoader or PrefetchLoader for the validation data.
            epistemic_params_regex: Regex to select parameters for the epistemic LA.
            aleatoric_params_regex: Regex to select parameters for the aleatoric LA.
            approx_method: The method to use for Laplace approximation.
            rank: The rank for the low-rank approximation.
        """
        self._validate_subsets(epistemic_params_regex, aleatoric_params_regex)

        enable_grads_by_regex(self.epistemic_wrapper.model, epistemic_params_regex)
        self.epistemic_wrapper.adapted_perform_laplace_approximation(
            train_loader=train_loader,
            val_loader=val_loader,
            approx_method=approx_method,
            rank=rank,
        )

        enable_grads_by_regex(self.aleatoric_wrapper.model, aleatoric_params_regex)
        self.aleatoric_wrapper.adapted_perform_laplace_approximation(
            train_loader=train_loader,
            val_loader=val_loader,
            approx_method=approx_method,
            rank=rank,
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Forward pass that combines both Laplace branches.

        Returns:
            Joint predictive logits together with the branch-specific logits.
        """
        epistemic_dict = self.epistemic_wrapper(x)
        aleatoric_dict = self.aleatoric_wrapper(x)

        epistemic_logits = epistemic_dict["logit"]
        aleatoric_logits = aleatoric_dict["logit"]

        with torch.no_grad():
            base_logits = self.epistemic_wrapper.model(x)

        base_logits = base_logits.unsqueeze(1)
        joint_logits = epistemic_logits + aleatoric_logits - base_logits

        return {
            "logit": joint_logits,
            "epistemic_logit": epistemic_logits,
            "aleatoric_logit": aleatoric_logits,
        }
