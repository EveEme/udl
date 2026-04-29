"""Laplace approximation wrapper class on Subnetworks.

- loads model (already trained)
- build laplace posterior using low-rank or KFAC tools
"""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from backpack import extend
from torch import Tensor, nn
from torch.func import jvp
from torch.utils.data import DataLoader

from bul.utils.flatten import unflatten_to_param_dict_like
from bul.utils.kfac import KFACScaleOperator, get_kfac_list_loader, get_kfac_scale_op
from bul.utils.linear_operators import make_forward_fn
from bul.utils.low_rank import (
    DiagonalPlusLowRankOperator,
    compute_low_rank_eigendecomposition,
    create_ggn_linear_operator,
    get_low_rank_scale_op,
)
from untangle.utils.loader import PrefetchLoader
from untangle.utils.metric import calibration_error
from untangle.wrappers.model_wrapper import DistributionalWrapper

logger = logging.getLogger(__name__)


class AdaptedLaplaceWrapper(DistributionalWrapper):
    """Wrapper that creates a Laplace-approximated posterior from an input model.

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

        self._num_mc_samples = num_mc_samples
        self._num_mc_samples_cv = num_mc_samples_cv
        self._is_laplace_approximated = False
        self._low_rank = False
        self._kfac = False
        self._num_params = None
        self._weight_path = weight_path
        self._load_model(weight_path)

    def adapted_perform_laplace_approximation(
        self,
        train_loader: DataLoader | PrefetchLoader,
        val_loader: DataLoader | PrefetchLoader,
        approx_method: str,
        rank: int | None = None,
    ) -> None:
        """Performs Laplace approximation and optimize prior precision.

        Args:
            train_loader: DataLoader or PrefetchLoader for the training data.
            val_loader: DataLoader or PrefetchLoader for the validation data.
            approx_method: The method to use for Laplace approximation.
            rank: The rank for the low-rank approximation.
        """
        # calc number of parameters in the subnetwork
        self._num_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        logger.info(f"Number of parameters in the subnetwork: {self._num_params}")

        with torch.enable_grad():
            logger.info("Starting Laplace approximation with defined subset")

            # compute GGN with low-rank approximation
            if approx_method == "low-rank":
                self._low_rank_approx(train_loader=train_loader, rank=rank)
                self._low_rank = True
            elif approx_method == "kfac":
                self._kfac_approx(train_loader=train_loader)
                self._kfac = True

            logger.info("Laplace approximation done.")

        logger.info("Starting prior precision optimization.")
        self._optimize_prior_precision_cv(
            val_loader=val_loader,
        )
        self._is_laplace_approximated = True
        logger.info("Prior precision optimization done.")

    def forward_head(self, *args: Any, **kwargs: Any) -> Tensor | dict[str, Tensor]:
        """Raises an error as it cannot be called directly for this class.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            ValueError: Always raised when this method is called.
        """
        del args, kwargs
        msg = f"forward_head cannot be called directly for {type(self)}"
        raise ValueError(msg)

    def forward(self, input: Tensor) -> Tensor | dict[str, Tensor]:
        """Performs forward pass with Laplace-approximated model.

        Args:
            input: Input tensor to the model.

        Returns:
            Dictionary containing the logit tensor.

        Raises:
            ValueError: If the model hasn't been Laplace-approximated yet.
        """
        if not self._is_laplace_approximated:
            msg = "Model has to be Laplace-approximated first"
            raise ValueError(msg)

        if self.training:
            return self.model(input)

        if self._low_rank:
            low_rank_scale_op = get_low_rank_scale_op(
                self._eigenvalues, self._eigenvectors, self.prior_precision
            )
            logit = self._predict(
                input,
                scale_op=low_rank_scale_op,
                num_samples=self._num_mc_samples,
            )
        elif self._kfac:
            kfac_scale_op = get_kfac_scale_op(self._kfac_list, self.prior_precision)
            logit = self._predict(
                input, scale_op=kfac_scale_op, num_samples=self._num_mc_samples
            )
        else:
            msg = "Invalid approximation method. Must be 'low_rank' or 'kfac'."
            raise ValueError(msg)

        return {
            "logit": logit,
        }

    @staticmethod
    def _get_ece(out_dist: Tensor, targets: Tensor) -> Tensor:
        """Calculates the Expected Calibration Error.

        Args:
            out_dist: Output distribution from the model.
            targets: True labels.

        Returns:
            Calculated Expected Calibration Error.
        """
        confidences, predictions = out_dist.max(dim=-1)  # [B]
        correctnesses = predictions.eq(targets).int()

        return calibration_error(
            confidences=confidences, correctnesses=correctnesses, num_bins=15, norm="l1"
        )

    def _optimize_prior_precision_cv(
        self,
        val_loader: DataLoader | PrefetchLoader,
        log_prior_prec_min: float = -1,
        log_prior_prec_max: float = 2,
        grid_size: int = 500,
    ) -> None:
        """Optimizes prior precision using cross-validation.

        Args:
            val_loader: DataLoader or PrefetchLoader for the validation data.
            log_prior_prec_min: Minimum log prior precision.
            log_prior_prec_max: Maximum log prior precision.
            grid_size: Number of grid points for optimization.
        """
        interval = torch.logspace(log_prior_prec_min, log_prior_prec_max, grid_size)
        self.prior_precision = self._grid_search(
            interval=interval,
            val_loader=val_loader,
        )

        logger.info(f"Optimized prior precision is {self.prior_precision}.")

    def _grid_search(
        self,
        interval: Tensor,
        val_loader: DataLoader | PrefetchLoader,
    ) -> float:
        """Performs grid search to find optimal prior precision.

        Args:
            interval: Tensor of prior precision values to search.
            val_loader: DataLoader or PrefetchLoader for the validation data.

        Returns:
            Optimal prior precision value.
        """
        results = []
        prior_precs = []

        for prior_prec in interval:
            prior_prec_value = prior_prec.item()

            logger.info(f"Trying {prior_prec}...")
            start_time = time.perf_counter()

            if self._low_rank:
                # Get scale operator: Sigma^{1/2}
                scale_op = get_low_rank_scale_op(
                    self._eigenvalues, self._eigenvectors, prior_prec_value
                )
            elif self._kfac:
                # Get scale operator using KFAC
                scale_op = get_kfac_scale_op(self._kfac_list, prior_prec_value)

            try:
                out_dist, targets = self._validate(
                    val_loader=val_loader, scale_op=scale_op
                )
                result = self._get_ece(out_dist, targets).item()
                accuracy = out_dist.argmax(dim=-1).eq(targets).float().mean()
            except RuntimeError as error:
                logger.info(f"Caught an exception in validate: {error}")
                result = float("inf")
                accuracy = float("NaN")

            logger.info(
                f"Took {time.perf_counter() - start_time} seconds, result: {result}, "
                f"accuracy: {accuracy}."
            )
            results.append(result)
            prior_precs.append(prior_prec_value)

        return prior_precs[int(np.argmin(results))]

    @torch.no_grad()
    def _validate(
        self,
        val_loader: DataLoader | PrefetchLoader,
        scale_op: DiagonalPlusLowRankOperator | KFACScaleOperator,
    ) -> tuple[Tensor, Tensor]:
        """Validates the model on the validation set.

        Args:
            val_loader: DataLoader or PrefetchLoader for the validation data.
            scale_op: Low-rank or KFAC scale operator for sampling.

        Returns:
            Tuple of output means and targets.
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        output_means = []
        targets = []

        for input, target in val_loader:
            if not isinstance(val_loader, PrefetchLoader):
                input, target = input.to(device), target.to(device)

            # linearize and sample
            out = self._predict(input, scale_op, num_samples=self._num_mc_samples_cv)
            out = F.softmax(out, dim=-1).mean(dim=1)  # [B, C]

            if out.device.type == "cuda":
                torch.cuda.synchronize()

            output_means.append(out)
            targets.append(target)

        return torch.cat(output_means, dim=0), torch.cat(targets, dim=0)

    def _kfac_approx(
        self,
        train_loader: DataLoader | PrefetchLoader,
    ) -> None:
        """Performs KFAC approximation of the Laplace posterior."""
        self.model.eval()

        model_full = extend(self.model, use_converter=True)
        params = [p for p in model_full.parameters() if p.requires_grad]
        device = next(model_full.parameters()).device

        self._kfac_list = get_kfac_list_loader(
            loader=train_loader,
            model=model_full,
            device=device,
            params=params,
            loss="ce",
            mc_samples=self._num_mc_samples_cv,
        )
        logger.info("KFAC approximation computed or loaded from checkpoint.")

    def _low_rank_approx(
        self, train_loader: DataLoader | PrefetchLoader, rank: int
    ) -> None:
        """Performs low-rank approximation of the Laplace posterior."""
        self.model.eval()

        # compute GGN linear operator for the subnetwork
        ggn = create_ggn_linear_operator(
            model=self.model,
            loss="ce",
            reduction="sum",
            loader=train_loader,
        )

        # compute low-rank eigendecomposition
        eigenvalues, eigenvectors = compute_low_rank_eigendecomposition(
            op=ggn,
            rank=rank,
            backend="arpack",
            arpack_kwargs={"curv": {"scipy_dtype": np.float64}},
            skerch_kwargs={},
        )
        self._eigenvalues = eigenvalues
        self._eigenvectors = eigenvectors

    def _compute_jvp_single(
        self,
        params_dict: dict[str, Tensor],
        param_keys: list[str],
        inputs: Tensor,
        tangent_flat: Tensor,
    ) -> Tensor:
        """Compute JVP for a single tangent vector.

        Args:
            params_dict: Dictionary of model parameters.
            param_keys: List of parameter names.
            inputs: Input tensor [N, input_dim].
            tangent_flat: Flattened tangent vector [P].

        Returns:
            JVP result [N, output_dim].
        """
        # We need to compute: J_x @ (Sigma^{1/2} @ v) for each sample.
        # This is done by: jvp(f, (theta*,), (Sigma^{1/2} @ v,))

        # Unflatten tangent using repo utility
        tangent_dict = unflatten_to_param_dict_like(
            tangent_flat, params_dict, param_keys
        )

        # Use repo's make_forward_fn
        forward_fn = make_forward_fn(
            original_model=self.model,
            buffers=dict(self.model.named_buffers()),
            inputs=inputs,
        )

        # Compute JVP
        _, jvp_output = jvp(forward_fn, (params_dict,), (tangent_dict,))

        return jvp_output

    def _predict(
        self,
        input: Tensor,
        scale_op: DiagonalPlusLowRankOperator | KFACScaleOperator,
        *,
        num_samples: int,
    ) -> Tensor:
        """Sample gaussian from parameter space and compute logit samples.

        Args:
            input: input to the model.
            scale_op: Scale operator for sampling.
            num_samples: Number of Monte Carlo samples to draw.

        Returns:
            logit samples.
        """
        params_dict = {
            name: p for name, p in self.model.named_parameters() if p.requires_grad
        }
        param_keys = list(params_dict.keys())
        device = next(iter(params_dict.values())).device
        dtype = next(iter(params_dict.values())).dtype

        # Sample standard normals: v ~ N(0, I)
        v_samples = torch.randn(
            self._num_params,
            num_samples,
            device=device,
            dtype=dtype,
        )

        # Transform to posterior samples: Sigma^{1/2} @ v
        perturbations = scale_op @ v_samples  # [P, n_samples]

        # Posterior samples: theta = theta* + Sigma^{1/2} @ v
        # theta_samples = theta_map.unsqueeze(1) + perturbations

        # compute JVP to get logit
        jvp_results = []

        with torch.enable_grad():
            for i in range(num_samples):
                tangent = perturbations[:, i]  # [P]
                jvp_output = self._compute_jvp_single(
                    params_dict, param_keys, input, tangent
                )
                jvp_output = jvp_output.detach()
                jvp_results.append(jvp_output)

        # Stack results: [n_samples, N, C]
        jvp_results = torch.stack(jvp_results, dim=0)

        with torch.no_grad():
            base_logits = self.model(input)

        # Compute sampled logits: f(x, theta*) + J_x Sigma^{1/2} v
        sampled_logits = base_logits.unsqueeze(0) + jvp_results

        sampled_logits = sampled_logits.permute(1, 0, 2)  # [B, S, C]

        return sampled_logits
