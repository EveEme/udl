"""Laplace predictive utilities using linearized JVP sampling."""

import logging
import math
from collections.abc import Callable, Iterable

import numpy as np
import torch
from torch import Tensor
from torch.func import jvp, vmap
from torch.utils.data import DataLoader

from bul.models.linearized import LinearizedModel
from bul.utils.flatten import (
    flatten_param_dict,
    get_param_keys,
    unflatten_to_param_dict_like,
)
from bul.utils.linear_operators import MatmulMixin, make_forward_fn
from bul.utils.metrics import calibration_error

logger = logging.getLogger(__name__)


def _jvp_nonlinear_at_map(
    linearized_model: LinearizedModel, inputs: Tensor, tangent: dict[str, Tensor]
) -> tuple[Tensor, Tensor]:
    """Compute f(x, theta_*) and J(x, theta_*) @ s via jvp.

    Args:
        linearized_model: Linearized model (provides original model/params/buffers).
        inputs: Input batch tensor.
        tangent: Dictionary of per-parameter tangent tensors (same structure as params).

    Returns:
        A tuple (f_base, jvp_term) where f_base = f(x, theta_*) and jvp_term = J s.
    """
    base_fn = make_forward_fn(
        original_model=linearized_model.original_model,
        buffers=linearized_model.original_buffers,
        inputs=inputs,
    )

    f_base, j_term = jvp(base_fn, (linearized_model.original_params,), (tangent,))
    return f_base, j_term


def _prepare_laplace_state(
    linearized_model: LinearizedModel,
) -> tuple[list[str], int, Tensor]:
    param_keys = get_param_keys(linearized_model)
    total_dim = sum(p.numel() for p in linearized_model.original_params.values())
    mean_flat = flatten_param_dict(
        {
            name: param - linearized_model.original_params[name]
            for name, param in linearized_model.current_params.items()
        },
        param_keys,
    )
    return param_keys, total_dim, mean_flat


def _make_batched_jvp(
    linearized_model: LinearizedModel,
    inputs: Tensor,
    param_keys: list[str],
    *,
    use_vmap: bool,
) -> Callable[[Tensor], Tensor]:
    def single_jvp(s_flat: Tensor, inputs=inputs) -> Tensor:
        s_tangent = unflatten_to_param_dict_like(
            s_flat, linearized_model.original_params, param_keys
        )
        _, j_term = _jvp_nonlinear_at_map(linearized_model, inputs, s_tangent)
        return j_term

    if use_vmap:
        return vmap(single_jvp, in_dims=0, out_dims=0)

    def single_sample_batch(s_flat_batch: Tensor) -> Tensor:
        return single_jvp(s_flat_batch[0]).unsqueeze(0)

    return single_sample_batch


def evaluate_laplace_full(
    linearized_model: LinearizedModel,
    loader: DataLoader,
    device: torch.device,
    scale_op: MatmulMixin,
    mc_samples: int,
    *,
    mc_chunk_size: int | None = None,
) -> tuple[float, float, float, Tensor]:
    """Compute NLL, accuracy, ECE, and mean probabilities for Laplace predictive.

    Returns:
        Tuple containing average NLL, accuracy, ECE, and mean class probabilities.
    """
    total_nll = 0.0
    total_correct = 0.0
    total_count = 0
    all_probs_list = []
    all_confidences = []
    all_correctnesses = []

    param_keys, total_dim, mean_flat = _prepare_laplace_state(linearized_model)

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        base_logits = linearized_model.original_model(inputs)
        use_vmap = mc_chunk_size != 1
        batched_jvp = _make_batched_jvp(
            linearized_model, inputs, param_keys, use_vmap=use_vmap
        )

        probs_sum = torch.zeros_like(base_logits, device=base_logits.device)
        log_target_sum = None

        chunk = mc_chunk_size if mc_chunk_size is not None else mc_samples

        remaining = mc_samples
        while remaining > 0:
            m = min(chunk, remaining)
            remaining -= m
            z = torch.randn(
                (total_dim, m), device=base_logits.device, dtype=base_logits.dtype
            )

            s_noise = scale_op @ z  # [total_dim, m]
            perturbations = (mean_flat.unsqueeze(1) + s_noise).T

            probs_chunk, s_chunk = _evaluate_laplace_chunk_full(
                batched_jvp=batched_jvp,
                base_logits=base_logits,
                targets=targets,
                perturbations=perturbations,
            )

            probs_sum = probs_sum + probs_chunk
            if log_target_sum is None:
                log_target_sum = s_chunk
            else:
                log_target_sum = torch.logaddexp(log_target_sum, s_chunk)

        probs_mean = probs_sum / mc_samples
        all_probs_list.append(probs_mean.cpu())

        pred = probs_mean.argmax(dim=1)
        correctness = (pred == targets).float()
        total_correct += correctness.sum().item()
        total_count += targets.shape[0]

        confidences = probs_mean.max(dim=1).values
        all_confidences.append(confidences.cpu())
        all_correctnesses.append(correctness.cpu())

        log_mean_target = log_target_sum - torch.log(
            torch.tensor(
                mc_samples,
                device=log_target_sum.device,
                dtype=log_target_sum.dtype,
            )
        )
        total_nll += -log_mean_target.sum().item()

    average_nll = total_nll / max(total_count, 1)

    all_probs = torch.cat(all_probs_list, dim=0)
    all_confidences_t = torch.cat(all_confidences, dim=0)
    all_correctnesses_t = torch.cat(all_correctnesses, dim=0)

    accuracy = total_correct / max(total_count, 1)
    ece = calibration_error(
        confidences=all_confidences_t,
        correctnesses=all_correctnesses_t,
        num_bins=15,
        norm="l1",
    ).item()

    return average_nll, accuracy, ece, all_probs


def evaluate_laplace_nll(
    linearized_model: LinearizedModel,
    loader: DataLoader,
    device: torch.device,
    scale_op: MatmulMixin,
    mc_samples: int,
    *,
    mc_chunk_size: int | None = None,
) -> float:
    """Compute only the average NLL for the Laplace predictive.

    Returns:
        Average NLL across all samples in the loader.
    """
    total_nll = 0.0
    total_count = 0

    param_keys, total_dim, mean_flat = _prepare_laplace_state(linearized_model)

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        base_logits = linearized_model.original_model(inputs)
        use_vmap = mc_chunk_size != 1
        batched_jvp = _make_batched_jvp(
            linearized_model, inputs, param_keys, use_vmap=use_vmap
        )

        log_target_sum = None

        chunk = mc_chunk_size if mc_chunk_size is not None else mc_samples

        remaining = mc_samples
        while remaining > 0:
            m = min(chunk, remaining)
            remaining -= m
            z = torch.randn(
                (total_dim, m), device=base_logits.device, dtype=base_logits.dtype
            )

            s_noise = scale_op @ z  # [total_dim, m]
            perturbations = (mean_flat.unsqueeze(1) + s_noise).T

            s_chunk = _evaluate_laplace_chunk_nll(
                batched_jvp=batched_jvp,
                base_logits=base_logits,
                targets=targets,
                perturbations=perturbations,
            )

            if log_target_sum is None:
                log_target_sum = s_chunk
            else:
                log_target_sum = torch.logaddexp(log_target_sum, s_chunk)

        total_count += targets.shape[0]
        log_mean_target = log_target_sum - torch.log(
            torch.tensor(
                mc_samples,
                device=log_target_sum.device,
                dtype=log_target_sum.dtype,
            )
        )
        total_nll += -log_mean_target.sum().item()

    return total_nll / max(total_count, 1)


@torch.compile(dynamic=False)
def _evaluate_laplace_chunk_full(
    batched_jvp: Callable[[Tensor], Tensor],
    base_logits: Tensor,
    targets: Tensor,
    perturbations: Tensor,
) -> tuple[Tensor, Tensor]:
    j_terms = batched_jvp(perturbations)  # [m, B, C]
    logits = base_logits.unsqueeze(0) + j_terms

    probs_sum_chunk = logits.softmax(dim=2).sum(dim=0)

    tgt = targets.view(1, -1, 1).expand(perturbations.shape[0], -1, 1)
    log_probs_mc = logits.log_softmax(dim=2)
    target_log_probs = log_probs_mc.gather(dim=2, index=tgt).squeeze(2)
    s_chunk = torch.logsumexp(target_log_probs, dim=0)

    return probs_sum_chunk, s_chunk


@torch.compile(dynamic=False)
def _evaluate_laplace_chunk_nll(
    batched_jvp: Callable[[Tensor], Tensor],
    base_logits: Tensor,
    targets: Tensor,
    perturbations: Tensor,
) -> Tensor:
    j_terms = batched_jvp(perturbations)
    logits = base_logits.unsqueeze(0) + j_terms

    tgt = targets.view(1, -1, 1).expand(perturbations.shape[0], -1, 1)
    log_probs_mc = logits.log_softmax(dim=2)
    target_log_probs = log_probs_mc.gather(dim=2, index=tgt).squeeze(2)
    return torch.logsumexp(target_log_probs, dim=0)


def grid_search_laplace_prior_precision(
    *,
    linearized_model: LinearizedModel,
    scale_op_builder: Callable[[float], MatmulMixin],
    val_loader: DataLoader,
    device: torch.device,
    mc_samples: int,
    mc_chunk_size: int | None = None,
    candidates: Iterable[float] | None = None,
    log10_from: float = -2.0,
    log10_to: float = 2.0,
    num_points: int = 50,
) -> float:
    """Grid search Laplace prior precision on a validation loader.

    Tries the provided candidates (or a default logspace from 1e-3 to 10) and
    picks the precision that minimizes validation NLL under the Laplace
    predictive.

    Args:
        linearized_model: Linearized model at MAP.
        scale_op_builder: Returns a Sigma^{1/2} scaoe op for a given prior.
        val_loader: Validation DataLoader.
        device: Torch device.
        mc_samples: Number of MC samples.
        mc_chunk_size: Optional chunk size for MC sampling; defaults to
            ``mc_samples`` when ``None``.
        candidates: Optional explicit list to try.
        log10_from: Start power for default logspace grid.
        log10_to: End power for default logspace grid.
        num_points: Number of default grid points.

    Returns:
        Selected Laplace prior precision.
    """
    if candidates is None:
        grid = np.logspace(log10_from, log10_to, num=num_points, base=10.0)
        # Use numpy scalars directly; no Python float() coercion needed
        candidates = grid.tolist()

    best_precision = None
    best_nll = math.inf
    # Track best NLL for logging only

    logger.info("Starting Laplace prior precision grid search...")
    logger.info("Candidates: %s", ", ".join(f"{c:g}" for c in candidates))

    for lam in candidates:
        scale_op = scale_op_builder(lam)
        nll = evaluate_laplace_nll(
            linearized_model=linearized_model,
            loader=val_loader,
            device=device,
            scale_op=scale_op,
            mc_samples=mc_samples,
            mc_chunk_size=mc_chunk_size,
        )
        logger.info("Candidate %s -> NLL=%.6f", f"{lam:g}", nll)
        if nll < best_nll:
            best_nll = nll
            best_precision = lam

    if best_precision is None:
        # Fallback: pick first candidate if all NLLs were NaN/Inf
        best_precision = candidates[0]
    logger.info(
        "Selected Laplace prior precision %s (best NLL=%.6f)",
        f"{best_precision:g}",
        best_nll,
    )

    return best_precision
