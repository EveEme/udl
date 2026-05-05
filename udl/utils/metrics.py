"""Metrics."""

import re

import torch
from torch import Tensor


def kl_divergence(probs_p: Tensor, probs_q: Tensor, dim: int = -1) -> Tensor:
    """Compute KL divergence between two probability distributions.

    Args:
        probs_p: First probability distribution.
        probs_q: Second probability distribution.
        dim: Dimension along which to sum the KL divergence.

    Returns:
        KL divergence D(P||Q) between the two distributions.
    """
    log_probs_p = get_log_probs(probs_p)
    log_probs_q = get_log_probs(probs_q)
    return (probs_p * (log_probs_p - log_probs_q)).sum(dim=dim)


def get_log_probs(probs: Tensor) -> Tensor:
    """Convert probabilities to log probabilities with numerical stability.

    Args:
        probs: Probability tensor.

    Returns:
        Log probabilities with small values clamped for numerical stability.
    """
    eps = torch.finfo(probs.dtype).eps
    return probs.clamp(min=eps).log()


def calculate_bin_metrics(
    confidences: Tensor, correctnesses: Tensor, num_bins: int = 10
) -> tuple[Tensor, Tensor, Tensor]:
    """Calculates the binwise accuracies, confidences and proportions of samples.

    Args:
        confidences: Tensor of shape (n,) containing predicted confidences.
        correctnesses: Tensor of shape (n,) containing the true correctness labels.
        num_bins: Number of equally sized bins.

    Returns:
        bin_proportions: Float tensor of shape (num_bins,) containing proportion
            of samples in each bin. Sums up to 1.
        bin_confidences: Float tensor of shape (num_bins,) containing the average
            confidence for each bin.
        bin_accuracies: Float tensor of shape (num_bins,) containing the average
            accuracy for each bin.
    """
    correctnesses = correctnesses.float()

    bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=confidences.device)
    indices = torch.bucketize(confidences.contiguous(), bin_boundaries) - 1
    indices = torch.clamp(indices, min=0, max=num_bins - 1)

    bin_counts = torch.zeros(
        num_bins, dtype=confidences.dtype, device=confidences.device
    )
    bin_counts.scatter_add_(dim=0, index=indices, src=torch.ones_like(confidences))
    bin_proportions = bin_counts / bin_counts.sum()
    pos_counts = bin_counts > 0

    bin_confidences = torch.zeros(
        num_bins, dtype=confidences.dtype, device=confidences.device
    )
    bin_confidences.scatter_add_(dim=0, index=indices, src=confidences)
    bin_confidences[pos_counts] /= bin_counts[pos_counts]

    bin_accuracies = torch.zeros(
        num_bins, dtype=correctnesses.dtype, device=confidences.device
    )
    bin_accuracies.scatter_add_(dim=0, index=indices, src=correctnesses)
    bin_accuracies[pos_counts] /= bin_counts[pos_counts]

    return bin_proportions, bin_confidences, bin_accuracies


def calibration_error(
    confidences: Tensor, correctnesses: Tensor, num_bins: int, norm: str
) -> Tensor:
    """Computes the expected/maximum calibration error.

    Args:
        confidences: Tensor of shape (n,) containing predicted confidences.
        correctnesses: Tensor of shape (n,) containing the true correctness labels.
        num_bins: Number of equally sized bins.
        norm: Whether to return ECE (L1 norm) or MCE (inf norm)

    Returns:
        The ECE/MCE.

    Raises:
        ValueError: If the provided norm is neither 'l1' nor 'inf'.
    """
    bin_proportions, bin_confidences, bin_accuracies = calculate_bin_metrics(
        confidences, correctnesses, num_bins
    )

    abs_diffs = (bin_accuracies - bin_confidences).abs()

    if norm == "l1":
        score = (bin_proportions * abs_diffs).sum()
    elif norm == "inf":
        score = abs_diffs.max()
    else:
        msg = f"Provided norm {norm} not l1 nor inf"
        raise ValueError(msg)

    return score
