"""Plotting utilities."""

import logging

import matplotlib.colors as mcolors
import numpy as np


def setup_logging() -> None:
    """Sets up logging config."""
    logging.basicConfig(
        format="{asctime} - {levelname} - {message}",
        datefmt="%Y-%m-%d %H:%M",
        style="{",
        level=logging.INFO,
        force=True,
    )


def lighten_color(color: str | tuple, amount: float = 0.5) -> tuple:
    """Lightens the given color by mixing it with white.

    Args:
        color: Original color (hex or RGB).
        amount: Amount of white to mix in. 0 is the original color, 1 is white.

    Returns:
        Lightened color in RGB format.
    """
    try:
        c = mcolors.to_rgb(color)
    except ValueError:
        c = color
    c_white = np.array([1.0, 1.0, 1.0])
    new_color = c + (c_white - c) * amount
    return tuple(new_color)


COLOR_GT = np.array([234.0, 67.0, 53.0]) / 255.0
COLOR_ESTIMATE = np.array([66.0, 103.0, 210.0]) / 255.0
COLOR_ERROR_BAR = np.array([105.0, 109.0, 113.0]) / 255.0
COLOR_DISTRIBUTIONAL = np.array([52.0, 168.0, 83.0]) / 255.0
COLOR_BASELINE = np.array([154.0, 160.0, 166.0]) / 255.0
COLOR_DETERMINISTIC = np.array([251.0, 188.0, 4.0]) / 255.0

ID_TO_METHOD_CIFAR10 = {
    "splendid-cherry-131": "CE Baseline",
    "pious-lake-132": "Low-rank FLELLA-10",
    "lunar-durian-133": "KFAC FLELLA-10",
    "dashing-music-134": "KFAC FLELLA-30",
    "misty-lake-135": "Low-rank LLEFLA-30",
    "comfy-sun-136": "KFAC LLEFLA-30",
}

ID_TO_METHOD_IMAGENET = {}

ID_TO_METHOD = {
    "imagenet": ID_TO_METHOD_IMAGENET,
    "cifar10": ID_TO_METHOD_CIFAR10,
}

DATASET_PREFIX_LIST_IMAGENET = [
    "best_id_test",
    "best_ood_test_varied_soft_imagenet_s1",
    "best_ood_test_varied_soft_imagenet_s2",
    "best_ood_test_varied_soft_imagenet_s3",
    "best_ood_test_varied_soft_imagenet_s4",
    "best_ood_test_varied_soft_imagenet_s5",
    # "best_ood_test_avg_soft_imagenet_s1",
    # "best_ood_test_avg_soft_imagenet_s2",
    # "best_ood_test_avg_soft_imagenet_s3",
    # "best_ood_test_avg_soft_imagenet_s4",
    # "best_ood_test_avg_soft_imagenet_s5",
    "best_ood_test_varied_soft_imagenet_s1_mixed_soft_imagenet",
    "best_ood_test_varied_soft_imagenet_s2_mixed_soft_imagenet",
    "best_ood_test_varied_soft_imagenet_s3_mixed_soft_imagenet",
    "best_ood_test_varied_soft_imagenet_s4_mixed_soft_imagenet",
    "best_ood_test_varied_soft_imagenet_s5_mixed_soft_imagenet",
    # "best_ood_test_avg_soft_imagenet_s1_mixed_soft_imagenet",
    # "best_ood_test_avg_soft_imagenet_s2_mixed_soft_imagenet",
    # "best_ood_test_avg_soft_imagenet_s3_mixed_soft_imagenet",
    # "best_ood_test_avg_soft_imagenet_s4_mixed_soft_imagenet",
    # "best_ood_test_avg_soft_imagenet_s5_mixed_soft_imagenet",
]

DATASET_PREFIX_LIST_CIFAR10 = [
    "best_id_test",
    "best_ood_test_varied_soft_cifar10_s1",
    "best_ood_test_varied_soft_cifar10_s2",
    "best_ood_test_varied_soft_cifar10_s3",
    "best_ood_test_varied_soft_cifar10_s4",
    "best_ood_test_varied_soft_cifar10_s5",
    # "best_ood_test_avg_soft_cifar10_s1",
    # "best_ood_test_avg_soft_cifar10_s2",
    # "best_ood_test_avg_soft_cifar10_s3",
    # "best_ood_test_avg_soft_cifar10_s4",
    # "best_ood_test_avg_soft_cifar10_s5",
    "best_ood_test_varied_soft_cifar10_s1_mixed_soft_cifar10",
    "best_ood_test_varied_soft_cifar10_s2_mixed_soft_cifar10",
    "best_ood_test_varied_soft_cifar10_s3_mixed_soft_cifar10",
    "best_ood_test_varied_soft_cifar10_s4_mixed_soft_cifar10",
    "best_ood_test_varied_soft_cifar10_s5_mixed_soft_cifar10",
    # "best_ood_test_avg_soft_cifar10_s1_mixed_soft_cifar10",
    # "best_ood_test_avg_soft_cifar10_s2_mixed_soft_cifar10",
    # "best_ood_test_avg_soft_cifar10_s3_mixed_soft_cifar10",
    # "best_ood_test_avg_soft_cifar10_s4_mixed_soft_cifar10",
    # "best_ood_test_avg_soft_cifar10_s5_mixed_soft_cifar10",
]

DATASET_PREFIX_LIST = {
    "cifar10": DATASET_PREFIX_LIST_CIFAR10,
}

DISTRIBUTIONAL_METHODS = [
    "Laplace",
    "Dual Laplace",
    "Low-rank FLELLA-10",
    "KFAC FLELLA-10",
    "KFAC FLELLA-30",
    "Low-rank LLEFLA-30",
    "KFAC LLEFLA-30",
]

EVIDENTIAL_METHODS = []

ESTIMATOR_CONVERSION_DICT = {
    "entropies_of_bma": r"$\text{PU}^\text{it}$",
    "expected_entropies": r"$\text{AU}^\text{it}$",
    "jensen_shannon_divergences": r"$\text{EU}^\text{it}$",
    "gt_total_predictives_bregman_dual_bma": r"$\text{PU}^\text{b}$",
    "gt_aleatorics_bregman": r"$\text{AU}^\text{b}$",
    "expected_divergences": r"$\text{EU}^\text{b}$",
    "gt_predictives_bregman_dual_bma": r"$\text{AU}^\text{b} + \text{B}^\text{b}$",
    "gt_biases_bregman_dual_bma": r"$\text{B}^\text{b}$",
    "expected_entropies_plus_expected_divergences": (
        r"$\text{AU}^\text{it} + \text{EU}^\text{b}$"
    ),
    "entropies_of_dual_bma": r"$\mathbb{H}\left(\tilde{\bm{\pi}}\right)$",
    "one_minus_expected_max_probs": r"$1 - \mathbb{E}\left[\max \bm{\pi}\right]$",
    "one_minus_max_probs_of_bma": r"$1 - \max \bar{\bm{\pi}}$",
    "one_minus_max_probs_of_dual_bma": r"$1 - \max \tilde{\bm{\pi}}$",
    "expected_variances_of_logits": r"$\mathbb{E}\left[\text{var }\bm{f}\right]$",
    "expected_variances_of_internal_logits": (
        r"$\mathbb{E}\left[\text{var }\bm{f}^\text{int}\right]$"
    ),
    "expected_variances_of_probs": r"$\mathbb{E}\left[\text{var }\bm{\pi}\right]$",
    "expected_variances_of_internal_probs": (
        r"$\mathbb{E}\left[\text{var }\bm{\pi}^\text{int}\right]$"
    ),
    "au_it_au": r"$\text{AU}_{\text{AU-LA}}^\text{it}$",
    "au_it_eu": r"$\text{EU}_{\text{AU-LA}}^\text{it}$",
    "eu_it_au": r"$\text{AU}_{\text{EU-LA}}^\text{it}$",
    "eu_it_eu": r"$\text{EU}_{\text{EU-LA}}^\text{it}$",
    "au_bregman_au": r"$\text{AU}_{\text{AU-LA}}^\text{b}$",
    "au_bregman_eu": r"$\text{EU}_{\text{AU-LA}}^\text{b}$",
    "eu_bregman_au": r"$\text{AU}_{\text{EU-LA}}^\text{b}$",
    "eu_bregman_eu": r"$\text{EU}_{\text{EU-LA}}^\text{b}$",
}

ONLY_DISTRIBUTIONAL_ESTIMATORS = [
    "expected_entropies",
    "jensen_shannon_divergences",
    "gt_total_predictives_bregman_dual_bma",
    "expected_divergences",
    "expected_entropies_plus_expected_divergences",
    "entropies_of_dual_bma",
    "one_minus_expected_max_probs",
    "one_minus_max_probs_of_dual_bma",
    "expected_variances_of_logits",
    "expected_variances_of_internal_logits",
    "expected_variances_of_probs",
    "expected_variances_of_internal_probs",
    "au_it_au",
    "au_it_eu",
    "eu_it_au",
    "eu_it_eu",
    "au_bregman_au",
    "au_bregman_eu",
    "eu_bregman_au",
    "eu_bregman_eu",
]

ONLY_NON_EVIDENTIAL_ESTIMATORS = [
    "expected_variances_of_logits",
    "expected_variances_of_internal_logits",
    "expected_variances_of_probs",
    "expected_variances_of_internal_probs",
]

ESTIMATORLESS_METRICS = [
    "hard_bma_accuracy_original",
    "correlation_it_au_eu",
    "correlation_it_eu_pu",
    "correlation_it_au_pu",
    "correlation_bregman_au_b_dual_bma",
    "correlation_bregman_eu_au_hat",
    "correlation_bregman_au_eu",
    "correlation_kendall_gal_au_eu_prob",
    "correlation_kendall_gal_au_eu_logit",
    "correlation_kendall_gal_au_eu_internal_prob",
    "correlation_kendall_gal_au_eu_internal_logit",
    "correlation_au_bregman_au_vs_eu",
    "correlation_eu_bregman_au_vs_eu",
    "correlation_au_bregman_au_vs_eu_bregman_eu",
    "correlation_eu_bregman_au_vs_au_bregman_eu",
    "correlation_au_it_au_vs_eu",
    "correlation_eu_it_au_vs_eu",
    "correlation_au_it_au_vs_eu_it_eu",
    "correlation_eu_it_au_vs_au_it_eu",
    "rank_correlation_au_it_au_vs_eu",
    "rank_correlation_eu_it_au_vs_eu",
    "rank_correlation_au_it_au_vs_eu_it_eu",
    "rank_correlation_eu_it_au_vs_au_it_eu",
    "rank_correlation_au_bregman_au_vs_eu",
    "rank_correlation_eu_bregman_au_vs_eu",
    "rank_correlation_au_bregman_au_vs_eu_bregman_eu",
    "rank_correlation_eu_bregman_au_vs_au_bregman_eu",
    "rank_correlation_it_au_eu",
    "rank_correlation_it_eu_pu",
    "rank_correlation_it_au_pu",
    "rank_correlation_bregman_au_b_dual_bma",
    "rank_correlation_bregman_eu_au_hat",
    "rank_correlation_bregman_au_eu",
    "rank_correlation_kendall_gal_au_eu_prob",
    "rank_correlation_kendall_gal_au_eu_logit",
    "rank_correlation_kendall_gal_au_eu_internal_prob",
    "rank_correlation_kendall_gal_au_eu_internal_logit",
    "log_prob_score_hard_bma_aleatoric_original",
    "brier_score_hard_bma_aleatoric_original",
    "time_forward_m",
]

CONSTRAINED_METRICS = [
    "ece_hard_bma_correctness_original",
    "ece_soft_bma_correctness_original",
    "brier_score_hard_bma_correctness_original",
    "brier_score_soft_bma_correctness_original",
    "log_prob_score_hard_bma_correctness_original",
    "log_prob_score_soft_bma_correctness_original",
]

CORRELATION_MATRIX_ESTIMATORS = [
    "entropies_of_bma",
    "au_it_au",
    "au_it_eu",
    "eu_it_au",
    "eu_it_eu",
    "au_bregman_au",
    "au_bregman_eu",
    "eu_bregman_au",
    "eu_bregman_eu",
    "expected_entropies",
    "jensen_shannon_divergences",
]
