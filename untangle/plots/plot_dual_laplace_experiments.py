"""Generate dual-Laplace comparison plots from W&B summaries and saved tensors.
"""
from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from scipy.stats import gaussian_kde, pearsonr, spearmanr
from tueplots import bundles
from utils import ESTIMATOR_CONVERSION_DICT, ID_TO_METHOD, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

IT_PAIRWISE_METRICS = {
    "rank_correlation_au_it_au_vs_eu": r"$\text{AU}_{\text{last}} \quad \text{vs} \quad \text{EU}_{\text{last}}$",
    "rank_correlation_au_it_au_vs_eu_it_eu": r"$\text{AU}_{\text{last}} \quad \text{vs} \quad \text{EU}_{\text{first}}$",
    "rank_correlation_eu_it_au_vs_eu": r"$\text{AU}_{\text{first}} \quad \text{vs} \quad \text{EU}_{\text{first}}$",
    "rank_correlation_eu_it_au_vs_au_it_eu": r"$\text{AU}_{\text{first}} \quad \text{vs} \quad \text{EU}_{\text{last}}$",
}

BREGMAN_PAIRWISE_METRICS = {
    "rank_correlation_au_bregman_au_vs_eu": r"$\text{AU}_{\text{last}} \quad \text{vs} \quad \text{EU}_{\text{last}}$",
    "rank_correlation_au_bregman_au_vs_eu_bregman_eu": r"$\text{AU}_{\text{last}} \quad \text{vs} \quad \text{EU}_{\text{first}}$",
    "rank_correlation_eu_bregman_au_vs_eu": r"$\text{AU}_{\text{first}} \quad \text{vs} \quad \text{EU}_{\text{first}}$",
    "rank_correlation_eu_bregman_au_vs_au_bregman_eu": r"$\text{AU}_{\text{first}} \quad \text{vs} \quad \text{EU}_{\text{last}}$",
}

PREDICTIVE_ENTROPY_METRICS = [
    "auroc_hard_bma_correctness_original",
    "auroc_oodness",
    "ece_hard_bma_correctness_original",
    "brier_score_hard_bma_correctness_original",
    "log_prob_score_hard_bma_correctness_original",
]

METRIC_NAME = {
    "auroc_hard_bma_correctness_original": "ID Correctness AUROC",
    "auroc_oodness": "OOD AUROC",
    "ece_hard_bma_correctness_original": "ECE",
    "brier_score_hard_bma_correctness_original": "Brier",
    "log_prob_score_hard_bma_correctness_original": "Log Prob.",
    "rank_correlation_bregman_au": "Rank Corr. with GT Aleatoric",
    "rank_correlation_bregman_au_b_dual_bma": "Bregman AU vs Bias Rank Corr.",
}
ECE_METRICS = [
    "ece_hard_bma_correctness_original",
]

EPISTEMIC_ESTIMATORS = ["au_it_eu", "eu_it_eu", "jensen_shannon_divergences", "one_minus_max_probs_of_bma"]
ALEATORIC_ESTIMATORS = ["au_it_au", "eu_it_au", "expected_entropies"]
PREDICTIVE_ESTIMATORS = ["entropies_of_bma", "one_minus_max_probs_of_bma"]
ECE_ESTIMATORS = list(dict.fromkeys(EPISTEMIC_ESTIMATORS + PREDICTIVE_ESTIMATORS))

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

MATRIX_METRICS = [
    "auroc_hard_bma_correctness_original",
    "auroc_oodness",
    "ece_hard_bma_correctness_original",
    "brier_score_hard_bma_correctness_original",
    "log_prob_score_hard_bma_correctness_original",
    "rank_correlation_bregman_au",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Plot dual-Laplace results from W&B.")
    parser.add_argument("dataset", help="Dataset key used in ID_TO_METHOD.")
    parser.add_argument(
        "--wandb-project",
        default="evelyn-emelanov-university-of-t-bingen/udl-thesis",
        help="W&B project path in entity/project form.",
    )
    parser.add_argument(
        "--approximation",
        choices=("kfac", "low-rank"),
        action="append",
        default=[],
        help="Filter by approximation family. Repeat to include both.",
    )
    parser.add_argument(
        "--laplace-type",
        choices=("eu", "au"),
        action="append",
        default=[],
        help="Filter by dual-laplace variant in method label. Repeat to include both.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to results/<dataset>/dual_laplace.",
    )
    parser.add_argument(
        "--ood-prefix",
        default=None,
        help=(
            "Explicit prefix to use for OOD AUROC metrics. "
            "By default, the first prefix containing 'mixed' is used."
        ),
    )
    parser.add_argument(
        "--tensor-dir",
        type=Path,
        default=None,
        help="Directory containing saved .pt tuples for density scatter plots.",
    )
    parser.add_argument(
        "--tensor-glob",
        default="**/*_au_*_eu.pt",
        help="Glob used under --tensor-dir to find decomposition tuples.",
    )
    return parser.parse_args()


def setup_plot_style() -> None:
    """Use plotting style."""
    config = bundles.neurips2024()
    plt.rcParams.update(config)
    plt.rcParams["text.latex.preamble"] += (
        r"\usepackage{amsmath} \usepackage{amsfonts} \usepackage{bm}"
    )


def normalize_text(text: str) -> str:
    """Normalize text for robust substring matching."""
    return text.lower().replace("_", "-").replace(" ", "-")


def contains(text: str, token: str) -> bool:
    """Return whether normalized token is contained in normalized text."""
    return normalize_text(token) in normalize_text(text)


def estimator_label(estimator: str) -> str:
    """Return a readable estimator label."""
    return ESTIMATOR_CONVERSION_DICT.get(estimator, estimator)


def infer_method_tags(method_label: str) -> tuple[str | None, str | None]:
    """Infer approximation and dual-laplace type from configured method label."""
    approx = None
    if contains(method_label, "kfac"):
        approx = "kfac"
    elif contains(method_label, "low-rank"):
        approx = "low-rank"

    laplace_type = None
    if contains(method_label, "LA"):
        laplace_type = "LA"

    return approx, laplace_type


def should_include_method(
    method_label: str,
    approximations: list[str],
    laplace_types: list[str],
) -> bool:
    """Return whether a configured method label should be included."""
    if contains(method_label, "CE Baseline"):
        return True

    approx, laplace_type = infer_method_tags(method_label)
    if approx is None or laplace_type is None:
        return False
    approx_ok = not approximations or approx in approximations
    laplace_ok = not laplace_types or laplace_type in laplace_types
    return approx_ok and laplace_ok


def metric_key_parts(
    key: str,
    metric: str,
    estimator: str | None,
) -> tuple[str, str] | None:
    """Return summary-key prefix and metric segment for a matching key."""
    metric_suffix = f"_{metric}"
    if not key.endswith(metric_suffix):
        return None

    stem = key.removesuffix(metric_suffix)
    if estimator is None:
        return stem, metric

    estimator_suffix = f"_{estimator}"
    if not stem.endswith(estimator_suffix):
        return None

    return stem.removesuffix(estimator_suffix), estimator


def discover_prefixes(
    runs: Iterable[wandb.apis.public.Run],
    metrics: Iterable[str],
    estimators: Iterable[str | None],
) -> list[str]:
    """Discover summary-key prefixes that contain requested metrics."""
    prefixes = set()
    for run in runs:
        for key in run.summary.keys():
            for metric in metrics:
                for estimator in estimators:
                    if metric_key_parts(key, metric, estimator) is not None:
                        prefixes.add(metric_key_parts(key, metric, estimator)[0])
    
    if not prefixes:
        logger.warning(
            "No prefixes found for metrics %s and estimators %s. Available W&B keys:",
            list(metrics)[:3],
            list(estimators)[:3],
        )
        for run in runs:
            for key in sorted(run.summary.keys())[:10]:
                logger.warning("  %s", key)
    return sorted(prefixes)


def choose_prefix(
    metric: str,
    available_prefixes: list[str],
    explicit_ood_prefix: str | None,
) -> str | None:
    """Choose an ID/OOD prefix for the requested metric."""
    if not available_prefixes:
        return None

    if metric == "auroc_oodness":
        if explicit_ood_prefix and explicit_ood_prefix in available_prefixes:
            return explicit_ood_prefix
        mixed = [prefix for prefix in available_prefixes if "mixed" in prefix]
        return mixed[0] if mixed else available_prefixes[0]

    non_ood = [
        prefix
        for prefix in available_prefixes
        if "mixed" not in prefix and "ood" not in prefix
    ]
    return non_ood[0] if non_ood else available_prefixes[0]


def read_value(
    run: wandb.apis.public.Run,
    metric: str,
    estimator: str | None,
    prefix: str,
) -> float | None:
    """Read a scalar metric from W&B run summary."""
    key = f"{prefix}_{metric}" if estimator is None else f"{prefix}_{estimator}_{metric}"
    value = run.summary.get(key)
    if value is None or isinstance(value, str):
        return None

    scalar = float(value)
    return scalar


def load_runs(args: argparse.Namespace) -> list[tuple[wandb.apis.public.Run, str]]:
    """Load configured runs from ID_TO_METHOD mapping for the dataset."""
    id_to_method = ID_TO_METHOD.get(args.dataset, {})
    if not id_to_method:
        msg = f"No run mapping configured in ID_TO_METHOD for dataset {args.dataset!r}."
        raise ValueError(msg)

    api = wandb.Api()
    loaded: list[tuple[wandb.apis.public.Run, str]] = []

    for run_id, method_label in sorted(id_to_method.items()):
        if not should_include_method(method_label, args.approximation, args.laplace_type):
            continue

        path = run_id if run_id.count("/") >= 2 else f"{args.wandb_project}/{run_id}"
        run = api.run(path)
        if run.state != "finished":
            logger.info("Skipping %s (%s): state=%s", run.id, method_label, run.state)
            continue
        loaded.append((run, method_label))

    if not loaded:
        msg = "No finished runs matched filters from ID_TO_METHOD mapping."
        raise RuntimeError(msg)

    logger.info("Loaded %d runs from ID_TO_METHOD mapping.", len(loaded))
    return loaded


def collect_metric_values(
    run_entries: list[tuple[wandb.apis.public.Run, str]],
    metrics: list[str],
    estimators: list[str | None],
    args: argparse.Namespace,
) -> dict[tuple[str, str | None, str], list[float]]:
    """Collect values keyed by metric, estimator, and method label."""
    runs = [run for run, _ in run_entries]
    prefixes = discover_prefixes(runs, metrics, estimators)
    unique_estimators = list(dict.fromkeys(estimators))

    values: dict[tuple[str, str | None, str], list[float]] = defaultdict(list)
    for metric in metrics:
        prefix = choose_prefix(metric, prefixes, args.ood_prefix)
        if prefix is None:
            logger.info("No prefix available for metric %s", metric)
            continue

        for run, method_label in run_entries:
            for estimator in unique_estimators:
                scalar = read_value(run, metric, estimator, prefix)
                if scalar is None or np.isnan(scalar):
                    continue
                values[(metric, estimator, method_label)].append(scalar)

    return values


def plot_metric_by_method(
    values: dict[tuple[str, str | None, str], list[float]],
    metric: str,
    estimators: list[str | None],
    title: str,
    save_path: Path,
) -> None:
    """Plot bars by method for one metric across one or more estimators."""
    methods = sorted({method for _, _, method in values})
    unique_estimators = list(dict.fromkeys(estimators))
    rows: list[tuple[str, str, float, float]] = []

    for method in methods:
        for estimator in unique_estimators:
            samples = values.get((metric, estimator, method), [])
            if not samples:
                continue
            est_name = "Estimatorless" if estimator is None else estimator_label(estimator)
            rows.append((method, est_name, float(np.mean(samples)), float(np.std(samples))))

    if not rows:
        logger.info("No values found for plot %s", title)
        return

    x = np.arange(len(rows))
    means = [item[2] for item in rows]
    stds = [item[3] for item in rows]
    tick_labels = [f"{method}\n{est}" for method, est, _, _ in rows]

    _, ax = plt.subplots()
    ax.bar(x, means, yerr=stds, capsize=2, color="#2a77b3", zorder=2)
    ax.grid(axis="y", linewidth=0.4, zorder=1)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_ylabel(METRIC_NAME.get(metric, metric))
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_pairwise_rank_correlation(
    values: dict[tuple[str, str | None, str], list[float]],
    pairwise_metrics: dict[str, str],
    title: str,
    save_path: Path,
) -> None:
    """Plot explicit dual-laplace pairwise rank correlations."""
    methods = sorted({method for _, _, method in values})
    pairs = list(pairwise_metrics.items())

    if not methods:
        logger.info("No methods available for %s", title)
        return

    x = np.arange(len(pairs))
    width = 0.8 / max(len(methods), 1)

    _, ax = plt.subplots()
    for index, method in enumerate(methods):
        means = []
        stds = []
        for metric, _ in pairs:
            samples = values.get((metric, None, method), [])
            if not samples:
                means.append(np.nan)
                stds.append(0.0)
            else:
                means.append(float(np.mean(samples)))
                stds.append(float(np.std(samples)))

        positions = x - 0.4 + (index + 0.5) * width
        ax.bar(positions, means, width=width, yerr=stds, capsize=2, label=method, zorder=2)

    ax.grid(axis="y", linewidth=0.4, zorder=1)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name in pairs], rotation=20, ha="right")
    ax.set_ylabel("Spearman rank correlation")
    ax.legend(frameon=False, fontsize=7)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def build_estimator_observation_matrix(
    run_entries: list[tuple[wandb.apis.public.Run, str]],
    metrics: list[str],
    estimators: list[str],
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[str]]:
    """Create estimator x observation matrix from metrics across runs."""
    runs = [run for run, _ in run_entries]
    prefixes = discover_prefixes(runs, metrics, estimators)
    method_order = [method for _, method in run_entries]

    rows: dict[str, list[float]] = {estimator: [] for estimator in estimators}
    for method_label in method_order:
        run = next(run for run, label in run_entries if label == method_label)
        for metric in metrics:
            prefix = choose_prefix(metric, prefixes, args.ood_prefix)
            if prefix is None:
                continue
            observation: dict[str, float] = {}
            for estimator in estimators:
                value = read_value(run, metric, estimator, prefix)
                if value is None or np.isnan(value):
                    observation = {}
                    break
                observation[estimator] = value
            for estimator, value in observation.items():
                rows[estimator].append(value)

    kept_estimators = [estimator for estimator, row in rows.items() if len(row) >= 2]
    if not kept_estimators:
        return np.array([]), []

    matrix = np.array([rows[estimator] for estimator in kept_estimators])
    labels = [estimator_label(estimator) for estimator in kept_estimators]
    return matrix, labels


def compute_correlation(matrix: np.ndarray, corr_type: str) -> np.ndarray:
    """Compute pairwise Pearson or Spearman row-wise correlation."""
    corr = np.eye(matrix.shape[0])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            if corr_type == "spearman":
                corr[i, j] = spearmanr(matrix[i], matrix[j])[0]
            else:
                corr[i, j] = pearsonr(matrix[i], matrix[j])[0]
    return corr


def plot_correlation_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    corr_type: str,
    title: str,
    save_path: Path,
) -> None:
    """Plot and save a correlation heatmap with annotations."""
    if matrix.size == 0 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        logger.info("Skipping %s: matrix is too small.", title)
        return

    corr = compute_correlation(matrix, corr_type)

    fig, ax = plt.subplots()
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.outline.set_visible(False)

    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=5)

    ax.spines[["right", "top"]].set_visible(False)
    ax.set_title("")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_loglog_density_scatter(path: Path, save_path: Path) -> None:
    """Plot density-colored log-log scatter from a saved tensor pair."""
    first, second = torch.load(path, map_location="cpu", weights_only=True)
    x = first.detach().float().flatten().numpy()
    y = second.detach().float().flatten().numpy()

    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        logger.info("Skipping %s: not enough finite positive points.", path)
        return

    log_xy = np.vstack([np.log10(x), np.log10(y)])
    density = gaussian_kde(log_xy)(log_xy)
    order = density.argsort()

    x = x[order]
    y = y[order]
    density = density[order]

    _, ax = plt.subplots()
    scatter = ax.scatter(x, y, c=density, s=7, cmap="plasma")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Aleatoric uncertainty")
    ax.set_ylabel("Epistemic uncertainty")
    ax.set_title(path.stem)
    ax.spines[["right", "top"]].set_visible(False)
    plt.colorbar(scatter, ax=ax, label="Density")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def main() -> None:
    """Generate all requested dual-Laplace plots."""
    args = parse_args()
    setup_plot_style()

    output_dir = args.output_dir or Path(f"results/{args.dataset}/dual_laplace")
    run_entries = load_runs(args)

    # 1) IT rank correlations with explicit AU/EU pairings.
    it_values = collect_metric_values(
        run_entries,
        metrics=list(IT_PAIRWISE_METRICS),
        estimators=[None],
        args=args,
    )
    plot_pairwise_rank_correlation(
        values=it_values,
        pairwise_metrics=IT_PAIRWISE_METRICS,
        title="",
        save_path=output_dir / "01_it_pairwise_rank_correlation.pdf",
    )

    # Bregman pairings requested alongside IT comparisons.
    bregman_values = collect_metric_values(
        run_entries,
        metrics=list(BREGMAN_PAIRWISE_METRICS),
        estimators=[None],
        args=args,
    )
    plot_pairwise_rank_correlation(
        values=bregman_values,
        pairwise_metrics=BREGMAN_PAIRWISE_METRICS,
        title="",
        save_path=output_dir / "01b_bregman_pairwise_rank_correlation.pdf",
    )

    # 3) Predictive entropy results.
    predictive_values = collect_metric_values(
        run_entries,
        metrics=PREDICTIVE_ENTROPY_METRICS,
        estimators=PREDICTIVE_ESTIMATORS,
        args=args,
    )
    for metric in PREDICTIVE_ENTROPY_METRICS:
        plot_metric_by_method(
            values=predictive_values,
            metric=metric,
            estimators=PREDICTIVE_ESTIMATORS,
            title="",
            save_path=output_dir / f"03_predictive_entropy_{metric}.pdf",
        )

    # 3b) ECE results with fallback metric names.
    ece_predictive_values = collect_metric_values(
        run_entries,
        metrics=ECE_METRICS,
        estimators=PREDICTIVE_ESTIMATORS,
        args=args,
    )
    for metric in ECE_METRICS:
        if any(ece_predictive_values.get((metric, est, method))
               for est in PREDICTIVE_ESTIMATORS
               for method in [m for _, m in run_entries]):
            plot_metric_by_method(
                values=ece_predictive_values,
                metric=metric,
                estimators=PREDICTIVE_ESTIMATORS,
                title="",
                save_path=output_dir / f"03b_ece_{metric}.pdf",
            )
            break

    # 4) OOD detection AUROC.
    ood_values = collect_metric_values(
        run_entries,
        metrics=["auroc_oodness"],
        estimators=EPISTEMIC_ESTIMATORS,
        args=args,
    )
    plot_metric_by_method(
        values=ood_values,
        metric="auroc_oodness",
        estimators=EPISTEMIC_ESTIMATORS,
        title="",
        save_path=output_dir / "04_ood_detection_auroc.pdf",
    )

    # 5) Rank correlation with ground-truth aleatoric uncertainty.
    gt_aleatoric_values = collect_metric_values(
        run_entries,
        metrics=["rank_correlation_bregman_au"],
        estimators=ALEATORIC_ESTIMATORS,
        args=args,
    )
    plot_metric_by_method(
        values=gt_aleatoric_values,
        metric="rank_correlation_bregman_au",
        estimators=ALEATORIC_ESTIMATORS,
        title="",
        save_path=output_dir / "05_gt_aleatoric_rank_correlation.pdf",
    )

    # 6) ID correctness AUROC.
    id_auroc_values = collect_metric_values(
        run_entries,
        metrics=["auroc_hard_bma_correctness_original"],
        estimators=EPISTEMIC_ESTIMATORS,
        args=args,
    )
    plot_metric_by_method(
        values=id_auroc_values,
        metric="auroc_hard_bma_correctness_original",
        estimators=EPISTEMIC_ESTIMATORS,
        title="",
        save_path=output_dir / "06_id_correctness_auroc.pdf",
    )

    # 7) Expected calibration error.
    ece_values = collect_metric_values(
        run_entries,
        metrics=ECE_METRICS,
        estimators=ECE_ESTIMATORS,
        args=args,
    )
    for metric in ECE_METRICS:
        plot_metric_by_method(
            values=ece_values,
            metric=metric,
            estimators=ECE_ESTIMATORS,
            title="",
            save_path=output_dir / f"07_expected_calibration_error_{metric}.pdf",
        )

    # 8) Rank correlation of Bregman aleatoric and bias terms.
    au_bias_values = collect_metric_values(
        run_entries,
        metrics=["rank_correlation_bregman_au_b_dual_bma"],
        estimators=[None],
        args=args,
    )
    plot_metric_by_method(
        values=au_bias_values,
        metric="rank_correlation_bregman_au_b_dual_bma",
        estimators=[None],
        title="",
        save_path=output_dir / "08_bregman_aleatoric_bias_rank_correlation.pdf",
    )

    # 2/9/10) Correlation matrices between uncertainty estimators.
    estimator_matrix, estimator_labels = build_estimator_observation_matrix(
        run_entries=run_entries,
        metrics=MATRIX_METRICS,
        estimators=CORRELATION_MATRIX_ESTIMATORS,
        args=args,
    )
    plot_correlation_heatmap(
        matrix=estimator_matrix,
        labels=estimator_labels,
        corr_type="pearson",
        title="",
        save_path=output_dir / "02_uncertainty_estimator_correlation_matrix.pdf",
    )
    plot_correlation_heatmap(
        matrix=estimator_matrix,
        labels=estimator_labels,
        corr_type="pearson",
        title="",
        save_path=output_dir / "09_pearson_correlation_matrix.pdf",
    )
    plot_correlation_heatmap(
        matrix=estimator_matrix,
        labels=estimator_labels,
        corr_type="spearman",
        title="",
        save_path=output_dir / "10_spearman_rank_correlation_matrix.pdf",
    )

    # 11) Density-colored log-log scatter plots from saved tensors.
    if args.tensor_dir is not None:
        for path in sorted(args.tensor_dir.glob(args.tensor_glob)):
            relative = path.relative_to(args.tensor_dir)
            filename = relative.with_suffix(".pdf").as_posix().replace("/", "__")
            plot_loglog_density_scatter(
                path=path,
                save_path=output_dir / "11_loglog_density_scatter" / filename,
            )
    else:
        logger.info("Skipping density scatter plots: --tensor-dir not provided.")


if __name__ == "__main__":
    main()
