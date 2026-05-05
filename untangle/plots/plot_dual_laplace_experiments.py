"""Plots dual-Laplace experiment summaries from W&B and tensors."""

import argparse
import logging
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from scipy.stats import gaussian_kde, pearsonr, spearmanr
from tqdm import tqdm
from tueplots import bundles
from utils import ESTIMATOR_CONVERSION_DICT, ID_TO_METHOD, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

EXPERIMENT_ALIASES = {
    "flella": ("flella", "first-layer-epistemic-last-layer-aleatoric"),
    "llefla": ("llefla", "last-layer-epistemic-first-layer-aleatoric"),
}

APPROXIMATION_ALIASES = {
    "kfac": ("kfac",),
    "low-rank": ("low-rank", "low_rank", "lowrank"),
}

METRIC_LABELS = {
    "rank_correlation_au_it_au_vs_eu": "IT AU-LA AU vs EU",
    "rank_correlation_eu_it_au_vs_eu": "IT EU-LA AU vs EU",
    "rank_correlation_au_it_au_vs_eu_it_eu": "IT AU-LA AU vs EU-LA EU",
    "rank_correlation_eu_it_au_vs_au_it_eu": "IT EU-LA AU vs AU-LA EU",
    "correlation_au_it_au_vs_eu": "Pearson IT AU-LA AU vs EU",
    "correlation_eu_it_au_vs_eu": "Pearson IT EU-LA AU vs EU",
    "correlation_au_it_au_vs_eu_it_eu": "Pearson IT AU-LA AU vs EU-LA EU",
    "correlation_eu_it_au_vs_au_it_eu": "Pearson IT EU-LA AU vs AU-LA EU",
    "rank_correlation_au_bregman_au_vs_eu": "Bregman AU-LA AU vs EU",
    "rank_correlation_eu_bregman_au_vs_eu": "Bregman EU-LA AU vs EU",
    "rank_correlation_au_bregman_au_vs_eu_bregman_eu": (
        "Bregman AU-LA AU vs EU-LA EU"
    ),
    "rank_correlation_eu_bregman_au_vs_au_bregman_eu": (
        "Bregman EU-LA AU vs AU-LA EU"
    ),
    "correlation_au_bregman_au_vs_eu": "Pearson Bregman AU-LA AU vs EU",
    "correlation_eu_bregman_au_vs_eu": "Pearson Bregman EU-LA AU vs EU",
    "correlation_au_bregman_au_vs_eu_bregman_eu": (
        "Pearson Bregman AU-LA AU vs EU-LA EU"
    ),
    "correlation_eu_bregman_au_vs_au_bregman_eu": (
        "Pearson Bregman EU-LA AU vs AU-LA EU"
    ),
    "rank_correlation_bregman_au_b_dual_bma": "Bregman AU vs Bias",
    "correlation_bregman_au_b_dual_bma": "Pearson Bregman AU vs Bias",
    "rank_correlation_bregman_au": "AU vs Bregman GT AU",
    "auroc_oodness": "OOD AUROC",
    "auroc_hard_bma_correctness_original": "ID Correctness AUROC",
    "ece_hard_bma_correctness_original": "ECE",
}

ESTIMATOR_LABELS = {
    **ESTIMATOR_CONVERSION_DICT,
    "rank_correlation_au_it_au_vs_eu": "AU-LA: AU vs EU",
    "rank_correlation_eu_it_au_vs_eu": "EU-LA: AU vs EU",
    "rank_correlation_au_it_au_vs_eu_it_eu": "AU-LA AU vs EU-LA EU",
    "rank_correlation_eu_it_au_vs_au_it_eu": "EU-LA AU vs AU-LA EU",
    "correlation_au_it_au_vs_eu": "AU-LA: AU vs EU",
    "correlation_eu_it_au_vs_eu": "EU-LA: AU vs EU",
    "correlation_au_it_au_vs_eu_it_eu": "AU-LA AU vs EU-LA EU",
    "correlation_eu_it_au_vs_au_it_eu": "EU-LA AU vs AU-LA EU",
    "rank_correlation_au_bregman_au_vs_eu": "AU-LA: AU vs EU",
    "rank_correlation_eu_bregman_au_vs_eu": "EU-LA: AU vs EU",
    "rank_correlation_au_bregman_au_vs_eu_bregman_eu": (
        "AU-LA AU vs EU-LA EU"
    ),
    "rank_correlation_eu_bregman_au_vs_au_bregman_eu": (
        "EU-LA AU vs AU-LA EU"
    ),
    "correlation_au_bregman_au_vs_eu": "AU-LA: AU vs EU",
    "correlation_eu_bregman_au_vs_eu": "EU-LA: AU vs EU",
    "correlation_au_bregman_au_vs_eu_bregman_eu": "AU-LA AU vs EU-LA EU",
    "correlation_eu_bregman_au_vs_au_bregman_eu": "EU-LA AU vs AU-LA EU",
    "rank_correlation_bregman_au_b_dual_bma": "AU vs Bias",
    "correlation_bregman_au_b_dual_bma": "AU vs Bias",
}

IT_RANK_CORRELATION_METRICS = [
    "rank_correlation_au_it_au_vs_eu",
    "rank_correlation_eu_it_au_vs_eu",
    "rank_correlation_au_it_au_vs_eu_it_eu",
    "rank_correlation_eu_it_au_vs_au_it_eu",
]

IT_PEARSON_CORRELATION_METRICS = [
    "correlation_au_it_au_vs_eu",
    "correlation_eu_it_au_vs_eu",
    "correlation_au_it_au_vs_eu_it_eu",
    "correlation_eu_it_au_vs_au_it_eu",
]

BREGMAN_RANK_CORRELATION_METRICS = [
    "rank_correlation_au_bregman_au_vs_eu",
    "rank_correlation_eu_bregman_au_vs_eu",
    "rank_correlation_au_bregman_au_vs_eu_bregman_eu",
    "rank_correlation_eu_bregman_au_vs_au_bregman_eu",
]

BREGMAN_PEARSON_CORRELATION_METRICS = [
    "correlation_au_bregman_au_vs_eu",
    "correlation_eu_bregman_au_vs_eu",
    "correlation_au_bregman_au_vs_eu_bregman_eu",
    "correlation_eu_bregman_au_vs_au_bregman_eu",
]

EPISTEMIC_ESTIMATORS = [
    "au_it_eu",
    "eu_it_eu",
    "jensen_shannon_divergences",
]

ALEATORIC_ESTIMATORS = [
    "au_it_au",
    "eu_it_au",
    "expected_entropies",
]

PREDICTIVE_ENTROPY_ESTIMATORS = ["entropies_of_bma"]

MATRIX_ESTIMATORS = [
    "entropies_of_bma",
    "au_it_au",
    "au_it_eu",
    "eu_it_au",
    "eu_it_eu",
    "expected_entropies",
    "jensen_shannon_divergences",
]

RANKING_SPECS = {
    "a_it_rank_correlation": (IT_RANK_CORRELATION_METRICS, [None]),
    "a_bregman_rank_correlation": (BREGMAN_RANK_CORRELATION_METRICS, [None]),
    "c_predictive_entropy": (
        [
            "auroc_hard_bma_correctness_original",
            "auroc_oodness",
            "ece_hard_bma_correctness_original",
        ],
        PREDICTIVE_ENTROPY_ESTIMATORS,
    ),
    "d_ood_auroc_epistemic": (["auroc_oodness"], EPISTEMIC_ESTIMATORS),
    "e_gt_aleatoric_rank_correlation": (
        ["rank_correlation_bregman_au"],
        ALEATORIC_ESTIMATORS,
    ),
    "f_id_correctness_auroc_epistemic": (
        ["auroc_hard_bma_correctness_original"],
        EPISTEMIC_ESTIMATORS,
    ),
    "g_expected_calibration_error": (
        ["ece_hard_bma_correctness_original"],
        EPISTEMIC_ESTIMATORS + PREDICTIVE_ENTROPY_ESTIMATORS,
    ),
    "h_bregman_aleatoric_bias_rank_correlation": (
        ["rank_correlation_bregman_au_b_dual_bma"],
        [None],
    ),
}


parser = argparse.ArgumentParser(description="Plot W&B summaries.")
parser.add_argument("dataset", help="Dataset name used in the output path.")
parser.add_argument(
    "--wandb-project",
    default="evelyn-emelanov-university-of-t-bingen/udl-thesis",
    help="W&B project path in entity/project form.",
)
parser.add_argument(
    "--experiment",
    choices=tuple(EXPERIMENT_ALIASES),
    action="append",
    default=[],
    help="Experiment to include. Repeat to include several. Defaults to both.",
)
parser.add_argument(
    "--experiment-filter",
    action="append",
    default=[],
    metavar="EXPERIMENT=SUBSTRING",
    help=(
        "Map an experiment to the substring used in your W&B run names. "
        "Example: --experiment-filter flella=my-first-layer-run."
    ),
)
parser.add_argument(
    "--approximation",
    choices=tuple(APPROXIMATION_ALIASES),
    action="append",
    default=[],
    help="Approximation method to include. Defaults to kfac and low-rank.",
)
parser.add_argument(
    "--run-name-filter",
    action="append",
    default=[],
    help="Additional substring that must appear in the W&B run name.",
)
parser.add_argument(
    "--run-group-filter",
    default=None,
    help="Optional exact W&B run group filter.",
)
parser.add_argument(
    "--prefix",
    action="append",
    default=[],
    help="Restrict summary-key prefixes. Repeat for multiple prefixes.",
)
parser.add_argument(
    "--ood-prefix",
    default=None,
    help=(
        "Summary-key prefix to prefer for OOD AUROC. "
        "Defaults to a mixed prefix if found."
    ),
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=None,
    help="Directory for generated plots. Defaults to results/<dataset>/dual_laplace.",
)
parser.add_argument(
    "--tensor-dir",
    type=Path,
    default=None,
    help="Directory containing saved .pt decomposition pairs for scatter plots.",
)
parser.add_argument(
    "--tensor-glob",
    default="**/*it_au_eu.pt",
    help="Glob, relative to --tensor-dir, for scatter input tensors.",
)
parser.add_argument(
    "--no-wandb",
    action="store_true",
    help="Only generate scatter plots from --tensor-dir.",
)


def setup_plot_style() -> None:
    """Set a compact paper plotting style."""
    config = bundles.neurips2024()
    config["figure.figsize"] = (4.8, 2.6)
    plt.rcParams.update(config)
    plt.rcParams["text.latex.preamble"] += (
        r"\usepackage{amsmath} \usepackage{amsfonts} \usepackage{bm}"
    )


def normalize_text(text: str) -> str:
    """Normalize text for loose run-name matching."""
    return text.lower().replace("_", "-").replace(" ", "-")


def contains_alias(text: str, aliases: Iterable[str]) -> bool:
    """Return whether text contains one of the aliases."""
    normalized = normalize_text(text)
    return any(normalize_text(alias) in normalized for alias in aliases)


def get_configured_run_label(
    run: wandb.apis.public.Run,
    args: argparse.Namespace,
) -> str | None:
    """Return the configured method label for a run from ID_TO_METHOD."""
    id_to_method = ID_TO_METHOD.get(args.dataset, {})
    candidates = [
        run.id,
        run.name or "",
        f"{args.wandb_project}/{run.id}",
        f"{args.wandb_project}/{run.name or ''}",
    ]
    for candidate in candidates:
        if candidate in id_to_method:
            return id_to_method[candidate]
    return None


def get_experiment_aliases(args: argparse.Namespace) -> dict[str, tuple[str, ...]]:
    """Return default experiment aliases plus user-provided W&B substrings."""
    aliases = {key: list(value) for key, value in EXPERIMENT_ALIASES.items()}
    for spec in args.experiment_filter:
        experiment, separator, substring = spec.partition("=")
        if not separator or experiment not in aliases or not substring:
            msg = (
                "--experiment-filter must look like EXPERIMENT=SUBSTRING, "
                f"where EXPERIMENT is one of {sorted(EXPERIMENT_ALIASES)}."
            )
            raise ValueError(msg)
        aliases[experiment].append(substring)
    return {key: tuple(value) for key, value in aliases.items()}


def matched_experiment(label: str, args: argparse.Namespace) -> str | None:
    """Return the experiment name matched by a W&B run label."""
    experiment_aliases = get_experiment_aliases(args)
    experiments = args.experiment or list(EXPERIMENT_ALIASES)
    for experiment in experiments:
        if contains_alias(label, experiment_aliases[experiment]):
            return experiment
    return None


def method_matches_filters(
    method_label: str,
    run_label: str,
    args: argparse.Namespace,
) -> bool:
    """Check whether a configured method belongs to the requested slice."""
    approximations = args.approximation or list(APPROXIMATION_ALIASES)

    return (
        matched_experiment(method_label, args) is not None
        and any(
            contains_alias(method_label, APPROXIMATION_ALIASES[item])
            for item in approximations
        )
        and all(
            contains_alias(f"{run_label} {method_label}", (item,))
            for item in args.run_name_filter
        )
    )


def run_matches(run: wandb.apis.public.Run, args: argparse.Namespace) -> bool:
    """Check whether a W&B run is configured and requested."""
    method_label = get_configured_run_label(run, args)
    if method_label is None:
        return False
    return method_matches_filters(method_label, run.name or run.id, args)


def run_group_label(run: wandb.apis.public.Run, args: argparse.Namespace) -> str:
    """Build a compact label from experiment and approximation aliases."""
    configured_label = get_configured_run_label(run, args)
    if configured_label is not None:
        return configured_label

    label = run.name or run.id
    experiment = (matched_experiment(label, args) or "unknown").upper()
    approximation = next(
        (
            name.upper()
            for name, aliases in APPROXIMATION_ALIASES.items()
            if contains_alias(label, aliases)
        ),
        "UNKNOWN",
    )
    return f"{experiment} {approximation}"


def metric_key_parts(
    key: str,
    metric: str,
    estimator: str | None,
) -> tuple[str, str] | None:
    """Return prefix and metric segment for a W&B summary key, if it matches."""
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
    """Discover prefixes that contain one of the requested metrics."""
    prefixes = set()
    for run in runs:
        for key in run.summary.keys():
            for metric in metrics:
                for estimator in estimators:
                    parts = metric_key_parts(key, metric, estimator)
                    if parts is not None:
                        prefixes.add(parts[0])
    return sorted(prefixes)


def choose_prefix(
    prefixes: list[str],
    metric: str,
    args: argparse.Namespace,
) -> str | None:
    """Choose the best prefix for a metric."""
    if args.prefix:
        candidates = [prefix for prefix in prefixes if prefix in args.prefix]
    else:
        candidates = prefixes

    if not candidates:
        return None
    if metric == "auroc_oodness":
        if args.ood_prefix in candidates:
            return args.ood_prefix
        mixed = [prefix for prefix in candidates if "mixed" in prefix]
        return mixed[0] if mixed else candidates[0]

    id_like = [
        prefix
        for prefix in candidates
        if "mixed" not in prefix and "ood" not in prefix
    ]
    return id_like[0] if id_like else candidates[0]


def read_summary_value(
    run: wandb.apis.public.Run,
    metric: str,
    estimator: str | None,
    prefix: str,
) -> float | None:
    """Read a scalar metric from a W&B run summary."""
    key = (
        f"{prefix}_{metric}"
        if estimator is None
        else f"{prefix}_{estimator}_{metric}"
    )
    value = run.summary.get(key)
    if value is None or isinstance(value, str):
        return None
    return float(value)


def collect_values(
    runs: list[wandb.apis.public.Run],
    metrics: list[str],
    estimators: list[str | None],
    args: argparse.Namespace,
) -> dict[tuple[str, str | None, str], list[float]]:
    """Collect values keyed by metric, estimator, and method label."""
    prefixes = discover_prefixes(runs, metrics, estimators)
    values: dict[tuple[str, str | None, str], list[float]] = defaultdict(list)

    for metric in metrics:
        prefix = choose_prefix(prefixes, metric, args)
        if prefix is None:
            logger.info("No prefix found for %s", metric)
            continue
        for run in runs:
            label = run_group_label(run, args)
            for estimator in estimators:
                value = read_summary_value(run, metric, estimator, prefix)
                if value is None or np.isnan(value):
                    continue
                if metric == "ece_hard_bma_correctness_original":
                    value *= -1
                values[(metric, estimator, label)].append(value)

    return values


def plot_grouped_bars(
    values: dict[tuple[str, str | None, str], list[float]],
    metrics: list[str],
    estimators: list[str | None],
    title: str,
    save_path: Path,
) -> None:
    """Plot grouped bars for metric/estimator values."""
    rows = []
    labels = sorted({label for _, _, label in values})
    for label in labels:
        for metric in metrics:
            for estimator in estimators:
                samples = values.get((metric, estimator, label), [])
                if not samples:
                    continue
                name = METRIC_LABELS.get(metric, metric)
                if estimator is not None:
                    est_label = ESTIMATOR_LABELS.get(estimator, estimator)
                    name = f"{name}\n{est_label}"
                rows.append((label, name, np.mean(samples), np.std(samples)))

    if not rows:
        logger.info("No values to plot for %s", title)
        return

    row_labels = [f"{method}\n{name}" for method, name, _, _ in rows]
    means = [mean for _, _, mean, _ in rows]
    stds = [std for _, _, _, std in rows]
    x = np.arange(len(rows))

    fig_width = max(5.0, 0.36 * len(rows))
    _, ax = plt.subplots(figsize=(fig_width, 3.0))
    ax.bar(x, means, yerr=stds, capsize=2, color="#4267d2", zorder=2)
    ax.grid(axis="y", linewidth=0.4, zorder=1)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(row_labels, rotation=45, ha="right")
    ax.set_ylabel("Mean over runs")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def build_run_metric_matrix(
    runs: list[wandb.apis.public.Run],
    metrics: list[str],
    estimators: list[str | None],
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[str]]:
    """Build a matrix with rows as metric/estimator pairs and columns as runs."""
    prefixes = discover_prefixes(runs, metrics, estimators)
    rows = []
    row_labels = []

    for metric in metrics:
        prefix = choose_prefix(prefixes, metric, args)
        if prefix is None:
            continue
        for estimator in estimators:
            row = []
            for run in runs:
                value = read_summary_value(run, metric, estimator, prefix)
                if value is None:
                    row = []
                    break
                row.append(
                    -value
                    if metric == "ece_hard_bma_correctness_original"
                    else value
                )
            if row:
                label = METRIC_LABELS.get(metric, metric)
                if estimator is not None:
                    label = f"{label}: {ESTIMATOR_LABELS.get(estimator, estimator)}"
                rows.append(row)
                row_labels.append(label)

    return np.array(rows), row_labels


def build_estimator_matrix(
    runs: list[wandb.apis.public.Run],
    metrics: list[str],
    estimators: list[str],
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[str]]:
    """Build rows as estimators and columns as metric/run observations."""
    prefixes = discover_prefixes(runs, metrics, estimators)
    rows = {estimator: [] for estimator in estimators}

    for metric in metrics:
        prefix = choose_prefix(prefixes, metric, args)
        if prefix is None:
            continue
        for run in runs:
            observation_values = {}
            for estimator in estimators:
                value = read_summary_value(run, metric, estimator, prefix)
                if value is None:
                    observation_values = {}
                    break
                observation_values[estimator] = (
                    -value
                    if metric == "ece_hard_bma_correctness_original"
                    else value
                )
            for estimator, value in observation_values.items():
                rows[estimator].append(value)

    row_labels = [
        ESTIMATOR_LABELS.get(estimator, estimator)
        for estimator, row in rows.items()
        if len(row) >= 2
    ]
    matrix_rows = [row for row in rows.values() if len(row) >= 2]
    if not matrix_rows:
        return np.array([]), []

    return np.array(matrix_rows), row_labels


def plot_correlation_matrix(
    matrix: np.ndarray,
    labels: list[str],
    name: str,
    save_path: Path,
) -> None:
    """Plot a correlation matrix with annotations."""
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        logger.info("Skipping %s; need at least two rows and two runs.", name)
        return

    corr = np.eye(matrix.shape[0])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            corr[i, j] = (
                spearmanr(matrix[i], matrix[j])[0]
                if name == "spearman"
                else pearsonr(matrix[i], matrix[j])[0]
            )

    fig_width = max(5.0, 0.35 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, fig_width))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_visible(False)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{corr[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=5,
            )
    ax.spines[["right", "top"]].set_visible(False)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_scatter_from_tensor(path: Path, save_path: Path) -> None:
    """Plot a density-colored log-log scatter from a saved decomposition pair."""
    first, second = torch.load(path, map_location="cpu", weights_only=True)
    x = first.detach().float().flatten().numpy()
    y = second.detach().float().flatten().numpy()
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        logger.info(
            "Skipping scatter for %s; not enough positive finite points.",
            path,
        )
        return

    xy = np.vstack([np.log10(x), np.log10(y)])
    density = gaussian_kde(xy)(xy)
    order = density.argsort()
    x = x[order]
    y = y[order]
    density = density[order]

    _, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(x, y, c=density, s=10, cmap="plasma")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Aleatoric Uncertainty")
    ax.set_ylabel("Epistemic Uncertainty")
    ax.spines[["right", "top"]].set_visible(False)
    plt.colorbar(scatter, ax=ax, label="Density")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def fetch_runs(args: argparse.Namespace) -> list[wandb.apis.public.Run]:
    """Fetch matching finished W&B runs."""
    if not ID_TO_METHOD.get(args.dataset):
        msg = f"No run mapping configured in ID_TO_METHOD for dataset {args.dataset!r}."
        raise ValueError(msg)

    filters: dict[str, Any] = {"state": "finished"}
    if args.run_group_filter:
        filters["group"] = args.run_group_filter

    api = wandb.Api()
    runs = [
        run
        for run in tqdm(api.runs(args.wandb_project, filters=filters))
        if run_matches(run, args)
    ]
    logger.info("Found %d matching W&B runs.", len(runs))
    return runs


def main() -> None:
    """Generate all requested FLELLA/LLEFLA plots."""
    setup_plot_style()
    args = parser.parse_args()
    output_dir = args.output_dir or Path(f"results/{args.dataset}/dual_laplace")

    if not args.no_wandb:
        runs = fetch_runs(args)
        for plot_name, (metrics, estimators) in RANKING_SPECS.items():
            values = collect_values(runs, metrics, estimators, args)
            plot_grouped_bars(
                values=values,
                metrics=metrics,
                estimators=estimators,
                title=plot_name.replace("_", " ").title(),
                save_path=output_dir / f"{plot_name}.pdf",
            )

        matrix_metrics = [
            "auroc_hard_bma_correctness_original",
            "auroc_oodness",
            "ece_hard_bma_correctness_original",
            "rank_correlation_bregman_au",
        ]
        matrix, labels = build_run_metric_matrix(
            runs=runs,
            metrics=matrix_metrics,
            estimators=MATRIX_ESTIMATORS,
            args=args,
        )
        plot_correlation_matrix(
            matrix=matrix,
            labels=labels,
            name="spearman",
            save_path=output_dir / "j_spearman_rank_correlation_matrix.pdf",
        )
        plot_correlation_matrix(
            matrix=matrix,
            labels=labels,
            name="pearson",
            save_path=output_dir / "i_pearson_correlation_matrix.pdf",
        )

        estimator_matrix, estimator_labels = build_estimator_matrix(
            runs=runs,
            metrics=matrix_metrics,
            estimators=MATRIX_ESTIMATORS,
            args=args,
        )
        plot_correlation_matrix(
            matrix=estimator_matrix,
            labels=estimator_labels,
            name="spearman",
            save_path=output_dir / "b_uncertainty_estimator_matrix_spearman.pdf",
        )
        plot_correlation_matrix(
            matrix=estimator_matrix,
            labels=estimator_labels,
            name="pearson",
            save_path=output_dir / "b_uncertainty_estimator_matrix_pearson.pdf",
        )

        decomp_matrix, decomp_labels = build_run_metric_matrix(
            runs=runs,
            metrics=IT_RANK_CORRELATION_METRICS
            + IT_PEARSON_CORRELATION_METRICS
            + BREGMAN_RANK_CORRELATION_METRICS
            + BREGMAN_PEARSON_CORRELATION_METRICS
            + [
                "rank_correlation_bregman_au_b_dual_bma",
                "correlation_bregman_au_b_dual_bma",
            ],
            estimators=[None],
            args=args,
        )
        plot_correlation_matrix(
            matrix=decomp_matrix,
            labels=decomp_labels,
            name="spearman",
            save_path=output_dir / "b_decomposition_correlation_matrix_spearman.pdf",
        )

    if args.tensor_dir is not None:
        for path in sorted(args.tensor_dir.glob(args.tensor_glob)):
            relative = path.relative_to(args.tensor_dir)
            save_name = relative.with_suffix(".pdf").as_posix().replace("/", "__")
            plot_scatter_from_tensor(path, output_dir / "scatter" / save_name)


if __name__ == "__main__":
    main()
