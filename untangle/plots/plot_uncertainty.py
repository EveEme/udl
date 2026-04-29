"""Plots aleatoric and epistemic uncertainty diagnostics from W&B runs.
"""

import argparse
import csv
import logging
import re
from collections import defaultdict
from copy import copy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import AxesImage
from matplotlib.patches import Patch
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from tueplots import bundles
from utils import (
    COLOR_BASELINE,
    COLOR_DISTRIBUTIONAL,
    COLOR_ERROR_BAR,
    COLOR_ESTIMATE,
    CORRELATION_MATRIX_ESTIMATORS,
    DATASET_PREFIX_LIST,
    DISTRIBUTIONAL_METHODS,
    ESTIMATORLESS_METRICS,
    ESTIMATOR_CONVERSION_DICT,
    ID_TO_METHOD,
    setup_logging,
)
from wandb.apis.public.sweeps import Sweep

import wandb

setup_logging()
logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_SELECTION_METRIC = (
    "id_eval_log_prob_score_hard_bma_aleatoric_original"
)

ALEATORIC_METRIC_DICT = {
    "rank_correlation_bregman_au": "Aleatoric Rank Corr. to Soft Labels",
    "auroc_multiple_labels": "Aleatoric AUROC vs Soft Labels",
}

EPISTEMIC_METRIC_DICT = {
    "auroc_oodness": "OOD AUROC",
}

TARGET_TO_LONG_LABEL = {
    "au": r"Matched to aleatoric task",
    "eu": r"Matched to epistemic task",
}

FOCUSED_ESTIMATOR_ORDER = [
    "expected_entropies",
    "expected_divergences",
    "jensen_shannon_divergences",
    "entropies_of_bma",
    "entropies_of_dual_bma",
    "expected_variances_of_probs",
    "expected_variances_of_logits",
    "expected_variances_of_internal_probs",
    "expected_variances_of_internal_logits",
    "au_bregman_au",
    "au_bregman_eu",
    "eu_bregman_au",
    "eu_bregman_eu",
    "au_it_au",
    "au_it_eu",
    "eu_it_au",
    "eu_it_eu",
]

ESTIMATOR_ROLE = {
    "expected_entropies": "au",
    "expected_divergences": "eu",
    "jensen_shannon_divergences": "eu",
    "expected_variances_of_probs": "eu",
    "expected_variances_of_logits": "eu",
    "expected_variances_of_internal_probs": "eu",
    "expected_variances_of_internal_logits": "eu",
    "au_bregman_au": "au",
    "au_bregman_eu": "eu",
    "eu_bregman_au": "au",
    "eu_bregman_eu": "eu",
    "au_it_au": "au",
    "au_it_eu": "eu",
    "eu_it_au": "au",
    "eu_it_eu": "eu",
}

EXTRA_ESTIMATOR_LABELS = {
    "au_bregman_au": r"Last layer" "\n" r"$\mathrm{AU}^{b}$",
    "au_bregman_eu": r"Last layer" "\n" r"$\mathrm{EU}^{b}$",
    "eu_bregman_au": r"First layer (EU)" "\n" r"$\mathrm{AU}^{b}$",
    "eu_bregman_eu": r"First layer (EU)" "\n" r"$\mathrm{EU}^{b}$",
    "au_it_au": r"Last layer" "\n" r"$\mathrm{AU}^{it}$",
    "au_it_eu": r"Last layer" "\n" r"$\mathrm{EU}^{it}$",
    "eu_it_au": r"First layer (EU)" "\n" r"$\mathrm{AU}^{it}$",
    "eu_it_eu": r"First layer (EU)" "\n" r"$\mathrm{EU}^{it}$",
    "expected_variances_of_internal_probs": (
        r"Internal" "\n" r"$\mathbb{E}[\mathrm{var}\,\pi]$"
    ),
    "expected_variances_of_internal_logits": (
        r"Internal" "\n" r"$\mathbb{E}[\mathrm{var}\,f]$"
    ),
}

DIRECT_CORRELATION_METRIC_DICT = {
    "bregman_au_eu": "GT\nAU vs EU",
    "kendall_gal_au_eu_prob": "K-G\nprob layer",
    "kendall_gal_au_eu_logit": "K-G\nlogit layer",
    "kendall_gal_au_eu_internal_prob": "K-G\ninternal prob",
    "kendall_gal_au_eu_internal_logit": "K-G\ninternal logit",
    "au_bregman_au_vs_eu": "Last layer\nBregman AU vs EU",
    "eu_bregman_au_vs_eu": "First layer (EU)\nBregman AU vs EU",
    "au_bregman_au_vs_eu_bregman_eu": (
        "Cross\nLast-layer AU vs\nFirst-layer (EU) EU"
    ),
    "eu_bregman_au_vs_au_bregman_eu": (
        "Cross\nFirst-layer (EU) AU vs\nLast-layer EU"
    ),
    "au_it_au_vs_eu": "Last layer\nIT AU vs EU",
    "eu_it_au_vs_eu": "First layer (EU)\nIT AU vs EU",
    "au_it_au_vs_eu_it_eu": "Cross\nLast-layer IT AU vs\nFirst-layer (EU) IT EU",
    "eu_it_au_vs_au_it_eu": "Cross\nFirst-layer (EU) IT AU vs\nLast-layer IT EU",
}

ESTIMATOR_CORRELATION_METRIC_DICT = {
    "auroc_hard_bma_correctness_original": "Correctness AUROC",
    "ece_hard_bma_correctness_original": "-ECE",
    "brier_score_hard_bma_correctness_original": "Correctness Brier",
    "log_prob_score_hard_bma_correctness_original": "Correctness Log Prob.",
    "hard_bma_raulc_original": "rAULC",
    "hard_bma_eaurc_original": "-E-AURC",
    "cumulative_hard_bma_abstinence_auc_original": "AUAC",
    "hard_bma_accuracy_original": "Accuracy",
    "log_prob_score_hard_bma_aleatoric_original": "Aleatoric Log Prob.",
    "brier_score_hard_bma_aleatoric_original": "Aleatoric Brier",
    "rank_correlation_bregman_au": "Aleatoric Rank Corr.",
    "auroc_multiple_labels": "Aleatoric AUROC",
    "auroc_oodness": "OOD AUROC",
}

NEGATED_ESTIMATOR_CORRELATION_METRICS = {
    "ece_hard_bma_correctness_original",
    "hard_bma_eaurc_original",
}

ESTIMATOR_CORRELATION_ID_PREFIX = "best_id_test"
ESTIMATOR_CORRELATION_MIXTURE_PREFIX = {
    "imagenet": "best_ood_test_varied_soft_imagenet_s2_mixed_soft_imagenet",
    "cifar10": "best_ood_test_varied_soft_cifar10_s2_mixed_soft_cifar10",
}

SWEEP_ID_PATTERN = re.compile(r"^[a-z0-9]{8}$")
SEVERITY_PATTERN = re.compile(r"_s(?P<severity>\d)(?P<mixed>_mixed.*)?$")

parser = argparse.ArgumentParser(
    description="Analyze aleatoric and epistemic uncertainty estimators from W&B runs"
)
parser.add_argument("dataset", choices=sorted(DATASET_PREFIX_LIST), help="Dataset to use")
parser.add_argument(
    "--mode",
    choices=["uncertainty", "estimator-correlation-matrix"],
    default="uncertainty",
    help=(
        "Plot mode. 'uncertainty' creates the aleatoric/epistemic diagnostic plots, "
        "while 'estimator-correlation-matrix' reproduces the estimator correlation "
        "matrix workflow with run-based W&B inputs."
    ),
)
parser.add_argument(
    "--method",
    action="append",
    default=[],
    help=(
        "Known method name from utils.py. Can be passed multiple times. "
        "Defaults to all distributional methods with registered sweep IDs."
    ),
)
parser.add_argument(
    "--sweep",
    action="append",
    default=[],
    help=(
        "Custom sweep specification in the form METHOD_NAME=SWEEP_ID or "
        "SWEEP_ID=METHOD_NAME."
    ),
)
parser.add_argument(
    "--run",
    action="append",
    default=[],
    help=(
        "Custom run specification in the form METHOD_NAME=RUN_ID, "
        "METHOD_NAME=ENTITY/PROJECT/RUN_ID, RUN_ID=METHOD_NAME, or "
        "ENTITY/PROJECT/RUN_ID=METHOD_NAME. Can be passed multiple times to "
        "aggregate several runs under one method."
    ),
)
parser.add_argument(
    "--estimators",
    nargs="*",
    default=None,
    help="Optional subset of estimator IDs to plot.",
)
parser.add_argument(
    "--matrix-estimator",
    choices=CORRELATION_MATRIX_ESTIMATORS,
    default="one_minus_max_probs_of_bma",
    help=(
        "Distributional estimator used in estimator-correlation-matrix mode for "
        "methods that log more than one estimator."
    ),
)
parser.add_argument(
    "--aleatoric-metric",
    choices=sorted(ALEATORIC_METRIC_DICT),
    default="rank_correlation_bregman_au",
    help="Metric used to judge aleatoric estimators against soft labels.",
)
parser.add_argument(
    "--epistemic-metric",
    choices=sorted(EPISTEMIC_METRIC_DICT),
    default="auroc_oodness",
    help="Metric used to judge epistemic estimators.",
)
parser.add_argument(
    "--correlation-kind",
    choices=["rank", "pearson"],
    default="rank",
    help="Correlation type used for the direct AU/EU correlation heatmap.",
)
parser.add_argument(
    "--include-mixed",
    action="store_true",
    help="Include mixed prefixes for the aleatoric task in addition to ID/OOD ones.",
)
parser.add_argument(
    "--checkpoint-selection-metric",
    default=DEFAULT_CHECKPOINT_SELECTION_METRIC,
    help=(
        "Training-time metric used to select the checkpoint whose best_* results are "
        "stored in W&B. This is recorded as metadata only."
    ),
)
parser.add_argument(
    "--entity",
    default="evelynemelanov",
    help="W&B entity that owns the sweeps.",
)
parser.add_argument(
    "--project",
    default="thesis",
    help="W&B project that stores the sweeps.",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=None,
    help="Directory in which to save the generated plots and CSV summaries.",
)


def setup_plot_style() -> None:
    """Sets up the plot style using tueplots and custom configurations."""
    config = bundles.neurips2024()
    plt.rcParams.update(config)
    plt.rcParams["text.latex.preamble"] += (
        r"\usepackage{amsmath} \usepackage{amsfonts} \usepackage{bm}"
    )


def get_metric_prefixes(dataset: str, metric_id: str, include_mixed: bool) -> list[str]:
    """Returns the dataset prefixes relevant for a specific metric."""
    prefixes = DATASET_PREFIX_LIST[dataset]

    if metric_id == "auroc_oodness":
        return [prefix for prefix in prefixes if "_mixed_" in prefix]

    if include_mixed:
        return prefixes

    return [prefix for prefix in prefixes if "_mixed_" not in prefix]


def merge_prefix_lists(*prefix_lists: list[str]) -> list[str]:
    """Merges ordered prefix lists while preserving first occurrence."""
    merged_prefixes = []

    for prefix_list in prefix_lists:
        for prefix in prefix_list:
            if prefix not in merged_prefixes:
                merged_prefixes.append(prefix)

    return merged_prefixes


def prefix_to_label(prefix: str) -> str:
    """Turns a long metric prefix into a concise axis label."""
    if prefix == "best_id_test":
        return "ID"

    match = SEVERITY_PATTERN.search(prefix)
    if match is None:
        return prefix.replace("best_", "").replace("_", " ")

    severity = match.group("severity")
    if match.group("mixed") is None:
        return f"OOD S{severity}"
    return f"Mix S{severity}"


def slugify(text: str) -> str:
    """Creates a filesystem-friendly slug."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def parse_sweep_spec(spec: str) -> tuple[str, str]:
    """Parses a custom sweep specification."""
    if "=" not in spec:
        msg = (
            "Invalid sweep specification. Use METHOD_NAME=SWEEP_ID or "
            f"SWEEP_ID=METHOD_NAME, got {spec!r}."
        )
        raise ValueError(msg)

    left, right = (part.strip() for part in spec.split("=", maxsplit=1))
    left_is_id = SWEEP_ID_PATTERN.fullmatch(left) is not None
    right_is_id = SWEEP_ID_PATTERN.fullmatch(right) is not None

    if left_is_id == right_is_id:
        msg = (
            "Could not infer which side is the sweep ID. Please provide exactly one "
            f"8-character sweep ID in {spec!r}."
        )
        raise ValueError(msg)

    if left_is_id:
        return right, left

    return left, right


def is_run_reference(value: str) -> bool:
    """Checks whether a string looks like a W&B run reference."""
    return "/" in value or SWEEP_ID_PATTERN.fullmatch(value) is not None


def parse_run_spec(spec: str) -> tuple[str, str]:
    """Parses a custom run specification."""
    if "=" not in spec:
        msg = (
            "Invalid run specification. Use METHOD_NAME=RUN_ID or "
            f"RUN_ID=METHOD_NAME, got {spec!r}."
        )
        raise ValueError(msg)

    left, right = (part.strip() for part in spec.split("=", maxsplit=1))
    left_is_run_ref = is_run_reference(left)
    right_is_run_ref = is_run_reference(right)

    if left_is_run_ref == right_is_run_ref:
        msg = (
            "Could not infer which side is the run reference."
        )
        raise ValueError(msg)

    if left_is_run_ref:
        return right, left

    return left, right


def resolve_method_sources(
    dataset: str,
    requested_methods: list[str],
    custom_sweep_specs: list[str],
    custom_run_specs: list[str],
    include_all_known_methods: bool = False,
) -> dict[str, dict[str, list[str]]]:
    """Builds the method-to-source mapping that should be analyzed."""
    known_method_to_sweep = {
        method_name: sweep_id
        for sweep_id, method_name in ID_TO_METHOD[dataset].items()
        if include_all_known_methods or method_name in DISTRIBUTIONAL_METHODS
    }

    selected_method_names = (
        requested_methods
        or list(known_method_to_sweep)
        if not custom_run_specs and not custom_sweep_specs
        else requested_methods
    )
    resolved_sources = defaultdict(lambda: {"sweeps": [], "runs": []})

    for method_name in selected_method_names:
        sweep_id = known_method_to_sweep.get(method_name)
        if sweep_id is None:
            logger.warning(
                "Method %s does not have a registered sweep ID for %s. "
                "Pass it explicitly with --sweep.",
                method_name,
                dataset,
            )
            continue

        resolved_sources[method_name]["sweeps"].append(sweep_id)

    for spec in custom_sweep_specs:
        method_name, sweep_id = parse_sweep_spec(spec)
        if sweep_id not in resolved_sources[method_name]["sweeps"]:
            resolved_sources[method_name]["sweeps"].append(sweep_id)

    for spec in custom_run_specs:
        method_name, run_ref = parse_run_spec(spec)
        if run_ref not in resolved_sources[method_name]["runs"]:
            resolved_sources[method_name]["runs"].append(run_ref)

    return dict(resolved_sources)


def get_finished_runs_from_sweep(
    api: wandb.Api,
    entity: str,
    project: str,
    sweep_id: str,
) -> list[wandb.apis.public.Run]:
    """Fetches the finished runs from a W&B sweep."""
    sweep: Sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    finished_runs = [run for run in sweep.runs if run.state == "finished"]
    logger.info("Sweep %s contains %d finished runs.", sweep_id, len(finished_runs))
    return finished_runs


def normalize_run_ref(entity: str, project: str, run_ref: str) -> str:
    """Normalizes a run reference into ENTITY/PROJECT/RUN_ID form."""
    if run_ref.count("/") == 2:
        return run_ref

    if run_ref.count("/") == 1:
        return f"{entity}/{run_ref}"

    return f"{entity}/{project}/{run_ref}"


def get_finished_runs_from_refs(
    api: wandb.Api,
    entity: str,
    project: str,
    run_refs: list[str],
) -> list[wandb.apis.public.Run]:
    """Fetches finished W&B runs from explicit run references."""
    finished_runs = []

    for run_ref in run_refs:
        normalized_ref = normalize_run_ref(entity=entity, project=project, run_ref=run_ref)
        run = api.run(normalized_ref)

        if run.state != "finished":
            logger.info("Run %s is in state %s and will be skipped.", normalized_ref, run.state)
            continue

        finished_runs.append(run)

    return finished_runs


def collect_finished_runs(
    api: wandb.Api,
    entity: str,
    project: str,
    sweep_ids: list[str],
    run_refs: list[str],
) -> list[wandb.apis.public.Run]:
    """Collects and deduplicates finished runs from sweeps and explicit refs."""
    collected_runs = []

    for sweep_id in sweep_ids:
        collected_runs.extend(
            get_finished_runs_from_sweep(
                api=api,
                entity=entity,
                project=project,
                sweep_id=sweep_id,
            )
        )

    collected_runs.extend(
        get_finished_runs_from_refs(
            api=api,
            entity=entity,
            project=project,
            run_refs=run_refs,
        )
    )

    deduplicated_runs = []
    seen_run_ids = set()

    for run in collected_runs:
        if run.id in seen_run_ids:
            continue

        seen_run_ids.add(run.id)
        deduplicated_runs.append(run)

    return deduplicated_runs


def is_valid_summary_value(value: Any) -> bool:
    """Checks whether a W&B summary value can be treated as a finite scalar."""
    if isinstance(value, str):
        return value != "NaN"

    try:
        float_value = float(value)
    except (TypeError, ValueError):
        return False

    return np.isfinite(float_value)


def match_prefix(key: str, prefixes: list[str]) -> str | None:
    """Matches a summary key to the longest compatible dataset prefix."""
    for prefix in sorted(prefixes, key=len, reverse=True):
        if key.startswith(f"{prefix}_"):
            return prefix

    return None


def direct_metric_name(stem: str, correlation_kind: str) -> str:
    """Returns the summary key suffix for a direct correlation metric."""
    prefix = "rank_correlation" if correlation_kind == "rank" else "correlation"
    return f"{prefix}_{stem}"


def order_items(items: set[str], preferred_order: list[str]) -> list[str]:
    """Orders discovered metric names using a preferred order first."""
    preferred = [item for item in preferred_order if item in items]
    leftovers = sorted(items.difference(preferred))
    return preferred + leftovers


def get_estimator_label(estimator: str) -> str:
    """Returns a human-readable estimator label."""
    return EXTRA_ESTIMATOR_LABELS.get(
        estimator, ESTIMATOR_CONVERSION_DICT.get(estimator, estimator)
    )


def get_estimator_correlation_prefix(dataset: str, metric_id: str) -> str:
    """Returns the metric prefix used by estimator-correlation-matrix mode."""
    if metric_id == "auroc_oodness":
        return ESTIMATOR_CORRELATION_MIXTURE_PREFIX[dataset]

    return ESTIMATOR_CORRELATION_ID_PREFIX


def extract_metric_token(key: str, prefix: str, metric_id: str) -> str | None:
    """Extracts the estimator token or estimatorless metric token from a summary key."""
    prefix_with_sep = f"{prefix}_"
    if not key.startswith(prefix_with_sep):
        return None

    stripped_key = key.removeprefix(prefix_with_sep)
    estimator_suffix = f"_{metric_id}"

    if stripped_key == metric_id:
        return metric_id

    if stripped_key.endswith(estimator_suffix):
        return stripped_key.removesuffix(estimator_suffix)

    return None


def is_estimator_correlation_token(token: str) -> bool:
    """Checks whether a summary-key token belongs in the estimator correlation matrix."""
    return (
        "mixed" not in token
        and "gt" not in token
        and (
            token in ESTIMATOR_CONVERSION_DICT or token in ESTIMATORLESS_METRICS
        )
    )


def discover_estimators(
    runs: list[wandb.apis.public.Run],
    prefixes: list[str],
    aleatoric_metric: str,
    epistemic_metric: str,
) -> list[str]:
    """Discovers estimator names that have relevant metrics logged."""
    suffixes = {f"_{aleatoric_metric}", f"_{epistemic_metric}"}
    discovered_estimators = set()

    for run in runs:
        for key, value in run.summary.items():
            if not is_valid_summary_value(value):
                continue

            prefix = match_prefix(key, prefixes)
            if prefix is None:
                continue

            for suffix in suffixes:
                if key.endswith(suffix):
                    estimator = key.removeprefix(f"{prefix}_").removesuffix(suffix)
                    discovered_estimators.add(estimator)
                    break

    focused_estimators = [
        estimator
        for estimator in FOCUSED_ESTIMATOR_ORDER
        if estimator in discovered_estimators
    ]
    if focused_estimators:
        return focused_estimators

    return order_items(discovered_estimators, FOCUSED_ESTIMATOR_ORDER)


def collect_metric_values(
    runs: list[wandb.apis.public.Run],
    prefixes: list[str],
    estimators: list[str],
    metric_id: str,
) -> dict[str, dict[str, list[float]]]:
    """Collects estimator-wise values for a single metric from run summaries."""
    estimator_set = set(estimators)
    suffix = f"_{metric_id}"
    results = defaultdict(lambda: defaultdict(list))

    for run in runs:
        for key, value in run.summary.items():
            if not is_valid_summary_value(value):
                continue

            prefix = match_prefix(key, prefixes)
            if prefix is None or not key.endswith(suffix):
                continue

            estimator = key.removeprefix(f"{prefix}_").removesuffix(suffix)
            if estimator not in estimator_set:
                continue

            results[prefix][estimator].append(float(value))

    return results


def collect_estimator_correlation_metric_values(
    runs: list[wandb.apis.public.Run],
    dataset: str,
    metric_id: str,
) -> dict[str, list[float]]:
    """Collects estimator-wise values for estimator-correlation-matrix mode."""
    prefix = get_estimator_correlation_prefix(dataset=dataset, metric_id=metric_id)
    results = defaultdict(list)

    for run in runs:
        for key, value in run.summary.items():
            if not is_valid_summary_value(value):
                continue

            token = extract_metric_token(key=key, prefix=prefix, metric_id=metric_id)
            if token is None or not is_estimator_correlation_token(token):
                continue

            results[token].append(float(value))

    return results


def collect_direct_correlation_values(
    runs: list[wandb.apis.public.Run],
    prefixes: list[str],
    correlation_kind: str,
) -> dict[str, dict[str, list[float]]]:
    """Collects direct AU-vs-EU correlation metrics from run summaries."""
    metric_names = {
        direct_metric_name(metric_stem, correlation_kind): metric_stem
        for metric_stem in DIRECT_CORRELATION_METRIC_DICT
    }
    results = defaultdict(lambda: defaultdict(list))

    for run in runs:
        for key, value in run.summary.items():
            if not is_valid_summary_value(value):
                continue

            prefix = match_prefix(key, prefixes)
            if prefix is None:
                continue

            for metric_name, metric_stem in metric_names.items():
                if key == f"{prefix}_{metric_name}":
                    results[prefix][metric_stem].append(float(value))
                    break

    return results


def summarize_values(values: list[float]) -> tuple[float, float, float, int]:
    """Returns mean, min, max, and count for a list of scalars."""
    if not values:
        return np.nan, np.nan, np.nan, 0

    array = np.asarray(values, dtype=float)
    return (
        float(array.mean()),
        float(array.min()),
        float(array.max()),
        int(array.size),
    )


def process_estimator_correlation_metric_values(
    metric_values: dict[str, list[float]],
    metric_id: str,
) -> dict[str, float]:
    """Averages estimator-correlation metric values and applies sign flips when needed."""
    processed_values = {}

    for token, values in metric_values.items():
        mean_value = summarize_values(values)[0]
        if np.isnan(mean_value):
            continue

        if metric_id in NEGATED_ESTIMATOR_CORRELATION_METRICS:
            mean_value = -mean_value

        processed_values[token] = mean_value

    return processed_values


def get_estimator_correlation_value(
    processed_values: dict[str, float],
    distributional_estimator: str,
    method_name: str,
    metric_id: str,
) -> float:
    """Selects the plotted value for one method and one metric."""
    if not processed_values:
        logger.warning(
            "No values found for %s on %s in estimator-correlation-matrix mode.",
            method_name,
            metric_id,
        )
        return np.nan

    if len(processed_values) == 1:
        return next(iter(processed_values.values()))

    if distributional_estimator not in processed_values:
        logger.warning(
            "Estimator %s is missing for %s on %s; available keys are %s.",
            distributional_estimator,
            method_name,
            metric_id,
            sorted(processed_values),
        )
        return np.nan

    return processed_values[distributional_estimator]


def available_estimators(
    aleatoric_values: dict[str, dict[str, list[float]]],
    aleatoric_prefixes: list[str],
    epistemic_values: dict[str, dict[str, list[float]]],
    epistemic_prefixes: list[str],
    estimators: list[str],
) -> list[str]:
    """Filters the estimator list down to those that have at least one metric."""
    filtered = []

    for estimator in estimators:
        has_aleatoric_data = any(
            aleatoric_values.get(prefix, {}).get(estimator, [])
            for prefix in aleatoric_prefixes
        )
        has_epistemic_data = any(
            epistemic_values.get(prefix, {}).get(estimator, [])
            for prefix in epistemic_prefixes
        )
        if has_aleatoric_data or has_epistemic_data:
            filtered.append(estimator)

    return filtered


def available_direct_metrics(
    direct_values: dict[str, dict[str, list[float]]],
    prefixes: list[str],
) -> list[str]:
    """Filters direct correlation metrics down to those that have values."""
    filtered = []

    for metric_stem in DIRECT_CORRELATION_METRIC_DICT:
        has_data = any(
            direct_values.get(prefix, {}).get(metric_stem, []) for prefix in prefixes
        )
        if has_data:
            filtered.append(metric_stem)

    return filtered


def calculate_pairwise_correlation(
    values_a: np.ndarray,
    values_b: np.ndarray,
    correlation_kind: str,
) -> float:
    """Calculates a pairwise correlation while ignoring methods with missing values."""
    mask = np.isfinite(values_a) & np.isfinite(values_b)
    if mask.sum() < 2:
        return np.nan

    filtered_a = values_a[mask]
    filtered_b = values_b[mask]

    if correlation_kind == "spearman":
        return float(spearmanr(filtered_a, filtered_b)[0])

    return float(pearsonr(filtered_a, filtered_b)[0])


def calculate_estimator_correlation_matrices(
    performance_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates Spearman and Pearson correlation matrices across metrics."""
    num_metrics = performance_matrix.shape[0]
    spearman_matrix = np.full((num_metrics, num_metrics), np.nan)
    pearson_matrix = np.full((num_metrics, num_metrics), np.nan)

    for row_idx in range(num_metrics):
        for col_idx in range(num_metrics):
            values_a = performance_matrix[row_idx, :]
            values_b = performance_matrix[col_idx, :]
            spearman_matrix[row_idx, col_idx] = calculate_pairwise_correlation(
                values_a=values_a,
                values_b=values_b,
                correlation_kind="spearman",
            )
            pearson_matrix[row_idx, col_idx] = calculate_pairwise_correlation(
                values_a=values_a,
                values_b=values_b,
                correlation_kind="pearson",
            )

    return spearman_matrix, pearson_matrix


def build_metric_matrix(
    metric_values: dict[str, dict[str, list[float]]],
    prefixes: list[str],
    estimators_or_metrics: list[str],
) -> np.ndarray:
    """Builds a prefix-by-column matrix from a nested metric dictionary."""
    matrix = np.full((len(prefixes), len(estimators_or_metrics)), np.nan)

    for row_idx, prefix in enumerate(prefixes):
        for col_idx, item in enumerate(estimators_or_metrics):
            values = metric_values.get(prefix, {}).get(item, [])
            matrix[row_idx, col_idx] = summarize_values(values)[0]

    return matrix


def render_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    xlabels: list[str],
    ylabels: list[str],
    title: str,
) -> AxesImage:
    """Renders a heatmap with missing values shown in light gray."""
    cmap = copy(plt.get_cmap("coolwarm"))
    cmap.set_bad("#f3f3f3")
    masked_matrix = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked_matrix, aspect="auto", cmap=cmap, vmin=-1, vmax=1)

    ax.set_title(title)
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels)
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.spines[["right", "top"]].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if matrix.size <= 96:
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                if np.isnan(value):
                    ax.text(
                        col_idx,
                        row_idx,
                        "n/a",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="black",
                    )
                    continue

                color = "white" if abs(value) > 0.5 else "black"
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=color,
                )

    return image


def plot_quality_heatmaps(
    method_name: str,
    estimators: list[str],
    aleatoric_values: dict[str, dict[str, list[float]]],
    aleatoric_prefixes: list[str],
    aleatoric_metric: str,
    epistemic_values: dict[str, dict[str, list[float]]],
    epistemic_prefixes: list[str],
    epistemic_metric: str,
    save_path: Path,
) -> None:
    """Plots aleatoric-task and epistemic-task heatmaps side by side."""
    if not estimators:
        logger.info("No estimators available for %s.", method_name)
        return

    aleatoric_matrix = build_metric_matrix(
        metric_values=aleatoric_values,
        prefixes=aleatoric_prefixes,
        estimators_or_metrics=estimators,
    )
    epistemic_matrix = build_metric_matrix(
        metric_values=epistemic_values,
        prefixes=epistemic_prefixes,
        estimators_or_metrics=estimators,
    )

    figure_width = max(7.2, 1.2 + 0.55 * len(estimators) * 2)
    figure_height = max(
        2.8, 1.4 + 0.45 * max(len(aleatoric_prefixes), len(epistemic_prefixes))
    )
    fig, axes = plt.subplots(
        ncols=2,
        figsize=(figure_width, figure_height),
        constrained_layout=True,
    )

    estimator_labels = [get_estimator_label(estimator) for estimator in estimators]

    image = render_heatmap(
        ax=axes[0],
        matrix=aleatoric_matrix,
        xlabels=estimator_labels,
        ylabels=[prefix_to_label(prefix) for prefix in aleatoric_prefixes],
        title=ALEATORIC_METRIC_DICT[aleatoric_metric],
    )
    render_heatmap(
        ax=axes[1],
        matrix=epistemic_matrix,
        xlabels=estimator_labels,
        ylabels=[prefix_to_label(prefix) for prefix in epistemic_prefixes],
        title=EPISTEMIC_METRIC_DICT[epistemic_metric],
    )

    fig.suptitle(f"{method_name}: Aleatoric vs Epistemic Estimator Quality")
    colorbar = fig.colorbar(image, ax=axes, fraction=0.035, pad=0.02)
    colorbar.outline.set_visible(False)
    colorbar.set_label("Metric value")
    fig.savefig(save_path)
    plt.close(fig)


def compute_disentanglement_scores(
    estimators: list[str],
    aleatoric_values: dict[str, dict[str, list[float]]],
    aleatoric_prefixes: list[str],
    epistemic_values: dict[str, dict[str, list[float]]],
    epistemic_prefixes: list[str],
) -> dict[str, tuple[float, float, float]]:
    """Computes matched-task minus mismatched-task scores per estimator."""
    scores = {}

    for estimator in estimators:
        role = ESTIMATOR_ROLE.get(estimator)
        if role is None:
            continue

        aleatoric_score = summarize_values([
            value
            for prefix in aleatoric_prefixes
            for value in aleatoric_values.get(prefix, {}).get(estimator, [])
        ])[0]
        epistemic_score = summarize_values([
            value
            for prefix in epistemic_prefixes
            for value in epistemic_values.get(prefix, {}).get(estimator, [])
        ])[0]

        if np.isnan(aleatoric_score) or np.isnan(epistemic_score):
            continue

        matched_score = aleatoric_score if role == "au" else epistemic_score
        mismatched_score = epistemic_score if role == "au" else aleatoric_score
        scores[estimator] = (matched_score - mismatched_score, matched_score, mismatched_score)

    return scores


def plot_disentanglement_scores(
    method_name: str,
    disentanglement_scores: dict[str, tuple[float, float, float]],
    save_path: Path,
) -> None:
    """Plots matched-minus-mismatched task scores as horizontal bars."""
    if not disentanglement_scores:
        logger.info("No disentanglement scores available for %s.", method_name)
        return

    summarized_scores = [
        (estimator, score_values[0], score_values[1], score_values[2])
        for estimator, score_values in disentanglement_scores.items()
    ]
    summarized_scores.sort(key=lambda item: item[1], reverse=True)

    estimators = [item[0] for item in summarized_scores]
    score_differences = np.array([item[1] for item in summarized_scores])
    positions = np.arange(len(estimators))
    colors = [
        COLOR_ESTIMATE
        if ESTIMATOR_ROLE.get(estimator) == "au"
        else COLOR_DISTRIBUTIONAL
        if ESTIMATOR_ROLE.get(estimator) == "eu"
        else COLOR_BASELINE
        for estimator in estimators
    ]

    figure_height = max(2.6, 1.0 + 0.38 * len(estimators))
    fig, ax = plt.subplots(figsize=(5.7, figure_height), constrained_layout=True)

    ax.barh(positions, score_differences, color=colors, zorder=2)
    ax.axvline(0, color=COLOR_ERROR_BAR, linewidth=1, zorder=1)
    ax.grid(axis="x", zorder=1, linewidth=0.5)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_yticks(positions)
    ax.set_yticklabels([get_estimator_label(estimator) for estimator in estimators])
    ax.set_xlabel("Matched task score - mismatched task score")
    ax.set_title(f"{method_name}: Disentanglement Score")

    legend_handles = [
        Patch(facecolor=COLOR_ESTIMATE, label=TARGET_TO_LONG_LABEL["au"]),
        Patch(facecolor=COLOR_DISTRIBUTIONAL, label=TARGET_TO_LONG_LABEL["eu"]),
    ]
    ax.legend(frameon=False, handles=legend_handles, fontsize="small", loc="lower right")

    fig.savefig(save_path)
    plt.close(fig)


def plot_direct_correlation_heatmap(
    method_name: str,
    prefixes: list[str],
    direct_values: dict[str, dict[str, list[float]]],
    correlation_kind: str,
    save_path: Path,
) -> None:
    """Plots direct AU-vs-EU correlation metrics as a heatmap."""
    metric_stems = available_direct_metrics(direct_values=direct_values, prefixes=prefixes)
    if not metric_stems:
        logger.info("No direct AU/EU correlation metrics available for %s.", method_name)
        return

    matrix = build_metric_matrix(
        metric_values=direct_values,
        prefixes=prefixes,
        estimators_or_metrics=metric_stems,
    )
    figure_width = max(5.2, 1.2 + 0.58 * len(metric_stems))
    figure_height = max(2.6, 1.4 + 0.45 * len(prefixes))
    fig, ax = plt.subplots(
        figsize=(figure_width, figure_height),
        constrained_layout=True,
    )
    image = render_heatmap(
        ax=ax,
        matrix=matrix,
        xlabels=[
            DIRECT_CORRELATION_METRIC_DICT.get(metric_stem, metric_stem)
            for metric_stem in metric_stems
        ],
        ylabels=[prefix_to_label(prefix) for prefix in prefixes],
        title=f"{method_name}: Direct {correlation_kind.title()} AU/EU Correlations",
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colorbar.outline.set_visible(False)
    colorbar.set_label("Correlation")
    fig.savefig(save_path)
    plt.close(fig)


def plot_estimator_correlation_matrix(
    correlation_matrix: np.ndarray,
    correlation_kind: str,
    matrix_estimator: str,
    save_path: Path,
) -> None:
    """Plots the metric-correlation matrix used by estimator-correlation-matrix mode."""
    metric_labels = list(ESTIMATOR_CORRELATION_METRIC_DICT.values())
    figure_size = max(5.4, 0.48 * len(metric_labels))
    fig, ax = plt.subplots(figsize=(figure_size, figure_size), constrained_layout=True)
    image = render_heatmap(
        ax=ax,
        matrix=correlation_matrix,
        xlabels=metric_labels,
        ylabels=metric_labels,
        title=(
            f"{get_estimator_label(matrix_estimator)}: "
            f"{correlation_kind.title()} Metric Correlations"
        ),
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.outline.set_visible(False)
    colorbar.set_label("Correlation")
    fig.savefig(save_path)
    plt.close(fig)


def write_metric_values_csv(
    method_name: str,
    metric_id: str,
    metric_label: str,
    prefixes: list[str],
    estimators: list[str],
    metric_values: dict[str, dict[str, list[float]]],
    save_path: Path,
) -> None:
    """Writes the aggregated estimator-wise values for one metric to CSV."""
    with save_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "method",
                "metric_id",
                "metric_label",
                "prefix",
                "prefix_label",
                "estimator",
                "estimator_label",
                "mean",
                "min",
                "max",
                "count",
            ]
        )

        for prefix in prefixes:
            for estimator in estimators:
                values = metric_values.get(prefix, {}).get(estimator, [])
                mean_value, min_value, max_value, count = summarize_values(values)
                writer.writerow(
                    [
                        method_name,
                        metric_id,
                        metric_label,
                        prefix,
                        prefix_to_label(prefix),
                        estimator,
                        get_estimator_label(estimator),
                        mean_value,
                        min_value,
                        max_value,
                        count,
                    ]
                )


def write_direct_correlation_csv(
    method_name: str,
    prefixes: list[str],
    direct_values: dict[str, dict[str, list[float]]],
    save_path: Path,
) -> None:
    """Writes the aggregated direct AU-vs-EU correlations to CSV."""
    with save_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "method",
                "prefix",
                "prefix_label",
                "metric",
                "metric_label",
                "mean",
                "min",
                "max",
                "count",
            ]
        )

        for prefix in prefixes:
            for metric_stem in DIRECT_CORRELATION_METRIC_DICT:
                values = direct_values.get(prefix, {}).get(metric_stem, [])
                mean_value, min_value, max_value, count = summarize_values(values)
                writer.writerow(
                    [
                        method_name,
                        prefix,
                        prefix_to_label(prefix),
                        metric_stem,
                        DIRECT_CORRELATION_METRIC_DICT.get(metric_stem, metric_stem),
                        mean_value,
                        min_value,
                        max_value,
                        count,
                    ]
                )


def write_metadata_csv(
    method_name: str,
    sweep_ids: list[str],
    run_refs: list[str],
    dataset: str,
    checkpoint_selection_metric: str,
    aleatoric_metric: str,
    epistemic_metric: str,
    save_path: Path,
) -> None:
    """Writes run-selection and plotting metadata to CSV."""
    with save_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["key", "value"])
        writer.writerow(["method", method_name])
        writer.writerow(["sweep_ids", ",".join(sweep_ids)])
        writer.writerow(["run_refs", ",".join(run_refs)])
        writer.writerow(["dataset", dataset])
        writer.writerow(["checkpoint_selection_metric", checkpoint_selection_metric])
        writer.writerow(["aleatoric_metric", aleatoric_metric])
        writer.writerow(["epistemic_metric", epistemic_metric])


def write_estimator_correlation_performance_csv(
    method_names: list[str],
    performance_matrix: np.ndarray,
    save_path: Path,
) -> None:
    """Writes the metric-by-method performance matrix to CSV."""
    metric_ids = list(ESTIMATOR_CORRELATION_METRIC_DICT)
    with save_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric_id", "metric_label", *method_names])

        for row_idx, metric_id in enumerate(metric_ids):
            writer.writerow(
                [
                    metric_id,
                    ESTIMATOR_CORRELATION_METRIC_DICT[metric_id],
                    *performance_matrix[row_idx, :].tolist(),
                ]
            )


def write_square_matrix_csv(
    matrix: np.ndarray,
    metric_labels: list[str],
    save_path: Path,
) -> None:
    """Writes a square metric-by-metric matrix to CSV."""
    with save_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", *metric_labels])

        for row_label, row_values in zip(metric_labels, matrix, strict=False):
            writer.writerow([row_label, *row_values.tolist()])


def build_estimator_correlation_performance_matrix(
    api: wandb.Api,
    args: argparse.Namespace,
) -> tuple[list[str], np.ndarray]:
    """Builds the metric-by-method performance matrix for correlation-matrix mode."""
    method_sources = resolve_method_sources(
        dataset=args.dataset,
        requested_methods=args.method,
        custom_sweep_specs=args.sweep,
        custom_run_specs=args.run,
        include_all_known_methods=True,
    )

    if not method_sources:
        msg = (
            "No methods resolved. Use --run for plain W&B runs, or --method/--sweep "
            "if you also have sweep-based experiments."
        )
        raise ValueError(msg)

    metric_ids = list(ESTIMATOR_CORRELATION_METRIC_DICT)
    performance_rows = [[] for _ in metric_ids]
    method_names = []

    for method_name, method_source in tqdm(method_sources.items(), desc="Methods"):
        runs = collect_finished_runs(
            api=api,
            entity=args.entity,
            project=args.project,
            sweep_ids=method_source["sweeps"],
            run_refs=method_source["runs"],
        )
        if not runs:
            logger.warning("Skipping %s because no finished runs were found.", method_name)
            continue

        method_names.append(method_name)

        for row_idx, metric_id in enumerate(metric_ids):
            metric_values = collect_estimator_correlation_metric_values(
                runs=runs,
                dataset=args.dataset,
                metric_id=metric_id,
            )
            processed_values = process_estimator_correlation_metric_values(
                metric_values=metric_values,
                metric_id=metric_id,
            )
            performance_rows[row_idx].append(
                get_estimator_correlation_value(
                    processed_values=processed_values,
                    distributional_estimator=args.matrix_estimator,
                    method_name=method_name,
                    metric_id=metric_id,
                )
            )

    if not method_names:
        msg = "No finished runs were found for estimator-correlation-matrix mode."
        raise ValueError(msg)

    return method_names, np.asarray(performance_rows, dtype=float)


def run_uncertainty_mode(api: wandb.Api, args: argparse.Namespace) -> None:
    """Runs the original uncertainty diagnostics."""
    aleatoric_prefixes = get_metric_prefixes(
        dataset=args.dataset,
        metric_id=args.aleatoric_metric,
        include_mixed=args.include_mixed,
    )
    epistemic_prefixes = get_metric_prefixes(
        dataset=args.dataset,
        metric_id=args.epistemic_metric,
        include_mixed=True,
    )
    direct_prefixes = merge_prefix_lists(aleatoric_prefixes, epistemic_prefixes)

    method_sources = resolve_method_sources(
        dataset=args.dataset,
        requested_methods=args.method,
        custom_sweep_specs=args.sweep,
        custom_run_specs=args.run,
    )

    if not method_sources:
        msg = (
            "No methods resolved. Use --run for plain W&B runs, or --method/--sweep "
            "if you also have sweep-based experiments."
        )
        raise ValueError(msg)

    output_root = args.output_dir or Path(f"results/{args.dataset}/uncertainty")
    output_root.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Assuming best_* metrics were produced using checkpoint selection metric %s.",
        args.checkpoint_selection_metric,
    )

    for method_name, method_source in tqdm(method_sources.items(), desc="Methods"):
        sweep_ids = method_source["sweeps"]
        run_refs = method_source["runs"]
        logger.info(
            "Processing %s with %d sweeps and %d explicit runs.",
            method_name,
            len(sweep_ids),
            len(run_refs),
        )
        runs = collect_finished_runs(
            api=api,
            entity=args.entity,
            project=args.project,
            sweep_ids=sweep_ids,
            run_refs=run_refs,
        )

        if not runs:
            logger.warning("Skipping %s because no finished runs were found.", method_name)
            continue

        estimators = args.estimators or discover_estimators(
            runs=runs,
            prefixes=direct_prefixes,
            aleatoric_metric=args.aleatoric_metric,
            epistemic_metric=args.epistemic_metric,
        )
        aleatoric_values = collect_metric_values(
            runs=runs,
            prefixes=aleatoric_prefixes,
            estimators=estimators,
            metric_id=args.aleatoric_metric,
        )
        epistemic_values = collect_metric_values(
            runs=runs,
            prefixes=epistemic_prefixes,
            estimators=estimators,
            metric_id=args.epistemic_metric,
        )
        estimators = available_estimators(
            aleatoric_values=aleatoric_values,
            aleatoric_prefixes=aleatoric_prefixes,
            epistemic_values=epistemic_values,
            epistemic_prefixes=epistemic_prefixes,
            estimators=estimators,
        )
        direct_values = collect_direct_correlation_values(
            runs=runs,
            prefixes=direct_prefixes,
            correlation_kind=args.correlation_kind,
        )
        disentanglement_scores = compute_disentanglement_scores(
            estimators=estimators,
            aleatoric_values=aleatoric_values,
            aleatoric_prefixes=aleatoric_prefixes,
            epistemic_values=epistemic_values,
            epistemic_prefixes=epistemic_prefixes,
        )

        method_output_dir = output_root / slugify(method_name)
        method_output_dir.mkdir(parents=True, exist_ok=True)

        write_metadata_csv(
            method_name=method_name,
            sweep_ids=sweep_ids,
            run_refs=run_refs,
            dataset=args.dataset,
            checkpoint_selection_metric=args.checkpoint_selection_metric,
            aleatoric_metric=args.aleatoric_metric,
            epistemic_metric=args.epistemic_metric,
            save_path=method_output_dir / "metadata.csv",
        )
        write_metric_values_csv(
            method_name=method_name,
            metric_id=args.aleatoric_metric,
            metric_label=ALEATORIC_METRIC_DICT[args.aleatoric_metric],
            prefixes=aleatoric_prefixes,
            estimators=estimators,
            metric_values=aleatoric_values,
            save_path=method_output_dir / "aleatoric_metric.csv",
        )
        write_metric_values_csv(
            method_name=method_name,
            metric_id=args.epistemic_metric,
            metric_label=EPISTEMIC_METRIC_DICT[args.epistemic_metric],
            prefixes=epistemic_prefixes,
            estimators=estimators,
            metric_values=epistemic_values,
            save_path=method_output_dir / "epistemic_metric.csv",
        )
        write_direct_correlation_csv(
            method_name=method_name,
            prefixes=direct_prefixes,
            direct_values=direct_values,
            save_path=method_output_dir / "direct_correlations.csv",
        )
        plot_quality_heatmaps(
            method_name=method_name,
            estimators=estimators,
            aleatoric_values=aleatoric_values,
            aleatoric_prefixes=aleatoric_prefixes,
            aleatoric_metric=args.aleatoric_metric,
            epistemic_values=epistemic_values,
            epistemic_prefixes=epistemic_prefixes,
            epistemic_metric=args.epistemic_metric,
            save_path=method_output_dir / "quality_heatmaps.pdf",
        )
        plot_disentanglement_scores(
            method_name=method_name,
            disentanglement_scores=disentanglement_scores,
            save_path=method_output_dir / "disentanglement_scores.pdf",
        )
        plot_direct_correlation_heatmap(
            method_name=method_name,
            prefixes=direct_prefixes,
            direct_values=direct_values,
            correlation_kind=args.correlation_kind,
            save_path=method_output_dir / "direct_uncertainty_correlations.pdf",
        )
        logger.info("Saved %s outputs to %s.", method_name, method_output_dir)


def run_estimator_correlation_matrix_mode(api: wandb.Api, args: argparse.Namespace) -> None:
    """Runs the metric-correlation workflow from plot_estimator_correlation_matrix.py."""
    output_root = args.output_dir or Path(
        f"results/{args.dataset}/{args.matrix_estimator}_correlation_matrix"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    method_names, performance_matrix = build_estimator_correlation_performance_matrix(
        api=api,
        args=args,
    )
    spearman_matrix, pearson_matrix = calculate_estimator_correlation_matrices(
        performance_matrix=performance_matrix
    )

    plot_estimator_correlation_matrix(
        correlation_matrix=spearman_matrix,
        correlation_kind="spearman",
        matrix_estimator=args.matrix_estimator,
        save_path=output_root / "correlation_matrix_spearman.pdf",
    )
    plot_estimator_correlation_matrix(
        correlation_matrix=pearson_matrix,
        correlation_kind="pearson",
        matrix_estimator=args.matrix_estimator,
        save_path=output_root / "correlation_matrix_pearson.pdf",
    )
    write_estimator_correlation_performance_csv(
        method_names=method_names,
        performance_matrix=performance_matrix,
        save_path=output_root / "performance_matrix.csv",
    )
    write_square_matrix_csv(
        matrix=spearman_matrix,
        metric_labels=list(ESTIMATOR_CORRELATION_METRIC_DICT.values()),
        save_path=output_root / "correlation_matrix_spearman.csv",
    )
    write_square_matrix_csv(
        matrix=pearson_matrix,
        metric_labels=list(ESTIMATOR_CORRELATION_METRIC_DICT.values()),
        save_path=output_root / "correlation_matrix_pearson.csv",
    )
    logger.info("Saved estimator-correlation-matrix outputs to %s.", output_root)


def main() -> None:
    """Main entry point for uncertainty analysis plots."""
    setup_plot_style()
    args = parser.parse_args()
    api = wandb.Api()
    if args.mode == "estimator-correlation-matrix":
        run_estimator_correlation_matrix_mode(api=api, args=args)
        return

    run_uncertainty_mode(api=api, args=args)


if __name__ == "__main__":
    main()
