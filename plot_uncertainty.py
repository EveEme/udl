"""
WandB Plotting Script
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.stats import gaussian_kde, spearmanr
import warnings
warnings.filterwarnings('ignore')

from tueplots import bundles

TAB10 = plt.get_cmap("tab10").colors

COLORS = {
    'ce': TAB10[0],
    'laplace_first': TAB10[1],
    'laplace_last': TAB10[2],
    'laplace_full': TAB10[3],
}

SUMMARY_KEY_GROUPS = {
    'rank_correlation_epistemic_aleatoric': (
        'rank_correlation_epistemic_aleatoric',
        'best_id_test_rank_correlation_it_au_eu',
        'id_test_test_loader_rank_correlation_it_au_eu',
        'rank_correlation_it_au_eu',
        '*rank_correlation_it_au_eu',
    ),
    'rank_correlation_it_au_eu': (
        'rank_correlation_it_au_eu',
        '*rank_correlation_it_au_eu',
    ),
    'correlation_it_au_eu': (
        'correlation_it_au_eu',
        '*correlation_it_au_eu',
    ),
    'correlation_au_it_au_vs_eu': (
        'correlation_au_it_au_vs_eu',
        '*correlation_au_it_au_vs_eu',
    ),
    'correlation_eu_it_au_vs_eu': (
        'correlation_eu_it_au_vs_eu',
        '*correlation_eu_it_au_vs_eu',
    ),
    'correlation_au_it_au_vs_eu_it_eu': (
        'correlation_au_it_au_vs_eu_it_eu',
        '*correlation_au_it_au_vs_eu_it_eu',
    ),
    'correlation_eu_it_au_vs_au_it_eu': (
        'correlation_eu_it_au_vs_au_it_eu',
        '*correlation_eu_it_au_vs_au_it_eu',
    ),
    'rank_correlation_au_it_au_vs_eu': (
        'rank_correlation_au_it_au_vs_eu',
        '*rank_correlation_au_it_au_vs_eu',
    ),
    'rank_correlation_eu_it_au_vs_eu': (
        'rank_correlation_eu_it_au_vs_eu',
        '*rank_correlation_eu_it_au_vs_eu',
    ),
    'rank_correlation_au_it_au_vs_eu_it_eu': (
        'rank_correlation_au_it_au_vs_eu_it_eu',
        '*rank_correlation_au_it_au_vs_eu_it_eu',
    ),
    'rank_correlation_eu_it_au_vs_au_it_eu': (
        'rank_correlation_eu_it_au_vs_au_it_eu',
        '*rank_correlation_eu_it_au_vs_au_it_eu',
    ),
    'predictive_uncertainty': (
        'predictive_uncertainty',
        'entropies_of_bma',
        '*entropies_of_bma',
        '*entropies_of_bma_time_forward_m',
        '*correlation_it_au_pu',
    ),
    'auroc_ood': (
        'auroc_ood',
        'auroc_oodness',
        'best_ood_test_varied_soft_cifar10_s2_mixed_soft_cifar10_one_minus_max_probs_of_bma_auroc_oodness',
        'ood_test_test_loader_mixed_soft_cifar10_one_minus_max_probs_of_bma_auroc_oodness',
        '*au_it_eu_auroc_oodness',
        '*eu_it_eu_auroc_oodness',
        '*jensen_shannon_divergences_auroc_oodness',
        '*expected_divergences_auroc_oodness',
        '*one_minus_max_probs_of_bma_auroc_oodness',
    ),
    'auroc_oodness': (
        'auroc_ood',
        'auroc_oodness',
        '*au_it_eu_auroc_oodness',
        '*eu_it_eu_auroc_oodness',
        '*jensen_shannon_divergences_auroc_oodness',
        '*expected_divergences_auroc_oodness',
        '*one_minus_max_probs_of_bma_auroc_oodness',
    ),
    'epistemic_auroc_oodness': (
        '*eu_it_eu_auroc_oodness',
        '*jensen_shannon_divergences_auroc_oodness',
        '*expected_divergences_auroc_oodness',
    ),
    'predictive_entropy_auroc_oodness': (
        '*entropies_of_bma_auroc_oodness',
        '*one_minus_max_probs_of_bma_auroc_oodness',
    ),
    'aleatoric_vs_gt': (
        'aleatoric_vs_gt',
        'best_id_test_rank_correlation_bregman_au',
        'id_test_test_loader_rank_correlation_bregman_au',
        'rank_correlation_bregman_au',
        '*au_it_au_rank_correlation_bregman_au',
        '*eu_it_au_rank_correlation_bregman_au',
        '*expected_entropies_rank_correlation_bregman_au',
        '*rank_correlation_bregman_au',
    ),
    'aleatoric_estimate_vs_gt': (
        '*au_it_au_rank_correlation_bregman_au',
        '*expected_entropies_rank_correlation_bregman_au',
        '*rank_correlation_bregman_au',
    ),
    'epistemic_uncertainty': ('epistemic_uncertainty',),
    'test_accuracy': (
        'test_accuracy',
        'best_id_test_hard_bma_accuracy_original',
        'id_test_test_loader_hard_bma_accuracy_original',
        'hard_bma_accuracy_original',
    ),
}

SUMMARY_KEYS = tuple(
    key for keys in SUMMARY_KEY_GROUPS.values() for key in keys if not key.startswith("*")
)
FILTER_SUMMARY_KEYS = tuple(sorted(set(
    list(SUMMARY_KEYS)
    + [
        'id_test_test_loader_rank_correlation_au_it_au_vs_eu',
        'id_test_test_loader_rank_correlation_eu_it_au_vs_eu',
        'id_test_test_loader_rank_correlation_au_it_au_vs_eu_it_eu',
        'id_test_test_loader_rank_correlation_eu_it_au_vs_au_it_eu',
        'id_test_test_loader_correlation_au_it_au_vs_eu',
        'id_test_test_loader_correlation_eu_it_au_vs_eu',
        'id_test_test_loader_correlation_au_it_au_vs_eu_it_eu',
        'id_test_test_loader_correlation_eu_it_au_vs_au_it_eu',
        'id_test_test_loader_au_it_au_rank_correlation_bregman_au',
        'id_test_test_loader_eu_it_eu_auroc_oodness',
        'ood_test_test_loader_mixed_soft_cifar10_eu_it_eu_auroc_oodness',
        'ood_test_test_loader_mixed_soft_cifar10_entropies_of_bma_auroc_oodness',
    ]
)))

RUN_NAMES = {
    "splendid-cherry-131": "CE Baseline",
    "pious-lake-132": "Low-rank EU First 10",
    "lunar-durian-133": "Kfac EU First 10",
    "dashing-music-134": "Kfac EU First 30",
    "misty-lake-135": "Low-rank EU Last 30",
    "comfy-sun-136": "Kfac EU Last 30",
}

def to_float(value, default=np.nan) -> float:
    """Convert scalar W&B summary values to float; return NaN otherwise."""
    try:
        if value is None:
            return default
        if isinstance(value, (dict, list, tuple, str)):
            return default
        array = np.asarray(value)
        if array.shape != ():
            return default
        return float(array)
    except (TypeError, ValueError):
        return default


def is_valid_number(value) -> bool:
    """Return whether value is a finite scalar number."""
    return np.isfinite(to_float(value))


def first_summary_float(summary, *keys: str) -> float:
    """Return the first finite scalar value found in a W&B summary."""
    for key in keys:
        if key.startswith("*"):
            suffix = key[1:]
            matching_keys = sorted(
                summary_key for summary_key in summary.keys()
                if summary_key.endswith(suffix)
            )
            for matching_key in matching_keys:
                value = to_float(summary.get(matching_key, np.nan))
                if np.isfinite(value):
                    return value
            continue

        value = to_float(summary.get(key, np.nan))
        if np.isfinite(value):
            return value
    return np.nan


def draw_no_data(ax, title: str, message: str = "No finite scalar values found.") -> None:
    """Draw a placeholder instead of failing on empty W&B summaries."""
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.text(0.5, 0.5, message, ha='center', va='center', transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def infer_experiment(*names: str) -> str:
    """Infer flella/llefla experiment label from run names."""
    joined = " ".join(name for name in names if name).lower()
    if (
        "flella" in joined
        or "eu first" in joined
        or ("epistemic first" in joined and "aleatoric last" in joined)
    ):
        return "flella"
    if (
        "llefla" in joined
        or "eu last" in joined
        or ("epistemic last" in joined and "aleatoric first" in joined)
    ):
        return "llefla"
    return "other"


def split_metrics_by_experiment(metrics: Dict) -> Dict[str, Dict]:
    """Group extracted metrics by inferred experiment name."""
    grouped = {"flella": {}, "llefla": {}, "other": {}}
    for run_name, metric_dict in metrics.items():
        experiment = metric_dict.get("experiment", "other")
        grouped.setdefault(experiment, {})[run_name] = metric_dict
    return {key: value for key, value in grouped.items() if value}


def tensor_to_numpy(values) -> np.ndarray:
    """Convert tensors or arrays to a one-dimensional NumPy array."""
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        values = values.numpy()
    return np.asarray(values, dtype=float).reshape(-1)


def load_uncertainty_tuple(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a saved ``(aleatoric, epistemic)`` tuple from validate.py."""
    import torch

    try:
        values = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        values = torch.load(path, map_location="cpu")
    if not isinstance(values, (tuple, list)) or len(values) != 2:
        msg = f"{path} must contain a tuple/list with two tensors."
        raise ValueError(msg)
    return tensor_to_numpy(values[0]), tensor_to_numpy(values[1])


def plot_density_loglog_scatter(
    aleatoric: np.ndarray,
    epistemic: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """Plot aleatoric vs epistemic uncertainty as log-log density scatter."""
    mask = (
        np.isfinite(aleatoric)
        & np.isfinite(epistemic)
        & (aleatoric > 0)
        & (epistemic > 0)
    )
    aleatoric = aleatoric[mask]
    epistemic = epistemic[mask]

    fig, ax = plt.subplots(figsize=(6, 5))
    if aleatoric.size < 3:
        draw_no_data(ax, title, "Need at least 3 positive finite samples.")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    xy = np.vstack([np.log10(aleatoric), np.log10(epistemic)])
    try:
        density = gaussian_kde(xy)(xy)
    except np.linalg.LinAlgError:
        density = np.ones_like(aleatoric)

    idx = density.argsort()
    aleatoric = aleatoric[idx]
    epistemic = epistemic[idx]
    density = density[idx]

    rho = spearmanr(aleatoric, epistemic).statistic
    scatter = ax.scatter(
        aleatoric,
        epistemic,
        c=density,
        s=10,
        cmap="plasma",
        edgecolors="none",
    )
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Density")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Aleatoric Uncertainty", color="blue")
    ax.set_ylabel("Epistemic Uncertainty", color="red")
    ax.set_title(f"{title}\nSpearman $\\rho={rho:.3f}$")
    ax.grid(alpha=0.25)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_saved_uncertainty_scatters(scatter_dir: Path, output_dir: Path) -> None:
    """Create density scatter plots from saved validate.py uncertainty tuples."""
    output_dir.mkdir(parents=True, exist_ok=True)
    au_files = sorted(scatter_dir.rglob("*au_it_au_eu.pt"))
    for au_file in au_files:
        prefix = au_file.name.removesuffix("au_it_au_eu.pt")
        eu_file = au_file.with_name(f"{prefix}eu_it_au_eu.pt")
        if not eu_file.exists():
            continue

        au_laplace_au, au_laplace_eu = load_uncertainty_tuple(au_file)
        eu_laplace_au, eu_laplace_eu = load_uncertainty_tuple(eu_file)
        safe_prefix = prefix.strip("_").replace("/", "_") or au_file.parent.name

        pairs = {
            "au_laplace_au_vs_au_laplace_eu": (
                au_laplace_au,
                au_laplace_eu,
                "AU-Laplace AU vs AU-Laplace EU",
            ),
            "au_laplace_au_vs_eu_laplace_eu": (
                au_laplace_au,
                eu_laplace_eu,
                "AU-Laplace AU vs EU-Laplace EU",
            ),
            "eu_laplace_au_vs_au_laplace_eu": (
                eu_laplace_au,
                au_laplace_eu,
                "EU-Laplace AU vs AU-Laplace EU",
            ),
            "eu_laplace_au_vs_eu_laplace_eu": (
                eu_laplace_au,
                eu_laplace_eu,
                "EU-Laplace AU vs EU-Laplace EU",
            ),
        }

        for name, (aleatoric, epistemic, title) in pairs.items():
            plot_density_loglog_scatter(
                aleatoric,
                epistemic,
                title=f"{safe_prefix}: {title}",
                output_path=output_dir / f"{safe_prefix}_{name}.pdf",
            )


def setup_plot_style() -> None:
    """Sets up the plot style using tueplots and custom configurations."""
    config = bundles.neurips2024()
    config["figure.figsize"] = (10.0, 5.0)  # (2.64, 0.9)
    plt.rcParams.update(config)
    plt.rcParams["text.latex.preamble"] += (
        r"\usepackage{amsmath} \usepackage{amsfonts} \usepackage{bm}"
    )

class WandBThesisPlotter:
    """Fetch runs from WandB and generate thesis plots."""
    
    def __init__(
        self,
        project: str,
        entity: str = None,
        api_key: str = None,
        timeout: int = 60,
    ):
        """Initialize WandB connection."""
        if api_key:
            os.environ['WANDB_API_KEY'] = api_key
        os.environ.setdefault('WANDB_HTTP_TIMEOUT', str(timeout))
        
        self.api = wandb.Api(timeout=timeout)
        self.project = project
        self.entity = entity or 'your-entity'
        self.runs_data = {}
        
    def fetch_runs(
        self,
        filters: Dict = None,
        *,
        only_with_metrics: bool = True,
        run_name_filter: str = None,
        run_group_filter: str = None,
        per_page: int = 50,
    ) -> pd.DataFrame:
        """
        Fetch runs from WandB project.
        
        Args:
            filters: Dict of field:value to filter runs (e.g., {"state": "finished"})
        
        Returns:
            DataFrame with run data
        """
        project_path = f"{self.entity}/{self.project}"
        filters = dict(filters or {})
        if run_group_filter:
            filters["group"] = run_group_filter
        if only_with_metrics:
            metric_filter = [
                {f"summary_metrics.{key}": {"$exists": True}}
                for key in FILTER_SUMMARY_KEYS
            ]
            filters = {"$and": [filters, {"$or": metric_filter}]}

        try:
            runs = self.api.runs(project_path, filters=filters, per_page=per_page)
        except TypeError:
            runs = self.api.runs(project_path, filters=filters)
        
        data = []
        for run in runs:
            if run_name_filter and run_name_filter not in (run.name or ""):
                continue
            summary = dict(run.summary)
            run_dict = {
                'name': RUN_NAMES.get(run.name, run.name),
                'raw_name': run.name,
                'id': run.id,
                'state': run.state,
                'summary': summary,
            }
            data.append(run_dict)
        
        return pd.DataFrame(data)
    
    def extract_metrics(self, runs_df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Extract relevant metrics from runs.
        
        Returns dict: {run_name: {metric_name: value}}
        """
        extracted = {}
        
        for _, row in runs_df.iterrows():
            run_name = row['name']
            raw_name = row.get('raw_name', run_name)
            summary = row['summary']
            
            extracted[run_name] = {
                'experiment': infer_experiment(raw_name, run_name),
                'correlation_au_it_au_vs_eu': first_summary_float(
                    summary,
                    *SUMMARY_KEY_GROUPS['correlation_au_it_au_vs_eu'],
                ),
                'correlation_eu_it_au_vs_eu': first_summary_float(
                    summary,
                    *SUMMARY_KEY_GROUPS['correlation_eu_it_au_vs_eu'], 
                ),
                'correlation_au_it_au_vs_eu_it_eu': first_summary_float(
                    summary,
                    *SUMMARY_KEY_GROUPS['correlation_au_it_au_vs_eu_it_eu'],
                ),
                'correlation_eu_it_au_vs_au_it_eu': first_summary_float(
                    summary,
                    *SUMMARY_KEY_GROUPS['correlation_eu_it_au_vs_au_it_eu'],
                ),
                'rank_correlation_au_it_au_vs_eu': first_summary_float(
                    summary,
                    *SUMMARY_KEY_GROUPS['rank_correlation_au_it_au_vs_eu'],
                ),
                'rank_correlation_eu_it_au_vs_eu': first_summary_float(
                    summary,
                    *SUMMARY_KEY_GROUPS['rank_correlation_eu_it_au_vs_eu'],
                ),  
                'rank_correlation_au_it_au_vs_eu_it_eu': first_summary_float(
                    summary,
                    *SUMMARY_KEY_GROUPS['rank_correlation_au_it_au_vs_eu_it_eu'],
                ),
                'rank_correlation_eu_it_au_vs_au_it_eu': first_summary_float(
                    summary,
                    *SUMMARY_KEY_GROUPS['rank_correlation_eu_it_au_vs_au_it_eu'],
                ),
                'rank_correlation_it_au_eu': first_summary_float(
                    summary,
                    *SUMMARY_KEY_GROUPS['rank_correlation_it_au_eu'],
                ),
                'rank_correlation_epistemic_aleatoric': first_summary_float(
                    summary,
                    *SUMMARY_KEY_GROUPS['rank_correlation_epistemic_aleatoric'],
                ),
                'predictive_uncertainty': first_summary_float(
                    summary,
                    *SUMMARY_KEY_GROUPS['predictive_uncertainty'],
                ),
                'auroc_oodness': first_summary_float(
                    summary,
                    *SUMMARY_KEY_GROUPS['auroc_oodness'],
                ),
                'auroc_ood': first_summary_float(
                    summary,
                    *SUMMARY_KEY_GROUPS['auroc_ood'],
                ),
                'epistemic_auroc_oodness': first_summary_float(
                    summary,
                    *SUMMARY_KEY_GROUPS['epistemic_auroc_oodness'],
                ),
                'predictive_entropy_auroc_oodness': first_summary_float(
                    summary,
                    *SUMMARY_KEY_GROUPS['predictive_entropy_auroc_oodness'],
                ),
                'aleatoric_vs_gt': first_summary_float(
                    summary,
                    *SUMMARY_KEY_GROUPS['aleatoric_vs_gt'],
                ),
                'aleatoric_estimate_vs_gt': first_summary_float(
                    summary,
                    *SUMMARY_KEY_GROUPS['aleatoric_estimate_vs_gt'],
                ),
                'epistemic_uncertainty': first_summary_float(
                    summary,
                    *SUMMARY_KEY_GROUPS['epistemic_uncertainty'],
                ),
                'test_accuracy': first_summary_float(
                    summary,
                    *SUMMARY_KEY_GROUPS['test_accuracy'],
                ),
            }
        
        return extracted
    
    def plot_rank_correlation(self, metrics: Dict, output_path: str = None):
        """
        Plot rank correlation between epistemic and aleatoric uncertainty.
        Compares across model variants (CE baseline, Laplace variants).
        """
        fig, ax = plt.subplots()
        
        # Prepare data
        runs = []
        correlations = []
        
        for run_name, metric_dict in metrics.items():
            corr = to_float(metric_dict['rank_correlation_epistemic_aleatoric'])
            if np.isfinite(corr):
                runs.append(run_name)
                correlations.append(corr)
        
        # Sort for better visualization
        sorted_pairs = sorted(zip(runs, correlations), key=lambda x: x[1])
        if not sorted_pairs:
            draw_no_data(ax, 'Uncertainty Measure Correlation')
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
            return fig, ax
        runs, correlations = zip(*sorted_pairs)
        
        # Determine colors based on model type
        bar_colors = []
        for run_name in runs:
            if 'ce' in run_name.lower():
                bar_colors.append(COLORS['ce'])
            elif 'first' in run_name.lower():
                bar_colors.append(COLORS['laplace_first'])
            elif 'last' in run_name.lower():
                bar_colors.append(COLORS['laplace_last'])
            else:
                bar_colors.append(COLORS['laplace_full'])
        
        bars = ax.barh(range(len(runs)), correlations, color=bar_colors)
        
        ax.set_yticks(range(len(runs)))
        ax.set_yticklabels(runs, fontsize=9)
        ax.set_xlabel('Rank Correlation (Epistemic vs Aleatoric)', fontsize=10)
        ax.set_title('Uncertainty Measure Correlation', fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        fig.subplots_adjust(left=0.24, right=0.92, bottom=0.24, top=0.86)
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        return fig, ax
    
    def plot_predictive_uncertainty(self, metrics: Dict, output_path: str = None, Tuple = None):
        """
        Plot predictive uncertainty across model variants.
        """
        fig, ax = plt.subplots()
        
        runs = []
        uncertainties = []
        
        for run_name, metric_dict in metrics.items():
            unc = to_float(metric_dict['predictive_uncertainty'])
            if np.isfinite(unc):
                runs.append(run_name)
                uncertainties.append(unc)
        
        sorted_pairs = sorted(zip(runs, uncertainties), key=lambda x: x[1])
        if not sorted_pairs:
            draw_no_data(ax, 'Predictive Uncertainty Across Models')
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
            return fig, ax
        runs, uncertainties = zip(*sorted_pairs)
        
        bar_colors = []
        for run_name in runs:
            if 'ce' in run_name.lower():
                bar_colors.append(COLORS['ce'])
            elif 'first' in run_name.lower():
                bar_colors.append(COLORS['laplace_first'])
            elif 'last' in run_name.lower():
                bar_colors.append(COLORS['laplace_last'])
            else:
                bar_colors.append(COLORS['laplace_full'])
        
        bars = ax.barh(range(len(runs)), uncertainties, color=bar_colors)
        
        ax.set_yticks(range(len(runs)))
        ax.set_yticklabels(runs, fontsize=9)
        ax.set_xlabel('Predictive Uncertainty (bits)', fontsize=10)
        ax.set_title('Predictive Uncertainty Across Models', fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        return fig, ax
    
    def plot_auroc_ood(self, metrics: Dict, output_path: str = None):
        """
        Plot AUROC for OOD detection across model variants.
        """
        fig, ax = plt.subplots()
        
        runs = []
        aurocs = []
        
        for run_name, metric_dict in metrics.items():
            # Try both 'auroc_oddness' and 'auroc_ood'
            auroc = to_float(metric_dict.get('auroc_ood'))
            if np.isfinite(auroc):
                runs.append(run_name)
                aurocs.append(auroc)
        
        sorted_pairs = sorted(zip(runs, aurocs), key=lambda x: x[1], reverse=True)
        if not sorted_pairs:
            draw_no_data(ax, 'OOD Detection Performance (AUROC)')
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
            return fig, ax
        runs, aurocs = zip(*sorted_pairs)
        
        bar_colors = []
        for run_name in runs:
            if 'ce' in run_name.lower():
                bar_colors.append(COLORS['ce'])
            elif 'first' in run_name.lower():
                bar_colors.append(COLORS['laplace_first'])
            elif 'last' in run_name.lower():
                bar_colors.append(COLORS['laplace_last'])
            else:
                bar_colors.append(COLORS['laplace_full'])
        
        bars = ax.barh(range(len(runs)), aurocs, color=bar_colors)
        
        ax.set_yticks(range(len(runs)))
        ax.set_yticklabels(runs, fontsize=9)
        ax.set_xlabel('AUROC', fontsize=10)
        ax.set_xlim([0.5, 1.0])
        ax.set_title('OOD Detection Performance (AUROC)', fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, aurocs)):
            ax.text(val - 0.02, i, f'{val:.3f}', va='center', ha='right', 
                   fontsize=8, fontweight='bold')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        return fig, ax
    
    def plot_aleatoric_vs_ground_truth(self, metrics: Dict, output_path: str = None,):
        """
        Plot aleatoric uncertainty vs ground truth calibration.
        If you have individual sample data, this will be a scatter plot.
        If only summary metrics, shows comparison across runs.
        """
        fig, ax = plt.subplots()
        
        runs = []
        aleatoric_vals = []
        
        for run_name, metric_dict in metrics.items():
            alea = to_float(metric_dict['aleatoric_vs_gt'])
            if np.isfinite(alea):
                runs.append(run_name)
                aleatoric_vals.append(alea)
        
        sorted_pairs = sorted(zip(runs, aleatoric_vals), key=lambda x: x[1])
        if not sorted_pairs:
            draw_no_data(ax, 'Aleatoric Uncertainty Calibration')
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
            return fig, ax
        runs, aleatoric_vals = zip(*sorted_pairs)
        
        bar_colors = []
        for run_name in runs:
            if 'ce' in run_name.lower():
                bar_colors.append(COLORS['ce'])
            elif 'first' in run_name.lower():
                bar_colors.append(COLORS['laplace_first'])
            elif 'last' in run_name.lower():
                bar_colors.append(COLORS['laplace_last'])
            else:
                bar_colors.append(COLORS['laplace_full'])
        
        bars = ax.barh(range(len(runs)), aleatoric_vals, color=bar_colors)
        
        ax.set_yticks(range(len(runs)))
        ax.set_yticklabels(runs, fontsize=9)
        ax.set_xlabel('Aleatoric Uncertainty (correlation with GT)', fontsize=10)
        ax.set_title('Aleatoric Uncertainty Calibration', fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        return fig, ax

    def plot_metric_bar(
        self,
        metrics: Dict,
        metric_key: str,
        xlabel: str,
        title: str,
        output_path: str = None,
        *,
        descending: bool = True,
        xlim: Tuple[float, float] = None,
    ):
        """Plot one scalar W&B metric per run."""
        fig, ax = plt.subplots()
        runs = []
        values = []
        for run_name, metric_dict in metrics.items():
            value = to_float(metric_dict.get(metric_key))
            if np.isfinite(value):
                runs.append(run_name)
                values.append(value)

        sorted_pairs = sorted(
            zip(runs, values),
            key=lambda item: item[1],
            reverse=descending,
        )
        if not sorted_pairs:
            draw_no_data(ax, title)
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
            return fig, ax

        runs, values = zip(*sorted_pairs)
        bars = ax.barh(range(len(runs)), values, color=self._get_colors(runs))
        ax.set_yticks(range(len(runs)))
        ax.set_yticklabels(runs, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.grid(axis='x', alpha=0.3)
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(
                value,
                i,
                f' {value:.3f}',
                va='center',
                ha='left',
                fontsize=8,
            )
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        return fig, ax

    def plot_decomposition_matrix(
        self,
        metrics: Dict,
        metric_keys: Tuple[Tuple[str, str], Tuple[str, str]],
        title: str,
        output_path: str = None,
    ):
        """Plot a 2x2 matrix of IT/Bregman cross-estimator correlations."""
        fig, ax = plt.subplots()
        matrix = np.full((2, 2), np.nan)
        for row_idx, row_keys in enumerate(metric_keys):
            for col_idx, metric_key in enumerate(row_keys):
                values = [
                    to_float(metric_dict.get(metric_key))
                    for metric_dict in metrics.values()
                ]
                values = [value for value in values if np.isfinite(value)]
                if values:
                    matrix[row_idx, col_idx] = float(np.mean(values))

        if np.isfinite(matrix).any():
            image = ax.imshow(matrix, cmap='coolwarm', vmin=-1, vmax=1)
            cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            cbar.outline.set_visible(False)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['AU-Laplace EU', 'EU-Laplace EU'], rotation=25, ha='right')
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['AU-Laplace AU', 'EU-Laplace AU'])
            for row_idx in range(2):
                for col_idx in range(2):
                    value = matrix[row_idx, col_idx]
                    label = 'nan' if not np.isfinite(value) else f'{value:.2f}'
                    ax.text(col_idx, row_idx, label, ha='center', va='center', color='black')
            ax.set_title(title, fontsize=11, fontweight='bold')
        else:
            draw_no_data(ax, title)

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        return fig, ax

    def plot_spearman_scatter(
        self,
        metrics: Dict,
        title: str,
        output_path: str = None,
    ):
        """Plot Spearman rank correlations as points for each run."""
        fig, ax = plt.subplots()
        metric_labels = {
            'rank_correlation_au_it_au_vs_eu': 'AU-LA AU vs AU-LA EU',
            'rank_correlation_au_it_au_vs_eu_it_eu': 'AU-LA AU vs EU-LA EU',
            'rank_correlation_eu_it_au_vs_au_it_eu': 'EU-LA AU vs AU-LA EU',
            'rank_correlation_eu_it_au_vs_eu': 'EU-LA AU vs EU-LA EU',
        }
        offsets = np.linspace(-0.24, 0.24, len(metric_labels))
        colors = plt.get_cmap('tab10').colors
        run_names = list(metrics)
        has_values = False

        for offset, (metric_key, label), color in zip(
            offsets,
            metric_labels.items(),
            colors,
        ):
            x_values = []
            y_values = []
            for idx, run_name in enumerate(run_names):
                value = to_float(metrics[run_name].get(metric_key))
                if np.isfinite(value):
                    x_values.append(idx + offset)
                    y_values.append(value)
            if y_values:
                has_values = True
                ax.scatter(
                    x_values,
                    y_values,
                    s=42,
                    alpha=0.85,
                    color=color,
                    label=label,
                    zorder=3,
                )

        if not has_values:
            draw_no_data(ax, title)
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
            return fig, ax

        ax.axhline(0.0, color='0.35', linewidth=0.8, linestyle='--', zorder=1)
        ax.set_xticks(range(len(run_names)))
        ax.set_xticklabels(run_names, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('Spearman rank correlation', fontsize=10)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=7, ncol=2)

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        return fig, ax

    def plot_experiment_suite(self, metrics: Dict, output_dir: Path) -> None:
        """Generate all plots for each experiment group."""
        grouped_metrics = split_metrics_by_experiment(metrics)
        correlation_keys = (
            ('correlation_au_it_au_vs_eu', 'correlation_au_it_au_vs_eu_it_eu'),
            ('correlation_eu_it_au_vs_au_it_eu', 'correlation_eu_it_au_vs_eu'),
        )
        rank_correlation_keys = (
            ('rank_correlation_au_it_au_vs_eu', 'rank_correlation_au_it_au_vs_eu_it_eu'),
            ('rank_correlation_eu_it_au_vs_au_it_eu', 'rank_correlation_eu_it_au_vs_eu'),
        )

        for experiment, experiment_metrics in grouped_metrics.items():
            experiment_dir = output_dir / experiment
            experiment_dir.mkdir(parents=True, exist_ok=True)
            title_prefix = experiment.upper()

            self.plot_decomposition_matrix(
                experiment_metrics,
                correlation_keys,
                f'{title_prefix}: IT Correlation Matrix',
                experiment_dir / '01_it_correlation_matrix.pdf',
            )
            self.plot_decomposition_matrix(
                experiment_metrics,
                rank_correlation_keys,
                f'{title_prefix}: IT Rank-Correlation Matrix',
                experiment_dir / '02_it_rank_correlation_matrix.pdf',
            )
            self.plot_metric_bar(
                experiment_metrics,
                'predictive_entropy_auroc_oodness',
                'OOD AUROC',
                f'{title_prefix}: Predictive Entropy',
                experiment_dir / '03_predictive_entropy_ood_auroc.pdf',
                xlim=(0.5, 1.0),
            )
            self.plot_metric_bar(
                experiment_metrics,
                'epistemic_auroc_oodness',
                'OOD AUROC',
                f'{title_prefix}: Epistemic Estimate by Layer',
                experiment_dir / '04_epistemic_ood_auroc.pdf',
                xlim=(0.5, 1.0),
            )
            self.plot_metric_bar(
                experiment_metrics,
                'aleatoric_estimate_vs_gt',
                'Rank correlation with GT aleatoric uncertainty',
                f'{title_prefix}: Aleatoric Estimate vs Ground Truth',
                experiment_dir / '05_aleatoric_vs_gt.pdf',
                descending=True,
                xlim=(-1.0, 1.0),
            )
            self.plot_spearman_scatter(
                experiment_metrics,
                f'{title_prefix}: IT Spearman Correlations',
                experiment_dir / '06_it_spearman_scatter.pdf',
            )
    
    def plot_comparison_grid(self, metrics: Dict, output_path: str = None):
        """
        Create a 2x2 grid comparing all four key metrics.
        """
        fig, axes = plt.subplots(2, 2)
        
        # 1. Rank correlation
        ax = axes[0, 0]
        runs = []
        correlations = []
        for run_name, metric_dict in metrics.items():
            corr = to_float(metric_dict['rank_correlation_epistemic_aleatoric'])
            if np.isfinite(corr):
                runs.append(run_name)
                correlations.append(corr)
        
        sorted_pairs = sorted(zip(runs, correlations), key=lambda x: x[1])
        if sorted_pairs:
            runs, correlations = zip(*sorted_pairs)
            colors_list = self._get_colors(runs)
            ax.barh(range(len(runs)), correlations, color=colors_list)
            ax.set_yticks(range(len(runs)))
            ax.set_yticklabels(runs, fontsize=8)
            ax.set_xlabel('Rank Correlation', fontsize=9)
            ax.set_title('(a) Epistemic vs Aleatoric Correlation', fontsize=10, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
        else:
            draw_no_data(ax, '(a) Epistemic vs Aleatoric Correlation')
        
        # 2. Predictive uncertainty
        ax = axes[0, 1]
        runs = []
        uncertainties = []
        for run_name, metric_dict in metrics.items():
            unc = to_float(metric_dict['predictive_uncertainty'])
            if np.isfinite(unc):
                runs.append(run_name)
                uncertainties.append(unc)
        
        sorted_pairs = sorted(zip(runs, uncertainties), key=lambda x: x[1])
        if sorted_pairs:
            runs, uncertainties = zip(*sorted_pairs)
            colors_list = self._get_colors(runs)
            ax.barh(range(len(runs)), uncertainties, color=colors_list)
            ax.set_yticks(range(len(runs)))
            ax.set_yticklabels(runs, fontsize=8)
            ax.set_xlabel('Uncertainty (bits)', fontsize=9)
            ax.set_title('(b) Predictive Uncertainty', fontsize=10, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
        else:
            draw_no_data(ax, '(b) Predictive Uncertainty')
        
        # 3. AUROC OOD
        ax = axes[1, 0]
        runs = []
        aurocs = []
        for run_name, metric_dict in metrics.items():
            auroc = to_float(metric_dict.get('auroc_ood'))
            if np.isfinite(auroc):
                runs.append(run_name)
                aurocs.append(auroc)
        
        sorted_pairs = sorted(zip(runs, aurocs), key=lambda x: x[1], reverse=True)
        if sorted_pairs:
            runs, aurocs = zip(*sorted_pairs)
            colors_list = self._get_colors(runs)
            ax.barh(range(len(runs)), aurocs, color=colors_list)
            ax.set_yticks(range(len(runs)))
            ax.set_yticklabels(runs, fontsize=8)
            ax.set_xlabel('AUROC', fontsize=9)
            ax.set_xlim([0.5, 1.0])
            ax.set_title('(c) OOD Detection (AUROC)', fontsize=10, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
        else:
            draw_no_data(ax, '(c) OOD Detection (AUROC)')
        
        # 4. Aleatoric vs GT
        ax = axes[1, 1]
        runs = []
        aleatoric_vals = []
        for run_name, metric_dict in metrics.items():
            alea = to_float(metric_dict['aleatoric_vs_gt'])
            if np.isfinite(alea):
                runs.append(run_name)
                aleatoric_vals.append(alea)
        
        sorted_pairs = sorted(zip(runs, aleatoric_vals), key=lambda x: x[1])
        if sorted_pairs:
            runs, aleatoric_vals = zip(*sorted_pairs)
            colors_list = self._get_colors(runs)
            ax.barh(range(len(runs)), aleatoric_vals, color=colors_list)
            ax.set_yticks(range(len(runs)))
            ax.set_yticklabels(runs, fontsize=8)
            ax.set_xlabel('Correlation with GT', fontsize=9)
            ax.set_title('(d) Aleatoric Uncertainty Calibration', fontsize=10, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
        else:
            draw_no_data(ax, '(d) Aleatoric Uncertainty Calibration')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        return fig, axes
    
    def _get_colors(self, runs: List[str]) -> List[str]:
        """Get color list based on run names."""
        colors_list = []
        for run_name in runs:
            if 'ce' in run_name.lower():
                colors_list.append(COLORS['ce'])
            elif 'first' in run_name.lower():
                colors_list.append(COLORS['laplace_first'])
            elif 'last' in run_name.lower():
                colors_list.append(COLORS['laplace_last'])
            else:
                colors_list.append(COLORS['laplace_full'])
        return colors_list


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Generate thesis plots from W&B runs.")
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Use all finished runs in the configured W&B project.",
    )
    parser.add_argument(
        "--include-runs-without-metrics",
        action="store_true",
        help="Disable server-side filtering for relevant summary metrics.",
    )
    parser.add_argument(
        "--run-name-filter",
        default=None,
        help="Optional substring filter applied to W&B run names.",
    )
    parser.add_argument(
        "--run-group-filter",
        default=None,
        help="Optional exact W&B run group filter.",
    )
    parser.add_argument(
        "--wandb-timeout",
        type=int,
        default=60,
        help="W&B API timeout in seconds.",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=50,
        help="Number of W&B runs fetched per API page.",
    )
    parser.add_argument(
        "--scatter-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing validate.py saved tensors "
            "(*au_it_au_eu.pt and *eu_it_au_eu.pt) for log-log density scatters."
        ),
    )
    args = parser.parse_args()
    
    # Configuration
    PROJECT = "udl-thesis"
    ENTITY = "evelyn-emelanov-university-of-t-bingen"
    OUTPUT_DIR = Path("./thesis_plots")
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    setup_plot_style()

    # Initialize plotter
    plotter = WandBThesisPlotter(
        project=PROJECT,
        entity=ENTITY,
        timeout=args.wandb_timeout,
    )
    
    # Fetch runs (filter for completed runs). The current script uses all runs by
    # default; --all-runs is accepted for consistency with the untangle plot scripts.
    print("Fetching runs from WandB...")
    runs_df = plotter.fetch_runs(
        filters={"state": "finished"},
        only_with_metrics=not args.include_runs_without_metrics,
        run_name_filter=args.run_name_filter,
        run_group_filter=args.run_group_filter,
        per_page=args.per_page,
    )
    print(f"Found {len(runs_df)} completed runs.")
    
    # Extract metrics
    print("Extracting metrics...")
    metrics = plotter.extract_metrics(runs_df)
    
    # Generate plots
    print("Generating plots...")

    plotter.plot_experiment_suite(metrics, OUTPUT_DIR)
    print(f"✓ Saved experiment plots in: {OUTPUT_DIR}")

    if args.scatter_dir is not None:
        scatter_output_dir = OUTPUT_DIR / "density_scatters"
        plot_saved_uncertainty_scatters(args.scatter_dir, scatter_output_dir)
        print(f"✓ Saved density scatter plots in: {scatter_output_dir}")
    
    plotter.plot_rank_correlation(metrics, 
                                 output_path=OUTPUT_DIR / "01_rank_correlation.pdf")
    print(f"✓ Saved: {OUTPUT_DIR / '01_rank_correlation.pdf'}")
    
    plotter.plot_predictive_uncertainty(metrics,
                                       output_path=OUTPUT_DIR / "02_predictive_uncertainty.pdf")
    print(f"✓ Saved: {OUTPUT_DIR / '02_predictive_uncertainty.pdf'}")
    
    plotter.plot_auroc_ood(metrics,
                          output_path=OUTPUT_DIR / "03_auroc_ood.pdf")
    print(f"✓ Saved: {OUTPUT_DIR / '03_auroc_ood.pdf'}")
    
    plotter.plot_aleatoric_vs_ground_truth(metrics,
                                          output_path=OUTPUT_DIR / "04_aleatoric_vs_gt.pdf")
    print(f"✓ Saved: {OUTPUT_DIR / '04_aleatoric_vs_gt.pdf'}")
    
    plotter.plot_comparison_grid(metrics,
                                output_path=OUTPUT_DIR / "05_comparison_grid.pdf")
    print(f"✓ Saved: {OUTPUT_DIR / '05_comparison_grid.pdf'}")
    
    print("\n✓ All plots generated successfully!")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
