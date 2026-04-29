"""Tests for dual Laplace specialization and joint predictive wiring."""

import contextlib
from argparse import Namespace
from importlib import import_module
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

metric_module = import_module("untangle.utils.metric")
validate_module = import_module("untangle.validate")
adapted_laplace_module = import_module("untangle.wrappers.adapted_laplace_wrapper")
dual_laplace_module = import_module("untangle.wrappers.dual_laplace_wrapper")

entropy = metric_module.entropy
kl_divergence = metric_module.kl_divergence
convert_inference_dict = validate_module.convert_inference_dict
evaluate_on_correlation_of_estimators = (
    validate_module.evaluate_on_correlation_of_estimators
)
evaluate_on_proper_scoring_and_calibration = (
    validate_module.evaluate_on_proper_scoring_and_calibration
)
get_bundle = validate_module.get_bundle
AdaptedLaplaceWrapper = adapted_laplace_module.AdaptedLaplaceWrapper
DualLaplaceWrapper = dual_laplace_module.DualLaplaceWrapper


class _ToyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature = nn.Linear(2, 2, bias=False)
        self.classifier = nn.Linear(2, 2, bias=False)
        self.num_classes = 2

        with torch.no_grad():
            self.feature.weight.copy_(torch.tensor([[1.0, 0.0], [0.5, -1.0]]))
            self.classifier.weight.copy_(torch.tensor([[1.0, -0.5], [-0.25, 0.75]]))

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.feature(inputs)

        return self.classifier(hidden)


class _StaticLaplaceWrapper(nn.Module):
    def __init__(self, model: nn.Module, logits: Tensor) -> None:
        super().__init__()
        self.model = model
        self._logits = logits

    def forward(self, _: Tensor) -> dict[str, Tensor]:
        return {"logit": self._logits}


class _StaticPredictiveModel(nn.Module):
    def __init__(self, logits: Tensor) -> None:
        super().__init__()
        self.logits = logits
        self.num_classes = logits.shape[-1]

    def forward(self, inputs: Tensor) -> dict[str, Tensor]:
        batch_size = inputs.shape[0]

        return {"logit": self.logits[:batch_size]}


def _build_dual_wrapper(
    monkeypatch: pytest.MonkeyPatch,
    loaded_paths: list[Path] | None = None,
) -> DualLaplaceWrapper:
    def fake_load(
        self: AdaptedLaplaceWrapper,
        weight_path: Path,
        *,
        strict: bool = True,
    ) -> None:
        del self, strict

        if loaded_paths is not None:
            loaded_paths.append(weight_path)

    monkeypatch.setattr(AdaptedLaplaceWrapper, "_load_model", fake_load)

    return DualLaplaceWrapper(
        model=_ToyNet(),
        num_mc_samples=2,
        num_mc_samples_cv=2,
        weight_path=Path("weights.pt"),
    )


def _summarize(logits: Tensor) -> dict[str, Tensor]:
    if logits.dim() == 2:
        logits = logits.unsqueeze(dim=1)

    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    bma = probs.mean(dim=1)
    log_dual_bma = F.log_softmax(log_probs.mean(dim=1), dim=-1)
    expected_entropy = entropy(probs).mean(dim=-1)

    return {
        "expected_entropy": expected_entropy,
        "expected_divergence": kl_divergence(
            log_dual_bma, log_probs.permute(1, 0, 2)
        ).mean(dim=0),
        "jensen_shannon_divergence": entropy(bma) - expected_entropy,
    }


def test_dual_laplace_wrapper_uses_weight_path_and_builds_joint_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_paths = []
    wrapper = _build_dual_wrapper(monkeypatch, loaded_paths)
    inputs = torch.tensor([[1.0, -0.5]], dtype=torch.float32)

    with torch.no_grad():
        base_logits = wrapper.epistemic_wrapper.model(inputs)

    epistemic_logits = base_logits.unsqueeze(1) + torch.tensor(
        [[[0.4, -0.2], [0.1, 0.3]]], dtype=torch.float32
    )
    aleatoric_logits = base_logits.unsqueeze(1) + torch.tensor(
        [[[-0.3, 0.5], [0.2, -0.1]]], dtype=torch.float32
    )
    wrapper.epistemic_wrapper = _StaticLaplaceWrapper(
        wrapper.epistemic_wrapper.model, epistemic_logits
    )
    wrapper.aleatoric_wrapper = _StaticLaplaceWrapper(
        wrapper.aleatoric_wrapper.model, aleatoric_logits
    )

    outputs = wrapper(inputs)
    expected_joint_logits = (
        base_logits.unsqueeze(1)
        + (epistemic_logits - base_logits.unsqueeze(1))
        + (aleatoric_logits - base_logits.unsqueeze(1))
    )

    assert loaded_paths == [Path("weights.pt"), Path("weights.pt")]
    assert torch.allclose(outputs["logit"], expected_joint_logits)
    assert torch.allclose(outputs["epistemic_logit"], epistemic_logits)
    assert torch.allclose(outputs["aleatoric_logit"], aleatoric_logits)


@pytest.mark.parametrize(
    ("epistemic_regex", "aleatoric_regex", "message"),
    [
        (None, "^classifier", "epistemic_params_regex must be provided"),
        ("^missing", "^classifier", "did not match any model parameters"),
        ("^feature", "^feature", "must select disjoint parameter subsets"),
    ],
)
def test_dual_laplace_wrapper_validates_parameter_subsets(
    monkeypatch: pytest.MonkeyPatch,
    epistemic_regex: str | None,
    aleatoric_regex: str | None,
    message: str,
) -> None:
    wrapper = _build_dual_wrapper(monkeypatch)

    with pytest.raises(ValueError, match=message):
        wrapper.perform_dual_laplace_approximation(
            train_loader=None,
            val_loader=None,
            approx_method="kfac",
            rank=2,
            epistemic_params_regex=epistemic_regex,
            aleatoric_params_regex=aleatoric_regex,
        )


def test_dual_laplace_wrapper_enables_expected_parameter_subsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _build_dual_wrapper(monkeypatch)
    selected_subsets = []

    def fake_perform(
        self: AdaptedLaplaceWrapper,
        train_loader: object,
        val_loader: object,
        approx_method: str,
        rank: int,
    ) -> None:
        del train_loader, val_loader, approx_method, rank
        enabled_parameters = [
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        selected_subsets.append(enabled_parameters)

    monkeypatch.setattr(
        AdaptedLaplaceWrapper,
        "adapted_perform_laplace_approximation",
        fake_perform,
    )
    wrapper.perform_dual_laplace_approximation(
        train_loader=None,
        val_loader=None,
        approx_method="low_rank",
        rank=3,
        epistemic_params_regex="^feature",
        aleatoric_params_regex="^classifier",
    )

    assert selected_subsets == [["feature.weight"], ["classifier.weight"]]


def test_adapted_laplace_forward_drops_prediction_graphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_load(
        self: AdaptedLaplaceWrapper,
        weight_path: Path,
        *,
        strict: bool = True,
    ) -> None:
        del self, weight_path, strict

    monkeypatch.setattr(AdaptedLaplaceWrapper, "_load_model", fake_load)
    wrapper = AdaptedLaplaceWrapper(
        model=_ToyNet(),
        num_mc_samples=2,
        num_mc_samples_cv=2,
        weight_path=Path("weights.pt"),
    )
    dim = sum(p.numel() for p in wrapper.model.parameters() if p.requires_grad)

    def fake_create_ggn_linear_operator(**_: object) -> object:
        return object()

    def fake_compute_low_rank_eigendecomposition(**_: object) -> tuple[Tensor, Tensor]:
        return torch.ones(1), torch.ones(dim, 1)

    def fake_optimize(
        self: AdaptedLaplaceWrapper,
        val_loader: object,
    ) -> None:
        del val_loader
        self.prior_precision = 1.0

    monkeypatch.setattr(
        AdaptedLaplaceWrapper,
        "_optimize_prior_precision_cv",
        fake_optimize,
    )
    monkeypatch.setattr(
        adapted_laplace_module,
        "create_ggn_linear_operator",
        fake_create_ggn_linear_operator,
    )
    monkeypatch.setattr(
        adapted_laplace_module,
        "compute_low_rank_eigendecomposition",
        fake_compute_low_rank_eigendecomposition,
    )
    wrapper.adapted_perform_laplace_approximation(
        train_loader=None,
        val_loader=None,
        approx_method="low-rank",
        rank=1,
    )
    wrapper.eval()

    with torch.enable_grad():
        outputs = wrapper(torch.randn(2, 2))

    assert not outputs["logit"].requires_grad


def test_convert_inference_dict_uses_joint_logits_and_branch_specialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _build_dual_wrapper(monkeypatch)
    joint_logits = torch.tensor(
        [[[2.0, 0.0], [0.0, 2.0]], [[1.5, -0.5], [1.0, 0.0]]],
        dtype=torch.float32,
    )
    epistemic_logits = torch.tensor(
        [[[3.0, 0.0], [0.0, 3.0]], [[1.2, 0.3], [0.5, 0.8]]],
        dtype=torch.float32,
    )
    aleatoric_logits = torch.tensor(
        [[[1.5, 0.5], [0.5, 1.5]], [[1.0, 0.2], [0.7, 0.6]]],
        dtype=torch.float32,
    )
    converted = convert_inference_dict(
        model=wrapper,
        inference_dict={
            "logit": joint_logits,
            "epistemic_logit": epistemic_logits,
            "aleatoric_logit": aleatoric_logits,
        },
        time_forward=0.25,
        args=Namespace(num_mc_samples=2),
    )
    joint_summary = _summarize(joint_logits)
    epistemic_summary = _summarize(epistemic_logits)
    aleatoric_summary = _summarize(aleatoric_logits)

    assert converted["time_forward"] == 0.25
    assert torch.allclose(
        converted["expected_entropy"], joint_summary["expected_entropy"]
    )
    assert torch.allclose(
        converted["expected_divergence"], joint_summary["expected_divergence"]
    )
    assert torch.allclose(
        converted["eu_bregman_eu"], epistemic_summary["expected_divergence"]
    )
    assert torch.allclose(
        converted["au_bregman_au"], aleatoric_summary["expected_entropy"]
    )
    assert torch.allclose(
        converted["eu_it_eu"], epistemic_summary["jensen_shannon_divergence"]
    )
    assert torch.allclose(
        converted["au_it_eu"], aleatoric_summary["jensen_shannon_divergence"]
    )


def test_evaluate_on_correlation_of_estimators_supports_dual_laplace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    wrapper = _build_dual_wrapper(monkeypatch)
    estimates = {
        "au_bregman_au": torch.tensor([0.1, 0.2]),
        "au_bregman_eu": torch.tensor([0.2, 0.3]),
        "eu_bregman_au": torch.tensor([0.3, 0.4]),
        "eu_bregman_eu": torch.tensor([0.4, 0.5]),
        "au_it_au": torch.tensor([0.5, 0.6]),
        "au_it_eu": torch.tensor([0.6, 0.7]),
        "eu_it_au": torch.tensor([0.7, 0.8]),
        "eu_it_eu": torch.tensor([0.8, 0.9]),
        "expected_entropies": torch.tensor([0.2, 0.4]),
        "expected_variances_of_probs": torch.tensor([0.3, 0.5]),
        "expected_variances_of_logits": torch.tensor([0.4, 0.6]),
        "jensen_shannon_divergences": torch.tensor([0.5, 0.7]),
        "entropies_of_bma": torch.tensor([0.6, 0.8]),
        "expected_divergences": torch.tensor([0.7, 0.9]),
        "expected_entropies_plus_expected_divergences": torch.tensor([0.9, 1.3]),
    }
    metrics = evaluate_on_correlation_of_estimators(
        model=wrapper,
        estimates=estimates,
        output_dir=tmp_path,
        save_prefix="dual_",
        args=Namespace(dataset_id="hard/imagenet"),
        is_soft_upstream_dataset=None,
    )

    assert "correlation_au_bregman_au_vs_eu" in metrics
    assert (tmp_path / "dual_au_bregman_au_eu.pt").exists()


def test_proper_scoring_uses_predictive_metric_names() -> None:
    metrics = evaluate_on_proper_scoring_and_calibration(
        estimates={
            "one_minus_expected_max_probs": torch.tensor([0.2, 0.3]),
            "one_minus_max_probs_of_dual_bma": torch.tensor([0.1, 0.4]),
            "one_minus_max_probs_of_bma": torch.tensor([0.25, 0.35]),
        },
        log_probs={
            "log_dual_bmas": torch.log(
                torch.tensor([[0.8, 0.2], [0.3, 0.7]], dtype=torch.float32)
            ),
            "log_bmas": torch.log(
                torch.tensor([[0.75, 0.25], [0.4, 0.6]], dtype=torch.float32)
            ),
        },
        targets={
            "gt_hard_dual_bma_correctnesses_original": torch.tensor([1, 0]),
            "gt_hard_dual_bma_correctnesses": torch.tensor([1, 0]),
            "gt_hard_bma_correctnesses_original": torch.tensor([1, 1]),
            "gt_hard_bma_correctnesses": torch.tensor([1, 1]),
            "gt_hard_dual_bma_correctnesses_original_top5": torch.tensor([1, 1]),
            "gt_hard_dual_bma_correctnesses_top5": torch.tensor([1, 1]),
            "gt_hard_bma_correctnesses_original_top5": torch.tensor([1, 1]),
            "gt_hard_bma_correctnesses_top5": torch.tensor([1, 1]),
            "gt_hard_labels_original": torch.tensor([0, 1]),
            "gt_hard_labels": torch.tensor([0, 1]),
        },
        is_soft_dataset=False,
        args=Namespace(dataset_id="hard/imagenet"),
        is_soft_upstream_dataset=None,
    )

    assert (
        "one_minus_max_probs_of_dual_bma_log_prob_score_hard_dual_bma_correctness_original"
        in metrics
    )
    assert "one_minus_max_probs_of_bma_brier_score_hard_bma_correctness" in metrics
    assert "log_prob_score_hard_dual_bma_aleatoric_original" in metrics
    assert "brier_score_hard_bma_aleatoric" in metrics

    assert "log_prob_score_hard_dual_bma_predictive_original" not in metrics
    assert "brier_score_hard_bma_predictive" not in metrics


def test_get_bundle_populates_soft_aleatoric_targets() -> None:
    logits = torch.tensor(
        [[2.0, 0.0, -1.0, -2.0, -3.0], [0.0, 2.0, -1.0, -2.0, -3.0]],
        dtype=torch.float32,
    )
    model = _StaticPredictiveModel(logits)
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    labels = torch.tensor(
        [
            [0.5, 0.2, 0.1, 0.1, 0.1, 0.0],
            [0.1, 0.6, 0.1, 0.1, 0.1, 1.0],
        ],
        dtype=torch.float32,
    )
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=2)

    _, _, targets, _ = get_bundle(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
        storage_device=torch.device("cpu"),
        amp_autocast=contextlib.nullcontext,
        is_soft_dataset=True,
        args=Namespace(prefetcher=False, channels_last=False, num_mc_samples=1),
    )
    probs = labels[:, :-1]
    expected_aleatoric = entropy(probs)

    assert torch.allclose(targets["gt_aleatorics_bregman"], expected_aleatoric)
