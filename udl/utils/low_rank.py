"""Low-rank GGN computation utilities."""

import logging
import math
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigsh
from skerch.algorithms import seigh
from torch import Tensor, nn
from torch.utils.data import DataLoader

from bul.utils.decompositions import select_skerch_measurements
from bul.utils.linear_operators import GGNLinearOperator, MatmulMixin

logger = logging.getLogger(__name__)


def make_scipy_linear_operator(
    op: MatmulMixin,
    *,
    scipy_dtype: np.dtype,
) -> LinearOperator:
    def _to_numpy(tensor: Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy().astype(scipy_dtype)

    def _from_numpy(array: np.ndarray) -> Tensor:
        np_array = np.asarray(array, dtype=scipy_dtype)
        tensor = torch.from_numpy(np_array)
        return tensor.to(device=op.device, dtype=op.dtype)

    def matvec(vec: np.ndarray) -> np.ndarray:
        vec_t = _from_numpy(vec)
        result_t = op.matmat(vec_t)
        return _to_numpy(result_t)

    def matmat(matrix: np.ndarray) -> np.ndarray:
        mat_t = _from_numpy(matrix)
        result_t = op.matmat(mat_t)
        return _to_numpy(result_t)

    return LinearOperator(
        shape=op.shape, matvec=matvec, matmat=matmat, dtype=scipy_dtype
    )


class LowRankMatrixOperator(MatmulMixin):
    """Low-rank operator computing ``U diag(lambda) U^T``."""

    def __init__(self, eigvals: Tensor, eigvecs: Tensor) -> None:
        self._eigvecs = eigvecs
        self._eigvals = eigvals
        self.shape = (eigvecs.shape[0], eigvecs.shape[0])
        self.device = eigvecs.device
        self.dtype = eigvecs.dtype

    def matmat(self, X: Tensor) -> Tensor:
        squeeze = X.ndim == 1

        if squeeze:
            X_mat = X.unsqueeze(-1)
        elif X.ndim == 2:
            X_mat = X
        else:
            msg = "Input must be 1D or 2D"
            raise ValueError(msg)

        projection = self._eigvecs.T @ X_mat
        scaled = self._eigvals.unsqueeze(-1) * projection
        result = self._eigvecs @ scaled

        if squeeze:
            return result.squeeze(-1)

        return result

    def rmatmat(self, X: Tensor) -> Tensor:
        if X.ndim == 1:
            return self.matmat(X)

        if X.ndim != 2:
            msg = "Input must be 1D or 2D"
            raise ValueError(msg)

        return self.matmat(X.T).T


class DiagonalPlusLowRankOperator(LowRankMatrixOperator):
    """Operator computing ``alpha I + U diag(lambda) U^T``."""

    def __init__(self, eigvals: Tensor, eigvecs: Tensor, diag_scale: float) -> None:
        super().__init__(eigvals=eigvals, eigvecs=eigvecs)
        self._diag_scale = diag_scale

    def matmat(self, X: Tensor) -> Tensor:
        squeeze = X.ndim == 1

        if squeeze:
            X_mat = X.unsqueeze(-1)
        elif X.ndim == 2:
            X_mat = X
        else:
            msg = "Input must be 1D or 2D"
            raise ValueError(msg)

        low_rank = super().matmat(X_mat)
        result = low_rank + self._diag_scale * X_mat

        if squeeze:
            return result.squeeze(-1)

        return result


def create_ggn_linear_operator(
    model: nn.Module,
    loss: Literal["mse", "ce"],
    reduction: Literal["sum", "mean"],
    loader: DataLoader,
) -> GGNLinearOperator:
    """Create a torch-native GGN linear operator for a dataset.

    Args:
        model: Network used to evaluate gradients.
        loss: Loss function to use.
        reduction: Aggregation applied to the per-sample gradients.
        loader: Data loader that streams the dataset to evaluate.

    Returns:
        GGN linear operator built on top of the provided model and loader.
    """
    operator = GGNLinearOperator(
        model=model,
        loader=loader,
        loss=loss,
        reduction=reduction,
        progressbar=True,
    )

    return operator


def compute_low_rank_eigendecomposition(
    op: MatmulMixin,
    rank: int,
    *,
    backend: Literal["arpack", "skerch"] = "arpack",
    arpack_kwargs: dict[str, Any],
    skerch_kwargs: dict[str, Any],
) -> tuple[Tensor, Tensor]:
    """Compute top-``rank`` eigenpairs using ARPACK or Skerch.

    Args:
        op: Linear operator that exposes ``matmat`` and ``shape``.
        rank: Number of leading eigenpairs to return.
        backend: Solver backend, either ``"arpack"`` or ``"skerch"``.
        arpack_kwargs: Keyword arguments forwarded to the ARPACK solver.
        skerch_kwargs: Keyword arguments forwarded to the Skerch solver.

    Returns:
        Eigenvalues and eigenvectors ordered by magnitude.

    Raises:
        ValueError: If ``op`` is not square.
    """
    num_rows, num_cols = op.shape

    if num_rows != num_cols:
        msg = "Operator must be square."
        raise ValueError(msg)

    if backend == "arpack":
        curv_kwargs = arpack_kwargs["curv"].copy()

        scipy_op = make_scipy_linear_operator(
            op,
            scipy_dtype=curv_kwargs.pop("scipy_dtype"),
        )

        curv_kwargs |= {
            "A": scipy_op,
            "k": rank,
            "which": "LM",
            "return_eigenvectors": True,
        }

        eigvals_np, eigvecs_np = eigsh(**curv_kwargs)
        eigvals = torch.from_numpy(eigvals_np).to(device=op.device, dtype=op.dtype)
        eigvecs = torch.from_numpy(eigvecs_np).to(device=op.device, dtype=op.dtype)

        return eigvals, eigvecs

    clamped_kwargs = select_skerch_measurements(
        total_dim=num_rows,
        section="curv",
        skerch_kwargs=skerch_kwargs,
    )
    vals, vecs = seigh(
        lop=op,
        lop_device=op.device,
        lop_dtype=op.dtype,
        **clamped_kwargs,
    )

    vals = vals[:rank].clamp(min=0.0)
    vecs = vecs[:, :rank]

    return vals, vecs


def clamp_eigenvalues(
    eigenvalues: Tensor,
    min_value: float = 0.0,
) -> Tensor:
    """Clamp eigenvalues to ensure positive semi-definiteness.

    Args:
        eigenvalues: Eigenvalues to clamp.
        min_value: Minimum allowed eigenvalue.

    Returns:
        Eigenvalues with entries below ``min_value`` replaced.
    """
    logger.info(
        "Clamping eigenvalues (%s and above) to %s...",
        f"{eigenvalues.min().item():.2e}",
        min_value,
    )
    return torch.clamp(eigenvalues, min=min_value)


def setup_ggn_operators(
    models: nn.Module | list[nn.Module],
    loaders: list[DataLoader],
    loss: Literal["mse", "ce"],
    reduction: Literal["sum", "mean"],
) -> list[GGNLinearOperator]:
    """Set up GGN linear operators for training and forgotten sets.

    Args:
        models: Model or list of models aligned with ``loaders``.
        loaders: Data loaders that produce batches for each model.
        loss: Loss function to use.
        reduction: Aggregation used inside the operator.

    Returns:
        Collection of GGN operators matching the order of ``loaders``.
    """
    if isinstance(models, nn.Module):
        models = len(loaders) * [models]  # Use the same model for all loaders

    ggn_ops = []
    for model, loader in zip(models, loaders, strict=True):
        ggn_op = create_ggn_linear_operator(
            model=model,
            loss=loss,
            reduction=reduction,
            loader=loader,
        )
        ggn_ops.append(ggn_op)

    return ggn_ops


def compute_ggn_eigendecomposition_with_checkpointing(
    op: MatmulMixin,
    rank: int,
    checkpoint_loader_fn: Callable[
        [Callable[[], tuple[Tensor, Tensor]]], tuple[Tensor, Tensor]
    ],
    *,
    backend: Literal["arpack", "skerch"] = "arpack",
    arpack_kwargs: dict[str, Any],
    skerch_kwargs: dict[str, Any],
) -> tuple[Tensor, Tensor]:
    """Compute GGN eigendecomposition with checkpointing support.

    Args:
        op: Linear operator to decompose.
        rank: Number of eigenpairs to compute.
        checkpoint_loader_fn: Callable that materialises cached eigenpairs on demand.
        backend: Solver backend, either ``"arpack"`` or ``"skerch"``.
        arpack_kwargs: Keyword arguments forwarded to the ARPACK solver.
        skerch_kwargs: Keyword arguments forwarded to the Skerch solver.

    Returns:
        Eigenvalues and eigenvectors obtained after checkpointing.
    """

    def compute_fn() -> tuple[Tensor, Tensor]:
        return compute_low_rank_eigendecomposition(
            op=op,
            rank=rank,
            backend=backend,
            arpack_kwargs=arpack_kwargs,
            skerch_kwargs=skerch_kwargs,
        )

    eigenvalues, eigenvectors = checkpoint_loader_fn(compute_fn)
    eigenvalues = clamp_eigenvalues(eigenvalues)

    return eigenvalues, eigenvectors


def get_low_rank_ggn_op(
    low_rank_eigvals: Tensor, low_rank_eigvecs: Tensor
) -> LowRankMatrixOperator:
    """Return the low-rank operator ``U diag(lambda) U^T``."""
    return LowRankMatrixOperator(eigvals=low_rank_eigvals, eigvecs=low_rank_eigvecs)


def get_low_rank_cov_op(
    ggn_eigvals: Tensor, ggn_eigvecs: Tensor, prior_precision: float
) -> DiagonalPlusLowRankOperator:
    """Return the covariance operator ``(G + lambda I)^{-1}`` in low-rank form."""
    cov_sigma_square = 1 / prior_precision
    cov_eigvals = -ggn_eigvals / (prior_precision * (ggn_eigvals + prior_precision))
    return DiagonalPlusLowRankOperator(
        eigvals=cov_eigvals,
        eigvecs=ggn_eigvecs,
        diag_scale=cov_sigma_square,
    )


def get_low_rank_scale_op(
    ggn_eigvals: Tensor, ggn_eigvecs: Tensor, prior_precision: float
) -> DiagonalPlusLowRankOperator:
    """Return the scale operator ``(G + lambda I)^{-1/2}`` in low-rank form."""
    cov_sigma = 1 / math.sqrt(prior_precision)
    scale_eigvals = 1 / (ggn_eigvals + prior_precision).sqrt() - cov_sigma
    return DiagonalPlusLowRankOperator(
        eigvals=scale_eigvals,
        eigvecs=ggn_eigvecs,
        diag_scale=cov_sigma,
    )


def get_low_rank_inv_scale_op(
    ggn_eigvals: Tensor, ggn_eigvecs: Tensor, prior_precision: float
) -> DiagonalPlusLowRankOperator:
    """Return the inverse-scale operator ``(G + lambda I)^{1/2}`` in low-rank form."""
    precision_sigma = math.sqrt(prior_precision)
    inv_scale_eigvals = (ggn_eigvals + prior_precision).sqrt() - precision_sigma
    return DiagonalPlusLowRankOperator(
        eigvals=inv_scale_eigvals,
        eigvecs=ggn_eigvecs,
        diag_scale=precision_sigma,
    )


def get_low_rank_precision_op(
    ggn_eigvals: Tensor, ggn_eigvecs: Tensor, prior_precision: float
) -> DiagonalPlusLowRankOperator:
    """Return the precision operator ``G + lambda I``."""
    return DiagonalPlusLowRankOperator(
        eigvals=ggn_eigvals,
        eigvecs=ggn_eigvecs,
        diag_scale=prior_precision,
    )


class SumOperator(MatmulMixin):
    """Linear operator for A + B."""

    def __init__(self, op_a: MatmulMixin, op_b: MatmulMixin) -> None:
        if op_a.shape != op_b.shape:
            msg = f"Operator shapes differ: {op_a.shape} vs {op_b.shape}"
            raise ValueError(msg)

        self._a = op_a
        self._b = op_b
        self.shape = op_a.shape
        self.device = op_a.device
        self.dtype = op_a.dtype

    def matmat(self, X: Tensor) -> Tensor:
        return self._a.matmat(X) + self._b.matmat(X)

    def rmatmat(self, X: Tensor) -> Tensor:
        if X.ndim == 1:
            return self.matmat(X)

        if X.ndim != 2:
            msg = "Input must be 1D or 2D"
            raise ValueError(msg)

        return self.matmat(X.T).T


class DifferenceOperator(MatmulMixin):
    """Linear operator for A - B."""

    def __init__(self, op_a: MatmulMixin, op_b: MatmulMixin) -> None:
        if op_a.shape != op_b.shape:
            msg = f"Operator shapes differ: {op_a.shape} vs {op_b.shape}"
            raise ValueError(msg)

        self._a = op_a
        self._b = op_b
        self.shape = op_a.shape
        self.device = op_a.device
        self.dtype = op_a.dtype

    def matmat(self, X: Tensor) -> Tensor:
        return self._a.matmat(X) - self._b.matmat(X)

    def rmatmat(self, X: Tensor) -> Tensor:
        if X.ndim == 1:
            return self.matmat(X)

        if X.ndim != 2:
            msg = "Input must be 1D or 2D"
            raise ValueError(msg)

        return self.matmat(X.T).T


def get_approximate_retained_set_ggn_op(
    training_set_ggn_eigvals: Tensor,
    training_set_ggn_eigvecs: Tensor,
    forgotten_set_ggn_op: MatmulMixin,
) -> MatmulMixin:
    """Torch-native approximate retained-set GGN operator.

    Computes G_retained approx (U Lambda U^T) - G_forgotten, where (U, Lambda) are the
    top eigenpairs of G_training. Returned operator implements MatmulMixin so it
    can be fed to either ARPACK or Skerch backends.

    Returns:
        The approximate retained set GGN linear operator.
    """
    low_rank_training = get_low_rank_ggn_op(
        low_rank_eigvals=training_set_ggn_eigvals,
        low_rank_eigvecs=training_set_ggn_eigvecs,
    )
    return DifferenceOperator(low_rank_training, forgotten_set_ggn_op)


def get_approximate_learned_set_ggn_op(
    training_set_ggn_eigvals: Tensor,
    training_set_ggn_eigvecs: Tensor,
    forgotten_set_ggn_op: MatmulMixin,
    learned_set_ggn_op: MatmulMixin,
) -> MatmulMixin:
    """Approximate retained-plus-learned GGN operator.

    Returns the operator representing ``(U diag(lambda) U^T - G_forgotten) + G_learned``
    using the available low-rank training decomposition.

    Returns:
        The approximate learned set GGN linear operator.
    """
    retained_op = get_approximate_retained_set_ggn_op(
        training_set_ggn_eigvals=training_set_ggn_eigvals,
        training_set_ggn_eigvecs=training_set_ggn_eigvecs,
        forgotten_set_ggn_op=forgotten_set_ggn_op,
    )
    return SumOperator(retained_op, learned_set_ggn_op)


def log_eigenvalue_statistics(
    eigenvalues: Tensor,
    name: str,
) -> None:
    """Log statistics about eigenvalues."""
    logger.info("%s eigenvalue statistics:", name)
    logger.info("  Min: %.2e", eigenvalues.min().item())
    logger.info("  Max: %.2e", eigenvalues.max().item())
    logger.info("  Mean: %.2e", eigenvalues.mean().item())
    logger.info("  Median: %.2e", eigenvalues.median().item())

    negative_count = (eigenvalues < 0).sum().item()
    total_count = len(eigenvalues)
    logger.info("  Negative eigenvalues: %d/%d", negative_count, total_count)

    if negative_count > 0:
        logger.warning("  Most negative: %.2e", eigenvalues.min().item())
