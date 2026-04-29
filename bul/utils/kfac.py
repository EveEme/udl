"""KFAC computation utilities."""

import logging
import warnings
from typing import Literal

import torch
import torch.nn.functional as F
from backpack import backpack, extend
from backpack.core.derivatives.batchnorm_nd import BatchNormNdDerivatives
from backpack.extensions import KFAC, KFLR
from backpack.extensions.mat_to_mat_jac_base import MatToJacMat
from torch import Tensor, nn
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d
from torch.utils.data import DataLoader

from bul.utils.linear_operators import MatmulMixin

warnings.filterwarnings(
    "ignore",
    message=(
        r"Full backward hook is firing when gradients are computed with respect.*"
    ),
    category=UserWarning,
)

logger = logging.getLogger(__name__)


class SummingKFLR(KFLR):
    """KFLR variant that sums branch contributions."""

    @staticmethod
    def accumulate_backpropagated_quantities(existing: Tensor, other: Tensor) -> Tensor:
        return existing + other


class KFACBatchNormNd(MatToJacMat):
    """KFAC extension for BatchNorm1d/2d/3d in evaluation mode.

    Computes KFAC blocks for BN parameters using MC-sampled backprop vectors.
    Follows the tutorial semantics: keep per-sample contributions (no batch
    aggregation inside J^T v), then form the Gram over all samples.
    """

    def __init__(self) -> None:
        super().__init__(
            derivatives=BatchNormNdDerivatives(), params=["weight", "bias"]
        )

    def weight(
        self,
        ext,
        module: nn.Module,
        g_inp: tuple[Tensor, ...],
        g_out: tuple[Tensor, ...],
        backproped: Tensor,
    ) -> list[Tensor]:
        del ext

        if module.training:
            msg = (
                "KFAC for BatchNorm is only supported in evaluation mode. "
                "Call module.eval() before computing KFAC."
            )
            raise RuntimeError(msg)

        v = backproped
        # KFAC (MC): keep per-sample contributions -> [V, N, C]
        JTv = self.derivatives.param_mjp(
            "weight", module, g_inp, g_out, v, sum_batch=False
        )
        # Flatten [V, N, C] -> [V*N, C]
        JTv = JTv.reshape(-1, module.num_features)

        kfac_gamma = JTv.T @ JTv
        return [kfac_gamma]

    def bias(
        self,
        ext,
        module: nn.Module,
        g_inp: tuple[Tensor, ...],
        g_out: tuple[Tensor, ...],
        backproped: Tensor,
    ) -> list[Tensor]:
        del ext

        if module.training:
            msg = (
                "KFAC for BatchNorm is only supported in evaluation mode. "
                "Call module.eval() before computing KFAC."
            )
            raise RuntimeError(msg)

        v = backproped
        # KFAC (MC): keep per-sample contributions -> [V, N, C]
        JTv = self.derivatives.param_mjp(
            "bias", module, g_inp, g_out, v, sum_batch=False
        )
        # Flatten [V, N, C] -> [V*N, C]
        JTv = JTv.reshape(-1, module.num_features)

        kfac_beta = JTv.T @ JTv
        return [kfac_beta]


class KFLRBatchNormNd(MatToJacMat):
    """KFLR extension for BatchNorm1d/2d/3d in evaluation mode.

    Computes exact (per-sample) BN blocks for KFLR: keeps the batch dimension
    in J^T v (no aggregation) and forms the Gram over [V * N, C].
    """

    def __init__(self) -> None:
        super().__init__(
            derivatives=BatchNormNdDerivatives(), params=["weight", "bias"]
        )

    def weight(
        self,
        ext,
        module: nn.Module,
        g_inp: tuple[Tensor, ...],
        g_out: tuple[Tensor, ...],
        backproped: Tensor,
    ) -> list[Tensor]:
        del ext

        if module.training:
            msg = (
                "KFAC for BatchNorm is only supported in evaluation mode. "
                "Call module.eval() before computing KFAC."
            )

            raise RuntimeError(msg)

        v = backproped
        # KFLR: keep batch dimension -> param_mjp(..., sum_batch=False) => [V, N, C]
        JTv = self.derivatives.param_mjp(
            "weight", module, g_inp, g_out, v, sum_batch=False
        )
        # Flatten [V, N, C] -> [V*N, C]
        JTv = JTv.reshape(-1, module.num_features)

        kfac_gamma = JTv.T @ JTv
        return [kfac_gamma]

    def bias(
        self,
        ext,
        module: nn.Module,
        g_inp: tuple[Tensor, ...],
        g_out: tuple[Tensor, ...],
        backproped: Tensor,
    ) -> list[Tensor]:
        del ext

        if module.training:
            msg = (
                "KFAC for BatchNorm is only supported in evaluation mode. "
                "Call module.eval() before computing KFAC."
            )
            raise RuntimeError(msg)

        v = backproped
        JTv = self.derivatives.param_mjp(
            "bias", module, g_inp, g_out, v, sum_batch=False
        )
        JTv = JTv.reshape(-1, module.num_features)

        kfac_beta = JTv.T @ JTv
        return [kfac_beta]


def create_kfac_with_batchnorm(mc_samples: int = 1) -> KFAC | KFLR:
    """Create a KFAC/KFLR extension with BatchNorm support.

    This function creates and returns a curvature extension and registers the
    BatchNorm extension for all BatchNorm variants (1d, 2d, 3d).
    If ``mc_samples == 0``, it returns a ``KFLR`` extension; otherwise it returns
    ``KFAC`` with the requested number of Monte Carlo samples.

    Args:
        mc_samples: Number of Monte Carlo samples for KFAC approximation.
                   Defaults to 1 for exact computation when batch_size=1.

    Returns:
        KFAC or KFLR extension with BatchNorm support registered.

    Example:
        >>> kfac_ext = create_kfac_with_batchnorm(mc_samples=1)
        >>> with backpack(kfac_ext):
        ...     loss.backward()
    """
    if mc_samples == 0:
        # KFLR path: keep batch dimension in BN blocks
        kflr_extension = SummingKFLR()
        kflr_batchnorm = KFLRBatchNormNd()
        kflr_extension.set_module_extension(BatchNorm1d, kflr_batchnorm)
        kflr_extension.set_module_extension(BatchNorm2d, kflr_batchnorm)
        kflr_extension.set_module_extension(BatchNorm3d, kflr_batchnorm)
        return kflr_extension

    # KFAC path: sum over batch in BN blocks
    kfac_extension = KFAC(mc_samples=mc_samples)
    kfac_batchnorm = KFACBatchNormNd()
    kfac_extension.set_module_extension(BatchNorm1d, kfac_batchnorm)
    kfac_extension.set_module_extension(BatchNorm2d, kfac_batchnorm)
    kfac_extension.set_module_extension(BatchNorm3d, kfac_batchnorm)
    return kfac_extension


def get_kfac_list_loader(
    loader: DataLoader,
    model: nn.Module,
    device: str | torch.device,
    params: list[Tensor],
    mc_samples: int = 1,
    *,
    loss: Literal["ce", "mse"] = "ce",
) -> list[list[Tensor]]:
    """Compute the KFAC approximation based on a list of minibatches.

    Args:
        loader: DataLoader containing minibatches.
        model: Neural network model extended with BackPACK.
        device: Device to run computations on.
        params: Parameters to compute KFAC with respect to.
        loss: Loss function to use.
        mc_samples: Number of Monte Carlo samples for KFAC approximation.

    Returns:
        KFAC approximation accumulated over all minibatches.
    """
    loss_fn = extend(
        nn.CrossEntropyLoss(reduction="sum")
        if loss == "ce"
        else nn.MSELoss(reduction="sum")
    )

    # Accumulate KFAC approximations over all minibatches
    num_data = 0
    for i, (X, y) in enumerate(loader):
        batch_size = X.shape[0]
        new_num_data = num_data + batch_size

        if i == 0:
            # Initialize `kfac`.
            # If there is only one minibatch, return the KFAC approximation.
            kfac = get_kfac_list_minibatch(
                X,
                y,
                model,
                loss_fn,
                device,
                params,
                mc_samples,
                loss,
            )

            if len(loader) == 1:
                return kfac
        else:
            # Compute minibatch KFAC approximation
            mb_kfac = get_kfac_list_minibatch(
                X,
                y,
                model,
                loss_fn,
                device,
                params,
                mc_samples,
                loss,
            )

            # Add minibatch KFAC approximation
            kfac = add_kfac_lists(
                kfac,
                mb_kfac,
                num_data,
                batch_size,
            )

        # Update number of data points
        num_data = new_num_data

    return kfac


def get_kfac_list_minibatch(
    X: Tensor,
    y: Tensor,
    model: nn.Module,
    loss_fn: nn.Module,
    device: str | torch.device,
    params: list[Tensor],
    mc_samples: int = 1,
    loss: Literal["ce", "mse"] = "ce",
) -> list[list[Tensor]]:
    """Get the KFAC approximation based on one minibatch `(X, y)`.

    This returns a list-representation of KFAC. Its entries are lists that contain
    either a single matrix `[Fi]` or a pair of matrices `[Ai, Bi]` such that
    `Fi = Ai kron Bi`. An example could look like this:
    ```
    kfac = [
        [F1],
        [A2, B2],  # F2 = A2 kron B2
        [F3],
        [A4, B4],  # F4 = A4 kron B4
        [A5, B5],  # F5 = A5 kron B5
        ...
    ]
    ```

    Args:
        X: Input tensor batch.
        y: Target tensor batch.
        model: Neural network model extended with BackPACK.
        loss_fn: Loss function extended with BackPACK.
        device: Device to run computations on.
        params: Parameters to compute KFAC with respect to.
        mc_samples: Number of Monte Carlo samples for KFAC approximation.
        loss: Loss function to use.

    Returns:
        KFAC approximation for the given minibatch.
    """
    # Forward and backward pass
    X, y = X.to(device), y.to(device)

    with torch.enable_grad():
        logits = model(X)

        if loss == "mse":
            y = prepare_targets_for_mse_loss(logits, y)
        loss_value = loss_fn(logits, y)

        # Choose KFLR via unified API when mc_samples == 0, else KFAC
        ext = create_kfac_with_batchnorm(mc_samples=mc_samples)

        with backpack(ext):
            loss_value.backward()

        # Extract factor lists depending on extension used
        if mc_samples == 0:
            kfac = [
                [elem.detach() for elem in param.kflr]
                for param in params
                if param.requires_grad
            ]
        else:
            kfac = [
                [elem.detach() for elem in param.kfac]
                for param in params
                if param.requires_grad
            ]

    return kfac


def add_kfac_lists(
    kfac_1: list[list[Tensor]],
    kfac_2: list[list[Tensor]],
    num_samples_1: float,
    num_samples_2: float,
) -> list[list[Tensor]]:
    """Add two KFAC approximations.

    Args:
        kfac_1: First KFAC approximation.
        kfac_2: Second KFAC approximation.
        num_samples_1: Number of data points used in `kfac_1`.
        num_samples_2: Number of data points used in `kfac_2`.

    Returns:
        Summed KFAC approximation.
    """
    res = []
    new_num_samples = num_samples_1 + num_samples_2
    for factors_1, factors_2 in zip(kfac_1, kfac_2, strict=True):
        # We simply add single factors
        if len(factors_1) == 1:
            res.append([factors_1[0] + factors_2[0]])
        # When there are two factors, we add the first pair and scale-add the second
        elif len(factors_1) == 2:
            res.append([
                factors_1[0] + factors_2[0],
                (num_samples_1 * factors_1[1] + num_samples_2 * factors_2[1])
                / new_num_samples,
            ])

    return res


def prepare_targets_for_mse_loss(logits: Tensor, targets: Tensor) -> Tensor:
    one_hot = F.one_hot(targets, num_classes=logits.shape[-1])
    return one_hot.to(dtype=logits.dtype, device=logits.device)


def subtract_kfac_lists(
    kfac_1: list[list[Tensor]],
    kfac_2: list[list[Tensor]],
    num_samples_1: int,
    num_samples_2: int,
) -> list[list[Tensor]]:
    """Subtract two KFAC approximations.

    Args:
        kfac_1: First KFAC approximation.
        kfac_2: Second KFAC approximation.
        num_samples_1: Number of data points used in `kfac_1`.
        num_samples_2: Number of data points used in `kfac_2`.

    Returns:
        Subtracted KFAC approximation.
    """
    new_num_samples = num_samples_1 - num_samples_2
    res = []

    for factors_full, factors_forgotten in zip(kfac_1, kfac_2, strict=True):
        # For single factors, simply subtract
        if len(factors_full) == 1:
            res.append([factors_full[0] - factors_forgotten[0]])
        # For two factors, subtract first and scale-subtract second
        elif len(factors_full) == 2:
            res.append([
                factors_full[0] - factors_forgotten[0],
                (num_samples_1 * factors_full[1] - num_samples_2 * factors_forgotten[1])
                / new_num_samples,
            ])

    return res


class KFACOperatorBase(MatmulMixin):
    """Shared logic for KFAC-based linear operators."""

    def __init__(self, kfac_list: list[list[Tensor]], prior_precision: float) -> None:
        if not kfac_list:
            msg = "kfac_list must be non-empty"
            raise ValueError(msg)

        self._prior = prior_precision
        self._specs: list[tuple] = []
        total_dim = 0
        first_tensor = kfac_list[0][0]
        self.device = first_tensor.device
        self.dtype = first_tensor.dtype

        for block in kfac_list:
            if len(block) == 1:
                (F,) = block
                s, U = torch.linalg.eigh(F)
                s = torch.clamp_min(s, 0)
                self._specs.append(("single", U, s))
                total_dim += U.shape[0]
            elif len(block) == 2:
                A, B = block
                sA, UA = torch.linalg.eigh(A)
                sB, UB = torch.linalg.eigh(B)
                sA = torch.clamp_min(sA, 0)
                sB = torch.clamp_min(sB, 0)
                self._specs.append((
                    "kron",
                    UA,
                    sA,
                    UB,
                    sB,
                ))
                total_dim += UA.shape[0] * UB.shape[0]
            else:
                msg = "Only blocks of length 1 or 2 are expected."
                raise ValueError(msg)

        self.shape = (total_dim, total_dim)

    def _apply_single(self, block: Tensor, U: Tensor, s: Tensor) -> Tensor:
        del self, block, U, s
        raise NotImplementedError

    def _apply_kron(
        self,
        block: Tensor,
        UA: Tensor,
        sA: Tensor,
        UB: Tensor,
        sB: Tensor,
    ) -> Tensor:
        del self, block, UA, sA, UB, sB
        raise NotImplementedError

    def matmat(self, X: Tensor) -> Tensor:
        squeeze = X.ndim == 1
        if squeeze:
            mat = X.unsqueeze(-1)
        elif X.ndim == 2:
            mat = X
        else:
            msg = "vector must be 1D or 2D"
            raise ValueError(msg)

        out = torch.zeros_like(mat, device=self.device, dtype=self.dtype)
        p = 0
        for spec in self._specs:
            tag = spec[0]
            if tag == "single":
                _, U, s = spec
                q = p + U.shape[0]
                block = mat[p:q, :]
                out[p:q, :] = self._apply_single(block, U, s)
                p = q
            else:
                _, UA, sA, UB, sB = spec
                m, n = UA.shape[0], UB.shape[0]
                q = p + m * n
                block = mat[p:q, :]
                out[p:q, :] = self._apply_kron(block, UA, sA, UB, sB)
                p = q

        if p != mat.shape[0]:
            msg = "vector has invalid size"
            raise ValueError(msg)

        if squeeze:
            return out.squeeze(-1)

        return out

    def rmatmat(self, X: Tensor) -> Tensor:
        if X.ndim == 1:
            return self.matmat(X)

        if X.ndim != 2:
            msg = "vector must be 1D or 2D"
            raise ValueError(msg)

        return self.matmat(X.T).T


class KFACCovarianceOperator(KFACOperatorBase):
    def _apply_single(self, block: Tensor, U: Tensor, s: Tensor) -> Tensor:
        z = U.T @ block
        z = z / (s.unsqueeze(-1) + self._prior)
        return U @ z

    def _apply_kron(
        self,
        block: Tensor,
        UA: Tensor,
        sA: Tensor,
        UB: Tensor,
        sB: Tensor,
    ) -> Tensor:
        m, n = UA.shape[0], UB.shape[0]
        block = block.T.reshape(-1, m, n).permute(0, 2, 1)
        transformed = torch.matmul(UB.T, block)
        transformed = torch.matmul(transformed, UA)
        denom = sB[:, None] * sA[None, :] + self._prior
        transformed = transformed / denom.unsqueeze(0)
        Y = torch.matmul(UB, transformed)
        Y = torch.matmul(Y, UA.T)
        Y = Y.permute(0, 2, 1)
        return Y.reshape(Y.shape[0], m * n).T


class KFACScaleOperator(KFACOperatorBase):
    def _apply_single(self, block: Tensor, U: Tensor, s: Tensor) -> Tensor:
        z = U.T @ block
        z = z / torch.sqrt(s.unsqueeze(-1) + self._prior)
        return U @ z

    def _apply_kron(
        self,
        block: Tensor,
        UA: Tensor,
        sA: Tensor,
        UB: Tensor,
        sB: Tensor,
    ) -> Tensor:
        m, n = UA.shape[0], UB.shape[0]
        block = block.T.reshape(-1, m, n).permute(0, 2, 1)
        transformed = torch.matmul(UB.T, block)
        transformed = torch.matmul(transformed, UA)
        denom = torch.sqrt(sB[:, None] * sA[None, :] + self._prior)
        transformed = transformed / denom.unsqueeze(0)
        Y = torch.matmul(UB, transformed)
        Y = torch.matmul(Y, UA.T)
        Y = Y.permute(0, 2, 1)
        return Y.reshape(Y.shape[0], m * n).T


class KFACInvScaleOperator(KFACOperatorBase):
    def _apply_single(self, block: Tensor, U: Tensor, s: Tensor) -> Tensor:
        z = U.T @ block
        z = torch.sqrt(s.unsqueeze(-1) + self._prior) * z
        return U @ z

    def _apply_kron(
        self,
        block: Tensor,
        UA: Tensor,
        sA: Tensor,
        UB: Tensor,
        sB: Tensor,
    ) -> Tensor:
        m, n = UA.shape[0], UB.shape[0]
        block = block.T.reshape(-1, m, n).permute(0, 2, 1)
        transformed = torch.matmul(UB.T, block)
        transformed = torch.matmul(transformed, UA)
        denom = torch.sqrt(sB[:, None] * sA[None, :] + self._prior)
        transformed = denom.unsqueeze(0) * transformed
        Y = torch.matmul(UB, transformed)
        Y = torch.matmul(Y, UA.T)
        Y = Y.permute(0, 2, 1)
        return Y.reshape(Y.shape[0], m * n).T


class KFACPrecisionOperator(KFACOperatorBase):
    def _apply_single(self, block: Tensor, U: Tensor, s: Tensor) -> Tensor:
        z = U.T @ block
        z = (s.unsqueeze(-1) + self._prior) * z
        return U @ z

    def _apply_kron(
        self,
        block: Tensor,
        UA: Tensor,
        sA: Tensor,
        UB: Tensor,
        sB: Tensor,
    ) -> Tensor:
        m, n = UA.shape[0], UB.shape[0]
        block = block.T.reshape(-1, m, n).permute(0, 2, 1)
        transformed = torch.matmul(UB.T, block)
        transformed = torch.matmul(transformed, UA)
        denom = sB[:, None] * sA[None, :] + self._prior
        transformed = denom.unsqueeze(0) * transformed
        Y = torch.matmul(UB, transformed)
        Y = torch.matmul(Y, UA.T)
        Y = Y.permute(0, 2, 1)
        return Y.reshape(Y.shape[0], m * n).T


def get_kfac_cov_op(kfac_list, prior_precision: float) -> KFACCovarianceOperator:
    return KFACCovarianceOperator(kfac_list, prior_precision)


def get_kfac_scale_op(kfac_list, prior_precision: float) -> KFACScaleOperator:
    return KFACScaleOperator(kfac_list, prior_precision)


def get_kfac_inv_scale_op(kfac_list, prior_precision: float) -> KFACInvScaleOperator:
    return KFACInvScaleOperator(kfac_list, prior_precision)


def get_kfac_precision_op(kfac_list, prior_precision: float) -> KFACPrecisionOperator:
    return KFACPrecisionOperator(kfac_list, prior_precision)
