"""Linear operator classes for matrix-free operations.

Notation:
- H: per-sample softmax cross-entropy Hessian in logit space (H = Diag(p) - p p^T).
- H^{1/2}: symmetric PSD square root of H (applied blockwise per sample).
- Psi_S: whitened features for a set S, Psi_S = H_S^{1/2} J_S.
- Sigma: parameter-space covariance/preconditioner.
"""

from collections.abc import Callable, Iterable
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.func import functional_call, jvp, vjp, vmap
from torch.utils.data import DataLoader
from tqdm import tqdm

from bul.models.linearized import LinearizedModel
from bul.utils.flatten import get_param_keys, unflatten_to_param_dict_like
from bul.utils.loaders import (
    PrefetchLoader,
    calculate_num_samples,
    calculate_total_dim,
    make_deterministic_loader,
)


def iter_progress(
    iterable: Iterable[Any],
    *,
    enabled: bool,
    desc: str,
) -> Iterable[Any]:
    return tqdm(iterable, desc=desc) if enabled else iterable


class MatmulMixin:
    """Adds right/left @ using matvec/rmatvec implemented by subclasses.

    Right multiply:  y = op @ x   -> column-wise matvec
    Left multiply:   Y = X @ op   -> row-wise rmatvec
    """

    device: torch.device
    dtype: torch.dtype
    shape: tuple[int, int]

    def __matmul__(self, X: Tensor) -> Tensor:
        return self.matmat(X)

    def __rmatmul__(self, X: Tensor) -> Tensor:
        return self.rmatmat(X)

    def _progress_desc(self, phase: str) -> str:
        return f"{self.__class__.__name__}::{phase} ({self.device!s})"

    def matmat(self, X: Tensor) -> Tensor:
        raise NotImplementedError

    def rmatmat(self, X: Tensor) -> Tensor:
        raise NotImplementedError


# Softmax CE Hessian helpers: H = Diag(p) - p p^T
@torch.compile(fullgraph=True, mode="reduce-overhead", dynamic=False)
def h_matmat(probs: Tensor, vec: Tensor) -> Tensor:
    """Compute y = H v with H = Diag(p) - p p^T, batched over samples.

    Args:
        probs: [B, C] softmax probabilities (rows sum to 1).
        vec: [B, C] or [K, B, C] vectors.

    Returns:
        [B, C] result.

    Raises:
        ValueError: If ``vec`` does not have 2 or 3 dimensions.
    """
    if vec.dim() == 2:
        # diag(p) v
        y = probs * vec
        # - p (p^T v)
        s = y.sum(dim=-1, keepdim=True)  # [B,1] == <p, v>
        return y - probs * s

    if vec.dim() == 3:
        probs_unsqueezed = probs.unsqueeze(0)  # [1, B, C]
        y = probs_unsqueezed * vec  # [K, B, C]
        s = y.sum(dim=-1, keepdim=True)  # [K, B, 1]
        return y - probs_unsqueezed * s

    msg = "vec must be [B, C] or [K, B, C]"
    raise ValueError(msg)


@torch.compile(fullgraph=True)
def apply_h_sqrt_eigh(
    probs: Tensor,  # [B, C]
    vec: Tensor,  # [B, C]
) -> Tensor:
    """Exact H^{1/2} via batched symmetric eigendecomposition (fallback/reference).

    Memory/time heavy for large C and B, but numerically exact.

    Returns:
        [B, C] or [K, B, C] result of H^{1/2} vec.

    Raises:
        ValueError: If ``vec`` does not have 2 or 3 dimensions.
    """
    H = torch.diag_embed(probs) - probs.unsqueeze(-1) * probs.unsqueeze(-2)  # [B, C, C]
    evals, evecs = torch.linalg.eigh(H)  # [B, C], [B, C, C]
    evals_sqrt = torch.clamp_min(evals, 0).sqrt()  # [B, C]

    if vec.dim() == 2:
        tmp = evecs.transpose(-2, -1) @ vec.unsqueeze(-1)  # [B, C, 1]
        tmp = evals_sqrt.unsqueeze(-1) * tmp  # [B, C, 1]
        out = evecs @ tmp
        return out.squeeze(-1)  # [B, C]

    if vec.dim() == 3:
        QT = evecs.transpose(-2, -1)  # [B, C, C]
        tmp = torch.einsum("bij,bjk->bik", QT, vec)  # [B, C, K]
        tmp = evals_sqrt.unsqueeze(-1) * tmp  # [B, C, K]
        return torch.einsum("bij,bjk->bik", evecs, tmp)  # [B, C, K]

    msg = "vec must be [B, C] or [B, C, K]"
    raise ValueError(msg)


@torch.compile(fullgraph=True)
def apply_h_sqrt_lanczos(
    probs: Tensor,  # [B, C]
    vec: Tensor,  # [B, C]
    *,
    num_steps: int = 64,
    tol: float = 1e-7,
    calc_dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Approximate y = H^{1/2} vec via batched Lanczos + small dense sqrt(T).

    Details:
      - For each sample b, run m-step Lanczos on the C x C SPD W_b with start v_b.
      - Build T_b (m x m tridiagonal) and compute f(T_b) e_1 with f(x) = sqrt(x).
      - Return y_b = ||v_b|| * Q_b @ [f(T_b) e_1], where Q_b are Lanczos basis vectors.

    Complexity per batch: O(B * C * m + B * C * m^2) flops; memory ~ B * C * m floats.

    Args:
        probs: [B, C] softmax probabilities per sample.
        vec: [B, C] vectors to whiten.
        num_steps: Lanczos steps (20-40 is typically enough).
        tol: breakdown threshold for beta_k; early stops per sample.
        calc_dtype: Dtype for the calculations.

    Returns:
        [B, C] approximate W^{1/2} vec.
    """
    device = probs.device
    return_dtype = probs.dtype
    B, C = probs.shape

    probs = probs.to(calc_dtype)
    vec = vec.to(calc_dtype)
    vec = vec - vec.mean(dim=-1, keepdim=True)

    # Handle zero vectors via masking (no early Python branching)
    v_norm = vec.norm(dim=-1)  # [B]
    out = torch.zeros_like(vec, device=device, dtype=calc_dtype)
    active = v_norm > 0
    initial_active = active.clone()

    # Normalize starting vectors where nonzero
    safe_v_norm = torch.where(active, v_norm, torch.ones_like(v_norm))
    q_cur = torch.where(
        active.unsqueeze(-1), vec / safe_v_norm.unsqueeze(-1), torch.zeros_like(vec)
    )

    q_prev = torch.zeros_like(vec)
    beta_prev = torch.zeros(B, dtype=calc_dtype, device=device)

    # Keep track of effective steps per sample (breakdown)
    eff_m = torch.full((B,), num_steps, device=device, dtype=torch.int32)
    current_active = active.clone()

    q_list = []
    alpha_list = []
    beta_list = []

    for k in range(num_steps):
        # w = H q_k - beta_{k-1} q_{k-1}
        w = h_matmat(probs, q_cur)
        if k > 0:
            w = w - beta_prev.unsqueeze(-1) * q_prev

        # alpha_k = <q_k, w>
        alpha_k = (q_cur * w).sum(dim=-1)  # [B]
        alpha_list.append(alpha_k)

        # w <- w - alpha_k q_k
        w = w - alpha_k.unsqueeze(-1) * q_cur

        # beta_k = ||w||
        beta_k = w.norm(dim=-1)  # [B]
        if k < num_steps - 1:
            beta_list.append(beta_k)

        # Save basis vector
        q_list.append(q_cur)

        # Early breakdown per sample
        broken = (beta_k <= tol) & current_active
        update_mask = (eff_m == num_steps) & broken
        new_val = torch.full_like(eff_m, k + 1)
        eff_m = torch.where(update_mask, new_val, eff_m)

        # Prepare next q with masking
        mask_next = (beta_k > tol) & current_active
        safe_beta = torch.where(mask_next, beta_k, torch.ones_like(beta_k))
        q_next = torch.where(
            mask_next.unsqueeze(-1),
            w / safe_beta.unsqueeze(-1),
            torch.zeros_like(w),
        )

        # Update recurrence
        q_prev = q_cur
        q_cur = q_next
        beta_prev = beta_k
        current_active = mask_next

    # Build outputs per sample from small T and basis
    # We will loop over samples
    # Samples that never broke keep full num_steps
    eff_m = torch.where(initial_active, eff_m, torch.zeros_like(eff_m))

    Q = (
        torch.stack(q_list, dim=1)
        if q_list
        else torch.zeros(B, num_steps, C, dtype=calc_dtype, device=device)
    )
    alphas = (
        torch.stack(alpha_list, dim=1)
        if alpha_list
        else torch.zeros(B, num_steps, dtype=calc_dtype, device=device)
    )
    if beta_list:
        betas = torch.stack(beta_list, dim=1)
    else:
        betas = torch.zeros(B, 0, dtype=calc_dtype, device=device)

    steps = torch.arange(num_steps, device=device)
    diag_mask = steps.unsqueeze(0) < eff_m.unsqueeze(1)
    if num_steps > 1:
        off_steps = torch.arange(num_steps - 1, device=device)
        off_mask = off_steps.unsqueeze(0) < (eff_m.clamp_min(1) - 1).unsqueeze(1)
    else:
        off_mask = torch.zeros(B, 0, device=device, dtype=torch.bool)

    diag_vals = alphas * diag_mask.to(calc_dtype)
    off_vals = betas * off_mask.to(calc_dtype)

    diag_T = torch.diag_embed(diag_vals)
    if num_steps > 1:
        off_T = torch.diag_embed(off_vals, offset=1)
        off_T = off_T + off_T.transpose(-1, -2)
    else:
        off_T = torch.zeros_like(diag_T)

    T = diag_T + off_T

    evals, evecs = torch.linalg.eigh(T)
    evals_sqrt = torch.clamp_min(evals, 0).sqrt()

    e1 = torch.zeros(num_steps, device=device, dtype=calc_dtype)
    e1[0] = 1.0
    tmp = torch.matmul(evecs.transpose(-2, -1), e1)
    tmp = evals_sqrt * tmp
    coeff = torch.matmul(evecs, tmp.unsqueeze(-1)).squeeze(-1)
    coeff = coeff * diag_mask.to(calc_dtype)

    y = (coeff.unsqueeze(-1) * Q).sum(dim=1)
    out = v_norm.unsqueeze(-1) * y
    out = torch.where(initial_active.unsqueeze(-1), out, torch.zeros_like(out))

    return out.to(return_dtype)


# Default dispatcher for H^{1/2}
@torch.compile(fullgraph=True)
def apply_h_sqrt(
    probs: Tensor,
    vec: Tensor,
    *,
    mode: str = "eigh",
    num_steps: int = 32,
    tol: float = 1e-7,
    calc_dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Dispatch to H^{1/2} implementations.

    mode="lanczos": fast, matrix-free; use num_steps/tol.
    mode="eigh": exact but expensive (forms H and eigendecomposes per sample).

    Returns:
        Result of applying H^{1/2} to vec using the specified mode.

    Raises:
        ValueError: If ``vec`` does not have 2 or 3 dimensions.
    """
    if mode == "eigh":
        return apply_h_sqrt_eigh(probs, vec)

    if vec.dim() == 2:
        return apply_h_sqrt_lanczos(
            probs,
            vec,
            num_steps=num_steps,
            tol=tol,
            calc_dtype=calc_dtype,
        )

    if vec.dim() == 3:

        def apply_h_sqrt_lanczos_one(vec_single: Tensor) -> Tensor:
            return apply_h_sqrt_lanczos(
                probs,
                vec_single,
                num_steps=num_steps,
                tol=tol,
                calc_dtype=calc_dtype,
            )

        out = vmap(apply_h_sqrt_lanczos_one, in_dims=0, out_dims=0)(
            vec.movedim(-1, 0)
        )  # [K, B, C]
        return out.movedim(0, -1)  # [B, C, K]

    msg = "vec must be [B, C] or [B, C, K]"
    raise ValueError(msg)


def matrix_to_batched_param_dict(
    matrix: Tensor,  # [P, K]
    params_like: dict[str, Tensor],
    keys: list[str],
) -> dict[str, Tensor]:
    """Convert a flat matrix into a batched param dict.

    Returns:
        Dictionary with leaves shaped ``[K, *param_shape]``.
    """
    sizes = [params_like[k].numel() for k in keys]
    chunks = list(matrix.split(sizes, dim=0))
    out: dict[str, Tensor] = {}
    for key, chunk in zip(keys, chunks, strict=True):
        shape = params_like[key].shape
        out[key] = chunk.T.contiguous().view(-1, *shape)  # [K, *shape]
    return out


def batched_param_dict_to_matrix(
    batched_param_dict: dict[str, Tensor],  # Leaves [K, *shape]
    keys: list[str],
) -> Tensor:
    """Convert a batched param dict back to a flat matrix.

    Returns:
        Tensor of shape ``[P, K]`` with column-stacked parameters.
    """
    matrices = []
    for key in keys:
        matrix = batched_param_dict[key].reshape(
            batched_param_dict[key].shape[0], -1
        )  # [K, p_k]
        matrices.append(matrix.T)  # [p_k, K]
    return torch.cat(matrices, dim=0)  # [P, K]


def make_forward_fn(
    original_model: torch.nn.Module,
    buffers: dict[str, Tensor],
    inputs: Tensor,
) -> Callable[[dict[str, Tensor]], Tensor]:
    """Returns f(params_dict) = model(params_dict, buffers)(inputs) -> logits [B,C]."""

    def forward_fn(params_dict: dict[str, Tensor]) -> Tensor:
        return functional_call(original_model, (params_dict, buffers), (inputs,))

    return forward_fn


@torch.compile(fullgraph=True, dynamic=False)
def psi_mat_batched_body(
    linearized_model: LinearizedModel,
    inputs_device: Tensor,
    batched_tangents: dict,
    h_sqrt_mode: str,
    apply_h_sqrt_kwargs: dict,
) -> Tensor:
    forward = make_forward_fn(
        linearized_model.original_model,
        linearized_model.original_buffers,
        inputs_device,
    )
    logits = forward(linearized_model.original_params)  # [B, C]
    probabilities = logits.softmax(dim=-1)

    def jvp_single(tangent: dict[str, Tensor], forward=forward) -> Tensor:
        _, Jv = jvp(forward, (linearized_model.original_params,), (tangent,))
        return Jv  # [B, C]

    logit_deltas = vmap(jvp_single, in_dims=0, out_dims=0)(
        batched_tangents
    )  # [K, B, C]
    logit_deltas_BCK = logit_deltas.movedim(0, -1)  # [B, C, K]

    whitened = apply_h_sqrt(
        probabilities, logit_deltas_BCK, mode=h_sqrt_mode, **apply_h_sqrt_kwargs
    )  # [B, C, K]

    return whitened


@torch.compile(fullgraph=True, dynamic=False)
def psi_mat_single_body(
    linearized_model: LinearizedModel,
    inputs_device: Tensor,
    direction: dict,
    h_sqrt_mode: str,
    apply_h_sqrt_kwargs: dict,
) -> Tensor:
    forward = make_forward_fn(
        linearized_model.original_model,
        linearized_model.original_buffers,
        inputs_device,
    )
    logits = forward(linearized_model.original_params)  # [B, C]
    probabilities = logits.softmax(dim=-1)

    _, logit_deltas = jvp(
        forward, (linearized_model.original_params,), (direction,)
    )  # [B, C]
    whitened = apply_h_sqrt(
        probabilities, logit_deltas, mode=h_sqrt_mode, **apply_h_sqrt_kwargs
    )  # [B, C]

    return whitened


def psi_mat(
    linearized_model: LinearizedModel,
    loader: DataLoader,
    num_classes: int,
    Pmat: Tensor,
    *,
    h_sqrt_mode: str,
    apply_h_sqrt_kwargs: dict,
    device: torch.device,
    dtype: torch.dtype,
    param_keys: list[str],
    progressbar: bool = False,
    progress_desc: str | None = None,
) -> Tensor:
    """Compute Psi_S @ Pmat with Psi_S := H_S^{1/2} J_S.

    Shapes:
        - Input  Pmat: [P]       -> Output: [|S| * C]
        - Input  Pmat: [P, K]    -> Output: [|S| * C, K]

    Args:
        linearized_model: Linearized model wrapper exposing `original_model`,
            `original_params`, and `original_buffers`.
        loader: Deterministic DataLoader over the set S (defines sample order).
        num_classes: Number of classes C (logit dimension).
        Pmat: Parameter-space vector or column-stacked matrix.
        h_sqrt_mode: Backend for H^{1/2} ("eigh" or "lanczos").
        apply_h_sqrt_kwargs: Extra kwargs forwarded to `apply_h_sqrt`.
        device: Torch device for computation.
        dtype: Torch dtype for intermediate tensors and outputs.
        param_keys: Parameter key order for (un)flattening.
        progressbar: Whether to wrap loader iteration with a tqdm progress bar.
        progress_desc: Description used for the progress bar when enabled.

    Returns:
        Psi_S @ Pmat with the output shape matching the input's column layout.

    Raises:
        ValueError: If ``Pmat`` has unsupported dimensionality or zero elements.
    """
    is_vector_input = Pmat.ndim == 1
    if is_vector_input:
        if Pmat.numel() == 0:
            msg = "Pmat must be non-empty."
            raise ValueError(msg)
        num_cols = 1
        Pmat_matrix = Pmat.view(-1, 1)
    elif Pmat.ndim == 2:
        num_cols = Pmat.shape[1]
        Pmat_matrix = Pmat
    else:
        msg = "Pmat must have shape [P] or [P, K]."
        raise ValueError(msg)

    # Output allocation in sample space, stacked in loader order
    total_samples = calculate_total_dim(loader, num_classes)
    output = torch.zeros(total_samples, num_cols, device=device, dtype=dtype)

    desc = progress_desc or "psi_mat"

    if num_cols == 1:
        # Unbatched path: one JVP per minibatch (no vmap).
        direction = unflatten_to_param_dict_like(
            Pmat_matrix[:, 0].to(device=device, dtype=dtype),
            linearized_model.original_params,
            param_keys,
        )
        write_ptr = 0
        data_iter = iter_progress(loader, enabled=progressbar, desc=desc)
        for inputs, _ in data_iter:
            inputs_device = inputs.to(device, non_blocking=True)

            whitened = psi_mat_single_body(
                linearized_model=linearized_model,
                inputs_device=inputs_device,
                direction=direction,
                h_sqrt_mode=h_sqrt_mode,
                apply_h_sqrt_kwargs=apply_h_sqrt_kwargs,
            )

            batch_size = inputs_device.shape[0]
            output[write_ptr : write_ptr + batch_size * num_classes, 0] = (
                whitened.flatten()
            )
            write_ptr += batch_size * num_classes
    else:
        # Batched path: vmap JVP over K directions per minibatch.
        batched_tangents = matrix_to_batched_param_dict(
            Pmat_matrix, linearized_model.original_params, param_keys
        )  # Leaves: [K, *shape]

        write_ptr = 0
        data_iter = iter_progress(loader, enabled=progressbar, desc=desc)
        for inputs, _ in data_iter:
            inputs_device = inputs.to(device, non_blocking=True)

            whitened = psi_mat_batched_body(
                linearized_model=linearized_model,
                inputs_device=inputs_device,
                batched_tangents=batched_tangents,
                h_sqrt_mode=h_sqrt_mode,
                apply_h_sqrt_kwargs=apply_h_sqrt_kwargs,
            )

            batch_size = inputs_device.shape[0]
            output[write_ptr : write_ptr + batch_size * num_classes, :] = (
                whitened.reshape(batch_size * num_classes, num_cols)
            )
            write_ptr += batch_size * num_classes

    return output.flatten() if is_vector_input else output


@torch.compile(fullgraph=True, dynamic=False)
def psiT_mat_batched_body(
    linearized_model: LinearizedModel,
    inputs_device: Tensor,
    s_batch: Tensor,
    h_sqrt_mode: str,
    apply_h_sqrt_kwargs: dict,
) -> dict:
    forward = make_forward_fn(
        linearized_model.original_model,
        linearized_model.original_buffers,
        inputs_device,
    )
    logits, vjp_fn = vjp(forward, linearized_model.original_params)
    probabilities = logits.softmax(dim=-1)

    whitened = apply_h_sqrt(
        probabilities, s_batch, mode=h_sqrt_mode, **apply_h_sqrt_kwargs
    )  # [B, C, K]

    whitened_KBC = whitened.movedim(-1, 0)  # [K, B, C]

    grads_batched = vmap(lambda cot: vjp_fn(cot)[0], in_dims=0, out_dims=0)(
        whitened_KBC
    )

    return grads_batched


@torch.compile(fullgraph=True, dynamic=False)
def psiT_mat_single_body(
    linearized_model: LinearizedModel,
    inputs_device: Tensor,
    s_batch: Tensor,
    h_sqrt_mode: str,
    apply_h_sqrt_kwargs: dict,
) -> dict:
    forward = make_forward_fn(
        linearized_model.original_model,
        linearized_model.original_buffers,
        inputs_device,
    )
    logits, vjp_fn = vjp(forward, linearized_model.original_params)
    probabilities = logits.softmax(dim=-1)

    whitened = apply_h_sqrt(
        probabilities, s_batch, mode=h_sqrt_mode, **apply_h_sqrt_kwargs
    )  # [B, C]

    grad_dict = vjp_fn(whitened)[0]

    return grad_dict


def psiT_mat(
    linearized_model: LinearizedModel,
    loader: DataLoader,
    num_classes: int,
    Smat: Tensor,
    *,
    h_sqrt_mode: str,
    apply_h_sqrt_kwargs: dict,
    device: torch.device,
    dtype: torch.dtype,
    param_keys: list[str],
    P: int,
    progressbar: bool = False,
    progress_desc: str | None = None,
) -> Tensor:
    """Compute Psi_S^T @ Smat with Psi_S := H_S^{1/2} J_S.

    Shapes:
        - Input  Smat: [|S| * C]     -> Output: [P]
        - Input  Smat: [|S| * C, K]  -> Output: [P, K]

    Args:
        linearized_model: Linearized model wrapper exposing `original_model`,
            `original_params`, and `original_buffers`.
        loader: Deterministic DataLoader over the set S (defines sample order).
        num_classes: Number of classes C (logit dimension).
        Smat: Sample-space vector or column-stacked matrix, stacked in loader order.
        h_sqrt_mode: Backend for H^{1/2} ("eigh" or "lanczos").
        apply_h_sqrt_kwargs: Extra kwargs forwarded to `apply_h_sqrt`.
        device: Torch device for computation.
        dtype: Torch dtype for intermediate tensors and outputs.
        param_keys: Parameter key order for flattening.
        P: Parameter-space dimension.
        progressbar: Whether to wrap loader iteration with a tqdm progress bar.
        progress_desc: Description used for the progress bar when enabled.

    Returns:
        Psi_S^T @ Smat with the output shape matching the input's column layout.

    Raises:
        ValueError: If ``Smat`` has unsupported dimensionality or zero elements.
    """
    is_vector_input = Smat.ndim == 1
    if is_vector_input:
        if Smat.numel() == 0:
            msg = "Smat must be non-empty."
            raise ValueError(msg)
        num_cols = 1
        Smat_matrix = Smat.view(-1, 1)
    elif Smat.ndim == 2:
        num_cols = Smat.shape[1]
        Smat_matrix = Smat
    else:
        msg = "Smat must have shape [|S| * C] or [|S| * C, K]."
        raise ValueError(msg)

    output = torch.zeros(P, num_cols, device=device, dtype=dtype)
    read_ptr = 0
    C = num_classes

    desc = progress_desc or "psiT_mat"

    if num_cols == 1:
        data_iter = iter_progress(loader, enabled=progressbar, desc=desc)
        for inputs, _ in data_iter:
            inputs_device = inputs.to(device, non_blocking=True)
            batch_size = inputs_device.shape[0]

            # Unbatched: single VJP.
            s_batch = Smat_matrix[read_ptr : read_ptr + batch_size * C, 0].reshape(
                batch_size, C
            )
            read_ptr += batch_size * C

            grad_dict = psiT_mat_single_body(
                linearized_model=linearized_model,
                inputs_device=inputs_device,
                s_batch=s_batch,
                h_sqrt_mode=h_sqrt_mode,
                apply_h_sqrt_kwargs=apply_h_sqrt_kwargs,
            )

            grad_flat = torch.cat([grad_dict[k].flatten() for k in param_keys], dim=0)
            output[:, 0].add_(grad_flat)
    else:
        data_iter = iter_progress(loader, enabled=progressbar, desc=desc)
        for inputs, _ in data_iter:
            inputs_device = inputs.to(device, non_blocking=True)
            batch_size = inputs_device.shape[0]

            # Batched: vmap VJP across K whitened columns.
            s_batch = Smat_matrix[read_ptr : read_ptr + batch_size * C, :].reshape(
                batch_size, C, num_cols
            )
            read_ptr += batch_size * C

            grads_batched = psiT_mat_batched_body(
                linearized_model=linearized_model,
                inputs_device=inputs_device,
                s_batch=s_batch,
                h_sqrt_mode=h_sqrt_mode,
                apply_h_sqrt_kwargs=apply_h_sqrt_kwargs,
            )

            output.add_(batched_param_dict_to_matrix(grads_batched, param_keys))

    return output.flatten() if is_vector_input else output


@torch.compile(fullgraph=True, dynamic=False)
def jt_h_j_mat_batched_body(
    linearized_model: LinearizedModel,
    inputs_device: Tensor,
    batched_tangents: dict,
) -> dict:
    forward = make_forward_fn(
        linearized_model.original_model,
        linearized_model.original_buffers,
        inputs_device,
    )
    logits, vjp_fn = vjp(forward, linearized_model.original_params)
    probabilities = logits.softmax(dim=-1)

    def jvp_single(tangent: dict[str, Tensor], forward=forward) -> Tensor:
        _, Jv = jvp(forward, (linearized_model.original_params,), (tangent,))
        return Jv  # [B, C]

    logit_deltas = vmap(jvp_single, in_dims=0, out_dims=0)(
        batched_tangents
    )  # [K, B, C]
    # Apply H to each of the K directions (broadcast over K).
    dot = (logit_deltas * probabilities).sum(dim=-1, keepdim=True)  # [K, B, 1]
    h_times_delta = probabilities * (logit_deltas - dot)  # [K, B, C]

    grads_batched = vmap(lambda cot: vjp_fn(cot)[0], in_dims=0, out_dims=0)(
        h_times_delta
    )
    return grads_batched


@torch.compile(fullgraph=True, mode="reduce-overhead", dynamic=False)
def jt_h_j_mat_single_body(
    linearized_model: LinearizedModel, inputs_device: Tensor, direction: dict
) -> dict:
    forward = make_forward_fn(
        linearized_model.original_model,
        linearized_model.original_buffers,
        inputs_device,
    )
    logits, vjp_fn = vjp(forward, linearized_model.original_params)
    probabilities = logits.softmax(dim=-1)

    _, logit_deltas = jvp(
        forward, (linearized_model.original_params,), (direction,)
    )  # [B, C]
    h_times_delta = h_matmat(probabilities, logit_deltas)  # [B, C]

    grad_dict = vjp_fn(h_times_delta)[0]

    return grad_dict


def jt_h_j_mat(
    linearized_model: LinearizedModel,
    loader: DataLoader,
    Pmat: Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    param_keys: list[str],
    progressbar: bool = False,
    progress_desc: str | None = None,
) -> Tensor:
    """Compute ``(J_S^T H_S J_S) @ Pmat``.

    Shapes:
        - Input  Pmat: [P]     -> Output: [P]
        - Input  Pmat: [P, K]  -> Output: [P, K]

    Args:
        linearized_model: Linearized model wrapper exposing `original_model`,
            `original_params`, and `original_buffers`.
        loader: Deterministic DataLoader over the set S (defines sample order).
        Pmat: Parameter-space vector or column-stacked matrix.
        device: Torch device for computation.
        dtype: Torch dtype for intermediate tensors and outputs.
        param_keys: Parameter key order for flattening.
        progressbar: Whether to wrap loader iteration with a tqdm progress bar.
        progress_desc: Description used for the progress bar when enabled.

    Returns:
        (J_S^T H_S J_S) @ Pmat with the output shape matching the input's column layout.

    Raises:
        ValueError: If ``Pmat`` has unsupported dimensionality or zero elements.
    """
    is_vector_input = Pmat.ndim == 1
    if is_vector_input:
        if Pmat.numel() == 0:
            msg = "Pmat must be non-empty."
            raise ValueError(msg)
        num_cols = 1
        Pmat_matrix = Pmat.view(-1, 1)
    elif Pmat.ndim == 2:
        num_cols = Pmat.shape[1]
        Pmat_matrix = Pmat
    else:
        msg = "Pmat must have shape [P] or [P, K]."
        raise ValueError(msg)

    output = torch.zeros_like(Pmat_matrix, device=device, dtype=dtype)

    desc = progress_desc or "jt_h_j_mat"

    if num_cols == 1:
        # Unbatched path.
        direction = unflatten_to_param_dict_like(
            Pmat_matrix[:, 0].to(device=device, dtype=dtype),
            linearized_model.original_params,
            param_keys,
        )
        data_iter = iter_progress(loader, enabled=progressbar, desc=desc)
        for inputs, _ in data_iter:
            inputs_device = inputs.to(device, non_blocking=True)
            grad_dict = jt_h_j_mat_single_body(
                linearized_model=linearized_model,
                inputs_device=inputs_device,
                direction=direction,
            )
            grad_flat = torch.cat([grad_dict[k].flatten() for k in param_keys], dim=0)
            output[:, 0].add_(grad_flat)
    else:
        # Batched path: vmap JVP and VJP over K.
        batched_tangents = matrix_to_batched_param_dict(
            Pmat_matrix, linearized_model.original_params, param_keys
        )  # Leaves: [K, *shape]

        data_iter = iter_progress(loader, enabled=progressbar, desc=desc)
        for inputs, _ in data_iter:
            inputs_device = inputs.to(device, non_blocking=True)

            grads_batched = jt_h_j_mat_batched_body(
                linearized_model=linearized_model,
                inputs_device=inputs_device,
                batched_tangents=batched_tangents,
            )
            output.add_(batched_param_dict_to_matrix(grads_batched, param_keys))

    return output[:, 0] if is_vector_input else output


@torch.compile(fullgraph=True, dynamic=False)
def jtj_mat_batched_body(
    linearized_model: LinearizedModel,
    inputs_device: Tensor,
    batched_tangents: dict,
) -> dict:
    forward = make_forward_fn(
        linearized_model.original_model,
        linearized_model.original_buffers,
        inputs_device,
    )
    _, vjp_fn = vjp(forward, linearized_model.original_params)

    def jvp_single(tangent: dict[str, Tensor], forward=forward) -> Tensor:
        _, Jv = jvp(forward, (linearized_model.original_params,), (tangent,))
        return Jv  # [B, C]

    logit_deltas = vmap(jvp_single, in_dims=0, out_dims=0)(
        batched_tangents
    )  # [K, B, C]
    grads_batched = vmap(
        lambda cot, vjp_fn=vjp_fn: vjp_fn(cot)[0], in_dims=0, out_dims=0
    )(logit_deltas)
    return grads_batched


@torch.compile(fullgraph=True, mode="reduce-overhead", dynamic=False)
def jtj_mat_single_body(
    linearized_model: LinearizedModel, inputs_device: Tensor, direction: dict
) -> dict:
    forward = make_forward_fn(
        linearized_model.original_model,
        linearized_model.original_buffers,
        inputs_device,
    )
    _, vjp_fn = vjp(forward, linearized_model.original_params)

    _, logit_deltas = jvp(
        forward, (linearized_model.original_params,), (direction,)
    )  # [B, C]

    grad_dict = vjp_fn(logit_deltas)[0]

    return grad_dict


def jtj_mat(
    linearized_model: LinearizedModel,
    loader: DataLoader,
    Pmat: Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    param_keys: list[str],
    progressbar: bool = False,
    progress_desc: str | None = None,
) -> Tensor:
    """Compute ``(J_S^T J_S) @ Pmat``.

    Shapes:
        - Input  Pmat: [P]     -> Output: [P]
        - Input  Pmat: [P, K]  -> Output: [P, K]

    Args:
        linearized_model: Linearized model wrapper exposing `original_model`,
            `original_params`, and `original_buffers`.
        loader: Deterministic DataLoader over the set S (defines sample order).
        Pmat: Parameter-space vector or column-stacked matrix.
        device: Torch device for computation.
        dtype: Torch dtype for intermediate tensors and outputs.
        param_keys: Parameter key order for flattening.
        progressbar: Whether to wrap loader iteration with a tqdm progress bar.
        progress_desc: Description used for the progress bar when enabled.

    Returns:
        (J_S^T J_S) @ Pmat with the output shape matching the input's column layout.

    Raises:
        ValueError: If ``Pmat`` has unsupported dimensionality or zero elements.
    """
    is_vector_input = Pmat.ndim == 1
    if is_vector_input:
        if Pmat.numel() == 0:
            msg = "Pmat must be non-empty."
            raise ValueError(msg)
        num_cols = 1
        Pmat_matrix = Pmat.view(-1, 1)
    elif Pmat.ndim == 2:
        num_cols = Pmat.shape[1]
        Pmat_matrix = Pmat
    else:
        msg = "Pmat must have shape [P] or [P, K]."
        raise ValueError(msg)

    output = torch.zeros_like(Pmat_matrix, device=device, dtype=dtype)

    desc = progress_desc or "jtj_mat"

    if num_cols == 1:
        # Unbatched path.
        direction = unflatten_to_param_dict_like(
            Pmat_matrix[:, 0].to(device=device, dtype=dtype),
            linearized_model.original_params,
            param_keys,
        )
        data_iter = iter_progress(loader, enabled=progressbar, desc=desc)
        for inputs, _ in data_iter:
            inputs_device = inputs.to(device, non_blocking=True)
            grad_dict = jtj_mat_single_body(
                linearized_model=linearized_model,
                inputs_device=inputs_device,
                direction=direction,
            )
            grad_flat = torch.cat([grad_dict[k].flatten() for k in param_keys], dim=0)
            output[:, 0].add_(grad_flat)
    else:
        # Batched path: vmap JVP and VJP over K.
        batched_tangents = matrix_to_batched_param_dict(
            Pmat_matrix, linearized_model.original_params, param_keys
        )  # Leaves: [K, *shape]

        data_iter = iter_progress(loader, enabled=progressbar, desc=desc)
        for inputs, _ in data_iter:
            inputs_device = inputs.to(device, non_blocking=True)

            grads_batched = jtj_mat_batched_body(
                linearized_model=linearized_model,
                inputs_device=inputs_device,
                batched_tangents=batched_tangents,
            )
            output.add_(batched_param_dict_to_matrix(grads_batched, param_keys))

    return output[:, 0] if is_vector_input else output


class PsiLeftGramOperator(MatmulMixin):
    """Left-gram operator K_S = Psi_S Sigma Psi_S^T for a set S.

    This operator acts in sample space (dimension N_S = num_classes * |S|).
    The matvec implements:
        s -> J_S^T (H_S^{1/2} s) -> Sigma @ (...) -> H_S^{1/2} (J_S ...).

    Args:
        linearized_model: Model wrapper.
        cov_op: Callable that applies Sigma to a flat parameter vector.
        loader: DataLoader enumerating set S and defining its order.
        num_classes: Number of classes.
        device: Torch device used for computation.
        apply_h_sqrt_kwargs: Extra keyword arguments forwarded to apply_h_sqrt.

    Attributes:
        total_dim: Sample-space dimension N_S = num_classes * |S|.
        device: Torch device used for computation.

    Raises:
        ValueError: If input vector has wrong device or dimension.
    """

    def __init__(
        self,
        *,
        linearized_model: LinearizedModel,
        cov_op: Callable[[Tensor], Tensor],
        loader: DataLoader,
        num_classes: int,
        device: torch.device,
        h_sqrt_mode: str = "eigh",
        apply_h_sqrt_kwargs: dict,
        progressbar: bool = False,
    ) -> None:
        self.linearized_model = linearized_model
        self._cov_op = cov_op
        self._loader = make_deterministic_loader(loader)
        self.num_classes = num_classes
        self.device = device
        self._h_sqrt_mode = h_sqrt_mode
        self._apply_h_sqrt_kwargs = apply_h_sqrt_kwargs
        self._progressbar = progressbar

        self.dtype = next(iter(self.linearized_model.original_params.values())).dtype
        self._param_keys = get_param_keys(self.linearized_model)
        self.total_dim = calculate_total_dim(self._loader, num_classes)
        self.shape = (self.total_dim, self.total_dim)

        self._P = sum(v.numel() for v in self.linearized_model.original_params.values())

    @property
    def is_symmetric(self) -> bool:
        """Whether the operator is symmetric in its action space."""
        return True

    def matmat(self, S: Tensor) -> Tensor:
        """Multiply by K_S.

        Args:
            S: Input vector on self.device.

        Returns:
            Y: Output vector K_S @ s.
        """
        A = psiT_mat(
            linearized_model=self.linearized_model,
            loader=self._loader,
            num_classes=self.num_classes,
            Smat=S,
            h_sqrt_mode=self._h_sqrt_mode,
            apply_h_sqrt_kwargs=self._apply_h_sqrt_kwargs,
            device=self.device,
            dtype=self.dtype,
            param_keys=self._param_keys,
            P=self._P,
            progressbar=self._progressbar,
            progress_desc=self._progress_desc("psiT_mat"),
        )  # [P, K]
        B = self._cov_op @ A
        Y = psi_mat(
            linearized_model=self.linearized_model,
            loader=self._loader,
            num_classes=self.num_classes,
            Pmat=B,
            h_sqrt_mode=self._h_sqrt_mode,
            apply_h_sqrt_kwargs=self._apply_h_sqrt_kwargs,
            device=self.device,
            dtype=self.dtype,
            param_keys=self._param_keys,
            progressbar=self._progressbar,
            progress_desc=self._progress_desc("psi_mat"),
        )  # [N_S, K]

        return Y

    def rmatmat(self, S: Tensor) -> Tensor:
        # Hermitian: left multiply equals right multiply
        if S.ndim == 1:
            return self.matmat(S)

        return self.matmat(S.T).T


class PsiCrossLeftGramOperator(MatmulMixin):
    """Sample-space cross left-gram operator.

    This operator forms either C C^T on the R-side (outside="R") or C^T C on
    the F-side (outside="F"), where C = Psi_R Sigma Psi_F^T. We always apply
    the operator on the smaller outside to reduce ARPACK memory.

    Args:
        linearized_model: Model wrapper with original_* fields.
        cov_op: Callable that applies Sigma to a flat parameter vector.
        retained_loader: DataLoader for set R.
        forgotten_loader: DataLoader for set F.
        outside: Literal "R" or "F" to choose the smaller side.
        num_classes: Number of classes.
        device: Torch device used for computation.
        apply_h_sqrt_kwargs: Extra keyword arguments forwarded to apply_h_sqrt.

    Attributes:
        total_dim: Dimension of the chosen outside side.
        device: Torch device.

    Raises:
        ValueError: If input vector has wrong device or dimension.
    """

    def __init__(
        self,
        *,
        linearized_model: LinearizedModel,
        cov_op: Callable[[Tensor], Tensor],  # Sigma MVP
        retained_loader: DataLoader,
        forgotten_loader: DataLoader,
        outside: Literal["R", "F"],
        num_classes: int,
        device: torch.device,
        h_sqrt_mode: str = "eigh",
        apply_h_sqrt_kwargs: dict,
        progressbar: bool = False,
    ) -> None:
        self.linearized_model = linearized_model
        self.device = device
        self._cov_op = cov_op
        # Reconstruct loaders because the ones passed in might not preserve iteration
        # order
        retained_loader = make_deterministic_loader(retained_loader)
        forgotten_loader = make_deterministic_loader(forgotten_loader)
        self._outside_loader = retained_loader if outside == "R" else forgotten_loader
        self._inside_loader = retained_loader if outside == "F" else forgotten_loader
        self._num_classes = num_classes
        self._h_sqrt_mode = h_sqrt_mode
        self._apply_h_sqrt_kwargs = apply_h_sqrt_kwargs
        self._progressbar = progressbar
        self._outside = outside

        self.dtype = next(iter(self.linearized_model.original_params.values())).dtype
        self._param_keys = get_param_keys(self.linearized_model)
        self._P = sum(v.numel() for v in self.linearized_model.original_params.values())

        dim_R = calculate_total_dim(retained_loader, num_classes)
        dim_F = calculate_total_dim(forgotten_loader, num_classes)
        self.total_dim = dim_R if outside == "R" else dim_F
        self.shape = (self.total_dim, self.total_dim)

    @property
    def is_symmetric(self) -> bool:
        """Whether the operator is symmetric in its action space."""
        return True

    def matmat(self, S: Tensor) -> Tensor:
        """Multiply by the selected cross left-gram.

        Args:
            S: Input vector on the chosen outside side.

        Returns:
            Y: Output vector.
        """
        inside_label = "F" if self._outside == "R" else "R"

        A = psiT_mat(
            linearized_model=self.linearized_model,
            loader=self._outside_loader,
            num_classes=self._num_classes,
            Smat=S,
            h_sqrt_mode=self._h_sqrt_mode,
            apply_h_sqrt_kwargs=self._apply_h_sqrt_kwargs,
            device=self.device,
            dtype=self.dtype,
            param_keys=self._param_keys,
            P=self._P,
            progressbar=self._progressbar,
            progress_desc=self._progress_desc(f"psiT_mat_{self._outside}"),
        )  # [P, K]
        B = self._cov_op @ A  # [P, K]
        C = jt_h_j_mat(
            linearized_model=self.linearized_model,
            loader=self._inside_loader,
            Pmat=B,
            device=self.device,
            dtype=self.dtype,
            param_keys=self._param_keys,
            progressbar=self._progressbar,
            progress_desc=self._progress_desc(f"jt_h_j_mat_{inside_label}"),
        )  # [P,K]
        D = self._cov_op @ C  # [P, K]
        Y = psi_mat(
            linearized_model=self.linearized_model,
            loader=self._outside_loader,
            num_classes=self._num_classes,
            Pmat=D,
            h_sqrt_mode=self._h_sqrt_mode,
            apply_h_sqrt_kwargs=self._apply_h_sqrt_kwargs,
            device=self.device,
            dtype=self.dtype,
            param_keys=self._param_keys,
            progressbar=self._progressbar,
            progress_desc=self._progress_desc(f"psi_mat_{self._outside}"),
        )  # [N_out, K]
        return Y

    def rmatmat(self, S: Tensor) -> Tensor:
        # Hermitian: left multiply equals right multiply
        if S.ndim == 1:
            return self.matmat(S)

        return self.matmat(S.T).T


class PsiRightGramOperator(MatmulMixin):
    """Right-gram operator G_S = Sigma^{1/2} Psi_S^T Psi_S Sigma^{1/2} for a set S.

    This operator acts in parameter space (dimension P). The matvec implements:
        v -> u = Sigma^{1/2} v
          -> z = J_S u
          -> h = H_S z
          -> t = J_S^T h
          -> y = Sigma^{1/2} t.

    Args:
        linearized_model: Model wrapper with original_* fields.
        scale_op: Callable that applies Sigma^{1/2} to flat parameter vectors.
        loader: DataLoader enumerating set S and defining its order.
        num_classes: Number of classes.
        device: Torch device used for computation.

    Attributes:
        total_dim: Parameter-space dimension P.
        device: Torch device.

    Raises:
        ValueError: If input vector has wrong device or dimension.
    """

    def __init__(
        self,
        *,
        linearized_model: LinearizedModel,
        scale_op: Callable[[Tensor], Tensor],
        loader: DataLoader,
        num_classes: int,
        device: torch.device,
        progressbar: bool = False,
    ) -> None:
        self.linearized_model = linearized_model
        self.device = device
        self._scale_op = scale_op
        self._loader = loader
        self._num_classes = num_classes
        self._progressbar = progressbar

        self.dtype = next(iter(self.linearized_model.original_params.values())).dtype
        self._param_keys = get_param_keys(self.linearized_model)
        self.total_dim = sum(
            v.numel() for v in self.linearized_model.original_params.values()
        )
        self.shape = (self.total_dim, self.total_dim)

    @property
    def is_symmetric(self) -> bool:
        """Whether the operator is symmetric in its action space."""
        return True

    def matmat(self, V: Tensor) -> Tensor:
        """Multiply by G_S.

        Args:
            V: Input vector on self.device.

        Returns:
            Y: Output vector G_S @ v.
        """
        # 1) u = Sigma^{1/2} v
        U = self._scale_op @ V  # [P, K]

        # 2) accumulate t = J_S^T (H_S (J_S u))
        T = jt_h_j_mat(
            linearized_model=self.linearized_model,
            loader=self._loader,
            Pmat=U,
            device=self.device,
            dtype=self.dtype,
            param_keys=self._param_keys,
            progressbar=self._progressbar,
            progress_desc=self._progress_desc("jt_h_j_mat"),
        )  # [P, K]

        # 3) y = Sigma^{1/2} t
        Y = self._scale_op @ T  # [P, K]
        return Y

    def rmatmat(self, V: Tensor) -> Tensor:
        # Hermitian: left multiply equals right multiply
        if V.ndim == 1:
            return self.matmat(V)

        return self.matmat(V.T).T


# Parameter-space right-gram operator without H-whitening:
# G = Sigma^{1/2} J_S^T J_S Sigma^{1/2}
class PhiRightGramOperator(MatmulMixin):
    """Right-gram operator for Phi_S = J_S without H whitening.

    This operator acts in parameter space (dimension P). The matvec implements:
        v -> u = Sigma^{1/2} v
          -> z = J_S u
          -> t = J_S^T z
          -> y = Sigma^{1/2} t.

    Its largest eigenvalue equals ||Phi_S Sigma^{1/2}||_2^2.

    Args:
        linearized_model: Model wrapper with original_* fields.
        scale_op: Callable that applies Sigma^{1/2} to flat parameter vectors.
        loader: DataLoader enumerating set S and defining its order.
        num_classes: Number of classes.
        device: Torch device used for computation.
    """

    def __init__(
        self,
        *,
        linearized_model: LinearizedModel,
        scale_op: Callable[[Tensor], Tensor],
        loader: DataLoader,
        num_classes: int,
        device: torch.device,
        progressbar: bool = False,
    ) -> None:
        self.linearized_model = linearized_model
        self.device = device
        self._scale_op = scale_op
        self._loader = make_deterministic_loader(loader)
        self._num_classes = num_classes
        self._progressbar = progressbar

        self.dtype = next(iter(self.linearized_model.original_params.values())).dtype
        self._param_keys = get_param_keys(self.linearized_model)
        self.total_dim = sum(
            v.numel() for v in self.linearized_model.original_params.values()
        )
        self.shape = (self.total_dim, self.total_dim)

    @property
    def is_symmetric(self) -> bool:
        """Whether the operator is symmetric in its action space."""
        return True

    def matmat(self, V: Tensor) -> Tensor:
        # 1) u = Sigma^{1/2} v
        U = self._scale_op @ V  # [P, K]

        # 2) accumulate t = J_S^T (J_S u)
        T = jtj_mat(
            linearized_model=self.linearized_model,
            loader=self._loader,
            Pmat=U,
            device=self.device,
            dtype=self.dtype,
            param_keys=self._param_keys,
            progressbar=self._progressbar,
            progress_desc=self._progress_desc("jtj_mat"),
        )  # [P,K]

        # 3) y = Sigma^{1/2} t
        Y = self._scale_op @ T  # [P, K]
        return Y

    def rmatmat(self, V: Tensor) -> Tensor:
        # Hermitian: left multiply equals right multiply
        if V.ndim == 1:
            return self.matmat(V)

        return self.matmat(V.T).T


class PsiCrossRightGramOperator(MatmulMixin):
    """Parameter-space cross right-gram operator.

    This operator acts in parameter space (dimension P). It builds the
    parameter-space operator
        G = Sigma^{1/2} Psi_R^T Psi_R Sigma Psi_F^T Psi_F Sigma^{1/2},
    whose largest eigenvalue equals sigma_max(Psi_R Sigma Psi_F^T)^2.

    Args:
        linearized_model: Model wrapper with original_* fields.
        scale_op: Callable that applies Sigma^{1/2} to a flat parameter vector.
        cov_op: Callable that applies Sigma to a flat parameter vector.
        retained_loader: DataLoader for set R.
        forgotten_loader: DataLoader for set F.
        num_classes: Number of classes.
        device: Torch device used for computation.
        apply_h_sqrt_kwargs: Extra keyword arguments forwarded to apply_h_sqrt.

    Attributes:
        total_dim: Parameter-space dimension P.
        device: Torch device.

    Raises:
        ValueError: If input vector has wrong device or dimension.
    """

    def __init__(
        self,
        *,
        linearized_model: LinearizedModel,
        scale_op: Callable[[Tensor], Tensor],
        cov_op: Callable[[Tensor], Tensor],
        retained_loader: DataLoader,
        forgotten_loader: DataLoader,
        num_classes: int,
        device: torch.device,
        h_sqrt_mode: str = "eigh",
        apply_h_sqrt_kwargs: dict,
        progressbar: bool = False,
    ) -> None:
        self.linearized_model = linearized_model
        self.device = device
        self._scale_op = scale_op
        self._cov_op = cov_op
        # Reconstruct loaders because the ones passed in might not preserve iteration
        # order
        self._R_loader = make_deterministic_loader(retained_loader)
        self._F_loader = make_deterministic_loader(forgotten_loader)
        self._num_classes = num_classes
        self._h_sqrt_mode = h_sqrt_mode
        self._apply_h_sqrt_kwargs = apply_h_sqrt_kwargs
        self._progressbar = progressbar

        self.dtype = next(iter(self.linearized_model.original_params.values())).dtype
        self._param_keys = get_param_keys(self.linearized_model)
        self.total_dim = sum(
            v.numel() for v in self.linearized_model.original_params.values()
        )
        self.shape = (self.total_dim, self.total_dim)

    @property
    def is_symmetric(self) -> bool:
        """Whether the operator is symmetric in its action space."""
        return False

    def matmat(self, V: Tensor) -> Tensor:
        # u0 = Sigma^{1/2} v
        U0 = self._scale_op @ V  # [P, K]

        # hR = J_R^T H_R J_R u0
        hR = jt_h_j_mat(
            linearized_model=self.linearized_model,
            loader=self._R_loader,
            Pmat=U0,
            device=self.device,
            dtype=self.dtype,
            param_keys=self._param_keys,
            progressbar=self._progressbar,
            progress_desc=self._progress_desc("jt_h_j_mat_R"),
        )  # [P, K]

        # t  = Sigma hR
        T = self._cov_op @ hR  # [P, K]

        # hF = J_F^T H_F J_F t
        hF = jt_h_j_mat(
            linearized_model=self.linearized_model,
            loader=self._F_loader,
            Pmat=T,
            device=self.device,
            dtype=self.dtype,
            param_keys=self._param_keys,
            progressbar=self._progressbar,
            progress_desc=self._progress_desc("jt_h_j_mat_F"),
        )  # [P, K]

        # y  = Sigma^{1/2} hF
        Y = self._scale_op @ hF  # [P, K]

        return Y

    def rmatmat(self, V: Tensor) -> Tensor:
        V = V.T  # [P, K]

        # u0 = Sigma^{1/2} v
        U0 = self._scale_op @ V  # [P, K]

        # hF = J_F^T H_F J_F u0
        hF = jt_h_j_mat(
            linearized_model=self.linearized_model,
            loader=self._F_loader,
            Pmat=U0,
            device=self.device,
            dtype=self.dtype,
            param_keys=self._param_keys,
            progressbar=self._progressbar,
            progress_desc=self._progress_desc("jt_h_j_mat_F"),
        )  # [P, K]

        # t  = Sigma hF
        T = self._cov_op @ hF  # [P, K]

        # hR = J_R^T H_R J_R t
        hR = jt_h_j_mat(
            linearized_model=self.linearized_model,
            loader=self._R_loader,
            Pmat=T,
            device=self.device,
            dtype=self.dtype,
            param_keys=self._param_keys,
            progressbar=self._progressbar,
            progress_desc=self._progress_desc("jt_h_j_mat_R"),
        )  # [P, K]

        # y  = Sigma^{1/2} hR
        y = self._scale_op @ hR  # [P, K]

        return y.T


class PsiCrossOperator(MatmulMixin):
    """Rectangular operator C := Psi_R Sigma Psi_F^T with left/right @ support.

    shape = (C|R|, C|F|). Right multiply: C @ x_F -> y_R. Left multiply: X_R @ C -> y_F.
    """

    def __init__(
        self,
        *,
        linearized_model: LinearizedModel,
        cov_op: Callable[[Tensor], Tensor],
        retained_loader: DataLoader,
        forgotten_loader: DataLoader,
        num_classes: int,
        device: torch.device,
        h_sqrt_mode: str = "eigh",
        apply_h_sqrt_kwargs: dict,
        progressbar: bool = False,
    ) -> None:
        self.linearized_model = linearized_model
        self.device = device
        self._cov_op = cov_op
        self._R_loader = make_deterministic_loader(retained_loader)
        self._F_loader = make_deterministic_loader(forgotten_loader)
        self._num_classes = num_classes
        self._h_sqrt_mode = h_sqrt_mode
        self._apply_h_sqrt_kwargs = apply_h_sqrt_kwargs
        self._progressbar = progressbar
        self.dtype = next(iter(self.linearized_model.original_params.values())).dtype
        self._param_keys = get_param_keys(self.linearized_model)
        self._P = sum(v.numel() for v in self.linearized_model.original_params.values())

        dim_R = calculate_total_dim(self._R_loader, num_classes)
        dim_F = calculate_total_dim(self._F_loader, num_classes)
        self.shape = (dim_R, dim_F)  # Rectangular linop

    @property
    def is_symmetric(self) -> bool:
        return False

    def matmat(self, X_F: Tensor) -> Tensor:
        A = psiT_mat(
            linearized_model=self.linearized_model,
            loader=self._F_loader,
            num_classes=self._num_classes,
            Smat=X_F,
            h_sqrt_mode=self._h_sqrt_mode,
            apply_h_sqrt_kwargs=self._apply_h_sqrt_kwargs,
            device=self.device,
            dtype=self.dtype,
            param_keys=self._param_keys,
            P=self._P,
            progressbar=self._progressbar,
            progress_desc=self._progress_desc("psiT_mat_F"),
        )  # [P, K]
        B = self._cov_op @ A
        Y = psi_mat(
            linearized_model=self.linearized_model,
            loader=self._R_loader,
            num_classes=self._num_classes,
            Pmat=B,
            h_sqrt_mode=self._h_sqrt_mode,
            apply_h_sqrt_kwargs=self._apply_h_sqrt_kwargs,
            device=self.device,
            dtype=self.dtype,
            param_keys=self._param_keys,
            progressbar=self._progressbar,
            progress_desc=self._progress_desc("psi_mat_R"),
        )  # [dim_R, K]
        return Y

    def rmatmat(self, Y_R: Tensor) -> Tensor:
        # Compute (X_R @ C) = (Psi_F Sigma Psi_R^T)^T @ X_R^T (implemented directly)
        A = psiT_mat(
            linearized_model=self.linearized_model,
            loader=self._R_loader,
            num_classes=self._num_classes,
            Smat=Y_R.T,
            h_sqrt_mode=self._h_sqrt_mode,
            apply_h_sqrt_kwargs=self._apply_h_sqrt_kwargs,
            device=self.device,
            dtype=self.dtype,
            param_keys=self._param_keys,
            P=self._P,
            progressbar=self._progressbar,
            progress_desc=self._progress_desc("psiT_mat_R"),
        )  # [P, K]
        B = self._cov_op @ A
        X_R = psi_mat(
            linearized_model=self.linearized_model,
            loader=self._F_loader,
            num_classes=self._num_classes,
            Pmat=B,
            h_sqrt_mode=self._h_sqrt_mode,
            apply_h_sqrt_kwargs=self._apply_h_sqrt_kwargs,
            device=self.device,
            dtype=self.dtype,
            param_keys=self._param_keys,
            progressbar=self._progressbar,
            progress_desc=self._progress_desc("psi_mat_F"),
        )  # [dim_F, K]
        return X_R.T  # [K, dim_F]


class PsiCrossParamGramOperator(MatmulMixin):
    """Hermitian P x P cross operator.

    Implements H = A_R^T A_F A_F^T A_R with A_S := Psi_S Sigma^{1/2}.

    Top eigenvalue satisfies: lambda_max(H) = sigma_max(Psi_R Sigma Psi_F^T)^2.
    This stays in parameter space (shape [P, P]), so Skerch.seigh scales at ImageNet.

    One matvec y = H v performs (in order):
      tR = A_R v = Psi_R (Sigma^{1/2} v)  [1 Pass(R)]
      pF = A_F^T tR = Psi_F^T tR  [1 Pass(F)]
      tF = A_F pF = Psi_F (Sigma^{1/2} pF)  [1 Pass(F)]
      y  = A_R^T tF = Psi_R^T tF  [1 Pass(R)]
    This requires 2 passes over R and 2 passes over F per matvec (exact).
    """

    def __init__(
        self,
        *,
        linearized_model: LinearizedModel,
        scale_op: Callable[[Tensor], Tensor],
        retained_loader: DataLoader,
        forgotten_loader: DataLoader,
        num_classes: int,
        device: torch.device,
        h_sqrt_mode: str = "eigh",
        apply_h_sqrt_kwargs: dict,
        progressbar: bool = False,
    ) -> None:
        self.linearized_model = linearized_model
        self.device = device
        self._scale_op = scale_op
        self._R_loader = make_deterministic_loader(retained_loader)
        self._F_loader = make_deterministic_loader(forgotten_loader)
        self._num_classes = num_classes
        self._h_sqrt_mode = h_sqrt_mode
        self._apply_h_sqrt_kwargs = apply_h_sqrt_kwargs
        self._progressbar = progressbar

        self.dtype = next(iter(self.linearized_model.original_params.values())).dtype
        self._param_keys = get_param_keys(self.linearized_model)
        self.total_dim = sum(
            v.numel() for v in self.linearized_model.original_params.values()
        )
        self.shape = (self.total_dim, self.total_dim)

    @property
    def is_symmetric(self) -> bool:
        return True

    def matmat(self, V: Tensor) -> Tensor:
        # tR = A_R v = Psi_R (Sigma^{1/2} v)
        tR = psi_mat(
            linearized_model=self.linearized_model,
            loader=self._R_loader,
            num_classes=self._num_classes,
            Pmat=self._scale_op @ V,
            h_sqrt_mode=self._h_sqrt_mode,
            apply_h_sqrt_kwargs=self._apply_h_sqrt_kwargs,
            device=self.device,
            dtype=self.dtype,
            param_keys=self._param_keys,
            progressbar=self._progressbar,
            progress_desc=self._progress_desc("psi_mat_R"),
        )  # [|R|*C, K]
        # pF = A_F^T tR = Psi_F^T tR
        pF = psiT_mat(
            linearized_model=self.linearized_model,
            loader=self._F_loader,
            num_classes=self._num_classes,
            Smat=tR,
            h_sqrt_mode=self._h_sqrt_mode,
            apply_h_sqrt_kwargs=self._apply_h_sqrt_kwargs,
            device=self.device,
            dtype=self.dtype,
            param_keys=self._param_keys,
            P=self.shape[0],
            progressbar=self._progressbar,
            progress_desc=self._progress_desc("psiT_mat_F"),
        )  # [P, K]
        # tF = A_F pF = Psi_F (Sigma^{1/2} pF)
        tF = psi_mat(
            linearized_model=self.linearized_model,
            loader=self._F_loader,
            num_classes=self._num_classes,
            Pmat=self._scale_op @ pF,
            h_sqrt_mode=self._h_sqrt_mode,
            apply_h_sqrt_kwargs=self._apply_h_sqrt_kwargs,
            device=self.device,
            dtype=self.dtype,
            param_keys=self._param_keys,
            progressbar=self._progressbar,
            progress_desc=self._progress_desc("psi_mat_F"),
        )  # [|F|*C, K]
        # y  = A_R^T tF = Psi_R^T tF
        Y = psiT_mat(
            linearized_model=self.linearized_model,
            loader=self._R_loader,
            num_classes=self._num_classes,
            Smat=tF,
            h_sqrt_mode=self._h_sqrt_mode,
            apply_h_sqrt_kwargs=self._apply_h_sqrt_kwargs,
            device=self.device,
            dtype=self.dtype,
            param_keys=self._param_keys,
            P=self.shape[0],
            progressbar=self._progressbar,
            progress_desc=self._progress_desc("psiT_mat_R"),
        )  # [P, K]
        return Y

    def rmatmat(self, V: Tensor) -> Tensor:
        # Hermitian: left multiply equals right multiply
        if V.ndim == 1:
            return self.matmat(V)

        return self.matmat(V.T).T


class GGNLinearOperator(MatmulMixin):
    r"""Matrix-free GGN/Fisher linear operator for cross-entropy (class axis = -1).

    Computes
        G = c * sum_n J_n^T H(p_n) J_n,
    where H(p) u = diag(p) u - p (p^T u),  with p = softmax(logits) and
    c = 1/N for reduction='mean', else 1 for 'sum'.

    Input to `__matmul__`:
      - flat vector v: shape [P]
      - flat matrix V: shape [P, K]  (K column vectors)

    Output mirrors input shape.

    Parameters are handled as a dict (same ordering as
    get_param_keys(self.linearized_model)).
    JVPs and VJPs are batched across K with `vmap`.

    Notes:
      * Uses self.linearized_model.original_model / original_params / original_buffers.
      * Uses loaders with deterministic iteration order (no side effects).
      * Assumes logits are returned directly by the model (no softmax inside).
    """

    def __init__(
        self,
        model: nn.Module,
        loader: DataLoader | PrefetchLoader,
        *,
        loss: Literal["ce", "mse"] = "ce",
        reduction: Literal["mean", "sum"] = "mean",
        progressbar: bool = False,
    ) -> None:
        if reduction not in {"mean", "sum"}:
            msg = "`reduction` must be 'mean' or 'sum'."
            raise ValueError(msg)

        self._model = model
        self._loader = loader
        self._loss = loss
        self._reduction = reduction
        self._progressbar = progressbar

        self._param_dict = {
            name: p for name, p in model.named_parameters() if p.requires_grad
        }
        self._buffer_dict = dict(model.named_buffers())

        self.device = next(iter(self._param_dict.values())).device
        self.dtype = next(iter(self._param_dict.values())).dtype

        self._param_keys = list(self._param_dict.keys())
        self._flat_sizes = [self._param_dict[k].numel() for k in self._param_keys]
        self._total_dim = sum(self._flat_sizes)
        self.shape = (self._total_dim, self._total_dim)

        self._num_data = calculate_num_samples(self._loader)

        self._matmat_body_single = torch.compile(
            self._matmat_body_single,
            fullgraph=True,
            mode="reduce-overhead",
            dynamic=False,
        )
        self._matmat_body_batched = torch.compile(
            self._matmat_body_batched,
            fullgraph=True,
            dynamic=False,
        )

    @property
    def is_symmetric(self) -> bool:
        return True

    def matmat(self, V: Tensor) -> Tensor:
        if V.ndim == 1:
            return self._matmat(V.view(self._total_dim, 1)).flatten()

        return self._matmat(V)

    def rmatmat(self, V: Tensor) -> Tensor:
        if V.ndim == 1:
            return self._matmat(V.view(self._total_dim, 1)).flatten()

        return self._matmat(V.T).T

    # Utility: split/reshape a [P, K] matrix into a *batched* pytree matching params
    # Leaves become [K, *param_shape] (K front) to feed `vmap` over tangents.
    def matrix_to_batched_param_dict(self, matrix: Tensor, K: int) -> dict[str, Tensor]:
        chunks = list(matrix.split(self._flat_sizes, dim=0))  # List of [p_i, K]
        batched_param_dict = {}
        for key, chunk in zip(self._param_keys, chunks, strict=True):
            shape = self._param_dict[key].shape
            # Reshape to [*param_shape, K]
            # then bring K to front -> [K, *param_shape]
            batched_param = chunk.view(*shape, K).movedim(-1, 0).contiguous()
            batched_param_dict[key] = batched_param
        return batched_param_dict

    # Utility: flatten a *batched* param dict with leaves [K, *param_shape]
    # back to a flat [P, K] with columns.
    def batched_param_dict_to_matrix(
        self,
        batched_param_dict: dict[str, Tensor],
        K: int,
    ) -> Tensor:
        matrix_blocks = []
        for key in self._param_keys:
            batched_param = batched_param_dict[key]  # [K, *param_shape]
            batched_param = batched_param.reshape(K, -1)  # [K, p_i]
            batched_param = batched_param.T  # [p_i, K]   (move K back to last dim)
            matrix_blocks.append(batched_param)
        return torch.cat(matrix_blocks, dim=0)  # [P, K]

    def _matmat(self, V: Tensor) -> Tensor:
        """Core computation. V: [P, K] with K>=1.

        Returns:
            Tensor shaped `[P, K]`.
        """
        P, K = V.shape

        # Allocate output accumulator
        out = torch.zeros(P, K, device=self.device, dtype=self.dtype)

        # Pre-batch the input tangents as pytree
        tangents_batched = self.matrix_to_batched_param_dict(V, K)

        # Iterate data once; use tqdm if requested
        data_iter = self._loader
        if self._progressbar:
            data_iter = tqdm(
                data_iter, desc=f"{self.__class__.__name__} ({self.device!s})"
            )

        alpha = 1.0 if self._reduction == "sum" else 1.0 / self._num_data

        for X, _ in data_iter:
            X = X.to(self.device, non_blocking=True)
            if K == 1:
                res = self._matmat_body_single(X, tangents_batched)
            else:
                res = self._matmat_body_batched(X, tangents_batched, K)

            res = res.detach()
            out.add_(res, alpha=alpha)

        return out

    def _matmat_body_single(self, X: Tensor, tangents_batched: dict) -> Tensor:
        # f(params_dict) -> logits [B, C]
        fwd = make_forward_fn(self._model, self._buffer_dict, X)

        # Evaluate logits and obtain VJP closure at current (frozen) params
        logits, vjp_fn = vjp(fwd, self._param_dict)  # logits: [B, C]
        probs = logits.softmax(dim=-1)  # [B, C]

        # Scalar (non-vmap) path
        # Build tangent dict (remove leading K)
        tangent = {k: v[0] for k, v in tangents_batched.items()}
        _, Jv = jvp(fwd, (self._param_dict,), (tangent,))  # [B, C]

        HJv = h_matmat(probs, Jv) if self._loss == "ce" else Jv  # [B, C]

        # Pull back
        JtHJv_dict = vjp_fn(HJv)[0]  # Dict of param-shaped

        # Flatten to [P, 1] and accumulate
        res = torch.cat([JtHJv_dict[k].reshape(-1, 1) for k in self._param_keys], dim=0)

        return res

    def _matmat_body_batched(self, X: Tensor, tangents_batched: dict, K: int) -> Tensor:
        # f(params_dict) -> logits [B, C]
        fwd = make_forward_fn(self._model, self._buffer_dict, X)

        # Evaluate logits and obtain VJP closure at current (frozen) params
        logits, vjp_fn = vjp(fwd, self._param_dict)  # logits: [B, C]
        probs = logits.softmax(dim=-1)  # [B, C]

        # K>1: vectorized JVPs
        # Define a function that maps a single tangent pytree -> delta logits
        def jvp_single(tangent: dict[str, Tensor], *, fwd_fn=fwd) -> Tensor:
            _, Jv = jvp(fwd_fn, (self._param_dict,), (tangent,))  # [B, C]
            return Jv

        # Map over K (leading axis of each leaf in tangents_batched)
        JV = vmap(jvp_single, in_dims=0, out_dims=0)(tangents_batched)  # [K, B, C]

        if self._loss == "ce":
            # Apply H along class dim (-1), broadcasting over K
            dot = (JV * probs).sum(dim=-1, keepdim=True)  # [K, B, 1]
            HJV = probs * (JV - dot)  # [K, B, C]
        else:
            HJV = JV  # MSE has an identity Hessian

        # Pull back each K-cotangent via vjp; vmap over K
        grads_batched_dict = vmap(
            lambda cot: vjp_fn(cot)[0],
            in_dims=0,
            out_dims=0,
        )(HJV)  # Leaves: [K, *param_shape]

        # Flatten to [P, K] and accumulate
        res = self.batched_param_dict_to_matrix(grads_batched_dict, K)  # [P, K]

        return res
