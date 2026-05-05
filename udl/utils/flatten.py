"""Parameter flattening and unflattening utilities."""

import torch
from torch import Tensor

from bul.models.linearized import LinearizedModel


def get_param_keys(linearized_model: LinearizedModel) -> list[str]:
    return list(linearized_model.original_params.keys())


def flatten_param_dict(params: dict[str, Tensor], param_keys: list[str]) -> Tensor:
    flats = [params[k].flatten() for k in param_keys]
    return torch.cat(flats, dim=0)


def unflatten_to_param_dict_like(
    vec: Tensor,
    like_params: dict[str, Tensor],
    param_keys: list[str],
) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    i = 0

    for k in param_keys:
        t = like_params[k]
        n = t.numel()
        out[k] = vec[i : i + n].reshape(t.shape)
        i += n

    if i != vec.numel():
        msg = f"Vector dimension mismatch: expected {vec.numel()}, got {i}"
        raise ValueError(msg)

    return out
