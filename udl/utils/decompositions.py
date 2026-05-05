from typing import Any


def select_skerch_measurements(
    *,
    total_dim: int,
    section: str,
    skerch_kwargs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Extract and clamp Skerch kwargs for a specific section.

    Returns:
        The clamped Skerch kwargs.
    """
    kwargs = skerch_kwargs[section].copy()
    cap = total_dim

    if section == "norm":
        kwargs["num_meas"] = min(kwargs["num_meas"], cap)
        kwargs["meas_blocksize"] = min(kwargs["meas_blocksize"], kwargs["num_meas"])
        kwargs["norm_types"] = ("op",)
        return kwargs

    kwargs["outer_dims"] = min(kwargs["outer_dims"], cap)
    recovery = kwargs["recovery_type"]
    if recovery.startswith("oversampled_"):
        kwargs["outer_dims"] = min(kwargs["outer_dims"], cap - 1)
        inner = int(recovery.split("_", maxsplit=1)[1])
        inner = max(inner, kwargs["outer_dims"] + 1)
        inner = min(inner, cap)
        kwargs["recovery_type"] = f"oversampled_{inner}"
    kwargs["meas_blocksize"] = min(kwargs["meas_blocksize"], kwargs["outer_dims"])

    return kwargs
