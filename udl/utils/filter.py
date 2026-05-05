"""Parameter filtering utilities."""

import logging
import re

from torch import nn

logger = logging.getLogger(__name__)


def disable_grads_by_regex(model: nn.Module, regex: str | None) -> None:
    """Set requires_grad=False for parameters whose names match the regex."""
    if not regex:
        return

    pattern = re.compile(regex)
    disabled = []
    for name, p in model.named_parameters():
        if pattern.match(name):
            p.requires_grad = False
            disabled.append(name)

    logger.info(
        "Disabled gradients for %d parameter(s) matching /%s/: %s",
        len(disabled),
        regex,
        ", ".join(disabled) if disabled else "{}",
    )


def enable_grads_by_regex(model: nn.Module, regex: str | None) -> None:
    """Set requires_grad=True for parameters whose names match the regex."""
    if not regex:
        return

    for p in model.parameters():
        p.requires_grad = False

    pattern = re.compile(regex)
    enabled = []
    for name, p in model.named_parameters():
        if pattern.match(name):
            p.requires_grad = True
            enabled.append(name)

    logger.info(
        "Enabled gradients for %d parameter(s) matching /%s/: %s",
        len(enabled),
        regex,
        ", ".join(enabled) if enabled else "{}",
    )
