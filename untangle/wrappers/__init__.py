"""Minimal Laplace-focused wrapper implementations.

This module contains only the Laplace approximation wrappers and base classes
needed for analyzing Laplace with subnetworks. Other uncertainty methods from
the full untangle repo have been removed to reduce dependencies.
"""

__all__ = [
    "AdaptedLaplaceWrapper",
    "CEBaselineWrapper",
    "DirichletWrapper",
    "DistributionalWrapper",
    "DualLaplaceWrapper",
    "ModelWrapper",
    "SpecialWrapper",
]

from .adapted_laplace_wrapper import AdaptedLaplaceWrapper
from .ce_baseline_wrapper import CEBaselineWrapper
from .dual_laplace_wrapper import DualLaplaceWrapper
from .model_wrapper import (
    DirichletWrapper,
    DistributionalWrapper,
    ModelWrapper,
    SpecialWrapper,
)
