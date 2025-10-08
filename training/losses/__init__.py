"""Loss package exposing registry, manager, and built-in components."""

from .registry import LOSS_REGISTRY
from .manager import LossManager, build_loss

# Import side effects to populate the registry
from . import basic_loss as _basic_loss  # noqa: F401
from . import gan_loss as _gan_loss  # noqa: F401

__all__ = ["LOSS_REGISTRY", "LossManager", "build_loss"]
