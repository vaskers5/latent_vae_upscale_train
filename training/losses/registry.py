"""Lightweight registry pattern used to track available loss modules.

This mirrors the minimal functionality required by the losses copied from
`new_folder`, allowing them to use ``@LOSS_REGISTRY.register()`` in the same
fashion as the original BasicSR implementation.
"""

from __future__ import annotations

from typing import Dict, Iterator, MutableMapping, Optional, TypeVar, Callable, Any

ModuleType = TypeVar("ModuleType")


class Registry(MutableMapping[str, ModuleType]):
    """Simple name -> module registry with decorator support."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._modules: Dict[str, ModuleType] = {}

    # -- registration helpers -------------------------------------------------
    def register(
        self,
        module: Optional[ModuleType] = None,
        *,
        name: Optional[str] = None,
        override: bool = False,
    ) -> Callable[[ModuleType], ModuleType] | ModuleType:
        """Register ``module`` under ``name`` (defaults to module.__name__)."""

        def _do_register(target: ModuleType) -> ModuleType:
            key = name or getattr(target, "__name__", None)
            if not key:
                raise ValueError("Registered object must define __name__ or provide an explicit name.")
            if not override and key in self._modules:
                raise KeyError(f"{key!r} is already registered in {self.name}.")
            self._modules[key] = target
            return target

        if module is None:
            return _do_register
        return _do_register(module)

    # -- mapping interface ----------------------------------------------------
    def __getitem__(self, key: str) -> ModuleType:
        return self._modules[key]

    def __setitem__(self, key: str, value: ModuleType) -> None:
        self._modules[key] = value

    def __delitem__(self, key: str) -> None:
        del self._modules[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._modules)

    def __len__(self) -> int:
        return len(self._modules)

    # -- convenience accessors ------------------------------------------------
    def get(self, key: str) -> ModuleType:
        if key not in self._modules:
            raise KeyError(f"{key!r} is not registered in {self.name}.")
        return self._modules[key]

    def build(self, key: str, *args: Any, **kwargs: Any) -> ModuleType:
        cls = self.get(key)
        return cls(*args, **kwargs)  # type: ignore[call-arg]


LOSS_REGISTRY: Registry = Registry("loss")

__all__ = ["Registry", "LOSS_REGISTRY"]
