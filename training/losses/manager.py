"""Composable loss manager that wraps registered loss modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

import torch
from torch import nn

from .registry import LOSS_REGISTRY


@dataclass
class LossComponent:
    """Container describing a single loss invocation."""

    name: str
    module: nn.Module
    inputs: List[str] = field(default_factory=lambda: ["pred", "target"])
    kw_inputs: Mapping[str, str] = field(default_factory=dict)

    def to(self, *args: Any, **kwargs: Any) -> None:
        self.module.to(*args, **kwargs)

    def train(self, mode: bool = True) -> None:
        self.module.train(mode)

    def eval(self) -> None:
        self.module.eval()

    def state_dict(self) -> Dict[str, Any]:
        return self.module.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.module.load_state_dict(state)


class LossManager(nn.Module):
    """Aggregates multiple registered losses into a single callable."""

    def __init__(self, losses: Iterable[Mapping[str, Any]]):
        super().__init__()
        components: List[LossComponent] = []
        for idx, spec in enumerate(losses):
            if "type" not in spec:
                raise ValueError(f"Loss specification at index {idx} is missing required key 'type': {spec}")
            spec_dict = dict(spec)
            loss_type = spec_dict.pop("type")
            name = str(spec_dict.pop("name", loss_type))
            inputs = list(spec_dict.pop("inputs", ["pred", "target"]))
            kw_inputs = dict(spec_dict.pop("kw_inputs", {}))
            module_cls = LOSS_REGISTRY.get(loss_type)
            module = module_cls(**spec_dict)
            components.append(LossComponent(name=name, module=module, inputs=inputs, kw_inputs=kw_inputs))

        self.components = nn.ModuleList([comp.module for comp in components])
        # Keep the metadata aligned with ModuleList entries
        self._meta: List[LossComponent] = components
        self._last_metrics: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ utils
    def _resolve_source(
        self,
        key: str,
        pred: Optional[torch.Tensor],
        target: Optional[torch.Tensor],
        runtime: MutableMapping[str, Any],
    ) -> Any:
        aliases = {
            "pred": pred,
            "prediction": pred,
            "input": pred,
            "x": pred,
            "outputs": pred,
            "target": target,
            "gt": target,
            "y": target,
        }
        if key in aliases and aliases[key] is not None:
            return aliases[key]
        if key in runtime:
            return runtime[key]
        raise KeyError(f"LossManager could not resolve input '{key}'. Provided inputs: {list(runtime.keys())}")

    # ----------------------------------------------------------------- Module
    def forward(self, pred: torch.Tensor, target: Optional[torch.Tensor] = None, **kwargs: Any) -> torch.Tensor:
        runtime: MutableMapping[str, Any] = dict(kwargs)
        total = pred.new_tensor(0.0)
        metrics: Dict[str, torch.Tensor] = {}
        for module, meta in zip(self.components, self._meta):
            args: List[Any] = []
            for source in meta.inputs:
                value = self._resolve_source(source, pred, target, runtime)
                args.append(value)

            call_kwargs: Dict[str, Any] = {}
            for param_name, source in meta.kw_inputs.items():
                call_kwargs[param_name] = self._resolve_source(source, pred, target, runtime)

            value = module(*args, **call_kwargs)
            if isinstance(value, (list, tuple)):
                tensors = [item for item in value if item is not None]
                if not tensors:
                    continue
                value = sum(tensors)

            metrics[meta.name] = value.detach()
            total = total + value

        self._last_metrics = metrics
        return total

    # -------------------------------------------------------------- exposure
    @property
    def metrics(self) -> Mapping[str, float]:
        return {name: float(value.detach().mean().item()) for name, value in self._last_metrics.items()}

    @property
    def raw_metrics(self) -> Mapping[str, torch.Tensor]:
        return self._last_metrics

    def extra_state(self) -> Dict[str, Any]:  # pragma: no cover - passthrough hook
        return {
            "components": [
                {
                    "name": meta.name,
                    "inputs": meta.inputs,
                    "kw_inputs": dict(meta.kw_inputs),
                }
                for meta in self._meta
            ]
        }


def build_loss(config: Mapping[str, Any]) -> nn.Module:
    """Build a single loss from a config dictionary."""
    spec = dict(config)
    if "type" not in spec:
        raise ValueError(f"Loss config is missing required key 'type': {config}")
    loss_type = spec.pop("type")
    module_cls = LOSS_REGISTRY.get(loss_type)
    return module_cls(**spec)


__all__ = ["LossManager", "build_loss"]
