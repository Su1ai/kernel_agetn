"""Roofline-style classification for normalized PPU Asight metrics."""

from __future__ import annotations

import logging
from typing import Any

from kernel_perf_agent.kernel_opt.roofline.ncu_roofline import (
    RooflineConfig,
    RooflineResult,
)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().strip('"')
    if not text or text == "-":
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


class PpuRooflineAnalyzer:
    """Classify PPU bottlenecks from normalized utilization metrics."""

    def __init__(
        self,
        logger: logging.Logger | None = None,
        roofline_config: RooflineConfig | None = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.config = roofline_config or RooflineConfig()
        self._efficiency_history: list[float] = []

    def analyze(self, ncu_metrics: dict[str, Any]) -> RooflineResult:
        metrics = ncu_metrics or {}
        compute = _optional_float(metrics.get("compute_util_pct"))
        memory = _optional_float(metrics.get("memory_util_pct"))
        warnings: list[str] = []
        if compute is None:
            warnings.append("PPU compute utilization unavailable")
        if memory is None:
            warnings.append("PPU HBM utilization unavailable")
        if compute is None and memory is None:
            return RooflineResult(0, 0, 0, False, 100, "unknown", False, warnings)

        compute_value = compute or 0.0
        memory_value = memory or 0.0
        efficiency = max(compute_value, memory_value)
        if compute is None:
            bottleneck = (
                "memory"
                if memory_value >= self.config.underutilized_threshold
                else "underutilized"
            )
        elif memory is None:
            bottleneck = (
                "compute"
                if compute_value >= self.config.underutilized_threshold
                else "underutilized"
            )
        elif efficiency < self.config.underutilized_threshold:
            bottleneck = "underutilized"
        else:
            bottleneck = "memory" if memory_value >= compute_value else "compute"

        return RooflineResult(
            compute_sol_pct=compute_value,
            memory_sol_pct=memory_value,
            efficiency_pct=efficiency,
            at_roofline=efficiency >= self.config.threshold_pct,
            headroom_pct=max(0.0, 100.0 - efficiency),
            bottleneck=bottleneck,
            uses_tensor_cores=False,
            warnings=warnings
            + ["Matrix-accelerator usage is not reported by this Asight mapping"],
        )

    def should_stop(self, result: RooflineResult) -> tuple[bool, str]:
        self._efficiency_history.append(result.efficiency_pct)
        if self.config.early_stop and result.at_roofline:
            return True, f"At measured PPU utilization limit ({result.efficiency_pct:.1f}%)"
        if len(self._efficiency_history) >= self.config.convergence_rounds:
            recent = self._efficiency_history[-self.config.convergence_rounds :]
            if max(recent) - min(recent) < self.config.min_improvement_pct:
                return True, "PPU utilization converged"
        return False, ""

    def reset_history(self) -> None:
        self._efficiency_history.clear()
