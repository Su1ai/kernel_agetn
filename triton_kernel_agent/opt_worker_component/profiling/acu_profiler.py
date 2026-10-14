"""PPU Asight Compute (``acu``) profiler and CSV normalization.

``acu`` is the kernel-level Metrics analyzer bundled with the PPU SDK. Unlike
Asight Systems (``asys``), which only traces operator timings, ``acu`` collects
the hardware counters the roofline analysis needs: compute and memory
throughput as a percentage of peak, occupancy, and cache behaviour.

Hardware counters are a single-owner resource on the PPU. A host-side monitor
(``amperf-collector``) normally holds them, so ``acu`` fails with
"Device is not ready for profiling" unless that collector is paused. The SDK
exposes a pause endpoint for it, which this profiler calls best-effort before
each run.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from triton_kernel_agent.opt_worker_component.profiling.ncu_wrapper_factory import (
    NCUWrapperFactory,
)

# Metric sections emitted by acu (CSV "Section Name" column).
_SOL_SECTION = "PPU Speed Of Light Throughput"
_OCCUPANCY_SECTION = "Occupancy"
_LAUNCH_SECTION = "Launch Statistics"
_COMPUTE_SECTION = "PPU Compute Workload Analysis"
_MEMORY_SECTION = "Memory Workload Analysis"

# Metric labels inside those sections.
_COMPUTE_SOL = "Compute (CU) Throughput"
_MEMORY_SOL = "Memory Throughput"
_DURATION = "Duration"

_NANOSECOND_UNITS = {"nsecond", "ns", "nanosecond"}
_MICROSECOND_UNITS = {"usecond", "us", "microsecond"}
_MILLISECOND_UNITS = {"msecond", "ms", "millisecond"}


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().strip('"')
    if not text or text in {"-", "n/a", "N/A"}:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    parsed = _optional_float(value)
    return int(parsed) if parsed is not None else None


def _to_nanoseconds(value: Any, unit: Any) -> float | None:
    """Convert a duration value/unit pair to nanoseconds."""
    parsed = _optional_float(value)
    if parsed is None:
        return None
    text = str(unit or "").strip().lower()
    if text in _NANOSECOND_UNITS:
        return parsed
    if text in _MICROSECOND_UNITS:
        return parsed * 1_000.0
    if text in _MILLISECOND_UNITS:
        return parsed * 1_000_000.0
    # Unknown unit: report the raw number rather than inventing a scale.
    return None


@dataclass
class PpuProfilerMetadata:
    kernel_file: str
    problem_file: str
    round_num: int
    timestamp: str
    profiler_name: str = "PPU Asight Compute (acu)"
    profiler_version: str | None = None
    report_file: str | None = None
    csv_file: str | None = None


@dataclass
class PpuProfilerResults:
    metrics: dict[str, dict[str, Any]]
    metadata: PpuProfilerMetadata

    def to_dict(self) -> dict[str, Any]:
        return {"metrics": self.metrics, "metadata": asdict(self.metadata)}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


class AcuKernelProfiler:
    """Collect and normalize PPU Asight Compute kernel metrics."""

    def __init__(
        self,
        logger: logging.Logger | None = None,
        log_dir: Path | None = None,
        artifacts_dir: Path | None = None,
        profiling_semaphore: Any | None = None,
        acu_bin_path: str | None = None,
        acu_timeout_seconds: int = 600,
        launch_count: int | None = None,
    ) -> None:
        # Imported lazily: a module-level import here would close a cycle
        # (acu_profiler -> platform/__init__ -> platform.ppu -> acu_profiler).
        from triton_kernel_agent.platform.ppu_runtime import ensure_ppu_environment

        ensure_ppu_environment()
        self.logger = logger or logging.getLogger(__name__)
        self.logs_dir = Path(log_dir or ".")
        self.artifacts_dir = Path(artifacts_dir or self.logs_dir / "artifacts")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.profiling_semaphore = profiling_semaphore
        self.acu_bin_path = acu_bin_path
        self.acu_timeout_seconds = acu_timeout_seconds
        # The wrapper launches many PyTorch setup kernels (input creation,
        # weight init) before the first Triton kernel, so a small cap would
        # profile only those. The cap is generous; `acu` stops when the app
        # ends, and the parser keeps just the Triton kernels.
        self.launch_count = launch_count or int(
            os.getenv("KERNELAGENT_ACU_LAUNCH_COUNT", "200")
        )
        self.wrapper_factory = NCUWrapperFactory(self.logger)

    # ------------------------------------------------------------------
    # Environment / tooling
    # ------------------------------------------------------------------

    @property
    def acu_bin(self) -> str:
        candidate = (
            self.acu_bin_path
            or shutil.which("acu")
            or "/usr/local/PPU_SDK/asight/bin/acu"
        )
        if not Path(candidate).exists() and shutil.which(candidate) is None:
            raise FileNotFoundError(
                "PPU Asight Compute 'acu' not found. "
                "Source /usr/local/PPU_SDK/envsetup.sh ppu first."
            )
        return candidate

    @property
    def profiler_version(self) -> str | None:
        try:
            result = subprocess.run(
                [self.acu_bin, "-v"], capture_output=True, text=True, timeout=10
            )
            text = (result.stdout or result.stderr).strip()
            return text.splitlines()[0] if result.returncode == 0 and text else None
        except Exception:
            return None

    def _request_metric_pause(self, seconds: int = 600) -> None:
        """Best-effort: free the PPU performance counters for profiling.

        The host-side collector owns the counters, so ``acu`` cannot profile
        while it is running. A rejected request (already paused) is fine.
        """
        url = os.getenv(
            "KERNELAGENT_PROFMETRIC_PAUSE_URL",
            "http://localhost:9501/profmetric/pause",
        )
        try:
            request = urllib.request.Request(
                url, data=f"time={seconds}".encode(), method="POST"
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                body = response.read().decode(errors="replace").strip()
            self.logger.info("Requested metric pause: %s", body[:160] or "ok")
        except Exception as exc:  # already paused / endpoint absent / no network
            self.logger.debug("Metric pause request ignored: %s", exc)

    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------

    def profile_kernel(
        self,
        kernel_file: Path,
        problem_file: Path,
        round_num: int,
        max_retries: int = 2,
    ) -> PpuProfilerResults | None:
        wrapper_file = self.wrapper_factory.create_ncu_wrapper(
            kernel_file, problem_file, self.artifacts_dir
        )
        acquired = False
        if self.profiling_semaphore is not None:
            acquired = self.profiling_semaphore.acquire(timeout=900)
            if not acquired:
                self.logger.warning("[%s] PPU profiler semaphore timed out", round_num)
                return None
        try:
            return self._profile_with_retries(
                wrapper_file, kernel_file, problem_file, round_num, max_retries
            )
        finally:
            if acquired:
                self.profiling_semaphore.release()

    def _profile_with_retries(
        self,
        wrapper_file: Path,
        kernel_file: Path,
        problem_file: Path,
        round_num: int,
        max_retries: int,
    ) -> PpuProfilerResults | None:
        for attempt in range(1, max_retries + 1):
            try:
                return self._profile_once(
                    wrapper_file, kernel_file, problem_file, round_num
                )
            except FileNotFoundError as exc:
                self.logger.error("PPU profiling unavailable: %s", exc)
                return None
            except (subprocess.TimeoutExpired, RuntimeError, OSError) as exc:
                self.logger.warning(
                    "PPU profiling attempt %s/%s failed: %s",
                    attempt,
                    max_retries,
                    exc,
                )
                if attempt < max_retries:
                    time.sleep(2**attempt)
        return None

    def _profile_once(
        self,
        wrapper_file: Path,
        kernel_file: Path,
        problem_file: Path,
        round_num: int,
    ) -> PpuProfilerResults:
        self._request_metric_pause()

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        report_file = self.artifacts_dir / f"acu_round_{round_num}_{stamp}.acurep"
        csv_file = self.logs_dir / f"round{round_num:03d}_acu_metrics.csv"
        env = os.environ.copy()
        # acu collects per-launch counters, so a single kernel execution per
        # wrapper run is enough; keep the wrapper's own warmup count.
        env.setdefault("KERNELAGENT_PROFILE_ITERATIONS", "1")

        cmd = [
            self.acu_bin,
            "-o",
            str(report_file),
            "-f",
            "--csv",
            "--csv-file",
            str(csv_file),
        ]

        # No `-k` filter: this acu build profiles nothing when a kernel-name
        # filter is supplied, and it also matches by substring rather than the
        # exact names we would pass. Instead profile broadly (bounded by the
        # launch count, which is generous enough to reach the Triton kernels
        # that run after PyTorch's setup kernels) and filter by name in the
        # parser, where the semantics are under our control.
        cmd += ["-c", str(self.launch_count), sys.executable, str(wrapper_file)]
        completed = subprocess.run(
            cmd,
            cwd=self.artifacts_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=self.acu_timeout_seconds,
        )
        log_file = self.logs_dir / f"round{round_num:03d}_acu_profile.log"
        log_file.write_text((completed.stdout or "") + "\n" + (completed.stderr or ""))
        if completed.returncode != 0:
            raise RuntimeError(
                f"acu profile failed ({completed.returncode}): "
                f"{(completed.stderr or completed.stdout)[-1000:]}"
            )
        if not csv_file.exists() or csv_file.stat().st_size == 0:
            raise RuntimeError(f"acu produced no CSV output at {csv_file}")

        metrics = self.parse_acu_csv(csv_file, kernel_file)
        if not metrics:
            raise RuntimeError("acu CSV contained no matching PPU kernel metrics")

        results = PpuProfilerResults(
            metrics=metrics,
            metadata=PpuProfilerMetadata(
                kernel_file=str(kernel_file),
                problem_file=str(problem_file),
                round_num=round_num,
                timestamp=datetime.now(timezone.utc).isoformat(),
                profiler_version=self.profiler_version,
                report_file=str(report_file),
                csv_file=str(csv_file),
            ),
        )
        (self.logs_dir / f"round{round_num:03d}_acu_metrics.json").write_text(
            results.to_json()
        )
        self.logger.info(
            "PPU Asight Compute profiling completed for round %s (%d kernel(s))",
            round_num,
            len(metrics),
        )
        return results

    # ------------------------------------------------------------------
    # CSV parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _triton_function_names(kernel_file: Path) -> set[str]:
        try:
            source = kernel_file.read_text()
        except OSError:
            return set()
        return set(
            re.findall(
                r"@(?:triton\.)?jit(?:\([^\n]*\))?\s*\n\s*def\s+([A-Za-z_]\w*)",
                source,
            )
        )

    @classmethod
    def parse_acu_csv(
        cls, csv_file: Path, kernel_file: Path
    ) -> dict[str, dict[str, Any]]:
        """Pivot the long-format acu CSV into one record per kernel."""
        with csv_file.open(newline="", encoding="utf-8", errors="replace") as handle:
            rows = list(csv.DictReader(handle))

        # {kernel_name: {(section, metric): (value, unit)}}
        grouped: dict[str, dict[tuple[str, str], tuple[str, str]]] = {}
        for row in rows:
            kernel = (row.get("Kernel Name") or "").strip()
            if not kernel:
                continue
            key = (row.get("Section Name") or "", row.get("Metric Name") or "")
            grouped.setdefault(kernel, {})[key] = (
                row.get("Metric Value") or "",
                row.get("Metric Unit") or "",
            )

        triton_names = cls._triton_function_names(kernel_file)
        selected = [name for name in grouped if any(t in name for t in triton_names)]
        if not selected:
            selected = [
                name
                for name in grouped
                if not name.startswith(("at::", "void at::"))
            ]

        metrics: dict[str, dict[str, Any]] = {}
        for kernel in selected:
            metrics[kernel] = cls._normalize_kernel(grouped[kernel], kernel)
        return metrics

    @staticmethod
    def _value(
        record: dict[tuple[str, str], tuple[str, str]], section: str, metric: str
    ) -> tuple[Any, Any]:
        raw = record.get((section, metric))
        if raw is None:
            return None, None
        return _optional_float(raw[0]), raw[1]

    @classmethod
    def _normalize_kernel(
        cls,
        record: dict[tuple[str, str], tuple[str, str]],
        kernel_name: str,
    ) -> dict[str, Any]:
        duration_ns = _to_nanoseconds(*(record.get((_SOL_SECTION, _DURATION)) or (None, None)))

        normalized: dict[str, Any] = {
            "profiler_name": "PPU Asight Compute (acu)",
            "kernel_name": kernel_name,
            "duration_ns": duration_ns,
            "duration_ms": duration_ns / 1_000_000 if duration_ns is not None else None,
            "compute_util_pct": _optional_float(
                (record.get((_SOL_SECTION, _COMPUTE_SOL)) or (None,))[0]
            ),
            "memory_util_pct": _optional_float(
                (record.get((_SOL_SECTION, _MEMORY_SOL)) or (None,))[0]
            ),
            "dram_util_pct": _optional_float(
                (record.get((_SOL_SECTION, "DRAM Throughput")) or (None,))[0]
            ),
            "l2_cache_util_pct": _optional_float(
                (record.get((_SOL_SECTION, "L2 Cache Throughput")) or (None,))[0]
            ),
            "occupancy_pct": _optional_float(
                (record.get((_OCCUPANCY_SECTION, "Achieved Occupancy")) or (None,))[0]
            ),
            "theoretical_occupancy_pct": _optional_float(
                (record.get((_OCCUPANCY_SECTION, "Theoretical Occupancy")) or (None,))[0]
            ),
            "active_warps_per_cu": _optional_float(
                (
                    record.get((_OCCUPANCY_SECTION, "Achieved Active Warps Per CU"))
                    or (None,)
                )[0]
            ),
            "cu_busy_pct": _optional_float(
                (record.get((_COMPUTE_SECTION, "CU Busy")) or (None,))[0]
            ),
            "issue_slots_busy_pct": _optional_float(
                (record.get((_COMPUTE_SECTION, "Issue Slots Busy")) or (None,))[0]
            ),
            "registers_per_thread": _optional_int(
                (
                    record.get((_LAUNCH_SECTION, "Registers Per Thread"))
                    or (None,)
                )[0]
            ),
            "block_size": _optional_int(
                (record.get((_LAUNCH_SECTION, "Block Size")) or (None,))[0]
            ),
            "grid_size": _optional_int(
                (record.get((_LAUNCH_SECTION, "Grid Size")) or (None,))[0]
            ),
            "l2_hit_rate_pct": _optional_float(
                (record.get((_MEMORY_SECTION, "L2 Hit Rate")) or (None,))[0]
            ),
            "mem_busy_pct": _optional_float(
                (record.get((_MEMORY_SECTION, "Mem Busy")) or (None,))[0]
            ),
            "raw_metrics": {
                f"{section}::{metric}": {"value": value, "unit": unit}
                for (section, metric), (value, unit) in record.items()
            },
        }

        # Backward-compatible aliases for code that still speaks the NCU names.
        compute = normalized["compute_util_pct"]
        memory = normalized["memory_util_pct"]
        if compute is not None:
            normalized["sm__throughput.avg.pct_of_peak_sustained_elapsed"] = compute
        if memory is not None:
            normalized[
                "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
            ] = memory
            normalized["dram__throughput.avg.pct_of_peak_sustained_elapsed"] = memory
        return normalized
