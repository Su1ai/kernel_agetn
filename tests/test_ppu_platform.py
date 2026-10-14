from pathlib import Path

import pytest

from triton_kernel_agent.opt_worker_component.profiling.acu_profiler import AcuKernelProfiler
from triton_kernel_agent.platform.ppu import PpuKernelProfiler, PpuRooflineAnalyzer
from triton_kernel_agent.platform.registry import registry
from triton_kernel_agent.platform_config import get_platform


def test_ppu_platform_uses_cuda_compatible_device_with_ppu_guidance():
    platform = get_platform("ppu")
    assert platform.device_string == "cuda"
    assert "CUDA-compatible" in platform.kernel_guidance
    assert "NVIDIA" in platform.kernel_guidance


def test_all_ppu_components_registered():
    for component in registry.list_components():
        assert registry.has(component, "ppu"), component


def _write_acu_csv(path: Path, rows: list[list[str]]) -> None:
    header = [
        "ID", "Process ID", "Process Name", "Host Name", "Kernel Name",
        "Kernel Mangled Name", "Context", "Stream", "Block Size", "Grid Size",
        "Device", "CC", "Section Name", "Metric Name", "Metric Unit",
        "Metric Value", "Rule Name", "Rule Type", "Rule Description",
    ]
    import csv as _csv

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = _csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _row(kernel: str, section: str, metric: str, value: str, unit: str) -> list[str]:
    return ["0", "1", "1", "host", kernel, "mangled", "", "", "256", "864",
            "0", "", section, metric, unit, value, "n/a", "n/a", "n/a"]


def test_acu_csv_is_normalized_and_prefers_sol_section(tmp_path: Path):
    kernel = tmp_path / "kernel.py"
    kernel.write_text("import triton\n@triton.jit\ndef add_kernel(x):\n    pass\n")
    csv_file = tmp_path / "metrics.csv"
    _write_acu_csv(csv_file, [
        # Triton kernel: SOL section carries the percentage values.
        _row("add_kernel", "PPU Speed Of Light Throughput", "Compute (CU) Throughput", "57.45", "%"),
        _row("add_kernel", "PPU Speed Of Light Throughput", "Memory Throughput", "5.63", "%"),
        _row("add_kernel", "PPU Speed Of Light Throughput", "Duration", "10647.65", "nsecond"),
        _row("add_kernel", "Occupancy", "Achieved Occupancy", "83.53", "%"),
        _row("add_kernel", "Launch Statistics", "Registers Per Thread", "32", "register/thread"),
        # Same metric name in the memory section is bytes/s and must NOT win.
        _row("add_kernel", "Memory Workload Analysis", "Memory Throughput", "500631999.11", "byte/second"),
        # PyTorch-internal kernel must be filtered out.
        _row("void at::native::other", "PPU Speed Of Light Throughput", "Compute (CU) Throughput", "1.0", "%"),
    ])

    metrics = AcuKernelProfiler.parse_acu_csv(csv_file, kernel)

    assert list(metrics) == ["add_kernel"]
    row = metrics["add_kernel"]
    assert row["compute_util_pct"] == pytest.approx(57.45)
    assert row["memory_util_pct"] == pytest.approx(5.63)
    assert row["occupancy_pct"] == pytest.approx(83.53)
    assert row["duration_ms"] == pytest.approx(10647.65 / 1_000_000)
    assert row["registers_per_thread"] == 32
    assert row["sm__throughput.avg.pct_of_peak_sustained_elapsed"] == pytest.approx(57.45)


def test_acu_missing_metrics_stay_missing_not_zero(tmp_path: Path):
    kernel = tmp_path / "kernel.py"
    kernel.write_text("import triton\n@triton.jit\ndef k(x):\n    pass\n")
    csv_file = tmp_path / "metrics.csv"
    _write_acu_csv(csv_file, [
        _row("k", "PPU Speed Of Light Throughput", "Memory Throughput", "-", "%"),
        _row("k", "Occupancy", "Achieved Occupancy", "n/a", "%"),
    ])

    row = AcuKernelProfiler.parse_acu_csv(csv_file, kernel)["k"]

    assert row["memory_util_pct"] is None
    assert row["occupancy_pct"] is None
    assert "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed" not in row


def test_ppu_roofline_uses_normalized_metrics():
    result = PpuRooflineAnalyzer().analyze(
        {"compute_util_pct": 25.0, "memory_util_pct": 80.0}
    )
    assert result.bottleneck == "memory"
    assert result.efficiency_pct == pytest.approx(80.0)
