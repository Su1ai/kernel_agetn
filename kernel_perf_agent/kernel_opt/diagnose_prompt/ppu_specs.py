"""PPU accelerator specification detection.

Runtime-observable values (CU count, SMEM, registers, L2, threads) are read from
the device.  Peak throughputs are hard-coded from a microbenchmark on 真武 810E
(2026-10-14); they are clearly labelled ``measured`` so they are never mistaken
for vendor spec or for NVIDIA values.
"""

from __future__ import annotations

from typing import Any


# Measured on PPU-ZW810E (Beethoven, PPU0010 / MMAv1, sm80, 64 CU @ 1700 MHz)
# via tiled-GEMM microbenchmark (8192^3, acc fp32) and copy-bandwidth test.
_MEASURED_PEAK_FP16_TFLOPS = 91.7
_MEASURED_PEAK_BF16_TFLOPS = 91.7
_MEASURED_PEAK_FP32_TFLOPS = 45.0  # tf32 path
_SPEC_MEMORY_BW_GBPS = 2765.0       # vendor spec (HBM2e)
_MEASURED_COPY_BW_GBPS = 2100.0     # measured copy (read+write)
_CU_CLOCK_MHZ = 1700
_MEM_CLOCK_MHZ = 1800


def get_ppu_specs(device_name: str | None = None) -> dict[str, Any]:
    detected_name = device_name
    memory_gb: float | None = None
    compute_units: int | None = None
    compatibility_arch: str | None = None
    max_threads_per_sm: int | None = None
    shared_memory_kb_per_sm: int | None = None
    registers_per_sm: int | None = None
    l2_cache_mb: float | None = None
    warp_size: int | None = None
    warnings: list[str] = []

    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA-compatible PPU runtime is not available")
        props = torch.cuda.get_device_properties(0)
        detected_name = detected_name or props.name
        memory_gb = round(props.total_memory / (1024**3), 2)
        compute_units = getattr(props, "multi_processor_count", None)
        max_threads_per_sm = getattr(props, "max_threads_per_multi_processor", None)
        _smem = getattr(props, "shared_memory_per_multiprocessor", None)
        shared_memory_kb_per_sm = _smem // 1024 if _smem else None
        registers_per_sm = getattr(props, "regs_per_multiprocessor", None)
        _l2 = getattr(props, "L2_cache_size", None)
        l2_cache_mb = round(_l2 / (1024**2), 0) if _l2 else None
        warp_size = getattr(props, "warp_size", None)
        major = getattr(props, "major", None)
        minor = getattr(props, "minor", None)
        if major is not None and minor is not None:
            compatibility_arch = f"CUDA-compatible capability {major}.{minor}"
    except Exception as exc:
        warnings.append(f"Runtime device detection unavailable: {exc}")

    return {
        "name": detected_name or "PPU (model unknown)",
        "vendor": "PPU",
        "architecture": "PPU native architecture (CUDA-compatible runtime)",
        "runtime_backend": "cuda-compatible",
        "profiler": "PPU Asight Compute (acu)",
        "compatibility_architecture": compatibility_arch,
        "memory_gb": memory_gb,
        "memory_type": "HBM2e",
        "compute_unit_count": compute_units,
        # Compatibility alias for templates that have not yet migrated from SM.
        "sm_count": compute_units,
        "max_threads_per_sm": max_threads_per_sm,
        "warp_size": warp_size,
        "shared_memory_kb_per_sm": shared_memory_kb_per_sm,
        "registers_per_sm": registers_per_sm,
        "l1_cache_kb": None,
        "l2_cache_mb": l2_cache_mb,
        "cu_clock_mhz": _CU_CLOCK_MHZ,
        "memory_clock_mhz": _MEM_CLOCK_MHZ,
        # Throughput: peak values below are microbenchmark-measured on 810E,
        # except peak_memory_bw_gbps which is the vendor spec.
        "peak_memory_bw_gbps": _SPEC_MEMORY_BW_GBPS,
        "peak_memory_bw_measured_gbps": _MEASURED_COPY_BW_GBPS,
        "peak_fp32_tflops": _MEASURED_PEAK_FP32_TFLOPS,
        "peak_fp16_tflops": _MEASURED_PEAK_FP16_TFLOPS,
        "peak_bf16_tflops": _MEASURED_PEAK_BF16_TFLOPS,
        "throughput_source": "measured on PPU-ZW810E (2026-10-14)",
        "capabilities": {
            "triton": True,
            "cuda_compatible_api": True,
            "bf16": True,
            "fp16": True,
            "tf32": True,
            "fp8": False,       # e4m3 compile-fails; e5m2 is emulated (< fp16)
            "int8_mma": False,  # tl.dot int8 not supported on PPU0010
            "persistent_kernel": True,
            "hardware_counters": True,
            "nvidia_tma": False,
            "nvidia_tl_extra_cuda": False,
            "warp_specialization": "unknown",
            "matrix_accelerator": True,
        },
        "warnings": warnings,
    }
