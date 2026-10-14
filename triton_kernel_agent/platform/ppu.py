"""PPU component adapters for the platform registry.

Concrete specs, profiler, and roofline implementations live beside their
NVIDIA counterparts; this module only wires them into the common interfaces.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from kernel_perf_agent.kernel_opt.diagnose_prompt.ppu_specs import get_ppu_specs
from kernel_perf_agent.kernel_opt.roofline.ncu_roofline import RooflineConfig
from kernel_perf_agent.kernel_opt.roofline.ppu_roofline import PpuRooflineAnalyzer
from triton_kernel_agent.opt_worker_component.profiling.acu_profiler import (
    AcuKernelProfiler,
)
from triton_kernel_agent.platform.interfaces import (
    AcceleratorSpecsProvider,
    BottleneckAnalyzerBase,
    RAGPrescriberBase,
)
from triton_kernel_agent.platform.nvidia import (
    NvidiaBenchmarker,
    NvidiaVerifier,
    NvidiaWorkerRunner,
)
from triton_kernel_agent.platform.ppu_runtime import ensure_ppu_environment


class PpuVerifier(NvidiaVerifier):
    def __init__(
        self, log_dir: Path, logger: logging.Logger, openai_model: str = "gpt-5"
    ) -> None:
        ensure_ppu_environment()
        super().__init__(log_dir=log_dir, logger=logger, openai_model=openai_model)

    def verify(
        self, kernel_code: str, problem_file: Path, test_code: list[str]
    ) -> bool:
        """Run CUDA-compatible verification with PPU prompt constraints."""
        from triton_kernel_agent.worker import VerificationWorker

        verify_dir = self.log_dir / "initial_verify"
        verify_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(problem_file, verify_dir / "problem.py")
        worker = VerificationWorker(
            worker_id=-1,
            workdir=verify_dir,
            log_dir=verify_dir,
            openai_model=self.openai_model,
            target_platform="ppu",
        )
        success, _, error = worker.verify_with_refinement(
            kernel_code=kernel_code,
            test_code=test_code,
            problem_description=problem_file.read_text(),
            max_refine_attempts=3,
        )
        if not success:
            self.logger.error("Initial PPU kernel verification failed: %s", error[:200])
        else:
            self.logger.info("Initial kernel passed PPU verification")
        return success


class PpuBenchmarker(NvidiaBenchmarker):
    def __init__(
        self,
        log_dir: Path,
        logger: logging.Logger,
        benchmark_lock: Any,
        warmup: int = 25,
        repeat: int = 100,
    ) -> None:
        ensure_ppu_environment()
        super().__init__(
            log_dir=log_dir,
            logger=logger,
            benchmark_lock=benchmark_lock,
            warmup=warmup,
            repeat=repeat,
        )


class PpuWorkerRunner(NvidiaWorkerRunner):
    def __init__(
        self,
        log_dir: Path,
        logger: logging.Logger,
        benchmark_lock: Any,
        profiling_semaphore: Any,
        openai_model: str,
        high_reasoning_effort: bool,
        bottleneck_override: str | None,
        worker_kwargs: dict[str, Any],
        gpu_ids: list[int] | None = None,
        gpu_locks: dict[int, Any] | None = None,
    ) -> None:
        ensure_ppu_environment()
        super().__init__(
            log_dir=log_dir,
            logger=logger,
            benchmark_lock=benchmark_lock,
            profiling_semaphore=profiling_semaphore,
            openai_model=openai_model,
            high_reasoning_effort=high_reasoning_effort,
            bottleneck_override=bottleneck_override,
            worker_kwargs=worker_kwargs,
            gpu_ids=gpu_ids,
            gpu_locks=gpu_locks,
        )


class PpuAcceleratorSpecsProvider(AcceleratorSpecsProvider):
    def get_specs(self, device_name: str | None = None) -> dict[str, Any]:
        ensure_ppu_environment()
        return get_ppu_specs(device_name)


# Keep the registry-facing name parallel to NvidiaKernelProfiler.  ``acu``
# (Asight Compute) is the kernel-Metrics analyzer; ``asys`` only traced timings
# and cannot supply the utilization counters the roofline analysis needs.
PpuKernelProfiler = AcuKernelProfiler


class PpuBottleneckAnalyzer(BottleneckAnalyzerBase):
    def __init__(
        self,
        logger: logging.Logger | None = None,
        log_dir: Path | None = None,
        openai_model: str = "gpt-5",
        gpu_name: str | None = None,
        roofline_config: RooflineConfig | None = None,
        num_bottlenecks: int = 1,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._log_dir = log_dir
        self._model = openai_model
        self._gpu_name = gpu_name
        self._num_bottlenecks = num_bottlenecks
        self._delegate: Any | None = None
        self.roofline = PpuRooflineAnalyzer(self._logger, roofline_config)

    def _get_delegate(self) -> Any:
        if self._delegate is None:
            from triton_kernel_agent.opt_worker_component.prescribing.bottleneck_analyzer import (
                BottleneckAnalyzer,
            )
            from utils.providers import get_model_provider

            self._delegate = BottleneckAnalyzer(
                provider=get_model_provider(self._model),
                model=self._model,
                gpu_specs=get_ppu_specs(self._gpu_name),
                logs_dir=self._log_dir,
                logger=self._logger,
                num_bottlenecks=self._num_bottlenecks,
                profiler_name="PPU Asight Compute (acu)",
            )
            self._delegate.roofline = self.roofline
        return self._delegate

    def analyze(
        self,
        kernel_code: str,
        ncu_metrics: dict[str, Any],
        round_num: int = 0,
        roofline_result: Any | None = None,
    ) -> list[Any]:
        return self._get_delegate().analyze(
            kernel_code, ncu_metrics, round_num, roofline_result
        )

    def analyze_source_only(
        self,
        kernel_code: str,
        problem_code: str,
        round_num: int = 0,
        baseline_time_ms: float | None = None,
        pytorch_baseline_ms: float | None = None,
    ) -> list[Any]:
        return self._get_delegate().analyze_source_only(
            kernel_code,
            problem_code,
            round_num,
            baseline_time_ms,
            pytorch_baseline_ms,
        )


class PpuRAGPrescriber(RAGPrescriberBase):
    """Wraps :class:`RAGPrescriber` with the PPU (真武 810E) knowledge base."""

    def __init__(
        self,
        logger: logging.Logger | None = None,
        database_path: Path | None = None,
        **_: Any,
    ) -> None:
        self._logger = logger
        self._database_path = database_path
        self._delegate: Any | None = None

    def _get_delegate(self) -> Any:
        if self._delegate is None:
            from triton_kernel_agent.opt_worker_component.prescribing.RAG_based_prescriber import (
                RAGPrescriber,
            )
            from kernel_perf_agent.kernel_opt.database.ppu.base import (
                build_ppu_hierarchy,
            )

            db_path = self._database_path
            if db_path is None:
                project_root = Path(__file__).resolve()
                while project_root.parent != project_root:
                    if (project_root / "pyproject.toml").exists():
                        break
                    project_root = project_root.parent
                db_path = (
                    project_root
                    / "kernel_perf_agent"
                    / "kernel_opt"
                    / "database"
                    / "ppu"
                    / "code_samples"
                )
            self._delegate = RAGPrescriber(
                database_path=db_path,
                logger=self._logger,
                hierarchy_builder=build_ppu_hierarchy,
            )
        return self._delegate

    def retrieve(self, query: str) -> tuple[Any | None, Any]:
        return self._get_delegate().retrieve(query)

    def build_context(self, opt_node: Any, **kwargs: Any) -> str:
        return self._get_delegate().build_context(opt_node, **kwargs)
