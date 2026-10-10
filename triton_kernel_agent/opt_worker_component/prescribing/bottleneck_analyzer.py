# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Bottleneck Analyzer - LLM-based NCU profiling analysis.

This module orchestrates LLM calls for bottleneck analysis using:
- judger_prompt.py: Prompt template, parsing, BottleneckResult dataclass
- ncu_roofline.py: Roofline analysis using NCU SOL metrics

Bottleneck Categories:
- memory: Memory bandwidth is the limiting factor
- compute: Compute throughput is the limiting factor
- underutilized: Neither saturated (<60% both), indicating stalls/occupancy issues
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from kernel_perf_agent.kernel_opt.diagnose_prompt.judger_prompt import (
    BottleneckResult,
    build_bottleneck_prompt,
    parse_bottleneck_response,
)
from kernel_perf_agent.kernel_opt.roofline.ncu_roofline import RooflineAnalyzer
from triton_kernel_agent.worker_util import _call_llm, _save_debug_file
from utils.providers.base import BaseProvider
from utils.llm_budget import ANALYSIS_MAX_TOKENS


_SOURCE_ONLY_PROMPT = """You are optimizing a Triton GPU kernel.

IMPORTANT: Nsight Compute (NCU) hardware profiling is UNAVAILABLE in this
environment. You MUST NOT invent, guess, or report any NCU metrics
(SOL %, memory bandwidth, compute utilization, occupancy, stall reasons,
cache hit rates, or any other hardware counters). Any such numbers you make
up are fabrication and will be discarded.

Available facts:
- GPU specs: {gpu_specs}
- Kernel source: (below)
- Problem source: (below)
- Baseline Triton kernel time: {baseline_time_ms} ms
- PyTorch eager baseline time: {pytorch_baseline_ms} ms

Analyze the kernel source and identify up to {num_bottlenecks} bottleneck(s).
Ground every claim in observable properties of the code and in the measured
runtimes, for example:
- memory access patterns, contiguity, coalescing, redundant global loads/stores
- tile / block sizes, num_warps, num_stages, loop structure
- fusion opportunities and kernel-launch count
- reduction / softmax / layernorm / GEMM implementation strategies
- whether Tensor Core paths are plausibly used (tl.dot on fp16/bf16/ints)
- whether the kernel is plausibly memory-bound, compute-bound, or underutilized
  given the measured time vs PyTorch eager

Return ONLY a JSON array (no prose) with exactly {num_bottlenecks} entries;
each entry MUST have this schema:
[
  {{
    "category": "memory|compute|underutilized",
    "summary": "one-line bottleneck summary",
    "reasoning": "analysis grounded in source + runtimes, no fabricated metrics",
    "root_causes": [
      {{"cause": "...", "evidence": ["..."], "fixes": [{{"fix": "...", "rationale": "..."}}]}}
    ],
    "recommended_fixes": [{{"fix": "...", "rationale": "..."}}]
  }}
]

KERNEL SOURCE:
```python
{kernel_code}
```

PROBLEM SOURCE:
```python
{problem_code}
```
"""


# Both analyses reason over a full kernel plus the problem source, and the
# model's thinking trace is billed against this same output budget. At 16384 the
# reasoning consumed the entire allowance on some attempts, leaving no room for
# the answer and producing an empty response. Raise this if empty responses
# reappear; the gateway accepts up to 65536.


class BottleneckAnalyzer:
    """LLM-based bottleneck analyzer using NCU metrics."""

    def __init__(
        self,
        provider: BaseProvider,
        model: str,
        gpu_specs: dict[str, Any],
        logs_dir: Path | None = None,
        logger: logging.Logger | None = None,
        num_bottlenecks: int = 1,
        num_causes: int = 2,
        num_fixes: int = 1,
        enable_debug: bool = True,
    ):
        """
        Initialize bottleneck analyzer.

        Args:
            provider: LLM provider instance
            model: Model name for LLM calls
            gpu_specs: GPU hardware specifications
            logs_dir: Directory for saving debug files
            logger: Logger instance
            num_bottlenecks: Number of bottlenecks to request from LLM
            num_causes: Number of root causes per bottleneck
            num_fixes: Number of recommended fixes per bottleneck
            enable_debug: Whether to save debug files (prompts/responses)
        """
        self.provider = provider
        self.model = model
        self.gpu_specs = gpu_specs
        self.logs_dir = logs_dir
        self.logger = logger or logging.getLogger(__name__)
        self.num_bottlenecks = num_bottlenecks
        self.num_causes = num_causes
        self.num_fixes = num_fixes
        self.enable_debug = enable_debug
        self.roofline = RooflineAnalyzer(logger=logger)

    def analyze(
        self,
        kernel_code: str,
        ncu_metrics: dict[str, Any],
        round_num: int = 0,
        roofline_result: Any = None,
    ) -> list[BottleneckResult]:
        """
        Analyze kernel bottlenecks using LLM.

        Args:
            kernel_code: The Triton kernel source code
            ncu_metrics: NCU profiling metrics dictionary
            round_num: Current optimization round (for logging)
            roofline_result: Pre-computed RooflineResult (if None, computed internally)

        Returns:
            List of BottleneckResult (ordered by importance).
            Empty list if analysis fails.
        """
        if roofline_result is None:
            # Filter out PyTorch kernels (at::*) and get Triton kernel metrics
            if ncu_metrics:
                triton_kernels = {
                    name: metrics
                    for name, metrics in ncu_metrics.items()
                    if not name.startswith("at::") and not name.startswith("void at::")
                }
                flat_metrics = (
                    next(iter(triton_kernels.values()))
                    if triton_kernels
                    else next(iter(ncu_metrics.values()), {})
                )
            else:
                flat_metrics = {}
            roofline_result = self.roofline.analyze(flat_metrics)

        prompt = build_bottleneck_prompt(
            kernel_code=kernel_code,
            ncu_metrics=ncu_metrics,
            roofline=roofline_result,
            gpu_specs=self.gpu_specs,
            num_bottlenecks=self.num_bottlenecks,
            num_causes=self.num_causes,
            num_fixes=self.num_fixes,
        )

        response = _call_llm(
            provider=self.provider,
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            logger=self.logger,
            max_tokens=ANALYSIS_MAX_TOKENS,
            empty_retries=1,
            retry_kwargs={"max_tokens": self.provider.get_max_tokens_limit(self.model)},
        )

        if self.enable_debug and self.logs_dir:
            _save_debug_file(
                self.logs_dir / f"round{round_num:03d}_bottleneck_prompt.txt",
                prompt,
                self.logger,
            )
            _save_debug_file(
                self.logs_dir / f"round{round_num:03d}_bottleneck_response.txt",
                response,
                self.logger,
            )

        results = parse_bottleneck_response(response)

        if results:
            categories = [r.category for r in results]
            self.logger.info(f"[{round_num}] Bottlenecks: {', '.join(categories)}")
        else:
            self.logger.warning(f"[{round_num}] Failed to parse bottleneck response")

        return results

    def analyze_source_only(
        self,
        kernel_code: str,
        problem_code: str,
        round_num: int = 0,
        baseline_time_ms: float | None = None,
        pytorch_baseline_ms: float | None = None,
    ) -> list[BottleneckResult]:
        """Analyze bottlenecks without NCU metrics (source-only fallback).

        Used when Nsight Compute is unavailable or fails. The LLM must reason
        from kernel source, problem source, and measured runtimes only, and
        must NOT fabricate hardware counters (SOL%, bandwidth, occupancy,
        stall reasons, etc.).

        Returns:
            List of BottleneckResult (same schema as :meth:`analyze`).
        """
        from kernel_perf_agent.kernel_opt.diagnose_prompt.judger_prompt import (
            _format_gpu_specs,
        )

        prompt = _SOURCE_ONLY_PROMPT.format(
            num_bottlenecks=self.num_bottlenecks,
            num_causes=self.num_causes,
            num_fixes=self.num_fixes,
            gpu_specs=_format_gpu_specs(self.gpu_specs),
            kernel_code=kernel_code,
            problem_code=problem_code,
            baseline_time_ms=(
                f"{baseline_time_ms:.4f}" if baseline_time_ms is not None else "unknown"
            ),
            pytorch_baseline_ms=(
                f"{pytorch_baseline_ms:.4f}"
                if pytorch_baseline_ms is not None
                else "unknown"
            ),
        )

        response = _call_llm(
            provider=self.provider,
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            logger=self.logger,
            max_tokens=ANALYSIS_MAX_TOKENS,
            empty_retries=1,
            retry_kwargs={"max_tokens": self.provider.get_max_tokens_limit(self.model)},
        )

        if self.enable_debug and self.logs_dir:
            _save_debug_file(
                self.logs_dir / f"round{round_num:03d}_source_only_prompt.txt",
                prompt,
                self.logger,
            )
            _save_debug_file(
                self.logs_dir / f"round{round_num:03d}_source_only_response.txt",
                response,
                self.logger,
            )

        results = parse_bottleneck_response(response)

        if results:
            categories = [r.category for r in results]
            self.logger.info(
                f"[{round_num}] Source-only bottlenecks: {', '.join(categories)}"
            )
        else:
            self.logger.warning(f"[{round_num}] Failed to parse source-only response")

        return results
