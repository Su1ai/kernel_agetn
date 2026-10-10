#!/usr/bin/env python3
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
Compose an end-to-end Triton kernel for the original KernelBench problem by
leveraging:
  1) Fuser's subgraphs JSON (decomposition + shapes)
  2) KernelAgent-generated Triton kernels for those subgraphs

We call an LLM to synthesize a final composed kernel that matches the original
problem semantics, returning one complete Python file that exposes
`kernel_function(...)` and includes a minimal self-test that checks numerical
equivalence against a PyTorch reference derived from the original problem.

Usage:
  python -m Fuser.compose_end_to_end \
      --problem /abs/path/to/kernelbench_problem.py \
      --subgraphs /abs/path/to/subgraphs.json \
      --kernels-summary /abs/path/to/kernels_out/summary.json \
      [--model gpt-5] [--out-dir ./compose_out] [--verify]

Notes:
- Requires an available LLM provider configured via KernelAgent providers
  (e.g., OPENAI_API_KEY for OpenAI models).
- Writes composed Python file to <out-dir>/composed_kernel.py and a
composition summary JSON.
"""

from __future__ import annotations

import ast
import argparse
import json
import os
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from triton_kernel_agent.platform_config import (
    get_platform,
    get_platform_choices,
    PlatformConfig,
)

# Reuse KernelAgent provider stack for LLM calls
try:
    from utils.providers.models import get_model_provider
except Exception:
    get_model_provider = None  # type: ignore

# Shared call helper: logs finish_reason / reasoning tokens and handles
# empty-response retries.
from utils.providers.llm_call import call_llm
from utils.llm_budget import COMPOSE_MAX_TOKENS

# Mirror the optimizer's kernel-source policy. The optimization stage rejects any
# kernel file mentioning torch.nn constructs, so catch that at generation time
# instead of failing later with an opaque initial-verification error.
try:
    from triton_kernel_agent.worker import (
        DISALLOWED_TORCH_PATTERNS as _DISALLOWED_TORCH_PATTERNS,
    )
except Exception:  # pragma: no cover - optional dependency
    _DISALLOWED_TORCH_PATTERNS = []

# Reuse extractor to capture clean python code blocks
from .code_extractor import extract_single_python_file

# Reuse Fuser runner for optional verification
from .runner import run_candidate


@dataclass
class KernelItem:
    subgraph_id: str
    kernel_path: Path
    code: str


def _validate_kernel_function_contract(code: str) -> str | None:
    """Validate the composed code's kernel_function follows the unified contract.

    The contract is ``kernel_function(*inputs, model=None)``: primary inputs are
    accepted positionally via ``*inputs``, and all weights/biases/buffers are
    read from an optional ``model`` keyword argument. Returns ``None`` when
    compliant, otherwise a short reason string describing the violation.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"generated code has a syntax error: {exc}"

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "kernel_function":
            args = node.args
            param_names = [a.arg for a in (list(args.args) + list(args.kwonlyargs))]
            if "model" not in param_names:
                return "kernel_function is missing the required `model` parameter"
            if args.vararg is None:
                return "kernel_function must accept positional inputs via `*inputs`"
            return None

    return "no `kernel_function` definition found in generated code"

_TRIPLE_DOUBLE = '"' * 3
_TRIPLE_SINGLE = "'" * 3
_STRIP_COMMENTS_STRINGS_RE = re.compile(
    "("
    + _TRIPLE_DOUBLE
    + r"[\s\S]*?"
    + _TRIPLE_DOUBLE
    + "|"
    + _TRIPLE_SINGLE
    + r"[\s\S]*?"
    + _TRIPLE_SINGLE
    + "|#.*)"
)


def _detect_disallowed_torch_usage(code: str) -> str | None:
    """Return a violation message if ``code`` uses PyTorch in a way the
    optimization stage forbids (notably an embedded torch.nn reference model).

    Comments and string literals are stripped first so that prose such as
    "do not import torch.nn" cannot cause a false positive.
    """
    if not _DISALLOWED_TORCH_PATTERNS:
        return None
    sanitized = _STRIP_COMMENTS_STRINGS_RE.sub("", code)
    for pattern, message in _DISALLOWED_TORCH_PATTERNS:
        if pattern.search(sanitized):
            return message
    return None


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_kernels_from_summary(summary_path: Path) -> list[KernelItem]:
    data = json.loads(_read_text(summary_path))
    if not isinstance(data, list):
        raise SystemExit("kernels summary must be a JSON array (from dispatch step)")
    items: list[KernelItem] = []
    summary_dir = summary_path.parent
    for it in data:
        if not isinstance(it, dict):
            continue
        if not it.get("success"):
            # Skip failed generations
            continue
        sid = str(it.get("id", ""))
        kpath_str = it.get("kernel_path") or ""
        if not sid or not kpath_str:
            continue
        kpath = Path(kpath_str)
        # If path is relative, resolve it relative to summary.json location
        if not kpath.is_absolute():
            kpath = summary_dir / kpath
        if not kpath.is_file():
            continue
        code = _read_text(kpath)
        items.append(KernelItem(subgraph_id=sid, kernel_path=kpath, code=code))
    if not items:
        raise SystemExit("no successful kernels in summary.json")
    return items


def _summarize_subgraphs_for_prompt(subgraphs: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for it in subgraphs:
        sid = str(it.get("id", "unknown"))
        typ = str(it.get("type", ""))
        layout = it.get("data_layout") or "NCHW"
        dtype = it.get("dtype") or "float32"
        inputs = it.get("inputs")
        in_shape = it.get("input_shape")
        out_shape = it.get("output_shape")
        ops = it.get("ops") or []
        # Compact rendering
        shapes_line = (
            f"inputs={inputs if inputs is not None else in_shape}, output={out_shape}"
        )
        lines.append(
            f"- ID={sid} type={typ} layout={layout} dtype={dtype} {shapes_line}"
        )
        try:
            ops_short = json.dumps(ops)[:400]
        except Exception:
            ops_short = str(ops)[:400]
        lines.append(f"  ops={ops_short}")
    return "\n".join(lines)


def _build_composition_prompt(
    problem_code: str,
    subgraphs: list[dict[str, Any]],
    kernel_items: list[KernelItem],
    target_platform: PlatformConfig,
) -> str:
    """Create a single user message to instruct composition by the LLM."""
    # Provide a succinct summary of subgraphs up front
    sg_summary = _summarize_subgraphs_for_prompt(subgraphs)

    # Include only essential snippets from each kernel to keep token usage sane
    # We include full files for now; callers can trim by model limits.
    kernels_section_parts: list[str] = []
    for ki in kernel_items:
        kernels_section_parts.append(
            f"### Subgraph {ki.subgraph_id}\n```python\n" + ki.code + "\n```\n"
        )
    kernels_section = "\n".join(kernels_section_parts)
    # Platform-specific guidance
    platform_guidance = target_platform.guidance_block

    guidance = textwrap.dedent(
        f"""
        You are given:
        - The original problem file (PyTorch module and helpers).
        - A decomposition of the model into fusable subgraphs with exact shapes.
        - Working Triton kernels generated for some subgraphs.

        TARGET PLATFORM: {target_platform.name}
        DEVICE STRING: {target_platform.device_string}
        {platform_guidance}

        Task:
        - Compose an end-to-end Triton implementation that matches the original
          model's forward pass for the provided shapes. You may inline, adapt,
          or reuse the given subgraph kernels. Prefer fusing into as few kernel
          launches as possible while preserving exact numerical semantics.

        Hard requirements:
        - Return ONE complete Python file only, fenced as a single ```python block.
        - Allocate inputs, weights, intermediates, and outputs on device='{target_platform.device_string}' and keep them there throughout forward/verification.
        - CPU is acceptable only for metadata, scalars, and export serialization—avoid `.cpu()` or `.to('cpu')` on compute tensors.
        - Provide at least one @triton.jit kernel and a top-level Python wrapper
          with the EXACT signature:  kernel_function(*inputs, model=None).
          Accept the primary input tensor(s) positionally (in Model.forward
          order) via *inputs. Read ALL required weights, biases, and buffers
          from the `model` argument (an instance of the problem's Model), via
          attribute access such as model.ln_1.weight, model.attn.c_attn.weight,
          or model.attn.bias. Do NOT list individual weight/bias tensors as
          separate positional parameters. When the model has no learnable
          parameters or buffers, `model` may be None and may be unused.
        - Keep this EXACT signature stable across refinement rounds: the
          positional input tensor(s) first, then the optional `model=None`
          keyword parameter. Never rename, reorder, or drop `model`.
        - No PyTorch math path: kernel_function MUST compute the final outputs
          using your Triton kernels only. Do NOT implement or fall back to
          torch.nn / torch.nn.functional / torch.* ops
          sigmoid, etc.) for producing the final result. Using PyTorch for
          reference comparisons is allowed only inside the self-test.
        - Use the data layout and dtype semantics indicated by subgraphs, defaulting
          to NCHW + float32 if unspecified. Respect stride/padding/dilation/groups,
          and exact op order.
        - TRITON-ONLY FILE (critical): this file must contain ONLY Triton
          kernels and plain Python helpers. Do NOT import torch.nn or
          torch.nn.functional, do NOT subclass nn.Module, do NOT declare or
          paste the reference Model (or any other nn.Module) into this file, and
          do NOT call `.forward(...)`. The downstream optimizer statically
          rejects any file containing these constructs, even when they appear
          only in the self-test.
        - Numerical equivalence: include a self-test (test_kernel or run_tests)
          that compares your Triton-based result to a PyTorch reference taken
          from the SIBLING `problem` module (a file named problem.py is placed
          next to yours). Inside the self-test function - not at module top
          level - do:
              import problem
              device = 'cuda'
              model = problem.Model(*problem.get_init_inputs()).to(device).eval()
              inputs = [t.to(device) for t in problem.get_inputs()]
          then call your wrapper as kernel_function(*inputs, model=model) and
          compare against model(*inputs) (invoke the module directly; never call
          `.forward(...)`). The test must print 'PASS' on success and exit with
          code 0. Use torch.allclose with rtol<=1e-3, atol<=1e-3 for fp32; for
          fp16/bf16 allow up to 2e-2.
        - Allowed imports: torch, triton, triton.language as tl, the sibling
          `problem` module, and the Python standard library. No other imports
          and no I/O.
        - Do NOT monkey-patch PyTorch device functions or torch.cuda.is_available()
        - Do NOT manipulate TRITON_BACKENDS environment variable
        - Do NOT disable or mock XPU/CUDA drivers

        Implementation tips:
        - If merging multiple subgraphs, ensure intermediate tensor shapes match.
        - Hoist constant weights or parameters to avoid reloading per block.
        - Use tl.load/tl.store with masks for boundary conditions.
        - Favor coalesced memory access; tile by blocks; compute grid from shape.
        - Common Triton pitfalls to avoid:
          * Do NOT call tl.broadcast on Python scalars; tl.maximum(x, 0.0) works.
          * Prefer scalar constants directly in elementwise ops (no explicit broadcast needed).
          * Keep BLOCK_SIZE power-of-two; mask stores at tail.
        """
    ).strip()

    user_lines: list[str] = []
    user_lines.append(guidance)
    user_lines.append("")
    user_lines.append("SUBGRAPHS (summary):")
    user_lines.append(sg_summary)
    user_lines.append("")
    user_lines.append("ORIGINAL PROBLEM FILE:")
    user_lines.append("```python")
    user_lines.append(problem_code)
    user_lines.append("```")
    user_lines.append("")
    user_lines.append("SUBGRAPH KERNELS (reference implementations):")
    user_lines.append(kernels_section)
    user_lines.append("")
    user_lines.append(
        "Return only one fenced Python code block with your final composed implementation."
    )
    return "\n".join(user_lines)

def _build_refinement_prompt(
    problem_code: str,
    subgraphs: list[dict[str, Any]],
    kernel_items: list[KernelItem],
    previous_code: str,
    error_info: dict[str, str],
    target_platform: PlatformConfig,
) -> str:
    """Prompt the LLM to refine the previously produced code based on errors."""
    err_tail = error_info.get("stderr_tail", "")
    out_tail = error_info.get("stdout_tail", "")

    guidance = textwrap.dedent(
        f"""
        You previously produced a composed Triton implementation, but it failed
        to run/compile. Analyze the ERROR_CONTEXT below and re-emit the entire
        corrected single-file implementation as one ```python block.

        TARGET PLATFORM: {target_platform.name}
        DEVICE STRING: {target_platform.device_string}

        Requirements remain the same. Additionally:
        - Fix any Triton compilation/runtime errors. For scalar constants in
          elementwise ops (e.g., ReLU), do not use tl.broadcast. Use direct
          scalars like 0.0 in tl.maximum(x, 0.0).
        - Keep the EXACT signature kernel_function(*inputs, model=None)
          unchanged (positional inputs first, then `model=None`), and retain
          the self-test that prints PASS on success and exits 0.
        - Keep the file Triton-only: no torch.nn / torch.nn.functional imports,
          no nn.Module subclass, no copied reference Model, no `.forward(...)`
          calls. The self-test must obtain its reference via `import problem`
          (inside the self-test function), not by redefining the model.
        - Do NOT reintroduce any PyTorch math path in kernel_function. The final
          outputs must be computed via your Triton kernels only (no fallback to
          torch.nn / torch.nn.functional ops).
        - Return the complete corrected file; do not send diffs.
        """
    ).strip()

    lines: list[str] = []
    lines.append(guidance)
    lines.append("")
    lines.append("ERROR_CONTEXT (stderr tail):\n```\n" + err_tail + "\n```")
    if out_tail.strip():
        lines.append("STDOUT tail:\n```\n" + out_tail + "\n```")
    lines.append("")
    lines.append("ORIGINAL PROBLEM FILE:\n```python\n" + problem_code + "\n```")
    lines.append("")
    lines.append("SUBGRAPHS (summary):\n" + _summarize_subgraphs_for_prompt(subgraphs))
    lines.append("")
    # Keep previous attempt for reference
    lines.append("PREVIOUS_ATTEMPT:\n```python\n" + previous_code + "\n```")
    lines.append("")
    lines.append(
        "Return only one fenced Python code block with the corrected implementation."
    )
    return "\n".join(lines)

def _auto_patch_common_triton_issues(
    code: str, target_platform: PlatformConfig
) -> tuple[str, bool]:
    """Apply tiny safe textual patches for known Triton pitfalls.

    - Replace tl.broadcast(0.0, ...) or tl.broadcast(1.0, ...) with scalar constants.
    Returns (patched_code, changed).
    """
    patched = code
    changed = False
    # Simple heuristics; keep conservative
    patterns = [
        ("tl.broadcast(0.0", "0.0"),
        ("tl.broadcast(1.0", "1.0"),
        ("tl.broadcast(0,", "0.0"),
        ("tl.broadcast(1,", "1.0"),
    ]
    for old, new in patterns:
        if old in patched:
            patched = patched.replace(old, new)
            changed = True
    # Remove cuda paterns
    cuda_hacks = target_platform.cuda_hacks_to_strip
    if cuda_hacks:
        lines = patched.split("\n")
        filtered_lines = []
        skip_until_blank = False
        for line in lines:
            # Check if we're in a block to skip
            if skip_until_blank:
                if line.strip() == "":
                    skip_until_blank = False
                continue
            # Check if line contains any CUDA hack pattern
            if any(hack in line for hack in cuda_hacks):
                changed = True
                # Start skipping if this is a function definition we need to remove entirely
                if "def _fake_torch_device" in line:
                    skip_until_blank = True
                continue
            filtered_lines.append(line)
        patched = "\n".join(filtered_lines)
    return patched, changed


def compose(
    problem_path: Path,
    subgraphs_path: Path,
    kernels_summary_path: Path,
    out_dir: Path,
    model_name: str,
    verify: bool = False,
    max_iters: int = 5,
    target_platform: str = "cuda",
) -> dict[str, Any]:
    if get_model_provider is None:
        raise SystemExit(
            "KernelAgent providers unavailable; ensure package import and dependencies"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    provider = get_model_provider(model_name)

    # Platform
    platform = get_platform(target_platform)

    # Load inputs
    problem_code = _read_text(problem_path)
    subgraphs = json.loads(_read_text(subgraphs_path))
    if not isinstance(subgraphs, list):
        raise SystemExit("subgraphs.json must be a JSON array")
    kernels = _load_kernels_from_summary(kernels_summary_path)

    attempts_dir = out_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)

    last_usage = None
    last_code = None
    verify_info: dict[str, Any] = {}

    for i in range(1, max_iters + 1):
        if i == 1 or last_code is None:
            prompt = _build_composition_prompt(
                problem_code, subgraphs, kernels, target_platform=platform
            )
        else:
            # Build refinement using previous error info
            stderr_tail = ""
            stdout_tail = ""
            try:
                if verify_info.get("stderr_path"):
                    with open(
                        verify_info["stderr_path"],
                        "r",
                        encoding="utf-8",
                        errors="ignore",
                    ) as f:
                        stderr_tail = f.read()[-2000:]
                if verify_info.get("stdout_path"):
                    with open(
                        verify_info["stdout_path"],
                        "r",
                        encoding="utf-8",
                        errors="ignore",
                    ) as f:
                        stdout_tail = f.read()[-2000:]
            except Exception:
                pass
            prompt = _build_refinement_prompt(
                problem_code,
                subgraphs,
                kernels,
                previous_code=last_code,
                error_info={"stderr_tail": stderr_tail, "stdout_tail": stdout_tail},
                target_platform=platform,
            )

        (attempts_dir / f"attempt_{i}.prompt.txt").write_text(prompt, encoding="utf-8")
        response = call_llm(
            provider,
            model_name,
            [{"role": "user", "content": prompt}],
            label="compose",
            max_tokens=COMPOSE_MAX_TOKENS,
        )
        last_usage = response.usage
        raw_text = response.content or ""
        finish_reason = getattr(response, "finish_reason", None)
        (attempts_dir / f"attempt_{i}.response.txt").write_text(
            raw_text, encoding="utf-8"
        )

        # A length-limited response is incomplete even if it happens to contain
        # a syntactically valid fenced block. Retry instead of compiling it.
        if finish_reason == "length":
            verify_info = {
                "verify_passed": False,
                "verify_reason": "LLM response was truncated at the token limit",
                "finish_reason": finish_reason,
            }
            continue

        try:
            extracted = extract_single_python_file(raw_text)
        except ValueError as exc:
            verify_info = {
                "verify_passed": False,
                "verify_reason": f"invalid LLM code response: {exc}",
                "finish_reason": finish_reason,
            }
            continue
        code = extracted.code
        # Auto-patch trivial Triton pitfalls before running
        code, changed = _auto_patch_common_triton_issues(code, platform)
        (attempts_dir / f"attempt_{i}.py").write_text(code, encoding="utf-8")
        last_code = code

        # Validate the unified kernel_function signature contract
        # (kernel_function(*inputs, model=None)) regardless of self-test result.
        contract_error = _validate_kernel_function_contract(code)
        # A composed file that still embeds torch.nn code passes the in-file
        # self-test but is rejected by the optimizer, so gate on it here too.
        policy_error = _detect_disallowed_torch_usage(code)
        violations = [
            (label, message)
            for label, message in (
                ("signature contract violation", contract_error),
                ("disallowed torch usage", policy_error),
            )
            if message
        ]

        # Verify each attempt if requested
        if verify:
            rr = run_candidate(
                artifacts_code_path=attempts_dir / f"attempt_{i}.py",
                run_root=out_dir / "runs",
                timeout_s=2400,
                isolated=False,
                deny_network=False,
                problem_path=problem_path,
            )
            verify_info = {
                "verify_rc": rr.rc,
                "verify_passed": rr.passed,
                "verify_reason": rr.reason,
                "validator": rr.validator_used,
                "stdout_path": str(rr.stdout_path),
                "stderr_path": str(rr.stderr_path),
            }
            if violations:
                # These override even a passing self-test, so the next
                # refinement round sees them through stderr_tail.
                reason = "; ".join(
                    f"{label}: {message}" for label, message in violations
                )
                verify_info["verify_passed"] = False
                verify_info["verify_reason"] = reason
                verify_info["violations"] = [m for _, m in violations]
                if contract_error:
                    verify_info["contract_violation"] = contract_error
                violation_stderr = out_dir / "policy_violation.txt"
                violation_stderr.write_text(reason, encoding="utf-8")
                verify_info["stderr_path"] = str(violation_stderr)
            if rr.passed and not violations:
                break
        else:
            # If not verifying, stop after first attempt
            break

    # Write final composed file as the last attempt
    composed_path = out_dir / "composed_kernel.py"
    composed_path.write_text(last_code or "", encoding="utf-8")

    result: dict[str, Any] = {
        "success": bool(last_code) and bool(verify_info.get("verify_passed", not verify)),
        "composed_path": str(composed_path.resolve()),
        "model": model_name,
        "usage": last_usage,
        "rounds": i,
        "target_platform": target_platform,
    }
    result.update(verify_info)

    # Persist a small summary
    (out_dir / "composition_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(
        description="Compose end-to-end Triton kernel from subgraphs + generated kernels"
    )
    p.add_argument(
        "--problem", required=True, help="Absolute path to KernelBench problem file"
    )
    p.add_argument(
        "--subgraphs", required=True, help="Path to subgraphs.json from Fuser"
    )
    p.add_argument(
        "--kernels-summary",
        required=True,
        help="Path to summary.json from dispatch step",
    )
    p.add_argument(
        "--out-dir",
        default="compose_out",
        help="Output directory for composed artifacts",
    )
    p.add_argument(
        "--model", default=os.getenv("OPENAI_MODEL") or "gpt-5", help="LLM model name"
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="Execute generated file and check PASS sentinel",
    )
    p.add_argument(
        "--target-platform",
        default="cuda",
        choices=get_platform_choices(),
        help="Target platform (default: cuda)",
    )
    p.add_argument("--max-iters", type=int, default=5, help="Max LLM refinement rounds")
    args = p.parse_args(argv)

    problem_path = Path(args.problem).resolve()
    subgraphs_path = Path(args.subgraphs).resolve()
    kernels_summary_path = Path(args.kernels_summary).resolve()
    out_dir = Path(args.out_dir).resolve()

    if not problem_path.is_file():
        print(f"problem file not found: {problem_path}")
        return 2
    if not subgraphs_path.is_file():
        print(f"subgraphs file not found: {subgraphs_path}")
        return 2
    if not kernels_summary_path.is_file():
        print(f"kernels summary not found: {kernels_summary_path}")
        return 2

    try:
        res = compose(
            problem_path=problem_path,
            subgraphs_path=subgraphs_path,
            kernels_summary_path=kernels_summary_path,
            out_dir=out_dir,
            model_name=args.model,
            verify=args.verify,
            max_iters=args.max_iters,
            target_platform=args.target_platform,
        )
        print(json.dumps(res, indent=2))
        return 0
    except Exception as exc:
        print(f"compose failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
