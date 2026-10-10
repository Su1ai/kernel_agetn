#!/usr/bin/env python3
"""Extract Softmax/RMSNorm subgraphs, save the fused PyTorch model, and
generate four unverified Triton kernels each."""

import concurrent.futures
import json
import os
import re
import shutil
import tarfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from Fuser.subgraph_extractor import extract_subgraphs_to_json
from triton_kernel_agent.platform_config import get_platform
from triton_kernel_agent.prompt_manager import PromptManager
from utils.providers import get_model_provider
from utils.llm_budget import EXTRACT_MAX_TOKENS


# ---- Models and generation settings --------------------------------------
EXTRACT_MODEL = "deepseek-v4-pro"
DISPATCH_MODEL = "deepseek-v4-flash"
EXTRACT_WORKERS = 1                  #extract模型数量（并行度）
MAX_EXTRACT_ITERS = 2                #extract工作最大迭代次数（模型一次提取子图可能并不细致）
GENERATION_WORKERS = 4               #kernel生成模型并行个数
# -------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
KERNELBENCH_LEVEL1 = ROOT.parent / "KernelBench" / "KernelBench" / "level1"
KERNELBENCH_LEVEL3 = ROOT.parent / "KernelBench" / "KernelBench" / "level3"
LOCAL_RUN_DIR = ROOT / "local_extract_run"          #保存运行中数据
PROBLEMS = (                                       #找到KernelBench中的mingpt model，重构shape，用dict存储name，sorce，和rule
    {
        "name": "mingpt_block",
        "source": KERNELBENCH_LEVEL3 / "44_MiniGPTBlock.py",
        "replacements": {
            "batch_size = 128": "batch_size = 1",
            "max_seqlen = 1024": "max_seqlen = 16",
            "seq_len = 512": "seq_len = 16",
            "n_embd = 768": "n_embd = 32",
            "n_head = 8": "n_head = 4",
            "attn_pdrop = 0.0": "attn_pdrop = 0.0",
            "resid_pdrop = 0.0": "resid_pdrop = 0.0",
        },
    },
)


def _required_key(model: str) -> str:
    if model.startswith("deepseek"):
        return "DEEPSEEK_API_KEY"
    if model.startswith("claude"):
        return "ANTHROPIC_API_KEY"
    return "OPENAI_API_KEY"         #否则默认openai key


def _check_configuration() -> None:
    for model in {EXTRACT_MODEL, DISPATCH_MODEL}:
        key_name = _required_key(model)
        if not os.getenv(key_name):      #从env中找key_name，找不到返回none
            raise RuntimeError(f"Set {key_name} in {ROOT / '.env'}")
    for problem in PROBLEMS:
        if not problem["source"].is_file():
            raise FileNotFoundError(f"KernelBench problem not found: {problem['source']}")


def _make_small_problem(problem: dict[str, Any], run_dir: Path) -> Path:        #替换problem的shape
    problems = problem["source"].read_text(encoding="utf-8")
    for original, replacement in problem["replacements"].items():
        if original not in problems:
            raise RuntimeError(f"expected fixture line not found: {original}")
        problems = problems.replace(original, replacement, 1)
    destination = run_dir / "input" / f"{problem['name']}_small.py"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(problems, encoding="utf-8")
    return destination


def _extract_subgraphs(problem_path: Path, run_dir: Path) -> list[dict[str, Any]]:
    official_run, generated_json = extract_subgraphs_to_json(
        problem_path=problem_path,
        model_name=EXTRACT_MODEL,
        workers=EXTRACT_WORKERS,
        max_iters=MAX_EXTRACT_ITERS,
        llm_timeout_s=1200,
        run_timeout_s=300,
        target_platform="cuda",
    )
    output_json = run_dir / "subgraphs.json"
    shutil.copyfile(generated_json, output_json)          #将生成的json保存到output路径
    subgraphs = json.loads(output_json.read_text(encoding="utf-8"))
    if not isinstance(subgraphs, list) or not subgraphs:
        raise RuntimeError("subgraphs.json must contain a non-empty array")
    _save_fused_model(official_run, run_dir)       
    print(f"official extract run: {official_run}")
    print(f"saved JSON: {output_json}")
    return subgraphs


def _read_winner(run_dir: Path) -> tuple[str, str]:           #没有跑优化流水线，闲置
    worker_id = "worker_01"
    iteration = "1"
    winner_worker = run_dir / "orchestrator" / "winner" / "worker_id.txt"
    winner_iter = run_dir / "orchestrator" / "winner" / "iter.txt"
    if winner_worker.is_file():
        worker_id = winner_worker.read_text(encoding="utf-8").strip()
    if winner_iter.is_file():
        iteration = winner_iter.read_text(encoding="utf-8").strip()
    return worker_id, iteration


def _extract_code_from_tar(tarball: Path) -> str | None:
    """Extract the fused model's code.py from the winner artifact tarball."""
    if not tarball.is_file():
        return None
    try:
        with tarfile.open(tarball, "r:gz") as tf:
            names = tf.getnames()
            if "code.py" not in names:
                return None
            member = tf.getmember("code.py")
            extracted = tf.extractfile(member)
            if extracted is None:
                return None
            return extracted.read().decode("utf-8")
    except (OSError, KeyError, tarfile.TarError):
        return None


def _save_fused_model(official_run: Path, run_dir: Path) -> None:
    """Copy the fused PyTorch model and its prompt into the local run folder."""
    fused_code: str | None = None

    # Prefer the winner artifact tarball (orchestrator summary).
    summary_path = official_run / "orchestrator" / "summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
        artifact = summary.get("artifact_path")
        if artifact:
            fused_code = _extract_code_from_tar(Path(artifact))

    # Fallback: winner worker's iteration artifacts.
    if not fused_code:
        worker_id, iteration = _read_winner(official_run)
        candidate = (
            official_run
            / "workers"
            / worker_id
            / "artifacts"
            / f"iteration_{iteration}"
            / "code.py"
        )
        if candidate.is_file():
            fused_code = candidate.read_text(encoding="utf-8")

    if fused_code:
        (run_dir / "fused_model.py").write_text(fused_code + "\n", encoding="utf-8")
        print(f"saved fused model: {run_dir / 'fused_model.py'}")
    else:
        print(f"WARNING: fused model code not found in {official_run}")

    # Save the fusion prompt that asked the LLM to refactor the model.
    worker_id, iteration = _read_winner(official_run)
    prompt_path = (
        official_run / "workers" / worker_id / "prompts" / f"iteration_{iteration}.txt"
    )
    if prompt_path.is_file():
        shutil.copyfile(prompt_path, run_dir / "fused_prompt.txt")
        print(f"saved fused prompt: {run_dir / 'fused_prompt.txt'}")


def _generation_problem(item: dict[str, Any]) -> str:
    source = item.get("source") if isinstance(item.get("source"), dict) else {}
    source_code = source.get("code") or "No source snippet was returned."
    return (
        "Implement the following extracted PyTorch subgraph as a complete Triton "
        "Python file. The wrapper must be named kernel_function. Preserve the exact "
        "shapes, dtype, dimensions, epsilon values, and operation semantics.\n\n"
        "SUBGRAPH JSON:\n"
        f"```json\n{json.dumps(item, indent=2, ensure_ascii=False)}\n```\n\n"
        "EXTRACTED PYTORCH SOURCE:\n"
        f"```python\n{source_code}\n```"
    )


def _extract_python(response: str) -> str | None:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip()
    return response.strip() or None


def _generate_one(worker_id: int, worker_dir: Path, prompt: str) -> dict[str, Any]:
    worker_dir.mkdir(parents=True, exist_ok=True)
    (worker_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    try:
        provider = get_model_provider(DISPATCH_MODEL)
        response = provider.get_response(
            DISPATCH_MODEL,
            [{"role": "user", "content": prompt}],
            max_tokens=EXTRACT_MAX_TOKENS,
            temperature=0.75 + worker_id * 0.05,
            high_reasoning_effort=True,
        )
        raw_reply = response.content or ""
        (worker_dir / "reply.txt").write_text(raw_reply, encoding="utf-8")
        kernel = _extract_python(raw_reply)
        if not kernel:
            raise RuntimeError("the model reply contained no kernel text")
        kernel_path = worker_dir / "kernel.py"
        kernel_path.write_text(kernel + "\n", encoding="utf-8")
        return {
            "worker_id": worker_id,
            "success": True,
            "kernel_path": str(kernel_path),
            "validation": "not_run",
        }
    except Exception as exc:
        return {
            "worker_id": worker_id,
            "success": False,
            "error": str(exc),
            "validation": "not_run",
        }


def _dispatch_generation(
    problem_name: str, subgraphs: list[dict[str, Any]], run_dir: Path
) -> None:
    prompt_manager = PromptManager(target_platform=get_platform("cuda"))
    kernels_dir = run_dir / "generated_kernels"
    summary: list[dict[str, Any]] = []

    for index, item in enumerate(subgraphs, start=1):
        subgraph_id = str(item.get("id") or f"subgraph_{index}").replace("/", "_")
        subgraph_dir = kernels_dir / subgraph_id
        subgraph_dir.mkdir(parents=True, exist_ok=True)
        problem = _generation_problem(item)
        (subgraph_dir / "problem.txt").write_text(problem, encoding="utf-8")
        prompt = prompt_manager.render_kernel_generation_prompt(
            problem_description=problem,
            test_code=(
                "Generation-only mode: no executable test is supplied. "
                "Implement the exact JSON and source semantics."
            ),
        )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=GENERATION_WORKERS
        ) as executor:
            futures = [
                executor.submit(
                    _generate_one,
                    worker_id,
                    subgraph_dir / f"worker_{worker_id:02d}",
                    prompt,
                )
                for worker_id in range(1, GENERATION_WORKERS + 1)
            ]
            results = [future.result() for future in futures]
        summary.append({"subgraph_id": subgraph_id, "workers": results})

    summary_path = kernels_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    generated = sum(r["success"] for entry in summary for r in entry["workers"])
    expected = len(subgraphs) * GENERATION_WORKERS
    print(f"{problem_name}: generated kernels {generated}/{expected}")
    print(f"{problem_name}: summary {summary_path}")
    if generated != expected:
        raise RuntimeError(f"{problem_name}: only {generated}/{expected} generated")


def main() -> None:
    load_dotenv(ROOT / ".env")
    _check_configuration()
    print(f"extract model: {EXTRACT_MODEL}")
    print(f"dispatch model: {DISPATCH_MODEL}")

    for problem in PROBLEMS:
        name = problem["name"]
        print(f"\n=== {name} ===")
        run_dir = LOCAL_RUN_DIR / name
        small_problem = _make_small_problem(problem, run_dir)
        subgraphs = _extract_subgraphs(small_problem, run_dir)
        _dispatch_generation(name, subgraphs, run_dir)

    print("\nvalidation, compilation, execution, and composition were not run")


if __name__ == "__main__":
    main()
