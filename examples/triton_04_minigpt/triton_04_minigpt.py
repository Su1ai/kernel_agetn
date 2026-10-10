#!/usr/bin/env python3
"""Run the MiniGPT extract/dispatch/compose pipeline and optimizer.

Launch commands:
    Generator stage:  python examples/triton_04_minigpt/triton_04_minigpt.py --stage generate
    Optimizer stage:  python examples/triton_04_minigpt/triton_04_minigpt.py --stage optimize
    Both stages:      python examples/triton_04_minigpt/triton_04_minigpt.py --stage all
"""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

import yaml
from dotenv import load_dotenv

from Fuser.config import new_run_id
from Fuser.pipeline import run_pipeline
from triton_kernel_agent.opt_manager import OptimizationManager

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = Path(__file__).resolve().parent
PROBLEM_PATH = EXAMPLE_DIR / "problem.py"
TEST_PATH = EXAMPLE_DIR / "test.py"
CONFIG_PATH = ROOT / "examples/configs/beam_search_04_test.yaml"
FUSE_ROOT = ROOT / ".fuse"

# Generator concurrency is independent of optimizer num_workers in the YAML.
GENERATOR_WORKERS = 1
DISPATCH_JOBS = 1


def _compose_succeeded(run_dir: Path) -> bool:
    summary_path = run_dir / "compose_out/composition_summary.json"
    composed_path = run_dir / "compose_out/composed_kernel.py"
    if not summary_path.is_file() or not composed_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(summary.get("success") and summary.get("verify_passed"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_matches_problem(run_dir: Path, problem_path: Path) -> bool:
    """Return whether compose validated this run against the current problem."""
    expected = _sha256(problem_path)
    snapshots = sorted(
        (run_dir / "compose_out/runs").glob("*/problem.py"), reverse=True
    )
    return any(_sha256(snapshot) == expected for snapshot in snapshots)


def find_latest_successful_run(
    out_root: Path, problem_path: Path = PROBLEM_PATH
) -> Path | None:
    for run_dir in sorted(out_root.glob("run_*"), reverse=True):
        if _compose_succeeded(run_dir) and _run_matches_problem(run_dir, problem_path):
            return run_dir
    return None


def load_config() -> dict:
    data = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {CONFIG_PATH}")
    return data


# Generator entry point -- launched with:
#     python examples/triton_04_minigpt/triton_04_minigpt.py --stage generate
# Extraction, kernel dispatch, and whole-model composition all happen
# inside run_pipeline(); results land in a timestamped .fuse/run_* dir.
def generate() -> Path:
    config = load_config()
    result = run_pipeline(
        problem_path=PROBLEM_PATH,
        extract_model=config["extract_model"],
        dispatch_model=config["dispatch_model"],
        compose_model=config["compose_model"],
        dispatch_jobs=DISPATCH_JOBS,
        workers=GENERATOR_WORKERS,
        max_iters=3,
        llm_timeout_s=1200,
        run_timeout_s=1200,
        out_root=FUSE_ROOT,
        verify=True,
        compose_max_iters=5,
        target_platform=config.get("target_platform", "cuda"),
        test_timeout_s=1200,
    )
    run_dir = Path(result["run_dir"])
    print(f"Generation completed: {run_dir}")
    return run_dir


# Optimizer entry point -- launched with:
#     python examples/triton_04_minigpt/triton_04_minigpt.py --stage optimize
# It consumes the most recent successful .fuse/run_* composed kernel
# (unless run_dir is given) and writes to a timestamped .fuse/opt_* dir.
def optimize(run_dir: Path | None = None) -> Path:
    run_dir = run_dir or find_latest_successful_run(FUSE_ROOT)
    if run_dir is None:
        raise RuntimeError(f"no successful composed run found under {FUSE_ROOT}")

    kernel_path = run_dir / "compose_out/composed_kernel.py"

    # Every optimizer run gets its own timestamped directory so a rerun never
    # overwrites the previous run's logs, artifacts, or program database.
    opt_run_id = new_run_id().replace("run_", "opt_", 1)
    opt_log_dir = FUSE_ROOT / opt_run_id
    manager = OptimizationManager(
        config=str(CONFIG_PATH),
        log_dir=opt_log_dir,
        database_path=opt_log_dir / "program_db.json",
    )
    result = manager.run_optimization(
        initial_kernel=kernel_path.read_text(),
        problem_file=PROBLEM_PATH,
        test_code=TEST_PATH.read_text(),
    )
    if not result.get("success") or not result.get("kernel_code"):
        raise RuntimeError(result.get("error", "optimization failed"))

    output_path = EXAMPLE_DIR / "optimized_kernel.py"
    temporary_output = output_path.with_suffix(".py.tmp")
    temporary_output.write_text(result["kernel_code"])
    temporary_output.replace(output_path)
    print(f"Optimizer logs: {opt_log_dir}")
    print(f"Optimized kernel: {output_path}")
    print("Best time: {:.4f} ms".format(result["best_time_ms"]))
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("generate", "optimize", "all"), default="all")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")

    generated_run = None
    # Generator: python <this file> --stage generate
    if args.stage in {"generate", "all"}:
        generated_run = generate()
    # Optimizer: python <this file> --stage optimize
    if args.stage in {"optimize", "all"}:
        optimize(run_dir=generated_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
