#!/usr/bin/env python3
"""Run the Fuser generation pipeline and/or the KernelAgent optimizer for a task.

Every task is a self-contained directory under ``tt_work/``::

    tt_work/<task>/
        problem.py                    # the model under test
        test.py                       # correctness test for each candidate

Run outputs land in the shared ``.fuse/`` root, whose names carry the task
label so several tasks stay distinguishable::

    .fuse/run_<task>_<timestamp>/     # generator output (--stage generate)
    .fuse/opt_<task>_<timestamp>/     # optimizer output (--stage optimize)
    .fuse/optimized_kernel_<task>.py  # best kernel of the last optimize run

Both stages read their own config file:

- generator: ``Fuser/config/autoagent_ppu.yml``
- optimizer: ``examples/configs/beam_search_04_test2.yaml``

Usage::

    python run_pipeline.py --stage all      --task relu_self_attention_bf16
    python run_pipeline.py --stage generate --task triton_04_minigpt
    python run_pipeline.py --stage optimize --task relu_self_attention_bf16

``--task`` may be omitted when ``tt_work`` holds exactly one task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Make direct execution from the repository root independent of an editable install.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
from dotenv import load_dotenv

from Fuser.config import new_run_id
from Fuser.paths import FUSE_ROOT, TT_WORK_ROOT
from Fuser.pipeline import run_pipeline
from triton_kernel_agent.opt_manager import OptimizationManager

TT_WORK = TT_WORK_ROOT          # task inputs (problem.py / test.py)
OUT_ROOT = FUSE_ROOT            # run + opt outputs, labelled by task
GENERATOR_CONFIG_PATH = ROOT / "Fuser/config/autoagent_ppu.yml"
OPTIMIZER_CONFIG_PATH = ROOT / "examples/configs/beam_search_04_test2.yaml"

# Fallback generator concurrency, used only when the config file omits
# ``workers`` / ``dispatch_jobs``.  The configured values take precedence.
GENERATOR_WORKERS = 1
DISPATCH_JOBS = 1


# ---------------------------------------------------------------------------
# Task discovery
# ---------------------------------------------------------------------------


def discover_tasks() -> list[Path]:
    """Return every ``tt_work`` subdirectory that looks like a task."""
    if not TT_WORK.is_dir():
        return []
    return sorted(
        d
        for d in TT_WORK.iterdir()
        if d.is_dir() and (d / "problem.py").is_file() and (d / "test.py").is_file()
    )


def resolve_task(name: str | None) -> Path:
    """Resolve a task directory under ``tt_work``.

    With no name, and exactly one task present, that task is used.  Otherwise
    the ambiguity is reported instead of guessing.
    """
    if name:
        task_dir = (TT_WORK / name) if not Path(name).is_absolute() else Path(name)
        if not (task_dir / "problem.py").is_file():
            raise SystemExit(f"task '{name}' has no problem.py: {task_dir}")
        if not (task_dir / "test.py").is_file():
            raise SystemExit(f"task '{name}' has no test.py: {task_dir}")
        return task_dir

    tasks = discover_tasks()
    if not tasks:
        raise SystemExit(
            f"no task found under {TT_WORK}. Create tt_work/<task>/ with "
            f"problem.py and test.py, or pass --task."
        )
    if len(tasks) > 1:
        names = ", ".join(t.name for t in tasks)
        raise SystemExit(f"multiple tasks found ({names}); pass --task <name>.")
    return tasks[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    out_root: Path, problem_path: Path, label: str | None = None
) -> Path | None:
    """Newest generator run that composed successfully for *problem_path*.

    Runs labelled for *label* are considered first; the shared output root may
    also hold runs from other tasks, and those are filtered out by comparing the
    problem snapshot recorded at compose time.
    """
    patterns = ([f"run_{label}_*"] if label else []) + ["run_*"]
    seen: set[Path] = set()
    for pattern in patterns:
        for run_dir in sorted(out_root.glob(pattern), reverse=True):
            if run_dir in seen:
                continue
            seen.add(run_dir)
            if _compose_succeeded(run_dir) and _run_matches_problem(
                run_dir, problem_path
            ):
                return run_dir
    return None


def load_config(config_path: Path) -> dict:
    data = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {config_path}")
    return data


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def generate(task_dir: Path, generator_config: Path = GENERATOR_CONFIG_PATH) -> Path:
    """Run extract -> dispatch -> compose for *task_dir*.

    Every generator knob comes from the config file so tuning happens in one
    place; the constants below are only fallbacks for configs that omit a key.
    Output lands in ``<task_dir>/run_<timestamp>/``.
    """
    config = load_config(generator_config)
    result = run_pipeline(
        problem_path=task_dir / "problem.py",
        extract_model=config["extract_model"],
        dispatch_model=config["dispatch_model"],
        compose_model=config["compose_model"],
        dispatch_jobs=config.get("dispatch_jobs", DISPATCH_JOBS),
        workers=config.get("workers", GENERATOR_WORKERS),
        max_iters=config.get("max_iters", 3),
        llm_timeout_s=config.get("llm_timeout_s", 1200),
        run_timeout_s=config.get("run_timeout_s", 1200),
        out_root=OUT_ROOT,
        run_label=task_dir.name,
        verify=config.get("verify", True),
        compose_max_iters=config.get("compose_max_iters", 5),
        target_platform=config.get("target_platform", "cuda"),
        test_timeout_s=config.get("test_timeout_s", 1200),
    )
    run_dir = Path(result["run_dir"])
    print(f"Generation completed: {run_dir}")
    return run_dir


def optimize(
    task_dir: Path,
    optimizer_config: Path = OPTIMIZER_CONFIG_PATH,
    run_dir: Path | None = None,
) -> Path:
    """Optimize the composed kernel of *task_dir*.

    Consumes the newest successful ``run_*`` composed kernel unless *run_dir*
    is given, and writes a timestamped ``opt_*`` directory next to it, so a
    rerun never overwrites earlier logs or the program database.
    """
    if not optimizer_config.is_file():
        raise SystemExit(f"optimizer config not found: {optimizer_config}")

    run_dir = run_dir or find_latest_successful_run(
        OUT_ROOT, task_dir / "problem.py", label=task_dir.name
    )
    if run_dir is None:
        raise RuntimeError(f"no successful composed run found under {OUT_ROOT}")

    kernel_path = run_dir / "compose_out/composed_kernel.py"

    opt_run_id = new_run_id(task_dir.name).replace("run_", "opt_", 1)
    opt_log_dir = OUT_ROOT / opt_run_id
    manager = OptimizationManager(
        config=str(optimizer_config),
        log_dir=opt_log_dir,
        database_path=opt_log_dir / "program_db.json",
    )
    result = manager.run_optimization(
        initial_kernel=kernel_path.read_text(),
        problem_file=task_dir / "problem.py",
        test_code=(task_dir / "test.py").read_text(),
    )
    if not result.get("success") or not result.get("kernel_code"):
        raise RuntimeError(result.get("error", "optimization failed"))

    output_path = OUT_ROOT / f"optimized_kernel_{task_dir.name}.py"
    temporary_output = output_path.with_suffix(".py.tmp")
    temporary_output.write_text(result["kernel_code"])
    temporary_output.replace(output_path)
    print(f"Optimizer logs: {opt_log_dir}")
    print(f"Optimized kernel: {output_path}")
    print("Best time: {:.4f} ms".format(result["best_time_ms"]))
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the generator and/or optimizer for a task in tt_work/",
    )
    parser.add_argument(
        "--stage", choices=("generate", "optimize", "all"), default="all"
    )
    parser.add_argument(
        "--task",
        default=None,
        help="task directory name under tt_work/ (optional when only one exists)",
    )
    parser.add_argument(
        "--generator-config", type=Path, default=GENERATOR_CONFIG_PATH
    )
    parser.add_argument(
        "--optimizer-config", type=Path, default=OPTIMIZER_CONFIG_PATH
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    task_dir = resolve_task(args.task)
    print(f"Task: {task_dir}")

    generated_run = None
    if args.stage in {"generate", "all"}:
        generated_run = generate(task_dir, args.generator_config)
    if args.stage in {"optimize", "all"}:
        optimize(task_dir, args.optimizer_config, run_dir=generated_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
