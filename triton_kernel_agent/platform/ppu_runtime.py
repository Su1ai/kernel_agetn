"""Runtime helpers shared by PPU platform adapters.

PPU compiles kernels at runtime and refuses to start when its SDK variables are
missing, so bubbles the environment up to every child process that may compile
or launch a kernel.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_PPU_SDK = "/usr/local/PPU_SDK"

# Library directories the PPU runtime and CUDA-compatibility layer need at
# load time. Kept explicit rather than inheriting LD_LIBRARY_PATH wholesale so a
# sandboxed verification child only receives the paths the accelerator needs.
_PPU_LIB_SUBDIRS = (
    "lib",
    "CUDA_SDK/lib64",
    "sailSHMEM/lib",
)

_PPU_BIN_SUBDIRS = (
    "bin",
    "CUDA_SDK/bin",
    "asight/bin",
    "ppu-smi/bin",
)


def _sdk_root() -> Path:
    return Path(os.environ.get("PPU_SDK") or DEFAULT_PPU_SDK)


def _prepend_path_list(
    current: str | None, additions: list[str], *, separator: str = os.pathsep
) -> str:
    seen: set[str] = set()
    ordered: list[str] = []
    for entry in list(additions) + [e for e in (current or "").split(separator) if e]:
        if entry and entry not in seen:
            seen.add(entry)
            ordered.append(entry)
    return separator.join(ordered)


def ppu_environment(overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Return the PPU variables needed to compile and launch kernels.

    Values already present in ``os.environ`` win, so an explicitly activated SDK
    (``source envsetup.sh ppu``) is never overridden. ``overrides`` optionally
    replaces variables for a specific child process.
    """
    sdk_root = _sdk_root()
    cuda_sdk = sdk_root / "CUDA_SDK"

    env: dict[str, str] = {
        "PPU_SDK": os.environ.get("PPU_SDK") or str(sdk_root),
        "PPU_HOME": os.environ.get("PPU_HOME") or str(sdk_root),
        "PPU_PATH": os.environ.get("PPU_PATH") or str(sdk_root),
        "CUDA_PATH": os.environ.get("CUDA_PATH") or str(cuda_sdk),
    }
    env["CUDA_HOME"] = os.environ.get("CUDA_HOME") or str(cuda_sdk)
    env["CUDA_TOOLKIT_ROOT"] = os.environ.get("CUDA_TOOLKIT_ROOT") or str(cuda_sdk)

    lib_dirs = [str(sdk_root / sub) for sub in _PPU_LIB_SUBDIRS]
    env["LD_LIBRARY_PATH"] = _prepend_path_list(
        os.environ.get("LD_LIBRARY_PATH"), lib_dirs
    )

    bin_dirs = [str(sdk_root / sub) for sub in _PPU_BIN_SUBDIRS]
    env["PATH"] = _prepend_path_list(os.environ.get("PATH"), bin_dirs)

    if overrides:
        env.update({k: v for k, v in overrides.items() if v is not None})
    return env


def ensure_ppu_environment() -> None:
    """Fill missing SDK variables in the current process."""
    for key, value in ppu_environment().items():
        if key in ("PATH", "LD_LIBRARY_PATH"):
            continue
        if not os.environ.get(key):
            os.environ[key] = value


def inject_ppu_environment(env: dict[str, str], *, enabled: bool = True) -> dict[str, str]:
    """Add PPU SDK variables to a child-process environment dict in place.

    ``enabled`` lets the caller skip injection on non-PPU platforms so the
    NVIDIA path keeps its existing allow-list behaviour unchanged.
    """
    if not enabled:
        return env
    for key, value in ppu_environment().items():
        if key in ("PATH", "LD_LIBRARY_PATH"):
            env[key] = value
        else:
            env.setdefault(key, value)
    return env
