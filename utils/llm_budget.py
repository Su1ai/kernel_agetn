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

"""Centralised output-token budgets for every LLM call site.

Reasoning-capable models (e.g. ``deepseek-v4.1-flash``) bill their thinking
trace against the same output budget as the answer.  A budget that is too small
therefore returns an *empty* response with ``finish_reason=length`` rather than a
truncated one, which is easy to mistake for a parse failure downstream.

The shared gateway offers no reliable switch for the thinking trace -- it
ignores ``enable_thinking`` and its handling of ``reasoning_effort`` is
model-dependent -- so the output budget is the one lever that always holds: it guarantees
room for the answer after the thinking trace takes its share.  Keep the values
here and override them per deployment through the environment instead of editing
each call site.
"""

from __future__ import annotations

import os
from pathlib import Path

# These budgets are module-level constants, so they are evaluated the moment
# this module is imported.  Entry points load ``.env`` inside ``main()``, which
# runs *after* the pipeline modules (and therefore this one) have been
# imported, so without this the environment overrides would silently fall back
# to the defaults below.  Load the project file first; ``override=False`` keeps
# a real environment variable authoritative over the file.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except Exception:  # python-dotenv absent, or no .env in this deployment
    pass


def _env_int(name: str, default: int) -> int:
    """Read a positive integer from the environment, falling back to ``default``."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


# Kernel / fused-source generation: must leave room for a full file after the
# thinking trace consumes its share.  Observed traces reach ~15K tokens.
CODE_GEN_MAX_TOKENS = _env_int("LLM_MAX_TOKENS_CODE_GEN", 32768)

# Bottleneck / roofline analysis over a kernel plus problem source.
ANALYSIS_MAX_TOKENS = _env_int("LLM_MAX_TOKENS_ANALYSIS", 32768)

# Reflexion and other small structured (JSON) responses.
REFLEXION_MAX_TOKENS = _env_int("LLM_MAX_TOKENS_REFLEXION", 16384)

# Whole-model composition, which emits several kernels in one response.
COMPOSE_MAX_TOKENS = _env_int("LLM_MAX_TOKENS_COMPOSE", 50000)

# Subgraph extraction prompts.
EXTRACT_MAX_TOKENS = _env_int("LLM_MAX_TOKENS_EXTRACT", 20000)

# Interactive UI / smoke scripts.
UI_MAX_TOKENS = _env_int("LLM_MAX_TOKENS_UI", 8000)
