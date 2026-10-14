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

"""DeepSeek provider using its OpenAI-compatible Chat Completions API.

Served either by the shared ADVibe gateway or by the official API at
``https://api.deepseek.com``; see :mod:`utils.providers.credentials`.

Implements the official thinking-mode controls documented at
https://api-docs.deepseek.com/guides/thinking_mode/ :

- Thinking switch  -> ``extra_body={"thinking": {"type": "enabled"|"disabled"}}``
  (a DeepSeek-specific field, so it is sent via ``extra_body`` rather than as a
  top-level kwarg the OpenAI SDK would not forward).
- Reasoning strength -> top-level ``reasoning_effort`` (``low`` / ``high`` /
  ``max``, with the documented aliases normalised onto those three).

Both are configurable through the environment (``THINKING`` and
``REASONING_EFFORT``) and overridable per call via ``thinking=...`` /
``reasoning_effort=...`` kwargs.
"""

import os

from .credentials import api_key_env_name, base_url
from .openai_base import OpenAICompatibleProvider


class DeepSeekProvider(OpenAICompatibleProvider):
    """OpenAI-compatible provider for the official DeepSeek API."""

    DEFAULT_BASE_URL = "https://api.deepseek.com"

    # The official API collapses a range of effort spellings onto three values
    # (see the thinking-mode docs).  We normalise so the config file stays
    # forgiving while the request always carries a value the API accepts.
    _EFFORT_NORMALIZE = {
        "minimal": "low",
        "low": "low",
        "medium": "high",
        "high": "high",
        "xhigh": "high",
        "max": "max",
        "ultra": "max",
    }

    def __init__(self):
        # Shared gateway when one is configured, else the official endpoint.
        super().__init__(
            api_key_env=api_key_env_name("deepseek"),
            base_url=base_url(self.DEFAULT_BASE_URL, vendor="deepseek"),
        )

    @property
    def name(self) -> str:
        return "deepseek"

    @staticmethod
    def _normalize_thinking(value: object) -> str:
        """Normalise a thinking switch to ``"enabled"`` / ``"disabled"``."""
        if isinstance(value, bool):
            return "enabled" if value else "disabled"
        text = str(value).strip().lower()
        if text in {"enabled", "on", "true", "1", "yes"}:
            return "enabled"
        if text in {"disabled", "off", "false", "0", "no"}:
            return "disabled"
        raise ValueError(
            f"invalid thinking value {value!r}: expected enabled/disabled"
        )

    @classmethod
    def _normalize_effort(cls, value: object) -> str:
        """Normalise a reasoning-effort value onto ``low`` / ``high`` / ``max``."""
        text = str(value).strip().lower()
        if text not in cls._EFFORT_NORMALIZE:
            raise ValueError(
                f"invalid reasoning_effort {value!r}: expected one of "
                f"{sorted(set(cls._EFFORT_NORMALIZE.values()))}"
            )
        return cls._EFFORT_NORMALIZE[text]

    def _build_api_params(self, model_name, messages, **kwargs):
        params = super()._build_api_params(model_name, messages, **kwargs)

        # Thinking switch (explicit per-call value wins over the environment).
        thinking = kwargs.get("thinking")
        if thinking is None:
            thinking = os.getenv("THINKING")
        if thinking is not None and str(thinking).strip() != "":
            extra_body = dict(params.get("extra_body") or {})
            extra_body["thinking"] = {"type": self._normalize_thinking(thinking)}
            params["extra_body"] = extra_body

        # Reasoning strength (explicit per-call value wins over the environment).
        effort = kwargs.get("reasoning_effort")
        if effort is None:
            effort = os.getenv("REASONING_EFFORT")
        if effort:
            params["reasoning_effort"] = self._normalize_effort(effort)

        return params

    def get_max_tokens_limit(self, model_name: str) -> int:
        # The API remains authoritative and may accept less.
        return 65536

    def supports_multiple_completions(self) -> bool:
        # DeepSeek-compatible gateways do not consistently implement ``n``.
        return False
