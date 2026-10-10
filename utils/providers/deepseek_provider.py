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

"""DeepSeek provider using its OpenAI-compatible Chat Completions API."""

import os

from .openai_base import OpenAICompatibleProvider


class DeepSeekProvider(OpenAICompatibleProvider):
    """OpenAI-compatible provider for official or proxied DeepSeek APIs."""

    DEFAULT_BASE_URL = "https://api.deepseek.com"
    supports_enable_thinking = True

    def __init__(self):
        super().__init__(
            api_key_env="DEEPSEEK_API_KEY",
            base_url=os.getenv("DEEPSEEK_BASE_URL", self.DEFAULT_BASE_URL),
        )

    @property
    def name(self) -> str:
        return "deepseek"

    @staticmethod
    def _parse_optional_bool(value: object, *, source: str) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        raise ValueError(
            f"{source} must be a boolean or one of true/false, 1/0, yes/no, on/off"
        )

    def _build_api_params(self, model_name, messages, **kwargs):
        params = super()._build_api_params(model_name, messages, **kwargs)

        # An explicit per-call option takes precedence over the environment.
        # Keep the option in extra_body so the OpenAI SDK forwards this
        # DeepSeek-specific field without rejecting an unknown top-level kwarg.
        raw_value = kwargs.get("enable_thinking")
        source = "enable_thinking"
        if raw_value is None:
            raw_value = os.getenv("DEEPSEEK_ENABLE_THINKING")
            source = "DEEPSEEK_ENABLE_THINKING"
        enabled = self._parse_optional_bool(raw_value, source=source)
        if enabled is not None:
            extra_body = dict(params.get("extra_body") or {})
            extra_body["enable_thinking"] = enabled
            params["extra_body"] = extra_body
        return params

    def get_max_tokens_limit(self, model_name: str) -> int:
        # Compose can require more than 32K output tokens for full-model kernels.
        # The upstream gateway remains authoritative and may reject a lower limit.
        return 65536

    def supports_multiple_completions(self) -> bool:
        # DeepSeek-compatible gateways do not consistently implement `n`.
        # Individual requests are slower but work across more deployments.
        return False
