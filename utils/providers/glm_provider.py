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

"""GLM provider for the shared OpenAI-compatible gateway.

GLM models are served either by the shared ADVibe gateway or by Zhipu's
official API (``open.bigmodel.cn``); :mod:`utils.providers.credentials` picks
between them from the environment.

Reasoning control is best-effort: ``THINKING`` (``enabled``/``disabled``) and
``REASONING_EFFORT`` (``low``/``high``/``max``) are forwarded when set, but the
gateway is not guaranteed to honour them.  The output-token budget in
``utils/llm_budget.py`` remains the reliable lever.
"""

import os

from .credentials import api_key_env_name, base_url
from .openai_base import OpenAICompatibleProvider


class GlmProvider(OpenAICompatibleProvider):
    """OpenAI-compatible provider for GLM models on the shared gateway."""

    DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"

    def __init__(self):
        # Shared gateway when one is configured, else Zhipu's official API.
        super().__init__(
            api_key_env=api_key_env_name("glm"),
            base_url=base_url(self.DEFAULT_BASE_URL, vendor="glm"),
        )

    @property
    def name(self) -> str:
        return "glm"

    def _build_api_params(self, model_name, messages, **kwargs):
        params = super()._build_api_params(model_name, messages, **kwargs)

        # Thinking-mode switch. The gateway takes it as a top-level request
        # field, which extra_body forwards verbatim; the OpenAI SDK would
        # otherwise reject the unknown top-level kwarg.
        thinking = kwargs.get("thinking") or os.getenv("THINKING")
        if thinking:
            params["extra_body"] = {"thinking": {"type": thinking}}

        effort = kwargs.get("reasoning_effort") or os.getenv(
            "REASONING_EFFORT"
        ) or os.getenv("GLM_REASONING_EFFORT")
        if effort:
            params["reasoning_effort"] = effort
        return params

    def get_max_tokens_limit(self, model_name: str) -> int:
        # The gateway is authoritative and may accept less.
        return 65536

    def supports_multiple_completions(self) -> bool:
        # Not confirmed for this gateway; individual calls are slower but safer.
        return False
