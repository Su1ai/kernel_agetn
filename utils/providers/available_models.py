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

"""External available models for KernelAgent."""

from utils.providers.model_config import ModelConfig
from utils.providers.openai_provider import OpenAIProvider
from utils.providers.anthropic_provider import AnthropicProvider
from utils.providers.relay_provider import RelayProvider
from utils.providers.deepseek_provider import DeepSeekProvider
from utils.providers.glm_provider import GlmProvider


# Registry of all available models (external/OSS version)
AVAILABLE_MODELS = [
    ModelConfig(
        name="deepseek-v4-pro",
        provider_classes=[DeepSeekProvider],
        description="DeepSeek V4 Pro via an OpenAI-compatible endpoint",
    ),
    ModelConfig(
        name="deepseek-v4-flash",
        provider_classes=[DeepSeekProvider],
        description="DeepSeek V4 Flash via an OpenAI-compatible endpoint",
    ),
    ModelConfig(
    name="deepseek-v4.1-flash",
    provider_classes=[DeepSeekProvider],
    description="DeepSeek V4.1 Flash via an OpenAI-compatible endpoint",
    ),
    # Official DeepSeek API model names (api.deepseek.com).
    ModelConfig(
        name="deepseek-flash",
        provider_classes=[DeepSeekProvider],
        description="DeepSeek Flash via the official API",
    ),
    ModelConfig(
        name="deepseek-chat",
        provider_classes=[DeepSeekProvider],
        description="DeepSeek Chat (non-reasoning) via the official API",
    ),
    ModelConfig(
        name="deepseek-reasoner",
        provider_classes=[DeepSeekProvider],
        description="DeepSeek Reasoner (reasoning) via the official API",
    ),
    ModelConfig(
        name="glm-5.3-flash",
        provider_classes=[GlmProvider],
        description="GLM 5.3 Flash on the shared OpenAI-compatible gateway",
    ),
    ModelConfig(
        name="glm-5.3",
        provider_classes=[GlmProvider],
        description="GLM 5.3 on the shared OpenAI-compatible gateway",
    ),
    ModelConfig(
        name="o4-mini",
        provider_classes=[OpenAIProvider],
        description="OpenAI o4-mini - fast reasoning model",
    ),
    # OpenAI GPT-5 Model (Only GPT-5)
    ModelConfig(
        name="gpt-5",
        provider_classes=[RelayProvider, OpenAIProvider],
        description="GPT-5 flagship model (Released Aug 2025)",
    ),
    ModelConfig(
        name="gpt-5.2",
        provider_classes=[OpenAIProvider],
        description="GPT-5.2 flagship model (Released Dec 2025)",
    ),
    # Anthropic Claude 4 Models (Latest)
    ModelConfig(
        name="claude-opus-4-6",
        provider_classes=[AnthropicProvider],
        description="Claude 4.6 Opus - most intelligent (Released Feb 2026)",
    ),
    ModelConfig(
        name="claude-sonnet-4-6",
        provider_classes=[AnthropicProvider],
        description="Claude 4.6 Sonnet - fast and powerful (Released Feb 2026)",
    ),
    ModelConfig(
        name="claude-opus-4-1-20250805",
        provider_classes=[AnthropicProvider],
        description="Claude 4.1 Opus - most capable (Released Aug 2025)",
    ),
    ModelConfig(
        name="claude-sonnet-4-20250514",
        provider_classes=[AnthropicProvider],
        description="Claude 4 Sonnet - high performance (Released May 2025)",
    ),
    ModelConfig(
        name="claude-sonnet-4-5-20250929",
        provider_classes=[AnthropicProvider],
        description="Claude 4.5 Sonnet - latest balanced model (Released Sep 2025)",
    ),
    ModelConfig(
        name="gcp-claude-4-sonnet",
        provider_classes=[RelayProvider],
        description="[Relay] Claude 4 Sonnet",
    ),
    ModelConfig(
        name="claude-opus-4.5",
        provider_classes=[RelayProvider],
        description="Claude 4.5 Opus (Released Nov 2025)",
    ),
    ModelConfig(
        name="gpt-5-2",
        provider_classes=[RelayProvider],
        description="GPT-5.2 flagship model (Dec 2025) - Note the name is different from the OpenAI model",
    ),
]
