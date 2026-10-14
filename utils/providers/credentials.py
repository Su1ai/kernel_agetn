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

"""Credential resolution for the OpenAI-compatible providers.

Two deployment shapes are supported, and the environment picks between them:

1. **Shared gateway (ADVibe).**  When a gateway key is configured, *every*
   provider talks to that gateway, because it serves several model families
   through one endpoint and one key.
2. **Direct vendor APIs.**  With no gateway key, each provider falls back to its
   own official endpoint.

Environment variables::

    gateway:   ADVIBE_API_KEY  / ADVIBE_BASE_URL      (legacy: API_KEY / BASE_URL)
    deepseek:  DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL   (official api.deepseek.com)
    glm:       GLM_API_KEY      / GLM_BASE_URL        (official open.bigmodel.cn)

The gateway names are checked first, so an existing deployment that only sets
``API_KEY`` / ``BASE_URL`` keeps working unchanged.
"""

import os

# Checked in order; the first name that is actually set wins.
GATEWAY_API_KEY_ENVS = ("ADVIBE_API_KEY", "API_KEY")
GATEWAY_BASE_URL_ENVS = ("ADVIBE_BASE_URL", "BASE_URL")

# vendor -> (api key env, base url env, official base url)
VENDORS: dict[str, tuple[str, str, str]] = {
    "deepseek": (
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "https://api.deepseek.com",
    ),
    "glm": (
        "GLM_API_KEY",
        "GLM_BASE_URL",
        "https://open.bigmodel.cn/api/paas/v4/",
    ),
}


def _first_set(names: tuple[str, ...]) -> str | None:
    """Return the first variable in *names* that has a non-empty value."""
    for name in names:
        if os.getenv(name):
            return name
    return None


def gateway_api_key_env() -> str | None:
    """Name of the env var holding the gateway key, or ``None`` if unset."""
    return _first_set(GATEWAY_API_KEY_ENVS)


def gateway_configured() -> bool:
    """Whether a shared gateway should be used instead of the vendor APIs."""
    return gateway_api_key_env() is not None


def api_key_env_name(vendor: str | None = None) -> str:
    """Name of the environment variable that will supply the API key.

    A configured gateway wins for every vendor.  Otherwise the vendor's own
    official key is used, and when neither is set the vendor's name is still
    returned so error messages point at the variable to fill in.
    """
    gateway = gateway_api_key_env()
    if gateway:
        return gateway
    if vendor in VENDORS:
        return VENDORS[vendor][0]
    return GATEWAY_API_KEY_ENVS[0]


def base_url(default: str | None = None, vendor: str | None = None) -> str | None:
    """Base URL paired with the credentials :func:`api_key_env_name` selects."""
    if gateway_configured():
        for name in GATEWAY_BASE_URL_ENVS:
            value = os.getenv(name)
            if value:
                return value
        # A gateway key without a URL is a misconfiguration; fall through so the
        # vendor default applies rather than returning None.
    if vendor in VENDORS:
        _, url_env, official = VENDORS[vendor]
        return os.getenv(url_env) or official
    return default
