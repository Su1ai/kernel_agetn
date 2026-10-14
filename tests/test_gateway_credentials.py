# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""Credential resolution: shared gateway first, official vendor API otherwise."""

from utils.providers.credentials import (
    api_key_env_name,
    base_url,
    gateway_configured,
)
from utils.providers.deepseek_provider import DeepSeekProvider
from utils.providers.glm_provider import GlmProvider

_ALL = (
    "ADVIBE_API_KEY", "ADVIBE_BASE_URL", "API_KEY", "BASE_URL",
    "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "GLM_API_KEY", "GLM_BASE_URL",
)


def _clear(monkeypatch):
    for name in _ALL:
        monkeypatch.delenv(name, raising=False)


def test_gateway_wins_for_every_vendor(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("ADVIBE_API_KEY", "gw-key")
    monkeypatch.setenv("ADVIBE_BASE_URL", "https://gateway.example/v1")
    # a vendor key must not override an explicitly configured gateway
    monkeypatch.setenv("GLM_API_KEY", "official-key")

    assert gateway_configured() is True
    for vendor in ("deepseek", "glm"):
        assert api_key_env_name(vendor) == "ADVIBE_API_KEY"
        assert base_url(None, vendor=vendor) == "https://gateway.example/v1"


def test_legacy_generic_names_act_as_the_gateway(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("API_KEY", "legacy-key")
    monkeypatch.setenv("BASE_URL", "https://legacy.example/v1")

    assert gateway_configured() is True
    assert api_key_env_name("glm") == "API_KEY"
    assert base_url(None, vendor="glm") == "https://legacy.example/v1"


def test_per_vendor_fallback_when_no_gateway(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("GLM_API_KEY", "glm-official")
    monkeypatch.setenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")

    assert gateway_configured() is False
    # deepseek has nothing configured: name points at the variable to fill in
    assert api_key_env_name("deepseek") == "DEEPSEEK_API_KEY"
    assert base_url(None, vendor="deepseek") == "https://api.deepseek.com"
    # glm uses its own official credentials
    assert api_key_env_name("glm") == "GLM_API_KEY"
    assert base_url(None, vendor="glm") == "https://open.bigmodel.cn/api/paas/v4/"


def test_official_defaults_used_when_only_keys_are_set(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds")

    assert base_url(None, vendor="deepseek") == "https://api.deepseek.com"
    assert base_url(None, vendor="glm") == "https://open.bigmodel.cn/api/paas/v4/"


def test_providers_wire_the_resolution(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("ADVIBE_API_KEY", "gw")
    monkeypatch.setenv("ADVIBE_BASE_URL", "https://gateway.example/v1")
    for provider in (DeepSeekProvider(), GlmProvider()):
        assert provider.api_key_env == "ADVIBE_API_KEY"
        assert provider.base_url == "https://gateway.example/v1"

    _clear(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds")
    monkeypatch.setenv("GLM_API_KEY", "glm")
    ds, glm = DeepSeekProvider(), GlmProvider()
    assert ds.api_key_env == "DEEPSEEK_API_KEY"
    assert ds.base_url == "https://api.deepseek.com"
    assert glm.api_key_env == "GLM_API_KEY"
    assert glm.base_url == "https://open.bigmodel.cn/api/paas/v4/"
    assert ds.name == "deepseek" and glm.name == "glm"
