# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from utils.providers.available_models import AVAILABLE_MODELS
from utils.providers.deepseek_provider import DeepSeekProvider


def _clear(monkeypatch):
    # Includes the gateway and official vendor names: a stray .env in the working
    # directory would otherwise make the resolution pick the gateway.
    for name in ("THINKING", "REASONING_EFFORT",
                 "ADVIBE_API_KEY", "ADVIBE_BASE_URL", "API_KEY", "BASE_URL",
                 "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL",
                 "GLM_API_KEY", "GLM_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


def test_deepseek_provider_configuration(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("BASE_URL", "https://gateway.example/v1")

    provider = DeepSeekProvider()

    assert provider.name == "deepseek"
    assert provider.api_key_env == "API_KEY"
    assert provider.base_url == "https://gateway.example/v1"
    assert provider.supports_multiple_completions() is False


def test_deepseek_models_are_registered():
    by_name = {model.name: model for model in AVAILABLE_MODELS}

    assert by_name["deepseek-v4-pro"].provider_classes == [DeepSeekProvider]
    assert by_name["deepseek-v4-flash"].provider_classes == [DeepSeekProvider]


def test_thinking_disabled_from_environment(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("THINKING", "disabled")
    provider = DeepSeekProvider()

    params = provider._build_api_params("deepseek-v4-flash", [])

    assert params["extra_body"]["thinking"] == {"type": "disabled"}


def test_thinking_explicit_kwarg_wins(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("THINKING", "disabled")
    provider = DeepSeekProvider()

    params = provider._build_api_params("deepseek-v4-flash", [], thinking="enabled")

    assert params["extra_body"]["thinking"] == {"type": "enabled"}


def test_thinking_omitted_by_default(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("API_KEY", "test-key")
    provider = DeepSeekProvider()

    params = provider._build_api_params("deepseek-v4-flash", [])

    assert "extra_body" not in params


def test_reasoning_effort_from_environment(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("REASONING_EFFORT", "low")
    provider = DeepSeekProvider()

    params = provider._build_api_params("deepseek-v4-flash", [])

    assert params["reasoning_effort"] == "low"


def test_reasoning_effort_normalization(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("API_KEY", "test-key")
    provider = DeepSeekProvider()

    assert provider._build_api_params(
        "deepseek-v4-flash", [], reasoning_effort="medium"
    )["reasoning_effort"] == "high"
    assert provider._build_api_params(
        "deepseek-v4-flash", [], reasoning_effort="minimal"
    )["reasoning_effort"] == "low"
    assert provider._build_api_params(
        "deepseek-v4-flash", [], reasoning_effort="ultra"
    )["reasoning_effort"] == "max"


def test_invalid_thinking_rejects(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("API_KEY", "test-key")
    provider = DeepSeekProvider()

    try:
        provider._build_api_params("deepseek-v4-flash", [], thinking="maybe")
    except ValueError as exc:
        assert "thinking" in str(exc)
    else:
        raise AssertionError("invalid thinking value should fail")


def test_invalid_effort_rejects(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("API_KEY", "test-key")
    provider = DeepSeekProvider()

    try:
        provider._build_api_params("deepseek-v4-flash", [], reasoning_effort="huge")
    except ValueError as exc:
        assert "reasoning_effort" in str(exc)
    else:
        raise AssertionError("invalid effort value should fail")
