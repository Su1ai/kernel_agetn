# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from utils.providers.available_models import AVAILABLE_MODELS
from utils.providers.deepseek_provider import DeepSeekProvider


def test_deepseek_provider_configuration(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://gateway.example/v1")

    provider = DeepSeekProvider()

    assert provider.name == "deepseek"
    assert provider.api_key_env == "DEEPSEEK_API_KEY"
    assert provider.base_url == "https://gateway.example/v1"
    assert provider.supports_multiple_completions() is False


def test_deepseek_models_are_registered():
    by_name = {model.name: model for model in AVAILABLE_MODELS}

    assert by_name["deepseek-v4-pro"].provider_classes == [DeepSeekProvider]
    assert by_name["deepseek-v4-flash"].provider_classes == [DeepSeekProvider]


def test_deepseek_enable_thinking_from_environment(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_ENABLE_THINKING", "false")
    provider = DeepSeekProvider()

    params = provider._build_api_params("deepseek-v4.1-flash", [])

    assert params["extra_body"]["enable_thinking"] is False


def test_deepseek_enable_thinking_explicit_value_wins(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_ENABLE_THINKING", "false")
    provider = DeepSeekProvider()

    params = provider._build_api_params(
        "deepseek-v4.1-flash", [], enable_thinking=True
    )

    assert params["extra_body"]["enable_thinking"] is True


def test_deepseek_enable_thinking_is_omitted_by_default(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.delenv("DEEPSEEK_ENABLE_THINKING", raising=False)
    provider = DeepSeekProvider()

    params = provider._build_api_params("deepseek-v4.1-flash", [])

    assert "extra_body" not in params


def test_deepseek_enable_thinking_rejects_invalid_environment(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_ENABLE_THINKING", "sometimes")
    provider = DeepSeekProvider()

    try:
        provider._build_api_params("deepseek-v4.1-flash", [])
    except ValueError as exc:
        assert "DEEPSEEK_ENABLE_THINKING" in str(exc)
    else:
        raise AssertionError("invalid boolean value should fail")
