# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""Shared LLM call helper with response diagnostics.

Every provider call should go through :func:`call_llm`.  An empty ``content``
returned by a reasoning model is otherwise indistinguishable from a downstream
parse failure: the finish reason and the reasoning-token count are what tell the
two apart.  The helper also centralises retry-on-empty so callers do not each
invent their own handling.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseProvider, LLMResponse

logger = logging.getLogger(__name__)

# Provider-specific request keys. Callers may always pass these; the helper drops
# any the target provider does not declare support for, because forwarding an
# unknown field to a gateway that validates its payload fails the whole request.
_OPTIONAL_REQUEST_KEYS = ("enable_thinking",)


def _filter_request_kwargs(
    provider: BaseProvider, kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Return a copy of ``kwargs`` with unsupported provider keys removed."""
    filtered = dict(kwargs)
    for key in _OPTIONAL_REQUEST_KEYS:
        if key in filtered and not getattr(provider, f"supports_{key}", False):
            filtered.pop(key)
    return filtered


def _describe(response: LLMResponse) -> tuple[str, int, int, Any, Any]:
    """Return (content, content_len, reasoning_len, completion, reasoning)."""
    content = response.content or ""
    reasoning_text = getattr(response, "reasoning_content", None) or ""
    usage = response.usage or {}
    details = usage.get("completion_tokens_details") or {}
    return (
        content,
        len(content),
        len(reasoning_text),
        usage.get("completion_tokens"),
        details.get("reasoning_tokens"),
    )


def call_llm(
    provider: BaseProvider,
    model: str,
    messages: list[dict[str, str]],
    *,
    log: logging.Logger | None = None,
    label: str = "",
    empty_retries: int = 0,
    retry_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> LLMResponse:
    """Call ``provider`` and log response diagnostics.

    Args:
        provider: Provider instance to call.
        model: Model name.
        messages: Chat messages.
        log: Logger for diagnostics; defaults to this module's logger.
        label: Short prefix identifying the call site in log lines.
        empty_retries: How many extra attempts to make when the response carries
            no usable text. Zero means "return the empty response as-is".
        retry_kwargs: Request keys merged into the request on retry, e.g.
            ``{"enable_thinking": False}`` to free the whole budget for the
            answer. Unsupported keys are dropped.
        **kwargs: Forwarded to ``provider.get_response``.

    Returns:
        The (last) :class:`LLMResponse`, empty content included.
    """
    log = log or logger
    prefix = f"{label}: " if label else ""
    request_kwargs = _filter_request_kwargs(provider, kwargs)

    attempt = 0
    while True:
        response = provider.get_response(model, messages, **request_kwargs)
        content, content_len, reasoning_len, completion, reasoning = _describe(response)
        finish_reason = getattr(response, "finish_reason", None)

        log.info(
            "%sLLM response: model=%s finish_reason=%s content_chars=%d "
            "reasoning_chars=%d completion_tokens=%s reasoning_tokens=%s",
            prefix,
            response.model,
            finish_reason,
            content_len,
            reasoning_len,
            completion,
            reasoning,
        )

        if content.strip():
            return response

        log.warning(
            "%sEmpty LLM content: finish_reason=%s reasoning_chars=%d "
            "completion_tokens=%s reasoning_tokens=%s",
            prefix,
            finish_reason,
            reasoning_len,
            completion,
            reasoning,
        )
        if attempt >= empty_retries:
            return response

        attempt += 1
        if retry_kwargs:
            request_kwargs.update(_filter_request_kwargs(provider, retry_kwargs))
        log.info(
            "%sRetrying LLM call after empty response (attempt %d/%d)",
            prefix,
            attempt,
            empty_retries,
        )
