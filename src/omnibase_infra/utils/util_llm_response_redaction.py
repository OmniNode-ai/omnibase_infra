# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""LLM API response redaction utilities for sensitive data protection.

This module provides functions to redact sensitive content from raw LLM API
responses before they are stored as ``ContractLlmUsageRaw`` blobs. The
redaction policy follows the OMN-2238 specification:

- Strip ``messages[*].content`` from system/user roles (store SHA-256 hash only)
- Strip tool call arguments (store function name + arg keys, not values)
- Cap total payload at 64KB with ``truncated: true`` + ``original_size_bytes``
  + SHA-256 hash of the original

NEVER include: passwords, API keys, PII, raw prompt content, tool call argument
values. SAFE to include: model names, token counts, finish reasons, function
names, argument key names, timing metadata.

Related:
    - OMN-2238: Extract and normalize token usage from LLM API responses
    - OMN-2235: LLM cost tracking contracts (SPI layer)
    - util_error_sanitization.py: General error message sanitization
"""

from __future__ import annotations

import hashlib
import json
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)

# Maximum size for raw response blob in bytes (64KB).
MAX_RAW_BLOB_BYTES: int = 65_536

# SHA-256 prefix for content hashes.
_HASH_PREFIX: str = "sha256:"


def _sha256_of(value: str) -> str:
    """Compute SHA-256 hex digest of a string with prefix.

    Args:
        value: The string to hash.

    Returns:
        Prefixed SHA-256 hex digest (e.g. ``sha256:a1b2c3...``).
    """
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"{_HASH_PREFIX}{digest}"


def _redact_messages(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    """Redact message content from system and user roles.

    For each message with role ``system`` or ``user``, the ``content`` field
    is replaced with a SHA-256 hash of the original. Assistant messages are
    preserved as-is (they contain model output, not user PII).

    Tool call results (role ``tool``) have their ``content`` hashed to avoid
    leaking tool execution outputs that may contain sensitive data.

    Args:
        messages: List of message dicts from the request/response payload.

    Returns:
        Copy of messages with sensitive content replaced by hashes.
    """
    redacted: list[dict[str, object]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        msg_copy = dict(msg)
        role = msg_copy.get("role", "")

        if role in ("system", "user", "tool"):
            content = msg_copy.get("content")
            if content is not None:
                content_str = str(content)
                msg_copy["content"] = _sha256_of(content_str)

        redacted.append(msg_copy)
    return redacted


def _redact_tool_calls(
    tool_calls: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Redact tool call argument values, preserving structure.

    For each tool call, the function name is preserved but argument values
    are stripped. Only argument keys are kept to show what data was passed
    without revealing the actual values.

    Args:
        tool_calls: List of tool call dicts from the response.

    Returns:
        Copy with argument values replaced by type placeholders.
    """
    redacted: list[dict[str, object]] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        tc_copy = dict(tc)
        func = tc_copy.get("function")
        if isinstance(func, dict):
            func_copy = dict(func)
            args_str = func_copy.get("arguments", "")
            if args_str:
                try:
                    args_parsed = json.loads(str(args_str))
                    if isinstance(args_parsed, dict):
                        func_copy["arguments"] = json.dumps(
                            {k: f"<{type(v).__name__}>" for k, v in args_parsed.items()}
                        )
                    else:
                        func_copy["arguments"] = "<redacted>"
                except (json.JSONDecodeError, TypeError):
                    func_copy["arguments"] = "<redacted>"
            tc_copy["function"] = func_copy
        redacted.append(tc_copy)
    return redacted


def redact_llm_response(
    raw_response: dict[str, object],
) -> dict[str, object]:
    """Redact sensitive data from a raw LLM API response.

    This function produces a sanitized copy of the response suitable for
    storage in ``ContractLlmUsageRaw.raw_data``. It applies the following
    redaction rules:

    1. Message content for system/user/tool roles is replaced with SHA-256 hashes
    2. Tool call argument values are stripped (function names + arg keys preserved)
    3. The overall blob is size-capped at 64KB

    Fields that are always safe and preserved as-is:
    - ``id``, ``object``, ``created``, ``model``
    - ``usage`` block (token counts)
    - ``finish_reason``, ``index``
    - ``system_fingerprint``

    Args:
        raw_response: The raw JSON-parsed API response dict.

    Returns:
        Redacted copy of the response, safe for storage.

    Example:
        >>> resp = {"choices": [{"message": {"role": "user", "content": "secret"}}]}
        >>> redacted = redact_llm_response(resp)
        >>> "secret" not in str(redacted)
        True
    """
    if not isinstance(raw_response, dict):
        return {}

    redacted = deepcopy(raw_response)

    # Redact top-level messages if present (request echo).
    if "messages" in redacted and isinstance(redacted["messages"], list):
        redacted["messages"] = _redact_messages(redacted["messages"])

    # Redact choices -> message content and tool calls.
    choices = redacted.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict):
                # Redact content from system/user/tool in the response choices
                role = message.get("role", "")
                if role in ("system", "user", "tool"):
                    content = message.get("content")
                    if content is not None:
                        message["content"] = _sha256_of(str(content))

                # Redact tool call arguments
                tool_calls = message.get("tool_calls")
                if isinstance(tool_calls, list) and tool_calls:
                    message["tool_calls"] = _redact_tool_calls(tool_calls)

    # Size cap enforcement
    return _enforce_size_cap(redacted)


def _enforce_size_cap(
    data: dict[str, object],
    max_bytes: int = MAX_RAW_BLOB_BYTES,
) -> dict[str, object]:
    """Enforce the 64KB size cap on the redacted response.

    If the JSON-serialized form exceeds ``max_bytes``, the data is replaced
    with a truncation marker containing the original size and a SHA-256 hash
    of the full content. The ``usage`` block is preserved even in the
    truncated form since it contains the token counts we need.

    Args:
        data: The redacted response dict.
        max_bytes: Maximum allowed size in bytes.

    Returns:
        Either the original data (if within limit) or a truncation marker.
    """
    try:
        serialized = json.dumps(data, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return {"truncated": True, "error": "serialization_failed"}

    size = len(serialized.encode("utf-8"))
    if size <= max_bytes:
        return data

    # Build truncation marker preserving the usage block.
    content_hash = _sha256_of(serialized)
    truncated: dict[str, object] = {
        "truncated": True,
        "original_size_bytes": size,
        "content_hash": content_hash,
    }

    # Preserve usage block if present (critical for token tracking).
    usage = data.get("usage")
    if isinstance(usage, dict):
        truncated["usage"] = usage

    # Preserve model and id for identification.
    for key in ("model", "id"):
        if key in data:
            truncated[key] = data[key]

    return truncated


__all__: list[str] = [
    "MAX_RAW_BLOB_BYTES",
    "redact_llm_response",
]
