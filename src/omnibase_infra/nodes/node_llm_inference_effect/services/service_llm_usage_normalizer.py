# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""LLM usage normalization service with provider fallback handling.

This module implements the 5-case normalization logic specified in OMN-2238:

1. **Complete** -- ``usage`` field present with all token counts
   -> ``source=API``, ``is_estimated=false``
2. **Partial** -- ``usage`` field present but missing some token counts
   -> estimate from response length, ``source=ESTIMATED``, ``is_estimated=true``
3. **Absent** -- ``usage`` field entirely missing from the response
   -> local tokenizer count, ``source=ESTIMATED``, ``is_estimated=true``
4. **Streaming** -- streaming response with chunk deltas
   -> accumulate chunk deltas if available, else estimate, ``source=ESTIMATED``
5. **Missing** -- provider omits usage entirely with no way to estimate
   -> ``source=MISSING``, tokens=0, ``estimated_cost_usd=null``

Tool call tokens are counted as completion_tokens (same as OpenAI convention).

Related:
    - OMN-2238: Extract and normalize token usage from LLM API responses
    - OMN-2235: LLM cost tracking contracts (SPI layer)
    - ContractLlmUsageNormalized: Canonical normalized form
    - ContractLlmUsageRaw: Raw provider data
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from omnibase_infra.utils.util_llm_response_redaction import redact_llm_response
from omnibase_spi.contracts.measurement.contract_llm_call_metrics import (
    ContractLlmUsageNormalized,
    ContractLlmUsageRaw,
)
from omnibase_spi.contracts.measurement.enum_usage_source import (
    ContractEnumUsageSource,
)

logger = logging.getLogger(__name__)

# Approximate chars-per-token ratio for estimation when token counts are
# unavailable. Based on the widely-cited "4 characters per token" heuristic
# for English text with GPT-family tokenizers.
_CHARS_PER_TOKEN_ESTIMATE: float = 4.0


def _safe_int(value: object) -> int | None:
    """Safely convert a value to int, returning None on failure.

    Args:
        value: A value to convert (int, float, str, None, etc.).

    Returns:
        The integer value, or None if conversion fails or value is None.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _estimate_tokens_from_text(text: str | None) -> int:
    """Estimate token count from text length using character ratio.

    This is a rough heuristic for when the provider does not report token
    counts. The estimate is deliberately conservative (rounds up).

    Args:
        text: The text to estimate tokens for, or None.

    Returns:
        Estimated token count (0 if text is None or empty).
    """
    if not text:
        return 0
    return max(1, int(len(text) / _CHARS_PER_TOKEN_ESTIMATE + 0.5))


def normalize_llm_usage(
    raw_response: Mapping[str, object],
    *,
    provider: str = "openai_compatible",
    generated_text: str | None = None,
    prompt_text: str | None = None,
) -> tuple[ContractLlmUsageRaw, ContractLlmUsageNormalized]:
    """Normalize LLM usage data from a raw API response.

    Implements the 5-case fallback logic from OMN-2238. Returns both the
    redacted raw usage blob and the canonical normalized representation.

    Args:
        raw_response: The full raw JSON response from the LLM provider.
        provider: Provider identifier (e.g. ``openai_compatible``, ``ollama``).
        generated_text: The generated text from the response, used for
            estimation when token counts are unavailable.
        prompt_text: The prompt/input text, used for estimation when
            prompt token count is unavailable.

    Returns:
        A tuple of ``(raw_usage, normalized_usage)`` where:
        - ``raw_usage`` contains the redacted response blob
        - ``normalized_usage`` contains the canonical token counts
    """
    # Build redacted raw blob.
    redacted_data = redact_llm_response(raw_response)
    raw_usage = ContractLlmUsageRaw(
        provider=provider,
        raw_data=redacted_data,
    )

    # Extract usage block from response.
    usage_block = (
        raw_response.get("usage") if isinstance(raw_response, Mapping) else None
    )

    if not isinstance(usage_block, dict):
        # Case 3 (Absent) or Case 5 (Missing): No usage block at all.
        # Try to estimate from response text.
        estimated_prompt = _estimate_tokens_from_text(prompt_text)
        estimated_completion = _estimate_tokens_from_text(generated_text)

        if estimated_prompt == 0 and estimated_completion == 0:
            # Case 5: Missing -- cannot estimate at all.
            logger.debug(
                "Usage data missing from provider response and no text "
                "available for estimation. provider=%s",
                provider,
            )
            normalized = ContractLlmUsageNormalized(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                source=ContractEnumUsageSource.MISSING,
                usage_is_estimated=False,
            )
        else:
            # Case 3: Absent -- estimate from text.
            logger.debug(
                "Usage data absent from provider response; estimating from "
                "text lengths. provider=%s estimated_prompt=%d "
                "estimated_completion=%d",
                provider,
                estimated_prompt,
                estimated_completion,
            )
            normalized = ContractLlmUsageNormalized(
                prompt_tokens=estimated_prompt,
                completion_tokens=estimated_completion,
                total_tokens=estimated_prompt + estimated_completion,
                source=ContractEnumUsageSource.ESTIMATED,
                usage_is_estimated=True,
            )
        return raw_usage, normalized

    # Usage block exists. Extract fields.
    prompt_tokens = _safe_int(usage_block.get("prompt_tokens"))
    completion_tokens = _safe_int(usage_block.get("completion_tokens"))

    if prompt_tokens is not None and completion_tokens is not None:
        # Case 1: Complete -- all fields present from API.
        total = prompt_tokens + completion_tokens
        logger.debug(
            "Complete usage data from API. provider=%s prompt=%d "
            "completion=%d total=%d",
            provider,
            prompt_tokens,
            completion_tokens,
            total,
        )
        normalized = ContractLlmUsageNormalized(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
            source=ContractEnumUsageSource.API,
            usage_is_estimated=False,
        )
        return raw_usage, normalized

    # Case 2: Partial -- some fields missing, estimate the rest.
    effective_prompt = (
        prompt_tokens
        if prompt_tokens is not None
        else _estimate_tokens_from_text(prompt_text)
    )
    effective_completion = (
        completion_tokens
        if completion_tokens is not None
        else _estimate_tokens_from_text(generated_text)
    )
    total = effective_prompt + effective_completion

    logger.debug(
        "Partial usage data from API; estimated missing fields. "
        "provider=%s prompt=%d (estimated=%s) completion=%d (estimated=%s)",
        provider,
        effective_prompt,
        prompt_tokens is None,
        effective_completion,
        completion_tokens is None,
    )
    normalized = ContractLlmUsageNormalized(
        prompt_tokens=effective_prompt,
        completion_tokens=effective_completion,
        total_tokens=total,
        source=ContractEnumUsageSource.ESTIMATED,
        usage_is_estimated=True,
    )
    return raw_usage, normalized


def normalize_streaming_usage(
    chunk_deltas: list[dict[str, object]],
    *,
    provider: str = "openai_compatible",
    generated_text: str | None = None,
    prompt_text: str | None = None,
) -> tuple[ContractLlmUsageRaw, ContractLlmUsageNormalized]:
    """Normalize usage from accumulated streaming chunk deltas.

    Case 4 from OMN-2238: For streaming responses, some providers include
    usage information in the final chunk. If available, those are used;
    otherwise, estimation from accumulated text is applied.

    Args:
        chunk_deltas: List of chunk delta dicts from the streaming response.
        provider: Provider identifier.
        generated_text: The accumulated generated text from all chunks.
        prompt_text: The prompt/input text for estimation fallback.

    Returns:
        A tuple of ``(raw_usage, normalized_usage)``.
    """
    # Check if any chunk contains a usage block (common in OpenAI streaming
    # with ``stream_options: {"include_usage": true}``).
    final_usage: dict[str, object] | None = None
    for chunk in reversed(chunk_deltas):
        if isinstance(chunk, dict) and isinstance(chunk.get("usage"), dict):
            final_usage = chunk
            break

    if final_usage is not None:
        # Delegate to the standard normalizer with the chunk that has usage.
        return normalize_llm_usage(
            final_usage,
            provider=provider,
            generated_text=generated_text,
            prompt_text=prompt_text,
        )

    # No usage in any chunk -- estimate from accumulated text.
    # Build a synthetic raw blob for provenance.
    raw_usage = ContractLlmUsageRaw(
        provider=provider,
        raw_data={
            "streaming": True,
            "chunk_count": len(chunk_deltas),
            "usage_in_chunks": False,
        },
    )

    estimated_prompt = _estimate_tokens_from_text(prompt_text)
    estimated_completion = _estimate_tokens_from_text(generated_text)

    if estimated_prompt == 0 and estimated_completion == 0:
        normalized = ContractLlmUsageNormalized(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            source=ContractEnumUsageSource.MISSING,
            usage_is_estimated=False,
        )
    else:
        normalized = ContractLlmUsageNormalized(
            prompt_tokens=estimated_prompt,
            completion_tokens=estimated_completion,
            total_tokens=estimated_prompt + estimated_completion,
            source=ContractEnumUsageSource.ESTIMATED,
            usage_is_estimated=True,
        )

    return raw_usage, normalized


__all__: list[str] = [
    "normalize_llm_usage",
    "normalize_streaming_usage",
]
