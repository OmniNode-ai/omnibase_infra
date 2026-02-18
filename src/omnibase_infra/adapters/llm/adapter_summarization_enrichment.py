# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Summarization enrichment adapter for ProtocolContextEnrichment.

When injected context exceeds the token threshold (default 8 000 tokens),
calls Qwen-72B to produce a concise summary capped at 2 000 tokens.  A
net-token guard discards the summary when it is longer than the original,
preserving the raw context instead.

Architecture:
    - Implements ProtocolContextEnrichment from omnibase_spi
    - Triggers only when context exceeds _TOKEN_THRESHOLD
    - Delegates LLM inference to HandlerLlmOpenaiCompatible via
      TransportHolderLlmHttp pointing at the Qwen-72B endpoint (:8100)
    - Returns ContractEnrichmentResult with enrichment_type="summarization"

Token Estimation:
    Token count is estimated at 4 characters per token (rough heuristic).
    Actual counts depend on the model tokenizer but this is sufficient
    for budget accounting and threshold comparisons.

Net Token Guard:
    If the summary token count >= original context token count, the summary
    is discarded and the raw context is returned as-is.  This prevents the
    enrichment from *increasing* the token budget rather than reducing it.

Related Tickets:
    - OMN-2262: Context summarization enrichment handler
    - OMN-2252: ProtocolContextEnrichment SPI contract
    - OMN-2257: LLM endpoint configuration
    - OMN-2107: HandlerLlmOpenaiCompatible
"""

from __future__ import annotations

import logging
import os
import time

from omnibase_infra.adapters.llm.adapter_llm_provider_openai import (
    TransportHolderLlmHttp,
)
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumLlmOperationType,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_openai_compatible import (
    HandlerLlmOpenaiCompatible,
)
from omnibase_infra.nodes.node_llm_inference_effect.models.model_llm_inference_request import (
    ModelLlmInferenceRequest,
)
from omnibase_spi.contracts.enrichment.contract_enrichment_result import (
    ContractEnrichmentResult,
)

logger = logging.getLogger(__name__)

# Prompt version -- bump when system/user prompt template changes.
_PROMPT_VERSION: str = "v1.0"

# Default model identifier sent to the Qwen-72B endpoint.
_DEFAULT_MODEL: str = "qwen2.5-72b"

# Token threshold: summarize only when context exceeds this many tokens.
_TOKEN_THRESHOLD: int = 8_000

# Maximum tokens requested for the summary response (~2k tokens).
_SUMMARY_MAX_TOKENS: int = 2_048

# Default temperature for summarization (slightly higher for natural prose).
_DEFAULT_TEMPERATURE: float = 0.3

# Rough token estimation: 4 characters per token.
_CHARS_PER_TOKEN: int = 4

# Relevance score when summarization succeeds and net guard passes.
_SUMMARIZATION_RELEVANCE_SCORE: float = 0.80

# Relevance score when context is below threshold (pass-through).
_PASSTHROUGH_RELEVANCE_SCORE: float = 1.0

# Relevance score when the net guard fires (summary was inflated).
_INFLATED_SUMMARY_RELEVANCE_SCORE: float = 1.0

# model_used value for pass-through (no LLM invoked).
_PASSTHROUGH_MODEL: str = "passthrough"

# summary_markdown value when the context is empty.
_EMPTY_CONTEXT_SUMMARY: str = "(empty context)"

# System prompt sent to the model.
_SYSTEM_PROMPT: str = (
    "You are an expert technical writer. Your task is to produce a concise, "
    "information-dense summary of the provided context. Preserve all key "
    "technical details, decisions, constraints, and entity names. "
    "Eliminate repetition and verbose prose. Output valid Markdown."
)

# User prompt template.
_USER_PROMPT_TEMPLATE: str = """\
Summarize the following context concisely. Target approximately {target_tokens} tokens.
Preserve all technical details, key decisions, constraints, and named entities.
Do not introduce information that is not present in the context.

## Context

{context}

## Summary
"""


def _estimate_tokens(text: str) -> int:
    """Estimate token count as len(text) // CHARS_PER_TOKEN.

    Args:
        text: Input text to estimate.

    Returns:
        Non-negative estimated token count.
    """
    return max(0, len(text) // _CHARS_PER_TOKEN)


class AdapterSummarizationEnrichment:
    """Context enrichment adapter that summarizes long context via Qwen-72B.

    Implements ``ProtocolContextEnrichment``.  When the ``context`` parameter
    exceeds ``_TOKEN_THRESHOLD`` tokens, the adapter calls Qwen-72B to produce
    a condensed summary.  A net-token guard ensures the summary is not longer
    than the original; if it is, the raw context is returned instead.

    Contexts that are already within the threshold are returned as-is with
    no LLM call, preserving the full content at near-zero latency.

    Attributes:
        handler_type: ``INFRA_HANDLER`` -- infrastructure-level handler.
        handler_category: ``EFFECT`` -- performs external I/O (HTTP call).

    Example:
        >>> adapter = AdapterSummarizationEnrichment()
        >>> result = await adapter.enrich(
        ...     prompt="Summarize the relevant background.",
        ...     context="<large context block>",
        ... )
        >>> print(result.summary_markdown)
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str = _DEFAULT_MODEL,
        token_threshold: int = _TOKEN_THRESHOLD,
        summary_max_tokens: int = _SUMMARY_MAX_TOKENS,
        temperature: float = _DEFAULT_TEMPERATURE,
        api_key: str | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            base_url: Base URL of the Qwen-72B endpoint.  Defaults to the
                ``LLM_QWEN_72B_URL`` environment variable, falling back to
                ``http://localhost:8100``.
            model: Model identifier string sent in inference requests.
            token_threshold: Minimum token count to trigger summarization.
                Contexts below this threshold are returned as-is.
            summary_max_tokens: Maximum tokens for the summary completion.
            temperature: Sampling temperature.
            api_key: Optional Bearer token for authenticated endpoints.
        """
        self._base_url: str = base_url or os.environ.get(
            "LLM_QWEN_72B_URL", "http://localhost:8100"
        )
        self._model: str = model
        self._token_threshold: int = token_threshold
        self._summary_max_tokens: int = summary_max_tokens
        self._temperature: float = temperature
        self._api_key: str | None = api_key

        self._transport = TransportHolderLlmHttp(
            target_name="qwen-72b-summarization",
            max_timeout_seconds=180.0,
        )
        self._handler = HandlerLlmOpenaiCompatible(self._transport)

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: INFRA_HANDLER."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification: EFFECT (HTTP call to Qwen-72B)."""
        return EnumHandlerTypeCategory.EFFECT

    async def enrich(
        self,
        prompt: str,
        context: str,
    ) -> ContractEnrichmentResult:
        """Enrich a prompt by summarizing long context via Qwen-72B.

        When ``context`` token count exceeds the threshold, calls Qwen-72B
        for a condensed summary.  A net-token guard discards the summary if
        it is longer than the original.  Contexts below the threshold are
        returned as-is without an LLM call.

        Args:
            prompt: The user prompt or query.  Provided for interface
                compatibility; the summarization is context-driven and does
                not directly incorporate the prompt into the LLM request.
            context: Raw context material to potentially summarize.

        Returns:
            ``ContractEnrichmentResult`` with:

            - ``enrichment_type="summarization"``
            - ``summary_markdown``: Summarized (or pass-through) context
            - ``token_count``: Estimated token count of the output
            - ``relevance_score``: 0.80 for successful summarization,
              1.0 for pass-through or net-guard bypass
            - ``model_used``: Model identifier, or ``"passthrough"`` sentinel
              when context is below threshold and no LLM was invoked
            - ``prompt_version``: Template version (``"v1.0"``)
            - ``latency_ms``: End-to-end wall time in milliseconds

        Raises:
            RuntimeHostError: Propagated from ``HandlerLlmOpenaiCompatible``
                on connection failures, timeouts, or authentication errors.
        """
        start = time.perf_counter()
        context_stripped = context.strip()
        original_token_count = _estimate_tokens(context_stripped)

        # Pass-through: context is below threshold -- no LLM call needed.
        if original_token_count <= self._token_threshold:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.debug(
                "Context below threshold (%d <= %d tokens); pass-through. latency_ms=%.1f",
                original_token_count,
                self._token_threshold,
                latency_ms,
            )
            # ContractEnrichmentResult requires summary_markdown min_length=1
            # and model_used min_length=1; use sentinel values for pass-through.
            passthrough_summary = context_stripped or _EMPTY_CONTEXT_SUMMARY
            return ContractEnrichmentResult(
                summary_markdown=passthrough_summary,
                token_count=original_token_count,
                relevance_score=_PASSTHROUGH_RELEVANCE_SCORE,
                enrichment_type="summarization",
                latency_ms=latency_ms,
                model_used=_PASSTHROUGH_MODEL,
                prompt_version=_PROMPT_VERSION,
            )

        # Context exceeds threshold -- call Qwen-72B for summarization.
        # Use str.replace() instead of str.format() so that curly braces in
        # context_stripped (e.g. JSON objects, Python dicts, YAML, code blocks)
        # are treated as literal characters and never interpreted as format
        # placeholders, which would raise KeyError / ValueError at runtime.
        user_message = _USER_PROMPT_TEMPLATE.replace(
            "{target_tokens}", str(self._summary_max_tokens)
        ).replace("{context}", context_stripped)

        request = ModelLlmInferenceRequest(
            base_url=self._base_url,
            operation_type=EnumLlmOperationType.CHAT_COMPLETION,
            model=self._model,
            messages=({"role": "user", "content": user_message},),
            system_prompt=_SYSTEM_PROMPT,
            max_tokens=self._summary_max_tokens,
            temperature=self._temperature,
            api_key=self._api_key,
        )

        response = await self._handler.handle(request)
        latency_ms = (time.perf_counter() - start) * 1000

        summary = (response.generated_text or "").strip()
        if not summary:
            summary = context_stripped
            logger.warning(
                "Qwen-72B returned empty generated_text; using raw context. "
                "model=%s latency_ms=%.1f",
                self._model,
                latency_ms,
            )
            return ContractEnrichmentResult(
                summary_markdown=summary,
                token_count=original_token_count,
                relevance_score=_PASSTHROUGH_RELEVANCE_SCORE,
                enrichment_type="summarization",
                latency_ms=latency_ms,
                model_used=self._model,
                prompt_version=_PROMPT_VERSION,
            )

        summary_token_count = _estimate_tokens(summary)

        # Net token guard: discard inflated summaries.
        if summary_token_count >= original_token_count:
            logger.debug(
                "Net token guard fired: summary (%d tokens) >= original (%d tokens); "
                "discarding summary. model=%s latency_ms=%.1f",
                summary_token_count,
                original_token_count,
                self._model,
                latency_ms,
            )
            return ContractEnrichmentResult(
                summary_markdown=context_stripped,
                token_count=original_token_count,
                relevance_score=_INFLATED_SUMMARY_RELEVANCE_SCORE,
                enrichment_type="summarization",
                latency_ms=latency_ms,
                model_used=self._model,
                prompt_version=_PROMPT_VERSION,
            )

        logger.debug(
            "Summarization complete. original_tokens=%d summary_tokens=%d "
            "savings_pct=%.0f model=%s latency_ms=%.1f",
            original_token_count,
            summary_token_count,
            100.0 * (1.0 - summary_token_count / original_token_count),
            self._model,
            latency_ms,
        )

        return ContractEnrichmentResult(
            summary_markdown=summary,
            token_count=summary_token_count,
            relevance_score=_SUMMARIZATION_RELEVANCE_SCORE,
            enrichment_type="summarization",
            latency_ms=latency_ms,
            model_used=self._model,
            prompt_version=_PROMPT_VERSION,
        )

    async def close(self) -> None:
        """Close the HTTP transport client."""
        await self._transport.close()


__all__: list[str] = ["AdapterSummarizationEnrichment"]
