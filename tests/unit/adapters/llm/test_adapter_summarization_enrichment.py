# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for AdapterSummarizationEnrichment.

Covers:
- handler_type / handler_category properties
- enrich() when context is below token threshold (pass-through, no LLM call)
- enrich() when context exceeds threshold (LLM called, summary returned)
- enrich() net token guard (summary >= original tokens -> raw context returned)
- enrich() when LLM returns empty/None text (pass-through with raw context)
- token estimation helper (_estimate_tokens)
- protocol compliance with ProtocolContextEnrichment
- close() propagates to transport
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from omnibase_infra.adapters.llm.adapter_summarization_enrichment import (
    _CHARS_PER_TOKEN,
    _EMPTY_CONTEXT_SUMMARY,
    _PASSTHROUGH_MODEL,
    _PROMPT_VERSION,
    _TOKEN_THRESHOLD,
    AdapterSummarizationEnrichment,
    _estimate_tokens,
)
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_spi.contracts.enrichment.contract_enrichment_result import (
    ContractEnrichmentResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter(**kwargs: object) -> AdapterSummarizationEnrichment:
    """Build an AdapterSummarizationEnrichment with a mocked transport."""
    adapter = AdapterSummarizationEnrichment(
        base_url="http://localhost:8100",
        **kwargs,  # type: ignore[arg-type]
    )
    # Replace transport and handler with mocks to avoid HTTP calls.
    adapter._transport = MagicMock()
    adapter._transport.close = AsyncMock()
    adapter._handler = AsyncMock()
    return adapter


def _make_llm_response(generated_text: str | None) -> MagicMock:
    """Build a minimal ModelLlmInferenceResponse mock."""
    resp = MagicMock()
    resp.generated_text = generated_text
    return resp


def _context_above_threshold(extra_chars: int = 1000) -> str:
    """Return a context string that exceeds _TOKEN_THRESHOLD tokens."""
    # _TOKEN_THRESHOLD tokens * _CHARS_PER_TOKEN chars each = minimum chars needed.
    min_chars = (_TOKEN_THRESHOLD + 1) * _CHARS_PER_TOKEN + extra_chars
    return "x" * min_chars


def _context_below_threshold() -> str:
    """Return a context string well below _TOKEN_THRESHOLD tokens."""
    # Half the threshold in tokens -> safe margin.
    target_chars = (_TOKEN_THRESHOLD // 2) * _CHARS_PER_TOKEN
    return "x" * target_chars


# ---------------------------------------------------------------------------
# _estimate_tokens
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEstimateTokens:
    """Tests for the _estimate_tokens helper."""

    def test_empty_string_returns_zero(self) -> None:
        assert _estimate_tokens("") == 0

    def test_chars_per_token_boundary(self) -> None:
        # Exactly _CHARS_PER_TOKEN chars => 1 token.
        text = "a" * _CHARS_PER_TOKEN
        assert _estimate_tokens(text) == 1

    def test_double_chars_per_token(self) -> None:
        text = "a" * (2 * _CHARS_PER_TOKEN)
        assert _estimate_tokens(text) == 2

    def test_non_multiple(self) -> None:
        # 7 chars with _CHARS_PER_TOKEN=4 => 7 // 4 = 1
        assert _estimate_tokens("a" * 7) == 7 // _CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterSummarizationEnrichmentProperties:
    """Tests for classification properties."""

    def test_handler_type(self) -> None:
        adapter = _make_adapter()
        assert adapter.handler_type is EnumHandlerType.INFRA_HANDLER

    def test_handler_category(self) -> None:
        adapter = _make_adapter()
        assert adapter.handler_category is EnumHandlerTypeCategory.EFFECT


# ---------------------------------------------------------------------------
# enrich() -- below threshold (pass-through)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterSummarizationEnrichmentPassThrough:
    """Tests for pass-through behavior when context is below threshold."""

    @pytest.mark.asyncio
    async def test_below_threshold_no_llm_call(self) -> None:
        """Context below threshold: LLM handler is NOT called."""
        adapter = _make_adapter()
        context = _context_below_threshold()

        result = await adapter.enrich(prompt="Summarize.", context=context)

        adapter._handler.handle.assert_not_awaited()  # type: ignore[attr-defined]
        assert isinstance(result, ContractEnrichmentResult)
        assert result.enrichment_type == "summarization"

    @pytest.mark.asyncio
    async def test_below_threshold_returns_raw_context(self) -> None:
        """Context below threshold: summary_markdown equals stripped context."""
        adapter = _make_adapter()
        context = "Short context."

        result = await adapter.enrich(prompt="Summarize.", context=context)

        assert result.summary_markdown == "Short context."

    @pytest.mark.asyncio
    async def test_below_threshold_relevance_score_is_one(self) -> None:
        """Pass-through result has relevance_score == 1.0."""
        adapter = _make_adapter()
        context = _context_below_threshold()

        result = await adapter.enrich(prompt="Q", context=context)

        assert result.relevance_score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_below_threshold_model_used_is_passthrough(self) -> None:
        """Pass-through result has model_used == 'passthrough' (no LLM involved)."""
        adapter = _make_adapter()
        context = _context_below_threshold()

        result = await adapter.enrich(prompt="Q", context=context)

        assert result.model_used == _PASSTHROUGH_MODEL

    @pytest.mark.asyncio
    async def test_empty_context_is_pass_through(self) -> None:
        """Empty context (0 tokens) is well below threshold -- pass-through."""
        adapter = _make_adapter()

        result = await adapter.enrich(prompt="Q", context="")

        adapter._handler.handle.assert_not_awaited()  # type: ignore[attr-defined]
        assert result.summary_markdown == _EMPTY_CONTEXT_SUMMARY
        assert result.token_count == 0

    @pytest.mark.asyncio
    async def test_whitespace_only_context_is_pass_through(self) -> None:
        """Whitespace-only context strips to empty -- pass-through."""
        adapter = _make_adapter()

        result = await adapter.enrich(prompt="Q", context="   \n\t  ")

        adapter._handler.handle.assert_not_awaited()  # type: ignore[attr-defined]
        assert result.summary_markdown == _EMPTY_CONTEXT_SUMMARY

    @pytest.mark.asyncio
    async def test_prompt_version_is_set(self) -> None:
        """prompt_version matches the module constant."""
        adapter = _make_adapter()

        result = await adapter.enrich(prompt="Q", context="short")

        assert result.prompt_version == _PROMPT_VERSION

    @pytest.mark.asyncio
    async def test_schema_version_default(self) -> None:
        """schema_version defaults to '1.0'."""
        adapter = _make_adapter()

        result = await adapter.enrich(prompt="Q", context="short")

        assert result.schema_version == "1.0"

    @pytest.mark.asyncio
    async def test_latency_ms_is_nonnegative(self) -> None:
        """latency_ms is always >= 0."""
        adapter = _make_adapter()

        result = await adapter.enrich(prompt="Q", context="short")

        assert result.latency_ms >= 0.0


# ---------------------------------------------------------------------------
# enrich() -- above threshold (LLM call, successful summarization)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterSummarizationEnrichmentSummarize:
    """Tests for summarization path (context exceeds threshold)."""

    @pytest.mark.asyncio
    async def test_above_threshold_calls_llm(self) -> None:
        """When context exceeds threshold, the LLM handler IS called."""
        adapter = _make_adapter()
        context = _context_above_threshold()
        # Summary is short -- net guard does NOT fire.
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response("## Summary\n\nBrief summary.")
        )

        result = await adapter.enrich(prompt="Summarize.", context=context)

        adapter._handler.handle.assert_awaited_once()
        assert result.summary_markdown == "## Summary\n\nBrief summary."

    @pytest.mark.asyncio
    async def test_above_threshold_enrichment_type_is_summarization(self) -> None:
        """Successful summarization has enrichment_type='summarization'."""
        adapter = _make_adapter()
        context = _context_above_threshold()
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response("Short summary.")
        )

        result = await adapter.enrich(prompt="Q", context=context)

        assert result.enrichment_type == "summarization"

    @pytest.mark.asyncio
    async def test_above_threshold_relevance_score_is_0_80(self) -> None:
        """Successful summarization has relevance_score == 0.80."""
        adapter = _make_adapter()
        context = _context_above_threshold()
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response("Short summary.")
        )

        result = await adapter.enrich(prompt="Q", context=context)

        assert result.relevance_score == pytest.approx(0.80)

    @pytest.mark.asyncio
    async def test_above_threshold_model_used_is_set(self) -> None:
        """Successful summarization has model_used set to the configured model."""
        adapter = _make_adapter(model="qwen2.5-72b")
        context = _context_above_threshold()
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response("Short summary.")
        )

        result = await adapter.enrich(prompt="Q", context=context)

        assert result.model_used == "qwen2.5-72b"

    @pytest.mark.asyncio
    async def test_above_threshold_token_count_is_summary_tokens(self) -> None:
        """token_count reflects the summary length, not the original."""
        adapter = _make_adapter()
        context = _context_above_threshold()
        summary = "A" * (8 * _CHARS_PER_TOKEN)  # 8 tokens
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response(summary)
        )

        result = await adapter.enrich(prompt="Q", context=context)

        assert result.token_count == 8

    @pytest.mark.asyncio
    async def test_above_threshold_latency_ms_is_nonnegative(self) -> None:
        """latency_ms is >= 0 on the summarization path."""
        adapter = _make_adapter()
        context = _context_above_threshold()
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response("Short.")
        )

        result = await adapter.enrich(prompt="Q", context=context)

        assert result.latency_ms >= 0.0


# ---------------------------------------------------------------------------
# enrich() -- net token guard
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterSummarizationEnrichmentNetTokenGuard:
    """Tests for the net token guard (summary >= original -> discard)."""

    @pytest.mark.asyncio
    async def test_net_guard_fires_when_summary_equals_original(self) -> None:
        """Guard fires when summary token count equals original token count."""
        adapter = _make_adapter()
        context = _context_above_threshold()
        original_tokens = _estimate_tokens(context.strip())
        # Make summary exactly as long as original.
        inflated_summary = "y" * (original_tokens * _CHARS_PER_TOKEN)
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response(inflated_summary)
        )

        result = await adapter.enrich(prompt="Q", context=context)

        # Should return raw context, not the summary.
        assert result.summary_markdown == context.strip()
        assert result.token_count == original_tokens

    @pytest.mark.asyncio
    async def test_net_guard_fires_when_summary_longer_than_original(self) -> None:
        """Guard fires when summary token count > original token count."""
        adapter = _make_adapter()
        context = _context_above_threshold()
        original_tokens = _estimate_tokens(context.strip())
        # Make summary 20% longer.
        inflated_summary = "y" * (int(original_tokens * 1.2) * _CHARS_PER_TOKEN)
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response(inflated_summary)
        )

        result = await adapter.enrich(prompt="Q", context=context)

        assert result.summary_markdown == context.strip()

    @pytest.mark.asyncio
    async def test_net_guard_relevance_score_is_one(self) -> None:
        """Net guard bypass yields relevance_score == 1.0."""
        adapter = _make_adapter()
        context = _context_above_threshold()
        original_tokens = _estimate_tokens(context.strip())
        inflated_summary = "y" * (original_tokens * _CHARS_PER_TOKEN)
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response(inflated_summary)
        )

        result = await adapter.enrich(prompt="Q", context=context)

        assert result.relevance_score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_net_guard_model_used_is_set(self) -> None:
        """model_used is set even when net guard fires (LLM was called)."""
        adapter = _make_adapter(model="qwen2.5-72b")
        context = _context_above_threshold()
        original_tokens = _estimate_tokens(context.strip())
        inflated_summary = "y" * (original_tokens * _CHARS_PER_TOKEN)
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response(inflated_summary)
        )

        result = await adapter.enrich(prompt="Q", context=context)

        assert result.model_used == "qwen2.5-72b"

    @pytest.mark.asyncio
    async def test_net_guard_does_not_fire_for_shorter_summary(self) -> None:
        """Guard does NOT fire when summary is shorter than original."""
        adapter = _make_adapter()
        context = _context_above_threshold()
        # Summary is much shorter -- guard should not fire.
        short_summary = "Brief."
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response(short_summary)
        )

        result = await adapter.enrich(prompt="Q", context=context)

        assert result.summary_markdown == short_summary
        assert result.relevance_score == pytest.approx(0.80)


# ---------------------------------------------------------------------------
# enrich() -- empty / None LLM response
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterSummarizationEnrichmentEmptyLlmResponse:
    """Tests for behavior when LLM returns empty or None text."""

    @pytest.mark.asyncio
    async def test_empty_llm_response_returns_raw_context(self) -> None:
        """When LLM returns empty string, raw context is returned."""
        adapter = _make_adapter()
        context = _context_above_threshold()
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response("")
        )

        result = await adapter.enrich(prompt="Q", context=context)

        assert result.summary_markdown == context.strip()

    @pytest.mark.asyncio
    async def test_none_llm_response_returns_raw_context(self) -> None:
        """When LLM returns None, raw context is returned."""
        adapter = _make_adapter()
        context = _context_above_threshold()
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response(None)
        )

        result = await adapter.enrich(prompt="Q", context=context)

        assert result.summary_markdown == context.strip()

    @pytest.mark.asyncio
    async def test_empty_llm_response_relevance_score_is_one(self) -> None:
        """Empty LLM response fall-back has relevance_score == 1.0."""
        adapter = _make_adapter()
        context = _context_above_threshold()
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response("")
        )

        result = await adapter.enrich(prompt="Q", context=context)

        assert result.relevance_score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_empty_llm_response_model_used_is_set(self) -> None:
        """model_used is set even on empty LLM response."""
        adapter = _make_adapter(model="qwen2.5-72b")
        context = _context_above_threshold()
        adapter._handler.handle = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_llm_response("")
        )

        result = await adapter.enrich(prompt="Q", context=context)

        assert result.model_used == "qwen2.5-72b"


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterSummarizationEnrichmentClose:
    """Tests for the close() method."""

    @pytest.mark.asyncio
    async def test_close_calls_transport_close(self) -> None:
        """close() delegates to the transport's close() method."""
        adapter = _make_adapter()
        await adapter.close()
        adapter._transport.close.assert_awaited_once()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProtocolContextEnrichmentCompliance:
    """Verify AdapterSummarizationEnrichment satisfies ProtocolContextEnrichment."""

    def test_isinstance_check(self) -> None:
        """isinstance() against ProtocolContextEnrichment is True."""
        from omnibase_spi.protocols.intelligence.protocol_context_enrichment import (
            ProtocolContextEnrichment,
        )

        adapter = _make_adapter()
        assert isinstance(adapter, ProtocolContextEnrichment)

    def test_has_enrich_method(self) -> None:
        """enrich() is callable on the adapter."""
        adapter = _make_adapter()
        assert callable(getattr(adapter, "enrich", None))
