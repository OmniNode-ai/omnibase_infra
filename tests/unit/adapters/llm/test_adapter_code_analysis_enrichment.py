# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for AdapterCodeAnalysisEnrichment.

Covers:
- handler_type / handler_category properties
- enrich() with provided diff (context non-empty)
- enrich() with no context (triggers git diff fallback)
- enrich() when git diff is empty (no changes)
- enrich() when LLM returns empty text
- diff truncation at _MAX_DIFF_CHARS
- protocol compliance with ProtocolContextEnrichment
- close() propagates to transport
- _read_git_diff helper (success, non-zero returncode, timeout, OSError)
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.adapters.llm.adapter_code_analysis_enrichment import (
    _MAX_DIFF_CHARS,
    _PROMPT_VERSION,
    AdapterCodeAnalysisEnrichment,
)
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_spi.contracts.enrichment.contract_enrichment_result import (
    ContractEnrichmentResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter(**kwargs: object) -> AdapterCodeAnalysisEnrichment:
    """Build an AdapterCodeAnalysisEnrichment with a mocked transport."""
    adapter = AdapterCodeAnalysisEnrichment(
        base_url="http://localhost:8000",
        **kwargs,  # type: ignore[arg-type]
    )
    # Replace the real transport and inner handler with mocks so tests do not
    # make actual HTTP calls.
    adapter._transport = MagicMock()
    adapter._transport.close = AsyncMock()
    adapter._handler = AsyncMock()
    return adapter


def _make_llm_response(generated_text: str) -> MagicMock:
    """Build a minimal ModelLlmInferenceResponse mock."""
    resp = MagicMock()
    resp.generated_text = generated_text
    return resp


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterCodeAnalysisEnrichmentProperties:
    """Tests for classification properties."""

    def test_handler_type(self) -> None:
        adapter = _make_adapter()
        assert adapter.handler_type is EnumHandlerType.INFRA_HANDLER

    def test_handler_category(self) -> None:
        adapter = _make_adapter()
        assert adapter.handler_category is EnumHandlerTypeCategory.EFFECT


# ---------------------------------------------------------------------------
# enrich() -- happy paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterCodeAnalysisEnrichmentEnrich:
    """Tests for the enrich() method."""

    @pytest.mark.asyncio
    async def test_enrich_with_provided_diff(self) -> None:
        """When context is non-empty, enrich() uses it as the diff."""
        adapter = _make_adapter()
        diff = "diff --git a/foo.py b/foo.py\n+def bar():\n+    pass\n"
        adapter._handler.handle = AsyncMock(
            return_value=_make_llm_response("## Analysis\n\nfunction bar added")
        )

        result = await adapter.enrich(
            prompt="What changed?",
            context=diff,
        )

        assert isinstance(result, ContractEnrichmentResult)
        assert result.enrichment_type == "code_analysis"
        assert result.summary_markdown == "## Analysis\n\nfunction bar added"
        assert result.relevance_score == pytest.approx(0.85)
        assert result.model_used == "qwen2.5-coder-14b"
        assert result.prompt_version == _PROMPT_VERSION
        assert result.token_count >= 0
        assert result.latency_ms >= 0.0

        # LLM handler must have been called once.
        adapter._handler.handle.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_enrich_context_whitespace_triggers_git_fallback(self) -> None:
        """Whitespace-only context triggers git diff fallback."""
        adapter = _make_adapter()
        git_diff_output = "diff --git a/bar.py b/bar.py\n-old\n+new\n"
        adapter._handler.handle = AsyncMock(
            return_value=_make_llm_response("## Analysis\n\nrelevant changes")
        )

        with patch(
            "omnibase_infra.adapters.llm.adapter_code_analysis_enrichment"
            ".AdapterCodeAnalysisEnrichment._read_git_diff",
            new=AsyncMock(return_value=git_diff_output),
        ):
            result = await adapter.enrich(
                prompt="What changed?",
                context="   ",
            )

        assert result.enrichment_type == "code_analysis"
        assert "relevant changes" in result.summary_markdown

    @pytest.mark.asyncio
    async def test_enrich_empty_context_uses_git_diff(self) -> None:
        """Empty context triggers git diff fallback and returns analysis."""
        adapter = _make_adapter()
        git_diff_output = "diff --git a/x.py b/x.py\n+x = 1\n"
        adapter._handler.handle = AsyncMock(
            return_value=_make_llm_response("## Analysis\n\nAdded constant x")
        )

        with patch(
            "omnibase_infra.adapters.llm.adapter_code_analysis_enrichment"
            ".AdapterCodeAnalysisEnrichment._read_git_diff",
            new=AsyncMock(return_value=git_diff_output),
        ):
            result = await adapter.enrich(prompt="Changes?", context="")

        assert result.relevance_score == pytest.approx(0.85)
        assert "Added constant x" in result.summary_markdown

    @pytest.mark.asyncio
    async def test_enrich_no_diff_returns_empty_result(self) -> None:
        """When no diff is available, returns low-relevance empty result."""
        adapter = _make_adapter()

        with patch(
            "omnibase_infra.adapters.llm.adapter_code_analysis_enrichment"
            ".AdapterCodeAnalysisEnrichment._read_git_diff",
            new=AsyncMock(return_value=""),
        ):
            result = await adapter.enrich(prompt="What changed?", context="")

        assert result.relevance_score == pytest.approx(0.0)
        assert result.token_count == 0
        assert "No Changes Detected" in result.summary_markdown
        # LLM handler must NOT have been called.
        adapter._handler.handle.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_enrich_llm_returns_empty_text(self) -> None:
        """When LLM returns None/empty text, summary is a fallback message."""
        adapter = _make_adapter()
        diff = "diff --git a/z.py b/z.py\n+z = 0\n"
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response(""))

        result = await adapter.enrich(prompt="Changes?", context=diff)

        assert "Analysis Unavailable" in result.summary_markdown

    @pytest.mark.asyncio
    async def test_enrich_llm_returns_none_text(self) -> None:
        """When LLM response generated_text is None, fallback message is used."""
        adapter = _make_adapter()
        diff = "diff --git a/z.py b/z.py\n+z = 0\n"
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response(None))  # type: ignore[arg-type]

        result = await adapter.enrich(prompt="Changes?", context=diff)

        assert "Analysis Unavailable" in result.summary_markdown

    @pytest.mark.asyncio
    async def test_enrich_diff_truncated_at_max_chars(self) -> None:
        """Diffs longer than _MAX_DIFF_CHARS are truncated."""
        adapter = _make_adapter()
        long_diff = "+" + "x" * (_MAX_DIFF_CHARS + 5000)
        adapter._handler.handle = AsyncMock(
            return_value=_make_llm_response("## Analysis\n\ntruncated diff analyzed")
        )

        await adapter.enrich(prompt="Changes?", context=long_diff)

        # Verify the handler was called and the user message content is within bounds.
        call_args = adapter._handler.handle.call_args
        request = call_args[0][0]
        user_msg_content = request.messages[0]["content"]
        # The diff portion in the message should be truncated.
        assert "... [diff truncated]" in user_msg_content

    @pytest.mark.asyncio
    async def test_enrich_token_count_is_estimated(self) -> None:
        """Token count is estimated as len(summary) // 4."""
        adapter = _make_adapter()
        summary = "A" * 400  # 400 chars => 100 tokens
        adapter._handler.handle = AsyncMock(return_value=_make_llm_response(summary))

        result = await adapter.enrich(
            prompt="Q", context="diff --git a/x.py b/x.py\n+pass\n"
        )

        assert result.token_count == 100

    @pytest.mark.asyncio
    async def test_enrich_result_schema_version_default(self) -> None:
        """schema_version defaults to '1.0'."""
        adapter = _make_adapter()
        adapter._handler.handle = AsyncMock(
            return_value=_make_llm_response("## Analysis\n\nsome analysis")
        )

        result = await adapter.enrich(
            prompt="Q", context="diff --git a/x.py b/x.py\n+pass\n"
        )

        assert result.schema_version == "1.0"


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterCodeAnalysisEnrichmentClose:
    """Tests for the close() method."""

    @pytest.mark.asyncio
    async def test_close_calls_transport_close(self) -> None:
        """close() delegates to the transport's close() method."""
        adapter = _make_adapter()
        await adapter.close()
        adapter._transport.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# _read_git_diff()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReadGitDiff:
    """Tests for AdapterCodeAnalysisEnrichment._read_git_diff()."""

    @pytest.mark.asyncio
    async def test_returns_diff_on_success(self) -> None:
        """Returns stdout when git diff exits with rc=0."""
        diff_bytes = b"diff --git a/x.py b/x.py\n+x = 1\n"
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(diff_bytes, b""))

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            result = await AdapterCodeAnalysisEnrichment._read_git_diff(".")

        assert result == diff_bytes.decode()

    @pytest.mark.asyncio
    async def test_returns_empty_on_nonzero_returncode(self) -> None:
        """Returns empty string when git exits with non-zero code."""
        mock_proc = AsyncMock()
        mock_proc.returncode = 128
        mock_proc.communicate = AsyncMock(
            return_value=(b"", b"fatal: not a git repository")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await AdapterCodeAnalysisEnrichment._read_git_diff(".")

        assert result == ""

    @pytest.mark.asyncio
    async def test_returns_empty_on_timeout(self) -> None:
        """Returns empty string when git diff times out."""
        mock_proc = AsyncMock()
        mock_proc.returncode = None
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=TimeoutError("timeout"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await AdapterCodeAnalysisEnrichment._read_git_diff(".")

        assert result == ""
        mock_proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_empty_on_oserror(self) -> None:
        """Returns empty string when git binary is not found (OSError)."""
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=OSError("git not found"),
        ):
            result = await AdapterCodeAnalysisEnrichment._read_git_diff(".")

        assert result == ""

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self) -> None:
        """asyncio.CancelledError is not swallowed."""
        mock_proc = AsyncMock()
        mock_proc.returncode = None
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=asyncio.CancelledError())

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(asyncio.CancelledError):
                await AdapterCodeAnalysisEnrichment._read_git_diff(".")

        mock_proc.kill.assert_called_once()


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProtocolContextEnrichmentCompliance:
    """Verify AdapterCodeAnalysisEnrichment satisfies ProtocolContextEnrichment."""

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
