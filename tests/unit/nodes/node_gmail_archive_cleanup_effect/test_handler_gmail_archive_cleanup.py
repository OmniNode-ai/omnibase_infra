# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for HandlerGmailArchiveCleanup.

Test scenarios:
    - 5 deletes all succeed
    - 2 of 5 deletes fail (soft failure)
    - search_messages raises (hard_failed=True, label skipped)
    - Model validation: retention_days=0 fails ge=1
    - Summary event emitted when purged_count > 0
    - No event emitted when no messages deleted and no errors

Related Tickets:
    - OMN-2731: Add node_gmail_archive_cleanup_effect
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from omnibase_infra.handlers.handler_gmail_api import HandlerGmailApi
from omnibase_infra.nodes.node_gmail_archive_cleanup_effect.handlers.handler_gmail_archive_cleanup import (
    HandlerGmailArchiveCleanup,
)
from omnibase_infra.nodes.node_gmail_archive_cleanup_effect.models.model_gmail_cleanup_config import (
    ModelGmailCleanupConfig,
)
from omnibase_infra.nodes.node_gmail_archive_cleanup_effect.models.model_gmail_cleanup_result import (
    ModelGmailCleanupResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stub(message_id: str) -> dict[str, str]:
    """Return a minimal Gmail message stub dict."""
    return {"id": message_id, "threadId": f"thread-{message_id}"}


def _make_gmail_api_mock() -> HandlerGmailApi:
    """Return a MagicMock with HandlerGmailApi spec."""
    mock: HandlerGmailApi = MagicMock(spec=HandlerGmailApi)
    return mock


# ---------------------------------------------------------------------------
# Model validation tests
# ---------------------------------------------------------------------------


class TestModelGmailCleanupConfig:
    """Validation tests for ModelGmailCleanupConfig."""

    def test_default_retention_days(self) -> None:
        """Default retention_days is 60."""
        config = ModelGmailCleanupConfig(archive_labels=["ARCHIVE"])
        assert config.retention_days == 60

    def test_retention_days_minimum(self) -> None:
        """retention_days=1 is the minimum valid value."""
        config = ModelGmailCleanupConfig(archive_labels=["ARCHIVE"], retention_days=1)
        assert config.retention_days == 1

    def test_retention_days_maximum(self) -> None:
        """retention_days=365 is the maximum valid value."""
        config = ModelGmailCleanupConfig(archive_labels=["ARCHIVE"], retention_days=365)
        assert config.retention_days == 365

    @pytest.mark.parametrize("invalid_days", [0, -1, 366, 1000])
    def test_retention_days_invalid(self, invalid_days: int) -> None:
        """retention_days outside [1, 365] raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ModelGmailCleanupConfig(
                archive_labels=["ARCHIVE"], retention_days=invalid_days
            )

    def test_archive_labels_required(self) -> None:
        """archive_labels field is required."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ModelGmailCleanupConfig()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Handler tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHandlerGmailArchiveCleanup:
    """Async tests for HandlerGmailArchiveCleanup."""

    # ------------------------------------------------------------------
    # Scenario 1: all 5 deletes succeed
    # ------------------------------------------------------------------

    async def test_all_five_deletes_succeed(self) -> None:
        """5 messages found and all deleted; purged_count=5."""
        gmail_api = _make_gmail_api_mock()
        stubs = [_make_stub(f"msg{i}") for i in range(5)]
        gmail_api.search_messages = AsyncMock(return_value=stubs)  # type: ignore[method-assign]
        gmail_api.delete_message = AsyncMock(return_value=True)  # type: ignore[method-assign]

        handler = HandlerGmailArchiveCleanup(gmail_api=gmail_api)
        config = ModelGmailCleanupConfig(
            archive_labels=["ARCHIVE_LABEL"], retention_days=30
        )
        result: ModelGmailCleanupResult = await handler.handle(config)

        assert result.purged_count == 5
        assert result.label_counts == {"ARCHIVE_LABEL": 5}
        assert not result.hard_failed
        assert result.errors == []
        # One summary event should be emitted
        assert len(result.pending_events) == 1
        event = result.pending_events[0]
        assert isinstance(event, dict)
        assert event["event_type"] == "onex.evt.omnibase-infra.gmail-archive-purged.v1"
        assert event["purged_count"] == 5
        assert event["partition_key"] == "gmail-archive-cleanup"

    # ------------------------------------------------------------------
    # Scenario 2: 2 of 5 deletes fail (soft failure)
    # ------------------------------------------------------------------

    async def test_two_of_five_deletes_fail(self) -> None:
        """5 messages found; 3 succeed, 2 fail; errors recorded."""
        gmail_api = _make_gmail_api_mock()
        stubs = [_make_stub(f"msg{i}") for i in range(5)]
        gmail_api.search_messages = AsyncMock(return_value=stubs)  # type: ignore[method-assign]
        # Fail for msg1 and msg3, succeed for rest
        fail_ids = {"msg1", "msg3"}
        gmail_api.delete_message = AsyncMock(  # type: ignore[method-assign]
            side_effect=lambda mid: mid not in fail_ids
        )

        handler = HandlerGmailArchiveCleanup(gmail_api=gmail_api)
        config = ModelGmailCleanupConfig(
            archive_labels=["ARCHIVE_LABEL"], retention_days=60
        )
        result: ModelGmailCleanupResult = await handler.handle(config)

        assert result.purged_count == 3
        assert result.label_counts == {"ARCHIVE_LABEL": 3}
        assert not result.hard_failed
        assert len(result.errors) == 2
        # Summary event should still be emitted (errors exist)
        assert len(result.pending_events) == 1
        event = result.pending_events[0]
        assert isinstance(event, dict)
        assert event["error_count"] == 2

    # ------------------------------------------------------------------
    # Scenario 3: search_messages raises (hard_failed=True)
    # ------------------------------------------------------------------

    async def test_search_messages_raises_hard_failed(self) -> None:
        """search_messages raises -> hard_failed=True, label skipped."""
        gmail_api = _make_gmail_api_mock()
        gmail_api.search_messages = AsyncMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("OAuth2 token refresh failed")
        )
        gmail_api.delete_message = AsyncMock(return_value=True)  # type: ignore[method-assign]

        handler = HandlerGmailArchiveCleanup(gmail_api=gmail_api)
        config = ModelGmailCleanupConfig(
            archive_labels=["ARCHIVE_LABEL"], retention_days=60
        )
        result: ModelGmailCleanupResult = await handler.handle(config)

        assert result.purged_count == 0
        assert result.label_counts == {}
        assert result.hard_failed is True
        assert len(result.errors) == 1
        assert "search_messages failed" in result.errors[0]
        assert "ARCHIVE_LABEL" in result.errors[0]
        # delete_message should never have been called
        gmail_api.delete_message.assert_not_called()
        # Summary event is emitted because errors exist
        assert len(result.pending_events) == 1

    # ------------------------------------------------------------------
    # Scenario 4: no messages found — no event emitted
    # ------------------------------------------------------------------

    async def test_no_messages_no_event(self) -> None:
        """No messages match query; no event emitted."""
        gmail_api = _make_gmail_api_mock()
        gmail_api.search_messages = AsyncMock(return_value=[])  # type: ignore[method-assign]
        gmail_api.delete_message = AsyncMock(return_value=True)  # type: ignore[method-assign]

        handler = HandlerGmailArchiveCleanup(gmail_api=gmail_api)
        config = ModelGmailCleanupConfig(
            archive_labels=["ARCHIVE_LABEL"], retention_days=60
        )
        result: ModelGmailCleanupResult = await handler.handle(config)

        assert result.purged_count == 0
        assert result.label_counts == {"ARCHIVE_LABEL": 0}
        assert not result.hard_failed
        assert result.errors == []
        # No event when nothing happened
        assert result.pending_events == []

    # ------------------------------------------------------------------
    # Scenario 5: multiple labels, mixed results
    # ------------------------------------------------------------------

    async def test_multiple_labels_mixed(self) -> None:
        """Two labels: first has 3 messages (all succeed), second raises."""
        gmail_api = _make_gmail_api_mock()
        stubs_label1 = [_make_stub(f"a{i}") for i in range(3)]

        async def search_side_effect(
            query: str, max_results: int = 500
        ) -> list[dict[str, str]]:
            if "label1" in query:
                return stubs_label1
            raise RuntimeError("API error for label2")

        gmail_api.search_messages = AsyncMock(side_effect=search_side_effect)  # type: ignore[method-assign]
        gmail_api.delete_message = AsyncMock(return_value=True)  # type: ignore[method-assign]

        handler = HandlerGmailArchiveCleanup(gmail_api=gmail_api)
        config = ModelGmailCleanupConfig(
            archive_labels=["label1", "label2"], retention_days=60
        )
        result: ModelGmailCleanupResult = await handler.handle(config)

        assert result.purged_count == 3
        assert result.label_counts.get("label1") == 3
        # label2 was skipped — not in label_counts
        assert "label2" not in result.label_counts
        assert result.hard_failed is True
        assert len(result.errors) == 1
        assert len(result.pending_events) == 1

    # ------------------------------------------------------------------
    # Scenario 6: query string uses correct label name and date
    # ------------------------------------------------------------------

    async def test_search_query_contains_date(self) -> None:
        """search_messages is called with a query containing the label and date."""
        gmail_api = _make_gmail_api_mock()
        gmail_api.search_messages = AsyncMock(return_value=[])  # type: ignore[method-assign]
        gmail_api.delete_message = AsyncMock(return_value=True)  # type: ignore[method-assign]

        handler = HandlerGmailArchiveCleanup(gmail_api=gmail_api)
        config = ModelGmailCleanupConfig(
            archive_labels=["MY_ARCHIVE"], retention_days=90
        )
        await handler.handle(config)

        call_args = gmail_api.search_messages.call_args
        query: str = (
            call_args.kwargs.get("query", "")
            if call_args.kwargs
            else str(call_args.args[0])
            if call_args.args
            else ""
        )
        assert "label:MY_ARCHIVE" in query
        assert "before:" in query

    # ------------------------------------------------------------------
    # Scenario 7: events_published starts at 0 (set by runtime)
    # ------------------------------------------------------------------

    async def test_events_published_starts_at_zero(self) -> None:
        """events_published is 0 — the runtime increments after publishing."""
        gmail_api = _make_gmail_api_mock()
        gmail_api.search_messages = AsyncMock(  # type: ignore[method-assign]
            return_value=[_make_stub("msg0")]
        )
        gmail_api.delete_message = AsyncMock(return_value=True)  # type: ignore[method-assign]

        handler = HandlerGmailArchiveCleanup(gmail_api=gmail_api)
        config = ModelGmailCleanupConfig(archive_labels=["ARCHIVE"], retention_days=30)
        result: ModelGmailCleanupResult = await handler.handle(config)

        assert result.events_published == 0
        assert len(result.pending_events) == 1
