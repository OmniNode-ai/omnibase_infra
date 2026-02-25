# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for the Gmail Intent Poller Effect node.

Covers:
- extract_urls: empty string, mixed content, duplicates, no URLs
- Poller handler:
  - skip-and-continue (get_message fails on 1 of 3 messages)
  - hard-fail label (list_messages returns [] due to failure, hard_failed=True)
  - idempotency (processed_label prevents re-emit in recovery pass)
  - mixed label states (some processed, some new)
- Gmail body parsing (delegated to ModelGmailMessage — tested via integration):
  - multipart/alternative (plain wins)
  - no text/plain (empty string)
  - deeply nested parts

Run with:
    uv run pytest tests/unit/nodes/node_gmail_intent_poller_effect/ -m unit -v

Related Tickets:
    - OMN-2730: feat(omnibase_infra): add node_gmail_intent_poller_effect
    - OMN-2728: Gmail Integration epic (omnibase_infra)
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from omnibase_infra.nodes.node_gmail_intent_poller_effect.handlers.handler_gmail_intent_poll import (
    HandlerGmailIntentPoll,
    extract_urls,
)
from omnibase_infra.nodes.node_gmail_intent_poller_effect.models.model_gmail_intent_poller_config import (
    ModelGmailIntentPollerConfig,
)
from omnibase_infra.nodes.node_gmail_intent_poller_effect.models.model_gmail_intent_poller_result import (
    ModelGmailIntentPollerResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _b64url(text: str) -> str:
    """Encode text as URL-safe base64 (no padding)."""
    return base64.urlsafe_b64encode(text.encode()).rstrip(b"=").decode()


def _make_raw_message(
    message_id: str = "msg001",
    subject: str = "Test Subject",
    sender: str = "test@example.com",
    body_text: str = "Hello world",
    label_ids: list[str] | None = None,
    received_ms: int = 1_700_000_000_000,
) -> dict[str, Any]:
    """Build a minimal Gmail API messages.get response dict."""
    return {
        "id": message_id,
        "threadId": f"thread-{message_id}",
        "labelIds": label_ids or [],
        "internalDate": str(received_ms),
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "Subject", "value": subject},
                {"name": "From", "value": sender},
            ],
            "body": {
                "data": _b64url(body_text),
            },
        },
    }


def _make_multipart_raw_message(
    message_id: str = "msg-mp",
    subject: str = "Multipart Subject",
    plain_text: str = "Plain body",
    html_text: str = "<b>HTML body</b>",
) -> dict[str, Any]:
    """Build a multipart/alternative Gmail API response."""
    return {
        "id": message_id,
        "threadId": f"thread-{message_id}",
        "labelIds": [],
        "internalDate": "1700000000000",
        "payload": {
            "mimeType": "multipart/alternative",
            "headers": [
                {"name": "Subject", "value": subject},
                {"name": "From", "value": "mp@example.com"},
            ],
            "parts": [
                {
                    "mimeType": "text/plain",
                    "body": {"data": _b64url(plain_text)},
                },
                {
                    "mimeType": "text/html",
                    "body": {"data": _b64url(html_text)},
                },
            ],
        },
    }


def _make_handler(gmail: Any) -> HandlerGmailIntentPoll:
    return HandlerGmailIntentPoll(gmail=gmail)


def _make_config(
    source_labels: list[str] | None = None,
    archive_label: str = "archived",
    processed_label: str = "processed",
    max_per_label: int = 50,
) -> ModelGmailIntentPollerConfig:
    return ModelGmailIntentPollerConfig(
        source_labels=source_labels or ["to-read"],
        archive_label=archive_label,
        processed_label=processed_label,
        max_per_label=max_per_label,
    )


# ---------------------------------------------------------------------------
# extract_urls tests
# ---------------------------------------------------------------------------


class TestExtractUrls:
    @pytest.mark.unit
    def test_empty_string_returns_empty(self) -> None:
        """Empty string yields no URLs."""
        assert extract_urls("") == []

    @pytest.mark.unit
    def test_no_urls_returns_empty(self) -> None:
        """Text with no URLs yields empty list."""
        assert extract_urls("Hello world, no links here.") == []

    @pytest.mark.unit
    def test_single_http_url(self) -> None:
        """Single http URL is extracted."""
        result = extract_urls("Check out http://example.com for more.")
        assert result == ["http://example.com"]

    @pytest.mark.unit
    def test_single_https_url(self) -> None:
        """Single https URL is extracted."""
        result = extract_urls("See https://secure.example.com/path?q=1")
        assert result == ["https://secure.example.com/path?q=1"]

    @pytest.mark.unit
    def test_mixed_content_extracts_urls_only(self) -> None:
        """URLs extracted from mixed text content."""
        text = "Hello! Visit https://a.com and http://b.org/page for info."
        result = extract_urls(text)
        assert result == ["https://a.com", "http://b.org/page"]

    @pytest.mark.unit
    def test_duplicate_urls_deduped_order_preserved(self) -> None:
        """Duplicate URLs are deduplicated, first-occurrence order preserved."""
        text = (
            "See https://example.com first "
            "then https://other.org "
            "then https://example.com again"
        )
        result = extract_urls(text)
        assert result == ["https://example.com", "https://other.org"]

    @pytest.mark.unit
    def test_url_not_preceded_by_protocol_not_extracted(self) -> None:
        """Plain domain without http/https is not extracted."""
        result = extract_urls("Visit www.example.com today")
        assert result == []

    @pytest.mark.unit
    def test_multiple_urls_in_order(self) -> None:
        """Multiple unique URLs returned in order of first occurrence."""
        text = "https://first.com https://second.com https://third.com"
        result = extract_urls(text)
        assert result == [
            "https://first.com",
            "https://second.com",
            "https://third.com",
        ]

    @pytest.mark.unit
    def test_url_stops_at_whitespace(self) -> None:
        """URL parsing stops at whitespace."""
        result = extract_urls("https://example.com/path rest of sentence")
        assert result == ["https://example.com/path"]


# ---------------------------------------------------------------------------
# HandlerGmailIntentPoll tests
# ---------------------------------------------------------------------------


class TestHandlerGmailIntentPollSkipAndContinue:
    """Test skip-and-continue behavior when get_message fails on some messages."""

    @pytest.mark.unit
    async def test_get_message_fails_on_one_of_three_skips_and_continues(
        self,
    ) -> None:
        """If get_message fails on 1 of 3 messages, other 2 are still processed."""
        gmail = MagicMock()
        gmail.resolve_label_ids = AsyncMock(
            return_value={
                "to-read": "Label_to-read",
                "archived": "Label_archived",
                "processed": "Label_processed",
            }
        )
        gmail.search_messages = AsyncMock(return_value=[])
        gmail.list_messages = AsyncMock(
            return_value=[
                {"id": "msg1"},
                {"id": "msg2"},
                {"id": "msg3"},
            ]
        )
        # modify_labels for processed_label always succeeds
        gmail.modify_labels = AsyncMock(return_value=True)
        # msg2 fails, msg1 and msg3 succeed
        raw_msg1 = _make_raw_message("msg1", body_text="body1")
        raw_msg3 = _make_raw_message("msg3", body_text="body3 https://example.com")

        async def get_message_side_effect(
            message_id: str, message_format: str = "full"
        ) -> dict[str, Any]:
            if message_id == "msg2":
                return {}
            if message_id == "msg1":
                return raw_msg1
            return raw_msg3

        gmail.get_message = get_message_side_effect

        handler = _make_handler(gmail)
        config = _make_config()
        result = await handler.handle(config)

        assert isinstance(result, ModelGmailIntentPollerResult)
        assert result.messages_processed == 2
        assert len(result.pending_events) == 2
        assert result.hard_failed is False
        # One error for msg2 failure
        assert len(result.errors) == 1
        assert "msg2" in result.errors[0]
        # Archived count: msg1 + msg3 archived, msg2 skipped
        assert result.messages_archived == 2

    @pytest.mark.unit
    async def test_processed_label_apply_fails_skips_message(self) -> None:
        """If applying processed_label fails, message is skipped."""
        gmail = MagicMock()
        gmail.resolve_label_ids = AsyncMock(
            return_value={
                "to-read": "Label_to-read",
                "archived": "Label_archived",
                "processed": "Label_processed",
            }
        )
        gmail.search_messages = AsyncMock(return_value=[])
        gmail.list_messages = AsyncMock(return_value=[{"id": "msg1"}, {"id": "msg2"}])

        # msg1 processed_label apply fails, msg2 succeeds
        apply_count = {"n": 0}

        async def modify_labels_side_effect(
            message_id: str,
            add_label_ids: list[str],
            remove_label_ids: list[str],
        ) -> bool:
            # First call = apply processed_label to msg1 → fail
            # Second call = apply processed_label to msg2 → succeed
            # Third call = archive msg2 → succeed
            apply_count["n"] += 1
            if apply_count["n"] == 1:
                return False
            return True

        gmail.modify_labels = modify_labels_side_effect
        gmail.get_message = AsyncMock(
            return_value=_make_raw_message("msg2", body_text="hello")
        )

        handler = _make_handler(gmail)
        config = _make_config()
        result = await handler.handle(config)

        # Only msg2 processed
        assert result.messages_processed == 1
        assert len(result.pending_events) == 1
        assert len(result.errors) == 1
        assert "msg1" in result.errors[0]


class TestHandlerGmailIntentPollHardFail:
    """Test hard_failed flag when list_messages fails (returns [] due to error)."""

    @pytest.mark.unit
    async def test_hard_failed_not_set_on_empty_label(self) -> None:
        """An empty inbox (list_messages returns []) does not set hard_failed.

        Note: HandlerGmailApi cannot distinguish empty-inbox from failure —
        it always returns []. hard_failed is NOT set by the handler when
        list_messages returns [] for a valid resolved label.
        """
        gmail = MagicMock()
        gmail.resolve_label_ids = AsyncMock(
            return_value={
                "to-read": "Label_to-read",
                "archived": "Label_archived",
                "processed": "Label_processed",
            }
        )
        gmail.search_messages = AsyncMock(return_value=[])
        gmail.list_messages = AsyncMock(return_value=[])
        gmail.modify_labels = AsyncMock(return_value=True)

        handler = _make_handler(gmail)
        config = _make_config()
        result = await handler.handle(config)

        assert result.messages_processed == 0
        assert result.hard_failed is False
        assert result.pending_events == []

    @pytest.mark.unit
    async def test_unresolved_source_label_adds_error(self) -> None:
        """Source label that cannot be resolved adds error string.

        When a source label is missing from label_id_map (label not found
        in Gmail), the handler skips it and appends an error message.
        """
        gmail = MagicMock()
        # "nonexistent" label is NOT in the resolved map
        gmail.resolve_label_ids = AsyncMock(
            return_value={
                "archived": "Label_archived",
                "processed": "Label_processed",
            }
        )
        gmail.search_messages = AsyncMock(return_value=[])
        gmail.list_messages = AsyncMock(return_value=[])
        gmail.modify_labels = AsyncMock(return_value=True)

        handler = _make_handler(gmail)
        config = _make_config(source_labels=["nonexistent"])
        result = await handler.handle(config)

        assert result.messages_processed == 0
        assert len(result.errors) == 1
        assert "nonexistent" in result.errors[0]


class TestHandlerGmailIntentPollIdempotency:
    """Test that the recovery pass handles crashed-mid-run idempotency."""

    @pytest.mark.unit
    async def test_recovery_pass_archives_without_re_emitting(self) -> None:
        """Messages with processed_label still in source are archived, not re-emitted."""
        gmail = MagicMock()
        gmail.resolve_label_ids = AsyncMock(
            return_value={
                "to-read": "Label_to-read",
                "archived": "Label_archived",
                "processed": "Label_processed",
            }
        )
        # Recovery pass finds one crashed message
        gmail.search_messages = AsyncMock(return_value=[{"id": "crashed-msg"}])
        # Main pass finds no new messages
        gmail.list_messages = AsyncMock(return_value=[])
        gmail.modify_labels = AsyncMock(return_value=True)

        handler = _make_handler(gmail)
        config = _make_config()
        result = await handler.handle(config)

        # No events emitted (recovery only archives)
        assert result.messages_processed == 0
        assert len(result.pending_events) == 0
        # One message archived during recovery
        assert result.messages_archived == 1
        # No errors
        assert result.errors == []

        # Verify search_messages was called with recovery query
        gmail.search_messages.assert_called_once()
        call_args = gmail.search_messages.call_args
        query = call_args.kwargs.get("query") or call_args.args[0]
        assert "processed" in query
        assert "to-read" in query

    @pytest.mark.unit
    async def test_processed_label_applied_before_fetching(self) -> None:
        """processed_label is applied to a message BEFORE get_message is called.

        This is the idempotency guarantee: if the run crashes after applying
        the marker but before archiving, the recovery pass will handle it
        without re-emitting.
        """
        call_order: list[str] = []

        gmail = MagicMock()
        gmail.resolve_label_ids = AsyncMock(
            return_value={
                "to-read": "Label_to-read",
                "archived": "Label_archived",
                "processed": "Label_processed",
            }
        )
        gmail.search_messages = AsyncMock(return_value=[])
        gmail.list_messages = AsyncMock(return_value=[{"id": "msg1"}])

        async def modify_labels_tracker(
            message_id: str,
            add_label_ids: list[str],
            remove_label_ids: list[str],
        ) -> bool:
            if "Label_processed" in add_label_ids:
                call_order.append("apply_processed")
            elif "Label_archived" in add_label_ids:
                call_order.append("archive")
            return True

        async def get_message_tracker(
            message_id: str, message_format: str = "full"
        ) -> dict[str, Any]:
            call_order.append("get_message")
            return _make_raw_message(message_id)

        gmail.modify_labels = modify_labels_tracker
        gmail.get_message = get_message_tracker

        handler = _make_handler(gmail)
        config = _make_config()
        await handler.handle(config)

        # processed_label must be applied before get_message
        assert call_order[0] == "apply_processed"
        assert call_order[1] == "get_message"
        assert call_order[2] == "archive"


class TestHandlerGmailIntentPollMixedLabelStates:
    """Test behavior across multiple source labels with mixed states."""

    @pytest.mark.unit
    async def test_multiple_source_labels_all_processed(self) -> None:
        """Messages from multiple source labels are all processed."""
        gmail = MagicMock()
        gmail.resolve_label_ids = AsyncMock(
            return_value={
                "label-a": "Label_a",
                "label-b": "Label_b",
                "archived": "Label_archived",
                "processed": "Label_processed",
            }
        )
        gmail.search_messages = AsyncMock(return_value=[])

        async def list_messages_side_effect(
            label_ids: list[str], max_results: int = 50
        ) -> list[dict[str, Any]]:
            if "Label_a" in label_ids:
                return [{"id": "a1"}, {"id": "a2"}]
            if "Label_b" in label_ids:
                return [{"id": "b1"}]
            return []

        gmail.list_messages = list_messages_side_effect
        gmail.modify_labels = AsyncMock(return_value=True)

        msg_map: dict[str, dict[str, Any]] = {
            "a1": _make_raw_message("a1", body_text="https://a1.com"),
            "a2": _make_raw_message("a2", body_text="no urls here"),
            "b1": _make_raw_message("b1", body_text="https://b1.org"),
        }

        async def get_message_side_effect(
            message_id: str, message_format: str = "full"
        ) -> dict[str, Any]:
            return msg_map.get(message_id, {})

        gmail.get_message = get_message_side_effect

        handler = _make_handler(gmail)
        config = _make_config(source_labels=["label-a", "label-b"])
        result = await handler.handle(config)

        assert result.messages_processed == 3
        assert len(result.pending_events) == 3
        assert result.hard_failed is False
        assert result.errors == []
        assert result.messages_archived == 3

        # Verify source_label field in events
        source_labels_in_events = {
            str(ev.get("source_label"))
            for ev in result.pending_events  # type: ignore[index]
            if isinstance(ev, dict)
        }
        assert source_labels_in_events == {"label-a", "label-b"}

    @pytest.mark.unit
    async def test_events_contain_required_fields(self) -> None:
        """Each pending event payload contains all required fields."""
        gmail = MagicMock()
        gmail.resolve_label_ids = AsyncMock(
            return_value={
                "to-read": "Label_to-read",
                "archived": "Label_archived",
                "processed": "Label_processed",
            }
        )
        gmail.search_messages = AsyncMock(return_value=[])
        gmail.list_messages = AsyncMock(return_value=[{"id": "msg1"}])
        gmail.modify_labels = AsyncMock(return_value=True)
        gmail.get_message = AsyncMock(
            return_value=_make_raw_message(
                "msg1",
                subject="Hello",
                sender="sender@example.com",
                body_text="Check https://example.com for details",
            )
        )

        handler = _make_handler(gmail)
        config = _make_config()
        result = await handler.handle(config)

        assert len(result.pending_events) == 1
        event = result.pending_events[0]
        assert isinstance(event, dict)

        required_keys = {
            "event_type",
            "message_id",
            "subject",
            "body_text",
            "urls",
            "source_label",
            "sender",
            "received_at",
            "partition_key",
        }
        assert required_keys.issubset(event.keys())
        assert event["partition_key"] == event["message_id"]
        assert event["source_label"] == "to-read"
        assert isinstance(event["urls"], list)
        assert "https://example.com" in event["urls"]

    @pytest.mark.unit
    async def test_body_text_truncated_to_4096_chars(self) -> None:
        """body_text in event payload is truncated to 4096 characters."""
        long_body = "A" * 5000 + " https://end.com"
        gmail = MagicMock()
        gmail.resolve_label_ids = AsyncMock(
            return_value={
                "to-read": "Label_to-read",
                "archived": "Label_archived",
                "processed": "Label_processed",
            }
        )
        gmail.search_messages = AsyncMock(return_value=[])
        gmail.list_messages = AsyncMock(return_value=[{"id": "msg1"}])
        gmail.modify_labels = AsyncMock(return_value=True)
        gmail.get_message = AsyncMock(
            return_value=_make_raw_message("msg1", body_text=long_body)
        )

        handler = _make_handler(gmail)
        config = _make_config()
        result = await handler.handle(config)

        assert len(result.pending_events) == 1
        event = result.pending_events[0]
        assert isinstance(event, dict)
        assert len(str(event["body_text"])) <= 4096

    @pytest.mark.unit
    async def test_events_published_zero_runtime_sets_it(self) -> None:
        """events_published is 0 in handler result (runtime sets it)."""
        gmail = MagicMock()
        gmail.resolve_label_ids = AsyncMock(
            return_value={
                "to-read": "Label_to-read",
                "archived": "Label_archived",
                "processed": "Label_processed",
            }
        )
        gmail.search_messages = AsyncMock(return_value=[])
        gmail.list_messages = AsyncMock(return_value=[{"id": "msg1"}])
        gmail.modify_labels = AsyncMock(return_value=True)
        gmail.get_message = AsyncMock(return_value=_make_raw_message("msg1"))

        handler = _make_handler(gmail)
        config = _make_config()
        result = await handler.handle(config)

        # Handler always returns 0 — runtime sets this after publishing
        assert result.events_published == 0


# ---------------------------------------------------------------------------
# Gmail body parsing tests (via ModelGmailMessage)
# ---------------------------------------------------------------------------


class TestGmailBodyParsing:
    """Test Gmail body parsing behavior via ModelGmailMessage.from_api_response."""

    @pytest.mark.unit
    def test_multipart_alternative_plain_wins_over_html(self) -> None:
        """multipart/alternative: text/plain is preferred over text/html."""
        from omnibase_infra.handlers.models.model_gmail_message import ModelGmailMessage

        raw = _make_multipart_raw_message(
            plain_text="Plain text wins",
            html_text="<b>HTML text</b>",
        )
        msg = ModelGmailMessage.from_api_response(raw)
        assert msg.body_text == "Plain text wins"

    @pytest.mark.unit
    def test_multipart_alternative_no_plain_falls_back_to_html(self) -> None:
        """multipart/alternative with no text/plain falls back to text/html."""
        from omnibase_infra.handlers.models.model_gmail_message import ModelGmailMessage

        raw: dict[str, Any] = {
            "id": "msg-html-only",
            "threadId": "thread-html",
            "labelIds": [],
            "internalDate": "1700000000000",
            "payload": {
                "mimeType": "multipart/alternative",
                "headers": [
                    {"name": "Subject", "value": "HTML Only"},
                    {"name": "From", "value": "html@example.com"},
                ],
                "parts": [
                    {
                        "mimeType": "text/html",
                        "body": {"data": _b64url("<b>HTML content</b>")},
                    },
                ],
            },
        }
        msg = ModelGmailMessage.from_api_response(raw)
        assert msg.body_text == "<b>HTML content</b>"

    @pytest.mark.unit
    def test_no_text_plain_or_html_returns_empty_string(self) -> None:
        """Message with no text parts yields empty body_text."""
        from omnibase_infra.handlers.models.model_gmail_message import ModelGmailMessage

        raw: dict[str, Any] = {
            "id": "msg-no-body",
            "threadId": "thread-no-body",
            "labelIds": [],
            "internalDate": "1700000000000",
            "payload": {
                "mimeType": "application/octet-stream",
                "headers": [
                    {"name": "Subject", "value": "No Body"},
                    {"name": "From", "value": "notext@example.com"},
                ],
                "body": {"data": ""},
            },
        }
        msg = ModelGmailMessage.from_api_response(raw)
        assert msg.body_text == ""

    @pytest.mark.unit
    def test_deeply_nested_parts_extracted(self) -> None:
        """Deeply nested MIME structure: text/plain found in nested multipart."""
        from omnibase_infra.handlers.models.model_gmail_message import ModelGmailMessage

        raw: dict[str, Any] = {
            "id": "msg-deep",
            "threadId": "thread-deep",
            "labelIds": [],
            "internalDate": "1700000000000",
            "payload": {
                "mimeType": "multipart/mixed",
                "headers": [
                    {"name": "Subject", "value": "Nested"},
                    {"name": "From", "value": "nested@example.com"},
                ],
                "parts": [
                    {
                        "mimeType": "multipart/alternative",
                        "parts": [
                            {
                                "mimeType": "text/plain",
                                "body": {"data": _b64url("Deeply nested plain text")},
                            },
                            {
                                "mimeType": "text/html",
                                "body": {"data": _b64url("<p>HTML</p>")},
                            },
                        ],
                    },
                ],
            },
        }
        msg = ModelGmailMessage.from_api_response(raw)
        assert "Deeply nested plain text" in msg.body_text

    @pytest.mark.unit
    def test_url_extraction_from_subject_and_body(self) -> None:
        """URLs are extracted from subject + body_text combined."""
        gmail = MagicMock()
        gmail.resolve_label_ids = AsyncMock(
            return_value={
                "to-read": "Label_to-read",
                "archived": "Label_archived",
                "processed": "Label_processed",
            }
        )
        gmail.search_messages = AsyncMock(return_value=[])
        gmail.list_messages = AsyncMock(return_value=[{"id": "msg1"}])
        gmail.modify_labels = AsyncMock(return_value=True)

        # URL in subject + URL in body
        raw = _make_raw_message(
            "msg1",
            subject="See https://subject-url.com for info",
            body_text="Also see https://body-url.org and https://subject-url.com again",
        )
        gmail.get_message = AsyncMock(return_value=raw)

        import asyncio

        handler = _make_handler(gmail)
        config = _make_config()

        async def run() -> ModelGmailIntentPollerResult:
            return await handler.handle(config)

        result = asyncio.get_event_loop().run_until_complete(run())

        assert len(result.pending_events) == 1
        event = result.pending_events[0]
        assert isinstance(event, dict)
        urls = event["urls"]
        assert isinstance(urls, list)
        # subject URL + body URL; subject URL deduplicated (appears in body too)
        assert "https://subject-url.com" in urls
        assert "https://body-url.org" in urls
        # deduplicated: subject-url.com appears only once
        assert urls.count("https://subject-url.com") == 1
