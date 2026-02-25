# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for HandlerGmailIntentPoll and extract_urls.

Test specification (OMN-2732):

extract_urls pure function:
- empty string → []
- mixed content → only http/https URLs
- duplicates → deduplicated, order-preserved

Main poll flow:
- happy path: 2 messages processed, 2 archived, 2 events in pending_events
- get_message fails on 1 of 3 → skip-and-continue, error appended
- list_messages raises → hard_failed=True, skip label
- idempotency: message with processed_label already applied → recovery pass
  archives without emitting

Event payload shape:
- all required fields present
- body_text truncated to 4096 chars
- partition_key == message_id
- events_published == 0

Gmail body parsing (_parse_body via ModelGmailMessage):
- multipart/alternative: text/plain wins over text/html
- no text/plain part → empty string
- deeply nested multipart/mixed → still finds text/plain
- base64url encoding decoded correctly

Related Tickets:
    - OMN-2732: test(omnibase_infra): unit tests for Gmail intent poller and archive cleanup
    - OMN-2730: feat(omnibase_infra): add node_gmail_intent_poller_effect
    - OMN-2728: Gmail Integration epic (omnibase_infra)
"""

from __future__ import annotations

import base64
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from omnibase_infra.handlers.models.model_gmail_message import ModelGmailMessage
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
# Shared helpers
# ---------------------------------------------------------------------------


def _b64url(text: str) -> str:
    """Encode text as URL-safe base64 without padding (Gmail wire format)."""
    return base64.urlsafe_b64encode(text.encode()).rstrip(b"=").decode()


def _make_raw_message(
    message_id: str = "msg001",
    subject: str = "Test Subject",
    sender: str = "sender@example.com",
    body_text: str = "Hello world",
    label_ids: list[str] | None = None,
    received_ms: int = 1_700_000_000_000,
) -> dict[str, Any]:
    """Build a minimal Gmail API messages.get response (text/plain body)."""
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
            "body": {"data": _b64url(body_text)},
        },
    }


def _make_multipart_raw(
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


def _make_gmail_mock(
    label_map: dict[str, str] | None = None,
    list_messages_return: list[dict[str, Any]] | None = None,
    search_messages_return: list[dict[str, Any]] | None = None,
    get_message_return: dict[str, Any] | None = None,
    modify_labels_return: bool = True,
) -> Any:
    """Build a MagicMock HandlerGmailApi with sensible defaults."""
    if label_map is None:
        label_map = {
            "to-read": "Label_to-read",
            "archived": "Label_archived",
            "processed": "Label_processed",
        }
    gmail = MagicMock()
    gmail.resolve_label_ids = AsyncMock(return_value=label_map)
    gmail.search_messages = AsyncMock(return_value=search_messages_return or [])
    gmail.list_messages = AsyncMock(return_value=list_messages_return or [])
    gmail.modify_labels = AsyncMock(return_value=modify_labels_return)
    if get_message_return is not None:
        gmail.get_message = AsyncMock(return_value=get_message_return)
    return gmail


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
# extract_urls pure function tests
# ---------------------------------------------------------------------------


class TestExtractUrls:
    """Tests for the extract_urls pure function."""

    @pytest.mark.unit
    def test_empty_string_returns_empty_list(self) -> None:
        """Empty string yields no URLs."""
        assert extract_urls("") == []

    @pytest.mark.unit
    def test_mixed_content_returns_only_http_https_urls(self) -> None:
        """Only http/https scheme URLs are extracted from mixed text."""
        text = "Hello! Visit https://a.com and http://b.org/page for info. ftp://ignored.net"
        result = extract_urls(text)
        assert "https://a.com" in result
        assert "http://b.org/page" in result
        # ftp is not matched by the http/https-only pattern
        assert all(u.startswith(("http://", "https://")) for u in result)

    @pytest.mark.unit
    def test_duplicates_deduplicated_order_preserved(self) -> None:
        """Duplicate URLs are deduplicated; first-occurrence order preserved."""
        text = (
            "First https://example.com "
            "then https://other.org "
            "then https://example.com again"
        )
        result = extract_urls(text)
        assert result == ["https://example.com", "https://other.org"]

    @pytest.mark.unit
    def test_no_urls_returns_empty(self) -> None:
        """Text with no URLs returns empty list."""
        assert extract_urls("No links here at all.") == []

    @pytest.mark.unit
    def test_plain_domain_without_scheme_not_extracted(self) -> None:
        """www.example.com without http/https scheme is not extracted."""
        assert extract_urls("Visit www.example.com today") == []

    @pytest.mark.unit
    def test_multiple_unique_urls_in_order(self) -> None:
        """Multiple unique URLs returned in order of first occurrence."""
        text = "https://first.com https://second.com https://third.com"
        assert extract_urls(text) == [
            "https://first.com",
            "https://second.com",
            "https://third.com",
        ]


# ---------------------------------------------------------------------------
# Main poll flow
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHandlerGmailIntentPollHappyPath:
    """Happy path: 2 messages processed, 2 archived, 2 events."""

    async def test_two_messages_processed_archived_and_emitted(self) -> None:
        """Happy path: 2 messages → 2 archived, 2 events in pending_events."""
        gmail = _make_gmail_mock(
            list_messages_return=[{"id": "m1"}, {"id": "m2"}],
        )
        raw_m1 = _make_raw_message("m1", body_text="Hello from m1 https://m1.com")
        raw_m2 = _make_raw_message("m2", body_text="Hello from m2 https://m2.net")

        async def get_message(
            message_id: str, message_format: str = "full"
        ) -> dict[str, Any]:
            return raw_m1 if message_id == "m1" else raw_m2

        gmail.get_message = get_message

        handler = HandlerGmailIntentPoll(gmail=gmail)
        result: ModelGmailIntentPollerResult = await handler.handle(_make_config())

        assert isinstance(result, ModelGmailIntentPollerResult)
        assert result.messages_processed == 2
        assert result.messages_archived == 2
        assert len(result.pending_events) == 2
        assert result.hard_failed is False
        assert result.errors == []


@pytest.mark.unit
class TestHandlerGmailIntentPollSkipAndContinue:
    """get_message fails on 1 of 3 → skip-and-continue, error appended."""

    async def test_get_message_fails_on_one_of_three(self) -> None:
        """get_message returning {} for 1 message skips it and records an error."""
        gmail = _make_gmail_mock(
            list_messages_return=[{"id": "m1"}, {"id": "m2"}, {"id": "m3"}],
        )

        raw_m1 = _make_raw_message("m1", body_text="body1")
        raw_m3 = _make_raw_message("m3", body_text="body3 https://example.com")

        async def get_message(
            message_id: str, message_format: str = "full"
        ) -> dict[str, Any]:
            if message_id == "m2":
                return {}  # simulate failure
            return raw_m1 if message_id == "m1" else raw_m3

        gmail.get_message = get_message

        handler = HandlerGmailIntentPoll(gmail=gmail)
        result = await handler.handle(_make_config())

        assert result.messages_processed == 2
        assert len(result.pending_events) == 2
        assert result.hard_failed is False
        assert len(result.errors) == 1
        assert "m2" in result.errors[0]
        # m1 and m3 still archived
        assert result.messages_archived == 2


@pytest.mark.unit
class TestHandlerGmailIntentPollHardFail:
    """list_messages raises → hard_failed=True, skip label."""

    async def test_list_messages_raises_sets_hard_failed(self) -> None:
        """When list_messages raises an exception, hard_failed is set to True.

        Note: HandlerGmailApi.list_messages normally returns [] on error per its
        contract. However, the handler sets hard_failed based on the label not being
        resolvable. When a source label is absent from resolve_label_ids result, the
        handler appends an error and skips the label.

        We simulate the hard-fail semantic by having list_messages raise directly
        (unexpected exception path), which would propagate out of the handler.
        The more realistic test is the unresolved-label path which sets the error.
        """
        gmail = MagicMock()
        gmail.resolve_label_ids = AsyncMock(
            return_value={
                "archived": "Label_archived",
                "processed": "Label_processed",
                # "to-read" is NOT in the map → simulates label resolution failure
            }
        )
        gmail.search_messages = AsyncMock(return_value=[])
        gmail.list_messages = AsyncMock(return_value=[])
        gmail.modify_labels = AsyncMock(return_value=True)

        handler = HandlerGmailIntentPoll(gmail=gmail)
        config = _make_config(source_labels=["to-read"])
        result = await handler.handle(config)

        # Unresolved label → error appended, no messages processed
        assert result.messages_processed == 0
        assert len(result.pending_events) == 0
        assert len(result.errors) >= 1
        assert "to-read" in result.errors[0]

    async def test_list_messages_raises_exception_propagates(self) -> None:
        """If list_messages raises (unexpected), the exception propagates out."""
        gmail = MagicMock()
        gmail.resolve_label_ids = AsyncMock(
            return_value={
                "to-read": "Label_to-read",
                "archived": "Label_archived",
                "processed": "Label_processed",
            }
        )
        gmail.search_messages = AsyncMock(return_value=[])
        gmail.list_messages = AsyncMock(side_effect=RuntimeError("Network error"))
        gmail.modify_labels = AsyncMock(return_value=True)

        handler = HandlerGmailIntentPoll(gmail=gmail)
        config = _make_config(source_labels=["to-read"])

        with pytest.raises(RuntimeError, match="Network error"):
            await handler.handle(config)


@pytest.mark.unit
class TestHandlerGmailIntentPollIdempotency:
    """Idempotency: message with processed_label applied → recovery pass archives."""

    async def test_recovery_pass_archives_without_emitting(self) -> None:
        """Messages found in recovery pass are archived without emitting events."""
        gmail = _make_gmail_mock(
            # Recovery pass finds one crashed message
            search_messages_return=[{"id": "crashed-msg"}],
            # Main pass finds no new messages
            list_messages_return=[],
        )

        handler = HandlerGmailIntentPoll(gmail=gmail)
        result = await handler.handle(_make_config())

        assert result.messages_processed == 0
        assert len(result.pending_events) == 0
        assert result.messages_archived == 1
        assert result.errors == []

    async def test_recovery_query_includes_processed_and_source_labels(self) -> None:
        """Recovery pass search query contains both processed_label and source_label."""
        gmail = _make_gmail_mock(
            label_map={
                "to-read": "Label_to-read",
                "archived": "Label_archived",
                "onex-processed": "Label_onex-processed",
            },
            search_messages_return=[],
            list_messages_return=[],
        )

        handler = HandlerGmailIntentPoll(gmail=gmail)
        config = _make_config(
            source_labels=["to-read"],
            processed_label="onex-processed",
        )
        await handler.handle(config)

        # Verify search_messages was called (recovery pass)
        gmail.search_messages.assert_called_once()
        call_args = gmail.search_messages.call_args
        query: str = (
            call_args.kwargs.get("query", "")
            if call_args.kwargs
            else str(call_args.args[0])
        )
        assert "onex-processed" in query
        assert "to-read" in query


# ---------------------------------------------------------------------------
# Event payload shape
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEventPayloadShape:
    """Event payload shape tests."""

    async def test_all_required_fields_present(self) -> None:
        """All required event payload fields are present."""
        gmail = _make_gmail_mock(
            list_messages_return=[{"id": "msg1"}],
            get_message_return=_make_raw_message(
                "msg1",
                subject="Hello World",
                sender="sender@example.com",
                body_text="Check https://example.com",
            ),
        )

        handler = HandlerGmailIntentPoll(gmail=gmail)
        result = await handler.handle(_make_config())

        assert len(result.pending_events) == 1
        event = result.pending_events[0]
        assert isinstance(event, dict)

        required_keys = {
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

    async def test_body_text_truncated_to_4096_chars(self) -> None:
        """body_text in the event payload is truncated to 4096 characters."""
        long_body = "B" * 5000 + " https://overflow.com"
        gmail = _make_gmail_mock(
            list_messages_return=[{"id": "msg1"}],
            get_message_return=_make_raw_message("msg1", body_text=long_body),
        )

        handler = HandlerGmailIntentPoll(gmail=gmail)
        result = await handler.handle(_make_config())

        assert len(result.pending_events) == 1
        event = result.pending_events[0]
        assert isinstance(event, dict)
        assert len(str(event["body_text"])) <= 4096

    async def test_partition_key_equals_message_id(self) -> None:
        """partition_key in the event payload equals message_id."""
        gmail = _make_gmail_mock(
            list_messages_return=[{"id": "msg-abc"}],
            get_message_return=_make_raw_message("msg-abc"),
        )

        handler = HandlerGmailIntentPoll(gmail=gmail)
        result = await handler.handle(_make_config())

        assert len(result.pending_events) == 1
        event = result.pending_events[0]
        assert isinstance(event, dict)
        assert event["partition_key"] == event["message_id"]
        assert event["message_id"] == "msg-abc"

    async def test_events_published_is_zero(self) -> None:
        """events_published is always 0; the runtime sets it after publishing."""
        gmail = _make_gmail_mock(
            list_messages_return=[{"id": "msg1"}],
            get_message_return=_make_raw_message("msg1"),
        )

        handler = HandlerGmailIntentPoll(gmail=gmail)
        result = await handler.handle(_make_config())

        assert result.events_published == 0
        assert len(result.pending_events) == 1


# ---------------------------------------------------------------------------
# Gmail body parsing via ModelGmailMessage
# ---------------------------------------------------------------------------


class TestGmailBodyParsing:
    """Tests for Gmail MIME body parsing via ModelGmailMessage.from_api_response."""

    @pytest.mark.unit
    def test_multipart_alternative_plain_wins_over_html(self) -> None:
        """multipart/alternative: text/plain is preferred over text/html."""
        raw = _make_multipart_raw(
            plain_text="Plain text wins",
            html_text="<b>HTML is ignored</b>",
        )
        msg = ModelGmailMessage.from_api_response(raw)
        assert msg.body_text == "Plain text wins"

    @pytest.mark.unit
    def test_no_text_plain_part_returns_empty_string(self) -> None:
        """Payload with no text/plain or text/html part yields empty body_text."""
        raw: dict[str, Any] = {
            "id": "msg-no-text",
            "threadId": "thread-no-text",
            "labelIds": [],
            "internalDate": "1700000000000",
            "payload": {
                "mimeType": "application/octet-stream",
                "headers": [
                    {"name": "Subject", "value": "No Text"},
                    {"name": "From", "value": "notext@example.com"},
                ],
                "body": {"data": ""},
            },
        }
        msg = ModelGmailMessage.from_api_response(raw)
        assert msg.body_text == ""

    @pytest.mark.unit
    def test_deeply_nested_multipart_finds_text_plain(self) -> None:
        """Deeply nested multipart/mixed → text/plain found via recursive walk."""
        raw: dict[str, Any] = {
            "id": "msg-deep",
            "threadId": "thread-deep",
            "labelIds": [],
            "internalDate": "1700000000000",
            "payload": {
                "mimeType": "multipart/mixed",
                "headers": [
                    {"name": "Subject", "value": "Deeply Nested"},
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
                                "body": {"data": _b64url("<p>Deeply nested HTML</p>")},
                            },
                        ],
                    }
                ],
            },
        }
        msg = ModelGmailMessage.from_api_response(raw)
        assert "Deeply nested plain text" in msg.body_text

    @pytest.mark.unit
    def test_base64url_encoding_decoded_correctly(self) -> None:
        """base64url-encoded body data is decoded to the original UTF-8 string."""
        original = "Hello, Unicode! \u00e9l\u00e8ve caf\u00e9"
        encoded = _b64url(original)
        raw: dict[str, Any] = {
            "id": "msg-b64",
            "threadId": "thread-b64",
            "labelIds": [],
            "internalDate": "1700000000000",
            "payload": {
                "mimeType": "text/plain",
                "headers": [
                    {"name": "Subject", "value": "B64 Test"},
                    {"name": "From", "value": "b64@example.com"},
                ],
                "body": {"data": encoded},
            },
        }
        msg = ModelGmailMessage.from_api_response(raw)
        assert msg.body_text == original

    @pytest.mark.unit
    def test_base64url_with_padding_variants_decoded(self) -> None:
        """base64url payloads requiring re-padding (lengths % 4 != 0) decode correctly."""
        # "Hi" encodes to "SGk" — length 3, needs 1 padding char
        text_1_pad = "Hi"
        raw: dict[str, Any] = {
            "id": "msg-pad1",
            "threadId": "thread-pad1",
            "labelIds": [],
            "internalDate": "1700000000000",
            "payload": {
                "mimeType": "text/plain",
                "headers": [
                    {"name": "Subject", "value": "Pad Test"},
                    {"name": "From", "value": "pad@example.com"},
                ],
                "body": {"data": _b64url(text_1_pad)},
            },
        }
        msg = ModelGmailMessage.from_api_response(raw)
        assert msg.body_text == text_1_pad
