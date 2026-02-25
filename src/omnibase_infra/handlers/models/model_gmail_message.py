# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Gmail Message Model.

Typed representation of a Gmail message returned by the Gmail REST API.
Fields are extracted from the raw API response (headers, body parts).

Related Tickets:
    - OMN-2729: Add HandlerGmailApi shared OAuth2 + REST client
    - OMN-2728: Gmail Integration epic (omnibase_infra)
"""

from __future__ import annotations

import base64
import logging
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, field_validator

from omnibase_core.types import JsonType

logger = logging.getLogger(__name__)

# Type alias for raw Gmail API response dicts
_ApiDict = dict[str, JsonType]


class ModelGmailMessage(BaseModel):
    """Immutable representation of a Gmail message.

    Constructed from the raw Gmail API ``messages.get`` response.
    Body text is decoded from base64url encoding; multipart/alternative
    payloads prefer text/plain over text/html.

    Attributes:
        message_id: Gmail message ID (``id`` field).
        thread_id: Gmail thread ID (``threadId`` field).
        subject: Subject header value, empty string if absent.
        sender: From header value, empty string if absent.
        received_at: UTC datetime parsed from ``internalDate`` (epoch ms).
        body_text: Decoded plain-text body (text/plain part); falls back
            to text/html if no plain part exists; empty string if neither.
        label_ids: List of Gmail label IDs applied to the message.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    message_id: str
    thread_id: str
    subject: str = ""
    sender: str = ""
    received_at: datetime
    body_text: str = ""
    label_ids: list[str] = []

    @field_validator("received_at")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        """Ensure received_at is timezone-aware UTC."""
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v.astimezone(UTC)

    @classmethod
    def from_api_response(cls, raw: _ApiDict) -> ModelGmailMessage:
        """Construct from a raw Gmail API ``messages.get`` response dict.

        Args:
            raw: Full response dict from Gmail API (format="full").

        Returns:
            ModelGmailMessage with all fields populated.
        """
        message_id: str = str(raw.get("id", ""))
        thread_id: str = str(raw.get("threadId", ""))
        raw_label_ids = raw.get("labelIds", [])
        label_ids: list[str] = _extract_string_list(raw_label_ids)

        # Parse internalDate (epoch milliseconds → UTC datetime)
        internal_date_ms_str: str = str(raw.get("internalDate", "0"))
        try:
            internal_date_ms = int(internal_date_ms_str)
        except (ValueError, TypeError):
            internal_date_ms = 0
        received_at = datetime.fromtimestamp(internal_date_ms / 1000.0, tz=UTC)

        # Extract headers from payload
        raw_payload = raw.get("payload", {})
        payload: _ApiDict = raw_payload if isinstance(raw_payload, dict) else {}
        headers: list[_ApiDict] = _extract_list_of_dicts(payload.get("headers", []))
        subject = ""
        sender = ""
        for header in headers:
            if not isinstance(header, dict):
                continue
            name = str(header.get("name", "")).lower()
            value = str(header.get("value", ""))
            if name == "subject":
                subject = value
            elif name == "from":
                sender = value

        # Extract body text by walking payload parts recursively
        body_text = _extract_body_text(payload)

        return cls(
            message_id=message_id,
            thread_id=thread_id,
            subject=subject,
            sender=sender,
            received_at=received_at,
            body_text=body_text,
            label_ids=label_ids,
        )


def _extract_body_text(payload: _ApiDict) -> str:
    """Recursively walk MIME payload parts to extract text content.

    Prefers text/plain over text/html within multipart/alternative.
    For other multipart types, concatenates all text parts found.
    Decodes base64url-encoded data.

    Args:
        payload: Gmail API payload dict (may contain ``parts`` list).

    Returns:
        Decoded text content, or empty string if none found.
    """
    mime_type: str = str(payload.get("mimeType", ""))

    if mime_type == "multipart/alternative":
        # Prefer plain text, fall back to HTML
        plain_text = ""
        html_text = ""
        parts: list[_ApiDict] = _extract_list_of_dicts(payload.get("parts", []))
        for part in parts:
            part_mime = str(part.get("mimeType", ""))
            if part_mime == "text/plain":
                plain_text = _decode_body_data(part)
            elif part_mime == "text/html":
                html_text = _decode_body_data(part)
        return plain_text if plain_text else html_text

    elif mime_type.startswith("multipart/"):
        # For multipart/mixed, multipart/related, etc.: recurse and concatenate
        parts = _extract_list_of_dicts(payload.get("parts", []))
        collected: list[str] = []
        for part in parts:
            text = _extract_body_text(part)
            if text:
                collected.append(text)
        return "\n".join(collected)

    elif mime_type in ("text/plain", "text/html"):
        return _decode_body_data(payload)

    return ""


def _decode_body_data(part: _ApiDict) -> str:
    """Decode base64url body data from a single MIME part.

    Args:
        part: A single MIME part dict from the Gmail API payload.

    Returns:
        Decoded text string, or empty string on decode failure.
    """
    raw_body = part.get("body", {})
    body: _ApiDict = raw_body if isinstance(raw_body, dict) else {}
    data: str = str(body.get("data", ""))
    if not data or data == "None":
        return ""
    try:
        # Gmail uses URL-safe base64 without padding
        padded = data + "=" * (4 - len(data) % 4) if len(data) % 4 else data
        decoded_bytes = base64.urlsafe_b64decode(padded)
        return decoded_bytes.decode("utf-8", errors="replace")
    except Exception as exc:
        logger.warning(
            "Failed to decode Gmail body part data",
            extra={"error": str(exc)},
        )
        return ""


def _extract_list_of_dicts(value: JsonType) -> list[_ApiDict]:
    """Safely extract a list of dicts from a JsonType value.

    Args:
        value: A JSON value that may be a list of dict objects.

    Returns:
        List of dict[str, JsonType] items, or empty list if value is not
        a list or contains non-dict items.
    """
    if not isinstance(value, list):
        return []
    result: list[_ApiDict] = []
    for item in value:
        if isinstance(item, dict):
            result.append(item)
    return result


def _extract_string_list(value: JsonType) -> list[str]:
    """Safely extract a list of strings from a JsonType value.

    Args:
        value: A JSON value that may be a list of strings.

    Returns:
        List of strings, or empty list if value is not a list.
    """
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


__all__: list[str] = ["ModelGmailMessage"]
