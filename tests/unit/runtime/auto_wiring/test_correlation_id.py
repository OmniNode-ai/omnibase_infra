# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for mandatory correlation_id enforcement in auto-wiring dispatch (OMN-7675)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from omnibase_infra.runtime.auto_wiring.context import ModelAutoWiringContext
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _make_dispatch_callback,
    _validate_correlation_id,
    _validate_output_events_correlation_id,
)

# ---------------------------------------------------------------------------
# ModelAutoWiringContext — correlation_id is mandatory
# ---------------------------------------------------------------------------


class TestModelAutoWiringContextCorrelationId:
    """correlation_id must be required, non-optional, non-empty."""

    def test_rejects_construction_without_correlation_id(self) -> None:
        """Omitting correlation_id must raise ValidationError."""
        with pytest.raises(ValidationError):
            ModelAutoWiringContext(
                handler_id="handler.test",
                node_kind="COMPUTE",
                phase="on_start",
            )

    def test_rejects_empty_string_correlation_id(self) -> None:
        """Empty string must be rejected by min_length=1."""
        with pytest.raises(ValidationError):
            ModelAutoWiringContext(
                handler_id="handler.test",
                node_kind="COMPUTE",
                phase="on_start",
                correlation_id="",
            )

    def test_accepts_valid_correlation_id(self) -> None:
        ctx = ModelAutoWiringContext(
            handler_id="handler.test",
            node_kind="COMPUTE",
            phase="on_start",
            correlation_id="abc-123",
        )
        assert ctx.correlation_id == "abc-123"


# ---------------------------------------------------------------------------
# _validate_correlation_id — envelope-level guard
# ---------------------------------------------------------------------------


class TestValidateCorrelationId:
    def test_raises_when_missing(self) -> None:
        envelope = MagicMock(spec=[])  # no attributes
        with pytest.raises(RuntimeError, match="missing correlation_id"):
            _validate_correlation_id(envelope)

    def test_raises_when_none(self) -> None:
        envelope = MagicMock()
        envelope.correlation_id = None
        with pytest.raises(RuntimeError, match="missing correlation_id"):
            _validate_correlation_id(envelope)

    def test_raises_when_empty_string(self) -> None:
        envelope = MagicMock()
        envelope.correlation_id = ""
        with pytest.raises(RuntimeError, match="correlation_id is empty"):
            _validate_correlation_id(envelope)

    def test_raises_when_whitespace_only(self) -> None:
        envelope = MagicMock()
        envelope.correlation_id = "   "
        with pytest.raises(RuntimeError, match="correlation_id is empty"):
            _validate_correlation_id(envelope)

    def test_returns_valid_id(self) -> None:
        envelope = MagicMock()
        envelope.correlation_id = "trace-42"
        result = _validate_correlation_id(envelope)
        assert result == "trace-42"


# ---------------------------------------------------------------------------
# _make_dispatch_callback — integration with correlation_id validation
# ---------------------------------------------------------------------------


class TestDispatchCallbackCorrelationId:
    @pytest.mark.asyncio
    async def test_raises_before_calling_handler_if_missing(self) -> None:
        handler = MagicMock()
        handler.handle = AsyncMock()
        callback = _make_dispatch_callback(handler)

        envelope = MagicMock(spec=[])  # no correlation_id
        with pytest.raises(RuntimeError, match="missing correlation_id"):
            await callback(envelope)

        handler.handle.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_before_calling_handler_if_empty(self) -> None:
        handler = MagicMock()
        handler.handle = AsyncMock()
        callback = _make_dispatch_callback(handler)

        envelope = MagicMock()
        envelope.correlation_id = ""
        with pytest.raises(RuntimeError, match="correlation_id is empty"):
            await callback(envelope)

        handler.handle.assert_not_called()

    @pytest.mark.asyncio
    async def test_handler_receives_envelope_with_correlation_id(self) -> None:
        handler = MagicMock()
        handler.handle = AsyncMock(return_value=None)
        callback = _make_dispatch_callback(handler)

        envelope = MagicMock()
        envelope.correlation_id = "cid-999"
        result = await callback(envelope)

        handler.handle.assert_called_once_with(envelope)
        assert result is None


# ---------------------------------------------------------------------------
# _validate_output_events_correlation_id — output event guard
# ---------------------------------------------------------------------------


class TestValidateOutputEventsCorrelationId:
    def test_passes_when_no_output_events(self) -> None:
        result = MagicMock()
        result.output_events = []
        _validate_output_events_correlation_id(result)  # no error

    def test_passes_when_events_have_correlation_id(self) -> None:
        event = MagicMock()
        event.correlation_id = "cid-ok"
        result = MagicMock()
        result.output_events = [event]
        _validate_output_events_correlation_id(result)  # no error

    def test_raises_when_event_missing_correlation_id(self) -> None:
        event = MagicMock(spec=[])  # no attributes
        result = MagicMock()
        result.output_events = [event]
        with pytest.raises(
            RuntimeError, match=r"output_event\[0\].*missing correlation_id"
        ):
            _validate_output_events_correlation_id(result)

    def test_raises_when_event_has_empty_correlation_id(self) -> None:
        event = MagicMock()
        event.correlation_id = ""
        result = MagicMock()
        result.output_events = [event]
        with pytest.raises(
            RuntimeError, match=r"output_event\[0\].*empty correlation_id"
        ):
            _validate_output_events_correlation_id(result)

    @pytest.mark.asyncio
    async def test_callback_validates_output_events(self) -> None:
        """End-to-end: callback raises if handler returns events without correlation_id."""
        bad_event = MagicMock(spec=[])  # no correlation_id
        dispatch_result = MagicMock()
        dispatch_result.output_events = [bad_event]

        handler = MagicMock()
        handler.handle = AsyncMock(return_value=dispatch_result)
        callback = _make_dispatch_callback(handler)

        envelope = MagicMock()
        envelope.correlation_id = "cid-valid"

        with pytest.raises(
            RuntimeError, match=r"output_event\[0\].*missing correlation_id"
        ):
            await callback(envelope)
