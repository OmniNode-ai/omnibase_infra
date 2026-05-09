# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Integration test: HandlerLedgerAppend emits WARNING when correlation_id is None.

OMN-10755: Forward gate — events without correlation_id must produce a warning
log entry so that missing-correlation violations are observable in production.
The ledger still accepts and stores the event (never drops).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import UUID

import pytest

if TYPE_CHECKING:
    from omnibase_infra.nodes.node_ledger_write_effect.handlers.handler_ledger_append import (
        HandlerLedgerAppend,
    )
    from omnibase_infra.nodes.node_registration_reducer.models.model_payload_ledger_append import (
        ModelPayloadLedgerAppend,
    )


class TestLedgerAppendNullCorrelation:
    """Verify null-correlation_id warning gate in HandlerLedgerAppend."""

    @pytest.mark.asyncio
    async def test_null_correlation_id_emits_warning(
        self,
        ledger_append_handler: HandlerLedgerAppend,
        make_ledger_payload: ...,
        cleanup_event_ledger: list[UUID | None],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Appending with correlation_id=None must emit a WARNING and still succeed."""
        payload: ModelPayloadLedgerAppend = make_ledger_payload(correlation_id=None)

        with caplog.at_level(logging.WARNING, logger="omnibase_infra"):
            result = await ledger_append_handler.append(payload)

        # Event must be stored — ledger never drops events
        assert result.success is True
        assert result.duplicate is False
        assert result.ledger_entry_id is not None
        cleanup_event_ledger.append(result.ledger_entry_id)

        # Warning must appear in logs
        warning_messages = [
            r.message for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert any("correlation_id" in msg for msg in warning_messages), (
            f"Expected correlation_id warning, got: {warning_messages}"
        )

    @pytest.mark.asyncio
    async def test_null_correlation_id_auto_generates_for_db(
        self,
        ledger_append_handler: HandlerLedgerAppend,
        make_ledger_payload: ...,
        cleanup_event_ledger: list[UUID | None],
    ) -> None:
        """With correlation_id=None the handler auto-generates one for the DB write."""
        payload: ModelPayloadLedgerAppend = make_ledger_payload(correlation_id=None)
        result = await ledger_append_handler.append(payload)

        # Append succeeds and entry is stored
        assert result.success is True
        assert result.ledger_entry_id is not None
        cleanup_event_ledger.append(result.ledger_entry_id)
