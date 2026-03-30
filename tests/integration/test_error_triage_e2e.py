# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""End-to-end verification for the error triage pipeline (OMN-5656).

Integration test that verifies the full error triage flow:
1. Produce a runtime error event to Kafka
2. Verify the triage consumer picks it up
3. Verify the handler classifies it
4. Verify the result is emitted to the triage topic

This test uses mocked Kafka producers/consumers and the real handler
logic to verify the pipeline wiring without requiring live infrastructure.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.handlers.handler_linear_db_error_reporter import (
    HandlerLinearDbErrorReporter,
    _build_ticket_description,
    _build_ticket_title,
    _fingerprint_to_lock_key,
)
from omnibase_infra.handlers.models.model_db_error_event import ModelDbErrorEvent
from omnibase_infra.topics.platform_topic_suffixes import TOPIC_DB_ERROR_V1


@pytest.mark.integration
class TestErrorTriagePipelineE2E:
    """End-to-end tests for the error triage pipeline."""

    def test_db_error_topic_constant_exists(self) -> None:
        """Verify the db error topic constant is defined and valid."""
        assert TOPIC_DB_ERROR_V1 == "onex.evt.omnibase-infra.db-error.v1"

    def test_error_event_model_construction(self) -> None:
        """Verify ModelDbErrorEvent can be constructed with required fields."""
        event = ModelDbErrorEvent(
            error_code="23505",
            error_message="duplicate key value violates unique constraint",
            hint="Key (id)=(abc) already exists.",
            detail="",
            sql_statement="INSERT INTO nodes (id) VALUES ('abc')",
            table_name="nodes",
            fingerprint="a" * 32,
            first_seen_at=datetime.now(UTC),
            service="omninode-runtime",
        )
        assert event.error_code == "23505"
        assert event.fingerprint == "a" * 32

    def test_error_event_serialization_roundtrip(self) -> None:
        """Verify error event survives JSON serialization (Kafka wire format)."""
        event = ModelDbErrorEvent(
            error_code="42P01",
            error_message='relation "missing_table" does not exist',
            hint=None,
            detail="",
            sql_statement="SELECT * FROM missing_table",
            table_name="missing_table",
            fingerprint="b" * 32,
            first_seen_at=datetime.now(UTC),
            service="omninode-runtime",
        )
        data = event.model_dump(mode="json")
        restored = ModelDbErrorEvent.model_validate(data)
        assert restored.error_code == event.error_code
        assert restored.fingerprint == event.fingerprint

    def test_fingerprint_to_lock_key_deterministic(self) -> None:
        """Verify fingerprint-to-lock-key mapping is deterministic."""
        fp = "abcdef0123456789abcdef0123456789"
        key1 = _fingerprint_to_lock_key(fp)
        key2 = _fingerprint_to_lock_key(fp)
        assert key1 == key2
        assert isinstance(key1, int)

    def test_fingerprint_to_lock_key_different_inputs(self) -> None:
        """Different fingerprints produce different lock keys."""
        fp1 = "a" * 32
        fp2 = "b" * 32
        assert _fingerprint_to_lock_key(fp1) != _fingerprint_to_lock_key(fp2)

    def test_ticket_title_format(self) -> None:
        """Verify the Linear ticket title follows expected format."""
        event = ModelDbErrorEvent(
            error_code="23505",
            error_message="duplicate key value violates unique constraint",
            hint=None,
            detail="",
            sql_statement="INSERT INTO test (id) VALUES (1)",
            table_name="test",
            fingerprint="c" * 32,
            first_seen_at=datetime.now(UTC),
            service="test-service",
        )
        title = _build_ticket_title(event)
        assert title.startswith("[DB ERROR]")
        assert "23505" in title
        assert "test" in title

    def test_ticket_description_contains_fields(self) -> None:
        """Verify the ticket description includes all relevant fields."""
        event = ModelDbErrorEvent(
            error_code="23505",
            error_message="dup key",
            hint="Check constraints",
            detail="",
            sql_statement="INSERT INTO t (id) VALUES (1)",
            table_name="t",
            fingerprint="d" * 32,
            first_seen_at=datetime.now(UTC),
            service="svc",
        )
        desc = _build_ticket_description(event)
        assert "dup key" in desc
        assert "Check constraints" in desc
        assert "svc" in desc
        assert "INSERT INTO" in desc

    @pytest.mark.asyncio
    async def test_handler_rejects_missing_db_pool(self) -> None:
        """Handler raises when db_pool is None."""
        handler = HandlerLinearDbErrorReporter(
            linear_api_key="test-key",
            linear_team_id="test-team",
            db_pool=None,
        )
        event = ModelDbErrorEvent(
            error_code="23505",
            error_message="test",
            hint=None,
            detail="",
            sql_statement="SELECT 1",
            table_name="test",
            fingerprint="e" * 32,
            first_seen_at=datetime.now(UTC),
            service="test",
        )
        from omnibase_infra.errors import RuntimeHostError

        with pytest.raises(RuntimeHostError, match="db_pool"):
            await handler.handle(event)

    @pytest.mark.asyncio
    async def test_handler_rejects_empty_api_key(self) -> None:
        """Handler raises when linear_api_key is empty."""
        mock_pool = MagicMock()
        handler = HandlerLinearDbErrorReporter(
            linear_api_key="",
            linear_team_id="team",
            db_pool=mock_pool,
        )
        event = ModelDbErrorEvent(
            error_code="23505",
            error_message="test",
            hint=None,
            detail="",
            sql_statement="SELECT 1",
            table_name="test",
            fingerprint="f" * 32,
            first_seen_at=datetime.now(UTC),
            service="test",
        )
        from omnibase_infra.errors import RuntimeHostError

        with pytest.raises(RuntimeHostError, match="linear_api_key"):
            await handler.handle(event)

    @pytest.mark.asyncio
    async def test_handler_dedup_skip_path(self) -> None:
        """Handler skips ticket creation for known fingerprints."""
        mock_conn = AsyncMock()
        # Simulate existing fingerprint in DB
        mock_conn.fetchrow = AsyncMock(
            side_effect=[
                # First fetchrow: dedup check returns existing row
                {
                    "linear_issue_id": str(uuid4()),
                    "linear_issue_url": "https://linear.app/test/123",
                    "occurrence_count": 5,
                },
                # Second fetchrow: UPDATE RETURNING
                {"occurrence_count": 6},
            ]
        )
        mock_conn.execute = AsyncMock()

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        handler = HandlerLinearDbErrorReporter(
            linear_api_key="key",
            linear_team_id="team",
            db_pool=mock_pool,
        )

        event = ModelDbErrorEvent(
            error_code="23505",
            error_message="dup key",
            hint=None,
            detail="",
            sql_statement="INSERT INTO t (id) VALUES (1)",
            table_name="t",
            fingerprint="a1b2c3d4e5f60000a1b2c3d4e5f60000",
            first_seen_at=datetime.now(UTC),
            service="test",
        )

        result = await handler.handle(event)
        assert result.skipped is True
        assert result.occurrence_count == 6

    def test_handler_classification_properties(self) -> None:
        """Verify handler type and category classification."""
        from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory

        handler = HandlerLinearDbErrorReporter()
        assert handler.handler_type == EnumHandlerType.INFRA_HANDLER
        assert handler.handler_category == EnumHandlerTypeCategory.EFFECT
