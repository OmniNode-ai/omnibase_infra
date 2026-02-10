# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for IntentEffectPostgresUpsert.

Tests the PostgreSQL upsert intent effect adapter which bridges
ModelPayloadPostgresUpsertRegistration payloads to ProjectorShell operations.

Related:
    - OMN-2050: Wire MessageDispatchEngine as single consumer path
    - IntentEffectPostgresUpsert: Implementation under test
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from pydantic import BaseModel, ConfigDict

from omnibase_infra.errors import ContainerWiringError, RuntimeHostError
from omnibase_infra.nodes.reducers.models.model_payload_postgres_upsert_registration import (
    ModelPayloadPostgresUpsertRegistration,
)
from omnibase_infra.runtime.intent_effects.intent_effect_postgres_upsert import (
    IntentEffectPostgresUpsert,
)


class MockProjectionRecord(BaseModel):
    """Mock projection record for testing."""

    model_config = ConfigDict(extra="allow", frozen=True)


class TestIntentEffectPostgresUpsertInit:
    """Tests for IntentEffectPostgresUpsert initialization."""

    def test_init_with_valid_projector(self) -> None:
        """Should initialize successfully with a valid projector."""
        mock_projector = MagicMock()

        effect = IntentEffectPostgresUpsert(projector=mock_projector)

        assert effect._projector is mock_projector

    def test_init_raises_on_none_projector(self) -> None:
        """Should raise ContainerWiringError when projector is None."""
        with pytest.raises(ContainerWiringError, match="ProjectorShell is required"):
            IntentEffectPostgresUpsert(projector=None)  # type: ignore[arg-type]


class TestIntentEffectPostgresUpsertExecute:
    """Tests for IntentEffectPostgresUpsert.execute method."""

    @pytest.fixture
    def mock_projector(self) -> MagicMock:
        """Create a mock ProjectorShell with async upsert_partial."""
        projector = MagicMock()
        projector.upsert_partial = AsyncMock(return_value=True)
        return projector

    @pytest.fixture
    def effect(self, mock_projector: MagicMock) -> IntentEffectPostgresUpsert:
        """Create an IntentEffectPostgresUpsert with mocked projector."""
        return IntentEffectPostgresUpsert(projector=mock_projector)

    @pytest.mark.asyncio
    async def test_execute_upserts_record(
        self, effect: IntentEffectPostgresUpsert, mock_projector: MagicMock
    ) -> None:
        """Should call projector.upsert_partial with record data."""
        entity_id = uuid4()
        correlation_id = uuid4()

        record = MockProjectionRecord(
            entity_id=str(entity_id),
            domain="registration",
            current_state="pending_registration",
            node_type="effect",
        )

        payload = ModelPayloadPostgresUpsertRegistration(
            correlation_id=correlation_id,
            record=record,
        )

        await effect.execute(payload, correlation_id=correlation_id)

        mock_projector.upsert_partial.assert_awaited_once()
        call_kwargs = mock_projector.upsert_partial.call_args
        assert call_kwargs.kwargs["correlation_id"] == correlation_id
        assert call_kwargs.kwargs["conflict_columns"] == ["entity_id", "domain"]

    @pytest.mark.asyncio
    async def test_execute_uses_payload_correlation_id_as_fallback(
        self, effect: IntentEffectPostgresUpsert, mock_projector: MagicMock
    ) -> None:
        """Should fall back to payload.correlation_id when none provided."""
        entity_id = uuid4()
        payload_correlation_id = uuid4()

        record = MockProjectionRecord(
            entity_id=str(entity_id),
            domain="registration",
        )

        payload = ModelPayloadPostgresUpsertRegistration(
            correlation_id=payload_correlation_id,
            record=record,
        )

        await effect.execute(payload)

        mock_projector.upsert_partial.assert_awaited_once()
        call_kwargs = mock_projector.upsert_partial.call_args
        assert call_kwargs.kwargs["correlation_id"] == payload_correlation_id

    @pytest.mark.asyncio
    async def test_execute_skips_when_record_is_none(
        self, effect: IntentEffectPostgresUpsert, mock_projector: MagicMock
    ) -> None:
        """Should skip upsert when record is None."""
        correlation_id = uuid4()

        # Create payload with None record by using model_construct to bypass validation
        payload = ModelPayloadPostgresUpsertRegistration.model_construct(
            intent_type="postgres.upsert_registration",
            correlation_id=correlation_id,
            record=None,
        )

        await effect.execute(payload, correlation_id=correlation_id)

        mock_projector.upsert_partial.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_execute_raises_when_entity_id_missing(
        self, effect: IntentEffectPostgresUpsert, mock_projector: MagicMock
    ) -> None:
        """Should raise RuntimeHostError when record has no entity_id."""
        correlation_id = uuid4()

        # Record without entity_id
        record = MockProjectionRecord(
            domain="registration",
            current_state="pending_registration",
        )

        payload = ModelPayloadPostgresUpsertRegistration(
            correlation_id=correlation_id,
            record=record,
        )

        with pytest.raises(RuntimeHostError, match="missing required entity_id"):
            await effect.execute(payload, correlation_id=correlation_id)

        mock_projector.upsert_partial.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_execute_raises_runtime_host_error_on_failure(
        self, effect: IntentEffectPostgresUpsert, mock_projector: MagicMock
    ) -> None:
        """Should raise RuntimeHostError when upsert fails."""
        entity_id = uuid4()
        correlation_id = uuid4()

        record = MockProjectionRecord(
            entity_id=str(entity_id),
            domain="registration",
        )

        payload = ModelPayloadPostgresUpsertRegistration(
            correlation_id=correlation_id,
            record=record,
        )

        mock_projector.upsert_partial.side_effect = Exception("DB connection failed")

        with pytest.raises(RuntimeHostError, match="Failed to execute PostgreSQL"):
            await effect.execute(payload, correlation_id=correlation_id)

    @pytest.mark.asyncio
    async def test_execute_extracts_values_from_record(
        self, effect: IntentEffectPostgresUpsert, mock_projector: MagicMock
    ) -> None:
        """Should pass all record fields as values to upsert_partial."""
        entity_id = uuid4()
        correlation_id = uuid4()

        record = MockProjectionRecord(
            entity_id=str(entity_id),
            domain="registration",
            current_state="pending_registration",
            node_type="effect",
            node_version="1.0.0",
        )

        payload = ModelPayloadPostgresUpsertRegistration(
            correlation_id=correlation_id,
            record=record,
        )

        await effect.execute(payload, correlation_id=correlation_id)

        call_kwargs = mock_projector.upsert_partial.call_args
        values = call_kwargs.kwargs["values"]
        assert values["entity_id"] == entity_id
        assert values["domain"] == "registration"
        assert values["current_state"] == "pending_registration"
        assert values["node_type"] == "effect"
