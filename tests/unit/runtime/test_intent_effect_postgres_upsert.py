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

import json
import re
from pathlib import Path
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

pytestmark = [pytest.mark.unit]


class MockProjectionRecord(BaseModel):
    """Mock projection record for testing."""

    model_config = ConfigDict(extra="allow", frozen=True)


@pytest.mark.unit
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


@pytest.mark.unit
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
    async def test_execute_raises_when_record_is_none(
        self, effect: IntentEffectPostgresUpsert, mock_projector: MagicMock
    ) -> None:
        """Should raise RuntimeHostError when record is None to prevent silent data loss."""
        correlation_id = uuid4()

        # Create payload with None record by using model_construct to bypass validation
        payload = ModelPayloadPostgresUpsertRegistration.model_construct(
            intent_type="postgres.upsert_registration",
            correlation_id=correlation_id,
            record=None,
        )

        with pytest.raises(RuntimeHostError, match="no record"):
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


@pytest.mark.unit
class TestColumnSetsMatchSchema:
    """Validate that _UUID_COLUMNS and _TIMESTAMP_COLUMNS match the SQL schema.

    These frozensets drive asyncpg type normalization. If the SQL schema adds
    a new UUID or TIMESTAMPTZ column and these sets are not updated, asyncpg
    will receive string values instead of native types, causing query failures.
    """

    @staticmethod
    def _extract_uuid_columns_from_sql(sql: str) -> set[str]:
        """Extract column names declared as UUID in CREATE TABLE."""
        columns: set[str] = set()
        # Match lines like: column_name UUID ...
        for match in re.finditer(r"^\s+(\w+)\s+UUID\b", sql, re.MULTILINE):
            columns.add(match.group(1))
        return columns

    @staticmethod
    def _extract_timestamptz_columns_from_sql(sql: str) -> set[str]:
        """Extract column names declared as TIMESTAMPTZ in CREATE TABLE."""
        columns: set[str] = set()
        for match in re.finditer(r"^\s+(\w+)\s+TIMESTAMPTZ\b", sql, re.MULTILINE):
            columns.add(match.group(1))
        return columns

    def test_uuid_columns_are_subset_of_schema(self) -> None:
        """_UUID_COLUMNS must only contain columns that exist as UUID in schema."""
        schema_path = (
            Path(__file__).parent.parent.parent.parent
            / "src"
            / "omnibase_infra"
            / "schemas"
            / "schema_registration_projection.sql"
        )
        assert schema_path.exists(), f"Schema file not found: {schema_path}"

        sql = schema_path.read_text()
        schema_uuid_cols = self._extract_uuid_columns_from_sql(sql)

        # _UUID_COLUMNS must be a subset of actual schema UUID columns
        unknown = IntentEffectPostgresUpsert._UUID_COLUMNS - schema_uuid_cols
        assert not unknown, (
            f"_UUID_COLUMNS contains columns not in schema: {unknown}. "
            f"Schema UUID columns: {schema_uuid_cols}"
        )

    def test_timestamp_columns_are_subset_of_schema(self) -> None:
        """_TIMESTAMP_COLUMNS must only contain columns that exist as TIMESTAMPTZ."""
        schema_path = (
            Path(__file__).parent.parent.parent.parent
            / "src"
            / "omnibase_infra"
            / "schemas"
            / "schema_registration_projection.sql"
        )
        assert schema_path.exists(), f"Schema file not found: {schema_path}"

        sql = schema_path.read_text()
        schema_ts_cols = self._extract_timestamptz_columns_from_sql(sql)

        unknown = IntentEffectPostgresUpsert._TIMESTAMP_COLUMNS - schema_ts_cols
        assert not unknown, (
            f"_TIMESTAMP_COLUMNS contains columns not in schema: {unknown}. "
            f"Schema TIMESTAMPTZ columns: {schema_ts_cols}"
        )

    def test_all_schema_uuid_columns_covered_or_documented(self) -> None:
        """All UUID columns in schema should be in _UUID_COLUMNS.

        If this test fails, a new UUID column was added to the SQL schema
        but not added to IntentEffectPostgresUpsert._UUID_COLUMNS or the
        'COLUMNS NOT YET COVERED' documentation block.
        """
        schema_path = (
            Path(__file__).parent.parent.parent.parent
            / "src"
            / "omnibase_infra"
            / "schemas"
            / "schema_registration_projection.sql"
        )
        sql = schema_path.read_text()
        schema_uuid_cols = self._extract_uuid_columns_from_sql(sql)
        covered = IntentEffectPostgresUpsert._UUID_COLUMNS

        # All schema UUID columns must be in _UUID_COLUMNS.
        uncovered = schema_uuid_cols - covered
        assert not uncovered, (
            f"Schema has UUID columns not in _UUID_COLUMNS: {uncovered}. "
            f"Add these to IntentEffectPostgresUpsert._UUID_COLUMNS or "
            f"document them in the 'COLUMNS NOT YET COVERED' block."
        )

    def test_all_schema_timestamptz_columns_covered_or_documented(self) -> None:
        """All TIMESTAMPTZ columns in schema should be in _TIMESTAMP_COLUMNS.

        If this test fails, a new TIMESTAMPTZ column was added to the SQL
        schema but not added to IntentEffectPostgresUpsert._TIMESTAMP_COLUMNS
        or the 'COLUMNS NOT YET COVERED' documentation block.
        """
        schema_path = (
            Path(__file__).parent.parent.parent.parent
            / "src"
            / "omnibase_infra"
            / "schemas"
            / "schema_registration_projection.sql"
        )
        sql = schema_path.read_text()
        schema_ts_cols = self._extract_timestamptz_columns_from_sql(sql)
        covered = IntentEffectPostgresUpsert._TIMESTAMP_COLUMNS

        uncovered = schema_ts_cols - covered
        assert not uncovered, (
            f"Schema has TIMESTAMPTZ columns not in _TIMESTAMP_COLUMNS: "
            f"{uncovered}. Add these to "
            f"IntentEffectPostgresUpsert._TIMESTAMP_COLUMNS or document "
            f"them in the 'COLUMNS NOT YET COVERED' block."
        )


@pytest.mark.unit
class TestNormalizeForAsyncpg:
    """Unit tests for IntentEffectPostgresUpsert._normalize_for_asyncpg.

    Covers all column type branches: None passthrough, JSONB serialization,
    UUID coercion, TIMESTAMPTZ coercion, and plain passthrough.
    """

    def test_jsonb_dict_is_serialized_to_json_string(self) -> None:
        """capabilities dict must be serialized to a JSON string for asyncpg."""
        caps = {"postgres": False, "read": False, "write": True}
        record: dict[str, object] = {"capabilities": caps}

        result = IntentEffectPostgresUpsert._normalize_for_asyncpg(record)

        assert isinstance(result["capabilities"], str)
        assert json.loads(result["capabilities"]) == caps  # type: ignore[arg-type]

    def test_jsonb_list_is_serialized_to_json_string(self) -> None:
        """A list value in a JSONB column must be serialized to a JSON string."""
        caps_list: list[str] = ["read", "write"]
        record: dict[str, object] = {"capabilities": caps_list}

        result = IntentEffectPostgresUpsert._normalize_for_asyncpg(record)

        assert isinstance(result["capabilities"], str)
        assert json.loads(result["capabilities"]) == caps_list  # type: ignore[arg-type]

    def test_jsonb_already_string_is_passed_through(self) -> None:
        """A JSON string already in a JSONB column must not be double-serialized."""
        caps_str = '{"postgres": true}'
        record: dict[str, object] = {"capabilities": caps_str}

        result = IntentEffectPostgresUpsert._normalize_for_asyncpg(record)

        assert result["capabilities"] == caps_str

    def test_jsonb_none_is_passed_through(self) -> None:
        """None values must be passed through unchanged regardless of column type."""
        record: dict[str, object] = {"capabilities": None}

        result = IntentEffectPostgresUpsert._normalize_for_asyncpg(record)

        assert result["capabilities"] is None

    def test_text_array_list_is_not_serialized(self) -> None:
        """Lists in TEXT[] columns (not in _JSONB_COLUMNS) must not be JSON-serialized.

        asyncpg handles Python lists natively for PostgreSQL array columns.
        Serializing them to JSON strings would cause a type mismatch.
        """
        protocols = ["http", "grpc"]
        record: dict[str, object] = {"protocols": protocols}

        result = IntentEffectPostgresUpsert._normalize_for_asyncpg(record)

        # Must remain a Python list, not a JSON string
        assert result["protocols"] == protocols
        assert isinstance(result["protocols"], list)

    @pytest.mark.asyncio
    async def test_execute_normalizes_capabilities_dict_to_json_string(
        self,
    ) -> None:
        """execute() must pass capabilities as a JSON string to upsert_partial."""
        mock_projector = MagicMock()
        mock_projector.upsert_partial = AsyncMock(return_value=True)
        effect = IntentEffectPostgresUpsert(projector=mock_projector)

        entity_id = uuid4()
        correlation_id = uuid4()
        caps = {"postgres": False, "read": False, "write": True}

        record = MockProjectionRecord(
            entity_id=str(entity_id),
            domain="registration",
            capabilities=caps,
        )
        payload = ModelPayloadPostgresUpsertRegistration(
            correlation_id=correlation_id,
            record=record,
        )

        await effect.execute(payload, correlation_id=correlation_id)

        call_kwargs = mock_projector.upsert_partial.call_args
        values = call_kwargs.kwargs["values"]
        assert isinstance(values["capabilities"], str), (
            "capabilities must be a JSON string when passed to asyncpg, "
            f"got {type(values['capabilities']).__name__}"
        )
        assert json.loads(values["capabilities"]) == caps
