# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for IntentEffectPostgresUpdate.

Tests the PostgreSQL UPDATE intent effect adapter which bridges
ModelPayloadPostgresUpdateRegistration payloads to raw asyncpg UPDATE
queries with optional monotonic heartbeat guard.

Related:
    - IntentEffectPostgresUpdate: Implementation under test
    - ModelPayloadPostgresUpdateRegistration: Intent payload model
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from omnibase_infra.errors import ContainerWiringError, RuntimeHostError
from omnibase_infra.nodes.reducers.models.model_payload_postgres_update_registration import (
    ModelPayloadPostgresUpdateRegistration,
)
from omnibase_infra.runtime.intent_effects.intent_effect_postgres_update import (
    _TIMESTAMP_COLUMNS,
    _UUID_COLUMNS,
    IntentEffectPostgresUpdate,
)

pytestmark = [pytest.mark.unit]


def _make_mock_pool() -> MagicMock:
    """Create a mock asyncpg pool with async context manager on acquire().

    Returns a MagicMock pool where pool.acquire() returns an async context
    manager yielding a mock connection with an async execute method.
    """
    mock_conn = MagicMock()
    mock_conn.execute = AsyncMock(return_value="UPDATE 1")

    pool = MagicMock()

    @asynccontextmanager
    async def _acquire() -> AsyncIterator[MagicMock]:
        yield mock_conn

    pool.acquire = _acquire
    pool._mock_conn = mock_conn  # Expose for assertions
    return pool


@pytest.mark.unit
class TestIntentEffectPostgresUpdateInit:
    """Tests for IntentEffectPostgresUpdate initialization."""

    def test_init_with_valid_pool(self) -> None:
        """Should initialize successfully with a valid pool."""
        mock_pool = MagicMock()

        effect = IntentEffectPostgresUpdate(pool=mock_pool)

        assert effect._pool is mock_pool

    def test_init_raises_on_none_pool(self) -> None:
        """Should raise ContainerWiringError when pool is None."""
        with pytest.raises(ContainerWiringError, match="asyncpg pool is required"):
            IntentEffectPostgresUpdate(pool=None)  # type: ignore[arg-type]


@pytest.mark.unit
class TestIntentEffectPostgresUpdateExecute:
    """Tests for IntentEffectPostgresUpdate.execute method."""

    @pytest.fixture
    def mock_pool(self) -> MagicMock:
        """Create a mock asyncpg pool with async acquire context manager."""
        return _make_mock_pool()

    @pytest.fixture
    def effect(self, mock_pool: MagicMock) -> IntentEffectPostgresUpdate:
        """Create an IntentEffectPostgresUpdate with mocked pool."""
        return IntentEffectPostgresUpdate(pool=mock_pool)

    @pytest.mark.asyncio
    async def test_execute_heartbeat_update(
        self, effect: IntentEffectPostgresUpdate, mock_pool: MagicMock
    ) -> None:
        """Verify SET clause, WHERE with monotonic guard, and params."""
        entity_id = uuid4()
        correlation_id = uuid4()
        heartbeat_ts = "2025-06-15T12:00:00+00:00"

        payload = ModelPayloadPostgresUpdateRegistration(
            correlation_id=correlation_id,
            entity_id=entity_id,
            domain="registration",
            updates={
                "last_heartbeat_at": heartbeat_ts,
                "liveness_deadline": "2025-06-15T12:05:00+00:00",
                "updated_at": "2025-06-15T12:00:00+00:00",
            },
        )

        await effect.execute(payload, correlation_id=correlation_id)

        conn = mock_pool._mock_conn
        conn.execute.assert_awaited_once()

        call_args = conn.execute.call_args
        sql = call_args.args[0]

        # Verify UPDATE target table
        assert '"registration_projections"' in sql
        assert sql.startswith("UPDATE")

        # Verify SET clause contains all update columns
        assert '"last_heartbeat_at"' in sql
        assert '"liveness_deadline"' in sql
        assert '"updated_at"' in sql

        # Verify WHERE clause has entity_id, domain, AND monotonic guard
        assert '"entity_id"' in sql
        assert '"domain"' in sql
        assert '"last_heartbeat_at" IS NULL OR "last_heartbeat_at" <' in sql, (
            "Monotonic guard should be present for heartbeat updates"
        )

        # Verify positional params: 3 SET values + entity_id + domain + heartbeat guard = 6
        positional_params = call_args.args[1:]
        assert len(positional_params) == 6

        # entity_id and domain should be in params
        assert entity_id in positional_params
        assert "registration" in positional_params

    @pytest.mark.asyncio
    async def test_execute_state_transition_update(
        self, effect: IntentEffectPostgresUpdate, mock_pool: MagicMock
    ) -> None:
        """Verify no monotonic guard for non-heartbeat updates."""
        entity_id = uuid4()
        correlation_id = uuid4()

        payload = ModelPayloadPostgresUpdateRegistration(
            correlation_id=correlation_id,
            entity_id=entity_id,
            domain="registration",
            updates={
                "current_state": "ack_received",
                "updated_at": "2025-06-15T12:00:00+00:00",
            },
        )

        await effect.execute(payload, correlation_id=correlation_id)

        conn = mock_pool._mock_conn
        conn.execute.assert_awaited_once()

        call_args = conn.execute.call_args
        sql = call_args.args[0]

        # Verify no monotonic guard in WHERE clause
        assert (
            "last_heartbeat_at" not in sql.split("WHERE")[1] if "WHERE" in sql else True
        ), "No monotonic guard should be present for state-transition updates"

        # Verify positional params: 2 SET values + entity_id + domain = 4
        positional_params = call_args.args[1:]
        assert len(positional_params) == 4

    @pytest.mark.asyncio
    async def test_execute_rejects_wrong_payload_type(
        self, effect: IntentEffectPostgresUpdate
    ) -> None:
        """Should raise RuntimeHostError for wrong payload type."""

        class FakePayload:
            pass

        with pytest.raises(
            RuntimeHostError,
            match="Expected ModelPayloadPostgresUpdateRegistration",
        ):
            await effect.execute(FakePayload())

    @pytest.mark.asyncio
    async def test_execute_rejects_empty_updates(
        self, effect: IntentEffectPostgresUpdate
    ) -> None:
        """Should raise RuntimeHostError when updates dict is empty."""
        entity_id = uuid4()
        correlation_id = uuid4()

        # Use model_construct to bypass validation since empty updates might be
        # rejected by model validators in the future.
        payload = ModelPayloadPostgresUpdateRegistration.model_construct(
            intent_type="postgres.update_registration",
            correlation_id=correlation_id,
            entity_id=entity_id,
            domain="registration",
            updates={},
        )

        with pytest.raises(RuntimeHostError, match="empty updates dict"):
            await effect.execute(payload, correlation_id=correlation_id)

    @pytest.mark.asyncio
    async def test_execute_wraps_db_error(
        self, effect: IntentEffectPostgresUpdate, mock_pool: MagicMock
    ) -> None:
        """Should wrap database errors as RuntimeHostError."""
        entity_id = uuid4()
        correlation_id = uuid4()

        payload = ModelPayloadPostgresUpdateRegistration(
            correlation_id=correlation_id,
            entity_id=entity_id,
            domain="registration",
            updates={"current_state": "active"},
        )

        conn = mock_pool._mock_conn
        conn.execute.side_effect = Exception("connection refused")

        with pytest.raises(
            RuntimeHostError, match="Failed to execute PostgreSQL UPDATE intent"
        ):
            await effect.execute(payload, correlation_id=correlation_id)

    @pytest.mark.asyncio
    async def test_execute_uses_payload_correlation_id_as_fallback(
        self, effect: IntentEffectPostgresUpdate, mock_pool: MagicMock
    ) -> None:
        """Should fall back to payload.correlation_id when none provided."""
        entity_id = uuid4()
        payload_correlation_id = uuid4()

        payload = ModelPayloadPostgresUpdateRegistration(
            correlation_id=payload_correlation_id,
            entity_id=entity_id,
            domain="registration",
            updates={"current_state": "active"},
        )

        # Call without explicit correlation_id
        await effect.execute(payload)

        conn = mock_pool._mock_conn
        conn.execute.assert_awaited_once()


@pytest.mark.unit
class TestNormalizeForAsyncpg:
    """Tests for IntentEffectPostgresUpdate._normalize_for_asyncpg."""

    def test_normalize_timestamps(self) -> None:
        """Should convert ISO datetime strings to datetime objects."""
        record = {
            "last_heartbeat_at": "2025-06-15T12:00:00+00:00",
            "updated_at": "2025-06-15T12:00:00",
        }

        result = IntentEffectPostgresUpdate._normalize_for_asyncpg(record)

        assert isinstance(result["last_heartbeat_at"], datetime)
        assert result["last_heartbeat_at"].tzinfo is not None

        # Naive datetime should get UTC attached
        assert isinstance(result["updated_at"], datetime)
        assert result["updated_at"].tzinfo == UTC

    def test_normalize_uuids(self) -> None:
        """Should convert string UUIDs to UUID objects."""
        test_uuid = uuid4()
        record = {
            "entity_id": str(test_uuid),
            "correlation_id": str(uuid4()),
        }

        result = IntentEffectPostgresUpdate._normalize_for_asyncpg(record)

        assert isinstance(result["entity_id"], UUID)
        assert result["entity_id"] == test_uuid
        assert isinstance(result["correlation_id"], UUID)

    def test_normalize_preserves_none_values(self) -> None:
        """Should preserve None values without conversion."""
        record = {
            "entity_id": None,
            "last_heartbeat_at": None,
            "current_state": None,
        }

        result = IntentEffectPostgresUpdate._normalize_for_asyncpg(record)

        assert result["entity_id"] is None
        assert result["last_heartbeat_at"] is None
        assert result["current_state"] is None

    def test_normalize_passes_through_non_special_columns(self) -> None:
        """Should pass through values for columns not in UUID or TIMESTAMP sets."""
        record = {
            "current_state": "active",
            "node_type": "effect",
            "last_applied_offset": 42,
        }

        result = IntentEffectPostgresUpdate._normalize_for_asyncpg(record)

        assert result["current_state"] == "active"
        assert result["node_type"] == "effect"
        assert result["last_applied_offset"] == 42

    def test_normalize_preserves_native_uuid(self) -> None:
        """Should pass through UUID objects without re-conversion."""
        native_uuid = uuid4()
        record = {"entity_id": native_uuid}

        result = IntentEffectPostgresUpdate._normalize_for_asyncpg(record)

        assert result["entity_id"] is native_uuid

    def test_normalize_preserves_native_datetime(self) -> None:
        """Should pass through datetime objects without re-conversion."""
        native_dt = datetime(2025, 6, 15, 12, 0, 0, tzinfo=UTC)
        record = {"last_heartbeat_at": native_dt}

        result = IntentEffectPostgresUpdate._normalize_for_asyncpg(record)

        assert result["last_heartbeat_at"] is native_dt


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

        unknown = _UUID_COLUMNS - schema_uuid_cols
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

        unknown = _TIMESTAMP_COLUMNS - schema_ts_cols
        assert not unknown, (
            f"_TIMESTAMP_COLUMNS contains columns not in schema: {unknown}. "
            f"Schema TIMESTAMPTZ columns: {schema_ts_cols}"
        )

    def test_all_schema_uuid_columns_covered(self) -> None:
        """All UUID columns in schema should be in _UUID_COLUMNS."""
        schema_path = (
            Path(__file__).parent.parent.parent.parent
            / "src"
            / "omnibase_infra"
            / "schemas"
            / "schema_registration_projection.sql"
        )
        sql = schema_path.read_text()
        schema_uuid_cols = self._extract_uuid_columns_from_sql(sql)

        uncovered = schema_uuid_cols - _UUID_COLUMNS
        assert not uncovered, (
            f"Schema has UUID columns not in _UUID_COLUMNS: {uncovered}. "
            f"Add these to intent_effect_postgres_update._UUID_COLUMNS."
        )

    def test_all_schema_timestamptz_columns_covered(self) -> None:
        """All TIMESTAMPTZ columns in schema should be in _TIMESTAMP_COLUMNS."""
        schema_path = (
            Path(__file__).parent.parent.parent.parent
            / "src"
            / "omnibase_infra"
            / "schemas"
            / "schema_registration_projection.sql"
        )
        sql = schema_path.read_text()
        schema_ts_cols = self._extract_timestamptz_columns_from_sql(sql)

        uncovered = schema_ts_cols - _TIMESTAMP_COLUMNS
        assert not uncovered, (
            f"Schema has TIMESTAMPTZ columns not in _TIMESTAMP_COLUMNS: "
            f"{uncovered}. Add these to "
            f"intent_effect_postgres_update._TIMESTAMP_COLUMNS."
        )
