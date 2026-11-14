# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the runtime_entity_id column on registration_projections.

OMN-24495 added a nullable ``runtime_entity_id UUID`` column (the runtime
that vouches for a node). Several layers enumerate the projection's columns
independently and must stay in sync: the SQL schema, forward migration 117,
the projector contract YAML, the intent effect column sets, and the
projection model. These tests pin the column in each layer and cover its
value round trips:

- S-C (schema sync): the column exists with the right type in every listing.
- V9 (upsert round trip): str and UUID forms reach ProjectorShell
  upsert_partial as native UUID objects; None stays None.
- Update adapter: the asyncpg normalizer converts/preserves the column and
  the _ALLOWED_COLUMNS allowlist accepts it while unknown columns are still
  refused.
- M1 (reader mapping, old rows): rows written before migration 117 (no
  runtime_entity_id key) read back as None.

Related:
    - OMN-24495: Add runtime_entity_id to registration projections
    - IntentEffectPostgresUpsert: Upsert adapter under test
    - IntentEffectPostgresUpdate: Update adapter under test
    - ProjectionReaderRegistration: Reader mapping under test
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import asyncpg
import pytest
import yaml
from pydantic import BaseModel, ConfigDict

from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.models.projection import ModelRegistrationProjection
from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)
from omnibase_infra.models.registration.model_payload_postgres_update_registration import (
    ModelPayloadPostgresUpdateRegistration,
)
from omnibase_infra.models.registration.model_payload_postgres_upsert_registration import (
    ModelPayloadPostgresUpsertRegistration,
)
from omnibase_infra.projectors.projection_reader_registration import (
    ProjectionReaderRegistration,
)
from omnibase_infra.runtime.intent_effects.intent_effect_postgres_update import (
    _UUID_COLUMNS as UPDATE_EFFECT_UUID_COLUMNS,
)
from omnibase_infra.runtime.intent_effects.intent_effect_postgres_update import (
    IntentEffectPostgresUpdate,
)
from omnibase_infra.runtime.intent_effects.intent_effect_postgres_upsert import (
    IntentEffectPostgresUpsert,
)

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).parent.parent.parent.parent

SCHEMA_FILE = (
    REPO_ROOT
    / "src"
    / "omnibase_infra"
    / "schemas"
    / "schema_registration_projection.sql"
)
MIGRATION_117_FILE = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "117_add_registration_runtime_entity_id.sql"
)
CONTRACT_FILE = (
    REPO_ROOT
    / "src"
    / "omnibase_infra"
    / "projectors"
    / "contracts"
    / "registration_projector.yaml"
)


class MockProjectionRecord(BaseModel):
    """Mock projection record for testing."""

    model_config = ConfigDict(extra="allow", frozen=True)


def _create_table_block(schema_sql: str) -> str:
    """Extract the CREATE TABLE block for registration_projections."""
    start = schema_sql.index(
        "CREATE TABLE IF NOT EXISTS public.registration_projections"
    )
    end = schema_sql.index("\n);", start)
    return schema_sql[start:end]


def _extract_uuid_columns_from_sql(sql: str) -> set[str]:
    """Extract column names declared as UUID in CREATE TABLE DDL.

    Same DDL-parsing approach as TestColumnSetsMatchSchema in the
    neighbouring adapter tests.
    """
    return set(re.findall(r"^\s+(\w+)\s+UUID\b", sql, re.MULTILINE))


def _make_upsert_payload(
    runtime_entity_id: object,
) -> ModelPayloadPostgresUpsertRegistration:
    """Build an upsert payload whose record carries runtime_entity_id."""
    record = MockProjectionRecord(
        entity_id=str(uuid4()),
        domain="registration",
        current_state="pending_registration",
        node_type="effect",
        runtime_entity_id=runtime_entity_id,
    )
    return ModelPayloadPostgresUpsertRegistration(
        correlation_id=uuid4(),
        record=record,
    )


def _make_mock_pool() -> MagicMock:
    """Create a mock asyncpg pool with async context manager on acquire().

    Same shape as _make_mock_pool in test_intent_effect_postgres_update.py.
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


def _make_update_payload(
    entity_id: UUID,
    correlation_id: UUID,
    updates_value: dict[str, object],
) -> ModelPayloadPostgresUpdateRegistration:
    """Build an update payload whose updates model dumps to updates_value."""
    updates = MagicMock()
    updates.model_dump.return_value = updates_value

    return ModelPayloadPostgresUpdateRegistration.model_construct(
        intent_type="postgres.update_registration",
        correlation_id=correlation_id,
        entity_id=entity_id,
        domain="registration",
        updates=updates,
    )


def create_mock_row(
    entity_id: UUID | None = None,
    state: EnumRegistrationState = EnumRegistrationState.ACTIVE,
    domain: str = "registration",
    ack_deadline: datetime | None = None,
    liveness_deadline: datetime | None = None,
    ack_timeout_emitted_at: datetime | None = None,
    liveness_timeout_emitted_at: datetime | None = None,
    contract_type: str | None = "effect",
    intent_types: list[str] | None = None,
    protocols: list[str] | None = None,
    capability_tags: list[str] | None = None,
    contract_version: str | None = "1.0.0",
) -> dict:
    """Create a mock database row with sensible defaults (pre-117 shape).

    Mirrors create_mock_row in test_projection_reader_registration.py. The
    returned dict deliberately carries no runtime_entity_id key so tests can
    exercise rows written before migration 117.
    """
    now = datetime.now(UTC)
    node_id = entity_id or uuid4()
    capabilities = ModelNodeCapabilities(postgres=True, read=True)

    return {
        "entity_id": node_id,
        "domain": domain,
        "current_state": state.value,
        "node_type": "effect",
        "node_version": "1.0.0",
        "capabilities": capabilities.model_dump_json(),
        # Capability fields (OMN-1134)
        "contract_type": contract_type,
        "intent_types": intent_types or [],
        "protocols": protocols or [],
        "capability_tags": capability_tags or [],
        "contract_version": contract_version,
        # Timeout fields
        "ack_deadline": ack_deadline,
        "liveness_deadline": liveness_deadline,
        "ack_timeout_emitted_at": ack_timeout_emitted_at,
        "liveness_timeout_emitted_at": liveness_timeout_emitted_at,
        "last_heartbeat_at": None,
        "last_applied_event_id": uuid4(),
        "last_applied_offset": 100,
        "last_applied_sequence": None,
        "last_applied_partition": "0",
        "registered_at": now,
        "updated_at": now,
        "correlation_id": uuid4(),
    }


@pytest.mark.unit
class TestSchemaSyncRuntimeEntityId:
    """S-C: runtime_entity_id must appear in every column listing.

    Each layer that enumerates registration_projections columns
    independently must list the column, otherwise writes and reads
    silently drop the runtime parent.
    """

    def test_create_table_declares_runtime_entity_id_nullable_uuid(self) -> None:
        """CREATE TABLE must declare runtime_entity_id as UUID, nullable."""
        schema_sql = SCHEMA_FILE.read_text()
        block = _create_table_block(schema_sql)

        match = re.search(r"^\s+runtime_entity_id\s+UUID\b", block, re.MULTILINE)
        assert match is not None, (
            f"runtime_entity_id UUID not found in CREATE TABLE:\n{block}"
        )
        assert "NOT NULL" not in match.group(0), (
            "runtime_entity_id must stay nullable (NULL = no runtime parent)"
        )

    def test_migration_117_adds_column_and_partial_index(self) -> None:
        """Migration 117 must add the column and its partial index idempotently."""
        assert MIGRATION_117_FILE.exists(), (
            f"Migration file not found: {MIGRATION_117_FILE}"
        )
        migration_sql = MIGRATION_117_FILE.read_text()

        assert re.search(
            r"ADD COLUMN IF NOT EXISTS\s+runtime_entity_id\s+UUID\b", migration_sql
        ), "Migration 117 must add runtime_entity_id UUID idempotently"
        assert (
            "CREATE INDEX IF NOT EXISTS idx_registration_runtime_entity_id"
            in migration_sql
        ), "Migration 117 must create the runtime parent index idempotently"
        assert "WHERE runtime_entity_id IS NOT NULL" in migration_sql, (
            "The runtime parent index must be partial (non-null rows only)"
        )

    def test_contract_yaml_lists_runtime_entity_id_as_uuid(self) -> None:
        """The projector contract must carry the column with type UUID."""
        assert CONTRACT_FILE.exists(), f"Contract file not found: {CONTRACT_FILE}"
        contract = yaml.safe_load(CONTRACT_FILE.read_text())

        columns = contract["projection_schema"]["columns"]
        matching = [
            column for column in columns if column["name"] == "runtime_entity_id"
        ]

        assert len(matching) == 1, (
            f"Expected exactly one runtime_entity_id column, got: {matching}"
        )
        assert matching[0]["type"] == "UUID"

    def test_update_effect_allowed_columns_include_runtime_entity_id(self) -> None:
        """_ALLOWED_COLUMNS must accept the column for UPDATE SET clauses."""
        assert "runtime_entity_id" in IntentEffectPostgresUpdate._ALLOWED_COLUMNS

    def test_update_effect_module_uuid_columns_include_runtime_entity_id(self) -> None:
        """The update module's _UUID_COLUMNS must cover the column."""
        assert "runtime_entity_id" in UPDATE_EFFECT_UUID_COLUMNS

    def test_upsert_effect_uuid_columns_include_runtime_entity_id(self) -> None:
        """The upsert effect's _UUID_COLUMNS must cover the column."""
        assert "runtime_entity_id" in IntentEffectPostgresUpsert._UUID_COLUMNS

    def test_projection_model_field_defaults_to_none(self) -> None:
        """ModelRegistrationProjection must default runtime_entity_id to None."""
        assert "runtime_entity_id" in ModelRegistrationProjection.model_fields

        field = ModelRegistrationProjection.model_fields["runtime_entity_id"]
        assert field.default is None
        assert not field.is_required()

    def test_all_schema_uuid_columns_in_both_uuid_column_sets(self) -> None:
        """Every schema UUID column must be covered by BOTH _UUID_COLUMNS sets."""
        schema_uuid_cols = _extract_uuid_columns_from_sql(SCHEMA_FILE.read_text())
        # Guard for this ticket: the new column must actually be parsed.
        assert "runtime_entity_id" in schema_uuid_cols

        missing_update = schema_uuid_cols - UPDATE_EFFECT_UUID_COLUMNS
        assert not missing_update, (
            f"Schema UUID columns missing from update _UUID_COLUMNS: {missing_update}"
        )

        missing_upsert = schema_uuid_cols - IntentEffectPostgresUpsert._UUID_COLUMNS
        assert not missing_upsert, (
            f"Schema UUID columns missing from upsert _UUID_COLUMNS: {missing_upsert}"
        )


@pytest.mark.unit
class TestRuntimeEntityIdRoundTripUpsert:
    """V9: runtime_entity_id must reach upsert_partial as a native UUID.

    Handlers emit JSON-serializable records (string UUIDs); asyncpg requires
    native UUID objects for UUID columns. The conversion is driven by
    IntentEffectPostgresUpsert._UUID_COLUMNS.
    """

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
    async def test_execute_converts_uuid_instance_to_uuid(
        self, effect: IntentEffectPostgresUpsert, mock_projector: MagicMock
    ) -> None:
        """A UUID-valued runtime_entity_id must arrive as the same UUID."""
        runtime_entity_id = uuid4()
        correlation_id = uuid4()

        await effect.execute(
            _make_upsert_payload(runtime_entity_id), correlation_id=correlation_id
        )

        mock_projector.upsert_partial.assert_awaited_once()
        values = mock_projector.upsert_partial.call_args.kwargs["values"]
        assert isinstance(values["runtime_entity_id"], UUID)
        assert values["runtime_entity_id"] == runtime_entity_id

    @pytest.mark.asyncio
    async def test_execute_converts_uuid_string_to_uuid(
        self, effect: IntentEffectPostgresUpsert, mock_projector: MagicMock
    ) -> None:
        """A string-valued runtime_entity_id must be converted to UUID."""
        runtime_entity_id = uuid4()
        correlation_id = uuid4()

        await effect.execute(
            _make_upsert_payload(str(runtime_entity_id)),
            correlation_id=correlation_id,
        )

        mock_projector.upsert_partial.assert_awaited_once()
        values = mock_projector.upsert_partial.call_args.kwargs["values"]
        assert isinstance(values["runtime_entity_id"], UUID)
        assert values["runtime_entity_id"] == runtime_entity_id

    @pytest.mark.asyncio
    async def test_execute_preserves_none_runtime_entity_id(
        self, effect: IntentEffectPostgresUpsert, mock_projector: MagicMock
    ) -> None:
        """None (no runtime parent) must survive the chain unchanged."""
        await effect.execute(_make_upsert_payload(None), correlation_id=uuid4())

        mock_projector.upsert_partial.assert_awaited_once()
        values = mock_projector.upsert_partial.call_args.kwargs["values"]
        assert values["runtime_entity_id"] is None

    @pytest.mark.asyncio
    async def test_execute_rejects_malformed_runtime_entity_id_string(
        self, effect: IntentEffectPostgresUpsert, mock_projector: MagicMock
    ) -> None:
        """A non-UUID string must fail the upsert without partial writes."""
        with pytest.raises(RuntimeHostError, match="Failed to execute PostgreSQL"):
            await effect.execute(
                _make_upsert_payload("not-a-uuid"), correlation_id=uuid4()
            )

        mock_projector.upsert_partial.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_control_column_missing_from_upsert_uuid_columns_leaves_string(
        self,
        effect: IntentEffectPostgresUpsert,
        mock_projector: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Failing control: without the column in _UUID_COLUMNS, str stays str.

        Removing runtime_entity_id from the normalization set disables the
        str -> UUID conversion, so the string form reaches upsert_partial
        unconverted. This documents the control that the UUID assertions in
        test_execute_converts_uuid_string_to_uuid would fail against.
        """
        monkeypatch.setattr(
            IntentEffectPostgresUpsert,
            "_UUID_COLUMNS",
            IntentEffectPostgresUpsert._UUID_COLUMNS - {"runtime_entity_id"},
        )
        runtime_entity_id = uuid4()

        await effect.execute(
            _make_upsert_payload(str(runtime_entity_id)), correlation_id=uuid4()
        )

        mock_projector.upsert_partial.assert_awaited_once()
        values = mock_projector.upsert_partial.call_args.kwargs["values"]
        assert isinstance(values["runtime_entity_id"], str)
        assert values["runtime_entity_id"] == str(runtime_entity_id)


@pytest.mark.unit
class TestRuntimeEntityIdUpdateAdapter:
    """Update adapter: normalization and allowlist behaviour for the column."""

    @pytest.fixture
    def mock_pool(self) -> MagicMock:
        """Create a mock asyncpg pool with async acquire context manager."""
        return _make_mock_pool()

    @pytest.fixture
    def effect(self, mock_pool: MagicMock) -> IntentEffectPostgresUpdate:
        """Create an IntentEffectPostgresUpdate with mocked pool."""
        return IntentEffectPostgresUpdate(pool=mock_pool)

    def test_normalize_converts_runtime_entity_id_string_to_uuid(self) -> None:
        """A string runtime_entity_id must become a UUID for asyncpg."""
        runtime_entity_id = uuid4()
        record: dict[str, object] = {"runtime_entity_id": str(runtime_entity_id)}

        result = IntentEffectPostgresUpdate._normalize_for_asyncpg(record)

        assert isinstance(result["runtime_entity_id"], UUID)
        assert result["runtime_entity_id"] == runtime_entity_id

    def test_normalize_preserves_runtime_entity_id_uuid_instance(self) -> None:
        """A native UUID runtime_entity_id must pass through unconverted."""
        runtime_entity_id = uuid4()
        record: dict[str, object] = {"runtime_entity_id": runtime_entity_id}

        result = IntentEffectPostgresUpdate._normalize_for_asyncpg(record)

        assert result["runtime_entity_id"] is runtime_entity_id

    def test_normalize_preserves_runtime_entity_id_none(self) -> None:
        """None (no runtime parent) must pass through unchanged."""
        record: dict[str, object] = {"runtime_entity_id": None}

        result = IntentEffectPostgresUpdate._normalize_for_asyncpg(record)

        assert result["runtime_entity_id"] is None

    @pytest.mark.asyncio
    async def test_execute_accepts_runtime_entity_id_column(
        self, effect: IntentEffectPostgresUpdate, mock_pool: MagicMock
    ) -> None:
        """runtime_entity_id must pass the allowlist and reach asyncpg as UUID."""
        entity_id = uuid4()
        correlation_id = uuid4()
        runtime_entity_id = uuid4()

        payload = _make_update_payload(
            entity_id=entity_id,
            correlation_id=correlation_id,
            updates_value={"runtime_entity_id": str(runtime_entity_id)},
        )

        await effect.execute(payload, correlation_id=correlation_id)

        conn = mock_pool._mock_conn
        conn.execute.assert_awaited_once()
        call_args = conn.execute.call_args
        sql = call_args.args[0]
        assert '"runtime_entity_id"' in sql

        # SET value first, then entity_id and domain in the WHERE clause.
        positional_params = call_args.args[1:]
        assert isinstance(positional_params[0], UUID)
        assert positional_params[0] == runtime_entity_id
        assert entity_id in positional_params

    @pytest.mark.asyncio
    async def test_execute_refuses_unknown_column(
        self, effect: IntentEffectPostgresUpdate, mock_pool: MagicMock
    ) -> None:
        """Error chain: an unknown column is refused before any query runs."""
        entity_id = uuid4()
        correlation_id = uuid4()

        payload = _make_update_payload(
            entity_id=entity_id,
            correlation_id=correlation_id,
            updates_value={"definitely_not_a_column": "x"},
        )

        with pytest.raises(RuntimeHostError, match="not in allowlist"):
            await effect.execute(payload, correlation_id=correlation_id)

        conn = mock_pool._mock_conn
        conn.execute.assert_not_awaited()


@pytest.mark.unit
class TestReaderMapsRuntimeEntityId:
    """M1: rows written before migration 117 read runtime_entity_id as None."""

    @pytest.fixture
    def mock_pool(self) -> MagicMock:
        """Create a mock asyncpg connection pool."""
        return MagicMock(spec=asyncpg.Pool)

    @pytest.fixture
    def reader(self, mock_pool: MagicMock) -> ProjectionReaderRegistration:
        """Create a ProjectionReaderRegistration with mocked pool."""
        return ProjectionReaderRegistration(pool=mock_pool)

    def test_row_without_runtime_entity_id_key_maps_to_none(
        self, reader: ProjectionReaderRegistration
    ) -> None:
        """A pre-117 row (no runtime_entity_id key) must map to None."""
        row = create_mock_row()
        assert "runtime_entity_id" not in row

        projection = reader._row_to_projection(row)

        assert projection.runtime_entity_id is None

    def test_row_with_runtime_entity_id_uuid_is_carried_through(
        self, reader: ProjectionReaderRegistration
    ) -> None:
        """A row carrying a runtime_entity_id UUID must map it through."""
        row = create_mock_row()
        runtime_entity_id = uuid4()
        row["runtime_entity_id"] = runtime_entity_id

        projection = reader._row_to_projection(row)

        assert projection.runtime_entity_id == runtime_entity_id
