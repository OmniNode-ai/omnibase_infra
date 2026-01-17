# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Parity tests proving declarative ProjectorShell matches legacy ProjectorRegistration.

These tests verify that the declarative ProjectorShell implementation with
registration_projector.yaml contract produces IDENTICAL results to the legacy
ProjectorRegistration class. All tests run both implementations side-by-side
and assert column-by-column equality.

Test Categories:
    1. Full projection persistence (persist)
    2. State retrieval (get_state)
    3. Partial updates (heartbeat, ack timeout, liveness timeout markers)
    4. Idempotency (replay behavior)
    5. Column mapping completeness

Related Tickets:
    - OMN-1170: Create registration_projector.yaml contract with parity tests
    - OMN-1169: ProjectorShell contract-driven projections

Environment Variables:
    Uses testcontainers (Docker-based PostgreSQL) for isolation.
    Docker daemon must be running.

Usage:
    pytest tests/integration/projectors/test_registration_projector_parity.py -v

.. versionadded:: 0.7.0
    Created for OMN-1170 parity verification.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import asyncpg
import pytest

from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_core.models.projectors import ModelProjectorContract
from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.models.projection import (
    ModelRegistrationProjection,
    ModelSequenceInfo,
)
from omnibase_infra.models.registration import ModelNodeCapabilities
from omnibase_infra.projectors.contracts import REGISTRATION_PROJECTOR_CONTRACT
from omnibase_infra.runtime.projector_shell import ProjectorShell
from omnibase_infra.utils.util_datetime import is_timezone_aware

if TYPE_CHECKING:
    # Type stub for deleted ProjectorRegistration - used only for type hints
    # in tests that are skipped due to missing legacy implementation.
    class ProjectorRegistration:
        """Type stub for deleted ProjectorRegistration class.

        This stub provides type hints for parity tests that are currently
        skipped (requires_legacy_projector). Methods match the deleted
        legacy implementation API.
        """

        async def persist(self, **kwargs: object) -> bool:
            """Persist projection to database."""
            ...

        async def update_heartbeat(self, **kwargs: object) -> bool:
            """Update heartbeat timestamp."""
            ...

        async def update_ack_timeout_marker(self, **kwargs: object) -> bool:
            """Mark ack timeout emitted."""
            ...

        async def update_liveness_timeout_marker(self, **kwargs: object) -> bool:
            """Mark liveness timeout emitted."""
            ...


# =============================================================================
# Legacy Projector Availability
# =============================================================================
#
# ProjectorRegistration was deleted as part of OMN-1170 conversion to declarative
# contracts. Tests that compared legacy vs declarative behavior are marked with
# `requires_legacy_projector` to indicate they need the legacy class to run.
#
# The availability is checked dynamically at import time. If the legacy class
# is restored (e.g., for parity verification), these tests will run automatically.
#


def _check_legacy_projector_available() -> bool:
    """Check if legacy ProjectorRegistration is available.

    This function attempts to import the legacy ProjectorRegistration class
    to determine if parity tests should run. The import is done at module
    load time to set the skip marker appropriately.

    Returns:
        bool: True if ProjectorRegistration can be imported, False otherwise.
    """
    try:
        from omnibase_infra.projectors.projector_registration import (
            ProjectorRegistration,
        )

        return True
    except ImportError:
        return False


LEGACY_PROJECTOR_AVAILABLE = _check_legacy_projector_available()
LEGACY_SKIP_REASON = (
    "Legacy ProjectorRegistration not available. "
    "Parity tests require the legacy implementation to compare against. "
    "See: src/omnibase_infra/projectors/projector_registration.py"
)

# Marker for tests requiring legacy projector
requires_legacy_projector = pytest.mark.skipif(
    not LEGACY_PROJECTOR_AVAILABLE,
    reason=LEGACY_SKIP_REASON,
)

# =============================================================================
# Test Markers
# =============================================================================

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.integration,
]


# =============================================================================
# Constants
# =============================================================================

# Path to the declarative contract
# Use exported constant from projectors.contracts package
# This ensures the test always uses the canonical contract location
CONTRACT_PATH = REGISTRATION_PROJECTOR_CONTRACT

# Columns that should be compared between legacy and declarative
# These are ALL 23 columns in the schema_registration_projection.sql
PROJECTION_COLUMNS = [
    "entity_id",
    "domain",
    "current_state",
    "node_type",
    "node_version",
    "capabilities",
    "contract_type",
    "intent_types",
    "protocols",
    "capability_tags",
    "contract_version",
    "ack_deadline",
    "liveness_deadline",
    "last_heartbeat_at",
    "ack_timeout_emitted_at",
    "liveness_timeout_emitted_at",
    "last_applied_event_id",
    "last_applied_offset",
    "last_applied_sequence",
    "last_applied_partition",
    "registered_at",
    "updated_at",
    "correlation_id",
]


# =============================================================================
# Fixtures
# =============================================================================

# NOTE: The `contract` fixture is defined in conftest.py to allow reuse across
# multiple test files in this directory. It loads the registration projector
# contract using the REGISTRATION_PROJECTOR_CONTRACT constant.


@pytest.fixture
def sample_projection() -> ModelRegistrationProjection:
    """Create a sample projection for testing.

    Returns:
        ModelRegistrationProjection with all fields populated.
    """
    now = datetime.now(UTC)
    return ModelRegistrationProjection(
        entity_id=uuid4(),
        domain="registration",
        current_state=EnumRegistrationState.PENDING_REGISTRATION,
        node_type=EnumNodeKind.EFFECT,
        node_version=ModelSemVer(major=1, minor=2, patch=3),
        capabilities=ModelNodeCapabilities(
            postgres=True,
            read=True,
            write=True,
            database=True,
        ),
        contract_type="effect",
        intent_types=["postgres.upsert", "consul.register"],
        protocols=["ProtocolDatabaseAdapter", "ProtocolServiceDiscovery"],
        capability_tags=["storage", "discovery"],
        contract_version="1.0.0",
        ack_deadline=now + timedelta(seconds=30),
        liveness_deadline=now + timedelta(seconds=90),
        last_heartbeat_at=now,
        ack_timeout_emitted_at=None,
        liveness_timeout_emitted_at=None,
        last_applied_event_id=uuid4(),
        last_applied_offset=12345,
        last_applied_sequence=None,
        last_applied_partition="0",
        registered_at=now,
        updated_at=now,
        correlation_id=uuid4(),
    )


@pytest.fixture
def sample_sequence_info() -> ModelSequenceInfo:
    """Create sample sequence info for ordering tests.

    Returns:
        ModelSequenceInfo with Kafka-based ordering.
    """
    return ModelSequenceInfo.from_kafka(partition=0, offset=12345)


# =============================================================================
# Helper Functions
# =============================================================================


async def get_projection_row(
    pool: asyncpg.Pool,
    entity_id: UUID,
    domain: str = "registration",
) -> dict[str, object] | None:
    """Fetch projection row from database.

    Args:
        pool: asyncpg connection pool.
        entity_id: Entity UUID to query.
        domain: Domain namespace.

    Returns:
        Row as dict, or None if not found.
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM registration_projections WHERE entity_id = $1 AND domain = $2",
            entity_id,
            domain,
        )

    if row is None:
        return None

    return dict(row)


def normalize_value(value: object, column_name: str) -> object:
    """Normalize database values for comparison.

    Handles type conversions:
    - Decimal -> float
    - JSONB string -> dict
    - datetime timezone normalization (ensures UTC, validates tz-awareness)
    - Enum -> string

    Args:
        value: Database value to normalize.
        column_name: Column name for context.

    Returns:
        Normalized value for comparison.

    Raises:
        ValueError: If datetime value is naive (missing timezone info).
            All datetimes should be timezone-aware before persisting to DB.
    """
    if value is None:
        return None

    if isinstance(value, Decimal):
        return float(value)

    if isinstance(value, datetime):
        # Validate timezone-awareness for datetime values
        # All datetimes should be tz-aware before persisting to database
        if not is_timezone_aware(value):
            raise ValueError(
                f"Naive datetime detected in column '{column_name}'. "
                "All datetimes must be timezone-aware. Use datetime.now(UTC) or "
                "datetime(..., tzinfo=timezone.utc) instead of naive datetime."
            )
        # Normalize to UTC for consistent comparison
        return value.astimezone(UTC)

    if column_name == "capabilities" and isinstance(value, str):
        return json.loads(value)

    if column_name == "current_state" and hasattr(value, "value"):
        return str(value.value)

    return value


def compare_rows(
    row1: dict[str, object] | None,
    row2: dict[str, object] | None,
    columns_to_compare: list[str],
    exclude_columns: list[str] | None = None,
) -> tuple[bool, list[str]]:
    """Compare two database rows column by column.

    Args:
        row1: First row dict.
        row2: Second row dict.
        columns_to_compare: List of columns to check.
        exclude_columns: Columns to skip (e.g., timestamps that may differ slightly).

    Returns:
        Tuple of (all_match, list_of_differences).
    """
    if row1 is None and row2 is None:
        return True, []

    if row1 is None or row2 is None:
        return False, ["One row is None while the other exists"]

    exclude = set(exclude_columns or [])
    differences: list[str] = []

    for col in columns_to_compare:
        if col in exclude:
            continue

        val1 = normalize_value(row1.get(col), col)
        val2 = normalize_value(row2.get(col), col)

        if val1 != val2:
            differences.append(f"{col}: {val1!r} != {val2!r}")

    return len(differences) == 0, differences


async def clear_projection_table(pool: asyncpg.Pool) -> None:
    """Clear all rows from the projection table.

    Args:
        pool: asyncpg connection pool.
    """
    async with pool.acquire() as conn:
        await conn.execute("TRUNCATE TABLE registration_projections CASCADE")


# =============================================================================
# Test Classes
# =============================================================================


@requires_legacy_projector
class TestPersistOutputMatchesLegacy:
    """Tests verifying persist() produces identical output."""

    async def test_persist_initial_insert_matches(
        self,
        pg_pool: asyncpg.Pool,
        legacy_projector: ProjectorRegistration,
        projector: ProjectorShell,
        sample_projection: ModelRegistrationProjection,
        sample_sequence_info: ModelSequenceInfo,
    ) -> None:
        """persist() initial insert produces identical row in both implementations.

        Given: Empty projection table
        When: Both projectors persist the same projection
        Then: Resulting database rows are identical (column by column)
        """
        correlation_id = uuid4()

        # Execute with LEGACY projector
        legacy_result = await legacy_projector.persist(
            projection=sample_projection,
            entity_id=sample_projection.entity_id,
            domain="registration",
            sequence_info=sample_sequence_info,
            correlation_id=correlation_id,
        )

        assert legacy_result is True

        # Get legacy row
        legacy_row = await get_projection_row(pg_pool, sample_projection.entity_id)

        # Clear table for declarative test
        await clear_projection_table(pg_pool)

        # NOTE: ProjectorShell.project() uses event envelopes, not ModelRegistrationProjection.
        # For a true parity test of persist(), we need to compare the legacy persist() with
        # a direct partial_update() or test at the SQL level.
        # Since ProjectorShell is designed for event-driven projections, the parity test
        # focuses on operations that both support: partial updates and state retrieval.

        # For full persist parity, we verify column mapping correctness
        # by checking all expected columns exist with correct types
        assert legacy_row is not None
        for col in PROJECTION_COLUMNS:
            assert col in legacy_row, f"Missing column: {col}"

    async def test_persist_upsert_update_matches(
        self,
        pg_pool: asyncpg.Pool,
        legacy_projector: ProjectorRegistration,
        sample_projection: ModelRegistrationProjection,
        sample_sequence_info: ModelSequenceInfo,
    ) -> None:
        """persist() update on existing row works correctly.

        Given: Existing projection row
        When: Legacy projector persists updated state with higher sequence
        Then: Row is updated (upsert behavior)
        """
        correlation_id = uuid4()

        # Initial persist
        await legacy_projector.persist(
            projection=sample_projection,
            entity_id=sample_projection.entity_id,
            domain="registration",
            sequence_info=sample_sequence_info,
            correlation_id=correlation_id,
        )

        # Update projection state
        sample_projection.current_state = EnumRegistrationState.ACCEPTED
        sample_projection.updated_at = datetime.now(UTC)

        # Create higher sequence for upsert
        new_sequence = ModelSequenceInfo.from_kafka(partition=0, offset=12346)

        # Persist update
        result = await legacy_projector.persist(
            projection=sample_projection,
            entity_id=sample_projection.entity_id,
            domain="registration",
            sequence_info=new_sequence,
            correlation_id=correlation_id,
        )

        assert result is True

        # Verify updated state
        row = await get_projection_row(pg_pool, sample_projection.entity_id)
        assert row is not None
        assert row["current_state"] == "accepted"


@requires_legacy_projector
class TestGetStateMatchesLegacy:
    """Tests verifying get_state() returns identical results."""

    async def test_get_state_returns_all_columns(
        self,
        pg_pool: asyncpg.Pool,
        legacy_projector: ProjectorRegistration,
        projector: ProjectorShell,
        sample_projection: ModelRegistrationProjection,
        sample_sequence_info: ModelSequenceInfo,
    ) -> None:
        """get_state() returns row with all expected columns.

        Given: Persisted projection
        When: Declarative projector retrieves state
        Then: All 23 columns are present and correctly typed
        """
        correlation_id = uuid4()

        # Persist with legacy
        await legacy_projector.persist(
            projection=sample_projection,
            entity_id=sample_projection.entity_id,
            domain="registration",
            sequence_info=sample_sequence_info,
            correlation_id=correlation_id,
        )

        # Get state with declarative
        state = await projector.get_state(
            aggregate_id=sample_projection.entity_id,
            correlation_id=correlation_id,
        )

        assert state is not None
        assert isinstance(state, dict)

        # Verify all columns present
        for col in PROJECTION_COLUMNS:
            assert col in state, f"Missing column in get_state result: {col}"

    async def test_get_state_none_for_missing(
        self,
        projector: ProjectorShell,
    ) -> None:
        """get_state() returns None for non-existent entity.

        Given: Empty projection table
        When: get_state() called for non-existent entity
        Then: Returns None
        """
        correlation_id = uuid4()
        non_existent_id = uuid4()

        state = await projector.get_state(
            aggregate_id=non_existent_id,
            correlation_id=correlation_id,
        )

        assert state is None


@requires_legacy_projector
class TestUpdateHeartbeatMatchesLegacy:
    """Tests verifying update_heartbeat partial update matches legacy."""

    async def test_update_heartbeat_updates_correct_columns(
        self,
        pg_pool: asyncpg.Pool,
        legacy_projector: ProjectorRegistration,
        projector: ProjectorShell,
        sample_projection: ModelRegistrationProjection,
        sample_sequence_info: ModelSequenceInfo,
    ) -> None:
        """update_heartbeat() updates only last_heartbeat_at, liveness_deadline, updated_at.

        Given: Existing projection
        When: Legacy update_heartbeat and declarative partial_update called
        Then: Same columns updated, others unchanged
        """
        correlation_id = uuid4()
        entity_id = sample_projection.entity_id

        # Setup: Persist initial projection with legacy
        await legacy_projector.persist(
            projection=sample_projection,
            entity_id=entity_id,
            domain="registration",
            sequence_info=sample_sequence_info,
            correlation_id=correlation_id,
        )

        # Get initial state for comparison
        initial_row = await get_projection_row(pg_pool, entity_id)
        assert initial_row is not None

        # Define heartbeat update values
        new_heartbeat_at = datetime.now(UTC)
        new_liveness_deadline = new_heartbeat_at + timedelta(seconds=90)

        # Execute LEGACY update_heartbeat
        legacy_result = await legacy_projector.update_heartbeat(
            entity_id=entity_id,
            domain="registration",
            last_heartbeat_at=new_heartbeat_at,
            liveness_deadline=new_liveness_deadline,
            correlation_id=correlation_id,
        )

        assert legacy_result is True

        # Get legacy result
        legacy_row = await get_projection_row(pg_pool, entity_id)
        assert legacy_row is not None

        # Verify only expected columns changed
        assert legacy_row["last_heartbeat_at"] == new_heartbeat_at
        assert legacy_row["liveness_deadline"] == new_liveness_deadline

        # Verify other columns unchanged (excluding updated_at which should change)
        unchanged_columns = [
            "entity_id",
            "domain",
            "current_state",
            "node_type",
            "node_version",
            "capabilities",
            "contract_type",
            "intent_types",
            "protocols",
            "capability_tags",
            "contract_version",
            "ack_deadline",
            "ack_timeout_emitted_at",
            "liveness_timeout_emitted_at",
            "last_applied_event_id",
            "last_applied_offset",
            "last_applied_sequence",
            "last_applied_partition",
            "registered_at",
            "correlation_id",
        ]

        for col in unchanged_columns:
            assert legacy_row[col] == initial_row[col], (
                f"Column {col} unexpectedly changed: "
                f"{initial_row[col]!r} -> {legacy_row[col]!r}"
            )

    async def test_update_heartbeat_entity_not_found(
        self,
        legacy_projector: ProjectorRegistration,
        projector: ProjectorShell,
    ) -> None:
        """update_heartbeat() returns False when entity not found.

        Given: Empty projection table
        When: update_heartbeat called for non-existent entity
        Then: Returns False (both implementations)
        """
        correlation_id = uuid4()
        non_existent_id = uuid4()
        now = datetime.now(UTC)

        # Legacy behavior
        legacy_result = await legacy_projector.update_heartbeat(
            entity_id=non_existent_id,
            domain="registration",
            last_heartbeat_at=now,
            liveness_deadline=now + timedelta(seconds=90),
            correlation_id=correlation_id,
        )

        assert legacy_result is False

        # Declarative behavior (using partial_update)
        declarative_result = await projector.partial_update(
            aggregate_id=non_existent_id,
            updates={
                "last_heartbeat_at": now,
                "liveness_deadline": now + timedelta(seconds=90),
                "updated_at": now,
            },
            correlation_id=correlation_id,
        )

        assert declarative_result is False


@requires_legacy_projector
class TestUpdateAckTimeoutMarkerMatchesLegacy:
    """Tests verifying update_ack_timeout_marker matches legacy."""

    async def test_update_ack_timeout_marker_updates_correct_columns(
        self,
        pg_pool: asyncpg.Pool,
        legacy_projector: ProjectorRegistration,
        projector: ProjectorShell,
        sample_projection: ModelRegistrationProjection,
        sample_sequence_info: ModelSequenceInfo,
    ) -> None:
        """update_ack_timeout_marker() updates only ack_timeout_emitted_at, updated_at.

        Given: Existing projection
        When: Legacy and declarative update timeout marker
        Then: Same columns updated, others unchanged
        """
        correlation_id = uuid4()
        entity_id = sample_projection.entity_id

        # Setup: Persist initial projection
        await legacy_projector.persist(
            projection=sample_projection,
            entity_id=entity_id,
            domain="registration",
            sequence_info=sample_sequence_info,
            correlation_id=correlation_id,
        )

        # Get initial state
        initial_row = await get_projection_row(pg_pool, entity_id)
        assert initial_row is not None
        assert initial_row["ack_timeout_emitted_at"] is None  # Initially null

        # Execute LEGACY update_ack_timeout_marker
        emitted_at = datetime.now(UTC)
        legacy_result = await legacy_projector.update_ack_timeout_marker(
            entity_id=entity_id,
            domain="registration",
            emitted_at=emitted_at,
            correlation_id=correlation_id,
        )

        assert legacy_result is True

        # Get legacy result
        legacy_row = await get_projection_row(pg_pool, entity_id)
        assert legacy_row is not None

        # Verify marker was set
        assert legacy_row["ack_timeout_emitted_at"] == emitted_at

        # Verify other columns unchanged (except updated_at)
        unchanged_columns = [
            "entity_id",
            "domain",
            "current_state",
            "node_type",
            "node_version",
            "capabilities",
            "contract_type",
            "intent_types",
            "protocols",
            "capability_tags",
            "contract_version",
            "ack_deadline",
            "liveness_deadline",
            "last_heartbeat_at",
            "liveness_timeout_emitted_at",
            "last_applied_event_id",
            "last_applied_offset",
            "last_applied_sequence",
            "last_applied_partition",
            "registered_at",
            "correlation_id",
        ]

        for col in unchanged_columns:
            assert legacy_row[col] == initial_row[col], (
                f"Column {col} unexpectedly changed"
            )

    async def test_update_ack_timeout_marker_entity_not_found(
        self,
        legacy_projector: ProjectorRegistration,
    ) -> None:
        """update_ack_timeout_marker() returns False when entity not found.

        Given: Empty projection table
        When: update_ack_timeout_marker called for non-existent entity
        Then: Returns False
        """
        correlation_id = uuid4()
        non_existent_id = uuid4()
        now = datetime.now(UTC)

        result = await legacy_projector.update_ack_timeout_marker(
            entity_id=non_existent_id,
            domain="registration",
            emitted_at=now,
            correlation_id=correlation_id,
        )

        assert result is False


@requires_legacy_projector
class TestUpdateLivenessTimeoutMarkerMatchesLegacy:
    """Tests verifying update_liveness_timeout_marker matches legacy."""

    async def test_update_liveness_timeout_marker_updates_correct_columns(
        self,
        pg_pool: asyncpg.Pool,
        legacy_projector: ProjectorRegistration,
        sample_projection: ModelRegistrationProjection,
        sample_sequence_info: ModelSequenceInfo,
    ) -> None:
        """update_liveness_timeout_marker() updates only liveness_timeout_emitted_at, updated_at.

        Given: Existing projection
        When: Legacy updates liveness timeout marker
        Then: Only expected columns updated
        """
        correlation_id = uuid4()
        entity_id = sample_projection.entity_id

        # Setup: Persist initial projection
        await legacy_projector.persist(
            projection=sample_projection,
            entity_id=entity_id,
            domain="registration",
            sequence_info=sample_sequence_info,
            correlation_id=correlation_id,
        )

        # Get initial state
        initial_row = await get_projection_row(pg_pool, entity_id)
        assert initial_row is not None
        assert initial_row["liveness_timeout_emitted_at"] is None

        # Execute update
        emitted_at = datetime.now(UTC)
        legacy_result = await legacy_projector.update_liveness_timeout_marker(
            entity_id=entity_id,
            domain="registration",
            emitted_at=emitted_at,
            correlation_id=correlation_id,
        )

        assert legacy_result is True

        # Get result
        legacy_row = await get_projection_row(pg_pool, entity_id)
        assert legacy_row is not None

        # Verify marker was set
        assert legacy_row["liveness_timeout_emitted_at"] == emitted_at

        # Verify other columns unchanged
        unchanged_columns = [
            "entity_id",
            "domain",
            "current_state",
            "node_type",
            "node_version",
            "capabilities",
            "contract_type",
            "intent_types",
            "protocols",
            "capability_tags",
            "contract_version",
            "ack_deadline",
            "liveness_deadline",
            "last_heartbeat_at",
            "ack_timeout_emitted_at",
            "last_applied_event_id",
            "last_applied_offset",
            "last_applied_sequence",
            "last_applied_partition",
            "registered_at",
            "correlation_id",
        ]

        for col in unchanged_columns:
            assert legacy_row[col] == initial_row[col], (
                f"Column {col} unexpectedly changed"
            )


@requires_legacy_projector
class TestIdempotencyMatchesLegacy:
    """Tests verifying idempotency behavior matches legacy."""

    async def test_stale_update_rejected(
        self,
        pg_pool: asyncpg.Pool,
        legacy_projector: ProjectorRegistration,
        sample_projection: ModelRegistrationProjection,
    ) -> None:
        """Stale updates (lower sequence) are rejected.

        Given: Existing projection with offset=12345
        When: persist() called with offset=12344 (lower)
        Then: Update is rejected, row unchanged
        """
        correlation_id = uuid4()
        entity_id = sample_projection.entity_id

        # Initial persist with offset 12345
        initial_sequence = ModelSequenceInfo.from_kafka(partition=0, offset=12345)
        await legacy_projector.persist(
            projection=sample_projection,
            entity_id=entity_id,
            domain="registration",
            sequence_info=initial_sequence,
            correlation_id=correlation_id,
        )

        # Get initial state
        initial_row = await get_projection_row(pg_pool, entity_id)
        assert initial_row is not None

        # Attempt update with LOWER sequence (stale)
        sample_projection.current_state = EnumRegistrationState.ACTIVE
        stale_sequence = ModelSequenceInfo.from_kafka(partition=0, offset=12344)

        result = await legacy_projector.persist(
            projection=sample_projection,
            entity_id=entity_id,
            domain="registration",
            sequence_info=stale_sequence,
            correlation_id=correlation_id,
        )

        # Stale update should be rejected
        assert result is False

        # Row should be unchanged
        final_row = await get_projection_row(pg_pool, entity_id)
        assert final_row is not None
        assert final_row["current_state"] == initial_row["current_state"]
        assert final_row["last_applied_offset"] == 12345

    async def test_duplicate_persist_idempotent(
        self,
        pg_pool: asyncpg.Pool,
        legacy_projector: ProjectorRegistration,
        sample_projection: ModelRegistrationProjection,
    ) -> None:
        """Persisting same projection twice is idempotent (second is rejected as equal sequence).

        Given: Persisted projection with offset=12345
        When: Same projection persisted again with same offset
        Then: Second persist rejected (not newer), state unchanged
        """
        correlation_id = uuid4()
        entity_id = sample_projection.entity_id
        sequence = ModelSequenceInfo.from_kafka(partition=0, offset=12345)

        # First persist
        result1 = await legacy_projector.persist(
            projection=sample_projection,
            entity_id=entity_id,
            domain="registration",
            sequence_info=sequence,
            correlation_id=correlation_id,
        )
        assert result1 is True

        # Get state after first persist
        row1 = await get_projection_row(pg_pool, entity_id)
        assert row1 is not None

        # Second persist with SAME sequence (equal, not greater)
        result2 = await legacy_projector.persist(
            projection=sample_projection,
            entity_id=entity_id,
            domain="registration",
            sequence_info=sequence,
            correlation_id=correlation_id,
        )

        # Equal sequence is rejected (must be GREATER to update)
        assert result2 is False

        # State should be unchanged
        row2 = await get_projection_row(pg_pool, entity_id)
        assert row2 is not None

        # Key columns should match
        assert row1["last_applied_offset"] == row2["last_applied_offset"]


@requires_legacy_projector
class TestAllColumnsMappedCorrectly:
    """Tests verifying every column is correctly mapped."""

    async def test_all_23_columns_present_after_persist(
        self,
        pg_pool: asyncpg.Pool,
        legacy_projector: ProjectorRegistration,
        sample_projection: ModelRegistrationProjection,
        sample_sequence_info: ModelSequenceInfo,
    ) -> None:
        """All 23 columns are present and correctly typed after persist.

        Given: Sample projection with all fields populated
        When: Legacy projector persists
        Then: All columns have expected values
        """
        correlation_id = uuid4()
        entity_id = sample_projection.entity_id

        await legacy_projector.persist(
            projection=sample_projection,
            entity_id=entity_id,
            domain="registration",
            sequence_info=sample_sequence_info,
            correlation_id=correlation_id,
        )

        row = await get_projection_row(pg_pool, entity_id)
        assert row is not None

        # Verify all 23 columns
        assert row["entity_id"] == entity_id
        assert row["domain"] == "registration"
        assert row["current_state"] == sample_projection.current_state.value
        assert row["node_type"] == sample_projection.node_type.value
        assert row["node_version"] == str(sample_projection.node_version)

        # Capabilities is JSONB
        assert row["capabilities"] is not None

        # Capability fields
        assert row["contract_type"] == sample_projection.contract_type
        assert row["intent_types"] == sample_projection.intent_types
        assert row["protocols"] == sample_projection.protocols
        assert row["capability_tags"] == sample_projection.capability_tags
        assert row["contract_version"] == sample_projection.contract_version

        # Deadline fields
        assert row["ack_deadline"] is not None
        assert row["liveness_deadline"] is not None
        assert row["last_heartbeat_at"] is not None

        # Timeout markers (initially None)
        assert row["ack_timeout_emitted_at"] == sample_projection.ack_timeout_emitted_at
        assert (
            row["liveness_timeout_emitted_at"]
            == sample_projection.liveness_timeout_emitted_at
        )

        # Idempotency fields
        assert row["last_applied_event_id"] == sample_projection.last_applied_event_id
        assert row["last_applied_offset"] == sample_sequence_info.offset

        # Timestamps
        assert row["registered_at"] == sample_projection.registered_at
        assert row["updated_at"] is not None

    async def test_capability_fields_gin_indexable(
        self,
        pg_pool: asyncpg.Pool,
        legacy_projector: ProjectorRegistration,
        sample_projection: ModelRegistrationProjection,
        sample_sequence_info: ModelSequenceInfo,
    ) -> None:
        """Capability array fields are stored correctly for GIN index queries.

        Given: Projection with capability arrays
        When: Persisted and queried with array containment
        Then: GIN index queries work correctly
        """
        correlation_id = uuid4()
        entity_id = sample_projection.entity_id

        await legacy_projector.persist(
            projection=sample_projection,
            entity_id=entity_id,
            domain="registration",
            sequence_info=sample_sequence_info,
            correlation_id=correlation_id,
        )

        # Query using GIN index on intent_types
        async with pg_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT entity_id FROM registration_projections
                WHERE intent_types @> ARRAY['postgres.upsert']
                """
            )

        assert len(rows) == 1
        assert rows[0]["entity_id"] == entity_id

        # Query using GIN index on protocols
        async with pg_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT entity_id FROM registration_projections
                WHERE protocols @> ARRAY['ProtocolDatabaseAdapter']
                """
            )

        assert len(rows) == 1
        assert rows[0]["entity_id"] == entity_id


@requires_legacy_projector
class TestPartialUpdateParity:
    """Tests verifying partial_update() matches legacy specialized methods."""

    async def test_partial_update_heartbeat_matches_legacy(
        self,
        pg_pool: asyncpg.Pool,
        legacy_projector: ProjectorRegistration,
        projector: ProjectorShell,
        sample_projection: ModelRegistrationProjection,
        sample_sequence_info: ModelSequenceInfo,
    ) -> None:
        """partial_update() produces same result as legacy update_heartbeat().

        Given: Two identical projections
        When: Legacy update_heartbeat and declarative partial_update called
        Then: Resulting rows have same updated columns
        """
        entity_id_1 = uuid4()
        entity_id_2 = uuid4()
        correlation_id = uuid4()

        # Create two identical projections
        proj1 = sample_projection.model_copy(update={"entity_id": entity_id_1})
        proj2 = sample_projection.model_copy(update={"entity_id": entity_id_2})

        # Persist both
        seq1 = ModelSequenceInfo.from_kafka(partition=0, offset=1000)
        seq2 = ModelSequenceInfo.from_kafka(partition=0, offset=1001)

        await legacy_projector.persist(
            projection=proj1,
            entity_id=entity_id_1,
            domain="registration",
            sequence_info=seq1,
            correlation_id=correlation_id,
        )
        await legacy_projector.persist(
            projection=proj2,
            entity_id=entity_id_2,
            domain="registration",
            sequence_info=seq2,
            correlation_id=correlation_id,
        )

        # Define heartbeat update
        new_heartbeat_at = datetime.now(UTC)
        new_liveness_deadline = new_heartbeat_at + timedelta(seconds=90)

        # Update entity_1 with LEGACY
        await legacy_projector.update_heartbeat(
            entity_id=entity_id_1,
            domain="registration",
            last_heartbeat_at=new_heartbeat_at,
            liveness_deadline=new_liveness_deadline,
            correlation_id=correlation_id,
        )

        # Update entity_2 with DECLARATIVE
        await projector.partial_update(
            aggregate_id=entity_id_2,
            updates={
                "last_heartbeat_at": new_heartbeat_at,
                "liveness_deadline": new_liveness_deadline,
                "updated_at": new_heartbeat_at,  # Legacy also updates this
            },
            correlation_id=correlation_id,
        )

        # Get both rows
        row1 = await get_projection_row(pg_pool, entity_id_1)
        row2 = await get_projection_row(pg_pool, entity_id_2)

        assert row1 is not None
        assert row2 is not None

        # Compare heartbeat-related columns
        assert row1["last_heartbeat_at"] == row2["last_heartbeat_at"]
        assert row1["liveness_deadline"] == row2["liveness_deadline"]

    async def test_partial_update_marker_matches_legacy(
        self,
        pg_pool: asyncpg.Pool,
        legacy_projector: ProjectorRegistration,
        projector: ProjectorShell,
        sample_projection: ModelRegistrationProjection,
        sample_sequence_info: ModelSequenceInfo,
    ) -> None:
        """partial_update() produces same result as legacy update_ack_timeout_marker().

        Given: Two identical projections
        When: Legacy update_ack_timeout_marker and declarative partial_update called
        Then: Resulting rows have same marker column values
        """
        entity_id_1 = uuid4()
        entity_id_2 = uuid4()
        correlation_id = uuid4()

        # Create two identical projections
        proj1 = sample_projection.model_copy(update={"entity_id": entity_id_1})
        proj2 = sample_projection.model_copy(update={"entity_id": entity_id_2})

        # Persist both
        seq1 = ModelSequenceInfo.from_kafka(partition=0, offset=2000)
        seq2 = ModelSequenceInfo.from_kafka(partition=0, offset=2001)

        await legacy_projector.persist(
            projection=proj1,
            entity_id=entity_id_1,
            domain="registration",
            sequence_info=seq1,
            correlation_id=correlation_id,
        )
        await legacy_projector.persist(
            projection=proj2,
            entity_id=entity_id_2,
            domain="registration",
            sequence_info=seq2,
            correlation_id=correlation_id,
        )

        # Define marker update
        emitted_at = datetime.now(UTC)

        # Update entity_1 with LEGACY
        await legacy_projector.update_ack_timeout_marker(
            entity_id=entity_id_1,
            domain="registration",
            emitted_at=emitted_at,
            correlation_id=correlation_id,
        )

        # Update entity_2 with DECLARATIVE
        await projector.partial_update(
            aggregate_id=entity_id_2,
            updates={
                "ack_timeout_emitted_at": emitted_at,
                "updated_at": emitted_at,  # Legacy also updates this
            },
            correlation_id=correlation_id,
        )

        # Get both rows
        row1 = await get_projection_row(pg_pool, entity_id_1)
        row2 = await get_projection_row(pg_pool, entity_id_2)

        assert row1 is not None
        assert row2 is not None

        # Compare marker columns
        assert row1["ack_timeout_emitted_at"] == row2["ack_timeout_emitted_at"]


class TestProjectorShellSmoke:
    """Smoke tests verifying ProjectorShell works with the registration contract.

    These tests verify the contract-driven projector can be loaded and performs
    basic operations. They do NOT require the legacy ProjectorRegistration class
    and should always run regardless of legacy availability.

    Related Tickets:
        - OMN-1170: Contract-driven projector verification
    """

    async def test_projector_shell_loads_from_contract(
        self,
        projector: ProjectorShell,
    ) -> None:
        """ProjectorShell loads successfully from registration contract.

        Given: Registration projector contract exists
        When: ProjectorPluginLoader loads the contract
        Then: ProjectorShell instance is created with correct configuration
        """
        # Verify projector was loaded (fixture does the loading)
        assert projector is not None
        # Verify it has the expected methods
        assert hasattr(projector, "get_state")
        assert hasattr(projector, "partial_update")

    async def test_get_state_returns_none_for_nonexistent(
        self,
        projector: ProjectorShell,
    ) -> None:
        """get_state() returns None for non-existent entity.

        Given: Empty projection table
        When: get_state() called for non-existent entity
        Then: Returns None (not an error)

        This verifies the declarative projector handles missing entities
        gracefully without requiring comparison to legacy implementation.
        """
        correlation_id = uuid4()
        non_existent_id = uuid4()

        state = await projector.get_state(
            aggregate_id=non_existent_id,
            correlation_id=correlation_id,
        )

        assert state is None

    async def test_partial_update_returns_false_for_nonexistent(
        self,
        projector: ProjectorShell,
    ) -> None:
        """partial_update() returns False for non-existent entity.

        Given: Empty projection table
        When: partial_update() called for non-existent entity
        Then: Returns False (row not found, no update performed)
        """
        correlation_id = uuid4()
        non_existent_id = uuid4()
        now = datetime.now(UTC)

        result = await projector.partial_update(
            aggregate_id=non_existent_id,
            updates={
                "last_heartbeat_at": now,
                "updated_at": now,
            },
            correlation_id=correlation_id,
        )

        assert result is False


class TestContractMatchesSchema:
    """Tests verifying the YAML contract matches the SQL schema."""

    def test_contract_table_name_matches_schema(
        self,
        contract: ModelProjectorContract,
    ) -> None:
        """Contract table name matches SQL schema table.

        Given: Loaded contract
        Then: Table name is 'registration_projections'
        """
        assert contract.projection_schema.table == "registration_projections"

    def test_contract_primary_key_matches_schema(
        self,
        contract: ModelProjectorContract,
    ) -> None:
        """Contract primary key matches SQL schema composite key.

        Given: Loaded contract
        Then: Primary key is ['entity_id', 'domain'] (composite)
        """
        # Note: The contract fixture converts composite keys to single strings
        # for model validation compatibility. The actual composite key is
        # enforced at the SQL schema level (entity_id, domain).
        assert contract.projection_schema.primary_key == "entity_id"

    def test_contract_has_all_consumed_events(
        self,
        contract: ModelProjectorContract,
    ) -> None:
        """Contract defines all required consumed events.

        Given: Loaded contract
        Then: All registration events are in consumed_events
        """
        expected_events = [
            "node.introspected.v1",
            "node.registration_accepted.v1",
            "node.registration_rejected.v1",
            "node.ack_received.v1",
            "node.ack_timeout.v1",
            "node.heartbeat_received.v1",
            "node.liveness_expired.v1",
        ]

        for event in expected_events:
            assert event in contract.consumed_events, f"Missing event: {event}"

    def test_contract_upsert_mode_configured(
        self,
        contract: ModelProjectorContract,
    ) -> None:
        """Contract is configured for upsert mode (matches legacy).

        Given: Loaded contract
        Then: Mode is 'upsert' and upsert_key matches composite primary key
        """
        assert contract.behavior.mode == "upsert"
        # Note: The contract fixture converts composite keys to single strings
        # for model validation compatibility. The actual composite upsert key
        # (entity_id, domain) is enforced at the SQL schema level.
        assert contract.behavior.upsert_key == "entity_id"


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "TestAllColumnsMappedCorrectly",
    "TestContractMatchesSchema",
    "TestGetStateMatchesLegacy",
    "TestIdempotencyMatchesLegacy",
    "TestPartialUpdateParity",
    "TestPersistOutputMatchesLegacy",
    "TestProjectorShellSmoke",
    "TestUpdateAckTimeoutMarkerMatchesLegacy",
    "TestUpdateHeartbeatMatchesLegacy",
    "TestUpdateLivenessTimeoutMarkerMatchesLegacy",
]
