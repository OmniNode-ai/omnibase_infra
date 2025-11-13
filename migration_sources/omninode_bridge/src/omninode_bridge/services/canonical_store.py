"""
CanonicalStoreService - Version-Controlled State Management with Optimistic Concurrency Control.

This service provides the canonical state store for workflow state management with:
- Optimistic concurrency control using version numbers
- PostgreSQL-backed persistence
- Event emission on state changes and conflicts
- Provenance tracking for all state mutations
- Prometheus metrics for observability

ONEX v2.0 Compliance:
- Suffix-based naming: CanonicalStoreService
- Strong typing with Pydantic models
- Comprehensive error handling
- Event-driven architecture with Kafka integration

Pure Reducer Refactor:
- Single source of truth for workflow state (Wave 2A)
- Version-based conflict detection
- Immutable state transitions with provenance
- Complete audit trail via events

Reference: docs/planning/PURE_REDUCER_REFACTOR_PLAN.md (Wave 2, Workstream 2A)

Example Usage:
    ```python
    # Initialize service
    service = CanonicalStoreService(postgres_client, kafka_client)

    # Get current state
    state = await service.get_state("workflow-123")
    print(f"Current version: {state.version}")

    # Compute new state (pure function)
    new_state = compute_new_state(state.state, action)

    # Try to commit with optimistic locking
    result = await service.try_commit(
        workflow_key="workflow-123",
        expected_version=state.version,
        state_prime=new_state,
        provenance={
            "effect_id": "effect-456",
            "timestamp": datetime.now(UTC).isoformat(),
            "action_id": "action-789"
        }
    )

    # Handle result
    if isinstance(result, EventStateCommitted):
        print(f"Success! New version: {result.new_version}")
    else:
        print(f"Conflict! Expected v{expected_version}, got v{result.actual_version}")
        # Retry with new version
    ```
"""

import json
import logging
import time
from datetime import UTC, datetime
from typing import Any, Union
from uuid import uuid4

from prometheus_client import Counter, Histogram
from pydantic import BaseModel, Field

from omninode_bridge.infrastructure.entities.model_workflow_state import (
    ModelWorkflowState,
)
from omninode_bridge.services.kafka_client import KafkaClient
from omninode_bridge.services.postgres_client import PostgresClient

logger = logging.getLogger(__name__)

# Prometheus Metrics
state_commits_total = Counter(
    "canonical_store_state_commits_total",
    "Total successful state commits",
    ["workflow_key"],
)

state_conflicts_total = Counter(
    "canonical_store_state_conflicts_total",
    "Total state conflicts encountered",
    ["workflow_key"],
)

commit_latency_ms = Histogram(
    "canonical_store_commit_latency_ms",
    "State commit latency in milliseconds",
    buckets=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
)

get_state_total = Counter(
    "canonical_store_get_state_total",
    "Total get_state calls",
)

get_state_errors = Counter(
    "canonical_store_get_state_errors",
    "Total get_state errors",
)


# ============================================================================
# Event Models
# ============================================================================


class EventStateCommitted(BaseModel):
    """
    Event emitted when a state commit succeeds.

    Published to Kafka topic: omninode_bridge_state_committed_v1
    Used for: Audit trail, downstream projections, monitoring

    Example:
        >>> event = EventStateCommitted(
        ...     workflow_key="workflow-123",
        ...     new_version=2,
        ...     state_snapshot={"items": [1, 2, 3], "count": 3},
        ...     provenance={"effect_id": "effect-456", "timestamp": "2025-10-21T12:00:00Z"}
        ... )
        >>> assert event.new_version == 2
    """

    event_type: str = Field(
        default="state_committed", description="Event type identifier"
    )
    workflow_key: str = Field(..., description="Workflow identifier")
    new_version: int = Field(..., description="New version number after commit", ge=1)
    state_snapshot: dict[str, Any] = Field(
        ..., description="Complete state snapshot after commit"
    )
    provenance: dict[str, Any] = Field(..., description="Provenance metadata")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Event timestamp (UTC)",
    )
    correlation_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Event correlation ID",
    )


class EventStateConflict(BaseModel):
    """
    Event emitted when a state commit fails due to version conflict.

    Published to Kafka topic: omninode_bridge_state_conflicts_v1
    Used for: Conflict monitoring, retry logic optimization, debugging

    Example:
        >>> event = EventStateConflict(
        ...     workflow_key="workflow-123",
        ...     expected_version=1,
        ...     actual_version=2,
        ...     reason="concurrent_modification"
        ... )
        >>> assert event.expected_version < event.actual_version
    """

    event_type: str = Field(
        default="state_conflict", description="Event type identifier"
    )
    workflow_key: str = Field(..., description="Workflow identifier")
    expected_version: int = Field(
        ..., description="Expected version number (optimistic lock)", ge=1
    )
    actual_version: int = Field(
        ..., description="Actual current version in database", ge=1
    )
    reason: str = Field(
        default="version_mismatch",
        description="Conflict reason (version_mismatch, concurrent_modification, etc.)",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Event timestamp (UTC)",
    )
    correlation_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Event correlation ID",
    )


# ============================================================================
# Service Implementation
# ============================================================================


class CanonicalStoreService:
    """
    Canonical state store with optimistic concurrency control.

    Provides version-controlled state management for workflows with:
    - Atomic read-modify-write operations via PostgreSQL
    - Optimistic locking using version numbers
    - Event emission for state changes and conflicts
    - Provenance tracking for complete audit trail
    - Prometheus metrics for observability

    Performance Targets:
    - get_state: < 5ms (single SELECT query)
    - try_commit: < 10ms (UPDATE with version check)
    - Event emission: < 20ms (async Kafka publish)

    Thread Safety: Safe for concurrent use with different workflow_keys.
    Concurrency: Uses optimistic locking for same workflow_key updates.

    Attributes:
        postgres_client: PostgreSQL client for database operations
        kafka_client: Kafka client for event publishing
        _metrics_commits_total: Counter for successful commits
        _metrics_conflicts_total: Counter for version conflicts
    """

    def __init__(
        self,
        postgres_client: PostgresClient,
        kafka_client: KafkaClient | None = None,
    ):
        """
        Initialize CanonicalStoreService.

        Args:
            postgres_client: Connected PostgreSQL client instance
            kafka_client: Optional Kafka client for event publishing
                         (if None, events are logged but not published)
        """
        self.postgres_client = postgres_client
        self.kafka_client = kafka_client

        # Kafka topics
        self._topic_state_committed = "omninode_bridge_state_committed_v1"
        self._topic_state_conflicts = "omninode_bridge_state_conflicts_v1"

        # Instance-level metrics (for testing and local tracking)
        self._metrics_get_state_total = 0
        self._metrics_get_state_errors = 0
        self._metrics_commits_total = 0
        self._metrics_conflicts_total = 0

    async def get_state(self, workflow_key: str) -> ModelWorkflowState:
        """
        Retrieve current workflow state with version number.

        This method performs a single SELECT query to fetch the canonical state
        for the given workflow. The returned state includes the current version
        number for use in subsequent optimistic locking operations.

        Performance: Typically < 5ms for hot data in connection pool.

        Args:
            workflow_key: Human-readable workflow identifier

        Returns:
            Complete workflow state with version, state, provenance, timestamps

        Raises:
            ValueError: If workflow_key is empty or invalid
            RuntimeError: If workflow not found in database
            Exception: For database connection or query errors

        Example:
            >>> state = await service.get_state("workflow-123")
            >>> print(f"Version {state.version}: {state.state}")
            Version 1: {'items': [], 'count': 0}
        """
        if not workflow_key or not workflow_key.strip():
            raise ValueError("workflow_key must be non-empty string")

        query = """
            SELECT workflow_key, version, state, updated_at, schema_version, provenance
            FROM workflow_state
            WHERE workflow_key = $1
        """

        try:
            get_state_total.inc()
            self._metrics_get_state_total += 1

            row = await self.postgres_client.fetch_one(query, workflow_key)

            if not row:
                get_state_errors.inc()
                self._metrics_get_state_errors += 1
                raise RuntimeError(
                    f"Workflow state not found for workflow_key='{workflow_key}'. "
                    "Ensure workflow exists before calling get_state."
                )

            # Parse JSONB fields if they're strings
            state = (
                row["state"]
                if isinstance(row["state"], dict)
                else json.loads(row["state"])
            )
            provenance = (
                row["provenance"]
                if isinstance(row["provenance"], dict)
                else json.loads(row["provenance"])
            )

            # Construct ModelWorkflowState from database row
            return ModelWorkflowState(
                workflow_key=row["workflow_key"],
                version=row["version"],
                state=state,
                updated_at=row["updated_at"],
                schema_version=row["schema_version"],
                provenance=provenance,
            )

        except ValueError as e:
            # Re-raise validation errors
            get_state_errors.inc()
            self._metrics_get_state_errors += 1
            raise
        except RuntimeError as e:
            # Re-raise not found errors
            raise
        except Exception as e:
            get_state_errors.inc()
            self._metrics_get_state_errors += 1
            logger.error(
                f"Failed to get workflow state for '{workflow_key}': {e}",
                exc_info=True,
                extra={"workflow_key": workflow_key},
            )
            raise RuntimeError(
                f"Database error retrieving workflow state: {e!s}"
            ) from e

    async def try_commit(
        self,
        workflow_key: str,
        expected_version: int,
        state_prime: dict[str, Any],
        provenance: dict[str, Any],
    ) -> Union[EventStateCommitted, EventStateConflict]:
        """
        Attempt to commit new state with optimistic concurrency control.

        This method implements optimistic locking using version numbers:
        1. UPDATE row WHERE workflow_key = $1 AND version = $2
        2. If updated (version matches) → increment version, emit EventStateCommitted
        3. If not updated (version mismatch) → emit EventStateConflict
        4. Publish event to Kafka for downstream consumption

        The UPDATE query is atomic and prevents race conditions via PostgreSQL's
        row-level locking. Multiple concurrent commits to the same workflow_key
        will serialize at the database level, with only one succeeding per version.

        Performance: Typically < 10ms for the UPDATE, < 20ms including Kafka.

        Args:
            workflow_key: Human-readable workflow identifier
            expected_version: Expected current version (for optimistic lock)
            state_prime: New state to commit (must be valid JSONB dict)
            provenance: Provenance metadata (must contain 'effect_id', 'timestamp')

        Returns:
            EventStateCommitted: On success (version matched, state updated)
            EventStateConflict: On conflict (version mismatch, concurrent update)

        Raises:
            ValueError: If inputs are invalid (empty workflow_key, version < 1, etc.)
            Exception: For database connection or query errors

        Example:
            >>> result = await service.try_commit(
            ...     workflow_key="workflow-123",
            ...     expected_version=1,
            ...     state_prime={"items": [1], "count": 1},
            ...     provenance={
            ...         "effect_id": "effect-456",
            ...         "timestamp": datetime.now(UTC).isoformat()
            ...     }
            ... )
            >>> if isinstance(result, EventStateCommitted):
            ...     print(f"Committed version {result.new_version}")
            ... else:
            ...     print(f"Conflict: expected v{result.expected_version}, got v{result.actual_version}")
        """
        # Input validation
        if not workflow_key or not workflow_key.strip():
            raise ValueError("workflow_key must be non-empty string")
        if expected_version < 1:
            raise ValueError(
                f"expected_version must be >= 1, got {expected_version}. "
                "Version numbers start at 1."
            )
        if not isinstance(state_prime, dict):
            raise ValueError(
                f"state_prime must be dict, got {type(state_prime).__name__}"
            )
        if not state_prime:
            raise ValueError("state_prime cannot be empty dict")
        if not isinstance(provenance, dict):
            raise ValueError(
                f"provenance must be dict, got {type(provenance).__name__}"
            )

        # Validate provenance required fields
        required_provenance_fields = ["effect_id", "timestamp"]
        missing_fields = [
            field for field in required_provenance_fields if field not in provenance
        ]
        if missing_fields:
            raise ValueError(
                f"provenance missing required fields: {missing_fields}. "
                f"Required: {required_provenance_fields}"
            )

        # Optimistic concurrency control UPDATE query
        # This query atomically:
        # 1. Checks that version = expected_version (optimistic lock)
        # 2. Updates state, increments version, updates timestamp, sets provenance
        # 3. Returns updated row if successful (RETURNING clause)
        update_query = """
            UPDATE workflow_state
            SET
                state = $1::jsonb,
                version = version + 1,
                updated_at = NOW(),
                provenance = $2::jsonb
            WHERE workflow_key = $3 AND version = $4
            RETURNING workflow_key, version, state, updated_at, schema_version, provenance
        """

        # Track commit latency
        start_time = time.time()

        try:
            # Execute atomic UPDATE with optimistic lock
            # Convert dicts to JSON strings for asyncpg
            row = await self.postgres_client.fetch_one(
                update_query,
                json.dumps(state_prime),
                json.dumps(provenance),
                workflow_key,
                expected_version,
            )

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            commit_latency_ms.observe(latency_ms)

            # Case 1: UPDATE succeeded (version matched)
            if row:
                new_version = row["version"]
                state_commits_total.labels(workflow_key=workflow_key).inc()
                self._metrics_commits_total += 1

                # Parse JSONB fields if they're strings
                state_snapshot = (
                    row["state"]
                    if isinstance(row["state"], dict)
                    else json.loads(row["state"])
                )
                provenance_result = (
                    row["provenance"]
                    if isinstance(row["provenance"], dict)
                    else json.loads(row["provenance"])
                )

                # Create success event
                event = EventStateCommitted(
                    workflow_key=workflow_key,
                    new_version=new_version,
                    state_snapshot=state_snapshot,
                    provenance=provenance_result,
                )

                # Publish event to Kafka (async, non-blocking)
                await self._publish_event(self._topic_state_committed, event)

                logger.info(
                    f"Successfully committed state for '{workflow_key}': "
                    f"v{expected_version} → v{new_version}",
                    extra={
                        "workflow_key": workflow_key,
                        "expected_version": expected_version,
                        "new_version": new_version,
                        "correlation_id": event.correlation_id,
                        "latency_ms": round(latency_ms, 2),
                    },
                )

                return event

            # Case 2: UPDATE failed (version mismatch or row not found)
            else:
                state_conflicts_total.labels(workflow_key=workflow_key).inc()
                self._metrics_conflicts_total += 1

                # Query current version to provide helpful conflict information
                current_state = await self.get_state(workflow_key)
                actual_version = current_state.version

                # Create conflict event
                event = EventStateConflict(
                    workflow_key=workflow_key,
                    expected_version=expected_version,
                    actual_version=actual_version,
                    reason=(
                        "concurrent_modification"
                        if actual_version > expected_version
                        else "version_mismatch"
                    ),
                )

                # Publish event to Kafka (async, non-blocking)
                await self._publish_event(self._topic_state_conflicts, event)

                logger.warning(
                    f"Version conflict for '{workflow_key}': "
                    f"expected v{expected_version}, actual v{actual_version}",
                    extra={
                        "workflow_key": workflow_key,
                        "expected_version": expected_version,
                        "actual_version": actual_version,
                        "correlation_id": event.correlation_id,
                    },
                )

                return event

        except ValueError as e:
            # Re-raise validation errors
            raise
        except Exception as e:
            logger.error(
                f"Failed to commit state for '{workflow_key}': {e}",
                exc_info=True,
                extra={
                    "workflow_key": workflow_key,
                    "expected_version": expected_version,
                },
            )
            raise RuntimeError(f"Database error during state commit: {e!s}") from e

    async def _publish_event(
        self,
        topic: str,
        event: Union[EventStateCommitted, EventStateConflict],
    ) -> None:
        """
        Publish event to Kafka topic.

        This method handles async event publishing with error handling.
        Events are published asynchronously and failures are logged but
        do not block the state commit operation.

        Args:
            topic: Kafka topic name
            event: Event to publish (StateCommitted or StateConflict)
        """
        if not self.kafka_client:
            logger.debug(
                f"Kafka client not configured, skipping event publish to '{topic}'",
                extra={"topic": topic, "event_type": event.event_type},
            )
            return

        try:
            # Convert Pydantic model to dict for Kafka serialization
            event_dict = event.model_dump()

            # Publish to Kafka (aiokafka handles async I/O)
            await self.kafka_client.publish_event(
                topic=topic,
                event=event_dict,
                key=event.workflow_key,  # Partition by workflow_key for ordering
            )

            logger.debug(
                f"Published {event.event_type} event to '{topic}' for workflow '{event.workflow_key}'",
                extra={
                    "topic": topic,
                    "event_type": event.event_type,
                    "workflow_key": event.workflow_key,
                    "correlation_id": event.correlation_id,
                },
            )

        except Exception as e:
            # Log error but don't fail the commit
            logger.error(
                f"Failed to publish {event.event_type} event to '{topic}': {e}",
                exc_info=True,
                extra={
                    "topic": topic,
                    "event_type": event.event_type,
                    "workflow_key": event.workflow_key,
                },
            )

    def get_metrics(self) -> dict[str, int]:
        """
        Get service metrics for monitoring and testing.

        Returns current instance-level metric values. These metrics are also
        exported to Prometheus for production monitoring.

        Returns:
            Dictionary with current metric counts

        Example:
            >>> metrics = service.get_metrics()
            >>> print(f"Commits: {metrics['canonical_commits_total']}")
            Commits: 42
        """
        return {
            "canonical_commits_total": self._metrics_commits_total,
            "canonical_conflicts_total": self._metrics_conflicts_total,
            "canonical_get_state_total": self._metrics_get_state_total,
            "canonical_get_state_errors": self._metrics_get_state_errors,
        }
