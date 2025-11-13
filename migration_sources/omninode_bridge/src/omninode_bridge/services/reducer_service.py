"""
ReducerService - Service Wrapper for Pure Reducer with Conflict Resolution.

Wraps the pure reducer and handles all I/O concerns:
- State management (read from canonical store)
- Conflict resolution with jittered backoff
- Action deduplication for idempotency
- Event coordination via Kafka
- Metrics tracking

ONEX v2.0 Compliance:
- Suffix-based naming: ReducerService
- Strong typing with Pydantic models
- Comprehensive error handling
- Event-driven architecture

Pure Reducer Refactor - Wave 3, Workstream 3B
Reference: docs/planning/PURE_REDUCER_REFACTOR_PLAN.md

Example:
    >>> from uuid import uuid4
    >>> action = ModelAction(
    ...     action_id=uuid4(),
    ...     workflow_key="workflow-123",
    ...     epoch=1,
    ...     lease_id=uuid4(),
    ...     payload={"operation": "add_stamp"}
    ... )
    >>> service = ReducerService(
    ...     reducer=reducer_node,
    ...     canonical_store=canonical_store,
    ...     projection_store=projection_store,
    ...     action_dedup=action_dedup,
    ...     kafka_client=kafka_client
    ... )
    >>> await service.handle_action(action)
"""

import asyncio
import hashlib
import json
import logging
import random
from datetime import UTC, datetime
from typing import Any, Optional
from uuid import uuid4

from prometheus_client import Counter, Histogram
from pydantic import BaseModel, Field

from omninode_bridge.infrastructure.entities.model_action import ModelAction
from omninode_bridge.services.action_dedup import ActionDedupService
from omninode_bridge.services.canonical_store import (
    CanonicalStoreService,
    EventStateCommitted,
    EventStateConflict,
)
from omninode_bridge.services.kafka_client import KafkaClient
from omninode_bridge.services.projection_store import ProjectionStoreService

logger = logging.getLogger(__name__)

# Prometheus Metrics
reducer_successful_actions_total = Counter(
    "reducer_successful_actions_total",
    "Total number of successfully processed actions",
    ["workflow_key"],
)

reducer_failed_actions_total = Counter(
    "reducer_failed_actions_total",
    "Total number of failed actions (max retries exceeded)",
    ["workflow_key"],
)

reducer_duplicate_actions_skipped_total = Counter(
    "reducer_duplicate_actions_skipped_total",
    "Total number of duplicate actions skipped via deduplication",
)

reducer_conflict_attempts_total = Counter(
    "reducer_conflict_attempts_total",
    "Total number of conflict retry attempts",
    ["workflow_key"],
)

reducer_backoff_ms_histogram = Histogram(
    "reducer_backoff_ms",
    "Backoff delay distribution in milliseconds",
    buckets=[10, 20, 40, 80, 160, 250, 500, 1000],
)

reducer_gave_up_total = Counter(
    "reducer_gave_up_total",
    "Total number of times reducer gave up after max retries",
    ["workflow_key"],
)


# ============================================================================
# Metrics Model
# ============================================================================


class ReducerServiceMetrics(BaseModel):
    """
    Metrics for ReducerService operations.

    Tracks success rates, conflicts, deduplication hits, and backoff performance
    for monitoring and optimization.
    """

    successful_actions: int = Field(
        default=0,
        description="Number of successfully processed actions",
        ge=0,
    )
    failed_actions: int = Field(
        default=0,
        description="Number of failed actions (max retries exceeded)",
        ge=0,
    )
    duplicate_actions_skipped: int = Field(
        default=0,
        description="Number of duplicate actions skipped via deduplication",
        ge=0,
    )
    conflict_attempts_total: int = Field(
        default=0,
        description="Total number of conflict retry attempts",
        ge=0,
    )
    total_backoff_time_ms: float = Field(
        default=0.0,
        description="Total time spent in backoff delays (milliseconds)",
        ge=0.0,
    )

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        total = self.successful_actions + self.failed_actions
        if total == 0:
            return 0.0
        return (self.successful_actions / total) * 100.0

    @property
    def avg_conflicts_per_action(self) -> float:
        """Calculate average conflicts per successful action."""
        if self.successful_actions == 0:
            return 0.0
        return self.conflict_attempts_total / self.successful_actions

    @property
    def dedup_hit_rate(self) -> float:
        """Calculate deduplication hit rate as percentage."""
        total_checks = (
            self.successful_actions
            + self.failed_actions
            + self.duplicate_actions_skipped
        )
        if total_checks == 0:
            return 0.0
        return (self.duplicate_actions_skipped / total_checks) * 100.0


# ============================================================================
# Reducer Service Implementation
# ============================================================================


class ReducerService:
    """
    Service wrapper for pure reducer with conflict resolution and retry logic.

    This service wraps a pure reducer and handles all I/O concerns:

    1. **Action Deduplication**: Prevents duplicate processing of actions
    2. **State Reading**: Reads current state from canonical store
    3. **Pure Reducer Invocation**: Calls reducer with current state
    4. **Conflict Resolution**: Retries on version conflicts with jittered backoff
    5. **Event Publishing**: Publishes intents to Kafka for downstream processing
    6. **Metrics Tracking**: Tracks success rates, conflicts, and performance

    Performance Characteristics:
    - Deduplication check: <5ms (indexed lookup)
    - State read: <5ms (canonical store)
    - Reducer invocation: <50ms (pure function, no I/O)
    - Commit attempt: <10ms (optimistic lock UPDATE)
    - Total latency (no conflicts): <70ms p95
    - Total latency (with conflicts): <150ms p95 (with backoff)

    Conflict Resolution:
    - Uses decorrelated jitter backoff (not exponential)
    - Default: 3 max attempts with 10-250ms backoff range
    - Prevents thundering herd via randomized backoff
    - Escalates to "ReducerGaveUp" event after max retries

    Thread Safety:
    - Designed for asyncio (not thread-safe)
    - Safe for concurrent use with different workflow_keys
    - Uses optimistic locking for same workflow_key updates

    Attributes:
        reducer: Pure reducer node (callable with state → intents)
        canonical_store: Canonical state store service
        projection_store: Projection store service (optional)
        action_dedup: Action deduplication service
        kafka_client: Kafka client for event publishing (optional)
        max_attempts: Maximum retry attempts on conflicts (default: 3)
        backoff_base_ms: Base backoff delay in milliseconds (default: 10)
        backoff_cap_ms: Maximum backoff delay in milliseconds (default: 250)
        metrics: Service metrics tracker
    """

    def __init__(
        self,
        reducer: Any,  # Pure reducer from workstream 3A
        canonical_store: CanonicalStoreService,
        projection_store: Optional[ProjectionStoreService] = None,
        action_dedup: Optional[ActionDedupService] = None,
        kafka_client: Optional[KafkaClient] = None,
        max_attempts: int = 3,
        backoff_base_ms: int = 10,
        backoff_cap_ms: int = 250,
    ):
        """
        Initialize ReducerService.

        Args:
            reducer: Pure reducer node (NodeBridgeReducer instance)
            canonical_store: Canonical state store service
            projection_store: Optional projection store service
            action_dedup: Optional action deduplication service
            kafka_client: Optional Kafka client for event publishing
            max_attempts: Maximum retry attempts on conflicts (default: 3)
            backoff_base_ms: Base backoff delay in milliseconds (default: 10ms)
            backoff_cap_ms: Maximum backoff delay in milliseconds (default: 250ms)
        """
        self.reducer = reducer
        self.canonical_store = canonical_store
        self.projection_store = projection_store
        self.action_dedup = action_dedup
        self.kafka_client = kafka_client

        # Retry configuration
        self.max_attempts = max_attempts
        self.backoff_base_ms = backoff_base_ms
        self.backoff_cap_ms = backoff_cap_ms

        # Metrics tracking
        self.metrics = ReducerServiceMetrics()

        # Kafka topics
        self._topic_reducer_gave_up = "omninode_bridge_reducer_gave_up_v1"
        self._topic_intents = "omninode_bridge_intents_v1"

    async def handle_action(self, action: ModelAction) -> None:
        """
        Handle action with deduplication, retry logic, and conflict resolution.

        This is the main entry point for processing actions. The workflow is:

        1. Check deduplication: Skip if already processed
        2. Retry loop (max_attempts):
           a. Read current state from canonical store
           b. Call pure reducer to get new state and intents
           c. Try to commit new state with optimistic lock
           d. If success: Publish intents and record action
           e. If conflict: Retry with jittered backoff
        3. If max retries exceeded: Publish ReducerGaveUp event

        Args:
            action: Action to process

        Raises:
            ValueError: If action is invalid
            RuntimeError: If canonical store is unavailable

        Example:
            >>> action = ModelAction(
            ...     action_id=uuid4(),
            ...     workflow_key="workflow-123",
            ...     epoch=1,
            ...     lease_id=uuid4(),
            ...     payload={"operation": "add_stamp"}
            ... )
            >>> await service.handle_action(action)
        """
        # Validate action
        if not action.workflow_key or not action.workflow_key.strip():
            raise ValueError("action.workflow_key must be non-empty string")

        # 1. Deduplication check
        if self.action_dedup:
            should_process = await self.action_dedup.should_process(
                action.workflow_key, action.action_id
            )
            if not should_process:
                logger.info(
                    f"Skipping duplicate action {action.action_id} for workflow {action.workflow_key}"
                )
                self.metrics.duplicate_actions_skipped += 1
                reducer_duplicate_actions_skipped_total.inc()
                return

        # 2. Retry loop with backoff
        for attempt in range(1, self.max_attempts + 1):
            try:
                # Read current state (canonical for consistency)
                state_record = await self.canonical_store.get_state(action.workflow_key)

                logger.debug(
                    f"Read state for {action.workflow_key}: version {state_record.version}"
                )

                # Call pure reducer
                # NOTE: This is compatible with both current and future pure reducer
                # Current reducer returns ModelReducerOutputState
                # Future pure reducer will return ModelReducerOutput with intents[]
                reducer_result = await self._call_reducer(state_record, action)

                # Extract state_prime from reducer result
                # This will be adapted based on reducer type
                state_prime = self._extract_state_prime(reducer_result)

                logger.debug(
                    f"Reducer produced new state for {action.workflow_key}: "
                    f"{len(str(state_prime))} bytes"
                )

                # Try to commit with optimistic lock
                commit_result = await self.canonical_store.try_commit(
                    workflow_key=action.workflow_key,
                    expected_version=state_record.version,
                    state_prime=state_prime,
                    provenance={
                        "effect_id": str(uuid4()),
                        "timestamp": datetime.now(UTC).isoformat(),
                        "action_id": str(action.action_id),
                        "epoch": action.epoch,
                        "lease_id": str(action.lease_id),
                    },
                )

                # Case 1: SUCCESS - Commit succeeded
                if isinstance(commit_result, EventStateCommitted):
                    logger.info(
                        f"Successfully committed state for {action.workflow_key}: "
                        f"v{state_record.version} → v{commit_result.new_version} "
                        f"(attempt {attempt}/{self.max_attempts})"
                    )

                    # Publish other intents (if reducer returned any)
                    await self._publish_intents(reducer_result, action)

                    # Remember action for dedup
                    if self.action_dedup:
                        result_hash = self._hash_result(state_prime)
                        await self.action_dedup.record_processed(
                            action.workflow_key,
                            action.action_id,
                            result_hash,
                            ttl_hours=6,
                        )

                    # Update metrics
                    self.metrics.successful_actions += 1
                    reducer_successful_actions_total.labels(
                        workflow_key=action.workflow_key
                    ).inc()
                    return

                # Case 2: CONFLICT - Version mismatch, retry with backoff
                elif isinstance(commit_result, EventStateConflict):
                    self.metrics.conflict_attempts_total += 1
                    reducer_conflict_attempts_total.labels(
                        workflow_key=action.workflow_key
                    ).inc()

                    # Calculate backoff delay with jitter
                    backoff_ms = self._backoff_delay(attempt)
                    self.metrics.total_backoff_time_ms += backoff_ms
                    reducer_backoff_ms_histogram.observe(backoff_ms)

                    logger.warning(
                        f"Version conflict on {action.workflow_key}: "
                        f"expected v{commit_result.expected_version}, "
                        f"actual v{commit_result.actual_version}. "
                        f"Attempt {attempt}/{self.max_attempts}, "
                        f"backoff {backoff_ms}ms"
                    )

                    # Last attempt? Don't sleep, just fail
                    if attempt < self.max_attempts:
                        await asyncio.sleep(backoff_ms / 1000.0)
                        continue
                    else:
                        logger.error(
                            f"Max retries ({self.max_attempts}) exceeded for {action.workflow_key}"
                        )
                        break

            except Exception as e:
                logger.error(
                    f"Error in attempt {attempt}/{self.max_attempts} for {action.workflow_key}: {e}",
                    exc_info=True,
                )

                # Last attempt? Fail immediately
                if attempt >= self.max_attempts:
                    break

                # Calculate backoff for errors too
                backoff_ms = self._backoff_delay(attempt)
                self.metrics.total_backoff_time_ms += backoff_ms
                reducer_backoff_ms_histogram.observe(backoff_ms)

                logger.warning(f"Retrying after error with backoff {backoff_ms}ms")
                await asyncio.sleep(backoff_ms / 1000.0)
                continue

        # 3. Max retries exceeded - escalate via ReducerGaveUp event
        logger.error(
            f"Reducer gave up on {action.workflow_key} after {self.max_attempts} attempts"
        )

        await self._publish_reducer_gave_up(action)

        # Update metrics
        self.metrics.failed_actions += 1
        reducer_failed_actions_total.labels(workflow_key=action.workflow_key).inc()
        reducer_gave_up_total.labels(workflow_key=action.workflow_key).inc()

    async def _call_reducer(self, state_record: Any, action: ModelAction) -> Any:
        """
        Call reducer with current state and action.

        This method is compatible with both:
        - Current reducer: execute_reduction(contract) → ModelReducerOutputState
        - Future pure reducer: process(input) → ModelReducerOutput with intents

        Args:
            state_record: Current workflow state from canonical store
            action: Action to process

        Returns:
            Reducer result (type depends on reducer implementation)
        """
        # Check if reducer has process() method (future pure reducer)
        if hasattr(self.reducer, "process"):
            # Future pure reducer interface
            # Build reducer input
            reducer_input = {
                "data": state_record.state,
                "metadata": {
                    "action": action.model_dump(),
                    "version": state_record.version,
                },
            }

            # Call process() method
            result = await self.reducer.process(reducer_input)
            return result

        # Current reducer interface (execute_reduction)
        elif hasattr(self.reducer, "execute_reduction"):
            # Import dynamically to avoid import errors in test environments
            try:
                from omnibase_core.enums import EnumNodeType
                from omnibase_core.models.contracts.model_contract_reducer import (
                    ModelContractReducer,
                )
                from omnibase_core.primitives.model_semver import ModelSemVer

                # Build contract for current reducer
                contract = ModelContractReducer(
                    name="reducer_action",
                    version=ModelSemVer(major=1, minor=0, patch=0),
                    description=f"Process action {action.action_id}",
                    node_type=EnumNodeType.REDUCER,
                    input_model="ModelReducerInputState",
                    output_model="ModelReducerOutputState",
                    input_state={
                        "items": [action.payload],  # Wrap payload in items array
                        "current_version": state_record.version,
                        "workflow_key": action.workflow_key,
                    },
                )

                # Call execute_reduction() method
                result = await self.reducer.execute_reduction(contract)
                return result

            except ImportError:
                # If omnibase_core is not available, call without contract wrapping
                # This allows testing with simple mocks
                result = await self.reducer.execute_reduction(None)
                return result

        else:
            raise TypeError(
                f"Reducer must have process() or execute_reduction() method, "
                f"got {type(self.reducer)}"
            )

    def _extract_state_prime(self, reducer_result: Any) -> dict[str, Any]:
        """
        Extract state_prime from reducer result.

        Supports both:
        - Future pure reducer: result.intents with PersistState intent
        - Current reducer: result.aggregations dict

        Args:
            reducer_result: Result from reducer

        Returns:
            New state dict to commit

        Raises:
            ValueError: If no state can be extracted
        """
        # Future pure reducer: Extract from PersistState intent
        if hasattr(reducer_result, "intents"):
            persist_intent = next(
                (
                    intent
                    for intent in reducer_result.intents
                    if getattr(intent, "intent_type", None) == "PersistState"
                ),
                None,
            )
            if persist_intent:
                return persist_intent.payload.get("aggregated_data", {})

        # Current reducer: Use aggregations dict
        if hasattr(reducer_result, "aggregations"):
            return {"aggregations": reducer_result.aggregations}

        # Fallback: Try to convert entire result to dict
        if hasattr(reducer_result, "model_dump"):
            return reducer_result.model_dump()

        raise ValueError(
            f"Cannot extract state_prime from reducer result type {type(reducer_result)}"
        )

    async def _publish_intents(self, reducer_result: Any, action: ModelAction) -> None:
        """
        Publish non-PersistState intents to Kafka.

        Future pure reducer will return multiple intents. We publish all
        intents except PersistState (which was already handled via commit).

        Args:
            reducer_result: Result from reducer
            action: Original action for context
        """
        if not self.kafka_client:
            logger.debug("Kafka client not available, skipping intent publishing")
            return

        # Future pure reducer: Publish all intents except PersistState
        if hasattr(reducer_result, "intents"):
            for intent in reducer_result.intents:
                intent_type = getattr(intent, "intent_type", None)

                # Skip PersistState (already handled via commit)
                if intent_type == "PersistState":
                    continue

                try:
                    await self.kafka_client.publish_event(
                        topic=self._topic_intents,
                        event={
                            "intent_type": intent_type,
                            "target": getattr(intent, "target", None),
                            "payload": getattr(intent, "payload", {}),
                            "source_action_id": str(action.action_id),
                            "workflow_key": action.workflow_key,
                            "timestamp": datetime.now(UTC).isoformat(),
                        },
                        key=action.workflow_key,
                    )

                    logger.debug(
                        f"Published intent {intent_type} for workflow {action.workflow_key}"
                    )

                except Exception as e:
                    logger.warning(
                        f"Failed to publish intent {intent_type}: {e}",
                        exc_info=True,
                    )

    async def _publish_reducer_gave_up(self, action: ModelAction) -> None:
        """
        Publish ReducerGaveUp event to Kafka.

        This event signals that the reducer exhausted all retry attempts
        and escalates to orchestrator for policy decision.

        Args:
            action: Action that failed
        """
        if not self.kafka_client:
            logger.warning(
                f"Kafka client not available, cannot publish ReducerGaveUp for {action.workflow_key}"
            )
            return

        try:
            await self.kafka_client.publish_event(
                topic=self._topic_reducer_gave_up,
                event={
                    "workflow_key": action.workflow_key,
                    "action_id": str(action.action_id),
                    "attempts": self.max_attempts,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "reason": "max_retries_exceeded",
                },
                key=action.workflow_key,
            )

            logger.info(
                f"Published ReducerGaveUp event for workflow {action.workflow_key}"
            )

        except Exception as e:
            logger.error(
                f"Failed to publish ReducerGaveUp event: {e}",
                exc_info=True,
            )

    def _backoff_delay(self, attempt: int) -> int:
        """
        Calculate jittered backoff delay for retry attempt.

        Uses decorrelated jitter (not exponential backoff) to prevent
        thundering herd problem when multiple reducers retry simultaneously.

        Formula: random.randint(base_ms, min(cap_ms, base_ms * 2^attempt))

        Example progression (base=10ms, cap=250ms):
        - Attempt 1: random(10, 20)   → ~10-20ms
        - Attempt 2: random(10, 40)   → ~10-40ms
        - Attempt 3: random(10, 80)   → ~10-80ms
        - Attempt 4: random(10, 160)  → ~10-160ms
        - Attempt 5: random(10, 250)  → ~10-250ms (capped)

        Args:
            attempt: Current attempt number (1-based)

        Returns:
            Backoff delay in milliseconds

        Example:
            >>> service = ReducerService(...)
            >>> delay = service._backoff_delay(1)
            >>> assert 10 <= delay <= 20  # First attempt
            >>> delay = service._backoff_delay(3)
            >>> assert 10 <= delay <= 80  # Third attempt
        """
        max_ms = min(self.backoff_cap_ms, self.backoff_base_ms * (2**attempt))
        return random.randint(self.backoff_base_ms, max_ms)

    def _hash_result(self, result: dict[str, Any]) -> str:
        """
        Calculate SHA256 hash of result for deduplication.

        Uses canonical JSON serialization (sorted keys) for deterministic
        hashing across different Python processes.

        Args:
            result: Result dict to hash

        Returns:
            SHA256 hex string (64 characters)

        Example:
            >>> service = ReducerService(...)
            >>> hash1 = service._hash_result({"a": 1, "b": 2})
            >>> hash2 = service._hash_result({"b": 2, "a": 1})  # Different order
            >>> assert hash1 == hash2  # Same hash due to sorted keys
        """
        # Sort keys for deterministic hashing
        canonical_json = json.dumps(result, sort_keys=True)
        return hashlib.sha256(canonical_json.encode()).hexdigest()

    def get_metrics(self) -> ReducerServiceMetrics:
        """
        Get current service metrics.

        Returns:
            Current metrics snapshot

        Example:
            >>> metrics = service.get_metrics()
            >>> print(f"Success rate: {metrics.success_rate:.2f}%")
            >>> print(f"Avg conflicts: {metrics.avg_conflicts_per_action:.2f}")
            >>> print(f"Dedup hit rate: {metrics.dedup_hit_rate:.2f}%")
        """
        return self.metrics

    def reset_metrics(self) -> None:
        """
        Reset all metrics counters to zero.

        Useful for testing or periodic metric reporting.
        """
        self.metrics = ReducerServiceMetrics()
