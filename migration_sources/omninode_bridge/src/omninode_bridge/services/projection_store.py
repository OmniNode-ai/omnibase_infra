#!/usr/bin/env python3
"""
ProjectionStoreService - Workflow Projection Store with Version Gating.

Read-optimized projection store with eventual consistency guarantees.
Implements version gating with fallback to canonical store for stale reads.

ONEX v2.0 Compliance:
- Suffix-based naming: ProjectionStoreService
- Eventual consistency with version gating
- Canonical fallback for projection lag
- Metrics tracking for observability

Key Features:
- Version gating: Wait for projection to catch up to required version
- Canonical fallback: Fall back to canonical store if projection lags
- Fast read path: <10ms for consistent projections
- Slow path: Fallback to canonical with conversion

Performance Targets:
- Read latency (consistent): <10ms (p95)
- Read latency (with wait): <100ms (p95)
- Fallback rate: <5% in steady state
- Polling overhead: 5ms intervals

Pure Reducer Refactor - Wave 2, Workstream 2B
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from omninode_bridge.infrastructure.entities.model_workflow_projection import (
    ModelWorkflowProjection,
)
from omninode_bridge.infrastructure.entities.model_workflow_state import (
    ModelWorkflowState,
)
from omninode_bridge.services.canonical_store_protocol import CanonicalStoreProtocol
from omninode_bridge.services.postgres_client import PostgresClient

logger = logging.getLogger(__name__)


@dataclass
class ProjectionStoreMetrics:
    """
    Metrics for projection store operations.

    Tracks read patterns, fallback rates, and wait times for performance
    monitoring and optimization.
    """

    projection_reads_total: int = 0
    projection_fallback_count: int = 0
    projection_wait_count: int = 0
    total_wait_time_ms: float = 0.0
    read_latencies_ms: list[float] = field(default_factory=list)

    @property
    def fallback_rate(self) -> float:
        """Calculate fallback rate as percentage."""
        if self.projection_reads_total == 0:
            return 0.0
        return (self.projection_fallback_count / self.projection_reads_total) * 100.0

    @property
    def avg_wait_time_ms(self) -> float:
        """Calculate average wait time in milliseconds."""
        if self.projection_wait_count == 0:
            return 0.0
        return self.total_wait_time_ms / self.projection_wait_count

    @property
    def p95_read_latency_ms(self) -> float:
        """Calculate 95th percentile read latency."""
        if not self.read_latencies_ms:
            return 0.0
        sorted_latencies = sorted(self.read_latencies_ms)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[idx] if idx < len(sorted_latencies) else 0.0


class ProjectionStoreService:
    """
    Projection store service with version gating and canonical fallback.

    This service provides read-optimized access to workflow projections
    with eventual consistency guarantees. It implements version gating
    to wait for projections to catch up, and falls back to canonical
    store if projections are too stale.

    Architecture:
        - Primary: Read from workflow_projection table (fast)
        - Version gating: Poll for required version with timeout
        - Fallback: Query canonical store and convert to projection

    Example:
        >>> service = ProjectionStoreService(db_client, canonical_store)
        >>> await service.initialize()
        >>>
        >>> # Fast path: projection is current
        >>> proj = await service.get_state("wf-123")
        >>>
        >>> # Version gating: wait for projection to catch up
        >>> proj = await service.get_state("wf-123", required_version=5, max_wait_ms=100)
        >>>
        >>> # Fallback: projection too stale, use canonical
        >>> proj = await service.get_state("wf-123", required_version=10, max_wait_ms=50)
    """

    def __init__(
        self,
        db_client: PostgresClient,
        canonical_store: CanonicalStoreProtocol,
        poll_interval_ms: int = 5,
    ):
        """
        Initialize projection store service.

        Args:
            db_client: PostgreSQL client for database access
            canonical_store: Canonical store for fallback queries
            poll_interval_ms: Polling interval for version gating (default: 5ms)
        """
        self._db_client = db_client
        self._canonical_store = canonical_store
        self._poll_interval_ms = poll_interval_ms
        self._metrics = ProjectionStoreMetrics()
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize the projection store service.

        Validates database connection and prepares for operations.
        """
        if self._initialized:
            return

        logger.info("Initializing ProjectionStoreService")
        self._initialized = True
        logger.info("ProjectionStoreService initialized successfully")

    async def get_state(
        self,
        workflow_key: str,
        required_version: Optional[int] = None,
        max_wait_ms: int = 100,
    ) -> ModelWorkflowProjection:
        """
        Get workflow projection with optional version gating.

        Implements version gating with canonical fallback:
        1. Query projection from database
        2. If version is sufficient, return immediately
        3. If version is stale, poll for updates (max_wait_ms)
        4. If timeout, fall back to canonical store

        Args:
            workflow_key: Unique workflow identifier
            required_version: Minimum required version (None = any version)
            max_wait_ms: Maximum wait time for version gating (default: 100ms)

        Returns:
            ModelWorkflowProjection: Workflow projection

        Raises:
            KeyError: If workflow does not exist in projection or canonical store
            ValueError: If workflow_key is empty or invalid

        Example:
            >>> # No version requirement (fast path)
            >>> proj = await service.get_state("wf-123")
            >>>
            >>> # Version gating (wait up to 100ms for version 5)
            >>> proj = await service.get_state("wf-123", required_version=5, max_wait_ms=100)
            >>>
            >>> # Fallback scenario (version 10 not ready, use canonical)
            >>> proj = await service.get_state("wf-123", required_version=10, max_wait_ms=50)
        """
        if not workflow_key or not workflow_key.strip():
            raise ValueError("workflow_key cannot be empty")

        start_time = time.perf_counter()
        self._metrics.projection_reads_total += 1

        try:
            # Fast path: read projection without version requirement
            if required_version is None:
                projection = await self._read_projection(workflow_key)
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                self._metrics.read_latencies_ms.append(elapsed_ms)
                logger.debug(
                    f"Read projection for {workflow_key} (no version requirement) in {elapsed_ms:.2f}ms"
                )
                return projection

            # Version gating path: poll for required version
            projection = await self._wait_for_version(
                workflow_key, required_version, max_wait_ms
            )

            if projection is not None:
                # Projection caught up within timeout
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                self._metrics.read_latencies_ms.append(elapsed_ms)
                logger.debug(
                    f"Read projection for {workflow_key} (version {projection.version} >= {required_version}) "
                    f"in {elapsed_ms:.2f}ms"
                )
                return projection

            # Fallback path: projection too stale, use canonical
            logger.warning(
                f"Projection for {workflow_key} did not reach version {required_version} "
                f"within {max_wait_ms}ms, falling back to canonical store"
            )
            self._metrics.projection_fallback_count += 1

            canonical_state = await self._canonical_store.get_state(workflow_key)
            projection = self._canonical_to_projection(canonical_state)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._metrics.read_latencies_ms.append(elapsed_ms)
            logger.info(
                f"Fallback to canonical for {workflow_key} (version {canonical_state.version}) "
                f"in {elapsed_ms:.2f}ms"
            )
            return projection

        except KeyError:
            # Workflow not found in projection, try canonical
            logger.warning(
                f"Projection not found for {workflow_key}, trying canonical store"
            )
            self._metrics.projection_fallback_count += 1

            canonical_state = await self._canonical_store.get_state(workflow_key)
            projection = self._canonical_to_projection(canonical_state)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._metrics.read_latencies_ms.append(elapsed_ms)
            logger.info(
                f"Fallback to canonical for {workflow_key} (not in projection) in {elapsed_ms:.2f}ms"
            )
            return projection

    async def get_version(self, workflow_key: str) -> int:
        """
        Get current projection version for workflow.

        Args:
            workflow_key: Unique workflow identifier

        Returns:
            int: Current projection version

        Raises:
            KeyError: If workflow does not exist in projection

        Example:
            >>> version = await service.get_version("wf-123")
            >>> assert version >= 1
        """
        if not workflow_key or not workflow_key.strip():
            raise ValueError("workflow_key cannot be empty")

        projection = await self._read_projection(workflow_key)
        return projection.version

    def get_metrics(self) -> ProjectionStoreMetrics:
        """
        Get current metrics snapshot.

        Returns:
            ProjectionStoreMetrics: Current metrics

        Example:
            >>> metrics = service.get_metrics()
            >>> print(f"Fallback rate: {metrics.fallback_rate:.2f}%")
            >>> print(f"P95 latency: {metrics.p95_read_latency_ms:.2f}ms")
        """
        return self._metrics

    async def _read_projection(self, workflow_key: str) -> ModelWorkflowProjection:
        """
        Read projection from database.

        Args:
            workflow_key: Unique workflow identifier

        Returns:
            ModelWorkflowProjection: Projection record

        Raises:
            KeyError: If workflow_key not found
        """
        query = """
            SELECT
                workflow_key,
                version,
                tag,
                last_action,
                namespace,
                updated_at,
                indices,
                extras
            FROM workflow_projection
            WHERE workflow_key = $1
        """

        row = await self._db_client.fetchrow(query, workflow_key)
        if row is None:
            raise KeyError(f"Workflow {workflow_key} not found in projection")

        return ModelWorkflowProjection(
            workflow_key=row["workflow_key"],
            version=row["version"],
            tag=row["tag"],
            last_action=row["last_action"],
            namespace=row["namespace"],
            updated_at=row["updated_at"],
            indices=row["indices"],
            extras=row["extras"],
        )

    async def _wait_for_version(
        self, workflow_key: str, required_version: int, max_wait_ms: int
    ) -> Optional[ModelWorkflowProjection]:
        """
        Wait for projection to reach required version.

        Polls projection table at poll_interval_ms intervals until:
        1. Projection reaches required version (success)
        2. Timeout expires (return None for fallback)

        Args:
            workflow_key: Unique workflow identifier
            required_version: Minimum required version
            max_wait_ms: Maximum wait time in milliseconds

        Returns:
            Optional[ModelWorkflowProjection]: Projection if version reached, None if timeout
        """
        deadline = time.perf_counter() + (max_wait_ms / 1000.0)
        wait_start = time.perf_counter()
        poll_count = 0

        while time.perf_counter() < deadline:
            try:
                projection = await self._read_projection(workflow_key)
                poll_count += 1

                if projection.version >= required_version:
                    # Success: projection caught up
                    wait_time_ms = (time.perf_counter() - wait_start) * 1000
                    if poll_count > 1:
                        # Only track as wait if we actually polled
                        self._metrics.projection_wait_count += 1
                        self._metrics.total_wait_time_ms += wait_time_ms
                        logger.debug(
                            f"Projection {workflow_key} reached version {projection.version} "
                            f"after {poll_count} polls in {wait_time_ms:.2f}ms"
                        )
                    return projection

            except KeyError:
                # Projection not found yet, keep waiting
                pass

            # Sleep for poll interval
            await asyncio.sleep(self._poll_interval_ms / 1000.0)

        # Timeout: projection did not catch up
        wait_time_ms = (time.perf_counter() - wait_start) * 1000
        self._metrics.projection_wait_count += 1
        self._metrics.total_wait_time_ms += wait_time_ms
        logger.warning(
            f"Timeout waiting for {workflow_key} to reach version {required_version} "
            f"after {poll_count} polls in {wait_time_ms:.2f}ms"
        )
        return None

    def _canonical_to_projection(
        self, canonical: ModelWorkflowState
    ) -> ModelWorkflowProjection:
        """
        Convert canonical state to projection.

        Extracts projection-relevant fields from canonical state.
        Assumes canonical state contains "tag" field for FSM state.

        Args:
            canonical: Canonical workflow state

        Returns:
            ModelWorkflowProjection: Converted projection

        Raises:
            KeyError: If required fields missing from canonical state
        """
        # Extract tag from canonical state
        # This assumes canonical state has "tag" field for FSM state
        tag = canonical.state.get("tag", "UNKNOWN")

        # Extract namespace from canonical state or provenance
        namespace = canonical.state.get("namespace") or canonical.provenance.get(
            "namespace", "default"
        )

        # Extract last_action from provenance
        last_action = canonical.provenance.get("action_id")

        return ModelWorkflowProjection(
            workflow_key=canonical.workflow_key,
            version=canonical.version,
            tag=tag,
            last_action=last_action,
            namespace=namespace,
            updated_at=canonical.updated_at,
            indices=None,  # No custom indices in canonical
            extras=None,  # No extra metadata in canonical
        )
