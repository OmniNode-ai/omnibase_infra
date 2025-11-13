"""
Event tracer for debugging code generation sessions.

This module provides event tracing capabilities for autonomous code generation
workflows, enabling developers to trace event flows across request/response
topics and understand the complete event chain for a session.

Key Features:
    - Session event tracing with timing analysis
    - Performance metrics calculation
    - Correlation ID-based event discovery
    - Comprehensive debugging support

Event Flow Tracing:
    omniclaude → Request Topics → omniarchon (intelligence processing)
    omniarchon → Response Topics → omniclaude (results)
    Both → Status Topics → monitoring/debugging

Usage:
    ```python
    from omninode_bridge.infrastructure.postgres_connection_manager import (
        PostgresConnectionManager,
        ModelPostgresConfig
    )
    from omninode_bridge.dashboard.codegen_event_tracer import CodegenEventTracer

    # Initialize database connection
    config = ModelPostgresConfig.from_environment()
    db_manager = PostgresConnectionManager(config)
    await db_manager.initialize()

    # Create event tracer
    tracer = CodegenEventTracer(db_manager)

    # Trace session events
    trace = await tracer.trace_session_events(session_id, time_range_hours=24)
    print(f"Session {session_id}: {trace.total_events} events")

    # Get performance metrics
    metrics = await tracer.get_session_metrics(session_id)
    print(f"Average response time: {metrics.avg_response_time_ms}ms")

    # Find correlated events
    events = await tracer.find_correlated_events(correlation_id)
    print(f"Found {len(events)} correlated events")
    ```

Database Schema:
    The tracer expects event logs stored in PostgreSQL with schema:
    - event_logs table with columns:
        - event_id (UUID)
        - session_id (UUID)
        - correlation_id (UUID)
        - event_type (TEXT): 'request', 'response', 'status', 'error'
        - topic (TEXT): Kafka topic name
        - timestamp (TIMESTAMPTZ)
        - status (TEXT): 'sent', 'received', 'failed', 'processing'
        - payload (JSONB): Event data
        - processing_time_ms (INTEGER): Response time
        - metadata (JSONB): Additional context
"""

import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Literal
from uuid import UUID

from omnibase_core import EnumCoreErrorCode, ModelOnexError

from omninode_bridge.dashboard.models import (
    ModelCorrelatedEvent,
    ModelEventTrace,
    ModelSessionMetrics,
)
from omninode_bridge.protocols import SupportsQuery

# Type alias for compatibility
OnexError = ModelOnexError

logger = logging.getLogger(__name__)


class CodegenEventTracer:
    """
    Trace event flow for code generation sessions.

    Provides comprehensive event tracing capabilities for debugging autonomous
    code generation workflows. Tracks events across request/response topics,
    calculates performance metrics, and enables correlation-based event discovery.

    Attributes:
        db: PostgresConnectionManager instance for database operations
        logger: Logger instance for tracing operations

    Methods:
        trace_session_events: Get complete event trace for a session
        get_session_metrics: Calculate performance metrics for a session
        find_correlated_events: Find all events with same correlation ID
    """

    def __init__(self, db_connection: SupportsQuery):
        """
        Initialize CodegenEventTracer with database connection.

        Args:
            db_connection: Database connection implementing SupportsQuery protocol
                for async query execution

        Example:
            ```python
            from omninode_bridge.infrastructure.postgres_connection_manager import (
                PostgresConnectionManager,
                ModelPostgresConfig
            )

            config = ModelPostgresConfig.from_environment()
            db_manager = PostgresConnectionManager(config)
            await db_manager.initialize()

            tracer = CodegenEventTracer(db_manager)
            ```
        """
        self.db = db_connection
        self.logger = logging.getLogger(__name__)

    async def trace_session_events(
        self, session_id: UUID, time_range_hours: int = 24
    ) -> ModelEventTrace:
        """
        Get complete event trace for a generation session.

        Retrieves all events for a specific code generation session within the
        specified time range, ordered chronologically. Calculates session-level
        metrics including total duration and completion status.

        Args:
            session_id: Code generation session ID to trace
            time_range_hours: How far back to look for events (default: 24 hours)

        Returns:
            ModelEventTrace with timing and status information containing:
                - session_id: Session UUID
                - events: List of ModelCorrelatedEvent objects
                - total_events: Count of events found
                - session_duration_ms: Duration from first to last event
                - status: "completed", "in_progress", "failed", or "unknown"
                - start_time: Timestamp of first event (or None)
                - end_time: Timestamp of last event (or None)
                - time_range_hours: Search range in hours

        Raises:
            OnexError: If database query fails or session_id is invalid

        Example:
            ```python
            session_id = UUID("123e4567-e89b-12d3-a456-426614174000")
            trace = await tracer.trace_session_events(session_id, time_range_hours=24)

            print(f"Session: {trace.session_id}")
            print(f"Total events: {trace.total_events}")
            print(f"Duration: {trace.session_duration_ms}ms")
            print(f"Status: {trace.status}")

            for event in trace.events:
                print(f"  [{event.timestamp}] {event.event_type}: {event.topic}")
            ```

        Event Types:
            - "request": Events sent to request topics (omniclaude → omniarchon)
            - "response": Events sent to response topics (omniarchon → omniclaude)
            - "status": Status update events
            - "error": Error/failure events

        Status Values:
            - "completed": All events processed successfully
            - "in_progress": Session still active
            - "failed": Session encountered errors
            - "unknown": No events found or indeterminate status
        """
        self.logger.info(
            f"Tracing events for session {session_id} (last {time_range_hours} hours)"
        )

        try:
            # Calculate time range cutoff
            cutoff_time = datetime.now(UTC) - timedelta(hours=time_range_hours)

            # Query event_logs table for session events
            query = """
            SELECT
                event_id, event_type, topic, timestamp,
                correlation_id, status, processing_time_ms,
                payload, metadata, session_id
            FROM event_logs
            WHERE session_id = $1 AND timestamp >= $2
            ORDER BY timestamp ASC
            """
            rows = await self.db.execute_query(query, session_id, cutoff_time)

            # Handle empty result set
            if not rows:
                self.logger.warning(
                    f"No events found for session {session_id} in last {time_range_hours} hours"
                )
                return ModelEventTrace(
                    session_id=session_id,
                    events=[],
                    total_events=0,
                    session_duration_ms=0,
                    status="unknown",
                    start_time=None,
                    end_time=None,
                    time_range_hours=time_range_hours,
                )

            # Convert rows to ModelCorrelatedEvent objects
            events = []
            for row in rows:
                # Parse JSONB fields if they're strings (testcontainers compatibility)
                # Handle payload with edge cases
                payload = row["payload"]
                if isinstance(payload, str):
                    payload = json.loads(payload) if payload else {}
                elif payload is None:
                    payload = {}

                # Handle metadata with edge cases
                metadata = row["metadata"]
                if isinstance(metadata, str):
                    metadata = json.loads(metadata) if metadata else {}
                elif metadata is None:
                    metadata = {}

                event = ModelCorrelatedEvent(
                    event_id=row["event_id"],
                    session_id=row["session_id"],
                    correlation_id=row["correlation_id"],
                    event_type=row["event_type"],
                    topic=row["topic"],
                    timestamp=row["timestamp"],
                    status=row["status"],
                    processing_time_ms=row["processing_time_ms"],
                    payload=payload,
                    metadata=metadata,
                )
                events.append(event)

            # Calculate session metrics
            total_events = len(events)
            timestamps = [event.timestamp for event in events]
            start_time = min(timestamps)
            end_time = max(timestamps)

            # Calculate session duration (0 for single event)
            if total_events == 1:
                session_duration_ms = 0
            else:
                session_duration_ms = int(
                    (end_time - start_time).total_seconds() * 1000
                )

            # Determine session status from latest event
            latest_event = events[-1]
            if latest_event.status in {"failed", "error"}:
                status = "failed"
            elif latest_event.status == "completed":
                status = "completed"
            elif latest_event.status in {"processing", "sent", "received"}:
                # Check if any events are failed
                has_failed = any(e.status in {"failed", "error"} for e in events)
                status = "failed" if has_failed else "in_progress"
            else:
                status = "unknown"

            self.logger.info(
                f"Session {session_id} trace: {total_events} events, "
                f"duration: {session_duration_ms}ms, status: {status}"
            )

            return ModelEventTrace(
                session_id=session_id,
                events=events,
                total_events=total_events,
                session_duration_ms=session_duration_ms,
                status=status,
                start_time=start_time,
                end_time=end_time,
                time_range_hours=time_range_hours,
            )

        except Exception as e:
            self.logger.error(
                f"Failed to trace session events: {e}",
                exc_info=True,
                extra={
                    "session_id": str(session_id),
                    "time_range_hours": time_range_hours,
                },
            )
            raise OnexError(
                code=EnumCoreErrorCode.DATABASE_QUERY_ERROR,
                message=f"Failed to trace session events: {e}",
                context={
                    "session_id": str(session_id),
                    "time_range_hours": time_range_hours,
                },
            ) from e

    async def get_session_metrics(self, session_id: UUID) -> ModelSessionMetrics:
        """
        Get performance metrics for a code generation session.

        Calculates comprehensive performance metrics including response times,
        success/failure rates, bottleneck identification, and throughput analysis.

        Args:
            session_id: Code generation session ID

        Returns:
            ModelSessionMetrics containing:
                - session_id: Session ID string
                - total_events: Total event count
                - successful_events: Number of successful events
                - failed_events: Number of failed events
                - success_rate: Ratio 0.0-1.0
                - avg_response_time_ms: Average response time
                - min/max/p50/p95/p99_response_time_ms: Response time statistics
                - total_processing_time_ms: Sum of processing times
                - event_type_breakdown: Dict mapping event types to counts
                - topic_breakdown: Dict mapping topics to counts
                - bottlenecks: List of ModelBottleneck objects
                - timeline: ModelTimeline with start/end/duration

        Raises:
            OnexError: If database query fails or session_id is invalid

        Example:
            ```python
            session_id = UUID("123e4567-e89b-12d3-a456-426614174000")
            metrics = await tracer.get_session_metrics(session_id)

            print(f"Success rate: {metrics.success_rate * 100:.1f}%")
            print(f"Average response time: {metrics.avg_response_time_ms}ms")
            print(f"P95 response time: {metrics.p95_response_time_ms}ms")

            if metrics.bottlenecks:
                print("Bottlenecks detected:")
                for bottleneck in metrics.bottlenecks:
                    print(f"  {bottleneck.topic}: {bottleneck.avg_response_time_ms}ms")
            ```

        Performance Thresholds:
            - Fast: < 500ms response time
            - Normal: 500-2000ms response time
            - Slow: 2000-5000ms response time
            - Critical: > 5000ms response time (flagged as bottleneck)

        Bottleneck Severity:
            - High: avg_response_time_ms > 5000ms
            - Medium: avg_response_time_ms > 2000ms
            - Low: avg_response_time_ms > 1000ms
        """
        self.logger.info(f"Calculating metrics for session {session_id}")

        try:
            # Query all events for session
            query = """
            SELECT
                event_type, status, processing_time_ms,
                topic, timestamp
            FROM event_logs
            WHERE session_id = $1
            ORDER BY timestamp ASC
            """
            rows = await self.db.execute_query(query, session_id)

            # Handle empty result set
            if not rows:
                self.logger.warning(f"No events found for session {session_id}")
                from omninode_bridge.dashboard.models.model_session_metrics import (
                    ModelTimeline,
                )

                return ModelSessionMetrics(
                    session_id=str(session_id),
                    total_events=0,
                    successful_events=0,
                    failed_events=0,
                    success_rate=0.0,
                    avg_response_time_ms=0.0,
                    min_response_time_ms=0,
                    max_response_time_ms=0,
                    p50_response_time_ms=0,
                    p95_response_time_ms=0,
                    p99_response_time_ms=0,
                    total_processing_time_ms=0,
                    event_type_breakdown={},
                    topic_breakdown={},
                    bottlenecks=[],
                    timeline=ModelTimeline(
                        start_time=None,
                        end_time=None,
                        duration_ms=0,
                    ),
                )

            # Calculate basic metrics
            total_events = len(rows)
            successful_statuses = {"sent", "received", "completed"}
            failed_statuses = {"failed", "error"}

            successful_events = sum(
                1 for row in rows if row["status"] in successful_statuses
            )
            failed_events = sum(1 for row in rows if row["status"] in failed_statuses)
            success_rate = successful_events / total_events if total_events > 0 else 0.0

            # Calculate response time metrics from processing times
            processing_times = [
                row["processing_time_ms"]
                for row in rows
                if row["processing_time_ms"] is not None
            ]

            if processing_times:
                processing_times.sort()
                avg_response_time_ms = sum(processing_times) / len(processing_times)
                min_response_time_ms = processing_times[0]
                max_response_time_ms = processing_times[-1]
                total_processing_time_ms = sum(processing_times)

                # Calculate percentiles
                def percentile(data: list[int], p: float) -> int:
                    """Calculate percentile value from sorted data."""
                    k = (len(data) - 1) * p
                    f = int(k)
                    c = f + 1
                    if c >= len(data):
                        return data[-1]
                    d0 = data[f]
                    d1 = data[c]
                    return int(d0 + (d1 - d0) * (k - f))

                p50_response_time_ms = percentile(processing_times, 0.50)
                p95_response_time_ms = percentile(processing_times, 0.95)
                p99_response_time_ms = percentile(processing_times, 0.99)
            else:
                avg_response_time_ms = 0.0
                min_response_time_ms = 0
                max_response_time_ms = 0
                total_processing_time_ms = 0
                p50_response_time_ms = 0
                p95_response_time_ms = 0
                p99_response_time_ms = 0

            # Calculate event type breakdown
            event_type_breakdown: dict[str, int] = {}
            for row in rows:
                event_type = row["event_type"]
                event_type_breakdown[event_type] = (
                    event_type_breakdown.get(event_type, 0) + 1
                )

            # Calculate topic breakdown
            topic_breakdown: dict[str, int] = {}
            for row in rows:
                topic = row["topic"]
                topic_breakdown[topic] = topic_breakdown.get(topic, 0) + 1

            # Identify bottlenecks by topic
            from omninode_bridge.dashboard.models.model_session_metrics import (
                ModelBottleneck,
                ModelTimeline,
            )

            topic_processing_times: dict[str, list[int]] = {}
            for row in rows:
                if row["processing_time_ms"] is not None:
                    topic = row["topic"]
                    if topic not in topic_processing_times:
                        topic_processing_times[topic] = []
                    topic_processing_times[topic].append(row["processing_time_ms"])

            bottlenecks: list[ModelBottleneck] = []
            for topic, times in topic_processing_times.items():
                if not times:
                    continue

                avg_time = sum(times) / len(times)

                # Determine severity based on avg response time
                severity: Literal["high", "medium", "low"] | None = None
                if avg_time > 5000:
                    severity = "high"
                elif avg_time > 2000:
                    severity = "medium"
                elif avg_time > 1000:
                    severity = "low"

                if severity:
                    bottlenecks.append(
                        ModelBottleneck(
                            topic=topic,
                            avg_response_time_ms=avg_time,
                            count=len(times),
                            severity=severity,
                        )
                    )

            # Sort bottlenecks by severity (high -> medium -> low) and avg time
            severity_order = {"high": 0, "medium": 1, "low": 2}
            bottlenecks.sort(
                key=lambda b: (severity_order[b.severity], -b.avg_response_time_ms)
            )

            # Calculate timeline metrics
            timestamps = [row["timestamp"] for row in rows]
            start_time = min(timestamps)
            end_time = max(timestamps)
            duration_ms = int((end_time - start_time).total_seconds() * 1000)

            timeline = ModelTimeline(
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
            )

            self.logger.info(
                f"Session {session_id} metrics: {total_events} events, "
                f"{success_rate:.1%} success rate, "
                f"{avg_response_time_ms:.1f}ms avg response time"
            )

            return ModelSessionMetrics(
                session_id=str(session_id),
                total_events=total_events,
                successful_events=successful_events,
                failed_events=failed_events,
                success_rate=success_rate,
                avg_response_time_ms=avg_response_time_ms,
                min_response_time_ms=min_response_time_ms,
                max_response_time_ms=max_response_time_ms,
                p50_response_time_ms=p50_response_time_ms,
                p95_response_time_ms=p95_response_time_ms,
                p99_response_time_ms=p99_response_time_ms,
                total_processing_time_ms=total_processing_time_ms,
                event_type_breakdown=event_type_breakdown,
                topic_breakdown=topic_breakdown,
                bottlenecks=bottlenecks,
                timeline=timeline,
            )

        except Exception as e:
            self.logger.error(
                f"Failed to calculate session metrics: {e}",
                exc_info=True,
                extra={"session_id": str(session_id)},
            )
            raise OnexError(
                code=EnumCoreErrorCode.DATABASE_QUERY_ERROR,
                message=f"Failed to calculate session metrics: {e}",
                context={"session_id": str(session_id)},
            ) from e

    async def find_correlated_events(
        self, correlation_id: UUID
    ) -> list[ModelCorrelatedEvent]:
        """
        Find all events with the same correlation ID.

        Retrieves complete event chain for a correlation ID, enabling request/response
        matching and multi-hop event tracing across topics.

        Args:
            correlation_id: Correlation ID to search for

        Returns:
            List of ModelCorrelatedEvent objects ordered by timestamp.
            Each event contains:
                - event_id: Event UUID
                - session_id: Session UUID
                - correlation_id: Correlation UUID
                - event_type: "request", "response", "status", or "error"
                - topic: Kafka topic name
                - timestamp: Event timestamp
                - status: "sent", "received", "failed", or "processing"
                - processing_time_ms: Optional processing time
                - payload: Event data
                - metadata: Additional context

        Raises:
            OnexError: If database query fails or correlation_id is invalid

        Example:
            ```python
            correlation_id = UUID("123e4567-e89b-12d3-a456-426614174000")
            events = await tracer.find_correlated_events(correlation_id)

            print(f"Found {len(events)} correlated events:")
            for event in events:
                print(f"  [{event.timestamp}] {event.event_type}")
                print(f"    Topic: {event.topic}")
                print(f"    Status: {event.status}")
                if event.processing_time_ms:
                    print(f"    Processing time: {event.processing_time_ms}ms")
            ```

        Use Cases:
            - Request/response matching: Find response for a specific request
            - Multi-hop tracing: Track events across multiple services
            - Debugging: Identify where correlation chain breaks
            - Performance analysis: Measure end-to-end latency

        Event Chain Example:
            1. Request event: omniclaude sends analysis request
            2. Status event: omniarchon acknowledges request
            3. Response event: omniarchon returns analysis results
            4. Status event: omniclaude processes response
        """
        self.logger.info(f"Finding events correlated with {correlation_id}")

        try:
            # Query event_logs by correlation_id
            query = """
            SELECT
                event_id, session_id, correlation_id, event_type,
                topic, timestamp, status, processing_time_ms,
                payload, metadata
            FROM event_logs
            WHERE correlation_id = $1
            ORDER BY timestamp ASC
            """
            rows = await self.db.execute_query(query, correlation_id)

            # Handle empty result set
            if not rows:
                self.logger.info(
                    f"No correlated events found for correlation_id {correlation_id}"
                )
                return []

            # Convert rows to ModelCorrelatedEvent objects
            correlated_events: list[ModelCorrelatedEvent] = []
            for row in rows:
                # Parse JSONB fields if they're strings (testcontainers compatibility)
                # Handle payload with edge cases
                payload = row.get("payload", {})
                if isinstance(payload, str):
                    payload = json.loads(payload) if payload else {}
                elif payload is None:
                    payload = {}

                # Handle metadata with edge cases
                metadata = row.get("metadata", {})
                if isinstance(metadata, str):
                    metadata = json.loads(metadata) if metadata else {}
                elif metadata is None:
                    metadata = {}

                event = ModelCorrelatedEvent(
                    event_id=row["event_id"],
                    session_id=row["session_id"],
                    correlation_id=row["correlation_id"],
                    event_type=row["event_type"],
                    topic=row["topic"],
                    timestamp=row["timestamp"],
                    status=row["status"],
                    processing_time_ms=row.get("processing_time_ms"),
                    payload=payload,
                    metadata=metadata,
                )
                correlated_events.append(event)

            self.logger.info(
                f"Found {len(correlated_events)} correlated events for "
                f"correlation_id {correlation_id}"
            )

            # Optional: Log event chain summary for debugging
            if correlated_events:
                chain_summary = " -> ".join(
                    [f"{e.event_type}({e.status})" for e in correlated_events]
                )
                self.logger.debug(f"Event chain: {chain_summary}")

            return correlated_events

        except Exception as e:
            self.logger.error(
                f"Failed to find correlated events: {e}",
                exc_info=True,
                extra={"correlation_id": str(correlation_id)},
            )
            raise OnexError(
                code=EnumCoreErrorCode.DATABASE_QUERY_ERROR,
                message=f"Failed to find correlated events: {e}",
                context={"correlation_id": str(correlation_id)},
            ) from e
