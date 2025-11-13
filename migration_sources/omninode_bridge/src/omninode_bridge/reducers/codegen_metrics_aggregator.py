#!/usr/bin/env python3
"""
Code Generation Metrics Aggregator - Streaming Reducer.

Aggregates code generation metrics from Kafka events with support for:
- Node type grouping (Effect/Compute/Reducer/Orchestrator)
- Quality bucket classification (Low/Medium/High)
- Time-windowed aggregation (Hourly/Daily/Weekly/Monthly)
- Domain grouping (API/ML/Data)
- Real-time statistical computations (avg, p50, p95, p99)

ONEX v2.0 Compliance:
- Suffix-based naming: CodegenMetricsAggregator
- Contract-driven configuration from codegen_metrics.yaml
- Streaming aggregation with async processing
- Event-driven architecture with Kafka integration
- PostgreSQL persistence for aggregated metrics

Performance Targets:
- <50ms per aggregation operation
- >1000 events/second throughput
- <100ms for 1000 items aggregation
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, ClassVar, Optional
from uuid import UUID, uuid4

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# === Enums ===


class EnumNodeType(str, Enum):
    """ONEX node types for code generation."""

    EFFECT = "effect"
    COMPUTE = "compute"
    REDUCER = "reducer"
    ORCHESTRATOR = "orchestrator"


class EnumQualityBucket(str, Enum):
    """Quality score buckets for classification."""

    LOW = "low"  # 0.0-0.6
    MEDIUM = "medium"  # 0.6-0.8
    HIGH = "high"  # 0.8-1.0


class EnumDomain(str, Enum):
    """Code generation domains."""

    API = "api"
    ML = "ml"
    DATA = "data"
    INFRASTRUCTURE = "infrastructure"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    UNKNOWN = "unknown"


class EnumWindowType(str, Enum):
    """Time window types for aggregation."""

    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class EnumAggregationType(str, Enum):
    """Aggregation strategy types."""

    NODE_TYPE_GROUPING = "node_type_grouping"
    QUALITY_BUCKETS = "quality_buckets"
    TIME_WINDOW = "time_window"
    DOMAIN_GROUPING = "domain_grouping"


# === Event Models ===


class ModelNodeGenerationStarted(BaseModel):
    """Event model for node generation started."""

    event_type: str = Field(default="NODE_GENERATION_STARTED")
    workflow_id: UUID
    node_type: EnumNodeType
    domain: Optional[EnumDomain] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ModelNodeGenerationStageCompleted(BaseModel):
    """Event model for generation stage completed."""

    event_type: str = Field(default="NODE_GENERATION_STAGE_COMPLETED")
    workflow_id: UUID
    stage_name: str
    stage_number: int = Field(ge=1, le=6)
    duration_ms: int = Field(ge=0)
    success: bool
    tokens_consumed: Optional[int] = None
    cost_usd: Optional[float] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ModelNodeGenerationCompleted(BaseModel):
    """Event model for node generation completed."""

    event_type: str = Field(default="NODE_GENERATION_COMPLETED")
    workflow_id: UUID
    node_type: EnumNodeType
    domain: Optional[EnumDomain] = None
    total_duration_seconds: float = Field(ge=0)
    quality_score: float = Field(ge=0.0, le=1.0)
    total_tokens: int = Field(ge=0, default=0)
    total_cost_usd: float = Field(ge=0.0, default=0.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ModelNodeGenerationFailed(BaseModel):
    """Event model for node generation failed."""

    event_type: str = Field(default="NODE_GENERATION_FAILED")
    workflow_id: UUID
    node_type: Optional[EnumNodeType] = None
    domain: Optional[EnumDomain] = None
    failed_stage: str
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# === Aggregation Models ===


@dataclass
class DurationStatistics:
    """Duration statistics for aggregation."""

    avg_seconds: float = 0.0
    p50_seconds: float = 0.0
    p95_seconds: float = 0.0
    p99_seconds: float = 0.0
    min_seconds: float = 0.0
    max_seconds: float = 0.0


@dataclass
class QualityStatistics:
    """Quality statistics for aggregation."""

    avg_score: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0


@dataclass
class CostStatistics:
    """Cost statistics for aggregation."""

    total_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_tokens_per_generation: float = 0.0
    avg_cost_per_generation: float = 0.0


@dataclass
class StageStatistics:
    """Per-stage statistics."""

    total_executions: int = 0
    avg_duration_ms: float = 0.0
    failure_count: int = 0


@dataclass
class AggregationMetadata:
    """Metadata about the aggregation process."""

    aggregation_duration_ms: int = 0
    events_processed: int = 0
    items_per_second: float = 0.0


class ModelAggregatedMetrics(BaseModel):
    """Aggregated metrics output model."""

    aggregation_id: UUID = Field(default_factory=uuid4)
    window_type: EnumWindowType
    window_start: datetime
    window_end: datetime
    aggregation_type: EnumAggregationType

    # Grouping key (node_type, quality_bucket, or domain)
    group_key: str

    # Counts
    total_generations: int = 0
    successful_generations: int = 0
    failed_generations: int = 0
    success_rate: float = 0.0

    # Statistics
    duration_statistics: Optional[dict[str, float]] = None
    quality_statistics: Optional[dict[str, float]] = None
    cost_statistics: Optional[dict[str, Any]] = None
    stage_statistics: Optional[dict[str, dict[str, Any]]] = None
    node_type_breakdown: Optional[dict[str, int]] = None

    # Metadata
    aggregation_metadata: Optional[dict[str, Any]] = None

    class Config:
        json_encoders: ClassVar[dict[type, Any]] = {
            datetime: lambda v: v.isoformat(),
            UUID: str,
        }


# === Aggregator Implementation ===


class CodegenMetricsAggregator:
    """
    Code Generation Metrics Aggregator.

    Streams code generation events from Kafka and computes real-time
    aggregations across multiple dimensions:
    - Node type (Effect/Compute/Reducer/Orchestrator)
    - Quality buckets (Low/Medium/High)
    - Time windows (Hourly/Daily/Weekly/Monthly)
    - Domains (API/ML/Data/etc.)

    Performance:
    - <50ms per aggregation operation
    - >1000 events/second throughput
    - <100ms for 1000 items

    Implementation:
    - In-memory buffers with periodic flushes
    - NumPy for statistical computations
    - Async processing with concurrent workers
    - PostgreSQL persistence for aggregated metrics
    """

    def __init__(
        self,
        max_buffer_size: int = 10000,
        flush_interval_seconds: int = 60,
        enable_postgres_persistence: bool = False,
    ) -> None:
        """
        Initialize the metrics aggregator.

        Args:
            max_buffer_size: Maximum events in buffer before forced flush
            flush_interval_seconds: Interval for periodic flushes
            enable_postgres_persistence: Enable PostgreSQL persistence
        """
        self.max_buffer_size = max_buffer_size
        self.flush_interval_seconds = flush_interval_seconds
        self.enable_postgres_persistence = enable_postgres_persistence

        # Event buffers by aggregation type
        self._buffers: dict[EnumAggregationType, list[Any]] = {
            agg_type: [] for agg_type in EnumAggregationType
        }

        # Active workflow tracking
        self._active_workflows: dict[UUID, dict[str, Any]] = {}

        # Time window tracking
        self._window_boundaries: dict[EnumWindowType, datetime] = {}

        # Stage statistics tracking
        self._stage_metrics: dict[str, StageStatistics] = defaultdict(StageStatistics)

        # Metrics
        self._events_processed_total = 0
        self._aggregations_completed = 0
        self._last_flush_time = datetime.now(UTC)

        logger.info(
            f"CodegenMetricsAggregator initialized: "
            f"buffer_size={max_buffer_size}, "
            f"flush_interval={flush_interval_seconds}s, "
            f"postgres_persistence={enable_postgres_persistence}"
        )

    async def process_event(self, event: Any) -> None:
        """
        Process a single code generation event.

        Args:
            event: Code generation event (Started/StageCompleted/Completed/Failed)
        """
        start_time = datetime.now(UTC)

        try:
            if isinstance(event, ModelNodeGenerationStarted):
                await self._handle_generation_started(event)
            elif isinstance(event, ModelNodeGenerationStageCompleted):
                await self._handle_stage_completed(event)
            elif isinstance(event, ModelNodeGenerationCompleted):
                await self._handle_generation_completed(event)
            elif isinstance(event, ModelNodeGenerationFailed):
                await self._handle_generation_failed(event)
            else:
                logger.warning(f"Unknown event type: {type(event)}")
                return

            self._events_processed_total += 1

            # Check if buffer flush needed
            if self._should_flush():
                await self._flush_buffers()

            # Performance tracking
            duration_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000
            if duration_ms > 50:  # Warn if exceeds target
                logger.warning(
                    f"Event processing took {duration_ms:.2f}ms (target <50ms)"
                )

        except Exception as e:
            logger.error(f"Error processing event: {e}", exc_info=True)

    async def _handle_generation_started(
        self, event: ModelNodeGenerationStarted
    ) -> None:
        """Handle generation started event."""
        self._active_workflows[event.workflow_id] = {
            "workflow_id": event.workflow_id,
            "node_type": event.node_type,
            "domain": event.domain or EnumDomain.UNKNOWN,
            "start_time": event.timestamp,
            "stages_completed": [],
            "total_tokens": 0,
            "total_cost_usd": 0.0,
        }
        logger.debug(f"Workflow {event.workflow_id} started: {event.node_type}")

    async def _handle_stage_completed(
        self, event: ModelNodeGenerationStageCompleted
    ) -> None:
        """Handle stage completed event."""
        if event.workflow_id in self._active_workflows:
            workflow = self._active_workflows[event.workflow_id]
            workflow["stages_completed"].append(
                {
                    "stage_name": event.stage_name,
                    "stage_number": event.stage_number,
                    "duration_ms": event.duration_ms,
                    "success": event.success,
                }
            )

            if event.tokens_consumed:
                workflow["total_tokens"] += event.tokens_consumed
            if event.cost_usd:
                workflow["total_cost_usd"] += event.cost_usd

            # Update stage statistics
            stage_key = event.stage_name
            self._stage_metrics[stage_key].total_executions += 1
            if not event.success:
                self._stage_metrics[stage_key].failure_count += 1

            logger.debug(
                f"Workflow {event.workflow_id} stage {event.stage_number} "
                f"({event.stage_name}): {event.duration_ms}ms"
            )

    async def _handle_generation_completed(
        self, event: ModelNodeGenerationCompleted
    ) -> None:
        """Handle generation completed event."""
        # Add to all relevant buffers for multi-dimensional aggregation
        self._buffers[EnumAggregationType.NODE_TYPE_GROUPING].append(event)
        self._buffers[EnumAggregationType.QUALITY_BUCKETS].append(event)
        self._buffers[EnumAggregationType.TIME_WINDOW].append(event)
        if event.domain:
            self._buffers[EnumAggregationType.DOMAIN_GROUPING].append(event)

        # Clean up active workflow
        if event.workflow_id in self._active_workflows:
            del self._active_workflows[event.workflow_id]

        logger.debug(
            f"Workflow {event.workflow_id} completed: "
            f"duration={event.total_duration_seconds}s, "
            f"quality={event.quality_score:.2f}"
        )

    async def _handle_generation_failed(self, event: ModelNodeGenerationFailed) -> None:
        """Handle generation failed event."""
        # Add to buffers for failure tracking
        self._buffers[EnumAggregationType.NODE_TYPE_GROUPING].append(event)
        if event.domain:
            self._buffers[EnumAggregationType.DOMAIN_GROUPING].append(event)

        # Clean up active workflow
        if event.workflow_id in self._active_workflows:
            del self._active_workflows[event.workflow_id]

        logger.warning(
            f"Workflow {event.workflow_id} failed at stage {event.failed_stage}: "
            f"{event.error_message}"
        )

    def _should_flush(self) -> bool:
        """Check if buffers should be flushed."""
        # Flush if any buffer exceeds max size
        if any(len(buf) >= self.max_buffer_size for buf in self._buffers.values()):
            return True

        # Flush if flush interval exceeded
        time_since_flush = (datetime.now(UTC) - self._last_flush_time).total_seconds()
        return time_since_flush >= self.flush_interval_seconds

    async def _flush_buffers(self) -> None:
        """Flush all buffers and compute aggregations."""
        start_time = datetime.now(UTC)

        try:
            # Aggregate by node type
            node_type_aggs = await self._aggregate_by_node_type()

            # Aggregate by quality buckets
            quality_aggs = await self._aggregate_by_quality_buckets()

            # Aggregate by time windows
            time_window_aggs = await self._aggregate_by_time_windows()

            # Aggregate by domain
            domain_aggs = await self._aggregate_by_domain()

            all_aggregations = (
                node_type_aggs + quality_aggs + time_window_aggs + domain_aggs
            )

            # Persist to PostgreSQL if enabled
            if self.enable_postgres_persistence and all_aggregations:
                await self._persist_aggregations(all_aggregations)

            # Clear buffers
            for buffer in self._buffers.values():
                buffer.clear()

            self._last_flush_time = datetime.now(UTC)
            self._aggregations_completed += len(all_aggregations)

            duration_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000
            logger.info(
                f"Flushed buffers: {len(all_aggregations)} aggregations "
                f"in {duration_ms:.2f}ms"
            )

        except Exception as e:
            logger.error(f"Error flushing buffers: {e}", exc_info=True)

    async def _aggregate_by_node_type(self) -> list[ModelAggregatedMetrics]:
        """Aggregate metrics by node type."""
        buffer = self._buffers[EnumAggregationType.NODE_TYPE_GROUPING]
        if not buffer:
            return []

        # Group by node type
        grouped: dict[EnumNodeType, list[Any]] = defaultdict(list)
        for event in buffer:
            if isinstance(
                event, ModelNodeGenerationCompleted | ModelNodeGenerationFailed
            ):
                node_type = event.node_type
                grouped[node_type].append(event)

        aggregations = []
        for node_type, events in grouped.items():
            agg = await self._compute_aggregation(
                events=events,
                aggregation_type=EnumAggregationType.NODE_TYPE_GROUPING,
                group_key=node_type.value,
                window_type=EnumWindowType.HOURLY,
            )
            aggregations.append(agg)

        return aggregations

    async def _aggregate_by_quality_buckets(self) -> list[ModelAggregatedMetrics]:
        """Aggregate metrics by quality buckets."""
        buffer = self._buffers[EnumAggregationType.QUALITY_BUCKETS]
        if not buffer:
            return []

        # Group by quality bucket
        grouped: dict[EnumQualityBucket, list[Any]] = defaultdict(list)
        for event in buffer:
            if isinstance(event, ModelNodeGenerationCompleted):
                bucket = self._classify_quality(event.quality_score)
                grouped[bucket].append(event)

        aggregations = []
        for bucket, events in grouped.items():
            agg = await self._compute_aggregation(
                events=events,
                aggregation_type=EnumAggregationType.QUALITY_BUCKETS,
                group_key=bucket.value,
                window_type=EnumWindowType.HOURLY,
            )
            aggregations.append(agg)

        return aggregations

    async def _aggregate_by_time_windows(self) -> list[ModelAggregatedMetrics]:
        """Aggregate metrics by time windows."""
        buffer = self._buffers[EnumAggregationType.TIME_WINDOW]
        if not buffer:
            return []

        aggregations = []

        # Aggregate for each window type
        for window_type in EnumWindowType:
            window_events = await self._get_window_events(buffer, window_type)
            if window_events:
                agg = await self._compute_aggregation(
                    events=window_events,
                    aggregation_type=EnumAggregationType.TIME_WINDOW,
                    group_key=window_type.value,
                    window_type=window_type,
                )
                aggregations.append(agg)

        return aggregations

    async def _aggregate_by_domain(self) -> list[ModelAggregatedMetrics]:
        """Aggregate metrics by domain."""
        buffer = self._buffers[EnumAggregationType.DOMAIN_GROUPING]
        if not buffer:
            return []

        # Group by domain
        grouped: dict[EnumDomain, list[Any]] = defaultdict(list)
        for event in buffer:
            if isinstance(
                event, ModelNodeGenerationCompleted | ModelNodeGenerationFailed
            ):
                domain = event.domain or EnumDomain.UNKNOWN
                grouped[domain].append(event)

        aggregations = []
        for domain, events in grouped.items():
            agg = await self._compute_aggregation(
                events=events,
                aggregation_type=EnumAggregationType.DOMAIN_GROUPING,
                group_key=domain.value,
                window_type=EnumWindowType.HOURLY,
            )
            aggregations.append(agg)

        return aggregations

    async def _compute_aggregation(
        self,
        events: list[Any],
        aggregation_type: EnumAggregationType,
        group_key: str,
        window_type: EnumWindowType,
    ) -> ModelAggregatedMetrics:
        """Compute aggregated metrics from events."""
        start_time = datetime.now(UTC)

        # Separate completed and failed events
        completed_events = [
            e for e in events if isinstance(e, ModelNodeGenerationCompleted)
        ]
        failed_events = [e for e in events if isinstance(e, ModelNodeGenerationFailed)]

        total_generations = len(events)
        successful_generations = len(completed_events)
        failed_generations = len(failed_events)
        success_rate = (
            successful_generations / total_generations if total_generations > 0 else 0.0
        )

        # Duration statistics (only for completed)
        duration_stats = None
        if completed_events:
            durations = np.array([e.total_duration_seconds for e in completed_events])
            duration_stats = {
                "avg_seconds": float(np.mean(durations)),
                "p50_seconds": float(np.percentile(durations, 50)),
                "p95_seconds": float(np.percentile(durations, 95)),
                "p99_seconds": float(np.percentile(durations, 99)),
                "min_seconds": float(np.min(durations)),
                "max_seconds": float(np.max(durations)),
            }

        # Quality statistics
        quality_stats = None
        if completed_events:
            quality_scores = np.array([e.quality_score for e in completed_events])
            quality_stats = {
                "avg_score": float(np.mean(quality_scores)),
                "min_score": float(np.min(quality_scores)),
                "max_score": float(np.max(quality_scores)),
            }

        # Cost statistics
        cost_stats = None
        if completed_events:
            total_tokens = sum(e.total_tokens for e in completed_events)
            total_cost = sum(e.total_cost_usd for e in completed_events)
            cost_stats = {
                "total_tokens": total_tokens,
                "total_cost_usd": total_cost,
                "avg_tokens_per_generation": total_tokens / len(completed_events),
                "avg_cost_per_generation": total_cost / len(completed_events),
            }

        # Aggregation metadata
        duration_ms = int((datetime.now(UTC) - start_time).total_seconds() * 1000)
        agg_metadata = {
            "aggregation_duration_ms": duration_ms,
            "events_processed": total_generations,
            "items_per_second": (
                total_generations / (duration_ms / 1000.0) if duration_ms > 0 else 0.0
            ),
        }

        # Window boundaries
        window_start, window_end = self._get_window_boundaries(window_type)

        return ModelAggregatedMetrics(
            window_type=window_type,
            window_start=window_start,
            window_end=window_end,
            aggregation_type=aggregation_type,
            group_key=group_key,
            total_generations=total_generations,
            successful_generations=successful_generations,
            failed_generations=failed_generations,
            success_rate=success_rate,
            duration_statistics=duration_stats,
            quality_statistics=quality_stats,
            cost_statistics=cost_stats,
            aggregation_metadata=agg_metadata,
        )

    def _classify_quality(self, quality_score: float) -> EnumQualityBucket:
        """Classify quality score into bucket."""
        if quality_score < 0.6:
            return EnumQualityBucket.LOW
        elif quality_score < 0.8:
            return EnumQualityBucket.MEDIUM
        else:
            return EnumQualityBucket.HIGH

    async def _get_window_events(
        self, events: list[Any], window_type: EnumWindowType
    ) -> list[Any]:
        """Get events within the current time window."""
        window_start, window_end = self._get_window_boundaries(window_type)

        return [e for e in events if window_start <= e.timestamp < window_end]

    def _get_window_boundaries(
        self, window_type: EnumWindowType
    ) -> tuple[datetime, datetime]:
        """Get window boundaries for the given window type."""
        now = datetime.now(UTC)

        if window_type == EnumWindowType.HOURLY:
            window_start = now.replace(minute=0, second=0, microsecond=0)
            window_end = window_start + timedelta(hours=1)
        elif window_type == EnumWindowType.DAILY:
            window_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            window_end = window_start + timedelta(days=1)
        elif window_type == EnumWindowType.WEEKLY:
            days_since_monday = now.weekday()
            window_start = (now - timedelta(days=days_since_monday)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            window_end = window_start + timedelta(weeks=1)
        else:  # MONTHLY
            window_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            # Next month
            if now.month == 12:
                window_end = window_start.replace(year=now.year + 1, month=1)
            else:
                window_end = window_start.replace(month=now.month + 1)

        return window_start, window_end

    async def _persist_aggregations(
        self, aggregations: list[ModelAggregatedMetrics]
    ) -> None:
        """Persist aggregations to PostgreSQL."""
        # PostgreSQL persistence (Phase 2 implementation)
        # Implementation will batch write metrics to time-series optimized table:
        #   CREATE TABLE codegen_metrics_timeseries (
        #     timestamp TIMESTAMPTZ, node_name TEXT, metric_name TEXT, value NUMERIC, tags JSONB
        #   ) PARTITION BY RANGE (timestamp);
        logger.debug(f"Would persist {len(aggregations)} aggregations to PostgreSQL")

    async def get_metrics(self) -> dict[str, Any]:
        """Get aggregator metrics."""
        return {
            "events_processed_total": self._events_processed_total,
            "aggregations_completed": self._aggregations_completed,
            "active_workflows": len(self._active_workflows),
            "buffer_sizes": {
                agg_type.value: len(buffer)
                for agg_type, buffer in self._buffers.items()
            },
            "stage_statistics": {
                stage: {
                    "total_executions": stats.total_executions,
                    "avg_duration_ms": stats.avg_duration_ms,
                    "failure_count": stats.failure_count,
                }
                for stage, stats in self._stage_metrics.items()
            },
        }
