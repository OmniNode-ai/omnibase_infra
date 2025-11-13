"""
Pure Aggregation Logic for Code Generation Metrics.

This module contains pure functions for aggregating code generation metrics
from event streams. No I/O operations - just data transformation.

Performance Targets:
- >1000 events/second throughput
- <100ms aggregation latency for 1000 items
- Streaming aggregation with windowing

Streaming Aggregation Design:
- Incremental state updates (O(1) memory per batch)
- Mergeable statistics for efficient processing
- Only buffers data needed for percentiles
"""

import math
import statistics
from collections import defaultdict, deque
from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from omninode_bridge.events.models.codegen_events import (
    ModelEventCodegenCompleted,
    ModelEventCodegenFailed,
    ModelEventCodegenStageCompleted,
    ModelEventCodegenStarted,
)
from omninode_bridge.events.models.typed_metrics import (
    ModelPerModelMetrics,
    ModelPerNodeTypeMetrics,
)

from .models.enum_metrics_window import EnumMetricsWindow
from .models.model_metrics_state import ModelMetricsState


class AggregationState(BaseModel):
    """
    Incrementally mergeable aggregation state for streaming metrics.

    This state can be updated batch-by-batch without buffering all events.
    Tracks running totals, counts, and only data needed for final computation.

    Memory Efficiency:
    - O(1) space for counts and sums
    - O(N) space only for durations (needed for percentiles)
    - O(M) space for model/node_type aggregates (M = unique models/types)
    """

    # Aggregation metadata
    aggregation_id: UUID = Field(default_factory=uuid4)
    window_start: datetime | None = None
    window_end: datetime | None = None

    # Running counts
    total_generations: int = 0
    successful_generations: int = 0
    failed_generations: int = 0

    # Duration tracking (for percentiles - unavoidable)
    all_durations: deque = Field(default_factory=lambda: deque(maxlen=10000))
    min_duration: float = float("inf")
    max_duration: float = 0.0

    # Quality metrics - running sums for averages
    quality_score_sum: float = 0.0
    quality_score_count: int = 0
    test_coverage_sum: float = 0.0
    test_coverage_count: int = 0
    complexity_sum: float = 0.0
    complexity_count: int = 0

    # Cost metrics - running totals
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    # Stage performance - running sums
    stage_durations: dict[str, list[float]] = Field(
        default_factory=lambda: defaultdict(list)
    )

    # Model metrics - running totals per model
    model_durations: dict[str, list[float]] = Field(
        default_factory=lambda: defaultdict(list)
    )
    model_quality_scores: dict[str, list[float]] = Field(
        default_factory=lambda: defaultdict(list)
    )
    model_tokens: dict[str, list[int]] = Field(
        default_factory=lambda: defaultdict(list)
    )
    model_costs: dict[str, list[float]] = Field(
        default_factory=lambda: defaultdict(list)
    )

    # Node type metrics - running totals per node type
    node_type_durations: dict[str, list[float]] = Field(
        default_factory=lambda: defaultdict(list)
    )
    node_type_quality_scores: dict[str, list[float]] = Field(
        default_factory=lambda: defaultdict(list)
    )

    # Intelligence usage
    intelligence_enabled_count: int = 0
    patterns_counts: list[int] = Field(default_factory=list)

    # Workflow tracking
    workflow_ids: set[UUID] = Field(default_factory=set)

    # Event count
    events_processed: int = 0

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


class MetricsAggregator:
    """
    Pure aggregation logic for code generation metrics.

    This class provides pure functions for streaming aggregation of
    code generation events into metrics state.

    All methods are pure functions (no I/O, no side effects) for
    easy testing and reasoning.
    """

    @staticmethod
    def aggregate_events(
        events: list[
            ModelEventCodegenStarted
            | ModelEventCodegenStageCompleted
            | ModelEventCodegenCompleted
            | ModelEventCodegenFailed
        ],
        window_type: EnumMetricsWindow = EnumMetricsWindow.HOURLY,
    ) -> ModelMetricsState:
        """
        Aggregate code generation events into metrics state.

        Pure function - no I/O, no side effects.

        Args:
            events: List of code generation events to aggregate
            window_type: Time window for aggregation

        Returns:
            ModelMetricsState with aggregated metrics
        """
        # Group events by type
        started_events: list[ModelEventCodegenStarted] = []
        stage_events: list[ModelEventCodegenStageCompleted] = []
        completed_events: list[ModelEventCodegenCompleted] = []
        failed_events: list[ModelEventCodegenFailed] = []

        for event in events:
            if isinstance(event, ModelEventCodegenStarted):
                started_events.append(event)
            elif isinstance(event, ModelEventCodegenStageCompleted):
                stage_events.append(event)
            elif isinstance(event, ModelEventCodegenCompleted):
                completed_events.append(event)
            elif isinstance(event, ModelEventCodegenFailed):
                failed_events.append(event)

        # Calculate window bounds
        if events:
            window_start = min(e.timestamp for e in events)
            window_end = max(e.timestamp for e in events)
        else:
            window_start = datetime.now(UTC)
            window_end = datetime.now(UTC)

        # Compute performance metrics
        total_generations = len(started_events)
        successful_generations = len(completed_events)
        failed_generations = len(failed_events)

        # Compute duration statistics from completed events
        durations = [e.total_duration_seconds for e in completed_events]
        if durations:
            avg_duration = statistics.mean(durations)
            p50_duration = statistics.median(durations)
            p95_duration = MetricsAggregator._percentile(durations, 0.95)
            p99_duration = MetricsAggregator._percentile(durations, 0.99)
            min_duration = min(durations)
            max_duration = max(durations)
        else:
            avg_duration = 0.0
            p50_duration = 0.0
            p95_duration = 0.0
            p99_duration = 0.0
            min_duration = 0.0
            max_duration = 0.0

        # Compute quality metrics from completed events
        quality_scores = [e.quality_score for e in completed_events]
        avg_quality_score = statistics.mean(quality_scores) if quality_scores else 0.0

        test_coverages = [
            e.test_coverage for e in completed_events if e.test_coverage is not None
        ]
        avg_test_coverage = statistics.mean(test_coverages) if test_coverages else None

        complexity_scores = [
            e.complexity_score
            for e in completed_events
            if e.complexity_score is not None
        ]
        avg_complexity = (
            statistics.mean(complexity_scores) if complexity_scores else None
        )

        # Compute cost metrics from completed events
        total_tokens = sum(e.total_tokens for e in completed_events)
        total_cost_usd = sum(e.total_cost_usd for e in completed_events)
        avg_cost_per_generation = (
            total_cost_usd / len(completed_events) if completed_events else 0.0
        )

        # Compute stage performance from stage_completed events
        stage_durations: dict[str, list[float]] = defaultdict(list)
        for event in stage_events:
            stage_durations[event.stage_name].append(event.duration_seconds)

        avg_stage_durations = {
            stage: statistics.mean(durations)
            for stage, durations in stage_durations.items()
        }

        # Compute per-model metrics from completed events
        model_metrics = MetricsAggregator._compute_model_metrics(completed_events)

        # Compute per-node-type metrics from completed events
        node_type_metrics = MetricsAggregator._compute_node_type_metrics(
            completed_events
        )

        # Compute intelligence usage from completed events
        intelligence_enabled_count = sum(
            1 for e in completed_events if e.patterns_applied
        )
        patterns_counts = [
            len(e.patterns_applied) for e in completed_events if e.patterns_applied
        ]
        avg_patterns_applied = (
            statistics.mean(patterns_counts) if patterns_counts else 0.0
        )

        # Count unique workflow IDs
        workflow_ids = {e.workflow_id for e in events}
        workflow_ids_tracked = len(workflow_ids)

        return ModelMetricsState(
            aggregation_id=uuid4(),
            window_type=window_type,
            window_start=window_start,
            window_end=window_end,
            total_generations=total_generations,
            successful_generations=successful_generations,
            failed_generations=failed_generations,
            avg_duration_seconds=avg_duration,
            p50_duration_seconds=p50_duration,
            p95_duration_seconds=p95_duration,
            p99_duration_seconds=p99_duration,
            min_duration_seconds=min_duration,
            max_duration_seconds=max_duration,
            avg_quality_score=avg_quality_score,
            avg_test_coverage=avg_test_coverage,
            avg_complexity_score=avg_complexity,
            total_tokens=total_tokens,
            total_cost_usd=total_cost_usd,
            avg_cost_per_generation=avg_cost_per_generation,
            avg_stage_durations=avg_stage_durations,
            model_metrics=model_metrics,
            node_type_metrics=node_type_metrics,
            intelligence_enabled_count=intelligence_enabled_count,
            avg_patterns_applied=avg_patterns_applied,
            workflow_ids_tracked=workflow_ids_tracked,
            events_processed=len(events),
            aggregation_duration_ms=0.0,  # Will be set by caller
            items_per_second=0.0,  # Will be set by caller
        )

    @staticmethod
    def _percentile(data: list[float], percentile: float) -> float:
        """
        Calculate percentile using nearest-rank method.

        Uses the ceiling method: ordinal_rank = ceil(p * N)
        This ensures correct results for all sample sizes, including small N.

        Args:
            data: List of values
            percentile: Percentile (0.0-1.0)

        Returns:
            Percentile value

        Examples:
            >>> _percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.95)
            10  # 95th percentile of 10 values
            >>> _percentile([1, 2, 3], 0.5)
            2  # 50th percentile (median) of 3 values
        """
        if not data:
            return 0.0

        # Edge case handling
        if percentile <= 0:
            return min(data)
        if percentile >= 1:
            return max(data)

        sorted_data = sorted(data)
        # Nearest-rank method: ordinal rank = ceil(p * N)
        ordinal_rank = math.ceil(percentile * len(sorted_data))
        # Convert to 0-based index (ordinal rank 1 = index 0)
        index = ordinal_rank - 1
        return sorted_data[index]

    @staticmethod
    def _compute_model_metrics(
        completed_events: list[ModelEventCodegenCompleted],
    ) -> dict[str, ModelPerModelMetrics]:
        """
        Compute per-model metrics from completed events.

        Args:
            completed_events: List of completed generation events

        Returns:
            Dict mapping model names to their strongly-typed metrics
        """
        model_data: dict[str, list[ModelEventCodegenCompleted]] = defaultdict(list)

        for event in completed_events:
            model_data[event.primary_model].append(event)

        model_metrics: dict[str, ModelPerModelMetrics] = {}
        for model_name, events in model_data.items():
            durations = [e.total_duration_seconds for e in events]
            quality_scores = [e.quality_score for e in events]
            tokens = [e.total_tokens for e in events]
            costs = [e.total_cost_usd for e in events]

            model_metrics[model_name] = {
                "total_generations": len(events),
                "avg_duration_seconds": statistics.mean(durations),
                "avg_quality_score": statistics.mean(quality_scores),
                "total_tokens": sum(tokens),
                "total_cost_usd": sum(costs),
                "avg_cost_per_generation": statistics.mean(costs),
            }

        return model_metrics

    @staticmethod
    def _compute_node_type_metrics(
        completed_events: list[ModelEventCodegenCompleted],
    ) -> dict[str, ModelPerNodeTypeMetrics]:
        """
        Compute per-node-type metrics from completed events.

        Args:
            completed_events: List of completed generation events

        Returns:
            Dict mapping node types to their strongly-typed metrics
        """
        node_type_data: dict[str, list[ModelEventCodegenCompleted]] = defaultdict(list)

        for event in completed_events:
            node_type_data[event.node_type].append(event)

        node_type_metrics: dict[str, ModelPerNodeTypeMetrics] = {}
        for node_type, events in node_type_data.items():
            durations = [e.total_duration_seconds for e in events]
            quality_scores = [e.quality_score for e in events]

            node_type_metrics[node_type] = {
                "total_generations": len(events),
                "avg_duration_seconds": statistics.mean(durations),
                "avg_quality_score": statistics.mean(quality_scores),
            }

        return node_type_metrics
