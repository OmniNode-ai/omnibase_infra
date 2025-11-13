"""
Metrics State Model for Code Generation Metrics Aggregation.

Tracks aggregated metrics for code generation workflows:
- Performance metrics (duration, throughput)
- Quality metrics (quality score, test coverage)
- Cost metrics (tokens, cost)
- Success/failure rates
- Per-model and per-node-type breakdowns
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field

from omninode_bridge.events.models.typed_metrics import (
    ModelPerModelMetrics,
    ModelPerNodeTypeMetrics,
)

from .enum_metrics_window import EnumMetricsWindow


class ModelMetricsState(BaseModel):
    """
    Aggregated metrics state for code generation workflows.

    This model represents the reduced/aggregated state after processing
    multiple NODE_GENERATION_* events within a time window.
    """

    # Aggregation metadata
    aggregation_id: UUID = Field(..., description="Unique aggregation identifier")
    window_type: EnumMetricsWindow = Field(
        ..., description="Time window type (hourly|daily|weekly|monthly)"
    )
    window_start: datetime = Field(..., description="Start of aggregation window")
    window_end: datetime = Field(..., description="End of aggregation window")

    # Performance metrics (from STARTED, STAGE_COMPLETED, COMPLETED events)
    total_generations: int = Field(default=0, description="Total generation requests")
    successful_generations: int = Field(default=0, description="Successful completions")
    failed_generations: int = Field(default=0, description="Failed generations")

    # Duration statistics (in seconds)
    avg_duration_seconds: float = Field(
        default=0.0, description="Average generation duration"
    )
    p50_duration_seconds: float = Field(
        default=0.0, description="50th percentile duration"
    )
    p95_duration_seconds: float = Field(
        default=0.0, description="95th percentile duration"
    )
    p99_duration_seconds: float = Field(
        default=0.0, description="99th percentile duration"
    )
    min_duration_seconds: float = Field(
        default=0.0, description="Minimum duration observed"
    )
    max_duration_seconds: float = Field(
        default=0.0, description="Maximum duration observed"
    )

    # Quality metrics (from COMPLETED events)
    avg_quality_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Average quality score (0-1)"
    )
    avg_test_coverage: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Average test coverage (0-1)"
    )
    avg_complexity_score: Optional[float] = Field(
        None, description="Average cyclomatic complexity"
    )

    # Cost metrics (from COMPLETED events)
    total_tokens: int = Field(default=0, description="Total tokens consumed")
    total_cost_usd: float = Field(default=0.0, description="Total cost in USD")
    avg_cost_per_generation: float = Field(
        default=0.0, description="Average cost per generation"
    )

    # Stage performance (from STAGE_COMPLETED events)
    avg_stage_durations: dict[str, float] = Field(
        default_factory=dict,
        description="Average duration per stage (stage_name -> avg_duration_seconds)",
    )

    # Model performance breakdown (from COMPLETED events)
    model_metrics: dict[str, ModelPerModelMetrics] = Field(
        default_factory=dict,
        description="Per-model performance stats (model_name -> metrics)",
    )

    # Node type breakdown (from COMPLETED events)
    node_type_metrics: dict[str, ModelPerNodeTypeMetrics] = Field(
        default_factory=dict,
        description="Per-node-type stats (node_type -> metrics)",
    )

    # Intelligence usage (from COMPLETED events)
    intelligence_enabled_count: int = Field(
        default=0, description="Generations with intelligence enabled"
    )
    avg_patterns_applied: float = Field(
        default=0.0, description="Average patterns applied per generation"
    )

    # Workflow tracking
    workflow_ids_tracked: int = Field(
        default=0, description="Number of unique workflows tracked"
    )

    # Aggregation performance
    aggregation_duration_ms: float = Field(
        default=0.0, description="Time to compute aggregation"
    )
    events_processed: int = Field(default=0, description="Number of events aggregated")
    items_per_second: float = Field(
        default=0.0, description="Aggregation throughput (events/sec)"
    )

    # Metadata
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When aggregation was computed"
    )
