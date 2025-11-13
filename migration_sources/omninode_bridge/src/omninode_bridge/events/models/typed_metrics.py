"""
Type definitions for strongly-typed metrics structures.

Provides TypedDict definitions for model metrics and node type metrics
to eliminate Any types in public APIs. Shared across event models and
metrics reducer components.
"""

from typing_extensions import TypedDict


class ModelPerModelMetrics(TypedDict):
    """
    Per-model performance metrics structure.

    Used in ModelMetricsState.model_metrics and ModelEventMetricsRecorded.
    Matches the exact fields computed by MetricsAggregator._compute_model_metrics.

    Fields:
        total_generations: Total generations using this model
        avg_duration_seconds: Average duration in seconds
        avg_quality_score: Average quality score (0-1)
        total_tokens: Total tokens consumed across all generations
        total_cost_usd: Total cost in USD across all generations
        avg_cost_per_generation: Average cost per generation in USD
    """

    total_generations: int
    avg_duration_seconds: float
    avg_quality_score: float
    total_tokens: int
    total_cost_usd: float
    avg_cost_per_generation: float


class ModelPerNodeTypeMetrics(TypedDict):
    """
    Per-node-type performance metrics structure.

    Used in ModelMetricsState.node_type_metrics and ModelEventMetricsRecorded.
    Matches the exact fields computed by MetricsAggregator._compute_node_type_metrics.

    Fields:
        total_generations: Total generations for this node type
        avg_duration_seconds: Average duration in seconds
        avg_quality_score: Average quality score (0-1)
    """

    total_generations: int
    avg_duration_seconds: float
    avg_quality_score: float
