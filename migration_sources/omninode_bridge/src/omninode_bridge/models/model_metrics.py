"""Model performance metrics tracking for Smart Responder Chain integration."""

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """Classification of task types for model performance tracking."""

    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    DEBUGGING = "debugging"
    DOCUMENTATION = "documentation"
    API_DESIGN = "api_design"
    TESTING = "testing"
    ARCHITECTURE = "architecture"
    SECURITY_AUDIT = "security_audit"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    WEBHOOK_PROCESSING = "webhook_processing"
    EVENT_TRANSFORMATION = "event_transformation"
    GENERAL_REASONING = "general_reasoning"


class ModelTier(str, Enum):
    """Model tiers from Smart Responder Chain."""

    TINY = "tiny"  # 3B models - fast, simple tasks
    SMALL = "small"  # 7B models - moderate complexity
    MEDIUM = "medium"  # 8B models - balanced performance
    LARGE = "large"  # 11B models - complex reasoning
    XLARGE = "xlarge"  # 14B models - advanced tasks
    HUGE = "huge"  # 70B+ models - maximum capability
    CUSTOM = "custom"  # Custom fine-tuned models
    CLOUD = "cloud"  # External API models


class ModelEndpoint(BaseModel):
    """Model endpoint configuration."""

    model_id: str
    endpoint_url: str
    tier: ModelTier
    node_location: str  # mac-studio, mac-mini, ai-pc, macbook-air
    model_name: str
    parameter_count: str | None = None
    specialized_for: list[TaskType] = Field(default_factory=list)
    max_context_tokens: int = 4096
    tokens_per_second: float | None = None


class TaskExecution(BaseModel):
    """Individual task execution record."""

    execution_id: UUID = Field(default_factory=uuid4)
    task_type: TaskType
    model_endpoint: str
    model_tier: ModelTier
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None

    # Context and complexity metrics
    input_tokens: int
    output_tokens: int
    context_size: int  # Total context including chat history
    task_complexity: str  # simple, moderate, complex, critical

    # Performance metrics
    latency_ms: float | None = None
    tokens_per_second: float | None = None
    first_token_latency_ms: float | None = None

    # Quality metrics
    success: bool = False
    quality_score: float | None = None  # 0.0-1.0 if available
    user_satisfaction: int | None = None  # 1-5 rating if provided
    retry_count: int = 0
    escalated_to_tier: ModelTier | None = None

    # Error tracking
    error_type: str | None = None
    error_message: str | None = None

    # Task-specific metrics
    code_lines_generated: int | None = None
    test_coverage: float | None = None
    security_issues_found: int | None = None
    performance_improvement: float | None = None

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelPerformanceStats(BaseModel):
    """Aggregated performance statistics for a model."""

    model_endpoint: str
    model_tier: ModelTier
    task_type: TaskType

    # Execution counts
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0

    # Performance averages
    avg_latency_ms: float = 0.0
    avg_tokens_per_second: float = 0.0
    avg_first_token_latency_ms: float = 0.0

    # Quality metrics
    avg_quality_score: float = 0.0
    avg_user_satisfaction: float = 0.0
    success_rate: float = 0.0

    # Context efficiency
    avg_context_size: float = 0.0
    max_context_handled: int = 0
    context_efficiency_score: float = 0.0  # success_rate / avg_context_size

    # Specialization metrics
    specialization_score: float = 0.0  # How well suited for this task type

    # Recent performance (last 100 executions)
    recent_success_rate: float = 0.0
    recent_avg_latency_ms: float = 0.0

    # Last updated
    last_updated: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ModelRecommendation(BaseModel):
    """Model recommendation for a specific task."""

    model_endpoint: str
    model_tier: ModelTier
    confidence_score: float  # 0.0-1.0
    expected_latency_ms: float
    expected_success_rate: float
    expected_quality_score: float
    reason: str

    # Cost considerations
    compute_cost_estimate: float | None = None
    energy_efficiency: float | None = None


class SmartResponderMetrics(BaseModel):
    """Enhanced metrics for Smart Responder Chain integration."""

    # Available models from the lab
    available_models: list[ModelEndpoint] = Field(default_factory=list)

    # Performance tracking
    execution_history: list[TaskExecution] = Field(default_factory=list)
    performance_stats: dict[str, list[ModelPerformanceStats]] = Field(
        default_factory=dict,
    )

    # Routing intelligence
    task_type_preferences: dict[TaskType, list[str]] = Field(default_factory=dict)
    context_size_routing: dict[str, int] = Field(
        default_factory=dict,
    )  # model -> max efficient context

    # Adaptive learning
    model_learning_curves: dict[str, list[float]] = Field(default_factory=dict)
    recent_escalation_patterns: list[dict[str, Any]] = Field(default_factory=list)

    # Lab utilization
    node_utilization: dict[str, float] = Field(default_factory=dict)
    load_balancing_weights: dict[str, float] = Field(default_factory=dict)

    def add_execution(self, execution: TaskExecution) -> None:
        """Add a task execution to the metrics."""
        self.execution_history.append(execution)

        # Keep only last 10000 executions to manage memory
        if len(self.execution_history) > 10000:
            self.execution_history = self.execution_history[-10000:]

        # Update performance stats
        self._update_performance_stats(execution)

        # Update routing intelligence
        self._update_routing_intelligence(execution)

    def _update_performance_stats(self, execution: TaskExecution) -> None:
        """Update aggregated performance statistics."""
        key = f"{execution.model_endpoint}_{execution.task_type}"

        if key not in self.performance_stats:
            self.performance_stats[key] = []

        # Find existing stats or create new
        stats = None
        for s in self.performance_stats[key]:
            if (
                s.model_endpoint == execution.model_endpoint
                and s.task_type == execution.task_type
            ):
                stats = s
                break

        if not stats:
            stats = ModelPerformanceStats(
                model_endpoint=execution.model_endpoint,
                model_tier=execution.model_tier,
                task_type=execution.task_type,
            )
            self.performance_stats[key].append(stats)

        # Update statistics
        stats.total_executions += 1
        if execution.success:
            stats.successful_executions += 1
        else:
            stats.failed_executions += 1

        # Update running averages
        if execution.latency_ms:
            stats.avg_latency_ms = self._update_running_average(
                stats.avg_latency_ms,
                execution.latency_ms,
                stats.total_executions,
            )

        if execution.tokens_per_second:
            stats.avg_tokens_per_second = self._update_running_average(
                stats.avg_tokens_per_second,
                execution.tokens_per_second,
                stats.total_executions,
            )

        if execution.quality_score:
            stats.avg_quality_score = self._update_running_average(
                stats.avg_quality_score,
                execution.quality_score,
                stats.total_executions,
            )

        # Update success rate
        stats.success_rate = stats.successful_executions / stats.total_executions

        # Update context metrics
        stats.avg_context_size = self._update_running_average(
            stats.avg_context_size,
            execution.context_size,
            stats.total_executions,
        )

        if execution.context_size > stats.max_context_handled:
            stats.max_context_handled = execution.context_size

        # Calculate context efficiency (success per context size)
        if stats.avg_context_size > 0:
            stats.context_efficiency_score = stats.success_rate / (
                stats.avg_context_size / 1000
            )

        stats.last_updated = datetime.now(UTC)

    def _update_routing_intelligence(self, execution: TaskExecution) -> None:
        """Update intelligent routing preferences."""
        # Track successful models for each task type
        if execution.success:
            if execution.task_type not in self.task_type_preferences:
                self.task_type_preferences[execution.task_type] = []

            prefs = self.task_type_preferences[execution.task_type]
            if execution.model_endpoint not in prefs:
                prefs.append(execution.model_endpoint)

            # Move successful model towards front of list
            prefs.remove(execution.model_endpoint)
            prefs.insert(0, execution.model_endpoint)

            # Keep only top 5 preferences
            self.task_type_preferences[execution.task_type] = prefs[:5]

        # Track escalation patterns
        if execution.escalated_to_tier:
            self.recent_escalation_patterns.append(
                {
                    "from_tier": execution.model_tier,
                    "to_tier": execution.escalated_to_tier,
                    "task_type": execution.task_type,
                    "context_size": execution.context_size,
                    "timestamp": execution.started_at.isoformat(),
                },
            )

            # Keep only last 100 escalations
            if len(self.recent_escalation_patterns) > 100:
                self.recent_escalation_patterns = self.recent_escalation_patterns[-100:]

    def _update_running_average(
        self,
        current_avg: float,
        new_value: float,
        count: int,
    ) -> float:
        """Update running average with new value."""
        if count == 1:
            return new_value
        return ((current_avg * (count - 1)) + new_value) / count

    def get_model_recommendation(
        self,
        task_type: TaskType,
        context_size: int,
        complexity: str = "moderate",
        max_latency_ms: float | None = None,
    ) -> list[ModelRecommendation]:
        """Get ranked model recommendations for a specific task."""
        recommendations = []

        # Get relevant performance stats
        relevant_stats = []
        for stats_list in self.performance_stats.values():
            for stats in stats_list:
                if stats.task_type == task_type and stats.total_executions >= 5:
                    relevant_stats.append(stats)

        # Score each model
        for stats in relevant_stats:
            confidence = self._calculate_confidence_score(
                stats,
                context_size,
                complexity,
            )

            if confidence > 0.1:  # Only recommend models with reasonable confidence
                recommendation = ModelRecommendation(
                    model_endpoint=stats.model_endpoint,
                    model_tier=stats.model_tier,
                    confidence_score=confidence,
                    expected_latency_ms=stats.avg_latency_ms,
                    expected_success_rate=stats.success_rate,
                    expected_quality_score=stats.avg_quality_score,
                    reason=self._generate_recommendation_reason(
                        stats,
                        context_size,
                        complexity,
                    ),
                )
                recommendations.append(recommendation)

        # Sort by confidence score
        recommendations.sort(key=lambda x: x.confidence_score, reverse=True)

        return recommendations[:5]  # Return top 5 recommendations

    def _calculate_confidence_score(
        self,
        stats: ModelPerformanceStats,
        context_size: int,
        complexity: str,
    ) -> float:
        """Calculate confidence score for a model recommendation."""
        # Base score from success rate
        score = stats.success_rate

        # Adjust for context size efficiency
        if stats.max_context_handled >= context_size:
            context_factor = min(1.0, stats.context_efficiency_score)
            score *= 0.7 + 0.3 * context_factor
        else:
            score *= 0.3  # Penalize if context too large

        # Adjust for task complexity
        complexity_weights = {
            "simple": 1.0,
            "moderate": 0.9,
            "complex": 0.8,
            "critical": 0.7,
        }
        if stats.model_tier in [ModelTier.TINY, ModelTier.SMALL] and complexity in [
            "complex",
            "critical",
        ]:
            score *= 0.5  # Small models struggle with complex tasks
        elif (
            stats.model_tier in [ModelTier.LARGE, ModelTier.XLARGE, ModelTier.HUGE]
            and complexity == "simple"
        ):
            score *= 0.8  # Overkill for simple tasks

        # Boost for recent good performance
        score *= 0.7 + 0.3 * stats.recent_success_rate

        # Data quality factor (more executions = higher confidence)
        data_quality = min(1.0, stats.total_executions / 100)
        score *= 0.5 + 0.5 * data_quality

        return min(1.0, score)

    def _generate_recommendation_reason(
        self,
        stats: ModelPerformanceStats,
        context_size: int,
        complexity: str,
    ) -> str:
        """Generate human-readable reason for recommendation."""
        reasons = []

        if stats.success_rate > 0.9:
            reasons.append(f"High success rate ({stats.success_rate:.1%})")
        elif stats.success_rate > 0.7:
            reasons.append(f"Good success rate ({stats.success_rate:.1%})")

        if stats.avg_latency_ms < 1000:
            reasons.append(f"Fast response ({stats.avg_latency_ms:.0f}ms avg)")
        elif stats.avg_latency_ms < 3000:
            reasons.append(f"Moderate latency ({stats.avg_latency_ms:.0f}ms avg)")

        if stats.max_context_handled >= context_size * 1.5:
            reasons.append("Handles large contexts well")

        if stats.context_efficiency_score > 0.5:
            reasons.append("Context efficient")

        if stats.total_executions > 50:
            reasons.append(f"Well-tested ({stats.total_executions} executions)")

        return "; ".join(reasons) if reasons else "Based on limited data"


# Global metrics instance for the application
global_metrics = SmartResponderMetrics()
