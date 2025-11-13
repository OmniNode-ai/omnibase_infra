"""
Routing models for Smart Routing Orchestration.

This module provides data models for routing decisions, contexts, and history
tracking. All models are Pydantic v2 compliant with strict validation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class RoutingDecision(str, Enum):
    """
    Routing decision enumeration.

    Defines all possible routing decisions for workflow orchestration.
    """

    ERROR = "error"
    END = "end"
    RETRY = "retry"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    BRANCH = "branch"
    SKIP = "skip"
    CONTINUE = "continue"


class RoutingStrategy(str, Enum):
    """Routing strategy types."""

    CONDITIONAL = "conditional"
    PARALLEL = "parallel"
    STATE_ANALYSIS = "state_analysis"
    PRIORITY = "priority"


class RoutingContext(BaseModel):
    """
    Context for routing evaluation.

    Provides all necessary information for routers to make informed decisions.

    Attributes:
        routing_id: Unique routing request identifier
        current_task: Task being evaluated
        state_summary: Summary of current state (for analysis)
        execution_time: Time spent executing current task (ms)
        retry_count: Number of retries for current task
        custom_data: Additional context data
        correlation_id: Optional correlation ID for tracing
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    routing_id: UUID = Field(default_factory=uuid4, description="Unique routing ID")
    current_task: str = Field(
        description="Task being evaluated", min_length=1, max_length=256
    )
    state_summary: dict[str, Any] = Field(
        default_factory=dict, description="Summary of current state"
    )
    execution_time: float = Field(
        default=0.0, ge=0.0, description="Execution time in milliseconds"
    )
    retry_count: int = Field(default=0, ge=0, description="Number of retries")
    custom_data: dict[str, Any] = Field(
        default_factory=dict, description="Additional context data"
    )
    correlation_id: Optional[str] = Field(
        default=None, description="Correlation ID for tracing"
    )


class RoutingResult(BaseModel):
    """
    Result of routing evaluation.

    Contains the routing decision, confidence score, and detailed reasoning.

    Attributes:
        decision: Routing decision made
        confidence: Confidence score (0.0-1.0)
        reasoning: Human-readable explanation
        strategy: Strategy that made the decision
        next_task: Next task to execute (if applicable)
        metadata: Additional metadata
    """

    decision: RoutingDecision = Field(description="Routing decision")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score (0.0-1.0)"
    )
    reasoning: str = Field(description="Human-readable explanation", min_length=1)
    strategy: RoutingStrategy = Field(description="Strategy that made decision")
    next_task: Optional[str] = Field(default=None, description="Next task to execute")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class RoutingHistoryRecord(BaseModel):
    """
    Immutable record of a routing decision.

    Tracks all routing decisions for debugging and analysis.

    Attributes:
        timestamp: When the routing decision was made
        routing_id: Unique routing request identifier
        context: Routing context at time of decision
        result: Routing result/decision
        routing_time_ms: Time taken to make decision (ms)
    """

    model_config = ConfigDict(frozen=True)

    timestamp: datetime = Field(
        default_factory=lambda: datetime.utcnow(),
        description="When the routing decision was made",
    )
    routing_id: UUID = Field(description="Unique routing request identifier")
    context: RoutingContext = Field(description="Routing context")
    result: RoutingResult = Field(description="Routing result/decision")
    routing_time_ms: float = Field(ge=0.0, description="Routing time in milliseconds")


class ConditionalRule(BaseModel):
    """
    Conditional routing rule.

    Defines a condition and the decision to make if the condition is met.

    Attributes:
        rule_id: Unique rule identifier
        name: Human-readable rule name
        condition_key: State key to evaluate
        condition_operator: Comparison operator (==, !=, >, <, >=, <=, in, not_in)
        condition_value: Value to compare against
        decision: Decision to make if condition is met
        next_task: Next task if condition is met
        priority: Rule priority (higher = evaluated first)
    """

    rule_id: str = Field(description="Unique rule identifier", min_length=1)
    name: str = Field(description="Human-readable rule name", min_length=1)
    condition_key: str = Field(description="State key to evaluate", min_length=1)
    condition_operator: str = Field(
        description="Comparison operator",
        pattern=r"^(==|!=|>|<|>=|<=|in|not_in|contains|not_contains)$",
    )
    condition_value: Any = Field(description="Value to compare against")
    decision: RoutingDecision = Field(description="Decision if condition met")
    next_task: Optional[str] = Field(
        default=None, description="Next task if condition met"
    )
    priority: int = Field(default=50, ge=0, le=100, description="Rule priority")


class ParallelizationHint(BaseModel):
    """
    Hint for parallel task execution.

    Identifies tasks that can be executed in parallel with dependencies.

    Attributes:
        task_group: Group of parallel tasks
        dependencies: Task IDs that must complete before this group
        estimated_duration_ms: Estimated duration in milliseconds
    """

    task_group: list[str] = Field(
        description="Group of parallel tasks", min_length=1
    )
    dependencies: list[str] = Field(
        default_factory=list, description="Task IDs that must complete first"
    )
    estimated_duration_ms: float = Field(
        default=0.0, ge=0.0, description="Estimated duration in milliseconds"
    )


class StateComplexityMetrics(BaseModel):
    """
    Metrics for state complexity analysis.

    Provides quantitative metrics for routing decisions based on state complexity.

    Attributes:
        key_count: Number of keys in state
        nested_depth: Maximum nesting depth
        total_size_bytes: Approximate total size in bytes
        complexity_score: Overall complexity score (0.0-1.0)
        has_errors: Whether state contains error indicators
        has_incomplete_data: Whether state is missing required data
    """

    key_count: int = Field(ge=0, description="Number of keys in state")
    nested_depth: int = Field(ge=0, description="Maximum nesting depth")
    total_size_bytes: int = Field(ge=0, description="Approximate total size in bytes")
    complexity_score: float = Field(
        ge=0.0, le=1.0, description="Overall complexity score"
    )
    has_errors: bool = Field(default=False, description="Contains error indicators")
    has_incomplete_data: bool = Field(
        default=False, description="Missing required data"
    )


class PriorityRoutingConfig(BaseModel):
    """
    Configuration for priority-based routing.

    Defines priority levels and routing rules.

    Attributes:
        high_priority_threshold: Tasks with priority >= this use high-priority routing
        low_priority_threshold: Tasks with priority <= this use low-priority routing
        high_priority_decision: Decision for high-priority tasks
        low_priority_decision: Decision for low-priority tasks
        default_decision: Decision for medium-priority tasks
    """

    high_priority_threshold: int = Field(
        default=80, ge=0, le=100, description="High-priority threshold"
    )
    low_priority_threshold: int = Field(
        default=20, ge=0, le=100, description="Low-priority threshold"
    )
    high_priority_decision: RoutingDecision = Field(
        default=RoutingDecision.CONTINUE, description="Decision for high-priority tasks"
    )
    low_priority_decision: RoutingDecision = Field(
        default=RoutingDecision.SKIP, description="Decision for low-priority tasks"
    )
    default_decision: RoutingDecision = Field(
        default=RoutingDecision.CONTINUE, description="Decision for medium-priority tasks"
    )
