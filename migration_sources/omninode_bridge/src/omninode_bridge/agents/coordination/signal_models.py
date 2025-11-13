"""
Coordination signal data models for agent-to-agent communication.

This module provides type-safe signal models for coordinating multi-agent
code generation workflows with event-driven communication patterns.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class SignalType(str, Enum):
    """Signal type enumeration for coordination events."""

    AGENT_INITIALIZED = "agent_initialized"
    AGENT_COMPLETED = "agent_completed"
    DEPENDENCY_RESOLVED = "dependency_resolved"
    INTER_AGENT_MESSAGE = "inter_agent_message"
    STAGE_COMPLETED = "stage_completed"  # Workflow stage completion
    ERROR_RECOVERY_COMPLETED = "error_recovery_completed"  # Error recovery completed


class CoordinationSignal(BaseModel):
    """
    Coordination signal for agent-to-agent communication.

    Attributes:
        signal_id: Unique signal identifier
        signal_type: Type of coordination signal
        sender_agent_id: Agent that sent the signal
        recipient_agents: List of recipient agent IDs (empty = broadcast)
        timestamp: Signal creation timestamp
        event_data: Signal-specific event data
        metadata: Additional metadata (correlation_id, priority, etc.)
        coordination_id: ID of coordination session
    """

    signal_id: str = Field(default_factory=lambda: str(uuid4()))
    signal_type: SignalType
    sender_agent_id: str = Field(..., min_length=1, max_length=100)
    recipient_agents: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_data: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    coordination_id: str = Field(..., min_length=1, max_length=100)

    class Config:
        """Pydantic configuration."""

        frozen = False
        validate_assignment = True


class AgentInitializedSignal(BaseModel):
    """
    Agent initialized signal - notifies other agents that an agent has started.

    Attributes:
        agent_id: ID of initialized agent
        capabilities: Agent capabilities
        ready: Whether agent is ready to receive tasks
        initialization_time_ms: Time taken to initialize
    """

    agent_id: str = Field(..., min_length=1)
    capabilities: list[str] = Field(default_factory=list)
    ready: bool = Field(default=True)
    initialization_time_ms: float = Field(default=0.0, ge=0)


class AgentCompletedSignal(BaseModel):
    """
    Agent completed signal - notifies completion with result summary.

    Attributes:
        agent_id: ID of completed agent
        result_summary: Summary of results produced
        quality_score: Quality score (0.0-1.0)
        execution_time_ms: Total execution time
        artifacts_generated: List of generated artifacts (file paths, etc.)
        error: Optional error message if failed
    """

    agent_id: str = Field(..., min_length=1)
    result_summary: str = Field(default="")
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0)
    execution_time_ms: float = Field(default=0.0, ge=0)
    artifacts_generated: list[str] = Field(default_factory=list)
    error: Optional[str] = None


class DependencyResolvedSignal(BaseModel):
    """
    Dependency resolved signal - notifies that a dependency is available.

    Attributes:
        dependency_type: Type of dependency (e.g., "model", "contract", "validator")
        dependency_id: Identifier of resolved dependency
        resolved_by: Agent that resolved the dependency
        resolution_data: Data about resolved dependency (file path, schema, etc.)
    """

    dependency_type: str = Field(..., min_length=1)
    dependency_id: str = Field(..., min_length=1)
    resolved_by: str = Field(..., min_length=1)
    resolution_data: dict[str, Any] = Field(default_factory=dict)


class InterAgentMessage(BaseModel):
    """
    Inter-agent message - general message passing between agents.

    Attributes:
        message_type: Type of message (e.g., "request", "notification", "query")
        message: Message content
        requires_response: Whether sender expects a response
        response_timeout_ms: Timeout for response (if requires_response=True)
        payload: Additional message payload
    """

    message_type: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    requires_response: bool = Field(default=False)
    response_timeout_ms: float = Field(default=5000.0, ge=0)
    payload: dict[str, Any] = Field(default_factory=dict)


class SignalSubscription(BaseModel):
    """
    Signal subscription for filtering signals.

    Attributes:
        subscription_id: Unique subscription identifier
        coordination_id: Coordination session to subscribe to
        agent_id: Subscribing agent ID
        signal_types: List of signal types to receive (empty = all)
        sender_filter: Filter by sender agent ID (None = all senders)
        created_at: Subscription creation timestamp
    """

    subscription_id: str = Field(default_factory=lambda: str(uuid4()))
    coordination_id: str = Field(..., min_length=1)
    agent_id: str = Field(..., min_length=1)
    signal_types: list[SignalType] = Field(default_factory=list)
    sender_filter: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        """Pydantic configuration."""

        frozen = False


class SignalMetrics(BaseModel):
    """
    Metrics for signal coordination.

    Attributes:
        total_signals_sent: Total signals sent
        total_signals_received: Total signals received
        average_propagation_ms: Average signal propagation time
        max_propagation_ms: Maximum signal propagation time
        signals_by_type: Signal counts by type
    """

    total_signals_sent: int = Field(default=0, ge=0)
    total_signals_received: int = Field(default=0, ge=0)
    average_propagation_ms: float = Field(default=0.0, ge=0)
    max_propagation_ms: float = Field(default=0.0, ge=0)
    signals_by_type: dict[str, int] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        frozen = False
