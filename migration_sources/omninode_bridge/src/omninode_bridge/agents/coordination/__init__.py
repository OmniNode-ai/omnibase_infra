"""
Agent coordination utilities for multi-agent workflows.

This package provides thread-safe state management, typed state models,
signal coordination for agent-to-agent communication, context distribution,
smart routing orchestration, dependency resolution, and custom exceptions for
coordinating parallel agent execution.

Public API:
    - CoordinationOrchestrator: Unified coordination system integrating all components
    - ThreadSafeState: Thread-safe state container with versioning
    - SignalCoordinator: Agent-to-agent signal coordination system
    - ContextDistributor: Agent context distribution system
    - SmartRoutingOrchestrator: Intelligent routing orchestration
    - DependencyResolver: Multi-agent dependency resolution system
    - StateChangeRecord: Immutable change record model
    - AgentCoordinationState: Typed state for agent coordination
    - CodeGenerationState: Typed state for code generation
    - CoordinationSignal: Signal model for agent communication
    - SignalType: Signal type enumeration
    - AgentInitializedSignal: Agent initialization signal
    - AgentCompletedSignal: Agent completion signal
    - DependencyResolvedSignal: Dependency resolution signal
    - InterAgentMessage: Inter-agent message passing
    - SignalSubscription: Signal subscription model
    - SignalMetrics: Signal metrics tracking
    - AgentContext: Complete context package for agent
    - CoordinationMetadata: Session and agent identification
    - SharedIntelligence: Shared data structures and patterns
    - AgentAssignment: Agent's specific assignment
    - CoordinationProtocols: Communication and update protocols
    - ResourceAllocation: Resource limits and constraints
    - ConditionalRouter: Conditional routing based on state
    - ParallelRouter: Parallel execution identification
    - StateAnalysisRouter: State complexity analysis routing
    - PriorityRouter: Priority-based routing
    - RoutingContext: Routing evaluation context
    - RoutingResult: Routing decision result
    - RoutingDecision: Routing decision enumeration
    - Dependency: Dependency specification model
    - DependencyType: Dependency type enumeration
    - DependencyStatus: Dependency status enumeration
    - DependencyResolutionResult: Dependency resolution result
    - AgentCompletionConfig: Agent completion dependency config
    - ResourceAvailabilityConfig: Resource availability dependency config
    - QualityGateConfig: Quality gate dependency config
    - StateKeyError: Key not found exception
    - StateVersionError: Version-related error
    - StateRollbackError: Rollback operation error
    - DependencyResolutionError: Dependency resolution error
    - DependencyTimeoutError: Dependency timeout error
"""

from .context_distribution import ContextDistributor
from .context_models import (
    AgentAssignment,
    AgentContext,
    ContextDistributionMetrics,
    ContextUpdateRequest,
    CoordinationMetadata,
    CoordinationProtocols,
    ResourceAllocation,
    SharedIntelligence,
)
from .dependency_models import (
    AgentCompletionConfig,
    Dependency,
    DependencyResolutionResult,
    DependencyStatus,
    DependencyType,
    QualityGateConfig,
    ResourceAvailabilityConfig,
)
from .dependency_resolution import DependencyResolver
from .exceptions import (
    DependencyResolutionError,
    DependencyTimeoutError,
    StateKeyError,
    StateLockTimeoutError,
    StateRollbackError,
    StateVersionError,
    ThreadSafeStateError,
)
from .models import AgentCoordinationState, CodeGenerationState, StateChangeRecord
from .orchestrator import CoordinationOrchestrator
from .routing import (
    ConditionalRouter,
    ParallelRouter,
    PriorityRouter,
    SmartRoutingOrchestrator,
    StateAnalysisRouter,
)
from .routing_models import (
    ConditionalRule,
    ParallelizationHint,
    PriorityRoutingConfig,
    RoutingContext,
    RoutingDecision,
    RoutingHistoryRecord,
    RoutingResult,
    RoutingStrategy,
    StateComplexityMetrics,
)
from .signal_models import (
    AgentCompletedSignal,
    AgentInitializedSignal,
    CoordinationSignal,
    DependencyResolvedSignal,
    InterAgentMessage,
    SignalMetrics,
    SignalSubscription,
    SignalType,
)
from .signals import SignalCoordinator
from .thread_safe_state import ThreadSafeState

__all__ = [
    # Core classes
    "CoordinationOrchestrator",
    "ThreadSafeState",
    "SignalCoordinator",
    "ContextDistributor",
    "SmartRoutingOrchestrator",
    "DependencyResolver",
    # State models
    "StateChangeRecord",
    "AgentCoordinationState",
    "CodeGenerationState",
    # Signal models
    "CoordinationSignal",
    "SignalType",
    "AgentInitializedSignal",
    "AgentCompletedSignal",
    "DependencyResolvedSignal",
    "InterAgentMessage",
    "SignalSubscription",
    "SignalMetrics",
    # Context models
    "AgentContext",
    "CoordinationMetadata",
    "SharedIntelligence",
    "AgentAssignment",
    "CoordinationProtocols",
    "ResourceAllocation",
    "ContextDistributionMetrics",
    "ContextUpdateRequest",
    # Routing components
    "ConditionalRouter",
    "ParallelRouter",
    "StateAnalysisRouter",
    "PriorityRouter",
    "RoutingContext",
    "RoutingResult",
    "RoutingDecision",
    "RoutingStrategy",
    "RoutingHistoryRecord",
    "ConditionalRule",
    "ParallelizationHint",
    "PriorityRoutingConfig",
    "StateComplexityMetrics",
    # Dependency models
    "Dependency",
    "DependencyType",
    "DependencyStatus",
    "DependencyResolutionResult",
    "AgentCompletionConfig",
    "ResourceAvailabilityConfig",
    "QualityGateConfig",
    # Exceptions
    "ThreadSafeStateError",
    "StateKeyError",
    "StateVersionError",
    "StateRollbackError",
    "StateLockTimeoutError",
    "DependencyResolutionError",
    "DependencyTimeoutError",
]
