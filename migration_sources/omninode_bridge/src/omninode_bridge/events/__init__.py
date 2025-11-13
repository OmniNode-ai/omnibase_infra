"""
Event model definitions for omninode_bridge

This package provides contract-first event schemas for the event-driven
architecture enabling parallel development across services.

All events follow the OnexEnvelopeV1 standard format for consistency
across the omninode ecosystem.
"""

from omninode_bridge.events.models.codegen_events import (  # Envelope; Node generation events; Metrics events; Pattern storage events; Intelligence gathering events; Orchestration events; Topic constants
    TOPIC_CODEGEN_COMPLETED,
    TOPIC_CODEGEN_FAILED,
    TOPIC_CODEGEN_METRICS_RECORDED,
    TOPIC_CODEGEN_REQUESTED,
    TOPIC_CODEGEN_STAGE_COMPLETED,
    TOPIC_CODEGEN_STARTED,
    TOPIC_INTELLIGENCE_QUERY_COMPLETED,
    TOPIC_INTELLIGENCE_QUERY_REQUESTED,
    TOPIC_ORCHESTRATOR_CHECKPOINT_REACHED,
    TOPIC_ORCHESTRATOR_CHECKPOINT_RESPONSE,
    TOPIC_PATTERN_STORAGE_REQUESTED,
    TOPIC_PATTERN_STORED,
    IntelligenceQueryCompletedEvent,
    IntelligenceQueryRequestedEvent,
    ModelEventCodegenCompleted,
    ModelEventCodegenFailed,
    ModelEventCodegenStageCompleted,
    ModelEventCodegenStarted,
    ModelEventMetricsRecorded,
    NodeGenerationRequestedEvent,
    OnexEnvelopeV1,
    OrchestratorCheckpointReachedEvent,
    OrchestratorCheckpointResponseEvent,
    PatternStorageRequestedEvent,
    PatternStoredEvent,
)

__all__ = [
    # Envelope
    "OnexEnvelopeV1",
    # Node generation events
    "NodeGenerationRequestedEvent",
    "ModelEventCodegenStarted",
    "ModelEventCodegenStageCompleted",
    "ModelEventCodegenCompleted",
    "ModelEventCodegenFailed",
    # Metrics events
    "ModelEventMetricsRecorded",
    # Pattern storage events
    "PatternStorageRequestedEvent",
    "PatternStoredEvent",
    # Intelligence gathering events
    "IntelligenceQueryRequestedEvent",
    "IntelligenceQueryCompletedEvent",
    # Orchestration events
    "OrchestratorCheckpointReachedEvent",
    "OrchestratorCheckpointResponseEvent",
    # Topic constants
    "TOPIC_CODEGEN_REQUESTED",
    "TOPIC_CODEGEN_STARTED",
    "TOPIC_CODEGEN_STAGE_COMPLETED",
    "TOPIC_CODEGEN_COMPLETED",
    "TOPIC_CODEGEN_FAILED",
    "TOPIC_CODEGEN_METRICS_RECORDED",
    "TOPIC_PATTERN_STORAGE_REQUESTED",
    "TOPIC_PATTERN_STORED",
    "TOPIC_INTELLIGENCE_QUERY_REQUESTED",
    "TOPIC_INTELLIGENCE_QUERY_COMPLETED",
    "TOPIC_ORCHESTRATOR_CHECKPOINT_REACHED",
    "TOPIC_ORCHESTRATOR_CHECKPOINT_RESPONSE",
]
