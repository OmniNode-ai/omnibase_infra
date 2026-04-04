# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Logical topic key constants for ProtocolTopicRegistry resolution.

These constants are logical identifiers used as keys in
``registry.resolve(topic_keys.X)`` calls. They are NOT Kafka topic strings --
the registry maps these keys to concrete topic strings at runtime.

Naming convention: drop the ``TOPIC_`` prefix from the legacy constant name.
For example, ``TOPIC_RESOLUTION_DECIDED`` becomes ``RESOLUTION_DECIDED``.

Usage:
    >>> from omnibase_infra.topics import topic_keys
    >>> from omnibase_infra.protocols import ProtocolTopicRegistry
    >>>
    >>> # Resolve a logical key to a concrete topic string
    >>> topic = registry.resolve(topic_keys.RESOLUTION_DECIDED)

Related:
    - OMN-5839: Topic registry consolidation epic
    - ProtocolTopicRegistry: Protocol that consumes these keys
    - ServiceTopicRegistry: Default implementation

.. versionadded:: 0.24.0
"""

from __future__ import annotations

from typing import Final

# ==============================================================================
# Session Topics
# ==============================================================================

SESSION_OUTCOME_CURRENT: Final[str] = "SESSION_OUTCOME_CURRENT"
"""Session-outcome command topic for intelligence processing (dual-publish cmd)."""

SESSION_OUTCOME_CANONICAL: Final[str] = "SESSION_OUTCOME_CANONICAL"
"""Session-outcome event topic for observability (dual-publish evt)."""

# ==============================================================================
# Injection Effectiveness Topics
# ==============================================================================

INJECTION_CONTEXT_UTILIZATION: Final[str] = "INJECTION_CONTEXT_UTILIZATION"
"""Context utilization metrics from omniclaude injection hooks."""

INJECTION_AGENT_MATCH: Final[str] = "INJECTION_AGENT_MATCH"
"""Agent match metrics from omniclaude injection hooks."""

INJECTION_LATENCY_BREAKDOWN: Final[str] = "INJECTION_LATENCY_BREAKDOWN"
"""Latency breakdown metrics from omniclaude injection hooks."""

INJECTION_CONTEXT_ENRICHMENT: Final[str] = "INJECTION_CONTEXT_ENRICHMENT"
"""Context enrichment metrics from omniclaude injection hooks (OMN-6158)."""

INJECTION_RECORDED: Final[str] = "INJECTION_RECORDED"
"""Injection recorded events from omniclaude injection hooks (OMN-6158)."""

MANIFEST_INJECTION_STARTED: Final[str] = "MANIFEST_INJECTION_STARTED"
"""Manifest injection started lifecycle event (OMN-1888)."""

MANIFEST_INJECTED: Final[str] = "MANIFEST_INJECTED"
"""Manifest injected lifecycle event (OMN-1888)."""

MANIFEST_INJECTION_FAILED: Final[str] = "MANIFEST_INJECTION_FAILED"
"""Manifest injection failed lifecycle event (OMN-1888)."""

# ==============================================================================
# LLM Topics
# ==============================================================================

LLM_CALL_COMPLETED: Final[str] = "LLM_CALL_COMPLETED"
"""LLM call completed metrics from omniintelligence."""

LLM_CALL_COMPLETED_INFRA: Final[str] = "LLM_CALL_COMPLETED_INFRA"
"""LLM call completed event from omnibase-infra inference effect."""

LLM_ENDPOINT_HEALTH: Final[str] = "LLM_ENDPOINT_HEALTH"
"""LLM endpoint health probe events."""

LLM_INFERENCE_REQUEST: Final[str] = "LLM_INFERENCE_REQUEST"
"""Inbound LLM inference request command topic."""

LLM_EMBEDDING_REQUEST: Final[str] = "LLM_EMBEDDING_REQUEST"
"""Inbound LLM embedding request command topic."""

EVAL_COMPLETED: Final[str] = "EVAL_COMPLETED"
"""Eval task completed event from ServiceEvalRunner."""

# ==============================================================================
# Effectiveness Topics
# ==============================================================================

EFFECTIVENESS_INVALIDATION: Final[str] = "EFFECTIVENESS_INVALIDATION"
"""Effectiveness data invalidation for dashboard refresh."""

# ==============================================================================
# Agent Topics
# ==============================================================================

AGENT_STATUS: Final[str] = "AGENT_STATUS"
"""Agent status events for real-time agent visibility."""

# ==============================================================================
# Reward Topics
# ==============================================================================

REWARD_ASSIGNED: Final[str] = "REWARD_ASSIGNED"
"""Per-target reward assignment with traceable evidence refs."""

# ==============================================================================
# Resolution Topics
# ==============================================================================

RESOLUTION_DECIDED: Final[str] = "RESOLUTION_DECIDED"
"""Resolution decision audit events."""

# ==============================================================================
# Circuit Breaker Topics
# ==============================================================================

CIRCUIT_BREAKER_STATE: Final[str] = "CIRCUIT_BREAKER_STATE"
"""Circuit breaker state transition events."""

# ==============================================================================
# Wiring Health Topics
# ==============================================================================

WIRING_HEALTH_SNAPSHOT: Final[str] = "WIRING_HEALTH_SNAPSHOT"
"""Wiring health snapshot events from WiringHealthChecker."""

# ==============================================================================
# Savings Estimation Topics
# ==============================================================================

SAVINGS_ESTIMATED: Final[str] = "SAVINGS_ESTIMATED"
"""Savings estimation event with tiered attribution breakdown."""

VALIDATOR_CATCH: Final[str] = "VALIDATOR_CATCH"
"""Validator catch events from pre-commit hooks, CI checks, poly enforcer."""

PATTERN_ENFORCEMENT: Final[str] = "PATTERN_ENFORCEMENT"
"""Pattern enforcement events from omniclaude hooks (severity-tagged catches)."""

HOOK_CONTEXT_INJECTED: Final[str] = "HOOK_CONTEXT_INJECTED"
"""Hook context injection events from omniclaude UserPromptSubmit hooks."""

# ==============================================================================
# Consumer Health Topics
# ==============================================================================

CONSUMER_HEALTH: Final[str] = "CONSUMER_HEALTH"
"""Consumer health events from ConsumerHealthEmitter."""

CONSUMER_RESTART_CMD: Final[str] = "CONSUMER_RESTART_CMD"
"""Consumer restart commands issued by triage node."""

# ==============================================================================
# Runtime Error Topics
# ==============================================================================

RUNTIME_ERROR: Final[str] = "RUNTIME_ERROR"
"""Runtime error events from RuntimeLogEventBridge."""

ERROR_TRIAGED: Final[str] = "ERROR_TRIAGED"
"""Runtime error triage result events from NodeRuntimeErrorTriageEffect."""

# ==============================================================================
# Routing Topics
# ==============================================================================

ROUTING_DECIDED: Final[str] = "ROUTING_DECIDED"
"""Routing decision event from AdapterModelRouter."""

# ==============================================================================
# Baselines Topics
# ==============================================================================

BASELINES_COMPUTED: Final[str] = "BASELINES_COMPUTED"
"""Baselines batch computation snapshot events."""

# ==============================================================================
# Waitlist Topics
# ==============================================================================

WAITLIST_SIGNUP: Final[str] = "WAITLIST_SIGNUP"
"""Waitlist signup events from omniweb Server Action (email_domain only, no PII)."""

# ==============================================================================
# Build Loop Topics (OMN-5113)
# ==============================================================================

# Commands (6)
BUILD_LOOP_START: Final[str] = "BUILD_LOOP_START"
"""Command to start the autonomous build loop."""

BUILD_LOOP_CLOSEOUT: Final[str] = "BUILD_LOOP_CLOSEOUT"
"""Command to initiate close-out phase."""

BUILD_LOOP_VERIFY: Final[str] = "BUILD_LOOP_VERIFY"
"""Command to initiate verification phase."""

BUILD_LOOP_FILL: Final[str] = "BUILD_LOOP_FILL"
"""Command to initiate sprint fill phase."""

BUILD_LOOP_CLASSIFY: Final[str] = "BUILD_LOOP_CLASSIFY"
"""Command to initiate ticket classification phase."""

BUILD_LOOP_BUILD: Final[str] = "BUILD_LOOP_BUILD"
"""Command to initiate build dispatch phase."""

# Events (8)
BUILD_LOOP_STARTED: Final[str] = "BUILD_LOOP_STARTED"
"""Event: build loop cycle started."""

BUILD_LOOP_CLOSEOUT_COMPLETED: Final[str] = "BUILD_LOOP_CLOSEOUT_COMPLETED"
"""Event: close-out phase completed."""

BUILD_LOOP_VERIFY_COMPLETED: Final[str] = "BUILD_LOOP_VERIFY_COMPLETED"
"""Event: verification phase completed."""

BUILD_LOOP_FILL_COMPLETED: Final[str] = "BUILD_LOOP_FILL_COMPLETED"
"""Event: sprint fill phase completed."""

BUILD_LOOP_CLASSIFY_COMPLETED: Final[str] = "BUILD_LOOP_CLASSIFY_COMPLETED"
"""Event: ticket classification phase completed."""

BUILD_LOOP_BUILD_COMPLETED: Final[str] = "BUILD_LOOP_BUILD_COMPLETED"
"""Event: build dispatch phase completed."""

BUILD_LOOP_CYCLE_COMPLETED: Final[str] = "BUILD_LOOP_CYCLE_COMPLETED"
"""Event: full build loop cycle completed."""

BUILD_LOOP_FAILED: Final[str] = "BUILD_LOOP_FAILED"
"""Event: build loop cycle failed (circuit breaker or unrecoverable error)."""


__all__: list[str] = [
    "AGENT_STATUS",
    "BASELINES_COMPUTED",
    "BUILD_LOOP_BUILD",
    "BUILD_LOOP_BUILD_COMPLETED",
    "BUILD_LOOP_CLASSIFY",
    "BUILD_LOOP_CLASSIFY_COMPLETED",
    "BUILD_LOOP_CLOSEOUT",
    "BUILD_LOOP_CLOSEOUT_COMPLETED",
    "BUILD_LOOP_CYCLE_COMPLETED",
    "BUILD_LOOP_FAILED",
    "BUILD_LOOP_FILL",
    "BUILD_LOOP_FILL_COMPLETED",
    "BUILD_LOOP_START",
    "BUILD_LOOP_STARTED",
    "BUILD_LOOP_VERIFY",
    "BUILD_LOOP_VERIFY_COMPLETED",
    "CIRCUIT_BREAKER_STATE",
    "CONSUMER_HEALTH",
    "CONSUMER_RESTART_CMD",
    "EFFECTIVENESS_INVALIDATION",
    "ERROR_TRIAGED",
    "EVAL_COMPLETED",
    "HOOK_CONTEXT_INJECTED",
    "INJECTION_AGENT_MATCH",
    "INJECTION_CONTEXT_ENRICHMENT",
    "INJECTION_CONTEXT_UTILIZATION",
    "INJECTION_LATENCY_BREAKDOWN",
    "INJECTION_RECORDED",
    "MANIFEST_INJECTED",
    "MANIFEST_INJECTION_FAILED",
    "MANIFEST_INJECTION_STARTED",
    "PATTERN_ENFORCEMENT",
    "LLM_CALL_COMPLETED",
    "LLM_CALL_COMPLETED_INFRA",
    "LLM_EMBEDDING_REQUEST",
    "LLM_ENDPOINT_HEALTH",
    "LLM_INFERENCE_REQUEST",
    "RESOLUTION_DECIDED",
    "ROUTING_DECIDED",
    "REWARD_ASSIGNED",
    "RUNTIME_ERROR",
    "SAVINGS_ESTIMATED",
    "SESSION_OUTCOME_CANONICAL",
    "SESSION_OUTCOME_CURRENT",
    "VALIDATOR_CATCH",
    "WAITLIST_SIGNUP",
    "WIRING_HEALTH_SNAPSHOT",
]
