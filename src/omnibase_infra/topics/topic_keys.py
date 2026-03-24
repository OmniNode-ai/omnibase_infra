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

# ==============================================================================
# LLM Topics
# ==============================================================================

LLM_CALL_COMPLETED: Final[str] = "LLM_CALL_COMPLETED"
"""LLM call completed metrics from omniintelligence."""

LLM_CALL_COMPLETED_INFRA: Final[str] = "LLM_CALL_COMPLETED_INFRA"
"""LLM call completed event from omnibase-infra inference effect."""

LLM_ENDPOINT_HEALTH: Final[str] = "LLM_ENDPOINT_HEALTH"
"""LLM endpoint health probe events."""

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


__all__: list[str] = [
    "AGENT_STATUS",
    "CIRCUIT_BREAKER_STATE",
    "CONSUMER_HEALTH",
    "CONSUMER_RESTART_CMD",
    "EFFECTIVENESS_INVALIDATION",
    "HOOK_CONTEXT_INJECTED",
    "INJECTION_AGENT_MATCH",
    "INJECTION_CONTEXT_UTILIZATION",
    "INJECTION_LATENCY_BREAKDOWN",
    "LLM_CALL_COMPLETED",
    "LLM_CALL_COMPLETED_INFRA",
    "LLM_ENDPOINT_HEALTH",
    "RESOLUTION_DECIDED",
    "REWARD_ASSIGNED",
    "RUNTIME_ERROR",
    "SAVINGS_ESTIMATED",
    "SESSION_OUTCOME_CANONICAL",
    "SESSION_OUTCOME_CURRENT",
    "VALIDATOR_CATCH",
    "WIRING_HEALTH_SNAPSHOT",
]
