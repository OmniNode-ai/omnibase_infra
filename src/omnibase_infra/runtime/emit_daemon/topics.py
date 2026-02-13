# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Centralized Kafka topic constants and pre-built registrations for the emit daemon.

All topic strings used by the emit daemon and notification infrastructure
are defined here as module-level constants. This avoids duplication and
provides a single place to update topic names.

Domain-specific ``ModelEventRegistration`` constants that reference these
topics also live here, keeping event_registry.py a generic utility with
no domain-specific knowledge.
"""

from omnibase_infra.runtime.emit_daemon.event_registry import ModelEventRegistration

TOPIC_PHASE_METRICS = "onex.evt.omniclaude.phase-metrics.v1"
TOPIC_NOTIFICATION_BLOCKED = "onex.evt.omniclaude.notification-blocked.v1"
TOPIC_NOTIFICATION_COMPLETED = "onex.evt.omniclaude.notification-completed.v1"

PHASE_METRICS_REGISTRATION = ModelEventRegistration(
    event_type="phase.metrics",
    topic_template=TOPIC_PHASE_METRICS,
    partition_key_field="run_id",
    required_fields=["run_id", "phase"],
    schema_version="1.0.0",
)

# All known event registrations for omnibase_infra.
# Add new registrations to this tuple. CLI stamp/verify and startup
# validation all use this list as the canonical source of truth.
ALL_EVENT_REGISTRATIONS: tuple[ModelEventRegistration, ...] = (
    PHASE_METRICS_REGISTRATION,
)

__all__: list[str] = [
    "ALL_EVENT_REGISTRATIONS",
    "TOPIC_PHASE_METRICS",
    "TOPIC_NOTIFICATION_BLOCKED",
    "TOPIC_NOTIFICATION_COMPLETED",
    "PHASE_METRICS_REGISTRATION",
]
