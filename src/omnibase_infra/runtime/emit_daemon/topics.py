# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Centralized Kafka topic constants and pre-built registrations for the emit daemon.

Topic strings are sourced from the canonical contract surfaces -- the
``SUFFIX_OMNICLAUDE_*`` constants in ``platform_topic_suffixes`` for
omniclaude-produced topics and ``EnumOmnibaseInfraTopic`` for omnibase_infra
topics -- so this module holds no raw topic literals (OMN-13700, OMN-3343).
This keeps the contract-driven topic registry the single source of truth.

Domain-specific ``ModelEventRegistration`` constants that reference these
topics also live here, keeping event_registry.py a generic utility with
no domain-specific knowledge.
"""

from omnibase_infra.enums.generated.enum_omnibase_infra_topic import (
    EnumOmnibaseInfraTopic,
)
from omnibase_infra.runtime.emit_daemon.event_registry import ModelEventRegistration
from omnibase_infra.topics.platform_topic_suffixes import (
    SUFFIX_OMNICLAUDE_NOTIFICATION_BLOCKED,
    SUFFIX_OMNICLAUDE_NOTIFICATION_COMPLETED,
    SUFFIX_OMNICLAUDE_PHASE_METRICS,
)

TOPIC_PHASE_METRICS = SUFFIX_OMNICLAUDE_PHASE_METRICS
# Consumed (not emitted) by omnibase_infra, so no ModelEventRegistration needed.
TOPIC_NOTIFICATION_BLOCKED = SUFFIX_OMNICLAUDE_NOTIFICATION_BLOCKED
TOPIC_NOTIFICATION_COMPLETED = SUFFIX_OMNICLAUDE_NOTIFICATION_COMPLETED

TOPIC_BASELINES_COMPUTED = EnumOmnibaseInfraTopic.EVT_BASELINES_COMPUTED_V1.value

PHASE_METRICS_REGISTRATION = ModelEventRegistration(
    event_type="phase.metrics",
    topic_template=TOPIC_PHASE_METRICS,
    partition_key_field="run_id",
    required_fields=("run_id", "phase"),
    schema_version="1.0.0",
)

BASELINES_COMPUTED_REGISTRATION = ModelEventRegistration(
    event_type="omnibase-infra.baselines.computed",
    topic_template=TOPIC_BASELINES_COMPUTED,
    partition_key_field="snapshot_id",
    required_fields=("snapshot_id", "contract_version", "computed_at_utc"),
    schema_version="1.0.0",
)

TCB_OUTCOME_REGISTRATION = ModelEventRegistration(
    event_type="tcb.outcome",
    topic_template=TOPIC_PHASE_METRICS,
    partition_key_field="ticket_id",
    required_fields=("ticket_id", "outcome"),
    schema_version="1.0.0",
)

# All known event registrations for omnibase_infra.
# Add new registrations to this tuple. CLI stamp/verify and startup
# validation all use this list as the canonical source of truth.
ALL_EVENT_REGISTRATIONS: tuple[ModelEventRegistration, ...] = (
    PHASE_METRICS_REGISTRATION,
    BASELINES_COMPUTED_REGISTRATION,
    TCB_OUTCOME_REGISTRATION,
)

__all__: list[str] = [
    "ALL_EVENT_REGISTRATIONS",
    "BASELINES_COMPUTED_REGISTRATION",
    "TCB_OUTCOME_REGISTRATION",
    "TOPIC_BASELINES_COMPUTED",
    "TOPIC_PHASE_METRICS",
    "TOPIC_NOTIFICATION_BLOCKED",
    "TOPIC_NOTIFICATION_COMPLETED",
    "PHASE_METRICS_REGISTRATION",
]
