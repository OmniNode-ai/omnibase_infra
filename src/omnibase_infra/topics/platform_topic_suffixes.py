"""Platform and domain topic suffixes for ONEX infrastructure.

This module defines topic suffixes for:
  1. Platform-reserved topics (producer: ``platform``) -- infrastructure internals
  2. Domain plugin topics (producer: ``omniintelligence``, ``pattern``, etc.) --
     provisioned by the runtime so domain plugins find their topics ready.

Domain services should NOT import individual suffix constants from this module.
They should subscribe to topics by name from their own contracts. The combined
``ALL_PROVISIONED_TOPIC_SPECS`` registry is consumed by ``TopicProvisioner`` at
startup to create all required topics.

Topic Suffix Format:
    onex.<kind>.<producer>.<event-name>.v<version>

    Structure:
        - onex: Required prefix for all ONEX topics
        - kind: Message category (evt, cmd, intent, snapshot, dlq)
        - producer: Routing domain -- ``platform`` for infrastructure,
          domain name for plugins (e.g., ``omniintelligence``)
        - event-name: Descriptive name using kebab-case
        - version: Semantic version (v1, v2, etc.)

    Kinds:
        evt - Event topics (state changes, notifications)
        cmd - Command topics (requests for action)
        intent - Intent topics (internal workflow coordination)
        snapshot - Snapshot topics (periodic state snapshots)
        dlq - Dead letter queue topics

    Examples:
        onex.evt.platform.node-registration.v1
        onex.cmd.platform.request-introspection.v1
        onex.intent.platform.runtime-tick.v1
        onex.cmd.omniintelligence.claude-hook-event.v1

Usage:
    from omnibase_infra.topics import SUFFIX_NODE_REGISTRATION

    # Compose full topic with tenant/namespace prefix
    full_topic = f"{tenant}.{namespace}.{SUFFIX_NODE_REGISTRATION}"

See Also:
    omnibase_core.validation.validate_topic_suffix - Validation function
    omnibase_core.validation.compose_full_topic - Topic composition utility
    model_topic_spec.ModelTopicSpec - Per-topic creation spec
"""

from __future__ import annotations

from omnibase_core.errors import OnexError
from omnibase_core.validation import validate_topic_suffix
from omnibase_infra.topics.model_topic_spec import ModelTopicSpec

# =============================================================================
# PLATFORM-RESERVED TOPIC SUFFIXES
# =============================================================================

# Node lifecycle events
SUFFIX_NODE_REGISTRATION: str = "onex.evt.platform.node-registration.v1"
"""Topic suffix for node registration events.

Published when a node registers with the runtime. Contains node metadata,
capabilities, and health check configuration.
"""

SUFFIX_NODE_INTROSPECTION: str = "onex.evt.platform.node-introspection.v1"
"""Topic suffix for node introspection events.

Published when a node responds to an introspection request. Contains node
capabilities, supported operations, and current state.
"""

SUFFIX_REGISTRY_REQUEST_INTROSPECTION: str = (
    "onex.evt.platform.registry-request-introspection.v1"
)
"""Topic suffix for registry-initiated introspection request events.

Published when the registry requests introspection from nodes during the
registration workflow. The registration orchestrator subscribes to this topic
to trigger node registration processing.
"""

SUFFIX_NODE_HEARTBEAT: str = "onex.evt.platform.node-heartbeat.v1"
"""Topic suffix for node heartbeat events.

Published periodically by nodes to indicate liveness. Contains timestamp,
resource usage metrics, and health status.
"""

# Command topics
SUFFIX_REQUEST_INTROSPECTION: str = "onex.cmd.platform.request-introspection.v1"
"""Topic suffix for introspection request commands.

Published to request introspection from a specific node or all nodes.
Nodes respond on the SUFFIX_NODE_INTROSPECTION topic.
"""

# FSM and state management
SUFFIX_FSM_STATE_TRANSITIONS: str = "onex.evt.platform.fsm-state-transitions.v1"
"""Topic suffix for FSM state transition events.

Published when a node's finite state machine transitions between states.
Contains previous state, new state, trigger event, and transition metadata.
"""

# Runtime coordination
SUFFIX_RUNTIME_TICK: str = "onex.intent.platform.runtime-tick.v1"
"""Topic suffix for runtime tick intents.

Internal topic for runtime orchestration. Triggers periodic tasks like
heartbeat collection, health checks, and scheduled workflows.
"""

# Registration snapshots
SUFFIX_REGISTRATION_SNAPSHOTS: str = "onex.snapshot.platform.registration-snapshots.v1"
"""Topic suffix for registration snapshot events.

Published periodically with aggregated registration state. Used for
dashboard displays and monitoring systems.
"""

# Contract lifecycle events (used by ContractRegistrationEventRouter in kernel)
SUFFIX_CONTRACT_REGISTERED: str = "onex.evt.platform.contract-registered.v1"
"""Topic suffix for contract registration events.

Published when a node contract is registered with the runtime.
"""

SUFFIX_CONTRACT_DEREGISTERED: str = "onex.evt.platform.contract-deregistered.v1"
"""Topic suffix for contract deregistration events.

Published when a node contract is deregistered from the runtime.
"""

# Registration ACK commands
SUFFIX_NODE_REGISTRATION_ACKED: str = "onex.cmd.platform.node-registration-acked.v1"
"""Topic suffix for node registration ACK commands.

Published by a node after it receives a registration-accepted event,
confirming that the node acknowledges successful registration.
"""

# =============================================================================
# INTELLIGENCE DOMAIN TOPIC SUFFIXES (omniintelligence plugin)
# =============================================================================
# These topics are consumed/produced by PluginIntelligence. They are provisioned
# alongside platform topics so the plugin finds them ready at startup.

# Command topics (inbound to intelligence pipeline)
SUFFIX_INTELLIGENCE_CLAUDE_HOOK_EVENT: str = (
    "onex.cmd.omniintelligence.claude-hook-event.v1"
)
"""Topic for Claude hook events dispatched to the intelligence pipeline."""

SUFFIX_INTELLIGENCE_SESSION_OUTCOME: str = (
    "onex.cmd.omniintelligence.session-outcome.v1"
)
"""Topic for session outcome commands (success/failure attribution)."""

SUFFIX_INTELLIGENCE_PATTERN_LIFECYCLE_TRANSITION: str = (
    "onex.cmd.omniintelligence.pattern-lifecycle-transition.v1"
)
"""Topic for pattern lifecycle transition commands."""

# Event topics (outbound from intelligence pipeline)
SUFFIX_INTELLIGENCE_INTENT_CLASSIFIED: str = (
    "onex.evt.omniintelligence.intent-classified.v1"
)
"""Topic for intent classification events."""

SUFFIX_INTELLIGENCE_PATTERN_LEARNED: str = (
    "onex.evt.omniintelligence.pattern-learned.v1"
)
"""Topic for pattern learning events (new pattern discovered)."""

SUFFIX_INTELLIGENCE_PATTERN_STORED: str = "onex.evt.omniintelligence.pattern-stored.v1"
"""Topic for pattern storage events (pattern persisted to DB)."""

SUFFIX_INTELLIGENCE_PATTERN_PROMOTED: str = (
    "onex.evt.omniintelligence.pattern-promoted.v1"
)
"""Topic for pattern promotion events (candidate -> validated)."""

SUFFIX_INTELLIGENCE_PATTERN_LIFECYCLE_TRANSITIONED: str = (
    "onex.evt.omniintelligence.pattern-lifecycle-transitioned.v1"
)
"""Topic for pattern lifecycle transition completion events."""

SUFFIX_INTELLIGENCE_LLM_CALL_COMPLETED: str = (
    "onex.evt.omniintelligence.llm-call-completed.v1"
)
"""Topic for LLM call completed metrics events.

Published by LLM inference handlers after each call. Contains per-call
token counts, cost, and latency for the cost aggregation pipeline.
"""

SUFFIX_INTELLIGENCE_PATTERN_DISCOVERED: str = "onex.evt.pattern.discovered.v1"
"""Topic for generic pattern discovery events."""

# =============================================================================
# TOPIC CATALOG TOPIC SUFFIXES
# =============================================================================

SUFFIX_TOPIC_CATALOG_QUERY: str = "onex.cmd.platform.topic-catalog-query.v1"
"""Topic suffix for topic catalog query commands.

Published when a client requests the current topic catalog. Contains optional
filters (topic_pattern, include_inactive) and a correlation_id for
request-response matching.
"""

SUFFIX_TOPIC_CATALOG_RESPONSE: str = "onex.evt.platform.topic-catalog-response.v1"
"""Topic suffix for topic catalog response events.

Published in response to a catalog query. Contains the full list of topic
entries with publisher/subscriber counts, plus catalog metadata.
"""

SUFFIX_TOPIC_CATALOG_CHANGED: str = "onex.evt.platform.topic-catalog-changed.v1"
"""Topic suffix for topic catalog change notification events.

Published when topics are added or removed from the catalog. Contains
delta tuples (topics_added, topics_removed) sorted alphabetically for
deterministic ordering.
"""

# =============================================================================
# PLATFORM TOPIC SPEC REGISTRY
# =============================================================================


# Build snapshot topic kafka_config from ModelSnapshotTopicConfig.default().
# Deferred import to avoid circular dependency; lazy initialization is safe
# because this module is only imported at startup.
def _snapshot_kafka_config() -> dict[str, str]:
    """Build Kafka config for the snapshot topic from ModelSnapshotTopicConfig."""
    from omnibase_infra.models.projection.model_snapshot_topic_config import (
        ModelSnapshotTopicConfig,
    )

    return ModelSnapshotTopicConfig.default().to_kafka_config()


ALL_PLATFORM_TOPIC_SPECS: tuple[ModelTopicSpec, ...] = (
    ModelTopicSpec(suffix=SUFFIX_NODE_REGISTRATION, partitions=6),
    ModelTopicSpec(suffix=SUFFIX_NODE_INTROSPECTION, partitions=6),
    ModelTopicSpec(suffix=SUFFIX_REGISTRY_REQUEST_INTROSPECTION, partitions=6),
    ModelTopicSpec(suffix=SUFFIX_NODE_HEARTBEAT, partitions=6),
    ModelTopicSpec(suffix=SUFFIX_REQUEST_INTROSPECTION, partitions=6),
    ModelTopicSpec(suffix=SUFFIX_FSM_STATE_TRANSITIONS, partitions=6),
    ModelTopicSpec(suffix=SUFFIX_RUNTIME_TICK, partitions=1),
    ModelTopicSpec(
        suffix=SUFFIX_REGISTRATION_SNAPSHOTS,
        partitions=1,
        kafka_config=_snapshot_kafka_config(),
    ),
    ModelTopicSpec(suffix=SUFFIX_CONTRACT_REGISTERED, partitions=6),
    ModelTopicSpec(suffix=SUFFIX_CONTRACT_DEREGISTERED, partitions=6),
    ModelTopicSpec(suffix=SUFFIX_NODE_REGISTRATION_ACKED, partitions=6),
    # Topic catalog topics (low-throughput coordination, 1 partition each)
    ModelTopicSpec(
        suffix=SUFFIX_TOPIC_CATALOG_QUERY,
        partitions=1,
        kafka_config={"retention.ms": "3600000", "cleanup.policy": "delete"},
    ),
    ModelTopicSpec(
        suffix=SUFFIX_TOPIC_CATALOG_RESPONSE,
        partitions=1,
        kafka_config={"retention.ms": "3600000", "cleanup.policy": "delete"},
    ),
    ModelTopicSpec(
        suffix=SUFFIX_TOPIC_CATALOG_CHANGED,
        partitions=1,
        kafka_config={"retention.ms": "604800000", "cleanup.policy": "delete"},
    ),
)
"""Complete tuple of all platform topic specs with per-topic configuration.

Each spec defines the topic suffix, partition count, replication factor, and
optional Kafka config overrides. TopicProvisioner iterates this registry
to create topics on startup.
"""

# =============================================================================
# INTELLIGENCE DOMAIN TOPIC SPEC REGISTRY
# =============================================================================

ALL_INTELLIGENCE_TOPIC_SPECS: tuple[ModelTopicSpec, ...] = (
    # Command topics (3 partitions each — matches e2e compose)
    ModelTopicSpec(suffix=SUFFIX_INTELLIGENCE_CLAUDE_HOOK_EVENT, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_INTELLIGENCE_SESSION_OUTCOME, partitions=3),
    ModelTopicSpec(
        suffix=SUFFIX_INTELLIGENCE_PATTERN_LIFECYCLE_TRANSITION, partitions=3
    ),
    # Event topics (3 partitions each)
    ModelTopicSpec(suffix=SUFFIX_INTELLIGENCE_INTENT_CLASSIFIED, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_INTELLIGENCE_PATTERN_LEARNED, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_INTELLIGENCE_PATTERN_STORED, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_INTELLIGENCE_PATTERN_PROMOTED, partitions=3),
    ModelTopicSpec(
        suffix=SUFFIX_INTELLIGENCE_PATTERN_LIFECYCLE_TRANSITIONED, partitions=3
    ),
    ModelTopicSpec(suffix=SUFFIX_INTELLIGENCE_PATTERN_DISCOVERED, partitions=3),
    ModelTopicSpec(suffix=SUFFIX_INTELLIGENCE_LLM_CALL_COMPLETED, partitions=3),
)
"""Intelligence domain topic specs provisioned for PluginIntelligence."""

# =============================================================================
# COMBINED PROVISIONED TOPIC SPECS
# =============================================================================

ALL_PROVISIONED_TOPIC_SPECS: tuple[ModelTopicSpec, ...] = (
    ALL_PLATFORM_TOPIC_SPECS + ALL_INTELLIGENCE_TOPIC_SPECS
)
"""All topic specs to be provisioned by TopicProvisioner at startup.

Combines platform-reserved and domain plugin topic specs into a single
registry consumed by service_topic_manager.py. This is the single source
of truth for topic creation.
"""

# =============================================================================
# AGGREGATE SUFFIX TUPLES
# =============================================================================

ALL_PLATFORM_SUFFIXES: tuple[str, ...] = tuple(
    spec.suffix for spec in ALL_PLATFORM_TOPIC_SPECS
)
"""Complete tuple of all platform-reserved topic suffixes.

Derived from ALL_PLATFORM_TOPIC_SPECS for backwards compatibility with
validation code that iterates suffix strings.
"""

ALL_PROVISIONED_SUFFIXES: tuple[str, ...] = tuple(
    spec.suffix for spec in ALL_PROVISIONED_TOPIC_SPECS
)
"""Complete tuple of all provisioned topic suffixes (platform + domain).

Derived from ALL_PROVISIONED_TOPIC_SPECS. Includes both platform-reserved
and domain plugin topics.
"""

# =============================================================================
# IMPORT-TIME VALIDATION
# =============================================================================


def _validate_all_suffixes() -> None:
    """Validate all suffixes at import time to fail fast on invalid format.

    Raises:
        OnexError: If any suffix fails validation with details about which
            suffix failed and why.
    """
    for suffix in ALL_PROVISIONED_SUFFIXES:
        result = validate_topic_suffix(suffix)
        if not result.is_valid:
            raise OnexError(f"Invalid topic suffix '{suffix}': {result.error}")


# Run validation at import time
_validate_all_suffixes()
