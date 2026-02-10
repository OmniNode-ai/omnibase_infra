"""Platform-reserved topic suffixes for ONEX infrastructure.

WARNING: These are platform-reserved suffixes. Domain services must NOT
import from this module. Domain topics should be defined in domain contracts.

Topic Suffix Format:
    onex.<kind>.<domain>.<event-name>.v<version>

    Structure:
        - onex: Required prefix for all ONEX topics
        - kind: Message category (evt, cmd, intent, snapshot, dlq)
        - domain: Routing domain -- ``platform`` for infrastructure
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
)
"""Complete tuple of all platform topic specs with per-topic configuration.

Each spec defines the topic suffix, partition count, replication factor, and
optional Kafka config overrides. TopicProvisioner iterates this registry
to create topics on startup.
"""

# =============================================================================
# AGGREGATE SUFFIX TUPLE (derived from specs for backwards compat)
# =============================================================================

ALL_PLATFORM_SUFFIXES: tuple[str, ...] = tuple(
    spec.suffix for spec in ALL_PLATFORM_TOPIC_SPECS
)
"""Complete tuple of all platform-reserved topic suffixes.

Derived from ALL_PLATFORM_TOPIC_SPECS for backwards compatibility with
validation code that iterates suffix strings.
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
    for suffix in ALL_PLATFORM_SUFFIXES:
        result = validate_topic_suffix(suffix)
        if not result.is_valid:
            raise OnexError(f"Invalid platform topic suffix '{suffix}': {result.error}")


# Run validation at import time
_validate_all_suffixes()
