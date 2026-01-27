"""Platform-reserved topic suffixes for ONEX infrastructure.

WARNING: These are platform-reserved suffixes. Domain services must NOT
import from this module. Domain topics should be defined in domain contracts.

Topic Suffix Format:
    onex.<kind>.<producer>.<event-name>.v<version>

    Structure:
        - onex: Required prefix for all ONEX topics
        - kind: Message category (evt, cmd, intent, snapshot, dlq)
        - producer: Service/module that produces the message
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
"""

from omnibase_core.errors import OnexError
from omnibase_core.validation import validate_topic_suffix

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

# =============================================================================
# AGGREGATE TUPLE
# =============================================================================

ALL_PLATFORM_SUFFIXES: tuple[str, ...] = (
    SUFFIX_NODE_REGISTRATION,
    SUFFIX_NODE_INTROSPECTION,
    SUFFIX_NODE_HEARTBEAT,
    SUFFIX_REQUEST_INTROSPECTION,
    SUFFIX_FSM_STATE_TRANSITIONS,
    SUFFIX_RUNTIME_TICK,
    SUFFIX_REGISTRATION_SNAPSHOTS,
)
"""Complete tuple of all platform-reserved topic suffixes.

Use this tuple for:
    - Validating that domain topics don't conflict with platform topics
    - Iterating over all platform topics for subscription setup
    - Documentation and discovery
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
