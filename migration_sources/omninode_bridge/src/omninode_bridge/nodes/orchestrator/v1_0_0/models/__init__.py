#!/usr/bin/env python3
"""
Models package for NodeBridgeOrchestrator v1.0.0.

Contains Pydantic models and enums for workflow orchestration,
including input/output state models and FSM state management.

ONEX v2.0 Compliance:
- Suffix-based naming conventions
- Strong typing with Pydantic v2
- O.N.E. v0.1 protocol compliance
"""

from typing import TYPE_CHECKING

from .enum_workflow_event import EnumWorkflowEvent
from .enum_workflow_state import EnumWorkflowState
from .model_node_heartbeat_event import EnumNodeHealthStatus, ModelNodeHeartbeatEvent
from .model_node_introspection_event import ModelNodeIntrospectionEvent
from .model_registry_request_event import (
    EnumIntrospectionReason,
    ModelRegistryRequestEvent,
)
from .model_stamp_request_input import ModelStampRequestInput
from .model_stamp_response_output import ModelStampResponseOutput


# Lazy imports for helper functions to avoid triggering registry imports
# These are only needed at runtime when actually creating envelopes
def __getattr__(name):
    """Lazy import of introspection event helpers to avoid import chain issues."""
    if name in (
        "create_heartbeat_envelope",
        "create_node_introspection_envelope",
        "create_registry_request_envelope",
    ):
        from .introspection_event_helpers import (  # noqa: F401
            create_heartbeat_envelope,
            create_node_introspection_envelope,
            create_registry_request_envelope,
        )

        globals()[name] = locals()[name]
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Import ModelOnexEnvelopeV1 only for type checking to avoid circular import
# Runtime usage is in introspection_event_helpers.py which has its own import
if TYPE_CHECKING:
    from omninode_bridge.nodes.registry.v1_0_0.models import (  # noqa: F401
        ModelOnexEnvelopeV1,
    )

__all__ = [
    # Enums
    "EnumWorkflowState",
    "EnumWorkflowEvent",
    "EnumNodeHealthStatus",
    "EnumIntrospectionReason",
    # Input/Output models
    "ModelStampRequestInput",
    "ModelStampResponseOutput",
    # Introspection event models
    "ModelNodeIntrospectionEvent",
    "ModelRegistryRequestEvent",
    "ModelNodeHeartbeatEvent",
    # Helper functions (lazy imported to avoid import chain)
    "create_node_introspection_envelope",
    "create_registry_request_envelope",
    "create_heartbeat_envelope",
]
