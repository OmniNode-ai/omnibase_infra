# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""
Runtime Heartbeat Refusal Reason Enumeration.

Defines the reasons a runtime refuses to start because it cannot heartbeat
(OMN-24496). Every runtime heartbeats: a runtime that cannot build its
introspection service, has an empty name, or cannot start its heartbeat task
refuses to start, and nothing is announced for it.

Thread Safety:
    All enums in this module are immutable and thread-safe.
    Enum values can be safely shared across threads without synchronization.
"""

from enum import Enum, unique


@unique
class EnumRuntimeHeartbeatRefusalReason(str, Enum):
    """
    Reasons a runtime refuses to start because it cannot heartbeat.

    Used by ModelRuntimeHeartbeatRefusal and RuntimeHeartbeatRequiredError to
    categorize why a runtime's startup was refused.

    Values:
        RUNTIME_NAME_EMPTY: The runtime's configured name is empty or whitespace.
        INTROSPECTION_SERVICE_UNBUILDABLE: The introspection service is unbuildable.
        HEARTBEAT_START_FAILED: The periodic heartbeat task could not be started.

    Example:
        >>> reason = EnumRuntimeHeartbeatRefusalReason.HEARTBEAT_START_FAILED
        >>> str(reason)
        'heartbeat_start_failed'
    """

    RUNTIME_NAME_EMPTY = "runtime_name_empty"
    """The runtime's configured name is empty or whitespace."""

    INTROSPECTION_SERVICE_UNBUILDABLE = "introspection_service_unbuildable"
    """The introspection service could not be built from the contracts."""

    HEARTBEAT_START_FAILED = "heartbeat_start_failed"
    """The periodic heartbeat task could not be started."""

    def __str__(self) -> str:
        """Return the string value for serialization."""
        return self.value


__all__: list[str] = ["EnumRuntimeHeartbeatRefusalReason"]
