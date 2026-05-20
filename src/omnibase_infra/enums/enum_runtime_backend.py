# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime backend selection enumeration.

Defines the canonical backend modes for node invocation dispatch.
Used by NodeInvocationAdapter to select between deployed Kafka runtime
and local in-memory runtime when the deployed runtime is unavailable.

Probing:
    Callers should probe the deployed runtime via health check before
    selecting a backend.  Auto mode performs this probe on each dispatch.

Usage:
    LOCAL  - force in-memory event bus + local state store (no Kafka required)
    DEPLOYED - force deployed Kafka runtime (fail if unavailable)
    AUTO - probe deployed runtime; fall back to LOCAL when unreachable
"""

from enum import Enum


class EnumRuntimeBackend(str, Enum):
    """Node invocation backend selection.

    Attributes:
        LOCAL: Use in-memory event bus and local state store.
            Preserves contract/topic/payload semantics without Kafka.
        DEPLOYED: Use the deployed Kafka-backed runtime.
            Raises if runtime is unreachable.
        AUTO: Probe deployed runtime health; fall back to LOCAL
            when the runtime is not reachable.
    """

    LOCAL = "local"
    DEPLOYED = "deployed"
    AUTO = "auto"


__all__ = ["EnumRuntimeBackend"]
