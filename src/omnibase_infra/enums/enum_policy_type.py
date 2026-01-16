# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Policy Type Enumeration.

Defines the canonical policy types for the RegistryPolicy plugin system.
Used to classify and route policies for orchestrator and reducer nodes.
"""

from enum import Enum


class EnumPolicyType(str, Enum):
    """Policy types for RegistryPolicy plugin classification.

    These represent the two primary policy categories in the ONEX
    4-node architecture: orchestrator policies and reducer policies.

    Attributes:
        ORCHESTRATOR: Policies for orchestrator nodes controlling workflow
            execution. Includes step ordering, retry strategies, backoff
            algorithms, routing decisions, timeout handling, and circuit
            breaker patterns.
        REDUCER: Policies for reducer nodes managing state consolidation.
            Includes state merge strategies, conflict resolution, idempotency
            enforcement, projection rules, and data redaction policies.
    """

    ORCHESTRATOR = "orchestrator"
    REDUCER = "reducer"


__all__ = ["EnumPolicyType"]
