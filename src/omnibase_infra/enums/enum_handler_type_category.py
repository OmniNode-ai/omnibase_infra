# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Type Category Enumeration for behavioral classification.

Defines the behavioral classification of handlers for ONEX architecture.
This enum is orthogonal to EnumHandlerType (architectural role) - both must
be specified on handler descriptors.

Classification Guide:
    | Category | Deterministic? | Side Effects? | Examples |
    |----------|----------------|---------------|----------|
    | COMPUTE | Yes | No | Validation, transformation, mapping |
    | EFFECT | N/A | Yes | DB, HTTP, Consul, Vault, Kafka, LLM calls |
    | NONDETERMINISTIC_COMPUTE | No | No | UUID generation, datetime.now(), random.choice() |

Note: LLM API calls are EFFECT (external I/O), not NONDETERMINISTIC_COMPUTE.

See Also:
    - EnumHandlerType: Architectural role classification (INFRA_HANDLER, NODE_HANDLER, etc.)
    - EnumNodeArchetype: Node archetypes for execution shape validation (EFFECT, COMPUTE, etc.)
    - HANDLER_PROTOCOL_DRIVEN_ARCHITECTURE.md: Full architecture documentation

Note on Naming Consistency:
    EnumHandlerTypeCategory and EnumNodeArchetype both use EFFECT and COMPUTE values but serve
    different purposes:
    - EnumHandlerTypeCategory: Classifies handler BEHAVIOR (side effects, determinism)
    - EnumNodeArchetype: Classifies node ARCHITECTURE (execution shape validation)
    The shared terminology reflects that EFFECT handlers are typically bound to EFFECT nodes,
    and COMPUTE handlers to COMPUTE nodes, but they represent different conceptual axes.
"""

from enum import Enum


class EnumHandlerTypeCategory(str, Enum):
    """Handler behavioral classification - selects policy envelope.

    Determines how the handler behaves at runtime and what policies apply.
    Drives: Security rules, determinism guarantees, replay safety, permissions.

    Note: ADAPTER is NOT a category - it's a policy tag (is_adapter: bool) on
    handler descriptors. Adapters are behaviorally EFFECT but have stricter defaults.

    Attributes:
        COMPUTE: Pure, deterministic computation. No side effects.
            Examples: Validation, transformation, mapping, calculations.
            Safe for replay, caching, and parallel execution.
        EFFECT: Side-effecting I/O operations. May not be deterministic.
            Examples: Database operations, HTTP calls, Consul, Vault, Kafka, LLM APIs.
            Requires idempotency handling for replay safety.
        NONDETERMINISTIC_COMPUTE: Pure (no I/O) but not deterministic.
            Examples: UUID generation, datetime.now(), random.choice().
            No external side effects but results may vary between runs.
    """

    COMPUTE = "compute"
    EFFECT = "effect"
    NONDETERMINISTIC_COMPUTE = "nondeterministic_compute"


__all__ = ["EnumHandlerTypeCategory"]
