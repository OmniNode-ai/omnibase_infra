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
    - EnumHandlerType: Architectural role classification
    - HANDLER_PROTOCOL_DRIVEN_ARCHITECTURE.md: Full architecture documentation
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
