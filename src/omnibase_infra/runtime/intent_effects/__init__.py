# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Intent effect adapters for contract-driven intent execution.

This package provides intent effect adapters that bridge intent payloads
(produced by handlers/reducers) to actual infrastructure operations. Each
adapter accepts a typed intent payload and executes the corresponding
effect-layer operation (PostgreSQL upsert, Consul registration, etc.).

Architecture:
    Handler -> ModelIntent(payload) -> IntentExecutor -> IntentEffect -> Infrastructure

    IntentEffects are registered with IntentExecutor via the contract's
    intent_routing_table. Each effect adapter exposes an async ``execute()``
    method matching the ProtocolIntentEffect protocol.

Related:
    - OMN-2050: Wire MessageDispatchEngine as single consumer path
    - IntentExecutor: Routes intents to effect adapters
    - contract.yaml: intent_consumption.intent_routing_table section

.. versionadded:: 0.7.0
"""

from omnibase_infra.runtime.intent_effects.intent_effect_consul_register import (
    IntentEffectConsulRegister,
)
from omnibase_infra.runtime.intent_effects.intent_effect_postgres_upsert import (
    IntentEffectPostgresUpsert,
)

__all__: list[str] = [
    "IntentEffectConsulRegister",
    "IntentEffectPostgresUpsert",
]
