# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Protocols for NodeRegistryEffect.

Migrated from: omnibase_infra.nodes.effects (OMN-3989)
"""

from __future__ import annotations

from omnibase_infra.nodes.node_registry_effect.protocols.protocol_effect_idempotency_store import (
    ProtocolEffectIdempotencyStore,
)
from omnibase_infra.nodes.node_registry_effect.protocols.protocol_postgres_adapter import (
    ProtocolPostgresAdapter,
)

__all__ = [
    "ProtocolEffectIdempotencyStore",
    "ProtocolPostgresAdapter",
]
