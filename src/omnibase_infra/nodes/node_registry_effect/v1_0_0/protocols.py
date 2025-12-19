# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Re-export module for backwards compatibility.

This module re-exports protocols and types from their split module locations
to maintain backwards compatibility with existing imports.

The protocols were split into separate files for better organization:
- protocol_types.py: Type aliases (EnvelopeDict, ResultDict)
- protocol_envelope_executor.py: ProtocolEnvelopeExecutor
- protocol_event_bus.py: ProtocolEventBus
"""

from __future__ import annotations

from omnibase_infra.nodes.node_registry_effect.v1_0_0.protocol_envelope_executor import (
    ProtocolEnvelopeExecutor,
)
from omnibase_infra.nodes.node_registry_effect.v1_0_0.protocol_event_bus import (
    ProtocolEventBus,
)
from omnibase_infra.nodes.node_registry_effect.v1_0_0.protocol_types import (
    EnvelopeDict,
    ResultDict,
)

__all__ = [
    "EnvelopeDict",
    "ProtocolEnvelopeExecutor",
    "ProtocolEventBus",
    "ResultDict",
]
