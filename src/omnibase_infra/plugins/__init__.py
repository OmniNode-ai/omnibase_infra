# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Compute plugin infrastructure for deterministic business logic.

The foundation for in-process, deterministic computation
that complements external effect handlers.

Components:
    - PluginComputeBase: Abstract base class with validation hooks

Architecture:
    Compute plugins perform pure computation with NO side effects:
    - NO external I/O (network, filesystem, database)
    - NO random number generation (unless seeded)
    - NO current time access (unless passed in context)
    - Deterministic: same inputs = same outputs
    - Replayable for debugging and testing

See Also:
    - omnibase_infra.protocols.ProtocolPluginCompute for protocol definition
"""

from omnibase_infra.plugins.plugin_compute_base import PluginComputeBase

__all__ = [
    "PluginComputeBase",
]
