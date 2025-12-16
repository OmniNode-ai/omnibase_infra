# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol definitions for omnibase_infra.

This module provides protocol definitions (duck-typed interfaces) for infrastructure
components in the ONEX ecosystem.

Protocols:
    - ProtocolComputePlugin: Interface for deterministic compute plugins

Architecture:
    Protocols enable duck typing and dependency injection without requiring
    inheritance. Classes implementing a protocol are automatically recognized
    through structural typing (matching method signatures).

Usage:
    ```python
    from omnibase_infra.protocols import ProtocolComputePlugin

    # Check if class implements protocol
    plugin = MyComputePlugin()
    assert isinstance(plugin, ProtocolComputePlugin)  # Runtime check
    ```

See Also:
    - omnibase_infra.plugins for base class implementations
    - ONEX 4-node architecture documentation
"""

from omnibase_infra.protocols.protocol_compute_plugin import ProtocolComputePlugin

__all__ = [
    "ProtocolComputePlugin",
]
