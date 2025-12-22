# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol definitions for omnibase_infra.

This module provides protocol definitions (duck-typed interfaces) for infrastructure
components in the ONEX ecosystem.

Protocols:
    - ProtocolPluginCompute: Interface for deterministic compute plugins
    - ProtocolSnapshotPublisher: Interface for snapshot publishing services (F2)

Architecture:
    Protocols enable duck typing and dependency injection without requiring
    inheritance. Classes implementing a protocol are automatically recognized
    through structural typing (matching method signatures).

Usage:
    ```python
    from omnibase_infra.protocols import ProtocolPluginCompute, ProtocolSnapshotPublisher

    # Check if class implements protocol
    plugin = MyComputePlugin()
    assert isinstance(plugin, ProtocolPluginCompute)  # Runtime check

    publisher = MySnapshotPublisher()
    assert isinstance(publisher, ProtocolSnapshotPublisher)  # Runtime check
    ```

See Also:
    - omnibase_infra.plugins for base class implementations
    - omnibase_infra.models.projection for projection models
    - ONEX 4-node architecture documentation
    - OMN-947 (F2) for snapshot publishing design
"""

from omnibase_infra.protocols.protocol_idempotency_store import (
    ProtocolIdempotencyStore,
)
from omnibase_infra.protocols.protocol_plugin_compute import ProtocolPluginCompute
from omnibase_infra.protocols.protocol_snapshot_publisher import (
    ProtocolSnapshotPublisher,
)

__all__ = [
    "ProtocolIdempotencyStore",
    "ProtocolPluginCompute",
    "ProtocolSnapshotPublisher",
]
