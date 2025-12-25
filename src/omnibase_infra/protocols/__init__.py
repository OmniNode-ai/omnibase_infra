# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol definitions for omnibase_infra.

This module provides protocol definitions (duck-typed interfaces) for infrastructure
components in the ONEX ecosystem.

Protocols:
    - ProtocolIdempotencyStore: Interface for idempotency checking and deduplication
    - ProtocolPluginCompute: Interface for deterministic compute plugins
    - ProtocolRegistryMetrics: Interface for registry metrics collection (optional)
    - ProtocolSnapshotPublisher: Interface for snapshot publishing services (F2)

Architecture:
    Protocols enable duck typing and dependency injection without requiring
    inheritance. Classes implementing a protocol are automatically recognized
    through structural typing (matching method signatures).

Usage:
    ```python
    from omnibase_infra.protocols import (
        ProtocolIdempotencyStore,
        ProtocolPluginCompute,
        ProtocolRegistryMetrics,
        ProtocolSnapshotPublisher,
    )

    # Verify protocol compliance via duck typing (per ONEX conventions)
    plugin = MyComputePlugin()
    assert hasattr(plugin, 'execute') and callable(plugin.execute)

    publisher = MySnapshotPublisher()
    assert hasattr(publisher, 'publish_snapshot') and callable(publisher.publish_snapshot)
    assert hasattr(publisher, 'delete_snapshot') and callable(publisher.delete_snapshot)

    store = MyIdempotencyStore()
    assert hasattr(store, 'check_and_record') and callable(store.check_and_record)
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
from omnibase_infra.protocols.protocol_registry_metrics import ProtocolRegistryMetrics
from omnibase_infra.protocols.protocol_snapshot_publisher import (
    ProtocolSnapshotPublisher,
)

__all__ = [
    "ProtocolIdempotencyStore",
    "ProtocolPluginCompute",
    "ProtocolRegistryMetrics",
    "ProtocolSnapshotPublisher",
]
