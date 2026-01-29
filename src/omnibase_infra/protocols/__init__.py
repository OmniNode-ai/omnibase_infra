# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol definitions for omnibase_infra.

This module provides protocol definitions (duck-typed interfaces) for infrastructure
components in the ONEX ecosystem.

Protocols:
    - ProtocolCapabilityProjection: Interface for capability-based projection queries
    - ProtocolCapabilityQuery: Interface for capability-based node discovery service
    - ProtocolDispatchEngine: Interface for message dispatch engines
    - ProtocolEventBusLike: Interface for event bus abstraction (used by introspection)
    - ProtocolIdempotencyStore: Interface for idempotency checking and deduplication
    - ProtocolMessageDispatcher: Interface for message dispatchers
    - ProtocolMessageTypeRegistry: Interface for message type registries
    - ProtocolPluginCompute: Interface for deterministic compute plugins
    - ProtocolRegistryMetrics: Interface for registry metrics collection (optional)
    - ProtocolSnapshotPublisher: Interface for snapshot publishing services (F2)
    - ProtocolSnapshotStore: Interface for snapshot storage backends

Note:
    ProtocolCircuitBreakerAware is defined in omnibase_infra.mixins (tightly coupled
    to MixinAsyncCircuitBreaker). Import it from there, not from this package.

Architecture:
    Protocols enable duck typing and dependency injection without requiring
    inheritance. Classes implementing a protocol are automatically recognized
    through structural typing (matching method signatures).

Usage:
    ```python
    from omnibase_infra.protocols import (
        ProtocolCapabilityQuery,
        ProtocolEventBusLike,
        ProtocolIdempotencyStore,
        ProtocolPluginCompute,
        ProtocolRegistryMetrics,
        ProtocolSnapshotPublisher,
        ProtocolSnapshotStore,
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
    - omnibase_infra.mixins for ProtocolCircuitBreakerAware
    - ONEX 4-node architecture documentation
    - OMN-947 (F2) for snapshot publishing design
"""

from omnibase_infra.protocols.protocol_capability_projection import (
    ProtocolCapabilityProjection,
)
from omnibase_infra.protocols.protocol_capability_query import ProtocolCapabilityQuery
from omnibase_infra.protocols.protocol_container_aware import ProtocolContainerAware
from omnibase_infra.protocols.protocol_dispatch_engine import ProtocolDispatchEngine
from omnibase_infra.protocols.protocol_event_bus_like import ProtocolEventBusLike
from omnibase_infra.protocols.protocol_event_projector import ProtocolEventProjector
from omnibase_infra.protocols.protocol_idempotency_store import (
    ProtocolIdempotencyStore,
)
from omnibase_infra.protocols.protocol_message_dispatcher import (
    ProtocolMessageDispatcher,
)
from omnibase_infra.protocols.protocol_message_type_registry import (
    ProtocolMessageTypeRegistry,
)
from omnibase_infra.protocols.protocol_plugin_compute import ProtocolPluginCompute
from omnibase_infra.protocols.protocol_projector_schema_validator import (
    ProtocolProjectorSchemaValidator,
)
from omnibase_infra.protocols.protocol_registry_metrics import ProtocolRegistryMetrics
from omnibase_infra.protocols.protocol_snapshot_publisher import (
    ProtocolSnapshotPublisher,
)
from omnibase_infra.protocols.protocol_snapshot_store import ProtocolSnapshotStore

__all__: list[str] = [
    "ProtocolCapabilityProjection",
    "ProtocolCapabilityQuery",
    "ProtocolEventBusLike",
    "ProtocolEventProjector",
    "ProtocolContainerAware",
    "ProtocolDispatchEngine",
    "ProtocolIdempotencyStore",
    "ProtocolMessageDispatcher",
    "ProtocolMessageTypeRegistry",
    "ProtocolPluginCompute",
    "ProtocolProjectorSchemaValidator",
    "ProtocolRegistryMetrics",
    "ProtocolSnapshotPublisher",
    "ProtocolSnapshotStore",
]
