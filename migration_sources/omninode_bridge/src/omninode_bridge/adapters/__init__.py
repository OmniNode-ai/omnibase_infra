"""Protocol adapters for omnibase_core classes.

This module provides adapter classes that make omnibase_core classes
protocol-compliant for duck typing, until upstream changes are merged.

The adapter pattern enables:
1. **Duck Typing**: Use protocols for type hints without inheritance
2. **Forward Compatibility**: Easy migration when upstream adds protocols
3. **Type Safety**: mypy validation with runtime_checkable protocols
4. **Change Tracking**: Document all needed upstream changes

Usage:
    >>> from omninode_bridge.adapters import (
    ...     ProtocolContainer,
    ...     ProtocolNode,
    ...     AdapterModelContainer,
    ...     AdapterNodeOrchestrator,
    ... )
    >>>
    >>> # Create container with protocol typing
    >>> container: ProtocolContainer = AdapterModelContainer.create(
    ...     config={"key": "value"}
    ... )
    >>>
    >>> # Create node with protocol typing
    >>> node: ProtocolNode = AdapterNodeOrchestrator(container)
    >>> result = await node.process(input_data)

Migration Path:
    Once omnibase_core v0.2.0+ and omnibase_spi v0.2.0+ are released:
    1. Update dependencies in pyproject.toml
    2. Search/replace adapter imports with omnibase_core imports
    3. Remove this adapters module
    4. Update type hints to use omnibase_spi.protocols

See Also:
    - UPSTREAM_CHANGES.md: Detailed tracking of needed upstream changes
    - protocols.py: Protocol definitions
    - container_adapter.py: Container adapter implementation
    - node_adapters.py: Node adapter implementations

Author: OmniNode Bridge Team
Created: 2025-10-30
"""

# Container adapters (make omnibase_core containers protocol-compliant)
from .container_adapter import AdapterModelContainer

# Node adapters (make omnibase_core nodes protocol-compliant)
from .node_adapters import (
    AdapterNodeCompute,
    AdapterNodeEffect,
    AdapterNodeOrchestrator,
    AdapterNodeReducer,
    ComputeNode,
    EffectNode,
    OrchestratorNode,
    ReducerNode,
)

# Protocol definitions (what omnibase_spi should provide)
from .protocols import (
    Container,
    Contract,
    Node,
    OnexError,
    ProtocolContainer,
    ProtocolContract,
    ProtocolNode,
    ProtocolOnexError,
)

__all__ = [
    # Protocols
    "ProtocolContainer",
    "ProtocolNode",
    "ProtocolOnexError",
    "ProtocolContract",
    # Protocol aliases
    "Container",
    "Node",
    "OnexError",
    "Contract",
    # Container adapters
    "AdapterModelContainer",
    # Node adapters
    "AdapterNodeOrchestrator",
    "AdapterNodeReducer",
    "AdapterNodeEffect",
    "AdapterNodeCompute",
    # Node adapter aliases
    "OrchestratorNode",
    "ReducerNode",
    "EffectNode",
    "ComputeNode",
]

# Version tracking
__version__ = "1.0.0"
__upstream_target__ = "omnibase_core>=0.2.0, omnibase_spi>=0.2.0"
