"""
Complete Node with All Mixins Example.

Comprehensive example showing the full power of the mixin system with all available mixins.

Example:
    >>> from examples.mixin_enhanced_nodes.complete_node_all_mixins.node import (
    ...     NodeIntelligentProcessingEffect
    ... )
    >>> from omnibase_core.models.core.model_container import ModelContainer
    >>>
    >>> container = ModelContainer()
    >>> node = NodeIntelligentProcessingEffect(container)
    >>> await node.initialize()
    >>>
    >>> # Process with caching
    >>> result = await node.execute_effect({
    ...     "operation": "analyze",
    ...     "data": {"values": [1, 2, 3]},
    ...     "cache_enabled": True
    ... })
    >>> print(f"Cached: {result['cached']}")
"""

from .node import NodeIntelligentProcessingEffect

__all__ = ["NodeIntelligentProcessingEffect"]
