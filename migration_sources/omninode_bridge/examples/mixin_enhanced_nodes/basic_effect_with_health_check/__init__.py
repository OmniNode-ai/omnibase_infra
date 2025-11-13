"""
Basic Effect with Health Check Example.

Demonstrates the simplest mixin integration - a basic EFFECT node with health checking.

Example:
    >>> from examples.mixin_enhanced_nodes.basic_effect_with_health_check.node import (
    ...     NodeDataProcessingEffect
    ... )
    >>> from omnibase_core.models.core.model_container import ModelContainer
    >>>
    >>> container = ModelContainer()
    >>> node = NodeDataProcessingEffect(container)
    >>> await node.initialize()
    >>>
    >>> # Check health
    >>> health = await node.check_health()
    >>> print(health.overall_status)
"""

from .node import NodeDataProcessingEffect

__all__ = ["NodeDataProcessingEffect"]
