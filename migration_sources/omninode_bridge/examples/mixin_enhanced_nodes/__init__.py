"""
Mixin-Enhanced Node Examples.

This package contains complete, runnable examples demonstrating the ONEX v2.0
mixin-enhanced code generation system with different mixin combinations.

Examples:
    - basic_effect_with_health_check: Simple effect with health monitoring
    - advanced_orchestrator_with_metrics: Orchestrator with full observability
    - complete_node_all_mixins: Comprehensive example with all mixins

Usage:
    # Run an example directly
    python -m examples.mixin_enhanced_nodes.basic_effect_with_health_check.node

    # Import in your code
    from examples.mixin_enhanced_nodes.basic_effect_with_health_check.node import (
        NodeDataProcessingEffect
    )
"""

__version__ = "1.0.0"
__all__ = [
    "basic_effect_with_health_check",
    "advanced_orchestrator_with_metrics",
    "complete_node_all_mixins",
]
