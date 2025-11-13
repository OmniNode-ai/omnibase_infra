"""
Advanced Orchestrator with Metrics Example.

Demonstrates a production-grade orchestrator with comprehensive observability.

Example:
    >>> from examples.mixin_enhanced_nodes.advanced_orchestrator_with_metrics.node import (
    ...     NodeWorkflowOrchestrator
    ... )
    >>> from omnibase_core.models.core.model_container import ModelContainer
    >>>
    >>> container = ModelContainer()
    >>> node = NodeWorkflowOrchestrator(container)
    >>> await node.initialize()
    >>>
    >>> # Execute workflow
    >>> result = await node.execute_orchestration({
    ...     "workflow_id": "wf-001",
    ...     "steps": [{"step_id": "step1", "operation": "process"}]
    ... })
"""

from .node import NodeWorkflowOrchestrator

__all__ = ["NodeWorkflowOrchestrator"]
