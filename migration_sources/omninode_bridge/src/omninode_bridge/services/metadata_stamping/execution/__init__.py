"""
Execution framework for O.N.E. v0.1 protocol compliance.

This module provides the transformer pattern, schema registry,
DAG execution engine, and simulation framework for schema-first execution.
"""

from .dag_engine import DAGExecutor, DAGNode, NodeStatus
from .schema_registry import SchemaRegistry, SchemaVersion, schema_registry
from .simulation import SimulationRequest, SimulationResult, WorkflowSimulator
from .stamping_transformers import (
    BatchStampingInput,
    BatchStampingOutput,
    StampingInput,
    StampingOutput,
    ValidationInput,
    ValidationOutput,
)
from .transformer import (
    BaseTransformer,
    ExecutionContext,
    get_transformer,
    list_transformers,
    transformer,
    unregister_transformer,
)

__all__ = [
    # Core transformer framework
    "ExecutionContext",
    "BaseTransformer",
    "transformer",
    "get_transformer",
    "list_transformers",
    "unregister_transformer",
    # Schema registry
    "SchemaRegistry",
    "SchemaVersion",
    "schema_registry",
    # DAG execution
    "DAGExecutor",
    "DAGNode",
    "NodeStatus",
    # Simulation framework
    "WorkflowSimulator",
    "SimulationRequest",
    "SimulationResult",
    # Stamping transformers
    "StampingInput",
    "StampingOutput",
    "ValidationInput",
    "ValidationOutput",
    "BatchStampingInput",
    "BatchStampingOutput",
]
