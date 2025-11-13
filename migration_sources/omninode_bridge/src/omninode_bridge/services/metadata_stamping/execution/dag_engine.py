"""
DAG execution engine for O.N.E. v0.1 protocol.

This module provides directed acyclic graph execution with
parallel processing and dependency resolution.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from .transformer import BaseTransformer, ExecutionContext

logger = logging.getLogger(__name__)


class NodeStatus(str, Enum):
    """Status of a DAG node."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DAGNode:
    """Node in the execution DAG."""

    id: str
    transformer: BaseTransformer
    dependencies: list[str]
    status: NodeStatus = NodeStatus.PENDING
    input_data: Any = None
    output_data: Any = None
    execution_time_ms: float = 0.0
    error: Optional[str] = None


class DAGExecutor:
    """
    Executor for directed acyclic graphs.

    Provides parallel execution with dependency resolution,
    cycle detection, and simulation support.
    """

    def __init__(self):
        """Initialize DAG executor."""
        self.nodes: dict[str, DAGNode] = {}
        self.execution_context: Optional[ExecutionContext] = None
        self.execution_history: list[str] = []

    def add_node(
        self,
        node_id: str,
        transformer: BaseTransformer,
        dependencies: Optional[list[str]] = None,
    ) -> DAGNode:
        """
        Add a node to the DAG.

        Args:
            node_id: Unique node identifier
            transformer: Transformer to execute
            dependencies: List of dependency node IDs

        Returns:
            DAGNode: Created node
        """
        node = DAGNode(
            id=node_id, transformer=transformer, dependencies=dependencies or []
        )
        self.nodes[node_id] = node
        logger.debug(
            f"Added DAG node: {node_id} with {len(node.dependencies)} dependencies"
        )
        return node

    def validate_dag(self) -> bool:
        """
        Validate DAG for cycles and missing dependencies.

        Returns:
            bool: True if valid

        Raises:
            ValueError: If validation fails
        """
        # Check for missing dependencies
        for node in self.nodes.values():
            for dep in node.dependencies:
                if dep not in self.nodes:
                    raise ValueError(f"Node {node.id} depends on missing node {dep}")

        # Check for cycles using DFS
        visited = set()
        rec_stack = set()

        def has_cycle(node_id: str) -> bool:
            """Check for cycle starting from node."""
            visited.add(node_id)
            rec_stack.add(node_id)

            # Check all dependencies
            node = self.nodes[node_id]
            for dep in node.dependencies:
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True

            rec_stack.remove(node_id)
            return False

        # Check from all nodes
        for node_id in self.nodes:
            if node_id not in visited:
                if has_cycle(node_id):
                    raise ValueError("DAG contains cycles")

        logger.debug("DAG validation successful")
        return True

    async def execute(
        self, input_data: dict[str, Any], simulation_mode: bool = False
    ) -> dict[str, Any]:
        """
        Execute the DAG.

        Args:
            input_data: Input data for nodes
            simulation_mode: Whether to run in simulation

        Returns:
            dict: Execution results
        """
        start_time = time.perf_counter()

        # Validate DAG
        try:
            if not self.validate_dag():
                raise ValueError("Invalid DAG")
        except ValueError as e:
            raise ValueError("Invalid DAG") from e

        # Create execution context
        self.execution_context = ExecutionContext(
            execution_id=str(uuid.uuid4()),
            input_schema="DAGInput",
            output_schema="DAGOutput",
            simulation_mode=simulation_mode,
            metadata={"node_count": len(self.nodes), "simulation": simulation_mode},
        )

        logger.info(
            f"Starting DAG execution: id={self.execution_context.execution_id}, "
            f"nodes={len(self.nodes)}, simulation={simulation_mode}"
        )

        # Reset node states
        self._reset_nodes()

        # Set initial input data
        self._set_initial_inputs(input_data)

        # Clear execution history
        self.execution_history = []

        # Execute nodes in topological order
        execution_results = {}
        completed_count = 0
        failed_count = 0

        while True:
            # Find ready nodes (no pending dependencies)
            ready_nodes = self._get_ready_nodes()

            if not ready_nodes:
                # Check if we're done or stuck
                pending_nodes = [
                    n for n in self.nodes.values() if n.status == NodeStatus.PENDING
                ]

                if not pending_nodes:
                    # All nodes processed
                    break
                else:
                    # Check if any pending nodes have failed dependencies
                    skipped_any = self._skip_nodes_with_failed_dependencies()

                    if not skipped_any:
                        # Stuck - this shouldn't happen after validation
                        logger.error(
                            f"DAG execution stuck with {len(pending_nodes)} pending nodes"
                        )
                        raise RuntimeError(
                            "DAG execution stuck - possible dependency issue"
                        )

            # Execute ready nodes in parallel
            logger.debug(f"Executing {len(ready_nodes)} ready nodes in parallel")
            await self._execute_nodes_parallel(ready_nodes)

            # Update counts
            for node in ready_nodes:
                if node.status == NodeStatus.COMPLETED:
                    completed_count += 1
                elif node.status == NodeStatus.FAILED:
                    failed_count += 1

        # Collect results
        for node_id, node in self.nodes.items():
            execution_results[node_id] = {
                "status": node.status.value,
                "output_data": node.output_data,
                "execution_time_ms": node.execution_time_ms,
                "error": node.error,
            }

        # Calculate total execution time
        total_time = (time.perf_counter() - start_time) * 1000

        # Determine overall status
        overall_status = "completed"
        if failed_count > 0:
            overall_status = "partial" if completed_count > 0 else "failed"

        result = {
            "execution_id": self.execution_context.execution_id,
            "simulation_mode": simulation_mode,
            "nodes": execution_results,
            "overall_status": overall_status,
            "completed_nodes": completed_count,
            "failed_nodes": failed_count,
            "total_execution_time_ms": total_time,
            "execution_order": self.execution_history,
        }

        logger.info(
            f"DAG execution completed: status={overall_status}, "
            f"completed={completed_count}, failed={failed_count}, "
            f"time={total_time:.2f}ms"
        )

        return result

    def _reset_nodes(self):
        """Reset all nodes to initial state."""
        for node in self.nodes.values():
            node.status = NodeStatus.PENDING
            node.output_data = None
            node.error = None
            node.execution_time_ms = 0.0

    def _set_initial_inputs(self, input_data: dict[str, Any]):
        """
        Set initial input data for nodes.

        Args:
            input_data: Map of node ID to input data
        """
        for node_id, data in input_data.items():
            if node_id in self.nodes:
                self.nodes[node_id].input_data = data
                logger.debug(f"Set input data for node {node_id}")

    def _get_ready_nodes(self) -> list[DAGNode]:
        """
        Get nodes ready for execution.

        Returns:
            list: Nodes with all dependencies completed
        """
        ready = []
        for node in self.nodes.values():
            if node.status == NodeStatus.PENDING:
                # Check if all dependencies are completed
                deps_completed = all(
                    self.nodes[dep_id].status == NodeStatus.COMPLETED
                    for dep_id in node.dependencies
                )
                if deps_completed:
                    ready.append(node)

        return ready

    def _skip_nodes_with_failed_dependencies(self) -> bool:
        """
        Skip nodes whose dependencies have failed.

        Returns:
            bool: True if any nodes were skipped
        """
        skipped_any = False

        for node in self.nodes.values():
            if node.status == NodeStatus.PENDING:
                # Check if any dependencies have failed
                has_failed_deps = any(
                    self.nodes[dep_id].status == NodeStatus.FAILED
                    for dep_id in node.dependencies
                )

                if has_failed_deps:
                    node.status = NodeStatus.SKIPPED
                    logger.debug(f"Skipped node {node.id} due to failed dependencies")
                    skipped_any = True

        return skipped_any

    async def _execute_nodes_parallel(self, nodes: list[DAGNode]):
        """
        Execute multiple nodes in parallel.

        Args:
            nodes: Nodes to execute
        """
        tasks = [self._execute_node(node) for node in nodes]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_node(self, node: DAGNode):
        """
        Execute a single node.

        Args:
            node: Node to execute
        """
        start_time = time.perf_counter()
        node.status = NodeStatus.RUNNING

        logger.debug(f"Executing node {node.id}")
        self.execution_history.append(node.id)

        try:
            # Prepare input from dependencies
            node_input = self._prepare_node_input(node)

            if self.execution_context.simulation_mode:
                # Simulation mode - don't actually execute
                await asyncio.sleep(0.01)  # Simulate some processing time
                node.output_data = {
                    "simulated": True,
                    "input": node_input,
                    "node_id": node.id,
                }
                node.status = NodeStatus.COMPLETED
                logger.debug(f"Node {node.id} simulated successfully")

            else:
                # Real execution
                # Create node-specific context
                node_context = ExecutionContext(
                    execution_id=f"{self.execution_context.execution_id}-{node.id}",
                    input_schema=(
                        node.transformer.input_schema.__name__
                        if node.transformer.input_schema
                        else "Any"
                    ),
                    output_schema=(
                        node.transformer.output_schema.__name__
                        if node.transformer.output_schema
                        else "Any"
                    ),
                    simulation_mode=False,
                    metadata={
                        "dag_execution_id": self.execution_context.execution_id,
                        "node_id": node.id,
                    },
                )

                # Execute transformer
                result = await node.transformer.execute_with_validation(
                    node_input, node_context
                )

                node.output_data = (
                    result.model_dump() if hasattr(result, "model_dump") else result
                )
                node.status = NodeStatus.COMPLETED
                logger.debug(f"Node {node.id} executed successfully")

        except Exception as e:
            node.error = str(e)
            node.status = NodeStatus.FAILED
            logger.error(f"Node {node.id} execution failed: {e}")

        finally:
            node.execution_time_ms = (time.perf_counter() - start_time) * 1000

    def _prepare_node_input(self, node: DAGNode) -> Any:
        """
        Prepare input for node from dependencies and initial data.

        Args:
            node: Node to prepare input for

        Returns:
            Any: Prepared input data
        """
        if not node.dependencies:
            # No dependencies - use initial input or provide minimal valid input
            if node.input_data is None:
                # Return minimal valid input to allow transformers to use defaults
                return {}
            elif isinstance(node.input_data, dict):
                return node.input_data
            else:
                # For Pydantic models or other objects, return as-is
                # Only wrap simple types like strings/numbers
                from pydantic import BaseModel

                if isinstance(node.input_data, BaseModel):
                    return node.input_data
                else:
                    # Wrap simple types in dict
                    return {"initial_input": node.input_data}

        # Combine outputs from dependencies
        combined_input = {}

        # Add initial input if available
        if node.input_data is not None:
            if isinstance(node.input_data, dict):
                combined_input.update(node.input_data)
            else:
                combined_input["initial_input"] = node.input_data

        # Add dependency outputs
        for dep_id in node.dependencies:
            dep_node = self.nodes[dep_id]
            if dep_node.output_data:
                combined_input[dep_id] = dep_node.output_data

        return combined_input

    def get_execution_plan(self) -> list[list[str]]:
        """
        Get execution plan showing parallel stages.

        Returns:
            list: List of stages, each containing node IDs that can run in parallel
        """
        try:
            if not self.validate_dag():
                return []
        except ValueError:
            # Invalid DAG - return empty plan
            return []

        # Reset nodes
        temp_completed = set()
        stages = []

        while len(temp_completed) < len(self.nodes):
            # Find nodes that can execute
            stage = []
            for node_id, node in self.nodes.items():
                if node_id not in temp_completed:
                    # Check if dependencies are completed
                    deps_ready = all(dep in temp_completed for dep in node.dependencies)
                    if deps_ready:
                        stage.append(node_id)

            if not stage:
                # No progress possible
                break

            stages.append(stage)
            temp_completed.update(stage)

        return stages

    def visualize_dag(self) -> str:
        """
        Generate simple text visualization of DAG.

        Returns:
            str: Text representation of DAG
        """
        lines = ["DAG Structure:"]
        lines.append("-" * 40)

        for node_id, node in self.nodes.items():
            deps = ", ".join(node.dependencies) if node.dependencies else "None"
            status = node.status.value
            lines.append(f"Node: {node_id}")
            lines.append(f"  Transformer: {node.transformer.name}")
            lines.append(f"  Dependencies: {deps}")
            lines.append(f"  Status: {status}")

        lines.append("-" * 40)
        lines.append("Execution Plan:")

        stages = self.get_execution_plan()
        for i, stage in enumerate(stages):
            lines.append(f"  Stage {i + 1}: {', '.join(stage)}")

        return "\n".join(lines)
