"""
Integration example showing all 4 coordination components working together.

This example demonstrates:
1. SignalCoordinator - Agent-to-agent communication
2. SmartRoutingOrchestrator - Intelligent task routing
3. ContextDistributor - Context distribution to agents
4. DependencyResolver - Dependency resolution

Scenario: Code generation workflow with 3 agents:
- model_generator: Generates Pydantic models from contract
- validator_generator: Generates validators (depends on model_generator)
- test_generator: Generates tests (depends on both)
"""

import asyncio
import logging
from typing import Any

from omninode_bridge.agents.coordination import (
    CoordinationOrchestrator,
    ThreadSafeState,
)
from omninode_bridge.agents.metrics.collector import MetricsCollector
from omninode_bridge.agents.registry.registry import AgentRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def simulate_agent_work(
    agent_id: str,
    orchestrator: CoordinationOrchestrator,
    coordination_id: str,
    duration_seconds: float = 1.0,
) -> dict[str, Any]:
    """
    Simulate agent work with coordination.

    Args:
        agent_id: Agent identifier
        orchestrator: Coordination orchestrator
        coordination_id: Coordination session ID
        duration_seconds: Simulated work duration

    Returns:
        Agent work result
    """
    logger.info(f"[{agent_id}] Starting work...")

    # Get agent context
    context = await orchestrator.get_agent_context(
        coordination_id=coordination_id,
        agent_id=agent_id,
    )

    if context:
        logger.info(
            f"[{agent_id}] Received context with {len(context.assignment.tasks)} tasks"
        )
    else:
        logger.warning(f"[{agent_id}] No context found!")

    # Simulate work
    await asyncio.sleep(duration_seconds)

    # Signal completion
    result_summary = {
        "status": "completed",
        "items_generated": 5,
        "quality_score": 0.95,
        "execution_time_ms": duration_seconds * 1000,
    }

    await orchestrator.signal_agent_completion(
        coordination_id=coordination_id,
        agent_id=agent_id,
        result_summary=result_summary,
    )

    logger.info(f"[{agent_id}] Work completed!")

    return result_summary


async def run_code_generation_workflow():
    """
    Run complete code generation workflow with all 4 coordination components.

    This demonstrates:
    1. Context distribution to 3 agents
    2. Dependency resolution (validator waits for model, test waits for both)
    3. Signal coordination between agents
    4. Metrics collection across all components
    """
    logger.info("=" * 80)
    logger.info("Starting Code Generation Workflow with Coordination Integration")
    logger.info("=" * 80)

    # Initialize infrastructure
    state = ThreadSafeState[dict[str, Any]]()
    metrics = MetricsCollector()
    await metrics.start()

    # Initialize agent registry (for routing)
    registry = AgentRegistry(
        state=state,
        metrics_collector=metrics,
    )

    # Register agents
    await registry.register_agent(
        agent_id="model_generator",
        agent_info={
            "name": "Model Generator",
            "capabilities": ["pydantic_models", "contract_parsing"],
            "status": "active",
        },
    )

    await registry.register_agent(
        agent_id="validator_generator",
        agent_info={
            "name": "Validator Generator",
            "capabilities": ["validators", "quality_checks"],
            "status": "active",
        },
    )

    await registry.register_agent(
        agent_id="test_generator",
        agent_info={
            "name": "Test Generator",
            "capabilities": ["unit_tests", "integration_tests"],
            "status": "active",
        },
    )

    # Initialize coordination orchestrator
    orchestrator = CoordinationOrchestrator(
        state=state,
        metrics_collector=metrics,
        agent_registry=registry,
        enable_routing=True,
        enable_dependency_resolution=True,
    )

    logger.info("✓ Orchestrator initialized with all 4 components")

    # Define workflow
    workflow_id = "codegen-session-001"
    agent_assignments = {
        "model_generator": {
            "objective": "Generate Pydantic models from contract",
            "tasks": ["parse_contract", "generate_models", "validate_syntax"],
            "input_data": {
                "contract_path": "./contract.yaml",
                "output_dir": "./generated/models",
            },
            "priority": 10,
        },
        "validator_generator": {
            "objective": "Generate validators for models",
            "tasks": ["generate_validators", "add_quality_checks"],
            "input_data": {
                "models_dir": "./generated/models",
                "output_dir": "./generated/validators",
            },
            "dependencies": ["model_generator"],  # Waits for model_generator
            "priority": 8,
        },
        "test_generator": {
            "objective": "Generate comprehensive tests",
            "tasks": ["generate_unit_tests", "generate_integration_tests"],
            "input_data": {
                "models_dir": "./generated/models",
                "validators_dir": "./generated/validators",
                "output_dir": "./generated/tests",
            },
            "dependencies": [
                "model_generator",
                "validator_generator",
            ],  # Waits for both
            "priority": 5,
        },
    }

    # Shared intelligence across agents
    shared_intelligence = {
        "patterns": ["singleton", "factory", "builder"],
        "conventions": {
            "naming": "snake_case",
            "max_line_length": 88,
            "docstring_style": "google",
        },
        "quality_requirements": {
            "min_coverage": 0.90,
            "max_complexity": 10,
        },
    }

    # Step 1: Coordinate workflow
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Coordinating Workflow (All 4 Components)")
    logger.info("=" * 80)

    coordination_result = await orchestrator.coordinate_workflow(
        workflow_id=workflow_id,
        agent_assignments=agent_assignments,
        shared_intelligence=shared_intelligence,
        enable_signals=True,
    )

    logger.info(f"\n✓ Workflow coordinated: {coordination_result['coordination_id']}")
    logger.info(f"  - Contexts distributed: {coordination_result['contexts_distributed']}")
    logger.info(
        f"  - Dependencies resolved: {coordination_result['dependencies_resolved']}"
    )
    logger.info(f"  - Signals sent: {coordination_result['signals_sent']}")
    logger.info(f"  - Duration: {coordination_result['duration_ms']:.2f}ms")

    coordination_id = coordination_result["coordination_id"]

    # Step 2: Simulate agent execution
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Simulating Agent Execution")
    logger.info("=" * 80)

    # Phase 1: Model generator runs first
    logger.info("\n--- Phase 1: Model Generation ---")
    model_result = await simulate_agent_work(
        agent_id="model_generator",
        orchestrator=orchestrator,
        coordination_id=coordination_id,
        duration_seconds=1.5,
    )

    # Phase 2: Validator generator runs after model generator
    logger.info("\n--- Phase 2: Validator Generation ---")
    validator_result = await simulate_agent_work(
        agent_id="validator_generator",
        orchestrator=orchestrator,
        coordination_id=coordination_id,
        duration_seconds=1.0,
    )

    # Phase 3: Test generator runs after both
    logger.info("\n--- Phase 3: Test Generation ---")
    test_result = await simulate_agent_work(
        agent_id="test_generator",
        orchestrator=orchestrator,
        coordination_id=coordination_id,
        duration_seconds=2.0,
    )

    # Step 3: Collect metrics
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Collecting Coordination Metrics")
    logger.info("=" * 80)

    coord_metrics = orchestrator.get_coordination_metrics(coordination_id)

    logger.info("\n📊 Signal Metrics:")
    signal_metrics = coord_metrics["signal_metrics"]
    logger.info(f"  - Total signals: {signal_metrics['total_signals_sent']}")
    logger.info(
        f"  - Avg propagation: {signal_metrics['average_propagation_ms']:.2f}ms"
    )
    logger.info(f"  - Max propagation: {signal_metrics['max_propagation_ms']:.2f}ms")

    logger.info("\n📊 Context Distribution Metrics:")
    context_metrics = coord_metrics["context_metrics"]
    logger.info(f"  - Contexts distributed: {context_metrics['contexts_distributed']}")
    logger.info(f"  - Total duration: {context_metrics['total_duration_ms']:.2f}ms")
    logger.info(
        f"  - Avg per agent: {context_metrics['average_duration_per_agent_ms']:.2f}ms"
    )

    # Step 4: Verify dependencies
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Verifying Dependency Resolution")
    logger.info("=" * 80)

    for agent_id in ["model_generator", "validator_generator", "test_generator"]:
        is_resolved = await orchestrator.check_dependency_status(
            coordination_id=coordination_id,
            dependency_id=f"{agent_id}_complete",
        )
        status = "✓" if is_resolved else "✗"
        logger.info(f"  {status} {agent_id}: {'RESOLVED' if is_resolved else 'PENDING'}")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("WORKFLOW COMPLETE")
    logger.info("=" * 80)
    logger.info("✓ All 4 coordination components successfully integrated:")
    logger.info("  1. SignalCoordinator - Handled agent communication")
    logger.info("  2. ContextDistributor - Distributed contexts to agents")
    logger.info("  3. DependencyResolver - Resolved agent dependencies")
    logger.info("  4. SmartRoutingOrchestrator - Made routing decisions")
    logger.info("=" * 80)

    # Cleanup
    await metrics.stop()


async def run_simple_coordination_example():
    """
    Simpler example showing basic coordination without full workflow.

    This is useful for understanding the orchestrator API.
    """
    logger.info("\n" + "=" * 80)
    logger.info("Simple Coordination Example")
    logger.info("=" * 80)

    # Setup
    state = ThreadSafeState[dict[str, Any]]()
    metrics = MetricsCollector()
    await metrics.start()

    orchestrator = CoordinationOrchestrator(
        state=state,
        metrics_collector=metrics,
        enable_routing=False,  # Disable routing for simplicity
        enable_dependency_resolution=False,  # Disable dependency resolution
    )

    # Simple workflow with 2 agents
    result = await orchestrator.coordinate_workflow(
        workflow_id="simple-workflow",
        agent_assignments={
            "agent-1": {
                "objective": "Task 1",
                "tasks": ["subtask-1", "subtask-2"],
            },
            "agent-2": {
                "objective": "Task 2",
                "tasks": ["subtask-3"],
            },
        },
    )

    logger.info(f"\n✓ Simple workflow completed in {result['duration_ms']:.2f}ms")
    logger.info(f"  - Contexts distributed: {result['contexts_distributed']}")
    logger.info(f"  - Signals sent: {result['signals_sent']}")

    await metrics.stop()


async def main():
    """Run all examples."""
    # Run full integration example
    await run_code_generation_workflow()

    # Wait a bit between examples
    await asyncio.sleep(1)

    # Run simple example
    await run_simple_coordination_example()


if __name__ == "__main__":
    asyncio.run(main())
