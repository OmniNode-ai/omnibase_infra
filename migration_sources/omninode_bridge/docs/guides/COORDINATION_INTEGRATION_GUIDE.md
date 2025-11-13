# Phase 4 Coordination Integration Guide

**Version**: 1.0
**Status**: ✅ Production-Ready
**Last Updated**: 2025-11-06
**Target Audience**: Developers integrating coordination into code generation pipelines

---

## Table of Contents

1. [Overview](#overview)
2. [Integration Steps](#integration-steps)
3. [Code Generation Pipeline Integration](#code-generation-pipeline-integration)
4. [Configuration](#configuration)
5. [Best Practices](#best-practices)
6. [Common Integration Patterns](#common-integration-patterns)
7. [Troubleshooting](#troubleshooting)

---

## Overview

This guide provides step-by-step instructions for integrating Phase 4 Coordination into your code generation pipeline. The coordination system enables:

- **Agent-to-agent communication** via signals
- **Intelligent routing** based on state and priorities
- **Context distribution** with shared intelligence
- **Dependency resolution** for sequential workflows

**Prerequisites**:
- Python 3.11+
- AsyncIO knowledge
- Basic understanding of code generation workflows

**Integration Time**: 30-60 minutes for basic integration, 2-4 hours for advanced features

---

## Integration Steps

### Step 1: Install Dependencies (5 minutes)

```bash
# Install omninode_bridge
cd /path/to/omninode_bridge
poetry install

# Or add to your project's pyproject.toml
[tool.poetry.dependencies]
omninode-bridge = {path = "../omninode_bridge", develop = true}
```

### Step 2: Setup Foundation Components (10 minutes)

```python
# setup_coordination.py
import asyncio
from omninode_bridge.agents.coordination import ThreadSafeState
from omninode_bridge.agents.metrics import MetricsCollector

async def setup_coordination_infrastructure():
    """
    Setup foundation components for coordination.

    Returns:
        tuple: (state, metrics)
    """
    # 1. Create shared state
    state = ThreadSafeState()

    # 2. Create metrics collector
    metrics = MetricsCollector(
        buffer_size=1000,
        kafka_enabled=True,  # Set to False for local development
        postgres_enabled=True  # Set to False for local development
    )

    # 3. Start metrics collector
    await metrics.start()

    return state, metrics

# Usage
if __name__ == "__main__":
    state, metrics = asyncio.run(setup_coordination_infrastructure())
    print("✅ Coordination infrastructure ready")
```

### Step 3: Initialize Coordination Components (10 minutes)

```python
# coordination_manager.py
from omninode_bridge.agents.coordination import (
    SignalCoordinator,
    SmartRoutingOrchestrator,
    ContextDistributor,
    DependencyResolver,
    ConditionalRouter,
    ParallelRouter,
    StateAnalysisRouter,
    PriorityRouter,
)

class CoordinationManager:
    """
    Central manager for all coordination components.
    """

    def __init__(self, state: ThreadSafeState, metrics: MetricsCollector):
        self.state = state
        self.metrics = metrics

        # Initialize components
        self.signal_coordinator = SignalCoordinator(
            state=state,
            metrics_collector=metrics,
            max_history_size=5000
        )

        self.routing_orchestrator = SmartRoutingOrchestrator(
            metrics_collector=metrics,
            state=state,
            max_history_size=1000
        )

        self.context_distributor = ContextDistributor(
            state=state,
            metrics_collector=metrics
        )

        self.dependency_resolver = DependencyResolver(
            signal_coordinator=self.signal_coordinator,
            metrics_collector=metrics,
            state=state,
            max_concurrent_resolutions=10
        )

        # Add default routers
        self._setup_default_routers()

    def _setup_default_routers(self):
        """Setup default routing strategies."""
        # Add routers based on your needs
        # Example: Add conditional router for error handling
        from omninode_bridge.agents.coordination import ConditionalRule, RoutingDecision

        rules = [
            ConditionalRule(
                rule_id="error_retry",
                name="Retry on Error",
                condition_key="error_count",
                condition_operator=">",
                condition_value=0,
                decision=RoutingDecision.RETRY,
                priority=100
            )
        ]

        self.routing_orchestrator.add_router(
            ConditionalRouter(rules=rules, metrics_collector=self.metrics)
        )

        # Add other routers as needed
        self.routing_orchestrator.add_router(
            ParallelRouter(metrics_collector=self.metrics)
        )
        self.routing_orchestrator.add_router(
            StateAnalysisRouter(metrics_collector=self.metrics)
        )
        self.routing_orchestrator.add_router(
            PriorityRouter(metrics_collector=self.metrics)
        )

# Usage
coordination_manager = CoordinationManager(state=state, metrics=metrics)
```

### Step 4: Integrate with Code Generation Pipeline (30 minutes)

```python
# code_generation_pipeline.py
import asyncio
from typing import Dict, Any
from uuid import uuid4

class CodeGenerationPipeline:
    """
    Code generation pipeline with Phase 4 Coordination integration.
    """

    def __init__(self, coordination_manager: CoordinationManager):
        self.coord = coordination_manager
        self.state = coordination_manager.state
        self.metrics = coordination_manager.metrics

    async def generate_code(
        self,
        contract_path: str,
        output_dir: str
    ) -> Dict[str, Any]:
        """
        Execute code generation pipeline with coordination.

        Args:
            contract_path: Path to contract YAML
            output_dir: Output directory for generated code

        Returns:
            Dictionary with generation results
        """
        # 1. Create coordination IDs
        coordination_id = f"codegen-{uuid4().hex[:8]}"
        session_id = f"session-{uuid4().hex[:8]}"

        print(f"📋 Starting code generation: {coordination_id}")

        try:
            # 2. Define agent assignments
            agent_assignments = {
                "model_generator": {
                    "agent_role": "model_generator",
                    "objective": "Generate Pydantic models from contract",
                    "tasks": ["parse_contract", "infer_types", "generate_models"],
                    "input_data": {
                        "contract_path": contract_path,
                        "output_dir": f"{output_dir}/models"
                    },
                    "dependencies": [],
                    "output_requirements": {"format": "pydantic_v2"},
                    "success_criteria": {"min_quality": 0.95}
                },
                "validator_generator": {
                    "agent_role": "validator_generator",
                    "objective": "Generate validators for models",
                    "tasks": ["analyze_models", "generate_validators"],
                    "input_data": {
                        "models_dir": f"{output_dir}/models",
                        "output_dir": f"{output_dir}/validators"
                    },
                    "dependencies": ["model_generator"],
                    "output_requirements": {"format": "pytest"},
                    "success_criteria": {"all_fields_validated": True}
                },
                "test_generator": {
                    "agent_role": "test_generator",
                    "objective": "Generate integration tests",
                    "tasks": ["analyze_workflow", "generate_tests"],
                    "input_data": {
                        "output_dir": f"{output_dir}/tests"
                    },
                    "dependencies": ["model_generator", "validator_generator"],
                    "output_requirements": {},
                    "success_criteria": {}
                }
            }

            # 3. Distribute context to all agents
            contexts = await self.coord.context_distributor.distribute_agent_context(
                coordination_state={
                    "coordination_id": coordination_id,
                    "session_id": session_id
                },
                agent_assignments=agent_assignments
            )

            print(f"✅ Context distributed to {len(contexts)} agents")

            # 4. Execute agents sequentially with dependency resolution
            results = {}

            for agent_id in ["model_generator", "validator_generator", "test_generator"]:
                print(f"\n🤖 Executing: {agent_id}")

                # Resolve dependencies before execution
                agent_context = contexts[agent_id]
                if agent_context.agent_assignment.dependencies:
                    print(f"⏳ Resolving dependencies: {agent_context.agent_assignment.dependencies}")

                    success = await self.coord.dependency_resolver.resolve_agent_dependencies(
                        coordination_id=coordination_id,
                        agent_context={
                            "agent_id": agent_id,
                            "dependencies": [
                                {
                                    "dependency_id": f"{dep}_complete",
                                    "type": "agent_completion",
                                    "target": dep,
                                    "timeout": 120,
                                    "metadata": {"agent_id": dep}
                                }
                                for dep in agent_context.agent_assignment.dependencies
                            ]
                        }
                    )

                    if not success:
                        print(f"❌ Dependency resolution failed for {agent_id}")
                        raise RuntimeError(f"Dependency resolution failed for {agent_id}")

                    print(f"✅ Dependencies resolved for {agent_id}")

                # Signal agent initialization
                await self.coord.signal_coordinator.signal_coordination_event(
                    coordination_id=coordination_id,
                    event_type="agent_initialized",
                    event_data={"agent_id": agent_id, "ready": True},
                    sender_agent_id=agent_id
                )

                # Execute agent
                agent_result = await self._execute_agent(agent_id, agent_context)
                results[agent_id] = agent_result

                # Signal agent completion
                await self.coord.signal_coordinator.signal_coordination_event(
                    coordination_id=coordination_id,
                    event_type="agent_completed",
                    event_data={
                        "agent_id": agent_id,
                        "result_summary": agent_result.get("summary", ""),
                        "quality_score": agent_result.get("quality_score", 0.0),
                        "execution_time_ms": agent_result.get("execution_time_ms", 0.0)
                    },
                    sender_agent_id=agent_id
                )

                print(f"✅ {agent_id} completed (quality: {agent_result.get('quality_score', 0.0):.2f})")

            # 5. Cleanup
            self.coord.context_distributor.clear_coordination_contexts(coordination_id)
            self.coord.dependency_resolver.clear_coordination_dependencies(coordination_id)

            print(f"\n✅ Code generation complete: {coordination_id}")

            return {
                "coordination_id": coordination_id,
                "session_id": session_id,
                "agents": results,
                "status": "success"
            }

        except Exception as e:
            print(f"❌ Code generation failed: {e}")
            raise

    async def _execute_agent(
        self,
        agent_id: str,
        agent_context: Any
    ) -> Dict[str, Any]:
        """
        Execute specific agent (placeholder for actual agent execution).

        Args:
            agent_id: Agent identifier
            agent_context: Agent context package

        Returns:
            Agent execution result
        """
        # TODO: Replace with actual agent execution logic
        # This is where you integrate your existing code generation agents

        import asyncio
        import time

        start_time = time.time()

        # Simulate agent work
        await asyncio.sleep(0.5)

        execution_time_ms = (time.time() - start_time) * 1000

        return {
            "agent_id": agent_id,
            "summary": f"Executed {agent_id} successfully",
            "quality_score": 0.95,
            "execution_time_ms": execution_time_ms,
            "files_generated": 5
        }

# Usage
async def main():
    # Setup
    state, metrics = await setup_coordination_infrastructure()
    coordination_manager = CoordinationManager(state, metrics)
    pipeline = CodeGenerationPipeline(coordination_manager)

    # Run code generation
    result = await pipeline.generate_code(
        contract_path="./contracts/user_service.yaml",
        output_dir="./generated"
    )

    print(f"\n📊 Results: {result}")

    await metrics.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Code Generation Pipeline Integration

### Pattern 1: Sequential with Dependencies

**Use Case**: Model → Validator → Test (each depends on previous)

```python
async def sequential_pipeline(coordination_manager, contract_path):
    coordination_id = generate_coordination_id()

    # Define agents with dependencies
    agent_assignments = {
        "model_gen": {"dependencies": []},
        "validator_gen": {"dependencies": ["model_gen"]},
        "test_gen": {"dependencies": ["model_gen", "validator_gen"]}
    }

    # Distribute context
    contexts = await coordination_manager.context_distributor.distribute_agent_context(
        coordination_state={"coordination_id": coordination_id, "session_id": generate_session_id()},
        agent_assignments=agent_assignments
    )

    # Execute agents sequentially
    for agent_id in ["model_gen", "validator_gen", "test_gen"]:
        # Resolve dependencies
        await coordination_manager.dependency_resolver.resolve_agent_dependencies(
            coordination_id=coordination_id,
            agent_context=prepare_agent_context(contexts[agent_id])
        )

        # Execute agent
        result = await execute_agent(agent_id, contexts[agent_id])

        # Signal completion
        await coordination_manager.signal_coordinator.signal_coordination_event(
            coordination_id=coordination_id,
            event_type="agent_completed",
            event_data={"agent_id": agent_id, "quality_score": result.quality}
        )
```

### Pattern 2: Parallel Execution

**Use Case**: Process 3 contracts simultaneously

```python
async def parallel_pipeline(coordination_manager, contract_paths):
    coordination_id = generate_coordination_id()

    # Define parallel agents (no dependencies)
    agent_assignments = {
        f"contract_{i}": {
            "objective": f"Process contract {i}",
            "dependencies": [],
            "input_data": {"contract_path": path}
        }
        for i, path in enumerate(contract_paths)
    }

    # Distribute context
    contexts = await coordination_manager.context_distributor.distribute_agent_context(
        coordination_state={"coordination_id": coordination_id, "session_id": generate_session_id()},
        agent_assignments=agent_assignments
    )

    # Execute agents in parallel
    results = await asyncio.gather(*[
        execute_agent(agent_id, contexts[agent_id])
        for agent_id in agent_assignments.keys()
    ])

    return results
```

### Pattern 3: Mixed Sequential and Parallel

**Use Case**: Parse contracts in parallel, then generate unified schema

```python
async def mixed_pipeline(coordination_manager, contract_paths):
    coordination_id = generate_coordination_id()

    # Define agents with mixed dependencies
    agent_assignments = {
        # Parallel contract parsing
        **{
            f"parser_{i}": {
                "objective": f"Parse contract {i}",
                "dependencies": []
            }
            for i in range(len(contract_paths))
        },
        # Sequential schema generation (depends on all parsers)
        "schema_gen": {
            "objective": "Generate unified schema",
            "dependencies": [f"parser_{i}" for i in range(len(contract_paths))]
        }
    }

    # Distribute context
    contexts = await coordination_manager.context_distributor.distribute_agent_context(
        coordination_state={"coordination_id": coordination_id, "session_id": generate_session_id()},
        agent_assignments=agent_assignments
    )

    # Execute parsers in parallel
    parser_ids = [f"parser_{i}" for i in range(len(contract_paths))]
    await asyncio.gather(*[
        execute_agent_with_completion_signal(agent_id, contexts[agent_id], coordination_id)
        for agent_id in parser_ids
    ])

    # Execute schema_gen (after dependencies resolved)
    await coordination_manager.dependency_resolver.resolve_agent_dependencies(
        coordination_id=coordination_id,
        agent_context=prepare_agent_context(contexts["schema_gen"])
    )

    result = await execute_agent("schema_gen", contexts["schema_gen"])

    return result
```

---

## Configuration

### Signal Coordinator Configuration

```python
signal_coordinator = SignalCoordinator(
    state=state,
    metrics_collector=metrics,
    max_history_size=10000  # Adjust based on expected signal volume
)
```

**Tuning**:
- **max_history_size**: Higher for debugging (10,000+), lower for production (1,000-5,000)

---

### Routing Orchestrator Configuration

```python
routing_orchestrator = SmartRoutingOrchestrator(
    metrics_collector=metrics,
    state=state,
    max_history_size=1000
)

# Add routers based on needs
routing_orchestrator.add_router(ConditionalRouter(rules=[...]))
routing_orchestrator.add_router(ParallelRouter(parallelization_hints=[...]))
routing_orchestrator.add_router(StateAnalysisRouter(max_complexity_score=0.8))
routing_orchestrator.add_router(PriorityRouter())
```

**Tuning**:
- **Conditional rules**: Add rules for error handling, phase transitions
- **Parallelization hints**: Define task groups that can run in parallel
- **Max complexity score**: Lower (0.5-0.7) for complex workflows, higher (0.8-0.9) for simple workflows

---

### Context Distributor Configuration

```python
context_distributor = ContextDistributor(
    state=state,
    metrics_collector=metrics,
    default_resource_allocation=ResourceAllocation(
        max_execution_time_ms=300000,  # 5 minutes
        max_retry_attempts=3,
        quality_threshold=0.8
    ),
    default_coordination_protocols=CoordinationProtocols(
        update_interval_ms=5000,
        heartbeat_interval_ms=10000
    )
)
```

**Tuning**:
- **max_execution_time_ms**: Increase for slow agents (300,000-600,000)
- **max_retry_attempts**: Increase for unreliable operations (3-5)
- **quality_threshold**: Adjust based on acceptable quality (0.7-0.95)

---

### Dependency Resolver Configuration

```python
dependency_resolver = DependencyResolver(
    signal_coordinator=signal_coordinator,
    metrics_collector=metrics,
    state=state,
    max_concurrent_resolutions=10  # Adjust based on expected concurrency
)
```

**Tuning**:
- **max_concurrent_resolutions**: Higher for more parallel dependencies (10-20)
- **Dependency timeout**: Set per dependency (default: 120s)

---

## Best Practices

### 1. Always Use Coordination IDs

**Good Practice**:
```python
coordination_id = f"codegen-{uuid4().hex[:8]}"  # Unique per workflow
session_id = f"session-{uuid4().hex[:8]}"  # Unique per session
```

**Why**: Enables tracking, debugging, and cleanup.

---

### 2. Signal Completion After Agent Execution

**Good Practice**:
```python
# Execute agent
result = await execute_agent(agent_id, context)

# Signal completion
await signal_coordinator.signal_coordination_event(
    coordination_id=coordination_id,
    event_type="agent_completed",
    event_data={"agent_id": agent_id, "quality_score": result.quality}
)
```

**Why**: Enables dependency resolution for downstream agents.

---

### 3. Always Cleanup After Workflow

**Good Practice**:
```python
try:
    # Execute workflow
    result = await execute_workflow()
finally:
    # Always cleanup
    context_distributor.clear_coordination_contexts(coordination_id)
    dependency_resolver.clear_coordination_dependencies(coordination_id)
```

**Why**: Prevents memory leaks and state pollution.

---

### 4. Use Specific Agent Roles

**Good Practice**:
```python
agent_assignments = {
    "model_generator": {"agent_role": "model_generator"},  # Specific role
    "validator_generator": {"agent_role": "validator_generator"}
}
```

**Why**: Enables role-based routing and capability matching.

---

### 5. Set Realistic Timeouts

**Good Practice**:
```python
dependency = Dependency(
    dependency_id="model_gen_complete",
    timeout=300,  # 5 minutes (realistic for model generation)
)
```

**Why**: Prevents premature timeouts while detecting failures.

---

## Common Integration Patterns

### Pattern 1: Error Recovery with Conditional Routing

```python
# Define error handling rule
error_rule = ConditionalRule(
    rule_id="error_retry",
    name="Retry on Error",
    condition_key="error_count",
    condition_operator="<=",
    condition_value=3,  # Max 3 retries
    decision=RoutingDecision.RETRY,
    priority=100
)

routing_orchestrator.add_router(ConditionalRouter(rules=[error_rule]))

# Execute agent with error recovery
for attempt in range(4):  # Max 3 retries + 1 initial attempt
    try:
        result = await execute_agent(agent_id, context)
        break
    except Exception as e:
        state["error_count"] = attempt + 1

        # Make routing decision
        routing_result = routing_orchestrator.route(
            state=state,
            current_task=agent_id
        )

        if routing_result["decision"] == "retry":
            print(f"Retrying {agent_id} (attempt {attempt + 2})")
        else:
            raise
```

---

### Pattern 2: Quality Gate Integration

```python
# Define quality gate dependency
quality_gate_dep = {
    "dependency_id": "coverage_gate",
    "type": "quality_gate",
    "target": "coverage_gate",
    "timeout": 60,
    "metadata": {
        "gate_id": "coverage_gate",
        "threshold": 0.8
    }
}

# Update quality gate score (from test execution)
await dependency_resolver.update_quality_gate_score("coverage_gate", 0.85)

# Resolve quality gate dependency
success = await dependency_resolver.resolve_agent_dependencies(
    coordination_id=coordination_id,
    agent_context={
        "agent_id": "deployment_agent",
        "dependencies": [quality_gate_dep]
    }
)

if success:
    print("Quality gate passed, proceeding with deployment")
else:
    print("Quality gate failed, aborting deployment")
```

---

### Pattern 3: Shared Intelligence Updates

```python
# After model generation, update type registry
from omninode_bridge.agents.coordination import ContextUpdateRequest

update_request = ContextUpdateRequest(
    coordination_id=coordination_id,
    update_type="type_registry",
    update_data={
        "UserModel": "class UserModel(BaseModel): ...",
        "PostModel": "class PostModel(BaseModel): ..."
    },
    target_agents=["validator_generator", "test_generator"],  # Update specific agents
    increment_version=True
)

results = context_distributor.update_shared_intelligence(update_request)

print(f"Updated {sum(results.values())}/{len(results)} agents")
```

---

## Troubleshooting

### Issue 1: Dependency Resolution Timeout

**Problem**: Agent dependencies time out before resolution.

**Solution**:
1. Increase dependency timeout
2. Verify target agent actually completes
3. Check signal coordinator for completion signals

```python
# Increase timeout
dependency = {
    "dependency_id": "slow_agent_complete",
    "timeout": 300,  # 5 minutes instead of default 120s
}

# Verify completion signal was sent
history = signal_coordinator.get_signal_history(
    coordination_id=coordination_id,
    filters={"signal_type": "agent_completed"}
)
print(f"Completion signals: {len(history)}")
```

---

### Issue 2: Context Distribution Slow

**Problem**: Context distribution takes longer than expected.

**Solution**:
1. Reduce shared intelligence size
2. Distribute context in batches
3. Check ThreadSafeState performance

```python
# Check metrics
stats = await metrics.get_stats()
print(f"Context distribution time: {stats.get('context_distribution_time_ms')}")

# Reduce shared intelligence
shared_intel = SharedIntelligence(
    type_registry={},  # Only include necessary types
    pattern_library={"validation": ["email_validator"]}  # Only necessary patterns
)
```

---

### Issue 3: Signal Not Received

**Problem**: Agent doesn't receive expected signal.

**Solution**:
1. Verify coordination_id matches
2. Ensure subscription before signal sent
3. Check signal type filter

```python
# Check signal history
history = signal_coordinator.get_signal_history(coordination_id=coordination_id)
print(f"Total signals: {len(history)}")

# Verify subscription
async for signal in signal_coordinator.subscribe_to_signals(
    coordination_id=coordination_id,
    agent_id="validator_gen",
    signal_types=["agent_completed"]  # Must include expected type
):
    print(f"Received signal: {signal.signal_type}")
```

---

## Next Steps

After completing integration:

1. **Performance Tuning**: See [Performance Tuning Guide](./COORDINATION_PERFORMANCE_TUNING.md)
2. **Advanced Patterns**: See [Architecture Guide](../architecture/PHASE_4_COORDINATION_ARCHITECTURE.md)
3. **API Reference**: See [API Reference](../api/COORDINATION_API_REFERENCE.md)
4. **Component Guides**: See individual component guides for deep dives

---

**Version**: 1.0
**Status**: ✅ Production-Ready
**Last Updated**: 2025-11-06
