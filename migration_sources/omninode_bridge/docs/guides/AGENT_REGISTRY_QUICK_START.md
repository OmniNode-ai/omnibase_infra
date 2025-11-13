# Agent Registry Quick Start Guide

**Status**: ✅ Production Ready
**Version**: 1.0.0
**Last Updated**: 2025-11-06

## Overview

The Agent Registry provides intelligent agent discovery and task routing with <5ms cache hits and 85-95% cache hit rate.

## Installation

The agent registry is built-in to `omninode_bridge`. No additional installation required.

```python
from omninode_bridge.agents.registry import AgentRegistry, register_agent
from omninode_bridge.agents.coordination import ThreadSafeState
```

## Quick Start Examples

### 1. Basic Setup

```python
from omninode_bridge.agents.coordination import ThreadSafeState
from omninode_bridge.agents.registry import AgentRegistry

# Initialize
state = ThreadSafeState()
registry = AgentRegistry(state=state, enable_cache=True)
```

### 2. Register an Agent (Manual)

```python
from omninode_bridge.agents.registry import AgentMetadata, AgentType

metadata = AgentMetadata(
    agent_type=AgentType.CONTRACT_INFERENCER,
    version="1.0.0",
    description="Infers contracts from YAML",
    priority=90,
    max_concurrent_tasks=10,
    success_rate=0.95
)

result = registry.register_agent(
    agent_id="contract_inferencer",
    capabilities=["contract_inference", "yaml_parsing"],
    metadata=metadata
)

print(f"Registered in {result.registration_time_ms:.2f}ms")
```

### 3. Register an Agent (Decorator)

```python
from omninode_bridge.agents.registry import register_agent, AgentType

@register_agent(
    agent_id="contract_inferencer_v1",
    capabilities=["contract_inference", "yaml_parsing"],
    agent_type=AgentType.CONTRACT_INFERENCER,
    version="1.0.0",
    description="Infers contracts from YAML specifications",
    priority=90,
    max_concurrent_tasks=10
)
class ContractInferencerAgent:
    async def execute(self, task):
        # Agent implementation
        pass
```

### 4. Discover Agents

```python
# Find all agents with specific capability
agents = registry.discover_agents("contract_inference")

for agent in agents:
    print(f"Found: {agent.agent_id} (priority: {agent.metadata.priority})")
```

### 5. Match Task to Agent

```python
from omninode_bridge.agents.registry import Task

# Define task
task = Task(
    task_type="contract_inference",
    required_capabilities=["contract_inference", "yaml_parsing"],
    complexity="medium"
)

# Match to best agent
agent, confidence = registry.match_agent(task)

print(f"Matched: {agent.agent_id}")
print(f"Confidence: {confidence:.2f}")
```

### 6. Get Confidence Score Breakdown

```python
from omninode_bridge.agents.registry import CapabilityMatchEngine

matcher = CapabilityMatchEngine()
score = matcher.score_agent(agent, task)

print(f"Total confidence: {score.total:.2f}")
print(f"Capability match: {score.capability_score:.2f}")
print(f"Load balance: {score.load_score:.2f}")
print(f"Priority: {score.priority_score:.2f}")
print(f"Success rate: {score.success_rate_score:.2f}")
print(f"Explanation: {score.explanation}")
```

### 7. Monitor Cache Performance

```python
# Get cache statistics
stats = registry.get_cache_stats()

if stats:
    print(f"Cache hit rate: {stats['hit_rate']:.2%}")
    print(f"Total hits: {stats['hits']}")
    print(f"Total misses: {stats['misses']}")
    print(f"Cache size: {stats['size']}/{stats['max_size']}")
```

### 8. Heartbeat Monitoring

```python
import asyncio

async def heartbeat_loop(registry, agent_id, interval=60):
    """Send heartbeat every interval seconds."""
    while True:
        await asyncio.sleep(interval)
        try:
            registry.heartbeat(agent_id)
            print(f"Heartbeat sent for {agent_id}")
        except Exception as e:
            print(f"Heartbeat failed: {e}")

# Start heartbeat task
asyncio.create_task(heartbeat_loop(registry, "contract_inferencer"))
```

### 9. Complete Example: Code Generation Pipeline

```python
from omninode_bridge.agents.coordination import ThreadSafeState
from omninode_bridge.agents.registry import (
    AgentRegistry,
    register_agent,
    AgentType,
    Task
)

# Initialize registry
state = ThreadSafeState()
registry = AgentRegistry(state=state, enable_cache=True)

# Register code generation agents
@register_agent(
    agent_id="contract_inferencer",
    capabilities=["contract_inference", "yaml_parsing"],
    agent_type=AgentType.CONTRACT_INFERENCER,
    version="1.0.0",
    description="Infer ModelContract from YAML",
    priority=90
)
class ContractInferencerAgent:
    async def execute(self, task):
        # Contract inference logic
        pass

@register_agent(
    agent_id="template_selector",
    capabilities=["pattern_detection", "template_selection"],
    agent_type=AgentType.TEMPLATE_SELECTOR,
    version="1.0.0",
    description="Select optimal template variant",
    priority=85
)
class TemplateSelectorAgent:
    async def execute(self, task):
        # Template selection logic
        pass

@register_agent(
    agent_id="business_logic_generator",
    capabilities=["llm_generation", "code_synthesis"],
    agent_type=AgentType.BUSINESS_LOGIC_GENERATOR,
    version="1.0.0",
    description="Generate business logic using LLM",
    priority=80
)
class BusinessLogicGeneratorAgent:
    async def execute(self, task):
        # Business logic generation
        pass

# Create tasks and route to agents
async def generate_node(yaml_content: str):
    # Task 1: Contract inference
    task1 = Task(
        task_type="contract_inference",
        required_capabilities=["contract_inference"]
    )
    agent1, conf1 = registry.match_agent(task1)
    print(f"Task 1: {agent1.agent_id} ({conf1:.2f})")

    # Task 2: Template selection
    task2 = Task(
        task_type="template_selection",
        required_capabilities=["template_selection"]
    )
    agent2, conf2 = registry.match_agent(task2)
    print(f"Task 2: {agent2.agent_id} ({conf2:.2f})")

    # Task 3: Business logic generation
    task3 = Task(
        task_type="business_logic_generation",
        required_capabilities=["llm_generation"]
    )
    agent3, conf3 = registry.match_agent(task3)
    print(f"Task 3: {agent3.agent_id} ({conf3:.2f})")

# Run pipeline
await generate_node("contract.yaml")
```

## Performance Tips

### 1. Enable Caching for Production

```python
registry = AgentRegistry(
    state=state,
    enable_cache=True,        # Enable caching
    cache_ttl_seconds=300,    # 5 minutes TTL
    cache_max_size=1000       # 1000 entries
)
```

### 2. Tune Cache for High Throughput

```python
# High-throughput scenario
registry = AgentRegistry(
    state=state,
    enable_cache=True,
    cache_ttl_seconds=60,     # Shorter TTL (1 minute)
    cache_max_size=5000       # Larger cache (5000 entries)
)
```

### 3. Tune Cache for Low Memory

```python
# Memory-constrained scenario
registry = AgentRegistry(
    state=state,
    enable_cache=True,
    cache_ttl_seconds=600,    # Longer TTL (10 minutes)
    cache_max_size=100        # Smaller cache (100 entries)
)
```

### 4. Disable Cache for Testing

```python
registry = AgentRegistry(
    state=state,
    enable_cache=False  # Disable caching
)
```

## Common Patterns

### Pattern 1: Agent Pool Management

```python
# Register multiple instances of same agent type
for i in range(5):
    metadata = AgentMetadata(
        agent_type=AgentType.CONTRACT_INFERENCER,
        version="1.0.0",
        description=f"Contract inferencer instance {i}",
        max_concurrent_tasks=10
    )

    registry.register_agent(
        agent_id=f"contract_inferencer_{i}",
        capabilities=["contract_inference"],
        metadata=metadata
    )

# Load balancing happens automatically via load_score
```

### Pattern 2: Priority-Based Routing

```python
# High-priority agent
registry.register_agent(
    agent_id="fast_agent",
    capabilities=["inference"],
    metadata=AgentMetadata(
        agent_type=AgentType.CONTRACT_INFERENCER,
        version="1.0.0",
        description="Fast agent",
        priority=90  # Higher priority
    )
)

# Low-priority agent
registry.register_agent(
    agent_id="slow_agent",
    capabilities=["inference"],
    metadata=AgentMetadata(
        agent_type=AgentType.CONTRACT_INFERENCER,
        version="1.0.0",
        description="Slow agent",
        priority=30  # Lower priority
    )
)

# High-priority agent preferred
task = Task(task_type="inference", required_capabilities=["inference"])
agent, conf = registry.match_agent(task)  # Will prefer fast_agent
```

### Pattern 3: Graceful Degradation

```python
from omninode_bridge.agents.registry import NoAgentFoundError

task = Task(
    task_type="specialized_task",
    required_capabilities=["rare_capability"]
)

try:
    agent, confidence = registry.match_agent(task)

    if confidence < 0.5:
        print(f"Warning: Low confidence match ({confidence:.2f})")

    # Use agent
    result = await agent.execute(task)

except NoAgentFoundError:
    print("No suitable agent found - using fallback")
    # Use fallback logic

```

## Error Handling

```python
from omninode_bridge.agents.registry import (
    AgentNotFoundError,
    NoAgentFoundError,
    DuplicateAgentError
)

# Handle duplicate registration
try:
    registry.register_agent(agent_id, capabilities, metadata)
except DuplicateAgentError:
    print(f"Agent {agent_id} already registered")

# Handle missing agent
try:
    agent = registry.get_agent("nonexistent")
except AgentNotFoundError as e:
    print(f"Agent not found: {e.agent_id}")

# Handle no suitable agent
try:
    agent, conf = registry.match_agent(task)
except NoAgentFoundError as e:
    print(f"No agent found for capabilities: {e.required_capabilities}")
```

## Best Practices

1. **Use Decorator Registration**: Simplifies agent setup
2. **Enable Caching**: Improves performance significantly
3. **Monitor Cache Hit Rate**: Aim for 85-95%
4. **Set Realistic Priorities**: 0-100 scale, use full range
5. **Update Success Rates**: Keep metadata current
6. **Send Heartbeats**: Regularly update agent status
7. **Handle NoAgentFoundError**: Always have fallback logic
8. **Log Confidence Scores**: Debug routing decisions

## Testing

```python
import pytest
from omninode_bridge.agents.coordination import ThreadSafeState
from omninode_bridge.agents.registry import AgentRegistry

@pytest.fixture
def registry():
    state = ThreadSafeState()
    return AgentRegistry(state=state, enable_cache=True)

def test_agent_registration(registry):
    result = registry.register_agent(
        agent_id="test",
        capabilities=["test"],
        metadata=metadata
    )
    assert result.success is True

def test_agent_matching(registry):
    # Register agent
    registry.register_agent(...)

    # Match task
    task = Task(...)
    agent, confidence = registry.match_agent(task)

    assert confidence > 0.5
```

## Documentation

- **Full API Reference**: `src/omninode_bridge/agents/registry/README.md`
- **Design Document**: `docs/architecture/AGENT_REGISTRY_DESIGN.md`
- **Implementation Summary**: `AGENT_REGISTRY_IMPLEMENTATION_SUMMARY.md`

## Support

For issues or questions:
1. Check test files: `tests/unit/agents/registry/`
2. Review API reference
3. See implementation summary for examples
