# Agent Registration & Discovery System

**Status**: ✅ Implemented (2025-11-06)
**Performance**: Validated against Phase 4 targets

## Overview

Production-ready agent registration and discovery system with:
- **Dynamic Registration**: Agents register at startup with capabilities
- **Fast Discovery**: <5ms agent lookup with caching (85-95% cache hit rate)
- **Capability Matching**: Intelligent task-to-agent routing with confidence scoring (0.0-1.0)
- **ThreadSafeState Integration**: Centralized state management
- **Type Safety**: Strong Pydantic models for all data structures

## Performance Targets

| Operation | Target | Implementation |
|-----------|--------|----------------|
| **Registration** | <50ms | ✅ Validated |
| **Discovery (cache hit)** | <5ms | ✅ Validated |
| **Discovery (cache miss)** | <100ms | ✅ Validated |
| **Cache hit rate** | 85-95% | ✅ LRU+TTL caching |

## Quick Start

### 1. Decorator-Based Registration

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
    async def execute(self, task: Task) -> Result:
        # Agent implementation
        pass
```

### 2. Manual Registration

```python
from omninode_bridge.agents.registry import AgentRegistry, AgentMetadata, AgentType
from omninode_bridge.agents.coordination import ThreadSafeState

# Initialize
state = ThreadSafeState()
registry = AgentRegistry(state=state, enable_cache=True)

# Register agent
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

### 3. Agent Discovery

```python
# Discover by capability
agents = registry.discover_agents("contract_inference")
for agent in agents:
    print(f"Found: {agent.agent_id} (priority: {agent.metadata.priority})")

# Match task to best agent
from omninode_bridge.agents.registry import Task

task = Task(
    task_type="contract_inference",
    required_capabilities=["contract_inference", "yaml_parsing"],
    complexity="medium"
)

agent, confidence = registry.match_agent(task)
print(f"Matched: {agent.agent_id} with confidence {confidence:.2f}")
```

### 4. Capability Matching

```python
from omninode_bridge.agents.registry import CapabilityMatchEngine

matcher = CapabilityMatchEngine()

# Score agent for task
score = matcher.score_agent(agent, task)
print(f"Total confidence: {score.total:.2f}")
print(f"Capability match: {score.capability_score:.2f}")
print(f"Load balance: {score.load_score:.2f}")
print(f"Priority: {score.priority_score:.2f}")
print(f"Success rate: {score.success_rate_score:.2f}")
print(f"Explanation: {score.explanation}")
```

### 5. Caching

```python
# Cache is enabled by default
registry = AgentRegistry(
    state=state,
    enable_cache=True,
    cache_ttl_seconds=300,  # 5 minutes
    cache_max_size=1000
)

# Check cache statistics
stats = registry.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2%}")
print(f"Total hits: {stats['hits']}")
print(f"Total misses: {stats['misses']}")
```

## API Reference

### Core Classes

#### `AgentRegistry`

Main registry for agent registration and discovery.

**Methods**:
- `register_agent(agent_id, capabilities, metadata)` - Register an agent
- `discover_agents(capability, status_filter)` - Discover agents by capability
- `match_agent(task)` - Match task to best agent with confidence scoring
- `get_agent(agent_id)` - Get agent by ID
- `unregister_agent(agent_id)` - Unregister an agent
- `list_agents()` - List all registered agents
- `heartbeat(agent_id)` - Update agent heartbeat
- `get_cache_stats()` - Get cache statistics

#### `CapabilityMatchEngine`

Multi-criteria agent matching engine.

**Scoring Algorithm**:
- Capability Match: 40% weight (Jaccard similarity)
- Load Balance: 20% weight (1 - active_tasks / max_tasks)
- Priority: 20% weight (agent priority / 100)
- Success Rate: 20% weight (success_rate)

**Methods**:
- `score_agent(agent, task)` - Score agent suitability for task

#### `CacheManager`

LRU cache with TTL for discovery results.

**Methods**:
- `get(key)` - Get value from cache
- `set(key, value)` - Set value in cache
- `invalidate(key)` - Invalidate specific entry
- `invalidate_all()` - Invalidate entire cache
- `get_stats()` - Get cache statistics

### Data Models

#### `AgentInfo`

Complete agent information.

**Fields**:
- `agent_id`: Unique identifier
- `capabilities`: List of capability tags
- `metadata`: Agent metadata
- `registered_at`: Registration timestamp
- `last_heartbeat`: Last heartbeat timestamp
- `status`: Agent status (ACTIVE, INACTIVE, etc.)
- `active_tasks`: Number of active tasks
- `total_tasks_completed`: Total tasks completed
- `total_tasks_failed`: Total tasks failed

#### `AgentMetadata`

Agent metadata.

**Fields**:
- `agent_type`: Agent type enum
- `version`: Semantic version
- `description`: Human-readable description
- `max_concurrent_tasks`: Maximum concurrent tasks
- `timeout_seconds`: Task timeout
- `priority`: Agent priority (0-100)
- `success_rate`: Historical success rate (0.0-1.0)
- `tags`: Additional tags
- `config`: Additional configuration

#### `Task`

Task for agent matching.

**Fields**:
- `task_id`: Unique task identifier
- `task_type`: Task type
- `required_capabilities`: Required capabilities
- `complexity`: Task complexity (low/medium/high)
- `priority`: Task priority (0-100)
- `timeout_seconds`: Task timeout

#### `ConfidenceScore`

Confidence score breakdown.

**Fields**:
- `total`: Total confidence (0.0-1.0)
- `capability_score`: Capability match score
- `load_score`: Load balance score
- `priority_score`: Priority score
- `success_rate_score`: Success rate score
- `explanation`: Human-readable explanation

## Advanced Usage

### Custom Weights for Capability Matching

```python
from omninode_bridge.agents.registry import CapabilityMatchEngine

# Custom weights (must sum to 1.0)
matcher = CapabilityMatchEngine(weights={
    "capability": 0.5,  # 50% weight on capability match
    "load": 0.2,        # 20% weight on load balance
    "priority": 0.2,    # 20% weight on priority
    "success_rate": 0.1 # 10% weight on success rate
})

score = matcher.score_agent(agent, task)
```

### Heartbeat Monitoring

```python
import asyncio

async def heartbeat_loop(registry, agent_id):
    """Send heartbeat every 60 seconds."""
    while True:
        await asyncio.sleep(60)
        try:
            registry.heartbeat(agent_id)
        except Exception as e:
            logger.error(f"Heartbeat failed for {agent_id}: {e}")

# Start heartbeat
asyncio.create_task(heartbeat_loop(registry, "contract_inferencer"))
```

### Agent Lifecycle Management

```python
# Register
result = registry.register_agent(agent_id, capabilities, metadata)

# Update heartbeat periodically
registry.heartbeat(agent_id)

# Unregister on shutdown
registry.unregister_agent(agent_id)
```

## Integration with Code Generation

### Phase 4 Agent Framework

```python
from omninode_bridge.agents.registry import register_agent, AgentType, Task

# 1. Register all code generation agents
@register_agent(
    agent_id="contract_inferencer",
    capabilities=["contract_inference", "yaml_parsing"],
    agent_type=AgentType.CONTRACT_INFERENCER,
    version="1.0.0",
    description="Infer ModelContract from YAML"
)
class ContractInferencerAgent:
    pass

@register_agent(
    agent_id="template_selector",
    capabilities=["pattern_detection", "template_selection"],
    agent_type=AgentType.TEMPLATE_SELECTOR,
    version="1.0.0",
    description="Select optimal template variant"
)
class TemplateSelectorAgent:
    pass

@register_agent(
    agent_id="business_logic_generator",
    capabilities=["llm_generation", "code_synthesis"],
    agent_type=AgentType.BUSINESS_LOGIC_GENERATOR,
    version="1.0.0",
    description="Generate business logic using LLM"
)
class BusinessLogicGeneratorAgent:
    pass

# 2. Use registry to route tasks
async def generate_node(yaml_content: str):
    # Create tasks
    tasks = [
        Task(
            task_type="contract_inference",
            required_capabilities=["contract_inference"]
        ),
        Task(
            task_type="template_selection",
            required_capabilities=["template_selection"]
        ),
        Task(
            task_type="business_logic_generation",
            required_capabilities=["llm_generation"]
        )
    ]

    # Match tasks to agents
    for task in tasks:
        agent, confidence = registry.match_agent(task)
        print(f"Task {task.task_type} → {agent.agent_id} ({confidence:.2f})")
```

## Testing

### Unit Tests

```python
import pytest
from omninode_bridge.agents.registry import (
    AgentRegistry, AgentMetadata, AgentType, Task
)
from omninode_bridge.agents.coordination import ThreadSafeState

@pytest.fixture
def registry():
    state = ThreadSafeState()
    return AgentRegistry(state=state, enable_cache=True)

def test_register_agent(registry):
    metadata = AgentMetadata(
        agent_type=AgentType.CONTRACT_INFERENCER,
        version="1.0.0",
        description="Test agent"
    )

    result = registry.register_agent(
        agent_id="test_agent",
        capabilities=["test_capability"],
        metadata=metadata
    )

    assert result.success is True
    assert result.registration_time_ms < 50

def test_match_agent(registry):
    # Register agent
    metadata = AgentMetadata(
        agent_type=AgentType.CONTRACT_INFERENCER,
        version="1.0.0",
        description="Test agent"
    )
    registry.register_agent(
        agent_id="test_agent",
        capabilities=["contract_inference"],
        metadata=metadata
    )

    # Match task
    task = Task(
        task_type="contract_inference",
        required_capabilities=["contract_inference"]
    )

    agent, confidence = registry.match_agent(task)
    assert agent.agent_id == "test_agent"
    assert confidence > 0.3
```

## Error Handling

### Exception Hierarchy

- `AgentRegistryError` - Base exception
  - `AgentNotFoundError` - Agent not found
  - `NoAgentFoundError` - No suitable agent for task
  - `DuplicateAgentError` - Agent already registered

### Example

```python
from omninode_bridge.agents.registry.exceptions import (
    AgentNotFoundError, NoAgentFoundError, DuplicateAgentError
)

try:
    agent, confidence = registry.match_agent(task)
except NoAgentFoundError as e:
    print(f"No agent found for capabilities: {e.required_capabilities}")
except Exception as e:
    print(f"Error: {e}")
```

## Performance Optimization

### Cache Tuning

```python
# High-throughput scenario: Larger cache, shorter TTL
registry = AgentRegistry(
    state=state,
    enable_cache=True,
    cache_ttl_seconds=60,   # 1 minute
    cache_max_size=5000     # 5000 entries
)

# Low-memory scenario: Smaller cache, longer TTL
registry = AgentRegistry(
    state=state,
    enable_cache=True,
    cache_ttl_seconds=600,  # 10 minutes
    cache_max_size=100      # 100 entries
)
```

### Disable Caching

```python
# For testing or debugging
registry = AgentRegistry(state=state, enable_cache=False)
```

## Documentation

- **Design Document**: `docs/architecture/AGENT_REGISTRY_DESIGN.md`
- **Requirements**: `docs/planning/PHASE_4_FOUNDATION_REQUIREMENTS.md` (Component 4)
- **API Reference**: This README

## License

Part of the omninode_bridge project.
