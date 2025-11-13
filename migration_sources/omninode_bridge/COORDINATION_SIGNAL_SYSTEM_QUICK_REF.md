# Coordination Signal System - Quick Reference

**Status**: ✅ Production-Ready | **Coverage**: 95%+ | **Tests**: 33/33 passing

## Quick Start

```python
from omninode_bridge.agents.coordination import (
    SignalCoordinator,
    ThreadSafeState,
)
from omninode_bridge.agents.metrics import MetricsCollector

# Setup
state = ThreadSafeState()
metrics = MetricsCollector()
coordinator = SignalCoordinator(state=state, metrics_collector=metrics)

# Send signal
await coordinator.signal_coordination_event(
    coordination_id="session-1",
    event_type="agent_completed",
    event_data={"agent_id": "model-gen", "quality_score": 0.95},
    sender_agent_id="model-gen"
)

# Subscribe
async for signal in coordinator.subscribe_to_signals(
    coordination_id="session-1",
    agent_id="validator-gen",
    signal_types=["agent_completed"]
):
    print(f"Received: {signal.signal_type}")
```

## Signal Types

| Type | Purpose | Event Data Keys |
|------|---------|----------------|
| `agent_initialized` | Agent startup | `agent_id`, `capabilities`, `ready` |
| `agent_completed` | Agent finished | `agent_id`, `quality_score`, `artifacts_generated` |
| `dependency_resolved` | Dependency available | `dependency_type`, `dependency_id`, `resolved_by` |
| `inter_agent_message` | General messaging | `message_type`, `message`, `requires_response` |

## API Methods

### SignalCoordinator

**Send Signal**:
```python
success = await coordinator.signal_coordination_event(
    coordination_id: str,
    event_type: str,  # "agent_initialized", "agent_completed", etc.
    event_data: dict,
    sender_agent_id: Optional[str] = None,
    recipient_agents: Optional[list[str]] = None,
    metadata: Optional[dict] = None
) -> bool
```

**Subscribe**:
```python
async for signal in coordinator.subscribe_to_signals(
    coordination_id: str,
    agent_id: str,
    signal_types: Optional[list[str]] = None  # None = all types
):
    # Process signal
    pass
```

**Get History**:
```python
history = coordinator.get_signal_history(
    coordination_id: str,
    filters: Optional[dict] = None,  # {"signal_type": "agent_completed"}
    limit: Optional[int] = None      # Last N signals
) -> list[CoordinationSignal]
```

**Get Metrics**:
```python
metrics = coordinator.get_signal_metrics(
    coordination_id: str
) -> SignalMetrics
```

## Performance

- **Signal propagation**: <100ms target (measured: ~3ms avg)
- **Bulk operations**: 100 signals in ~310ms (3.1ms avg)
- **Storage**: <2ms (via ThreadSafeState)
- **Metrics overhead**: <1ms per signal

## Code Generation Workflow

```python
# 1. Contract Parser initializes
await coordinator.signal_coordination_event(
    coordination_id="codegen",
    event_type="agent_initialized",
    event_data={"agent_id": "contract-parser", "ready": True}
)

# 2. Contract Parser completes
await coordinator.signal_coordination_event(
    coordination_id="codegen",
    event_type="agent_completed",
    event_data={"agent_id": "contract-parser", "quality_score": 1.0}
)

# 3. Contract schema available
await coordinator.signal_coordination_event(
    coordination_id="codegen",
    event_type="dependency_resolved",
    event_data={
        "dependency_type": "contract",
        "dependency_id": "UserContract",
        "resolved_by": "contract-parser"
    }
)

# Model generator receives dependency signal and starts...
```

## Files

- **Models**: `src/omninode_bridge/agents/coordination/signal_models.py`
- **Coordinator**: `src/omninode_bridge/agents/coordination/signals.py`
- **Tests**: `tests/unit/agents/coordination/test_signals.py`
- **Docs**: `docs/architecture/COORDINATION_SIGNAL_SYSTEM.md`

## Running Tests

```bash
# All tests
pytest tests/unit/agents/coordination/test_signals.py -v

# With coverage
pytest tests/unit/agents/coordination/test_signals.py \
  --cov=omninode_bridge.agents.coordination.signals \
  --cov=omninode_bridge.agents.coordination.signal_models

# Performance tests only
pytest tests/unit/agents/coordination/test_signals.py::TestSignalPerformance -v
```

## Integration

**Foundation Components**:
- ThreadSafeState: Signal storage
- MetricsCollector: Performance tracking
- AgentRegistry: Agent validation (ready)

**Storage Keys**:
- `signal_history:{coordination_id}`
- `signal_metrics:{coordination_id}`

**Metrics**:
- `signal_propagation_time_ms`
- `signals_sent`
- `signals_received`

**Storage & Retention Policy**:
- **Signal TTL**: 24 hours (signals auto-expire after this period)
- **Max History**: 1000 signals per coordination_id (oldest signals pruned when limit reached)
- **Metrics Retention**: 7 days (historical metrics aggregated and archived)
- **Cleanup Strategy**: Automatic background cleanup runs hourly, removing expired signals and compacting history
