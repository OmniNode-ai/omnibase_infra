# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for NodeInvocationAdapter local runtime fallback (OMN-8701).

Proves the platform-level requirement: node execution must not depend exclusively
on the deployed Kafka runtime being healthy. When the runtime is unavailable,
NodeInvocationAdapter must complete via the local in-memory event bus while
preserving contract/topic/payload semantics and producing auditable evidence.

These tests simulate the deployed runtime being unavailable and verify that:
1. Local runtime dispatch completes with full evidence metadata.
2. Commands are stored in the local state store for auditability.
3. Multiple distinct node topics (delegation + chain_orchestrator) work,
   proving platform-level coverage rather than a one-skill patch.
4. The AUTO backend probes health and falls back without error propagation.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_infra.enums.enum_runtime_backend import EnumRuntimeBackend
from omnibase_infra.runtime.models.model_local_state_store import ModelLocalStateStore
from omnibase_infra.runtime.node_invocation_adapter import NodeInvocationAdapter

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Simulated unavailable runtime bus
# ---------------------------------------------------------------------------

_DELEGATION_CMD = "onex.cmd.omnibase-infra.delegation-request.v1"
_DELEGATION_COMPLETED = "onex.evt.omnibase-infra.delegation-completed.v1"
_DELEGATION_FAILED = "onex.evt.omnibase-infra.delegation-failed.v1"

_CHAIN_CMD = "onex.cmd.omnibase-infra.chain-orchestration.v1"
_CHAIN_COMPLETED = "onex.evt.omnibase-infra.chain-completed.v1"
_CHAIN_FAILED = "onex.evt.omnibase-infra.chain-failed.v1"


class _KafkaDownBus:
    """Simulates Kafka being unreachable — health_check reports unhealthy."""

    async def health_check(self) -> dict[str, object]:
        return {"healthy": False, "reason": "broker_unreachable"}

    async def publish(self, *_a: object, **_kw: object) -> None:
        raise OSError("Kafka broker unreachable (simulated in integration test)")

    async def subscribe(self, *_a: object, **_kw: object) -> object:
        raise OSError("Kafka broker unreachable (simulated in integration test)")


class _KafkaExceptionBus:
    """Simulates a bus whose health_check raises — worst-case unavailability."""

    async def health_check(self) -> dict[str, object]:
        raise ConnectionRefusedError(  # kafka-fallback-ok
            "broker connection refused (simulated)"
        )


# ---------------------------------------------------------------------------
# Tests — delegation node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delegation_node_local_fallback_when_kafka_down() -> None:
    """When Kafka bus is unhealthy, delegation dispatch falls back to LOCAL runtime.

    Simulates the documented failure mode from OMN-8701: /onex:delegate
    publishes to onex.cmd.omnibase-infra.delegation-request.v1 via emit daemon;
    when Kafka is unavailable, execution must route through local runtime.
    """
    store = ModelLocalStateStore()
    cid = uuid4()
    adapter = NodeInvocationAdapter(
        event_bus=_KafkaDownBus(),  # type: ignore[arg-type]
        backend=EnumRuntimeBackend.AUTO,
        health_probe_timeout=0.1,
    )
    result = await adapter.dispatch(
        command_topic=_DELEGATION_CMD,
        terminal_events=(_DELEGATION_COMPLETED, _DELEGATION_FAILED),
        payload={"prompt": "implement OMN-8701", "task_type": "code"},
        correlation_id=cid,
        command_name="delegation.orchestrate",
        requester="test_suite",
        state_store=store,
    )

    # Evidence: local path was selected
    assert result["_runtime_backend"] == "local", (
        f"Expected 'local' runtime_backend but got: {result['_runtime_backend']}"
    )
    assert result["_event_bus_backend"] == "inmemory"
    assert result["_state_store_backend"] == "local"
    assert result["_node_contract"] == _DELEGATION_CMD
    assert result["_command_topic"] == _DELEGATION_CMD

    # Command was persisted in local state store
    command_key = f"command:{_DELEGATION_CMD}:{cid}"
    stored = store.get(command_key)
    assert stored is not None, "Command must be in local state store"
    assert stored["command_topic"] == _DELEGATION_CMD
    assert stored["correlation_id"] == str(cid)

    # Status is set
    assert "status" in result


@pytest.mark.asyncio
async def test_delegation_node_local_fallback_when_bus_health_raises() -> None:
    """When bus health_check raises, AUTO mode falls back to LOCAL without error."""
    adapter = NodeInvocationAdapter(
        event_bus=_KafkaExceptionBus(),  # type: ignore[arg-type]
        backend=EnumRuntimeBackend.AUTO,
        health_probe_timeout=0.1,
    )
    result = await adapter.dispatch(
        command_topic=_DELEGATION_CMD,
        terminal_events=(_DELEGATION_COMPLETED, _DELEGATION_FAILED),
        payload={"prompt": "probe", "task_type": "research"},
        correlation_id=uuid4(),
    )
    assert result["_runtime_backend"] == "local"
    assert result["_event_bus_backend"] == "inmemory"


# ---------------------------------------------------------------------------
# Tests — chain_orchestrator node (proves platform-level, not one-skill)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chain_orchestrator_node_local_fallback_when_kafka_down() -> None:
    """Non-delegation node (chain_orchestrator) also dispatches via local runtime.

    Proves NodeInvocationAdapter is platform-level: any node topic works,
    not only the delegation path.
    """
    store = ModelLocalStateStore()
    cid = uuid4()
    adapter = NodeInvocationAdapter(
        event_bus=_KafkaDownBus(),  # type: ignore[arg-type]
        backend=EnumRuntimeBackend.AUTO,
        health_probe_timeout=0.1,
    )
    result = await adapter.dispatch(
        command_topic=_CHAIN_CMD,
        terminal_events=(_CHAIN_COMPLETED, _CHAIN_FAILED),
        payload={"chain_id": "test-chain-integration-001", "steps": ["a", "b"]},
        correlation_id=cid,
        command_name="chain.orchestrate",
        requester="integration_test",
        state_store=store,
    )

    assert result["_runtime_backend"] == "local"
    assert result["_event_bus_backend"] == "inmemory"
    assert result["_node_contract"] == _CHAIN_CMD

    # State store captured the command
    command_key = f"command:{_CHAIN_CMD}:{cid}"
    stored = store.get(command_key)
    assert stored is not None
    assert stored["correlation_id"] == str(cid)


@pytest.mark.asyncio
async def test_local_backend_explicit_no_bus_completes() -> None:
    """LOCAL backend with no event_bus dispatches via in-memory bus directly."""
    store = ModelLocalStateStore()
    cid = uuid4()
    adapter = NodeInvocationAdapter(
        event_bus=None,
        backend=EnumRuntimeBackend.LOCAL,
    )
    result = await adapter.dispatch(
        command_topic=_DELEGATION_CMD,
        terminal_events=(_DELEGATION_COMPLETED, _DELEGATION_FAILED),
        payload={"prompt": "local-only dispatch", "task_type": "audit"},
        correlation_id=cid,
        state_store=store,
    )

    assert result["_runtime_backend"] == "local"
    assert result["_event_bus_backend"] == "inmemory"
    assert result["_state_store_backend"] == "local"
    assert store.size() >= 1


@pytest.mark.asyncio
async def test_local_state_store_is_auditable_after_dispatch() -> None:
    """State store snapshot contains command evidence after local dispatch."""
    store = ModelLocalStateStore()
    cid = uuid4()
    adapter = NodeInvocationAdapter(event_bus=None, backend=EnumRuntimeBackend.LOCAL)
    await adapter.dispatch(
        command_topic=_DELEGATION_CMD,
        terminal_events=(_DELEGATION_COMPLETED,),
        payload={"prompt": "audit-test", "task_type": "verify"},
        correlation_id=cid,
        state_store=store,
    )

    snapshot = store.snapshot()
    assert len(snapshot) >= 1

    command_key = f"command:{_DELEGATION_CMD}:{cid}"
    assert command_key in snapshot
    cmd_record = snapshot[command_key]
    assert cmd_record["command_name"] == "node.invocation"
    assert cmd_record["payload"] == {"prompt": "audit-test", "task_type": "verify"}
