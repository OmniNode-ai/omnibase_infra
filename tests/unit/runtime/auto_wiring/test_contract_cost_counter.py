# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Per-contract dispatch cost counter (OMN-26250).

Validates:
- Dispatching one record to two fake handlers with known sleeps gives each its
  own wall time within 20 percent, split into on-loop and off-loop time
- Count and wire bytes are charged to the contract that dispatched
- Every attached contract reads every window, including one that dispatched
  nothing, and closing a window zeroes the counters
- The reading refuses a missing window and a window that ends before it starts
- The wiring wraps the dispatcher of a prepared contract entry and registers
  the contract when it is committed
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from omnibase_infra.handlers.contract_dispatch_cost import (
    ContractDispatchCostCounter,
    measure_contract_dispatch,
    record_ingress_bytes,
)
from omnibase_infra.handlers.entry_method_offload import invoke_entry_method
from omnibase_infra.models.health.model_contract_cost_reading import (
    ModelContractCostReading,
)

pytestmark = pytest.mark.unit

_START = datetime(2026, 11, 21, 0, 0, 0, tzinfo=UTC)
_END = _START + timedelta(seconds=60)
_SLEEP_SYNC = 0.2
_SLEEP_ASYNC = 0.1


def _sync_handle(_arg: object) -> str:
    time.sleep(_SLEEP_SYNC)
    return "sync"


async def _async_handle(_arg: object) -> str:
    await asyncio.sleep(_SLEEP_ASYNC)
    return "async"


def _dispatcher(handle: Any) -> Any:
    async def _dispatch(envelope: object) -> object:
        return await invoke_entry_method(handle, envelope)

    return _dispatch


async def test_two_contracts_each_read_their_own_wall_time() -> None:
    counter = ContractDispatchCostCounter()
    counter.register("contract_sync")
    counter.register("contract_async")
    sync_dispatch = measure_contract_dispatch(
        "contract_sync", _dispatcher(_sync_handle), counter
    )
    async_dispatch = measure_contract_dispatch(
        "contract_async", _dispatcher(_async_handle), counter
    )

    record_ingress_bytes(123)
    await asyncio.gather(sync_dispatch("record"), async_dispatch("record"))

    by_name = {r.contract_name: r for r in counter.close_window(_START, _END)}
    sync_reading = by_name["contract_sync"]
    async_reading = by_name["contract_async"]

    sync_total = sync_reading.on_loop_wall_ms + sync_reading.off_loop_wall_ms
    async_total = async_reading.on_loop_wall_ms + async_reading.off_loop_wall_ms
    assert sync_total == pytest.approx(_SLEEP_SYNC * 1000, rel=0.2)
    assert async_total == pytest.approx(_SLEEP_ASYNC * 1000, rel=0.2)
    # A plain function runs on a worker thread; an async handler stays on the loop.
    assert sync_reading.off_loop_wall_ms == pytest.approx(_SLEEP_SYNC * 1000, rel=0.2)
    assert async_reading.off_loop_wall_ms == 0.0
    assert async_reading.on_loop_wall_ms == pytest.approx(_SLEEP_ASYNC * 1000, rel=0.2)
    assert sync_reading.dispatch_count == async_reading.dispatch_count == 1
    assert sync_reading.payload_bytes == async_reading.payload_bytes == 123


async def test_a_failing_dispatch_is_still_charged() -> None:
    counter = ContractDispatchCostCounter()

    async def _boom(_envelope: object) -> object:
        await asyncio.sleep(0.05)
        raise RuntimeError("handler failed")

    dispatch = measure_contract_dispatch("contract_boom", _boom, counter)
    with pytest.raises(RuntimeError):
        await dispatch("record")

    (reading,) = counter.close_window(_START, _END)
    assert reading.dispatch_count == 1
    assert reading.on_loop_wall_ms == pytest.approx(50.0, rel=0.5)


def test_every_attached_contract_reads_every_window() -> None:
    counter = ContractDispatchCostCounter()
    counter.register("contract_b")
    counter.register("contract_a")

    first = counter.close_window(_START, _END)
    assert [r.contract_name for r in first] == ["contract_a", "contract_b"]
    assert all(
        (r.dispatch_count, r.on_loop_wall_ms, r.off_loop_wall_ms, r.payload_bytes)
        == (0, 0.0, 0.0, 0)
        for r in first
    )

    counter.record(
        "contract_a", wall_seconds=0.3, off_loop_seconds=0.1, payload_bytes=7
    )
    second = {
        r.contract_name: r for r in counter.close_window(_END, _END + (_END - _START))
    }
    assert second["contract_a"].dispatch_count == 1
    assert second["contract_a"].on_loop_wall_ms == pytest.approx(200.0)
    assert second["contract_a"].off_loop_wall_ms == pytest.approx(100.0)
    assert second["contract_a"].payload_bytes == 7
    # Closing zeroed the counters but kept the contract.
    third = {
        r.contract_name: r for r in counter.close_window(_END, _END + timedelta(1))
    }
    assert third["contract_a"].dispatch_count == 0
    assert set(third) == {"contract_a", "contract_b"}


def _reading_fields() -> dict[str, Any]:
    return {
        "contract_name": "contract_a",
        "window_start": _START,
        "window_end": _END,
        "dispatch_count": 1,
        "on_loop_wall_ms": 1.0,
        "off_loop_wall_ms": 0.0,
        "payload_bytes": 10,
    }


@pytest.mark.parametrize("missing", sorted(_reading_fields()))
def test_reading_refuses_each_missing_field(missing: str) -> None:
    fields = _reading_fields()
    del fields[missing]
    with pytest.raises(ValidationError):
        ModelContractCostReading(**fields)


def test_reading_refuses_a_window_that_ends_before_it_starts() -> None:
    fields = {**_reading_fields(), "window_end": _START}
    with pytest.raises(ValidationError):
        ModelContractCostReading(**fields)


def test_reading_refuses_a_negative_measurement() -> None:
    with pytest.raises(ValidationError):
        ModelContractCostReading(**{**_reading_fields(), "on_loop_wall_ms": -1.0})


# ---------------------------------------------------------------------------
# Wiring seam: the dispatcher the runtime attaches is the measured one
# ---------------------------------------------------------------------------


class _AttachedHandler:
    async def handle(self, envelope: object) -> None:
        await asyncio.sleep(0.05)


def _attached_contract(name: str) -> Any:
    from pathlib import Path

    from omnibase_core.models.contracts.model_contract_version import (
        ModelContractVersion,
    )
    from omnibase_infra.runtime.auto_wiring.models import (
        ModelDiscoveredContract,
        ModelEventBusWiring,
        ModelHandlerRef,
        ModelHandlerRouting,
        ModelHandlerRoutingEntry,
    )

    return ModelDiscoveredContract(
        name=name,
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/contract.yaml"),
        entry_point_name=name,
        package_name="test-pkg",
        event_bus=ModelEventBusWiring(
            subscribe_topics=("onex.evt.platform.local-input.v1",), publish_topics=()
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(name="HandlerFoo", module="fake.module"),
                    event_model=None,
                    operation=None,
                ),
            ),
        ),
    )


async def test_attached_contract_is_registered_and_its_dispatch_is_charged() -> None:
    from unittest.mock import MagicMock, patch

    from omnibase_core.services.service_handler_resolver import (
        ServiceHandlerResolver,
    )
    from omnibase_core.services.service_local_handler_ownership_query import (
        ServiceLocalHandlerOwnershipQuery,
    )
    from omnibase_infra.handlers.contract_dispatch_cost import (
        get_process_contract_cost_counter,
    )
    from omnibase_infra.runtime.auto_wiring.handler_wiring import (
        _commit_handler_wiring,
        _prepare_handler_wiring,
    )
    from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

    name = "node_cost_counter_wiring_omn26250"
    contract = _attached_contract(name)
    entry = contract.handler_routing.handlers[0]
    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=_AttachedHandler,
    ):
        prepared = _prepare_handler_wiring(
            contract=contract,
            entry=entry,
            dispatch_engine=None,
            resolver=ServiceHandlerResolver(),
            ownership_query=ServiceLocalHandlerOwnershipQuery(
                local_node_names=frozenset({name})
            ),
            event_bus=None,
            container=None,
        )
    counter = get_process_contract_cost_counter()
    counter.close_window(_START, _END)  # drain whatever other tests left

    # Committing attaches the contract: it reads, with zero dispatches, at once,
    # and the dispatcher the engine receives is the measured one.
    engine = MagicMock(spec=MessageDispatchEngine)
    engine.is_frozen = False
    _commit_handler_wiring(prepared, engine, owner_contract_name=name)
    attached = {r.contract_name: r for r in counter.close_window(_START, _END)}
    assert attached[name].dispatch_count == 0

    registered = engine.register_dispatcher.call_args.kwargs["dispatcher"]
    await registered(MagicMock())
    charged = {r.contract_name: r for r in counter.close_window(_START, _END)}
    assert charged[name].dispatch_count == 1
    assert charged[name].on_loop_wall_ms == pytest.approx(50.0, rel=0.5)


def test_a_registration_with_no_owner_is_not_an_attached_contract() -> None:
    from omnibase_infra.handlers.contract_dispatch_cost import (
        attach_contract_dispatcher,
    )

    async def _dispatch(envelope: object) -> None:
        return None

    assert attach_contract_dispatcher(None, _dispatch) is _dispatch


async def test_a_consumed_message_carries_its_wire_size_to_the_dispatch() -> None:
    from omnibase_infra.handlers.handler_runtime_loop_lag_sampler import (
        RuntimeLoopLagSampler,
    )

    counter = ContractDispatchCostCounter()
    dispatch = measure_contract_dispatch(
        "contract_bytes", _dispatcher(_async_handle), counter
    )
    RuntimeLoopLagSampler().observe_message("onex.evt.test-service.other.v1", {}, 456)
    await dispatch("record")

    (reading,) = counter.close_window(_START, _END)
    assert reading.payload_bytes == 456
