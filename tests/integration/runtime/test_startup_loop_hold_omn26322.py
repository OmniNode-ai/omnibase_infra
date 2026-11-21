# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Kernel startup must not hold the event loop for over one second per call (OMN-26322).

A lab reproduction of the real kernel on one CPU (the dev runtime's ``cpus: 1.0``)
sampled the main thread once per second and recorded the loop lag of a one-second
heartbeat. Two calls held the loop for over one second on the omnibase_infra#6133
head, before /health was served:

- boot step 0 (``run_boot_step0`` -> ``HandlerPackageAnnouncer.handle``): the
  inventory read that imports every node entry point and hashes every contract,
  then the fold of every announcement on the in-memory bus, which never suspends;
- the topic provisioner's constructor, which reads every installed contract to
  build the topic specs, called on the loop by the kernel.

The fixture ``tests/fixtures/omn26322/devboot3_1250_loop_holds.json`` holds the
recorded holds. The tests replay them, capped by ``_REPLAY_CAP_S`` to keep the run
short, through the real handler with a heartbeat on the same loop, and read the
longest gap between heartbeats.

Needs no services: the blocking calls are replaced by sleeps of the recorded length.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import time
from collections.abc import Awaitable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from omnibase_core.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.handlers import handler_package_announcer
from omnibase_infra.handlers.handler_package_announcer import (
    HandlerPackageAnnouncer,
    local_announce_request,
    read_package_inventory,
)
from omnibase_infra.models.node_announce.model_package_inventory import (
    ModelPackageInventory,
)
from omnibase_infra.runtime import service_kernel

pytestmark = [pytest.mark.integration]

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "omn26322"
    / "devboot3_1250_loop_holds.json"
)
_REPLAY_CAP_S = 1.6
_HOLD_LIMIT_S = 1.0


def _fixture() -> dict[str, object]:
    data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _hold(name: str) -> dict[str, object]:
    holds = _fixture()["holds_before"]
    assert isinstance(holds, list)
    for hold in holds:
        if isinstance(hold, dict) and hold.get("name") == name:
            return hold
    raise AssertionError(f"fixture has no recorded hold named {name}")


async def _longest_gap(work: Awaitable[object]) -> tuple[float, object]:
    """Run ``work`` beside a 50 ms heartbeat and return the longest heartbeat gap."""
    gaps: list[float] = []
    done = asyncio.Event()

    async def heartbeat() -> None:
        last = time.monotonic()
        while not done.is_set():
            await asyncio.sleep(0.05)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    beat = asyncio.create_task(heartbeat())
    await asyncio.sleep(0)
    try:
        result = await work
    finally:
        done.set()
        await beat
    return max(gaps, default=0.0), result


def _small_inventory() -> ModelPackageInventory:
    inventory = read_package_inventory(frozenset({"omnibase_infra"}))
    assert inventory.records, "omnibase_infra declares onex.nodes entry points"
    return inventory


@pytest.mark.asyncio
async def test_startup_loop_hold_under_one_second_announcer_inventory() -> None:
    """The recorded inventory read replays on a worker thread, not on the loop."""
    recorded = float(str(_hold("boot_step0")["contiguous_s"]))
    assert recorded > _HOLD_LIMIT_S
    replay = min(recorded, _REPLAY_CAP_S)
    inventory = _small_inventory()

    def slow_read(active: frozenset[str] | None) -> ModelPackageInventory:
        time.sleep(replay)
        return inventory

    bus = EventBusInmemory(environment="test", group="node-announce")
    with patch.object(handler_package_announcer, "read_package_inventory", slow_read):
        gap, result = await _longest_gap(
            HandlerPackageAnnouncer().handle(
                local_announce_request(frozenset({"omnibase_infra"})),
                bus=bus,
                settle_timeout_seconds=30.0,
            )
        )
    assert gap < _HOLD_LIMIT_S, (
        f"boot step 0 held the loop {gap:.2f}s replaying the recorded "
        f"{recorded}s inventory read"
    )
    assert len(result.outcomes) == len(result.announcements) > 0


@pytest.mark.asyncio
async def test_startup_loop_hold_under_one_second_announcer_fold() -> None:
    """The fold gives the loop a turn between announcements."""
    fold = _fixture()["fold"]
    assert isinstance(fold, dict)
    per_item = float(str(fold["seconds_per_announcement"]))
    inventory = _small_inventory()
    count = len(inventory.records)
    # Replay the recorded per-announcement cost over enough items to exceed the cap.
    per_call = max(per_item, _REPLAY_CAP_S / count)
    original = handler_package_announcer.apply_announcement

    def slow_apply(*args: Any, **kwargs: Any) -> Any:
        time.sleep(per_call)
        return original(*args, **kwargs)

    bus = EventBusInmemory(environment="test", group="node-announce")
    with (
        patch.object(
            handler_package_announcer,
            "read_package_inventory",
            lambda active: inventory,
        ),
        patch.object(handler_package_announcer, "apply_announcement", slow_apply),
    ):
        gap, result = await _longest_gap(
            HandlerPackageAnnouncer().handle(
                local_announce_request(frozenset({"omnibase_infra"})),
                bus=bus,
                settle_timeout_seconds=60.0,
            )
        )
    assert per_call * count > _HOLD_LIMIT_S
    assert gap < _HOLD_LIMIT_S, (
        f"the announcement fold held the loop {gap:.2f}s over {count} items"
    )
    assert len(result.outcomes) == count


def test_startup_loop_hold_under_one_second_topic_provisioner() -> None:
    """The kernel builds the topic provisioner (recorded hold) on a worker thread."""
    recorded = float(str(_hold("topic_provisioner_init")["contiguous_s"]))
    assert recorded > _HOLD_LIMIT_S
    tree = ast.parse(inspect.getsource(service_kernel))
    constructions: list[ast.Call] = []
    offloaded: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "TopicProvisioner":
            constructions.append(node)
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "to_thread"
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "TopicProvisioner"
        ):
            offloaded.append(node)
    assert not constructions, (
        "service_kernel constructs TopicProvisioner on the event loop "
        f"(recorded hold {recorded}s on one CPU)"
    )
    assert len(offloaded) == 1


class _BootStepError(RuntimeError):
    """A typed failure raised inside the work moved off the loop."""


@pytest.mark.asyncio
async def test_startup_wiring_error_fails_startup_announcer_inventory() -> None:
    """An inventory failure reaches boot step 0's caller as the same error object."""
    raised = _BootStepError("inventory read failed")

    def failing_read(active: frozenset[str] | None) -> ModelPackageInventory:
        raise raised

    bus = EventBusInmemory(environment="test", group="node-announce")
    with patch.object(
        handler_package_announcer, "read_package_inventory", failing_read
    ):
        with pytest.raises(_BootStepError) as caught:
            await HandlerPackageAnnouncer().handle(
                local_announce_request(frozenset({"omnibase_infra"})),
                bus=bus,
                settle_timeout_seconds=5.0,
            )
    assert caught.value is raised


@pytest.mark.asyncio
async def test_startup_wiring_error_fails_startup_announcer_publish() -> None:
    """A publish failure in the yielding fold still propagates unchanged."""
    raised = _BootStepError("publish failed")
    inventory = _small_inventory()

    async def failing_publish(*args: object, **kwargs: object) -> None:
        raise raised

    bus = EventBusInmemory(environment="test", group="node-announce")
    with (
        patch.object(
            handler_package_announcer,
            "read_package_inventory",
            lambda active: inventory,
        ),
        patch.object(bus, "publish", failing_publish),
    ):
        with pytest.raises(_BootStepError) as caught:
            await HandlerPackageAnnouncer().handle(
                local_announce_request(frozenset({"omnibase_infra"})),
                bus=bus,
                settle_timeout_seconds=5.0,
            )
    assert caught.value is raised


@pytest.mark.asyncio
async def test_startup_wiring_error_fails_startup_topic_provisioner(
    tmp_path: Path,
) -> None:
    """The provisioner's constructor error crosses the worker thread unchanged."""
    from omnibase_infra.event_bus.service_topic_manager import TopicProvisioner

    with pytest.raises(FileNotFoundError):
        await asyncio.to_thread(
            TopicProvisioner,
            bootstrap_servers="127.0.0.1:1",
            contracts_root=tmp_path / "absent",
        )
