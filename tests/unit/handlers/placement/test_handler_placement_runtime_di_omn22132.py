# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22132: HandlerPlacementRuntime takes every dependency by constructor injection.

The handler is built twice per profile: once from fakes through the three ports,
and once through ``RegistryInfraPlacementRuntime`` over an in-memory store. The
``inmemory`` and ``redpanda`` profiles are the two binding blocks of the node
contract; each supplies every timing the handler uses.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import pytest
import yaml

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.handlers.placement import (
    EnumPlacementHealthVerdict,
    HandlerPlacementEtcd,
    HandlerPlacementRuntime,
    ModelPlacementPolicy,
    ModelPlacementServingReport,
    ProtocolPlacedNode,
    evaluate_serving_health,
)
from omnibase_infra.models.model_placement_runtime_settings import (
    ModelPlacementRuntimeSettings,
)
from omnibase_infra.models.model_placement_serving_request import (
    ModelPlacementServingRequest,
)
from omnibase_infra.nodes.node_placement_runtime_effect.registry.registry_infra_placement_runtime import (
    RegistryInfraPlacementRuntime,
)

pytestmark = pytest.mark.unit

CONTRACT = (
    Path(__file__).resolve().parents[4]
    / "src/omnibase_infra/nodes/node_placement_runtime_effect/contract.yaml"
)
PROFILES = ("inmemory", "redpanda")


class FakeSession:
    def __init__(
        self,
        runtime_id: str = "rt-1",
        node_names: tuple[str, ...] = ("any_node", "primary_node"),
    ) -> None:
        self.runtime_id = runtime_id
        self.node_names = node_names
        self.lease_id = 0
        self.calls: list[str] = []

    async def open(self) -> None:
        self.lease_id = 7
        self.calls.append("open")

    async def keepalive(self) -> bool:
        self.calls.append("keepalive")
        return True

    def drop(self) -> None:
        self.lease_id = 0

    async def close(self) -> None:
        self.calls.append("close")
        self.lease_id = 0


class FakeElection:
    def __init__(self, *, wins: bool) -> None:
        self.epoch = 0
        self._wins = wins
        self.stepped_down = 0

    def acting(self) -> bool:
        return self.epoch != 0

    async def campaign_once(self) -> bool:
        if self._wins:
            self.epoch = 11
        return self.acting()

    async def verify_holder(self) -> bool:
        return self.acting()

    def step_down(self) -> None:
        self.epoch = 0
        self.stepped_down += 1


class FakeNode:
    def __init__(self) -> None:
        self.log: list[tuple[str, int]] = []

    async def rebuild(self, placement_epoch: int) -> None:
        self.log.append(("rebuild", placement_epoch))

    async def start_consuming(self, placement_epoch: int) -> None:
        self.log.append(("start", placement_epoch))

    async def stop_consuming(self) -> None:
        self.log.append(("stop", 0))


class FakeStore:
    """An in-memory stand-in for the etcd gateway, behind the adapter's calls."""

    def __init__(self) -> None:
        self.keys: dict[str, tuple[str, int]] = {}
        self.revision = 100

    async def lease_grant(self, ttl_s: int) -> tuple[int, int]:
        return 5, ttl_s

    async def lease_keepalive(self, lease_id: int) -> int:
        return 4

    async def lease_revoke(self, lease_id: int) -> None:
        self.keys.clear()

    async def put(self, key: str, value: str, *, lease_id: int) -> int:
        self.revision += 1
        self.keys[key] = (value, self.revision)
        return self.revision

    async def create_if_absent(
        self, key: str, value: str, *, lease_id: int
    ) -> tuple[bool, int]:
        if key in self.keys:
            return False, 0
        self.revision += 1
        self.keys[key] = (value, self.revision)
        return True, self.revision

    async def get(self, key: str) -> tuple[str, int] | None:
        return self.keys.get(key)

    async def get_prefix(self, prefix: str) -> Mapping[str, str]:
        return {k: v for k, (v, _) in self.keys.items() if k.startswith(prefix)}


def _settings(profile: str) -> ModelPlacementRuntimeSettings:
    return RegistryInfraPlacementRuntime.settings_from_contract(
        yaml.safe_load(CONTRACT.read_text()), profile
    )


def _nodes(
    any_node: FakeNode, primary_node: FakeNode
) -> dict[str, tuple[ModelPlacementPolicy, ProtocolPlacedNode]]:
    return {
        "any_node": (ModelPlacementPolicy(mode="any"), any_node),
        "primary_node": (ModelPlacementPolicy(mode="primary"), primary_node),
    }


@pytest.mark.parametrize("profile", PROFILES)
async def test_handler_runs_on_injected_fakes(profile: str) -> None:
    """Every collaborator is a fake behind its port; the profile supplies the timings."""
    settings = _settings(profile)
    session, election = FakeSession(), FakeElection(wins=True)
    any_node, primary_node = FakeNode(), FakeNode()
    reads: list[tuple[Sequence[str], int]] = []

    async def serving_reader(
        *, expected_nodes: Sequence[str], min_serving: int
    ) -> ModelPlacementServingReport:
        reads.append((tuple(expected_nodes), min_serving))
        return evaluate_serving_health([], min_serving=min_serving)

    runtime = HandlerPlacementRuntime(
        runtime_id="rt-1",
        session=session,
        elections={"primary_node": election},
        nodes=_nodes(any_node, primary_node),
        serving_reader=serving_reader,
        lock=asyncio.Lock(),
        min_serving=settings.min_serving,
        keepalive_interval_s=settings.keepalive_interval_s,
        campaign_interval_s=0.01,
        fence_tick_s=0.01,
    )
    await runtime.start()
    try:
        for _ in range(100):
            if "primary_node" in runtime.started_nodes():
                break
            await asyncio.sleep(0.01)
        assert runtime.started_nodes() == {"any_node": 0, "primary_node": 11}
        assert any_node.log == [("start", 0)]
        assert primary_node.log == [("rebuild", 11), ("start", 11)]
        assert runtime.may_publish("primary_node")
        assert runtime.stamp("primary_node", "go").placement_epoch == 11
        report = await runtime.serving_report()
        assert report.verdict is EnumPlacementHealthVerdict.OK
        assert reads == [(("any_node", "primary_node"), settings.min_serving)]
    finally:
        await runtime.stop()
    assert session.calls[0] == "open"
    assert session.calls[-1] == "close"
    assert election.stepped_down >= 1
    assert primary_node.log[-1] == ("stop", 0)


@pytest.mark.parametrize("profile", PROFILES)
async def test_registry_wires_the_handler_over_an_in_memory_store(
    profile: str,
) -> None:
    """The registry builds session, elections and serving reader from the contract profile."""
    settings = _settings(profile)
    store = FakeStore()
    registry = RegistryInfraPlacementRuntime(
        adapter=cast("HandlerPlacementEtcd", store), settings=settings
    )
    any_node, primary_node = FakeNode(), FakeNode()
    runtime = registry.build(runtime_id="rt-1", nodes=_nodes(any_node, primary_node))
    await runtime.start()
    try:
        for _ in range(200):
            if "primary_node" in runtime.started_nodes():
                break
            await asyncio.sleep(0.01)
        started = runtime.started_nodes()
        assert started["any_node"] == 0
        assert started["primary_node"] > 0
        assert primary_node.log[0][0] == "rebuild"
        member = json.loads(store.keys["/onex/placement/members/rt-1"][0])
        assert member == {"runtime_id": "rt-1", "nodes": ["any_node", "primary_node"]}
        assert store.keys["/onex/placement/primary/primary_node"][0] == "rt-1"
        report = await runtime.serving_report()
        assert report.min_serving == settings.min_serving
        assert report.verdict is EnumPlacementHealthVerdict.FAIL
    finally:
        await runtime.stop()


def test_profiles_are_declared_and_differ_only_by_contract_values() -> None:
    inmemory, redpanda = _settings("inmemory"), _settings("redpanda")
    assert inmemory != redpanda
    assert redpanda.ttl_s > inmemory.ttl_s
    with pytest.raises(KeyError):
        _settings("undeclared")


def test_settings_refuse_a_margin_at_or_above_the_ttl() -> None:
    with pytest.raises(ValueError, match="margin_s must be below ttl_s"):
        ModelPlacementRuntimeSettings(
            ttl_s=2,
            margin_s=2.0,
            keepalive_interval_s=1.0,
            campaign_interval_s=0.25,
            fence_tick_s=0.1,
            min_serving=2,
        )


@pytest.mark.parametrize("profile", PROFILES)
async def test_handle_answers_one_placement_decision(profile: str) -> None:
    """``handle`` returns the serving report, with the profile's min_serving by default."""
    settings = _settings(profile)
    runtime = RegistryInfraPlacementRuntime(
        adapter=cast("HandlerPlacementEtcd", FakeStore()), settings=settings
    ).build(runtime_id="rt-1", nodes=_nodes(FakeNode(), FakeNode()))
    default = await runtime.handle(ModelPlacementServingRequest())
    override = await runtime.handle(ModelPlacementServingRequest(min_serving=1))
    assert default.min_serving == settings.min_serving
    assert override.min_serving == 1
    assert default.verdict is EnumPlacementHealthVerdict.FAIL
    assert default.under_served == ("any_node", "primary_node")


async def _no_serving(
    *, expected_nodes: Sequence[str], min_serving: int
) -> ModelPlacementServingReport:
    return evaluate_serving_health([], min_serving=min_serving)


def _build(
    *,
    session: FakeSession,
    elections: Mapping[str, FakeElection],
    runtime_id: str = "rt-1",
) -> HandlerPlacementRuntime:
    return HandlerPlacementRuntime(
        runtime_id=runtime_id,
        session=session,
        elections=elections,
        nodes=_nodes(FakeNode(), FakeNode()),
        serving_reader=_no_serving,
        lock=asyncio.Lock(),
        min_serving=1,
        keepalive_interval_s=0.1,
        campaign_interval_s=0.01,
        fence_tick_s=0.01,
    )


def test_handler_copies_the_nodes_and_elections_mappings() -> None:
    elections = {"primary_node": FakeElection(wins=True)}
    runtime = _build(session=FakeSession(), elections=elections)
    elections["ghost"] = FakeElection(wins=True)
    assert set(runtime._elections) == {"primary_node"}


@pytest.mark.parametrize(
    "elections",
    [
        {},
        {"primary_node": FakeElection(wins=True), "any_node": FakeElection(wins=True)},
        {"primary_node": FakeElection(wins=True), "ghost": FakeElection(wins=True)},
    ],
)
def test_handler_rejects_elections_not_matching_primary_nodes(
    elections: Mapping[str, FakeElection],
) -> None:
    with pytest.raises(ProtocolConfigurationError, match="primary-mode nodes"):
        _build(session=FakeSession(), elections=elections)


@pytest.mark.parametrize(
    "session",
    [
        FakeSession(runtime_id="other"),
        FakeSession(node_names=("any_node",)),
    ],
)
def test_handler_rejects_session_disagreeing_on_identity(
    session: FakeSession,
) -> None:
    with pytest.raises(ProtocolConfigurationError, match="disagree"):
        _build(session=session, elections={"primary_node": FakeElection(wins=True)})


@pytest.mark.parametrize(
    "override",
    [
        {"keepalive_interval_s": 2.5},
        {"keepalive_interval_s": 3.0},
        {"fence_tick_s": 1.5},
        {"fence_tick_s": 2.0},
    ],
)
def test_settings_reject_loops_slower_than_the_lease(
    override: dict[str, float],
) -> None:
    base = {
        "ttl_s": 4,
        "margin_s": 1.5,
        "keepalive_interval_s": 1.0,
        "campaign_interval_s": 0.25,
        "fence_tick_s": 0.1,
        "min_serving": 2,
    }
    with pytest.raises(ValueError):
        ModelPlacementRuntimeSettings.model_validate({**base, **override})


@pytest.mark.parametrize("bad", [0, -1])
async def test_serving_report_rejects_non_positive_min_serving(bad: int) -> None:
    runtime = _build(
        session=FakeSession(), elections={"primary_node": FakeElection(wins=True)}
    )
    with pytest.raises(ProtocolConfigurationError, match="min_serving"):
        await runtime.serving_report(min_serving=bad)
