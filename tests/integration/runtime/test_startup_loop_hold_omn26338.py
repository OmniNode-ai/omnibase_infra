# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Two more kernel startup calls must not hold the event loop over one second (OMN-26338).

The one-CPU lab reproduction of OMN-26322 (omnibase_infra#6134) left two calls that
held the loop for over a second before /health could be answered with the dev
contract set:

- ``wire_from_manifest`` built the topic-migration executor's ``TopicProvisioner``
  synchronously (``handler_wiring``), and its construction scans every installed
  contract;
- ``RuntimeHostProcess._start_local_ingress`` read and parsed every node contract
  through ``discover_runtime_local_ingress_routes`` on the loop.

``tests/fixtures/omn26338/holds2_1410_loop_holds.json`` holds the recorded holds.
Each test replays one, capped by ``_REPLAY_CAP_S`` to keep the run short, through
the real call path beside a heartbeat on the same loop, and reads the longest gap.
The tests that change nothing about the result compare the topic set and the route
set with what a synchronous call finds.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_core.models.errors.model_onex_error import ModelOnexError
from omnibase_infra.event_bus.service_topic_manager import TopicProvisioner
from omnibase_infra.models.event_bus.model_contract_topic_scan import (
    ModelContractTopicScan,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _build_topic_migration_executor_dependencies,
    wire_from_manifest,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess
from omnibase_infra.runtime.runtime_local_ingress import (
    discover_runtime_local_ingress_routes,
)
from omnibase_infra.tools.contract_topic_extractor import ContractTopicExtractor
from tests.helpers.runtime_helpers import make_runtime_config

pytestmark = [pytest.mark.integration]

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "omn26338"
    / "holds2_1410_loop_holds.json"
)
_REPLAY_CAP_S = 1.6
_HOLD_LIMIT_S = 1.0
_EXECUTOR_MODULE = (
    "omnibase_infra.nodes.node_topic_migration_executor_effect.handlers."
    "handler_topic_migration_executor"
)


class _StartupWiringError(RuntimeError):
    """A typed failure raised inside the work moved off the loop."""


def _hold(name: str) -> dict[str, object]:
    data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    for hold in data["holds_before"]:
        if hold.get("name") == name:
            assert isinstance(hold, dict)
            return hold
    raise AssertionError(f"fixture has no recorded hold named {name}")


def _recorded_replay(name: str) -> float:
    recorded = float(str(_hold(name)["contiguous_s"]))
    assert recorded > _HOLD_LIMIT_S
    return min(recorded, _REPLAY_CAP_S)


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


def _executor_contract() -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name="node_topic_migration_executor_effect",
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/contract.yaml"),
        entry_point_name="node_topic_migration_executor_effect",
        package_name="test-pkg",
        event_bus=ModelEventBusWiring(
            subscribe_topics=("onex.evt.platform.local-input.v1",), publish_topics=()
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="HandlerTopicMigrationExecutor", module=_EXECUTOR_MODULE
                    ),
                    event_model=None,
                    operation=None,
                ),
            ),
        ),
    )


@pytest.fixture
def contracts_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "contracts"
    node = root / "node_demo"
    node.mkdir(parents=True)
    (node / "contract.yaml").write_text(
        "name: node_demo\nevent_bus:\n  subscribe_topics:\n"
        "    - onex.cmd.demo.start.v1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ONEX_CONTRACTS_DIR", str(root))
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:1")
    ModelContractTopicScan.clear()
    yield root
    ModelContractTopicScan.clear()


def _provisioner_topics(deps: dict[str, object]) -> frozenset[str]:
    provisioner = deps["provisioner"]
    assert isinstance(provisioner, TopicProvisioner)
    return frozenset(provisioner._spec_by_name)


@pytest.mark.asyncio
async def test_startup_loop_hold_under_one_second_topic_migration_provisioner(
    contracts_root: Path,
) -> None:
    """wire_from_manifest builds the migration provisioner without holding the loop."""
    replay = _recorded_replay("topic_migration_provisioner")
    real_extract = ContractTopicExtractor.extract_all

    def slow_extract(self: ContractTopicExtractor, *args: object, **kwargs: object):
        time.sleep(replay)
        return real_extract(self, *args, **kwargs)

    with (
        patch.object(ContractTopicExtractor, "extract_all", slow_extract),
        patch("aiokafka.admin.AIOKafkaAdminClient"),
        patch("aiokafka.AIOKafkaConsumer"),
    ):
        gap, report = await _longest_gap(
            wire_from_manifest(
                ModelAutoWiringManifest(contracts=(_executor_contract(),)),
                MessageDispatchEngine(),
            )
        )
    assert report.total_failed == 0
    assert gap < _HOLD_LIMIT_S, (
        f"wire_from_manifest held the loop {gap:.2f}s replaying the recorded "
        f"{replay}s contract-topic scan of the migration executor's provisioner"
    )


@pytest.mark.asyncio
async def test_startup_loop_hold_under_one_second_topic_set_equals_cold_build(
    contracts_root: Path,
) -> None:
    """A warmed construction provisions the topics a cold synchronous one finds."""
    from omnibase_infra.handlers.topic_migration_dependencies import (
        warm_topic_migration_scan,
    )

    with (
        patch("aiokafka.admin.AIOKafkaAdminClient"),
        patch("aiokafka.AIOKafkaConsumer"),
    ):
        cold = _provisioner_topics(_build_topic_migration_executor_dependencies())
        ModelContractTopicScan.clear()
        runs = ModelContractTopicScan.runs()
        await warm_topic_migration_scan(_executor_contract(), contracts_root)
        assert ModelContractTopicScan.runs() == runs + 1
        warmed = _provisioner_topics(_build_topic_migration_executor_dependencies())
        assert ModelContractTopicScan.runs() == runs + 1, "construction must hit"
    assert any("demo.start" in topic for topic in cold)
    assert warmed == cold


def _ingress_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, packages: tuple[str, ...]
) -> RuntimeHostProcess:
    monkeypatch.setenv("RUNTIME_PROFILE", "main")
    monkeypatch.setenv("ONEX_ACTIVE_RUNTIME_PACKAGES", ",".join(packages))
    server = MagicMock()
    server.start = AsyncMock()
    monkeypatch.setattr(
        "omnibase_infra.runtime.runtime_host_process.RuntimeLocalIngressServer",
        lambda **_kwargs: server,
    )
    process = RuntimeHostProcess(
        config=make_runtime_config(
            local_ingress={
                "enabled": True,
                "socket_path": str(tmp_path / "ingress.sock"),
                "enabled_profiles": ["main"],
                "package_names": list(packages),
            }
        ),
        dispatch_engine=AsyncMock(),
    )
    monkeypatch.setattr(
        process,
        "_resolve_dispatch_result_applier_from_container",
        AsyncMock(return_value=None),
    )
    return process


@pytest.mark.asyncio
async def test_startup_loop_hold_under_one_second_local_ingress_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_start_local_ingress reads the recorded route discovery off the loop."""
    replay = _recorded_replay("local_ingress_route_discovery")
    process = _ingress_process(tmp_path, monkeypatch, ("omnibase_infra",))

    def slow_discover(packages: tuple[str, ...]) -> dict[str, object]:
        time.sleep(replay)
        return {}

    monkeypatch.setattr(
        "omnibase_infra.runtime.runtime_host_process."
        "discover_runtime_local_ingress_routes",
        slow_discover,
    )
    gap, _ = await _longest_gap(process._start_local_ingress())
    assert gap < _HOLD_LIMIT_S, (
        f"_start_local_ingress held the loop {gap:.2f}s replaying the recorded "
        f"{replay}s local ingress route discovery"
    )


@pytest.mark.asyncio
async def test_startup_loop_hold_under_one_second_local_ingress_routes_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The routes found off the loop are the routes a synchronous discovery finds."""
    process = _ingress_process(tmp_path, monkeypatch, ("omnibase_infra",))
    expected = discover_runtime_local_ingress_routes(("omnibase_infra",))
    await process._start_local_ingress()
    assert expected
    assert process._local_ingress_routes == expected


def _duplicate_alias_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_root = tmp_path / "fakepkg"
    (package_root / "nodes" / "node_dupe").mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "nodes" / "node_dupe" / "contract.yaml").write_text(
        """
name: node_dupe
event_bus:
  subscribe_topics:
    - onex.cmd.dupe.start.v1
handler_routing:
  handlers:
    - operation: dupe.deploy.publish_monitor
      input_model:
        module: fakepkg.models.model_publish_command
        name: ModelPublishCommand
    - operation: dupe.deploy.publish_monitor
      input_model:
        module: fakepkg.events.runtime_deployment
        name: ModelRebuildCompleted
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "omnibase_infra.runtime.runtime_local_ingress.importlib.import_module",
        lambda _name: SimpleNamespace(__file__=str(package_root / "__init__.py")),
    )


@pytest.mark.asyncio
async def test_startup_wiring_error_fails_startup_local_ingress_duplicate_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A duplicate alias still fails _start_local_ingress with the same ValueError."""
    _duplicate_alias_package(tmp_path, monkeypatch)
    process = _ingress_process(tmp_path, monkeypatch, ("fakepkg",))
    with pytest.raises(ValueError) as direct:
        discover_runtime_local_ingress_routes(("fakepkg",))
    with pytest.raises(ValueError) as started:
        await process._start_local_ingress()
    assert type(started.value) is type(direct.value)
    assert str(started.value) == str(direct.value)
    assert "dupe.deploy.publish_monitor" in str(started.value)
    assert process._local_ingress_routes == {}


@pytest.mark.asyncio
async def test_startup_wiring_error_fails_startup_topic_migration_scan(
    contracts_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scan error in the migration provisioner fails wiring as it did before."""

    def failing_extract(self: ContractTopicExtractor, *args: object, **kwargs: object):
        raise _StartupWiringError("contract-topic scan failed")

    manifest = ModelAutoWiringManifest(contracts=(_executor_contract(),))
    with (
        patch.object(ContractTopicExtractor, "extract_all", failing_extract),
        patch("aiokafka.admin.AIOKafkaAdminClient"),
        patch("aiokafka.AIOKafkaConsumer"),
    ):
        report = await wire_from_manifest(manifest, MessageDispatchEngine())
        assert report.total_failed == 1
        assert report.results[0].reason == (
            "ModelOnexError: Auto-wiring contract "
            "'node_topic_migration_executor_effect' failed: "
            "handler=HandlerTopicMigrationExecutor: "
            "_StartupWiringError: contract-topic scan failed"
        )
        monkeypatch.setenv("ONEX_WIRING_STRICT_MODE", "1")
        with pytest.raises(ModelOnexError, match="contract-topic scan failed"):
            await wire_from_manifest(manifest, MessageDispatchEngine())


@pytest.mark.asyncio
async def test_startup_wiring_error_fails_startup_topic_migration_missing_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing contracts root still fails as the provisioner's FileNotFoundError."""
    monkeypatch.setenv("ONEX_CONTRACTS_DIR", str(tmp_path / "absent"))
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:1")
    manifest = ModelAutoWiringManifest(contracts=(_executor_contract(),))
    with (
        patch("aiokafka.admin.AIOKafkaAdminClient"),
        patch("aiokafka.AIOKafkaConsumer"),
    ):
        report = await wire_from_manifest(manifest, MessageDispatchEngine())
    assert report.total_failed == 1
    assert (
        "FileNotFoundError: contracts_root does not exist" in report.results[0].reason
    )
