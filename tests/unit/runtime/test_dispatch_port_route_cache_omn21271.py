# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21271: the delegation dispatch port resolves its routes once, off the loop.

Before this change ``RuntimeDelegationDispatchPort._resolved_routes`` parsed
every installed node contract synchronously on every delegate-skill dispatch,
on the runtime's event loop. These tests pin that a resolved port reads no
contract again until the installed contract set changes, and that the first
resolution runs in a worker thread.
"""

from __future__ import annotations

import asyncio
import itertools
import os
import pathlib
import shutil
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from omnibase_core.models.dispatch.model_dispatch_bus_command import (
    ModelDispatchBusCommand,
)
from omnibase_core.models.dispatch.model_dispatch_bus_terminal_result import (
    ModelDispatchBusTerminalResult,
)
from omnibase_infra.enums.enum_consumer_group_purpose import EnumConsumerGroupPurpose
from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.event_bus.models.model_publish_receipt import (
    ModelPublishReceipt,
)
from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.runtime.runtime_local_ingress import ModelRuntimeLocalIngressRoute
from omnibase_infra.runtime.service_delegation_dispatch_port import (
    RuntimeDelegationDispatchPort,
)

pytestmark = pytest.mark.unit

_PORT_MODULE = "omnibase_infra.runtime.service_delegation_dispatch_port"
_INGRESS_MODULE = "omnibase_infra.runtime.runtime_local_ingress"

_DELEGATION_CONTRACT = """
name: node_delegation_orchestrator
event_bus:
  subscribe_topics:
    - {command_topic}
terminal_events:
  success: onex.evt.omnibase-infra.delegation-completed.v1
  failure: onex.evt.omnibase-infra.delegation-failed.v1
handler_routing:
  handlers:
    - operation: delegation.orchestrate
""".strip()

_DEMO_CONTRACT = """
name: {name}
event_bus:
  subscribe_topics:
    - {command_topic}
""".strip()


class _FakeRuntimeBus:
    """Satisfies the broker transport; the broker is faked, so it is never used."""

    environment = "test"
    bootstrap_servers = "localhost:9092"

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: ModelEventHeaders | None = None,
    ) -> ModelPublishReceipt | None:
        raise AssertionError("route resolution tests never publish")

    async def subscribe(
        self,
        topic: str,
        node_identity: ModelNodeIdentity | None = None,
        on_message: Callable[[ModelEventMessage], Awaitable[None]] | None = None,
        *,
        group_id: str | None = None,
        purpose: EnumConsumerGroupPurpose = EnumConsumerGroupPurpose.CONSUME,
        required_for_readiness: bool = False,
    ) -> Callable[[], Awaitable[None]]:
        raise AssertionError("route resolution tests never subscribe")


class _ContractReadCounter:
    """Counts ``Path.read_text`` calls on files named ``contract.yaml``."""

    def __init__(self) -> None:
        self.paths: list[str] = []
        self._lock = threading.Lock()

    @property
    def count(self) -> int:
        with self._lock:
            return len(self.paths)

    def record(self, path: Path) -> None:
        with self._lock:
            self.paths.append(str(path))


def _write_node(package_root: Path, node: str, text: str) -> Path:
    node_dir = package_root / "nodes" / node
    node_dir.mkdir(parents=True, exist_ok=True)
    contract_path = node_dir / "contract.yaml"
    contract_path.write_text(text, encoding="utf-8")
    return contract_path


def _fake_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, package_name: str
) -> Path:
    package_root = tmp_path / package_name
    package_root.mkdir()
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        f"{_INGRESS_MODULE}.importlib.import_module",
        lambda _name: SimpleNamespace(__file__=str(package_root / "__init__.py")),
    )
    return package_root


@pytest.fixture
def read_counter(monkeypatch: pytest.MonkeyPatch) -> _ContractReadCounter:
    counter = _ContractReadCounter()
    original = pathlib.Path.read_text

    def counting_read_text(
        self: Path, encoding: str | None = None, errors: str | None = None
    ) -> str:
        if self.name == "contract.yaml":
            counter.record(self)
        return original(self, encoding=encoding, errors=errors)

    monkeypatch.setattr(pathlib.Path, "read_text", counting_read_text)
    return counter


def _demo_route(command_topic: str) -> ModelRuntimeLocalIngressRoute:
    return ModelRuntimeLocalIngressRoute(
        node_name="node_demo",
        contract_name="demo",
        command_topic=command_topic,
        event_type=None,
        terminal_event=None,
        contract_path="/contracts/fakepkg/node_demo/contract.yaml",
        package_name="fakepkg",
    )


@pytest.mark.asyncio
async def test_second_resolution_reads_no_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    read_counter: _ContractReadCounter,
) -> None:
    package_root = _fake_package(tmp_path, monkeypatch, "fakepkg")
    _write_node(
        package_root,
        "node_demo",
        _DEMO_CONTRACT.format(name="demo", command_topic="onex.cmd.demo.start.v1"),
    )
    port = RuntimeDelegationDispatchPort(_FakeRuntimeBus(), package_names=("fakepkg",))

    first = await port._resolved_routes()
    assert read_counter.count == 1
    second = await port._resolved_routes()
    assert read_counter.count == 1

    assert first == second
    assert first["demo"].command_topic == "onex.cmd.demo.start.v1"
    assert first is not second
    first.clear()
    third = await port._resolved_routes()
    assert third == second
    assert read_counter.count == 1


@pytest.mark.asyncio
async def test_second_dispatch_reads_no_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    read_counter: _ContractReadCounter,
) -> None:
    package_root = _fake_package(tmp_path, monkeypatch, "omnimarket")
    _write_node(
        package_root,
        "node_delegation_orchestrator",
        _DELEGATION_CONTRACT.format(
            command_topic="onex.cmd.omnibase-infra.delegation-request.v1"
        ),
    )
    commands: list[ModelDispatchBusCommand] = []

    class FakeBroker:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def dispatch_request(
            self, command: ModelDispatchBusCommand
        ) -> tuple[None, ModelDispatchBusTerminalResult]:
            commands.append(command)
            return None, ModelDispatchBusTerminalResult(
                correlation_id=command.correlation_id,
                status="completed",
                payload={"content": "ok"},
                completed_at=datetime.now(UTC),
            )

    def no_bounded_route(**_kwargs: object) -> None:
        return None

    monkeypatch.setattr(f"{_PORT_MODULE}.RuntimePatternBBroker", FakeBroker)
    monkeypatch.setattr(
        f"{_PORT_MODULE}.resolve_bounded_delegation_route", no_bounded_route
    )
    port = RuntimeDelegationDispatchPort(
        _FakeRuntimeBus(), package_names=("omnimarket",)
    )

    reads_after: list[int] = []
    for _ in range(2):
        result = await port.dispatch(
            prompt="probe",
            task_type="document",
            correlation_id=uuid4(),
            max_tokens=64,
            source_file_path=None,
            source_session_id=None,
            wait=True,
        )
        assert result["status"] == "completed"
        reads_after.append(read_counter.count)

    assert reads_after == [1, 1]
    assert [command.command_name for command in commands] == [
        "delegation.orchestrate"
    ] * 2


@pytest.mark.asyncio
async def test_first_resolution_runs_off_the_event_loop_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_package(tmp_path, monkeypatch, "fakepkg")
    discovery_threads: list[int] = []

    def recording_discovery(
        package_names: Sequence[str],
    ) -> dict[str, ModelRuntimeLocalIngressRoute]:
        discovery_threads.append(threading.get_ident())
        return {"demo": _demo_route("onex.cmd.demo.start.v1")}

    monkeypatch.setattr(
        f"{_PORT_MODULE}.discover_runtime_local_ingress_routes", recording_discovery
    )
    port = RuntimeDelegationDispatchPort(_FakeRuntimeBus(), package_names=("fakepkg",))

    routes = await port._resolved_routes()

    assert routes["demo"].command_topic == "onex.cmd.demo.start.v1"
    assert len(discovery_threads) == 1
    assert discovery_threads[0] != threading.get_ident()


@pytest.mark.asyncio
async def test_slow_resolution_does_not_stall_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_package(tmp_path, monkeypatch, "fakepkg")

    def slow_discovery(
        package_names: Sequence[str],
    ) -> dict[str, ModelRuntimeLocalIngressRoute]:
        time.sleep(0.4)
        return {"demo": _demo_route("onex.cmd.demo.start.v1")}

    monkeypatch.setattr(
        f"{_PORT_MODULE}.discover_runtime_local_ingress_routes", slow_discovery
    )
    port = RuntimeDelegationDispatchPort(_FakeRuntimeBus(), package_names=("fakepkg",))

    ticks: list[float] = []
    done = asyncio.Event()

    async def heartbeat() -> None:
        ticks.append(time.monotonic())
        while not done.is_set():
            await asyncio.sleep(0.005)
            ticks.append(time.monotonic())

    beat = asyncio.create_task(heartbeat())
    await asyncio.sleep(0)
    try:
        await port._resolved_routes()
    finally:
        done.set()
        await beat

    largest_gap = max(later - earlier for earlier, later in itertools.pairwise(ticks))
    assert largest_gap < 0.2


def _edit_contract(package_root: Path) -> str:
    contract_path = package_root / "nodes" / "node_demo" / "contract.yaml"
    contract_path.write_text(
        _DEMO_CONTRACT.format(
            name="demo", command_topic="onex.cmd.demo.start-renamed.v1"
        ),
        encoding="utf-8",
    )
    return "edited"


def _add_contract(package_root: Path) -> str:
    _write_node(
        package_root,
        "node_extra",
        _DEMO_CONTRACT.format(name="extra", command_topic="onex.cmd.extra.start.v1"),
    )
    return "added"


def _remove_contract(package_root: Path) -> str:
    shutil.rmtree(package_root / "nodes" / "node_other")
    return "removed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [_edit_contract, _add_contract, _remove_contract],
    ids=["edited", "added", "removed"],
)
async def test_changed_contract_set_is_resolved_on_the_next_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    read_counter: _ContractReadCounter,
    change: Callable[[Path], str],
) -> None:
    package_root = _fake_package(tmp_path, monkeypatch, "fakepkg")
    _write_node(
        package_root,
        "node_demo",
        _DEMO_CONTRACT.format(name="demo", command_topic="onex.cmd.demo.start.v1"),
    )
    _write_node(
        package_root,
        "node_other",
        _DEMO_CONTRACT.format(name="other", command_topic="onex.cmd.other.start.v1"),
    )
    port = RuntimeDelegationDispatchPort(_FakeRuntimeBus(), package_names=("fakepkg",))
    before = await port._resolved_routes()
    assert before["demo"].command_topic == "onex.cmd.demo.start.v1"
    assert "other" in before
    assert "extra" not in before
    reads_before = read_counter.count

    kind = change(package_root)
    after = await port._resolved_routes()

    assert read_counter.count > reads_before
    if kind == "edited":
        assert after["demo"].command_topic == "onex.cmd.demo.start-renamed.v1"
    elif kind == "added":
        assert after["extra"].command_topic == "onex.cmd.extra.start.v1"
        assert "other" in after
    else:
        assert "other" not in after
        assert "node_other" not in after
        assert after["demo"].command_topic == "onex.cmd.demo.start.v1"


@pytest.mark.asyncio
async def test_same_size_edit_with_new_mtime_is_resolved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = _fake_package(tmp_path, monkeypatch, "fakepkg")
    contract_path = _write_node(
        package_root,
        "node_demo",
        _DEMO_CONTRACT.format(name="demo", command_topic="onex.cmd.demo.aaaaa.v1"),
    )
    pinned_ns = 1_700_000_000_000_000_000
    os.utime(contract_path, ns=(pinned_ns, pinned_ns))
    port = RuntimeDelegationDispatchPort(_FakeRuntimeBus(), package_names=("fakepkg",))
    assert (await port._resolved_routes())[
        "demo"
    ].command_topic == "onex.cmd.demo.aaaaa.v1"

    contract_path.write_text(
        _DEMO_CONTRACT.format(name="demo", command_topic="onex.cmd.demo.bbbbb.v1"),
        encoding="utf-8",
    )
    os.utime(contract_path, ns=(pinned_ns + 1_000, pinned_ns + 1_000))

    assert (await port._resolved_routes())[
        "demo"
    ].command_topic == "onex.cmd.demo.bbbbb.v1"


@pytest.mark.asyncio
async def test_concurrent_resolutions_read_each_contract_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    read_counter: _ContractReadCounter,
) -> None:
    package_root = _fake_package(tmp_path, monkeypatch, "fakepkg")
    for node, name in (("node_demo", "demo"), ("node_other", "other")):
        _write_node(
            package_root,
            node,
            _DEMO_CONTRACT.format(name=name, command_topic=f"onex.cmd.{name}.go.v1"),
        )
    port = RuntimeDelegationDispatchPort(_FakeRuntimeBus(), package_names=("fakepkg",))

    results = await asyncio.gather(*(port._resolved_routes() for _ in range(10)))

    assert sorted(read_counter.paths) == sorted(
        str(package_root / "nodes" / node / "contract.yaml")
        for node in ("node_demo", "node_other")
    )
    assert all(result == results[0] for result in results)


@pytest.mark.asyncio
async def test_constructor_routes_skip_discovery_and_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("constructor routes must not touch installed contracts")

    monkeypatch.setattr(f"{_PORT_MODULE}.discover_runtime_local_ingress_routes", refuse)
    monkeypatch.setattr(f"{_PORT_MODULE}.local_ingress_contract_fingerprint", refuse)
    route = _demo_route("onex.cmd.demo.start.v1")
    port = RuntimeDelegationDispatchPort(_FakeRuntimeBus(), routes={"demo": route})

    first = await port._resolved_routes()
    second = await port._resolved_routes()

    assert first == second == {"demo": route}
    assert first is not second


@pytest.mark.asyncio
async def test_fingerprint_failure_resolves_uncached_every_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unimportable(package_names: Sequence[str]) -> tuple[tuple[str, int, int]]:
        raise ModuleNotFoundError("No module named 'fakepkg'")

    answers: list[dict[str, ModelRuntimeLocalIngressRoute] | Exception] = [
        {"demo": _demo_route("onex.cmd.demo.one.v1")},
        {"demo": _demo_route("onex.cmd.demo.two.v1")},
        ModuleNotFoundError("No module named 'fakepkg'"),
    ]
    calls: list[tuple[str, ...]] = []

    def discovery(
        package_names: Sequence[str],
    ) -> dict[str, ModelRuntimeLocalIngressRoute]:
        calls.append(tuple(package_names))
        answer = answers[len(calls) - 1]
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(
        f"{_PORT_MODULE}.local_ingress_contract_fingerprint", unimportable
    )
    monkeypatch.setattr(
        f"{_PORT_MODULE}.discover_runtime_local_ingress_routes", discovery
    )
    port = RuntimeDelegationDispatchPort(_FakeRuntimeBus(), package_names=("fakepkg",))

    first = await port._resolved_routes()
    second = await port._resolved_routes()
    with pytest.raises(ModuleNotFoundError, match="fakepkg"):
        await port._resolved_routes()

    assert first["demo"].command_topic == "onex.cmd.demo.one.v1"
    assert second["demo"].command_topic == "onex.cmd.demo.two.v1"
    assert calls == [("fakepkg",)] * 3
