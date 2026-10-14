# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21041 -- one runner-fleet-maintain-start reaches runner-fleet-maintain-completed.

The workflow integration test next door calls the four handlers directly with
keyword arguments, so it never exercised what the runtime does: bind the
contract-declared event model, invoke ``handle`` with ONE validated payload, and
publish each returned model to the topic the contract declares for it. Two hops
were dropped on the dev lane because of that gap (deployed image read
2026-10-12: ``HandlerRunnerFleetMaintainStart.handle(self, correlation_id: UUID)``
and ``HandlerRunnerFleetSnapshot.handle(self, correlation_id: UUID)``):

* start -> snapshot-gather: the runtime hands the start handler the whole
  ``ModelRunnerFleetMaintainStartCommand``; pydantic rejects it as a ``UUID``.
* snapshot -> health-evaluate: the evaluate command has no ``published_events``
  entry, so it fell back to ``event_bus.publish_topics[0]`` (the snapshot-gather
  topic), where the snapshot effect rejects it.

These tests load the three REAL contracts from disk, build each handler's REAL
dispatch callback (``_make_dispatch_callback``) and each contract's REAL result
applier (``build_contract_result_applier``, fed the way ``handler_wiring`` feeds
it), and carry one start command along the topics until the completed event.
Only the ``gh`` / ``ssh`` subprocess boundary is mocked, and every call that
boundary receives is recorded for the read-only assertions.

A handler that rejects its payload is reported as a failed ASSERTION, not left as
a bare exception: the canonical-shape seam gate counts a parity test as
discriminating only when it fails on its own assertion against the pre-change tree.

The fixed-UUID inputs below (``_START_INPUT``, ``_SNAPSHOT_SCENARIOS``) are the
corpus the adequacy receipts and hand-flip proofs under
``scripts/ci/adequacy_receipts/`` bind by ``input_hash``.
"""

from __future__ import annotations

import importlib
import json
import re
import shlex
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ValidationError

import omnibase_infra.nodes as _nodes_pkg
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.nodes.node_runner_fleet_health_compute.models.enum_recommended_action_type import (
    EnumRecommendedActionType,
)
from omnibase_infra.nodes.node_runner_fleet_maintain_orchestrator.models.model_runner_fleet_maintain_completed_event import (
    ModelRunnerFleetMaintainCompletedEvent,
)
from omnibase_infra.nodes.node_runner_fleet_maintain_orchestrator.models.model_runner_fleet_maintain_start_command import (
    ModelRunnerFleetMaintainStartCommand,
)
from omnibase_infra.nodes.node_runner_health_snapshot_effect.handlers.handler_runner_fleet_snapshot import (
    HandlerRunnerFleetSnapshot,
)
from omnibase_infra.nodes.node_runner_health_snapshot_effect.models.model_runner_fleet_snapshot import (
    ModelRunnerFleetSnapshot,
)
from omnibase_infra.nodes.node_runner_health_snapshot_effect.models.model_runner_fleet_snapshot_gather_command import (
    ModelRunnerFleetSnapshotGatherCommand,
)
from omnibase_infra.observability.runner_health.model_runner_fleet_config import (
    ModelRunnerFleetConfig,
)
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    ProtocolHandleable,
    _make_dispatch_callback,
    _select_dispatch_result_output_topic,
)
from omnibase_infra.runtime.auto_wiring.models import ModelDiscoveredContract
from omnibase_infra.runtime.event_bus_subcontract_wiring import (
    load_published_events_map,
)
from omnibase_infra.runtime.service_dispatch_result_applier import (
    DispatchResultApplier,
    build_contract_result_applier,
)

pytestmark = [pytest.mark.unit]

_NODES_DIR = Path(_nodes_pkg.__file__).parent
_CONTRACT_PATHS = [
    _NODES_DIR / "node_runner_fleet_maintain_orchestrator" / "contract.yaml",
    _NODES_DIR / "node_runner_health_snapshot_effect" / "contract.yaml",
    _NODES_DIR / "node_runner_fleet_health_compute" / "contract.yaml",
]

_START_TOPIC = "onex.cmd.omnibase-infra.runner-fleet-maintain-start.v1"
_GATHER_TOPIC = "onex.cmd.omnibase-infra.runner-fleet-snapshot-gather.v1"
_SNAPSHOT_TOPIC = "onex.evt.omnibase-infra.runner-health-snapshot.v1"
_EVALUATE_TOPIC = "onex.cmd.omnibase-infra.runner-fleet-health-evaluate.v1"
_VERDICT_TOPIC = "onex.evt.omnibase-infra.runner-fleet-health-verdict.v1"
_COMPLETED_TOPIC = "onex.evt.omnibase-infra.runner-fleet-maintain-completed.v1"

# Output model class -> the topic its contract declares for it.
_EXPECTED_TOPIC_BY_OUTPUT_CLASS = {
    "ModelRunnerFleetSnapshotGatherCommand": _GATHER_TOPIC,
    "ModelRunnerFleetSnapshot": _SNAPSHOT_TOPIC,
    "ModelRunnerFleetHealthEvaluateCommand": _EVALUATE_TOPIC,
    "ModelRunnerFleetHealthVerdict": _VERDICT_TOPIC,
    "ModelRunnerFleetMaintainCompletedEvent": _COMPLETED_TOPIC,
}

_SMALLEST_SCOPE_REPO = "OmniNode-ai/omnibase_infra"
_LONG_AGO = "2020-01-01T00:00:00Z"

# Inputs bound by the adequacy receipts / hand-flip proofs (fixed, so they hash stably).
_START_INPUT = ModelRunnerFleetMaintainStartCommand(
    correlation_id=UUID("00000000-0000-4000-8000-000000021041")
)
_SNAPSHOT_SCENARIO_IDS = {
    "healthy": UUID("00000000-0000-4000-8000-0000000210a1"),
    "crash_loop_orphan_listener": UUID("00000000-0000-4000-8000-0000000210a2"),
    "probes_unreachable": UUID("00000000-0000-4000-8000-0000000210a3"),
    "wedged_queue_codeload": UUID("00000000-0000-4000-8000-0000000210a4"),
}
_SNAPSHOT_SCENARIOS = {
    name: ModelRunnerFleetSnapshotGatherCommand(correlation_id=cid)
    for name, cid in _SNAPSHOT_SCENARIO_IDS.items()
}


def _smallest_scope_config() -> ModelRunnerFleetConfig:
    """Lab-proof scope: two runners, one watched repo, one codeload scan."""
    return ModelRunnerFleetConfig(
        version="1.0",
        github_org="OmniNode-ai",
        runner_host="runner-host.invalid",
        runner_group="omnibase-ci",
        runner_name_prefix="omninode-runner",
        expected_count=2,
        watch_repos=(_SMALLEST_SCOPE_REPO,),
        codeload_scan_limit=1,
    )


# --------------------------------------------------------------------------- #
# The mocked gh / ssh boundary, recording every call
# --------------------------------------------------------------------------- #


class _FakeProc:
    def __init__(
        self, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr


@dataclass
class _Boundary:
    """Answers gh/ssh probes from a fixed fleet and records each argv."""

    github_runners: list[dict[str, object]]
    docker_lines: list[str]
    unreachable: bool = False
    queue_lines: list[dict[str, object]] = field(default_factory=list)
    failed_runs: list[dict[str, object]] = field(default_factory=list)
    failed_log: str = ""
    calls: list[tuple[str, ...]] = field(default_factory=list)

    async def __call__(self, *args: object, **_kwargs: object) -> _FakeProc:
        argv = tuple(str(a) for a in args)
        self.calls.append(argv)
        if self.unreachable:
            return _FakeProc(stderr=b"connection refused", returncode=255)
        if argv[:2] == ("gh", "api") and "/actions/runners" in argv[2]:
            lines = "\n".join(json.dumps(r) for r in self.github_runners)
            return _FakeProc(stdout=lines.encode())
        if argv[:2] == ("gh", "api"):
            return _FakeProc(
                stdout="\n".join(json.dumps(r) for r in self.queue_lines).encode()
            )
        if argv[:3] == ("gh", "run", "list"):
            return _FakeProc(stdout=json.dumps(self.failed_runs).encode())
        if argv[:3] == ("gh", "run", "view"):
            return _FakeProc(stdout=self.failed_log.encode())
        if argv[0] == "ssh" and "buildx" in argv[2]:
            return _FakeProc(stdout=b"OK\n")
        if argv[0] == "ssh" and argv[2].startswith("{ df -P"):
            return _FakeProc(stdout=b"41\n")
        if argv[0] == "ssh":
            return _FakeProc(stdout="\n".join(self.docker_lines).encode())
        raise AssertionError(f"Unexpected subprocess call: {argv}")


def _boundary(*, crash_looping: bool = False) -> _Boundary:
    restart_count = 9 if crash_looping else 0
    return _Boundary(
        github_runners=[
            {"name": "omninode-runner-1", "status": "online", "busy": False},
            {"name": "omninode-runner-2", "status": "online", "busy": False},
        ],
        docker_lines=[
            f"omninode-runner-1\trunning\t{restart_count}\tUp 2 days\t20\thealthy\t1\t0",
            "omninode-runner-2\trunning\t0\tUp 2 days\t20\thealthy\t1\t0",
        ],
    )


def _scenario_boundary(name: str) -> _Boundary:
    boundary = _boundary()
    if name == "crash_loop_orphan_listener":
        boundary.docker_lines = [
            "omninode-runner-1\trunning\t9\tUp 2 days\t5000\tunhealthy\t2\t1",
            "omninode-runner-2\trunning\t0\tUp 2 days\t20\thealthy\t1\t0",
        ]
        boundary.github_runners[0]["busy"] = True
    elif name == "probes_unreachable":
        boundary.unreachable = True
    elif name == "wedged_queue_codeload":
        boundary.queue_lines = [
            {"id": 7, "name": "CI", "status": "queued", "created_at": _LONG_AGO}
        ]
        boundary.failed_runs = [{"databaseId": 9, "displayTitle": "CI"}]
        boundary.failed_log = "fatal: GnuTLS recv error while fetching codeload"
    elif name != "healthy":
        raise AssertionError(f"unknown scenario {name!r}")
    return boundary


# --------------------------------------------------------------------------- #
# The in-process bus: real contracts, real callbacks, real result appliers
# --------------------------------------------------------------------------- #


@dataclass
class _Route:
    contract: ModelDiscoveredContract
    callback: Callable[
        [ModelEventEnvelope[object]], Awaitable[ModelDispatchResult | None]
    ]
    handler_name: str


@dataclass
class _Chain:
    """Routes keyed by subscribe topic, plus one applier per contract."""

    routes: dict[str, _Route]
    appliers: dict[str, DispatchResultApplier]
    buses: dict[str, AsyncMock]
    contracts: dict[str, ModelDiscoveredContract]


def _build_chain() -> _Chain:
    manifest = discover_contracts_from_paths(list(_CONTRACT_PATHS))
    assert not manifest.errors, f"contract discovery errors: {manifest.errors!r}"
    contracts = {c.name: c for c in manifest.contracts}
    assert set(contracts) == {
        "node_runner_fleet_maintain_orchestrator",
        "node_runner_health_snapshot_effect",
        "node_runner_fleet_health_compute",
    }

    routes: dict[str, _Route] = {}
    appliers: dict[str, DispatchResultApplier] = {}
    buses: dict[str, AsyncMock] = {}
    for contract in contracts.values():
        assert contract.event_bus is not None
        assert contract.handler_routing is not None
        # Fed exactly the way handler_wiring._subscribe_contract_topics feeds it.
        buses[contract.name] = AsyncMock(spec=ProtocolEventBusLike)
        appliers[contract.name] = build_contract_result_applier(
            event_bus=buses[contract.name],
            contract_path=contract.contract_path,
            publish_topics=contract.event_bus.publish_topics,
            output_topic=_select_dispatch_result_output_topic(contract),
            output_topic_map=load_published_events_map(contract.contract_path),
            allowed_output_topics=contract.event_bus.publish_topics,
        )
        for entry in contract.handler_routing.handlers:
            module = importlib.import_module(entry.handler.module)
            handler_cls = getattr(module, entry.handler.name)
            handler = cast(
                "ProtocolHandleable",
                HandlerRunnerFleetSnapshot(config=_smallest_scope_config())
                if entry.handler.name == "HandlerRunnerFleetSnapshot"
                else handler_cls(),
            )
            callback = _make_dispatch_callback(handler, entry.event_model)
            topics = (
                (entry.topic,)
                if entry.topic
                else tuple(contract.event_bus.subscribe_topics)
            )
            for topic in topics:
                assert topic not in routes, f"two handlers own {topic}"
                routes[topic] = _Route(contract, callback, entry.handler.name)
    return _Chain(routes=routes, appliers=appliers, buses=buses, contracts=contracts)


@dataclass
class _Hop:
    topic: str
    handler: str
    output_classes: list[str]
    output_topics: list[str]


async def _dispatch(
    route: _Route, payload: BaseModel, correlation_id: UUID
) -> tuple[ModelDispatchResult | None, str | None]:
    """Run one hop through its real dispatch callback.

    A handler that rejects its payload (pydantic ``ValidationError``, or a
    ``TypeError`` from a signature the runtime cannot bind) comes back as an error
    string for the caller to assert on, never as a bare exception.
    """
    envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
        payload=payload,
        correlation_id=correlation_id,
        event_type=type(payload).__name__,
    )
    try:
        return await route.callback(envelope), None
    except (ValidationError, TypeError) as exc:
        return None, f"{route.handler_name} rejected its payload: {exc}"


async def _drive(
    chain: _Chain, start: ModelRunnerFleetMaintainStartCommand
) -> tuple[list[_Hop], BaseModel | None, str]:
    """Publish ``start`` and follow every output to its topic.

    Returns ``(hops, completed_event_or_None, stop_reason)``.
    """
    pending: list[tuple[str, BaseModel]] = [(_START_TOPIC, start)]
    hops: list[_Hop] = []
    for _ in range(12):
        if not pending:
            return hops, None, "no message left on any topic"
        topic, payload = pending.pop(0)
        if topic == _COMPLETED_TOPIC:
            return hops, payload, "completed"
        route = chain.routes.get(topic)
        if route is None:
            return hops, None, f"no handler consumes {topic}"
        result, error = await _dispatch(route, payload, start.correlation_id)
        if error is not None:
            return hops, None, error
        if result is None or result.status is not EnumDispatchStatus.SUCCESS:
            return hops, None, f"{route.handler_name} dispatch status {result!r}"
        applier = chain.appliers[route.contract.name]
        bus = chain.buses[route.contract.name]
        bus.publish_envelope.reset_mock()
        await applier.apply(result, correlation_id=start.correlation_id)
        topics = [c.kwargs["topic"] for c in bus.publish_envelope.await_args_list]
        assert len(topics) == len(result.output_events)
        hops.append(
            _Hop(
                topic=topic,
                handler=route.handler_name,
                output_classes=[type(e).__name__ for e in result.output_events],
                output_topics=list(topics),
            )
        )
        pending.extend(zip(topics, result.output_events, strict=True))
    return hops, None, "hop budget exhausted"


async def _run_workflow(
    boundary: _Boundary,
) -> tuple[list[_Hop], BaseModel | None, str, UUID]:
    chain = _build_chain()
    start = ModelRunnerFleetMaintainStartCommand(correlation_id=uuid4())

    async def _spawn(*args: object, **kwargs: object) -> _FakeProc:
        return await boundary(*args, **kwargs)

    with patch("asyncio.create_subprocess_exec", AsyncMock(side_effect=_spawn)):
        hops, completed, reason = await _drive(chain, start)
    return hops, completed, reason, start.correlation_id


# --------------------------------------------------------------------------- #
# AC2 -- each hop is reachable and lands where its contract says
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_start_command_reaches_completed_event_with_health_verdict() -> None:
    """LOAD-BEARING: start -> gather -> snapshot -> evaluate -> verdict -> completed."""
    hops, completed, reason, cid = await _run_workflow(_boundary())

    assert completed is not None, (
        f"the workflow stopped before the completed event: {reason}; "
        f"hops so far: {[(h.handler, h.output_topics) for h in hops]}"
    )
    assert isinstance(completed, ModelRunnerFleetMaintainCompletedEvent)
    assert completed.correlation_id == cid
    assert completed.verdict.correlation_id == cid
    assert completed.verdict.expected_count == 2
    assert completed.verdict.online_count == 2
    assert [h.handler for h in hops] == [
        "HandlerRunnerFleetMaintainStart",
        "HandlerRunnerFleetSnapshot",
        "HandlerRunnerFleetSnapshotComplete",
        "HandlerRunnerFleetHealthEvaluate",
        "HandlerRunnerFleetHealthVerdictComplete",
    ]


@pytest.mark.asyncio
async def test_start_handler_hop_dispatches_through_the_real_callback() -> None:
    """Hop 1 alone: the start command in, the snapshot-gather command out."""
    chain = _build_chain()
    result, error = await _dispatch(
        chain.routes[_START_TOPIC], _START_INPUT, _START_INPUT.correlation_id
    )

    assert error is None, error
    assert result is not None
    assert result.status is EnumDispatchStatus.SUCCESS
    assert len(result.output_events) == 1
    gather = result.output_events[0]
    assert isinstance(gather, ModelRunnerFleetSnapshotGatherCommand)
    assert gather.correlation_id == _START_INPUT.correlation_id


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", sorted(_SNAPSHOT_SCENARIOS))
async def test_snapshot_effect_hop_dispatches_through_the_real_callback(
    scenario: str,
) -> None:
    """Hop 2 alone, per scenario: the gather command in, a facts-only snapshot out."""
    chain = _build_chain()
    command = _SNAPSHOT_SCENARIOS[scenario]
    boundary = _scenario_boundary(scenario)

    async def _spawn(*args: object, **kwargs: object) -> _FakeProc:
        return await boundary(*args, **kwargs)

    with patch("asyncio.create_subprocess_exec", AsyncMock(side_effect=_spawn)):
        result, error = await _dispatch(
            chain.routes[_GATHER_TOPIC], command, command.correlation_id
        )

    assert error is None, error
    assert result is not None
    assert result.status is EnumDispatchStatus.SUCCESS
    assert len(result.output_events) == 1
    snapshot = result.output_events[0]
    assert isinstance(snapshot, ModelRunnerFleetSnapshot)
    assert snapshot.correlation_id == command.correlation_id

    if scenario == "healthy":
        assert [r.name for r in snapshot.runners] == [
            "omninode-runner-1",
            "omninode-runner-2",
        ]
        assert snapshot.github_source_ok and snapshot.docker_source_ok
        assert snapshot.source_errors == ()
    elif scenario == "crash_loop_orphan_listener":
        first = snapshot.runners[0]
        assert first.docker_restart_count == 9
        assert first.listener_process_count == 2
        assert first.orphaned_listener_count == 1
        assert first.docker_health == "unhealthy"
    elif scenario == "probes_unreachable":
        # A probe that could not look reports unknown -- never a fabricated pass.
        assert snapshot.runners == ()
        assert not snapshot.github_source_ok
        assert not snapshot.docker_source_ok
        assert snapshot.host_disk_used_percent is None
        assert snapshot.buildx_available is None
        assert len(snapshot.source_errors) >= 4
    else:
        assert len(snapshot.zombie_run_candidates) == 1
        assert snapshot.zombie_run_candidates[0].run_id == 7
        assert snapshot.codeload_throttle_signal_count == 1


@pytest.mark.asyncio
async def test_every_output_lands_on_the_topic_its_contract_declares() -> None:
    """A returned model published to the wrong topic is a dropped hop.

    The evaluate command used to resolve to ``publish_topics[0]`` -- the
    snapshot-gather topic -- because no ``published_events`` entry named it.
    """
    hops, _completed, _reason, _cid = await _run_workflow(_boundary())

    seen: dict[str, str] = {}
    for hop in hops:
        for cls, topic in zip(hop.output_classes, hop.output_topics, strict=True):
            seen[cls] = topic
    assert seen == _EXPECTED_TOPIC_BY_OUTPUT_CLASS


def test_orchestrator_declares_a_terminal_event_for_onex_run_node() -> None:
    """``onex run-node`` reads a top-level ``terminal_event`` string and refuses without it.

    Read 2026-10-12: ``Node 'node_runner_fleet_maintain_orchestrator' contract ...
    does not declare a terminal_event``. It must be a publishable topic, so the
    applier's fallback topic is the completed event rather than the first command.
    """
    contract = _build_chain().contracts["node_runner_fleet_maintain_orchestrator"]
    assert contract.terminal_event == _COMPLETED_TOPIC
    assert contract.event_bus is not None
    assert _COMPLETED_TOPIC in contract.event_bus.publish_topics


# --------------------------------------------------------------------------- #
# AC3 -- read-only: recommended actions only, and no probe issues a write
# --------------------------------------------------------------------------- #

_GH_WRITE_FLAGS = {
    "-X",
    "--method",
    "-f",
    "-F",
    "--field",
    "--raw-field",
    "--input",
}
_GH_READ_VERBS = {("api",), ("run", "list"), ("run", "view")}
_DOCKER_READ_SUBCOMMANDS = {"ps", "inspect", "exec", "buildx"}
_REMOTE_WRITE_WORDS = re.compile(
    r"\b(rm|kill|pkill|killall|systemctl|service|reboot|shutdown|mv|cp|chmod|"
    r"chown|tee|dd|mkfs|sudo|apt|apt-get|pip|curl|wget|truncate)\b"
)
_REDIRECT_TO_FILE = re.compile(r">>?\s*(?!/dev/null|&\d)\S")


def _write_findings(call: tuple[str, ...]) -> list[str]:
    """Why ``call`` is not a read-only probe; empty when it is one."""
    findings: list[str] = []
    if call[0] == "gh":
        verb = call[1:3] if call[1] == "run" else call[1:2]
        if tuple(verb) not in _GH_READ_VERBS:
            findings.append(f"gh verb {' '.join(verb)!r} is not a read")
        findings.extend(
            f"gh write flag {arg!r}" for arg in call[1:] if arg in _GH_WRITE_FLAGS
        )
        if call[1] == "run" and call[2] == "view" and "--log-failed" not in call:
            findings.append("gh run view without --log-failed")
        return findings
    if call[0] == "ssh":
        remote = call[2]
        for sub in re.findall(r"\bdocker\s+(?:-\S+\s+)*([a-z]+)", remote):
            if sub not in _DOCKER_READ_SUBCOMMANDS:
                findings.append(f"docker {sub!r} is not an inspect")
        for match in re.finditer(r"\bdocker\s+buildx\s+([a-z]+)", remote):
            if match.group(1) != "inspect":
                findings.append(f"docker buildx {match.group(1)!r} is not an inspect")
        if _REMOTE_WRITE_WORDS.search(remote):
            findings.append(f"mutating word in remote command: {remote[:80]!r}")
        if _REDIRECT_TO_FILE.search(remote):
            findings.append("remote command redirects output to a file")
        return findings
    return [f"unexpected executable {call[0]!r}"]


@pytest.mark.parametrize(
    "call",
    [
        ("gh", "api", "-X", "POST", "/repos/o/r/actions/runs/1/cancel"),
        ("gh", "api", "/orgs/o/actions/runners", "-f", "x=1"),
        ("gh", "run", "rerun", "1", "--repo", "o/r"),
        ("gh", "run", "cancel", "1", "--repo", "o/r"),
        ("ssh", "h", "docker restart omninode-runner-1"),
        ("ssh", "h", "docker stop omninode-runner-1"),
        ("ssh", "h", "docker exec omninode-runner-1 sh -c 'kill 1'"),
        ("ssh", "h", "docker ps -a > /tmp/x"),
        ("ssh", "h", "docker ps -a >> /tmp/x"),
        ("ssh", "h", "docker ps -a 2>/tmp/x"),
        ("ssh", "h", "systemctl restart docker"),
        ("scp", "a", "b"),
    ],
)
def test_read_only_write_detector_flags_every_mutating_call(
    call: tuple[str, ...],
) -> None:
    """Positive control: a detector that flags nothing proves nothing."""
    assert _write_findings(call), f"{call!r} was not recognised as a write"


def _repos_queried(calls: list[tuple[str, ...]]) -> set[str]:
    repos: set[str] = set()
    for call in calls:
        for index, arg in enumerate(call):
            if arg.startswith("/repos/"):
                repos.add(arg.removeprefix("/repos/").split("/actions")[0])
            if arg == "--repo":
                repos.add(call[index + 1])
    return repos


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", sorted(_SNAPSHOT_SCENARIO_IDS))
async def test_read_only_no_snapshot_probe_issues_a_write_call(scenario: str) -> None:
    """Every gh call is a read; every ssh command is an inspect, ps, df or grep."""
    chain = _build_chain()
    boundary = _scenario_boundary(scenario)

    async def _spawn(*args: object, **kwargs: object) -> _FakeProc:
        return await boundary(*args, **kwargs)

    command = _SNAPSHOT_SCENARIOS[scenario]
    with patch("asyncio.create_subprocess_exec", AsyncMock(side_effect=_spawn)):
        result, error = await _dispatch(
            chain.routes[_GATHER_TOPIC], command, command.correlation_id
        )

    assert error is None, error
    assert result is not None
    assert boundary.calls, "the snapshot effect issued no probe at all"
    assert {c[0] for c in boundary.calls} == {"gh", "ssh"}
    for call in boundary.calls:
        assert not _write_findings(call), (
            f"{call!r} is not read-only: {_write_findings(call)}"
        )
    # The scope override took: only the one watched repo was ever queried.
    assert _repos_queried(boundary.calls) == {_SMALLEST_SCOPE_REPO}


@pytest.mark.asyncio
async def test_read_only_completed_event_carries_recommended_actions_only() -> None:
    """A crash-looping runner yields a RECOMMENDED restart; nothing performs it."""
    boundary = _boundary(crash_looping=True)
    _hops, completed, reason, _cid = await _run_workflow(boundary)

    assert isinstance(completed, ModelRunnerFleetMaintainCompletedEvent), reason
    actions = completed.verdict.recommended_actions
    assert any(
        a.action_type is EnumRecommendedActionType.RESTART_RUNNER
        and a.target_id == "omninode-runner-1"
        for a in actions
    ), f"expected a recorded restart recommendation, got {actions!r}"

    # The event has a report and a correlation id -- no field a mutation could fill.
    assert set(ModelRunnerFleetMaintainCompletedEvent.model_fields) == {
        "correlation_id",
        "verdict",
    }
    assert set(type(actions[0]).model_fields) == {
        "action_type",
        "target_id",
        "reason",
        "confidence",
    }
    # ... and the recommendation was not carried out: no container action ever left.
    assert boundary.calls
    for call in boundary.calls:
        assert not _write_findings(call), call
        assert not re.search(
            r"\bdocker\s+(restart|stop|kill|start|rm|pause|update)\b",
            shlex.join(call),
        ), f"{call!r} carried out a container action"


def test_read_only_asyncio_is_the_only_subprocess_seam() -> None:
    """The mock above is only sound if the effect shells out through asyncio alone."""
    source = (
        _NODES_DIR
        / "node_runner_health_snapshot_effect"
        / "handlers"
        / "handler_runner_fleet_snapshot.py"
    ).read_text(encoding="utf-8")
    assert "asyncio.create_subprocess_exec" in source
    for forbidden in ("subprocess.run", "subprocess.Popen", "os.system", "os.popen"):
        assert forbidden not in source
