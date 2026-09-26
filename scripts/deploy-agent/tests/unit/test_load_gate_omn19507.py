# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19507: the .200 deploy lane defers a job while the host's model servers need it.

Operator RULING, omni_home ledger 2026-09-25T12:40:36Z (items b and c): add a
deploy lane on .200 with a load gate, and leave .101 out of the deploy pool.

* the gate itself (``deploy_agent/load_gate.py``): a pure decision over one
  reading of the host, with thresholds declared per instance in the routing
  table, and an unreadable reading that defers rather than guesses;
* the consumer: a command for this instance is not accepted while the gate is
  shut. Nothing is committed and nothing is published; the record goes back
  onto the fetch path and the partition is paused until a re-check opens the
  gate. A command routed to another instance is still skipped as before;
* the agent: an idle converge waits for the gate the same way;
* the routing table: .101 (hostname ``stickybeatz``) is an excluded host that
  no instance may name and no agent may run as.

Every test name carries ``load_gate`` so ``pytest -k load_gate`` selects them.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
import yaml
from deploy_agent import load_gate as load_gate_mod
from deploy_agent.consumer import DeployConsumer
from deploy_agent.events import EnumRuntimeLane, ModelRebuildRequested
from deploy_agent.load_gate import (
    EnumLoadGateVerdict,
    LoadGate,
    ModelHostLoadReading,
    ModelLoadGateThresholds,
    decide,
    load_gate_for_instance,
    parse_gpu_in_use_bytes,
    parse_load_gates,
    parse_memory_free_percent,
)
from deploy_agent.routing import (
    ROUTING_TABLE_RELPATH,
    RoutingTableError,
    parse_routing_table,
    resolve_instance,
)

pytestmark = pytest.mark.unit

SHA = "b" * 40
GIB = 1024**3
REPO_ROOT = Path(load_gate_mod.__file__).resolve().parents[3]

THRESHOLDS = ModelLoadGateThresholds(
    max_load1_per_cpu=2.0,
    build_reserve_gib=24,
    model_reserve_gib=70,
    min_headroom_gib=16,
    recheck_seconds=60,
)


def _reading(**overrides: object) -> ModelHostLoadReading:
    values: dict[str, object] = {
        "load1": 24.0,
        "cpu_count": 24,
        "memory_total_gib": 192.0,
        "memory_free_percent": 50.0,
        "gpu_in_use_gib": 68.0,
    }
    values.update(overrides)
    return ModelHostLoadReading.model_validate(values)


# --- the decision -------------------------------------------------------------


def test_load_gate_opens_on_a_quiet_host_with_the_model_resident() -> None:
    decision = decide(THRESHOLDS, _reading())

    assert decision.verdict is EnumLoadGateVerdict.OPEN
    assert decision.reasons == ()
    # 96 GiB free, less the build's 24, less the 2 GiB the model has yet to map.
    assert decision.headroom_gib == pytest.approx(70.0)


def test_load_gate_defers_when_cpu_saturation_is_over_the_threshold() -> None:
    decision = decide(THRESHOLDS, _reading(load1=49.0))

    assert decision.verdict is EnumLoadGateVerdict.DEFER
    assert any("load1 49.00 over 24 cpu" in r for r in decision.reasons)


def test_load_gate_charges_the_model_reserve_when_the_model_is_not_resident() -> None:
    # 45% of 192 GiB is 86.4 GiB free. With the model resident that leaves
    # 86.4 - 24 - 2 = 60.4; with it paged out, the model's own 70 GiB comes
    # back the moment a request arrives, so the headroom is 86.4 - 24 - 70.
    resident = decide(THRESHOLDS, _reading(memory_free_percent=45.0))
    paged_out = decide(
        THRESHOLDS, _reading(memory_free_percent=45.0, gpu_in_use_gib=3.5)
    )

    assert resident.verdict is EnumLoadGateVerdict.OPEN
    assert paged_out.verdict is EnumLoadGateVerdict.DEFER
    assert paged_out.headroom_gib == pytest.approx(86.4 - 24 - 66.5)
    assert any("headroom" in r for r in paged_out.reasons)


def test_load_gate_defers_on_low_memory_even_with_the_model_resident() -> None:
    decision = decide(THRESHOLDS, _reading(memory_free_percent=20.0))

    assert decision.verdict is EnumLoadGateVerdict.DEFER
    assert decision.headroom_gib == pytest.approx(38.4 - 24 - 2)


@pytest.mark.parametrize(
    "field",
    ["load1", "cpu_count", "memory_total_gib", "memory_free_percent", "gpu_in_use_gib"],
)
def test_load_gate_an_unreadable_input_defers_and_names_itself(field: str) -> None:
    decision = decide(THRESHOLDS, _reading(**{field: None}))

    assert decision.verdict is EnumLoadGateVerdict.DEFER
    assert any(field in r and "unreadable" in r for r in decision.reasons)


def test_load_gate_describe_is_one_line_with_every_reading() -> None:
    line = decide(THRESHOLDS, _reading(load1=49.0)).describe()

    assert "\n" not in line
    for token in ("defer", "load1 49.00", "free 50%", "gpu in use 68.0 GiB"):
        assert token in line


# --- the host readers -----------------------------------------------------------


def test_load_gate_parses_the_macos_memory_pressure_line() -> None:
    out = (
        "The system has 206158430208 (12582912 pages with a page size of 16384).\n"
        "System-wide memory free percentage: 49%\n"
    )
    assert parse_memory_free_percent(out) == 49.0
    assert parse_memory_free_percent("no such line") is None


def test_load_gate_parses_the_ioreg_accelerator_in_use_bytes() -> None:
    out = (
        '  | "PerformanceStatistics" = {"In use system memory (driver)"=0,'
        '"Alloc system memory"=83515867136,"In use system memory"=72058109952,'
        '"Device Utilization %"=95}\n'
    )
    assert parse_gpu_in_use_bytes(out) == 72058109952
    assert parse_gpu_in_use_bytes('"Alloc system memory"=1') is None


def test_load_gate_the_probe_never_raises_and_records_failures_as_none() -> None:
    def boom() -> object:
        raise OSError("no such binary")

    reading = load_gate_mod.probe_host_load(
        read_loadavg=boom,  # type: ignore[arg-type]
        read_cpu_count=boom,  # type: ignore[arg-type]
        read_memory_total_bytes=boom,  # type: ignore[arg-type]
        read_memory_free_percent=boom,  # type: ignore[arg-type]
        read_gpu_in_use_bytes=boom,  # type: ignore[arg-type]
    )

    assert reading == ModelHostLoadReading(
        load1=None,
        cpu_count=None,
        memory_total_gib=None,
        memory_free_percent=None,
        gpu_in_use_gib=None,
    )


def test_load_gate_rechecks_no_more_often_than_declared() -> None:
    clock = SimpleNamespace(t=1000.0)
    reads: list[int] = []

    def read() -> ModelHostLoadReading:
        reads.append(1)
        return _reading(load1=49.0)

    gate = LoadGate(THRESHOLDS, read=read, clock=lambda: clock.t)
    assert gate.recheck_due()
    assert gate.check().verdict is EnumLoadGateVerdict.DEFER
    clock.t += 59
    assert not gate.recheck_due()
    clock.t += 1
    assert gate.recheck_due()
    assert len(reads) == 1


# --- the table ------------------------------------------------------------------


def _table(extra_dev_200: str = "", excluded: str = "") -> str:
    return (
        "default_instance: dev-201\n"
        f"{excluded}"
        "instances:\n"
        "  dev-201:\n"
        "    hostnames: [omninode-pc]\n"
        "    consumer_group: onex-deploy-agent\n"
        "  dev-200:\n"
        "    hostnames: [stickybeatz-studio]\n"
        "    consumer_group: onex-deploy-agent-dev-200\n"
        f"{extra_dev_200}"
        "routes: []\n"
    )


GATE_BLOCK = (
    "    load_gate:\n"
    "      max_load1_per_cpu: 2.0\n"
    "      build_reserve_gib: 24\n"
    "      model_reserve_gib: 70\n"
    "      min_headroom_gib: 16\n"
    "      recheck_seconds: 60\n"
)


def test_load_gate_is_declared_per_instance_in_the_routing_table() -> None:
    gates = parse_load_gates(_table(GATE_BLOCK))

    assert gates == {"dev-200": THRESHOLDS}, "an instance without a block has no gate"


@pytest.mark.parametrize(
    "block",
    [
        GATE_BLOCK + "      surprise: 1\n",
        GATE_BLOCK.replace("2.0", "-1"),
        GATE_BLOCK.replace("      recheck_seconds: 60\n", "      recheck_seconds: 0\n"),
        "    load_gate: yes\n",
    ],
)
def test_load_gate_a_malformed_block_refuses(block: str) -> None:
    with pytest.raises(RoutingTableError, match="load_gate"):
        parse_load_gates(_table(block))


def test_load_gate_the_routing_parser_still_routes_a_table_with_a_gate() -> None:
    table = parse_routing_table(_table(GATE_BLOCK))
    assert table.default_instance == "dev-201"


def test_load_gate_the_committed_table_gates_dev_200_and_nothing_else() -> None:
    text = (REPO_ROOT / ROUTING_TABLE_RELPATH).read_text(encoding="utf-8")
    gates = parse_load_gates(text)

    assert set(gates) == {"dev-200"}, "the .201 and .202 agents are unchanged"
    assert load_gate_for_instance(REPO_ROOT, "dev-200") == gates["dev-200"]
    assert load_gate_for_instance(REPO_ROOT, "dev-201") is None


# --- .101 is out of the pool ----------------------------------------------------


EXCLUDED = (
    "excluded_hosts:\n"
    "  stickybeatz: .101, out of the deploy pool (ledger RULING 2026-09-25T12:40:36Z)\n"
)


def test_load_gate_an_instance_may_not_name_an_excluded_host() -> None:
    text = _table(excluded=EXCLUDED).replace(
        "[stickybeatz-studio]", "[stickybeatz-studio, stickybeatz]"
    )
    with pytest.raises(RoutingTableError, match="excluded"):
        parse_routing_table(text)


def test_load_gate_an_agent_on_an_excluded_host_refuses_to_start() -> None:
    table = parse_routing_table(_table(excluded=EXCLUDED))

    with pytest.raises(RoutingTableError, match="excluded"):
        resolve_instance(table, env={}, hostname="Stickybeatz.local")
    with pytest.raises(RoutingTableError, match="excluded"):
        resolve_instance(
            table, env={"DEPLOY_AGENT_INSTANCE": "dev-200"}, hostname="Stickybeatz"
        )
    # The .200 host's first label differs and still resolves.
    assert (
        resolve_instance(table, env={}, hostname="Stickybeatz-Studio.local").name
        == "dev-200"
    )


def test_load_gate_the_committed_table_excludes_101_and_cites_the_ruling() -> None:
    text = (REPO_ROOT / ROUTING_TABLE_RELPATH).read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    table = parse_routing_table(text)

    assert "stickybeatz" in table.excluded_hosts
    assert "ledger:3893" in raw["excluded_hosts"]["stickybeatz"]
    for instance in table.instances.values():
        assert "stickybeatz" not in instance.hostnames


# --- the consumer ---------------------------------------------------------------


def _command(requested_by: str = "gha/omnimarket/pr-2890") -> ModelRebuildRequested:
    return ModelRebuildRequested.model_validate(
        {
            "correlation_id": str(uuid4()),
            "requested_by": requested_by,
            "scope": "full",
            "runtime_lane": "dev",
            "build_source": "workspace",
            "git_ref": SHA,
        }
    )


def _message(cmd: ModelRebuildRequested) -> SimpleNamespace:
    payload = cmd.model_dump(mode="json") | {"_signature": "a" * 64}
    return SimpleNamespace(
        value=payload, topic="t", partition=0, offset=100, key=None, timestamp=None
    )


class _Gate:
    def __init__(self, verdict: EnumLoadGateVerdict, due: bool = True) -> None:
        self.verdict = verdict
        self.due = due
        self.checks = 0

    def check(self) -> object:
        self.checks += 1
        return decide(
            THRESHOLDS,
            _reading(load1=49.0 if self.verdict is EnumLoadGateVerdict.DEFER else 1.0),
        )

    def recheck_due(self) -> bool:
        return self.due


def _consumer(gate: object | None) -> DeployConsumer:
    consumer = DeployConsumer.__new__(DeployConsumer)
    consumer.consumer = Mock()
    consumer.consumer.poll.return_value = {}
    consumer.job_store = Mock()
    consumer.job_store.has_active_job.return_value = False
    consumer.job_store.is_duplicate.return_value = False
    consumer.allowed_lanes = frozenset({EnumRuntimeLane.DEV})
    consumer.self_update_hook = lambda rewind: None
    consumer.notices = []  # type: ignore[attr-defined]
    consumer.on_rejected = consumer.notices.append  # type: ignore[attr-defined]
    consumer.load_gate = gate  # type: ignore[assignment]
    return consumer


def _process(consumer: DeployConsumer, cmd: ModelRebuildRequested) -> tuple:
    with patch("deploy_agent.consumer.verify_command", return_value=True):
        return consumer._process_message(_message(cmd))


def test_load_gate_consumer_defers_without_committing_or_publishing() -> None:
    consumer = _consumer(_Gate(EnumLoadGateVerdict.DEFER))

    assert _process(consumer, _command()) == (None, None)

    consumer.job_store.accept.assert_not_called()
    consumer.consumer.commit.assert_not_called()
    assert consumer.notices == []  # type: ignore[attr-defined]
    (tp, offset), _ = consumer.consumer.seek.call_args
    assert (tp.topic, tp.partition, offset) == ("t", 0, 100), "back onto the fetch path"
    consumer.consumer.pause.assert_called_once_with(tp)
    assert consumer.load_gate_paused == tp


def test_load_gate_consumer_accepts_when_the_gate_is_open() -> None:
    cmd = _command()
    consumer = _consumer(_Gate(EnumLoadGateVerdict.OPEN))

    assert _process(consumer, cmd) == (cmd, None)
    consumer.consumer.pause.assert_not_called()


def test_load_gate_consumer_still_skips_a_command_routed_elsewhere() -> None:
    gate = _Gate(EnumLoadGateVerdict.DEFER)
    consumer = _consumer(gate)
    router = Mock()
    router.is_mine.return_value = (
        False,
        SimpleNamespace(instance="dev-202", basis="t"),
    )
    router.instance.name = "dev-200"
    consumer.router = router

    assert _process(consumer, _command()) == (None, None)

    assert gate.checks == 0, "the gate is read only for this instance's own commands"
    consumer.consumer.commit.assert_called_once()
    consumer.consumer.pause.assert_not_called()


def test_load_gate_consumer_resumes_the_partition_once_a_recheck_opens() -> None:
    gate = _Gate(EnumLoadGateVerdict.DEFER)
    consumer = _consumer(gate)
    _process(consumer, _command())
    paused = consumer.load_gate_paused

    gate.due = False
    consumer.poll_and_accept()
    consumer.consumer.resume.assert_not_called()

    gate.due = True
    consumer.poll_and_accept()
    consumer.consumer.resume.assert_not_called()  # still shut

    gate.verdict = EnumLoadGateVerdict.OPEN
    consumer.poll_and_accept()
    consumer.consumer.resume.assert_called_once_with(paused)
    assert consumer.load_gate_paused is None


def test_load_gate_consumer_without_a_gate_is_unchanged() -> None:
    cmd = _command()
    assert _process(_consumer(None), cmd) == (cmd, None)
    bare = DeployConsumer.__new__(DeployConsumer)
    assert bare.load_gate is None, "declared on the class, off by default"
    assert bare.load_gate_paused is None


# --- the agent: an idle converge waits for the gate -----------------------------


def test_load_gate_idle_converge_waits_while_the_gate_is_shut() -> None:
    from deploy_agent import agent as agent_mod

    agent = agent_mod.DeployAgent.__new__(agent_mod.DeployAgent)
    agent._load_gate = _Gate(EnumLoadGateVerdict.DEFER)  # type: ignore[assignment]
    agent._idle_converge_attempted = set()
    agent._load_gate_last_verdict = None

    assert agent._idle_converge_gate_open() is False
    assert agent._idle_converge_attempted == set(), "the head is retried later"

    agent._load_gate = _Gate(EnumLoadGateVerdict.OPEN)  # type: ignore[assignment]
    assert agent._idle_converge_gate_open() is True

    agent._load_gate = None
    assert agent._idle_converge_gate_open() is True, "no gate, no wait"
