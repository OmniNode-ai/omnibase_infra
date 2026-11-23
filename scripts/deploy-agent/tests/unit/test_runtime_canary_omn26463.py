# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26463 -- the runtime-only canary lane and the dev roll's receipt read.

AC1 ``canary_isolation``, AC2 ``canary_wakes_every_dormant``, AC3
``canary_receipt_first_failing_line``, AC4 ``roll_refuses_without_canary_pass``,
AC5 ``canary_incident_23837926``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from deploy_agent.executor import load_dev_boot_guard
from deploy_agent.instance_lanes import load_instance_lanes
from deploy_agent.routing import load_routing_table

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[4]
_FIXTURES = Path(__file__).parent / "fixtures" / "canary_omn26463"
_OVERLAY = _REPO / "docker" / "docker-compose.canary.yml"
_TABLE = _REPO / "config" / "deploy_lane_routing.yaml"
_CANARY = "canary-h051"

_HANDLER = "handler_runtime_canary"

INFRA = "a" * 40
MARKET = "b" * 12
CORE = "c" * 12
INFRA_PIN = INFRA
TAGS = {
    "omnimarket": f"dev-20261122165900-{MARKET}",
    "omnibase_core": f"dev-20261122165800-{CORE}",
}


def _gate() -> Any:
    """The canary gate, loaded by the boot guard the way the agent's roll loads it."""
    return load_dev_boot_guard()._canary_gate()


def _modules() -> tuple[Any, Any, Any, Any]:
    _gate()
    names = (
        "handlers.handler_runtime_canary",
        "models.model_runtime_canary_candidate",
        "models.model_runtime_canary_request",
        "models.model_runtime_canary_receipt",
    )
    first, second, third, fourth = (sys.modules[f"omnibase_infra.{n}"] for n in names)
    return first, second, third, fourth


class FakeProbe:
    """A canary container that answers from a script, on a clock that moves when slept on."""

    def __init__(
        self,
        dormant: tuple[str, ...],
        *,
        log: str = "",
        refuses_wake: frozenset[str] = frozenset(),
        heartbeat: bool = True,
        ready_after: float = 0.0,
        clock: list[float] | None = None,
    ) -> None:
        self._dormant = dormant
        self._log = log
        self._refuses = refuses_wake
        self._heartbeat = heartbeat
        self._ready_after = ready_after
        self.clock = clock if clock is not None else [0.0]
        self.woken: list[str] = []

    def ready(self) -> bool:
        return self.clock[0] >= self._ready_after

    def dormant_contracts(self) -> tuple[str, ...]:
        return self._dormant

    def wake(self, contract: str) -> bool:
        self.woken.append(contract)
        return contract not in self._refuses

    def heartbeat_ok(self) -> bool:
        return self._heartbeat

    def log_text(self) -> str:
        return self._log


def _run(probe: FakeProbe, tmp_path: Path, **bounds: float) -> tuple[Any, Any]:
    handler_mod, candidate_mod, request_mod, _ = _modules()

    def sleep(seconds: float) -> None:
        probe.clock[0] += seconds

    candidate = candidate_mod.ModelRuntimeCanaryCandidate(
        infra_ref=INFRA, omnimarket_ref=MARKET, core_ref=CORE, overlay_rev="d" * 40
    )
    request = request_mod.ModelRuntimeCanaryRequest(
        candidate=candidate, receipt_dir=tmp_path / "receipts", **bounds
    )
    receipt = handler_mod.judge_candidate(
        request, probe, clock=lambda: probe.clock[0], sleep=sleep
    )
    return receipt, request


# --- AC2 -------------------------------------------------------------------------


def test_canary_wakes_every_dormant_contract_once(tmp_path: Path) -> None:
    dormant = ("node_a", "node_b", "node_c")
    probe = FakeProbe(dormant)
    receipt, _ = _run(probe, tmp_path, heartbeat_hold_seconds=30, poll_seconds=5)
    assert receipt.verdict == "PASS"
    assert probe.woken == list(dormant), "one synthetic first request per contract"
    assert receipt.contracts_woken == dormant
    assert receipt.contracts_unwoken == ()
    assert receipt.heartbeat_seconds is not None and receipt.heartbeat_seconds >= 30


def test_canary_wakes_every_dormant_fails_when_one_is_left_unwoken(
    tmp_path: Path,
) -> None:
    probe = FakeProbe(("node_a", "node_b"), refuses_wake=frozenset({"node_b"}))
    receipt, _ = _run(probe, tmp_path, heartbeat_hold_seconds=30)
    assert receipt.verdict == "FAIL"
    assert receipt.failed_check == "wake"
    assert receipt.contracts_unwoken == ("node_b",)
    assert receipt.first_failing_line is not None
    assert "node_b" in receipt.first_failing_line


def test_canary_ready_budget_is_five_minutes_by_default(tmp_path: Path) -> None:
    probe = FakeProbe(("node_a",), ready_after=10_000.0)
    receipt, request = _run(probe, tmp_path)
    assert request.ready_budget_seconds == 300.0
    assert request.heartbeat_hold_seconds == 900.0
    assert receipt.verdict == "FAIL" and receipt.failed_check == "ready"
    assert receipt.ready_seconds is not None and receipt.ready_seconds >= 300.0
    assert receipt.first_failing_line == "/ready not ready within 300s"


def test_canary_heartbeat_unhealthy_fails_the_hold(tmp_path: Path) -> None:
    probe = FakeProbe(("node_a",), heartbeat=False)
    receipt, _ = _run(probe, tmp_path, heartbeat_hold_seconds=60)
    assert receipt.verdict == "FAIL" and receipt.failed_check == "heartbeat"
    assert "heartbeat unhealthy" in (receipt.first_failing_line or "")


# --- AC3 -------------------------------------------------------------------------


def test_canary_receipt_first_failing_line_is_written_and_required(
    tmp_path: Path,
) -> None:
    log = (
        "INFO ok\n"
        "2026-11-22 ERROR omnibase_infra.x something broke\n"
        "later ERROR a second one\n"
    )
    probe = FakeProbe(("node_a",), log=log)
    _, request = _run(probe, tmp_path, heartbeat_hold_seconds=30)
    written = json.loads(
        (request.receipt_dir / f"{request.candidate.key}.json").read_text()
    )
    assert written["verdict"] == "FAIL"
    assert written["first_failing_line"].endswith("something broke")
    assert written["ready_seconds"] is not None and written["wake_seconds"] is not None


def test_canary_receipt_first_failing_line_a_fail_without_one_is_refused() -> None:
    _, candidate_mod, _, receipt_mod = _modules()
    candidate = candidate_mod.ModelRuntimeCanaryCandidate(
        infra_ref=INFRA, omnimarket_ref=MARKET, core_ref=CORE
    )
    with pytest.raises(ValueError, match="first failing line"):
        receipt_mod.ModelRuntimeCanaryReceipt(
            verdict="FAIL",
            candidate=candidate,
            started_at="2026-11-22T17:00:00Z",
            finished_at="2026-11-22T17:00:01Z",
            first_failing_line="  ",
        )


# --- AC5 -------------------------------------------------------------------------


def test_canary_incident_23837926_fails_on_the_recorded_boot(tmp_path: Path) -> None:
    log = (_FIXTURES / "canary_incident_23837926_boot.txt").read_text()
    handler_mod, *_ = _modules()
    line = handler_mod.first_failing_line(log)
    assert line is not None
    assert line.startswith("TypeError")
    assert "consumer_group_handover" in line
    probe = FakeProbe(("node_schedule_run_reducer",), log=log, heartbeat=False)
    receipt, _ = _run(probe, tmp_path, heartbeat_hold_seconds=30)
    assert receipt.verdict == "FAIL"
    assert receipt.first_failing_line == line


def test_canary_incident_23837926_passes_on_the_fixed_boot(tmp_path: Path) -> None:
    log = (_FIXTURES / "canary_incident_23837926_fixed_boot.txt").read_text()
    probe = FakeProbe(("node_schedule_run_reducer",), log=log)
    receipt, _ = _run(probe, tmp_path, heartbeat_hold_seconds=30)
    assert receipt.verdict == "PASS" and receipt.first_failing_line is None


# --- AC4 -------------------------------------------------------------------------


def _write_receipt(tmp_path: Path, verdict: str) -> Path:
    probe = FakeProbe(("node_a",), log="ERROR boom\n" if verdict == "FAIL" else "")
    _, request = _run(probe, tmp_path, heartbeat_hold_seconds=30)
    return request.receipt_dir


def test_roll_refuses_without_canary_pass_no_receipt(tmp_path: Path) -> None:
    gate = _gate()
    with pytest.raises(gate.RuntimeCanaryRefusedError, match="no canary receipt"):
        gate.require_pass(
            tmp_path,
            git_ref=INFRA_PIN,
            sibling_tags=TAGS,
            overlay_ref="d" * 40,
        )


def test_roll_refuses_without_canary_pass_a_fail_receipt(tmp_path: Path) -> None:
    gate = _gate()
    directory = _write_receipt(tmp_path, "FAIL")
    with pytest.raises(gate.RuntimeCanaryRefusedError, match=r"FAIL: .*boom"):
        gate.require_pass(
            directory, git_ref=INFRA_PIN, sibling_tags=TAGS, overlay_ref="d" * 40
        )


def test_roll_admits_a_candidate_with_a_pass_receipt(tmp_path: Path) -> None:
    gate = _gate()
    directory = _write_receipt(tmp_path, "PASS")
    path = gate.require_pass(
        directory, git_ref=INFRA_PIN, sibling_tags=TAGS, overlay_ref="d" * 40
    )
    assert path.is_file()


def test_roll_refuses_without_canary_pass_a_different_candidate(
    tmp_path: Path,
) -> None:
    gate = _gate()
    directory = _write_receipt(tmp_path, "PASS")
    with pytest.raises(gate.RuntimeCanaryRefusedError, match="no canary receipt"):
        gate.require_pass(
            directory, git_ref="e" * 40, sibling_tags=TAGS, overlay_ref="d" * 40
        )


def test_roll_refuses_without_canary_pass_an_unpinned_candidate(tmp_path: Path) -> None:
    gate = _gate()
    with pytest.raises(gate.RuntimeCanaryRefusedError, match="not pinned"):
        gate.require_pass(
            tmp_path, git_ref="origin/dev", sibling_tags={}, overlay_ref=""
        )


def _roll_command() -> dict[str, object]:
    return {"git_ref": INFRA_PIN, "sibling_tags": TAGS, "overlay_ref": "d" * 40}


def test_roll_refuses_without_canary_pass_in_the_boot_guard_prepare(
    tmp_path: Path,
) -> None:
    """With the receipt directory present, the roll's prepare refuses before any build."""
    guard_mod = load_dev_boot_guard()
    (tmp_path / "canary-receipts").mkdir()
    with pytest.raises(RuntimeError, match="no canary receipt"):
        guard_mod._require_canary_receipt(_roll_command(), tmp_path)


def test_roll_admits_in_the_boot_guard_prepare_with_a_pass_receipt(
    tmp_path: Path,
) -> None:
    guard_mod = load_dev_boot_guard()
    directory = _write_receipt(tmp_path / "run", "PASS")
    state = tmp_path / "state"
    state.mkdir()
    directory.rename(state / "canary-receipts")
    guard_mod._require_canary_receipt(_roll_command(), state)
    other = {**_roll_command(), "git_ref": "e" * 40}
    with pytest.raises(RuntimeError, match="no canary receipt"):
        guard_mod._require_canary_receipt(other, state)


def test_the_receipt_gate_is_off_until_the_state_dir_holds_receipts(
    tmp_path: Path,
) -> None:
    """Turning it on for a lane is the operator's act, after receipts are delivered."""
    guard_mod = load_dev_boot_guard()
    guard_mod._require_canary_receipt(
        {"git_ref": "origin/dev", "sibling_tags": {}, "overlay_ref": ""}, tmp_path
    )


# --- AC1 -------------------------------------------------------------------------

_DEV_BROKER = ".".join(("192", "168", "86", "201"))
_DEV_GROUP = re.compile(r"onex-deploy-agent(?!-canary)")


def _overlay_lines() -> list[str]:
    return [
        line
        for line in _OVERLAY.read_text().splitlines()
        if not line.lstrip().startswith("#")
    ]


def canary_isolation_violations(
    *, overlay: list[str], row: str, consumer_group: str, containers: list[str]
) -> list[str]:
    """What makes the canary share anything with dev; empty when it shares nothing."""
    found: list[str] = []
    if _DEV_GROUP.match(consumer_group) or "canary" not in consumer_group:
        found.append(f"consumer group {consumer_group!r} is not the canary's own")
    for name in containers:
        if "canary" not in name:
            found.append(f"container {name!r} does not carry the canary name")
    for line in overlay:
        if _DEV_BROKER in line:
            found.append(f"overlay line names the dev broker: {line.strip()}")
        published = re.match(r'\s*- "(?:[\d.]+:)?(\d+):\d+"', line)
        if published and not 62000 <= int(published.group(1)) <= 62999:
            found.append(f"published port outside the canary block: {line.strip()}")
    for env in row.splitlines():
        if env.lstrip().startswith("#"):
            continue
        if _DEV_BROKER in env or "SASL_ENV_PREFIX=DEV_" in env:
            found.append(f"row carries a dev broker or credential: {env}")
        if "broker_container: omnibase-infra-redpanda" in env:
            found.append(f"row names the dev broker container: {env}")
    return found


def _canary_row_text() -> str:
    """The canary instance's block of the committed table, comments included."""
    text = _TABLE.read_text()
    start = text.index(f"\n  {_CANARY}:\n") + 1
    end = text.index("\nroutes:\n", start)
    return text[start:end]


def _canary_row() -> tuple[str, list[str]]:
    table = load_routing_table(_REPO)
    group = table.instances[_CANARY].consumer_group
    lane = load_instance_lanes(_REPO)[_CANARY]
    return group, [lane.runtime_container, lane.postgres_container]


def test_canary_isolation_the_committed_lane_shares_nothing_with_dev() -> None:
    group, containers = _canary_row()
    assert (
        canary_isolation_violations(
            overlay=_overlay_lines(),
            row=_canary_row_text(),
            consumer_group=group,
            containers=containers,
        )
        == []
    )


def test_canary_isolation_one_runtime_container_with_its_own_ports() -> None:
    lane = load_instance_lanes(_REPO)[_CANARY]
    assert lane.runtime_container == "omninode-canary-runtime"
    assert lane.health_ports.main == 62085 and lane.health_ports.effects == 62086
    assert lane.proves == ()
    doc = yaml.safe_load(_OVERLAY.read_text().replace("!override", ""))
    assert doc["name"] == lane.compose_project
    assert (
        doc["services"]["omninode-runtime"]["container_name"] == lane.runtime_container
    )


def test_canary_isolation_fails_on_a_dev_consumer_group_or_dev_dsn() -> None:
    group, containers = _canary_row()
    row = _canary_row_text()
    bad_group = canary_isolation_violations(
        overlay=_overlay_lines(),
        row=row,
        consumer_group="onex-deploy-agent",
        containers=containers,
    )
    assert bad_group and "consumer group" in bad_group[0]
    bad_dsn = canary_isolation_violations(
        overlay=[*_overlay_lines(), f"  KAFKA: {_DEV_BROKER}:19092"],
        row=row + f"\n    kafka_bootstrap: {_DEV_BROKER}:19092\n",
        consumer_group=group,
        containers=containers,
    )
    assert len(bad_dsn) == 2
