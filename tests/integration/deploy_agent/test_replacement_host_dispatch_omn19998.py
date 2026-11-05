# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exercise host replacement admission through the real in-memory transport."""

from __future__ import annotations

import json
import os
import subprocess
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

import pytest
import yaml
from deploy_agent import executor
from deploy_agent.events import EnumRuntimeLane

from omnibase_core.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_core.validation.validator_runtime_profiles import ValidatorRuntimeProfiles
from omnibase_infra.handlers.handler_deploy_agent_replacement import (
    HotpatchLedgerPreflightError,
    run_replacement,
)
from tests.scripts.test_hotpatch_ledger_gate_omn19998 import _Host

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[3]
NODE = ROOT / "src/omnibase_infra/nodes/host/node_deploy_agent_replacement_effect"
CONTRACT = yaml.safe_load((NODE / "contract.yaml").read_text())


def test_host_contract_does_not_attach_a_shared_runtime_consumer() -> None:
    """Host-injected I/O cannot be resolved by a shared runtime process."""
    assert "event_bus" not in CONTRACT
    assert "runtime_profiles" not in CONTRACT
    assert ValidatorRuntimeProfiles().validate(NODE).is_valid


def test_host_contract_is_not_advertised_to_shared_runtime_discovery() -> None:
    """The agent loads this contract directly and supplies host-only services.

    Exporting it through onex.nodes adds a contract to shared runtime discovery
    even though that process cannot supply the injected host execution boundary.
    """
    assert CONTRACT["name"] not in {
        entry.name for entry in entry_points(group="onex.nodes")
    }


def test_request_and_terminal_cross_the_bus_with_real_ledger_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = _Host(tmp_path)
    host.record(host.merged)
    monkeypatch.setattr(executor, "REPO_DIR", str(host.repo))
    monkeypatch.setenv("PATH", f"{host.bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.delenv("HOTPATCH_PREFLIGHT_BYPASS", raising=False)
    bus = EventBusInmemory()
    publish = bus.publish
    messages: list[tuple[str, dict[str, Any]]] = []

    async def capture(topic: str, key: bytes | None, value: bytes, **kw: Any) -> Any:
        messages.append((topic, json.loads(value)))
        return await publish(topic, key, value, **kw)

    monkeypatch.setattr(bus, "publish", capture)
    result = run_replacement(
        executor,
        lane=EnumRuntimeLane.DEV,
        environment={
            **os.environ,
            "HOTPATCH_LEDGER_PATH": str(host.ledger),
            "OMNI_HOME": str(host.clones_root),
        },
        contract=CONTRACT,
        bus=bus,
        lifecycle=bus,
    )
    assert result.returncode == 0
    assert [topic for topic, _ in messages] == [
        CONTRACT["host_dispatch"]["command_topic"],
        CONTRACT["host_dispatch"]["terminal_topic"],
    ]
    assert messages[0][1]["operation"] == "preflight"
    assert messages[0][1]["lane"] == EnumRuntimeLane.DEV.value
    assert messages[1][1] == result.model_dump(mode="json")


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("missing_script", "gate script is missing"),
        ("missing_home", "OMNI_HOME is unset"),
        ("git_failure", "could not be read"),
        ("gate_timeout", "timed out"),
    ],
)
def test_admission_failure_never_reaches_recreate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str, message: str
) -> None:
    host = _Host(tmp_path)
    host.record(host.merged)
    monkeypatch.setattr(executor, "REPO_DIR", str(host.repo))
    environment = {
        **os.environ,
        "HOTPATCH_LEDGER_PATH": str(host.ledger),
        "OMNI_HOME": str(host.clones_root),
    }
    if failure == "missing_script":
        (host.repo / "scripts/preflight_hotpatch_ledger.py").unlink()
    if failure == "missing_home":
        environment.pop("OMNI_HOME")
    commands: list[list[str]] = []

    def run(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        commands.append(cmd)
        assert "compose" not in cmd
        if cmd[0] == "git":
            return subprocess.CompletedProcess(
                cmd, 1 if failure == "git_failure" else 0, host.merged, "unreadable"
            )
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])

    monkeypatch.setattr(executor, "_run", run)
    bus = EventBusInmemory()
    error = (
        subprocess.TimeoutExpired
        if failure == "gate_timeout"
        else HotpatchLedgerPreflightError
    )
    with pytest.raises(error, match=message):
        run_replacement(
            executor,
            lane=EnumRuntimeLane.DEV,
            environment=environment,
            contract=CONTRACT,
            bus=bus,
            lifecycle=bus,
            argv=["docker", "compose", "up", "-d"],
            timeout=5,
        )
    assert not any("compose" in cmd for cmd in commands)


@pytest.mark.parametrize("outcome", ["removal_race", "timeout", "double_timeout"])
def test_retry_settles_and_preserves_result_over_bus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    """Only the host I/O is replaced; the handler, bus and models are real."""
    monkeypatch.setattr(executor, "REPO_DIR", str(tmp_path))
    calls: list[list[str]] = []
    cleanups: list[str] = []
    settlements: list[str] = []
    argv = ["docker", "compose", "up", "-d", "service"]

    def run(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if outcome == "double_timeout" or (outcome == "timeout" and len(calls) == 1):
            raise subprocess.TimeoutExpired(cmd, kw["timeout"])
        if outcome == "removal_race" and len(calls) == 1:
            return subprocess.CompletedProcess(
                cmd, 1, "", "removal is already in progress"
            )
        return subprocess.CompletedProcess(cmd, 0, "recreated\n", "")

    def cleanup(*args: Any) -> list[str]:
        cleanups.append("leftover")
        return ["leftover"]

    def settle(*args: Any) -> Any:
        settlements.append("settled")
        return executor.ModelServiceSettle(removed=["settled"], still_removing=[])

    monkeypatch.setattr(executor, "_run", run)
    monkeypatch.setattr(executor, "remove_recreate_leftovers", cleanup)
    monkeypatch.setattr(executor, "settle_service_containers", settle)
    bus = EventBusInmemory()
    removed: list[str] = []

    def invoke() -> Any:
        return run_replacement(
            executor,
            lane=EnumRuntimeLane.DEV,
            environment={"HOTPATCH_LEDGER_PATH": str(tmp_path / "absent-ledger")},
            contract=CONTRACT,
            bus=bus,
            lifecycle=bus,
            argv=argv,
            timeout=5,
            services=["service"],
            removed=removed,
        )

    if outcome == "double_timeout":
        with pytest.raises(subprocess.TimeoutExpired):
            invoke()
        assert len(cleanups) == len(settlements) == 2
        assert removed == ["leftover", "settled", "leftover", "settled"]
    else:
        result = invoke()
        assert result.returncode == 0
        assert result.stdout == "recreated\n"
        assert len(settlements) == 1
        assert len(cleanups) == (1 if outcome == "timeout" else 0)
        assert removed == (
            ["leftover", "settled"] if outcome == "timeout" else ["settled"]
        )
    assert calls == [argv, argv]
