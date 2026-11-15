# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A restart on merge never touches a prod agent and never leaves an agent stopped (OMN-24626).

Review finding HIGH 3 of omnibase_infra#6027: the merge edited every tracked agent unit,
each agent syncs its own unit and restarts it, and the new code refuses to start without
settings the tracked units left to the host env store.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.events import EnumRuntimeLane
from deploy_agent.unit_drift import ModelUnitEntry, load_manifest

pytestmark = pytest.mark.unit

rom = executor_mod.load_dev_boot_guard("handler_restart_on_merge")

REPO_ROOT = Path(__file__).resolve().parents[4]
MANIFEST = REPO_ROOT / "deploy" / "unit-drift-manifest.yaml"
PROD_UNIT = "deploy-agent.service"
DEV_UNITS = {
    e.name
    for e in load_manifest(MANIFEST)
    if e.tracked.startswith("scripts/deploy-agent/deploy/") and e.name != PROD_UNIT
}


def _entries() -> list[ModelUnitEntry]:
    return load_manifest(MANIFEST)


CARRIED = dict.fromkeys(rom.REQUIRED_SETTINGS, "/set")


def test_restart_on_merge_set_is_the_declared_agents_minus_the_prod_fence() -> None:
    names = {entry.name for entry in rom.agent_units(_entries(), REPO_ROOT)}
    assert len(DEV_UNITS) == 2
    assert names == DEV_UNITS | {PROD_UNIT}
    by_name = {e.name: e for e in rom.agent_units(_entries(), REPO_ROOT)}
    assert rom.declared_fence(by_name[PROD_UNIT], REPO_ROOT) == {"prod"}
    for unit in DEV_UNITS:
        assert "prod" not in rom.declared_fence(by_name[unit], REPO_ROOT)
    # Even carrying every setting, the prod fence keeps it out of the set.
    assert set(rom.restart_set(_entries(), REPO_ROOT, CARRIED)) == DEV_UNITS


def test_restart_on_merge_the_fence_is_the_agents_own_parse() -> None:
    """The handler's reading of a fence names the same lanes ``lane_policy`` parses."""
    by_name = {e.name: e for e in rom.agent_units(_entries(), REPO_ROOT)}
    for name, entry in by_name.items():
        fence = rom.declared_fence(entry, REPO_ROOT)
        parsed = executor_mod.parse_allowed_lanes(",".join(sorted(fence)))
        assert {lane.value for lane in parsed} == fence, name
    assert EnumRuntimeLane.PROD.value == rom.PROD_LANE


def test_restart_on_merge_refuses_the_prod_unit_before_anything_is_replaced() -> None:
    with pytest.raises(rom.RestartRefusedError, match="fenced to prod"):
        rom.assert_restartable(PROD_UNIT, _entries(), REPO_ROOT, CARRIED)
    for unit in DEV_UNITS:
        rom.assert_restartable(unit, _entries(), REPO_ROOT, CARRIED)


def test_restart_on_merge_a_unit_missing_a_setting_is_never_synced(
    tmp_path: Path,
) -> None:
    unit_dir = tmp_path / "units"
    unit_dir.mkdir()
    (unit_dir / "agent.service").write_text(
        "Environment=DEPLOY_AGENT_TRACKING_REF=dev\n"
        "Environment=DEPLOY_AGENT_ALLOWED_LANES=dev\n"
        "Environment=DEPLOY_AGENT_REPO_DIR=/state/deploy-source/omnibase_infra\n"
    )
    entry = ModelUnitEntry(
        name="agent.service",
        tracked="units/agent.service",
        installed="~/.config/systemd/user/agent.service",
        hosts=["h"],
    )
    carried = {k: v for k, v in CARRIED.items() if k != "DEPLOY_AGENT_CANONICAL_CLONE"}
    with pytest.raises(rom.RestartRefusedError, match="DEPLOY_AGENT_CANONICAL_CLONE"):
        rom.assert_restartable(entry.name, [entry], tmp_path, carried)
    assert rom.restart_set([entry], tmp_path, carried) == []
    # Declared in the unit or carried by the process: either is verified present.
    assert rom.restart_set([entry], tmp_path, CARRIED) == [entry.name]


def test_restart_on_merge_handler_decides_the_same(tmp_path: Path) -> None:
    handler = rom.HandlerRestartOnMerge()
    request_type = sys.modules[
        "omnibase_infra.models.model_restart_on_merge"
    ].ModelRestartOnMerge
    prod = handler.handle(
        request_type(
            unit_name=PROD_UNIT,
            repo_root=str(REPO_ROOT),
            carried_settings=tuple(CARRIED),
        )
    )
    assert prod.restartable is False and "prod" in prod.reason
    dev = handler.handle(
        request_type(
            unit_name="deploy-agent-dev.service",
            repo_root=str(REPO_ROOT),
            carried_settings=tuple(CARRIED),
        )
    )
    assert dev.restartable is True
    bare = handler.handle(
        request_type(unit_name="deploy-agent-dev.service", repo_root=str(REPO_ROOT))
    )
    assert bare.restartable is False and "DEPLOY_AGENT_CANONICAL_CLONE" in bare.reason


def test_restart_on_merge_executor_syncs_and_restarts_nothing_for_a_prod_agent() -> (
    None
):
    with (
        patch.object(
            executor_mod.unit_drift, "own_unit_name_from_cgroup", return_value=PROD_UNIT
        ),
        patch.object(
            executor_mod.unit_drift,
            "sync_own_unit",
            side_effect=AssertionError("synced prod"),
        ),
    ):
        assert (
            executor_mod._sync_own_unit_after_pull(
                str(REPO_ROOT / "scripts" / "deploy-agent"), 60
            )
            is None
        )


def _installed_unit(home: Path, entry: ModelUnitEntry) -> Path:
    installed = home / ".config" / "systemd" / "user" / entry.name
    installed.parent.mkdir(parents=True)
    installed.write_text("new unit\n")
    backup = installed.with_name(installed.name + ".bak-unit-drift-20260101T000000Z")
    backup.write_text("previous unit\n")
    return installed


def _dev_entry() -> ModelUnitEntry:
    return next(e for e in _entries() if e.name == "deploy-agent-dev.service")


def test_restart_on_merge_failed_restart_restarts_the_previous_revision(
    tmp_path: Path,
) -> None:
    entry = _dev_entry()
    installed = _installed_unit(tmp_path, entry)
    commands: list[list[str]] = []
    restored: list[str] = []

    def run(command: list[str]) -> int:
        commands.append(command)
        is_restart = command[-2:-1] == ["restart"]
        return 1 if is_restart and len(restored) == 0 else 0

    rom.restart_synced_unit(
        entry.name,
        restart_command=["systemctl", "--user", "restart"],
        run=run,
        rollback=(
            lambda: restored.append("code"),
            lambda: rom.restore_previous_unit(
                executor_mod, entry.name, [entry], REPO_ROOT, tmp_path, run
            ),
        ),
    )
    assert installed.read_text() == "previous unit\n"
    assert restored == ["code"]
    restarts = [c for c in commands if c[-2:-1] == ["restart"]]
    assert [c[-1] for c in restarts] == [entry.name, entry.name]


def test_restart_on_merge_a_second_failure_is_loud_and_stops_no_other_agent(
    tmp_path: Path,
) -> None:
    entry = _dev_entry()
    _installed_unit(tmp_path, entry)
    commands: list[list[str]] = []

    def run(command: list[str]) -> int:
        commands.append(command)
        return 1

    with pytest.raises(rom.AgentLeftStoppedError, match="operator action needed"):
        rom.restart_synced_unit(
            entry.name,
            restart_command=["systemctl", "--user", "restart"],
            run=run,
            rollback=(lambda: None,),
        )
    named = {c[-1] for c in commands if c[-2:-1] == ["restart"]}
    assert named == {entry.name}
    assert not any("stop" in c for c in commands)


def test_restart_on_merge_a_clean_restart_runs_no_rollback(tmp_path: Path) -> None:
    entry = _dev_entry()
    installed = _installed_unit(tmp_path, entry)
    rom.restart_synced_unit(
        entry.name,
        restart_command=["systemctl", "--user", "restart"],
        run=lambda command: 0,
        rollback=(lambda: pytest.fail("rollback ran after a clean restart"),),
    )
    assert installed.read_text() == "new unit\n"


def test_restart_on_merge_a_process_without_the_settings_keeps_its_revision() -> None:
    carried = dict.fromkeys(rom.REQUIRED_SETTINGS, "/x")
    assert rom.reexec_blocked_by_settings(carried) is False
    for name in rom.REQUIRED_SETTINGS:
        partial = {k: v for k, v in carried.items() if k != name}
        assert rom.reexec_blocked_by_settings(partial) is True
