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
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.events import EnumRuntimeLane
from deploy_agent.unit_drift import ModelUnitEntry, load_manifest

pytestmark = pytest.mark.unit

rom = executor_mod.load_dev_boot_guard("handler_restart_on_merge")
MODELS = sys.modules["omnibase_infra.models.model_restart_on_merge"]

REPO_ROOT = Path(__file__).resolve().parents[4]
#: An execution with no ledger to write to: a refusal is logged as pending.
NO_LEDGER = SimpleNamespace(
    dev_boot_report_config=lambda: SimpleNamespace(
        plugin_dir=None, ledger_via_bus=False
    )
)
MANIFEST = REPO_ROOT / "deploy" / "unit-drift-manifest.yaml"
PROD_UNIT = "deploy-agent.service"
DEV_UNITS = {
    e.name
    for e in load_manifest(MANIFEST)
    if e.tracked.startswith("scripts/deploy-agent/deploy/") and e.name != PROD_UNIT
}


def _entries() -> list[ModelUnitEntry]:
    return load_manifest(MANIFEST)


def test_restart_on_merge_set_is_the_declared_agents_minus_the_prod_fence() -> None:
    names = {entry.name for entry in rom.agent_units(_entries(), REPO_ROOT)}
    assert len(DEV_UNITS) == 2
    assert names == DEV_UNITS | {PROD_UNIT}
    by_name = {e.name: e for e in rom.agent_units(_entries(), REPO_ROOT)}
    assert rom.declared_fence(by_name[PROD_UNIT], REPO_ROOT) == {"prod"}
    for unit in DEV_UNITS:
        assert "prod" not in rom.declared_fence(by_name[unit], REPO_ROOT)
    # Whatever its env files hold, the prod fence keeps it out of the set.
    assert PROD_UNIT not in rom.restart_set(_entries(), REPO_ROOT, Path("/nonexistent"))


def test_restart_on_merge_the_fence_is_the_agents_own_parse() -> None:
    """The handler's reading of a fence names the same lanes ``lane_policy`` parses."""
    by_name = {e.name: e for e in rom.agent_units(_entries(), REPO_ROOT)}
    for name, entry in by_name.items():
        fence = rom.declared_fence(entry, REPO_ROOT)
        parsed = executor_mod.parse_allowed_lanes(",".join(sorted(fence)))
        assert {lane.value for lane in parsed} == fence, name
    assert EnumRuntimeLane.PROD.value == rom.PROD_LANE


def _target(unit: str) -> object:
    return MODELS.ModelRestartOnMerge(
        unit=unit, repo_root=str(REPO_ROOT), home="/nonexistent"
    )


def test_restart_on_merge_refuses_the_prod_unit_before_anything_is_replaced() -> None:
    with pytest.raises(rom.RestartRefusedError, match="fenced to prod"):
        rom.assert_restartable(_target(PROD_UNIT), _entries())
    for unit in DEV_UNITS:
        with pytest.raises(rom.RestartRefusedError, match="would restart without"):
            rom.assert_restartable(_target(unit), _entries())


def test_restart_on_merge_handler_decides_the_same() -> None:
    handler = rom.HandlerRestartOnMerge()
    prod = handler.handle(
        MODELS.ModelRestartOnMerge(
            unit=PROD_UNIT, repo_root=str(REPO_ROOT), home="/nonexistent"
        )
    )
    assert prod.restartable is False and "prod" in prod.reason
    bare = handler.handle(
        MODELS.ModelRestartOnMerge(
            unit="deploy-agent-dev.service",
            repo_root=str(REPO_ROOT),
            home="/nonexistent",
        )
    )
    assert bare.restartable is False and "DEPLOY_AGENT_CANONICAL_CLONE" in bare.reason


def test_restart_on_merge_executor_syncs_and_restarts_nothing_for_a_prod_agent() -> (
    None
):
    refusals: list[str] = []
    with (
        patch.object(rom.sys, "platform", "linux"),
        patch.object(rom, "read_own_cgroup", return_value=""),
        patch.object(
            rom,
            "write_refusal_status",
            lambda execution, reason, env: refusals.append(reason),
        ),
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
    assert refusals and "fenced to prod" in refusals[0]


def test_restart_on_merge_a_process_without_the_settings_keeps_its_revision() -> None:
    carried = {
        **dict.fromkeys(rom.REQUIRED_SETTINGS, "/x"),
        "DEPLOY_AGENT_ALLOWED_LANES": "dev",
    }
    assert rom.reexec_blocked_by_settings(NO_LEDGER, carried) is False
    for name in rom.REQUIRED_SETTINGS:
        partial = {k: v for k, v in carried.items() if k != name}
        assert rom.reexec_blocked_by_settings(NO_LEDGER, partial) is True
