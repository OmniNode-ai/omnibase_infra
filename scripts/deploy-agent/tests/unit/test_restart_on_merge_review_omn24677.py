# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The blocking findings 2 to 5 of the omnibase_infra#6027 review at 72d8c39490e5 (OMN-24677).

* restart_on_merge_guard: a blocking per-host check names every non-prod agent whose
  env files lack the settings, and a non-lab fence needs no lab overlay source.
* exit_zero_then_dead: a start that dies after the restart command returned 0 is
  verified from the transient unit, rolled back, and a second death raises.
* settings_env_file: the settings check reads the env file the launcher sources, not
  the process environment, and a refusal writes one STATUS row.
* overlay_store_dev: the overlay store is refused only for a composition that mounts
  it, from the registry root; each routing-table dev instance whose overlay drops the
  mount renders its environment with no store (one test per instance, named for it).
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from deploy_agent import executor as executor_mod

pytestmark = pytest.mark.unit

rom = executor_mod.load_dev_boot_guard("handler_restart_on_merge")
clone = executor_mod.load_dev_boot_guard("handler_deploy_source_clone")
models = sys.modules["omnibase_infra.models.model_restart_on_merge"]

REPO_ROOT = Path(__file__).resolve().parents[4]
#: An execution with no ledger to write to: a refusal is logged as pending.
NO_LEDGER = SimpleNamespace(
    dev_boot_report_config=lambda: SimpleNamespace(
        plugin_dir=None, ledger_via_bus=False
    )
)
STORE_TARGET = "/etc/onex/skill-overlays-store"
SETTINGS = {
    "DEPLOY_AGENT_REPO_DIR": "/state/deploy-source/omnibase_infra",
    "DEPLOY_AGENT_CANONICAL_CLONE": "/registry/omnibase_infra",
    "DEPLOY_AGENT_LAB_OVERLAY_SOURCE": "/registry/overlay-source",
}


def _world(tmp_path: Path, *, fence: str = "dev", store: dict[str, str]) -> Path:
    """A repository root with one agent unit whose launcher sources ``store.env``."""
    repo = tmp_path / "repo"
    (repo / "deploy").mkdir(parents=True)
    (repo / "units").mkdir()
    store_path = tmp_path / "store.env"
    store_path.write_text(
        "".join(f"{name}={value}\n" for name, value in store.items()), "utf-8"
    )
    (repo / "units" / "agent.service").write_text(
        "Environment=DEPLOY_AGENT_TRACKING_REF=dev\n"
        f"Environment=DEPLOY_AGENT_ALLOWED_LANES={fence}\n"
        f"Environment=DEPLOY_AGENT_ENV_FILE={store_path}\n",
        "utf-8",
    )
    (repo / "deploy" / "unit-drift-manifest.yaml").write_text(
        "units:\n  - name: agent.service\n    tracked: units/agent.service\n"
        "    installed: ~/.config/systemd/user/agent.service\n    hosts: [h]\n",
        "utf-8",
    )
    installed = tmp_path / "home" / ".config" / "systemd" / "user" / "agent.service"
    installed.parent.mkdir(parents=True)
    installed.write_text("installed\n", "utf-8")
    return repo


def _target(repo: Path, tmp_path: Path) -> object:
    return models.ModelRestartOnMerge(
        unit="agent.service", repo_root=str(repo), home=str(tmp_path / "home")
    )


def _entries(repo: Path) -> list[object]:
    return [
        sys.modules["omnibase_infra.models.model_agent_unit"].ModelAgentUnit(
            name="agent.service", tracked="units/agent.service"
        )
    ]


# --- finding 2 -----------------------------------------------------------------


def test_restart_on_merge_guard_check_host_names_each_agent_lacking_settings(
    tmp_path: Path,
) -> None:
    partial = {k: v for k, v in SETTINGS.items() if k != "DEPLOY_AGENT_CANONICAL_CLONE"}
    repo = _world(tmp_path, store=partial)
    handler = rom.HandlerRestartOnMerge()
    request = models.ModelRestartOnMerge(
        operation="check_host", repo_root=str(repo), home=str(tmp_path / "home")
    )
    blocked = handler.handle(request)
    assert blocked.restartable is False
    assert [gap.unit for gap in blocked.gaps] == ["agent.service"]
    assert blocked.gaps[0].missing == ("DEPLOY_AGENT_CANONICAL_CLONE",)
    (tmp_path / "store.env").write_text(
        "".join(f"{k}={v}\n" for k, v in SETTINGS.items()), "utf-8"
    )
    assert handler.handle(request).restartable is True


def test_restart_on_merge_guard_check_host_skips_absent_units(
    tmp_path: Path,
) -> None:
    other = _world(tmp_path / "other", store={})
    assert rom.check_host(_entries(other), other, tmp_path / "nobody") == ()


def test_restart_on_merge_guard_a_non_lab_fence_needs_no_overlay_source() -> None:
    env = {
        "DEPLOY_AGENT_REPO_DIR": "/state/jobs/deploy-source/omnibase_infra",
        "DEPLOY_AGENT_CANONICAL_CLONE": "/registry/omnibase_infra",
        "DEPLOY_AGENT_TRACKING_REF": "dev",
        "DEPLOY_AGENT_ALLOWED_LANES": "prod",
    }
    assert clone.settings(env).lab_overlay_source == ""
    with pytest.raises(clone.DeploySourceCloneError, match="LAB_OVERLAY_SOURCE"):
        clone.settings({**env, "DEPLOY_AGENT_ALLOWED_LANES": "dev"})
    assert rom.required_settings(frozenset({"prod"})) == (
        "DEPLOY_AGENT_REPO_DIR",
        "DEPLOY_AGENT_CANONICAL_CLONE",
    )


def test_restart_on_merge_guard_settings_refuse_a_canonical_deploy_source() -> None:
    """Finding 6: ``settings()`` itself refuses the canonical clone as deploy source."""
    with pytest.raises(clone.DeploySourceIsCanonicalCloneError):
        clone.settings(
            {
                "DEPLOY_AGENT_REPO_DIR": "/registry/omnibase_infra",
                "DEPLOY_AGENT_CANONICAL_CLONE": "/registry/omnibase_infra",
                "DEPLOY_AGENT_TRACKING_REF": "dev",
                "DEPLOY_AGENT_ALLOWED_LANES": "prod",
            }
        )


def test_restart_on_merge_guard_the_tracked_units_name_the_store_they_source() -> None:
    entries = rom.as_agent_units(
        executor_mod.unit_drift.load_manifest(REPO_ROOT / rom.MANIFEST_RELPATH)
    )
    home = Path("/home/agent")
    files = {
        e.name: rom.unit_env_files(e, REPO_ROOT, home)
        for e in rom.agent_units(entries, REPO_ROOT)
    }
    assert len(files) == 3
    assert all(files.values()), files


# --- finding 3 -----------------------------------------------------------------


def _restart(world: SimpleNamespace, tmp_path: Path) -> None:
    rom.restart_synced_unit(
        models.ModelRestartOnMerge(unit="agent.service", repo_root="/r", home="/h"),
        transient_command=[],
        script=rom.transient_restart_script(
            "agent.service",
            installed=world.installed,
            agent_dir=str(tmp_path / "agent"),
            loaded_sha="a" * 40,
            settle=0,
        ),
        run_unit_command=world.run,
    )


def test_exit_zero_then_dead_is_rolled_back_from_the_transient_unit(
    fake_systemd: SimpleNamespace, tmp_path: Path
) -> None:
    fake_systemd.alive.write_text("dead\nup\n", "utf-8")
    _restart(fake_systemd, tmp_path)
    calls = fake_systemd.log.read_text("utf-8").splitlines()
    assert calls.count("systemctl --user restart agent.service") == 2
    assert f"git -C {tmp_path / 'agent'} checkout --quiet --detach {'a' * 40}" in calls
    assert fake_systemd.installed.read_text("utf-8") == "previous unit\n"


def test_exit_zero_then_dead_twice_raises_agent_left_stopped(
    fake_systemd: SimpleNamespace, tmp_path: Path
) -> None:
    fake_systemd.alive.write_text("dead\ndead\n", "utf-8")
    with pytest.raises(rom.AgentLeftStoppedError, match="operator action needed"):
        _restart(fake_systemd, tmp_path)
    calls = fake_systemd.log.read_text("utf-8").splitlines()
    assert not any(" stop " in call for call in calls)


def test_exit_zero_then_dead_a_live_start_runs_no_rollback(
    fake_systemd: SimpleNamespace, tmp_path: Path
) -> None:
    fake_systemd.alive.write_text("up\n", "utf-8")
    _restart(fake_systemd, tmp_path)
    calls = fake_systemd.log.read_text("utf-8").splitlines()
    assert calls.count("systemctl --user restart agent.service") == 1
    assert fake_systemd.installed.read_text("utf-8") == "new unit\n"


# --- finding 4 -----------------------------------------------------------------


def test_settings_env_file_not_the_process_environment(tmp_path: Path) -> None:
    partial = {k: v for k, v in SETTINGS.items() if k != "DEPLOY_AGENT_REPO_DIR"}
    repo = _world(tmp_path, store=partial)
    with pytest.raises(rom.RestartRefusedError, match="DEPLOY_AGENT_REPO_DIR"):
        rom.assert_restartable(_target(repo, tmp_path), _entries(repo))
    # The store is fixed after this process started: the next check reads the file.
    (tmp_path / "store.env").write_text(
        "".join(f"export {k}='{v}'\n" for k, v in SETTINGS.items()), "utf-8"
    )
    rom.assert_restartable(_target(repo, tmp_path), _entries(repo))


def test_settings_env_file_reexec_reads_the_store_the_launcher_sources(
    tmp_path: Path,
) -> None:
    store = tmp_path / "store.env"
    store.write_text(
        "".join(f"{k}={v}\n" for k, v in SETTINGS.items()) + "EMPTY=\n", "utf-8"
    )
    env = {"DEPLOY_AGENT_ENV_FILE": str(store), "DEPLOY_AGENT_ALLOWED_LANES": "dev"}
    assert rom.reexec_blocked_by_settings(NO_LEDGER, env) is False
    store.write_text("DEPLOY_AGENT_REPO_DIR=\n", "utf-8")
    assert rom.reexec_blocked_by_settings(NO_LEDGER, env) is True


def test_settings_env_file_refusal_writes_one_status_row(tmp_path: Path) -> None:
    ran: list[list[str]] = []
    execution = SimpleNamespace(
        dev_boot_report_config=lambda: SimpleNamespace(
            plugin_dir=tmp_path / "plugin",
            ledger_via_bus=True,
            registry_root=tmp_path / "registry",
        ),
        _run=lambda argv, timeout: ran.append(argv) or SimpleNamespace(returncode=0),
    )
    env = {"DEPLOY_AGENT_STATE_DIR": str(tmp_path), "ONEX_LEDGER_HOST_NAME": "lab-host"}
    rom.write_refusal_status(execution, "agent.service would restart without X", env)
    rom.write_refusal_status(execution, "agent.service would restart without X", env)
    assert len(ran) == 1
    row = ran[0][-1]
    assert " | STATUS | lane=deploy-agent-lab-host | " in row
    assert "restart refused: agent.service would restart without X" in row
    rom.write_refusal_status(execution, "agent.service would restart without Y", env)
    assert len(ran) == 2


# --- finding 5 -----------------------------------------------------------------


@pytest.fixture
def dev_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    registry = tmp_path / "registry"
    own_clone = tmp_path / "state" / "deploy-source" / "omnibase_infra"
    (registry / "omnibase_infra").mkdir(parents=True)
    own_clone.mkdir(parents=True)
    (own_clone / "docker").symlink_to(REPO_ROOT / "docker")
    for name, value in {
        "DEPLOY_AGENT_REPO_DIR": str(own_clone),
        "DEPLOY_AGENT_CANONICAL_CLONE": str(registry / "omnibase_infra"),
        "DEPLOY_AGENT_LAB_OVERLAY_SOURCE": str(registry / "overlay-source"),
        "DEPLOY_AGENT_TRACKING_REF": "dev",
        "DEPLOY_AGENT_ALLOWED_LANES": "dev",
        "ONEX_LANE_OVERLAY_DIR": str(tmp_path),
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("DEV_LANE_OVERLAY_STORE_ROOTS_DIR", raising=False)
    monkeypatch.delenv("UNIT_DRIFT_REGISTRY_ROOT", raising=False)
    for cached in (
        executor_mod.deploy_source,
        executor_mod._lane_configs,
        executor_mod.dev_instance_lane_configs,
    ):
        cached.cache_clear()
    monkeypatch.setattr(executor_mod, "operator_store_overlay", dict)
    monkeypatch.setattr(executor_mod, "_load_runtime_policy_env", dict)
    monkeypatch.setattr(executor_mod, "_lane_env_overlay", dict)
    monkeypatch.setattr(executor_mod, "_dev_broker_policy_env", lambda env: {})
    monkeypatch.setattr(
        executor_mod, "_llm_endpoint_env_overlay", lambda env: ({}, None)
    )
    monkeypatch.setattr(
        executor_mod._DevInstanceSelection, "name", executor_mod.DEFAULT_DEV_INSTANCE
    )
    return registry


def _instances_without_store_mount() -> list[str]:
    """Routing-table dev instances whose overlay replaces volumes and names no store.

    Read from the overlay files themselves, not through the code under test.
    """
    names = []
    for name, spec in executor_mod.load_instance_lanes(REPO_ROOT).items():
        text = (REPO_ROOT / spec.compose_overlay).read_text("utf-8")
        if "volumes: !override" in text and STORE_TARGET not in text:
            names.append(name)
    return names


def _renders_with_no_store(instance: str) -> object:
    def test(dev_world: Path) -> None:
        executor_mod.select_dev_instance(instance)
        env = executor_mod._compose_env()
        assert "DEV_LANE_OVERLAY_STORE_ROOTS_DIR" not in env

    return test


INSTANCES_WITHOUT_MOUNT = _instances_without_store_mount()
for _instance in INSTANCES_WITHOUT_MOUNT:
    globals()[
        f"test_overlay_store_{_instance.replace('-', '_')}_with_no_store_renders"
    ] = _renders_with_no_store(_instance)


def test_overlay_store_dev_instance_without_the_mount_exists() -> None:
    assert INSTANCES_WITHOUT_MOUNT


def test_overlay_store_dev_control_default_instance_refuses_an_absent_store(
    dev_world: Path,
) -> None:
    with pytest.raises(clone.DeploySourceCloneError, match="absent or empty"):
        executor_mod._compose_env()


def test_overlay_store_dev_store_is_derived_from_the_registry_root(
    dev_world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "declared-registry"
    store = registry / "overlays" / "overlays"
    store.mkdir(parents=True)
    (store / "overlay.yaml").write_text("schema_version: '1'\n", "utf-8")
    monkeypatch.setenv("UNIT_DRIFT_REGISTRY_ROOT", str(registry))
    executor_mod.deploy_source.cache_clear()
    assert executor_mod._compose_env()["DEV_LANE_OVERLAY_STORE_ROOTS_DIR"] == str(store)
