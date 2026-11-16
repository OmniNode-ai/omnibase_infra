# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The findings of the omnibase_infra#6027 review at eee30caf4fad (OMN-24713).

* check_host_prod_fenced / check_host_launchd: the per-host reading lists the prod-fenced
  unit and names every installed launchd instance, names only, never a value.
* rollback_marker / self_update_exit_3: a rollback leaves a marker from the transient
  unit, which outlives its stopped caller; ``self_update`` skips the refused revision and
  raises an agent left stopped, driven through ``self_update`` itself.
* nrestarts_nonzero: a unit that is active but restarted by systemd is not up.
* reexec_refusal_status: a refused re-exec writes its STATUS row and a pass clears it.
* deploy_source_clone: the guard names the code checkout, a clone has a timeout, the
  test settings are assigned, and the tracked units no longer misdescribe the seed.
"""

from __future__ import annotations

import ast
import os
import plistlib
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent import loaded_code
from deploy_agent.events import EnumSelfUpdateBoundary
from deploy_agent.executor import DeployExecutor

from .test_agent_own_clone_omn24562 import _ensure, _World, dsc
from .test_restart_on_merge_review_omn24677 import (
    SETTINGS,
    _entries,
    _world,
    models,
    rom,
)
from .test_self_update_loaded_code_omn18200 import (
    _advance_origin,
    _git,
    _make_origin_and_clone,
)
from .test_self_update_restart_omn22478 import SYNC, UNIT, _restart_ends_with

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
REFUSED = "c" * 40


# --- finding 1: the prod-fenced unit ------------------------------------------------


def test_restart_on_merge_check_host_prod_fenced_unit_is_listed(
    tmp_path: Path,
) -> None:
    repo = _world(tmp_path, fence="prod", store={})
    gaps = rom.check_host(_entries(repo), repo, tmp_path / "home")
    assert [gap.unit for gap in gaps] == ["agent.service"]
    assert gaps[0].fence == ("prod",)
    assert gaps[0].missing == (
        "DEPLOY_AGENT_REPO_DIR",
        "DEPLOY_AGENT_CANONICAL_CLONE",
    )
    request = models.ModelRestartOnMerge(
        operation="check_host", repo_root=str(repo), home=str(tmp_path / "home")
    )
    answer = rom.HandlerRestartOnMerge().handle(request)
    assert answer.restartable is False
    assert "agent.service: DEPLOY_AGENT_REPO_DIR" in answer.reason


def test_restart_on_merge_check_host_prod_fenced_reading_is_names_only(
    tmp_path: Path,
) -> None:
    secret = "do-not-print-this-value"
    store = {"DEPLOY_AGENT_CANONICAL_CLONE": secret, "ANOTHER_SETTING": secret}
    repo = _world(tmp_path, fence="prod", store=store)
    (gap,) = rom.check_host(_entries(repo), repo, tmp_path / "home")
    assert gap.found == ("DEPLOY_AGENT_CANONICAL_CLONE",)
    assert gap.missing == ("DEPLOY_AGENT_REPO_DIR",)
    assert secret not in repr(gap)
    answer = rom.HandlerRestartOnMerge().handle(
        models.ModelRestartOnMerge(
            operation="check_host", repo_root=str(repo), home=str(tmp_path / "home")
        )
    )
    assert secret not in answer.model_dump_json()


def test_restart_on_merge_check_host_prod_fenced_complete_store_is_restartable(
    tmp_path: Path,
) -> None:
    prod_settings = {k: v for k, v in SETTINGS.items() if "LAB_OVERLAY" not in k}
    repo = _world(tmp_path, fence="prod", store=prod_settings)
    request = models.ModelRestartOnMerge(
        operation="check_host", repo_root=str(repo), home=str(tmp_path / "home")
    )
    answer = rom.HandlerRestartOnMerge().handle(request)
    assert answer.restartable is True and answer.gaps == ()
    # The merge-time restart set itself still excludes the prod unit.
    assert rom.restart_set(_entries(repo), repo, tmp_path / "home") == []


# --- finding 7: launchd instances ---------------------------------------------------


def _launchd_instance(home: Path, instance: str, store: Path) -> str:
    label = f"ai.omninode.deploy-agent-{instance}"
    agents = home / "Library" / "LaunchAgents"
    agents.mkdir(parents=True, exist_ok=True)
    (agents / f"{label}.plist").write_bytes(
        plistlib.dumps(
            {
                "Label": label,
                "EnvironmentVariables": {
                    "DEPLOY_AGENT_ENV_FILE": str(store),
                    "DEPLOY_AGENT_ALLOWED_LANES": "dev",
                    "DEPLOY_AGENT_REPO_DIR": "/lane/state/deploy-source/omnibase_infra",
                },
            }
        )
    )
    return label


def test_restart_on_merge_check_host_launchd_instances_are_named(
    tmp_path: Path,
) -> None:
    repo = _world(tmp_path, store=dict(SETTINGS))
    home = tmp_path / "home"
    complete = tmp_path / "complete.env"
    complete.write_text("".join(f"{k}={v}\n" for k, v in SETTINGS.items()), "utf-8")
    partial = tmp_path / "partial.env"
    partial.write_text("DEPLOY_AGENT_LAB_OVERLAY_SOURCE=/overlay\n", "utf-8")
    ok = _launchd_instance(home, "dev-a", complete)
    gapped = _launchd_instance(home, "dev-b", partial)
    answer = rom.HandlerRestartOnMerge().handle(
        models.ModelRestartOnMerge(
            operation="check_host", repo_root=str(repo), home=str(home)
        )
    )
    assert answer.launchd_instances == (ok, gapped)
    assert [gap.unit for gap in answer.gaps] == [gapped]
    assert answer.gaps[0].missing == ("DEPLOY_AGENT_CANONICAL_CLONE",)
    assert answer.restartable is False


def test_restart_on_merge_check_host_launchd_absent_names_none(tmp_path: Path) -> None:
    repo = _world(tmp_path, store=dict(SETTINGS))
    answer = rom.HandlerRestartOnMerge().handle(
        models.ModelRestartOnMerge(
            operation="check_host", repo_root=str(repo), home=str(tmp_path / "home")
        )
    )
    assert answer.launchd_instances == () and answer.restartable is True


# --- finding 3: NRestarts ------------------------------------------------------------


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


def test_restart_on_merge_nrestarts_nonzero_is_not_up(
    fake_systemd: SimpleNamespace, tmp_path: Path
) -> None:
    """Active, but systemd restarted it twice by itself: that start died."""
    fake_systemd.alive.write_text("up\nup\n", "utf-8")
    fake_systemd.restarts.write_text("2\n0\n", "utf-8")
    _restart(fake_systemd, tmp_path)
    calls = fake_systemd.log.read_text("utf-8").splitlines()
    assert calls.count("systemctl --user restart agent.service") == 2
    assert fake_systemd.installed.read_text("utf-8") == "previous unit\n"


def test_restart_on_merge_nrestarts_nonzero_twice_raises_agent_left_stopped(
    fake_systemd: SimpleNamespace, tmp_path: Path
) -> None:
    fake_systemd.alive.write_text("up\nup\n", "utf-8")
    fake_systemd.restarts.write_text("1\n3\n", "utf-8")
    with pytest.raises(rom.AgentLeftStoppedError):
        _restart(fake_systemd, tmp_path)


# --- finding 2: the rollback marker --------------------------------------------------


def _script(world: SimpleNamespace, tmp_path: Path, marker: Path) -> str:
    return rom.transient_restart_script(
        "agent.service",
        installed=world.installed,
        agent_dir=str(tmp_path / "agent"),
        loaded_sha="a" * 40,
        settle=0,
        refused_marker=marker,
    )


def test_restart_on_merge_rollback_marker_survives_a_killed_caller(
    fake_systemd: SimpleNamespace, tmp_path: Path
) -> None:
    """The first restart stops the caller; the script still leaves the marker."""
    caller = subprocess.Popen(["sleep", "60"])
    (fake_systemd.bin_dir / "killpid").write_text(str(caller.pid), "utf-8")
    systemctl = fake_systemd.bin_dir / "systemctl"
    systemctl.write_text(
        systemctl.read_text("utf-8").replace(
            "case ",
            f"if [ -f {fake_systemd.bin_dir}/killpid ]; then "
            f"kill -TERM $(cat {fake_systemd.bin_dir}/killpid); "
            f"rm -f {fake_systemd.bin_dir}/killpid; fi\ncase ",
            1,
        ),
        "utf-8",
    )
    fake_systemd.alive.write_text("dead\nup\n", "utf-8")
    marker = tmp_path / "state" / rom.ROLLBACK_MARKER
    exit_file = tmp_path / "exit"
    script = _script(fake_systemd, tmp_path, marker)
    env = {
        **os.environ,
        "PATH": f"{fake_systemd.bin_dir}{os.pathsep}{os.environ['PATH']}",
    }
    # A new session, as a transient unit is: nothing the caller dies with reaches it.
    subprocess.Popen(
        ["bash", "-c", f"bash -c {shlex.quote(script)}; echo $? > {exit_file}"],
        env=env,
        start_new_session=True,
    )
    deadline = time.monotonic() + 20
    while not exit_file.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert exit_file.read_text("utf-8").strip() == str(rom.ROLLED_BACK)
    assert caller.wait(timeout=5) == -signal.SIGTERM
    assert marker.read_text("utf-8") == "b" * 40


def test_restart_on_merge_rollback_marker_is_not_left_by_a_live_start(
    fake_systemd: SimpleNamespace, tmp_path: Path
) -> None:
    fake_systemd.alive.write_text("up\n", "utf-8")
    marker = tmp_path / "state" / rom.ROLLBACK_MARKER
    env = {
        **os.environ,
        "PATH": f"{fake_systemd.bin_dir}{os.pathsep}{os.environ['PATH']}",
    }
    done = subprocess.run(
        ["bash", "-c", _script(fake_systemd, tmp_path, marker)], env=env, check=False
    )
    assert done.returncode == 0 and not marker.exists()


def test_restart_on_merge_rollback_marker_is_not_left_when_nothing_runs(
    fake_systemd: SimpleNamespace, tmp_path: Path
) -> None:
    fake_systemd.alive.write_text("dead\ndead\n", "utf-8")
    marker = tmp_path / "state" / rom.ROLLBACK_MARKER
    env = {
        **os.environ,
        "PATH": f"{fake_systemd.bin_dir}{os.pathsep}{os.environ['PATH']}",
    }
    done = subprocess.run(
        ["bash", "-c", _script(fake_systemd, tmp_path, marker)], env=env, check=False
    )
    assert done.returncode == rom.LEFT_STOPPED and not marker.exists()


def test_restart_on_merge_rollback_marker_path_reaches_the_script(
    tmp_path: Path,
) -> None:
    ran: list[list[str]] = []
    execution = SimpleNamespace(
        SELF_RESTART_COMMAND="systemd-run --user --collect --wait systemctl --user restart",
        _run=lambda argv, timeout: ran.append(argv) or SimpleNamespace(returncode=0),
    )
    agent_dir = tmp_path / "repo" / "scripts" / "deploy-agent"
    agent_dir.mkdir(parents=True)
    state = tmp_path / "state"
    rom.restart_own_unit(
        execution,
        "agent.service",
        agent_dir=str(agent_dir),
        loaded_sha="a" * 40,
        timeout=60,
        env={"HOME": str(tmp_path), "DEPLOY_AGENT_STATE_DIR": str(state)},
    )
    assert str(state / rom.ROLLBACK_MARKER) in ran[0][-1]


def _status_execution(tmp_path: Path, ran: list[list[str]]) -> SimpleNamespace:
    return SimpleNamespace(
        dev_boot_report_config=lambda: SimpleNamespace(
            plugin_dir=tmp_path / "plugin",
            ledger_via_bus=True,
            registry_root=tmp_path / "registry",
        ),
        _run=lambda argv, timeout: ran.append(argv) or SimpleNamespace(returncode=0),
    )


def test_restart_on_merge_rollback_marker_skips_the_refused_revision_with_one_row(
    tmp_path: Path,
) -> None:
    ran: list[list[str]] = []
    execution = _status_execution(tmp_path, ran)
    env = {"DEPLOY_AGENT_STATE_DIR": str(tmp_path), "ONEX_LEDGER_HOST_NAME": "lab-host"}
    assert rom.skip_refused_revision(execution, REFUSED, env) is False
    (tmp_path / rom.ROLLBACK_MARKER).write_text(REFUSED, "utf-8")
    for _ in range(3):
        assert rom.skip_refused_revision(execution, REFUSED, env) is True
    assert len(ran) == 1
    assert f"revision {REFUSED} refused" in ran[0][-1]
    # The remote moved on: the marker goes, and the next refusal writes a new row.
    assert rom.skip_refused_revision(execution, "d" * 40, env) is False
    assert not (tmp_path / rom.ROLLBACK_MARKER).exists()
    (tmp_path / rom.ROLLBACK_MARKER).write_text(REFUSED, "utf-8")
    assert rom.skip_refused_revision(execution, REFUSED, env) is True
    assert len(ran) == 2


def _recorded_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path]:
    origin, clone = _make_origin_and_clone(tmp_path)
    monkeypatch.setenv("DEPLOY_AGENT_DIR", str(clone))
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setenv("DEPLOY_AGENT_STATE_DIR", str(state))
    loaded_code.record_loaded_code_sha(str(clone))
    return origin, clone, state


def test_restart_on_merge_rollback_marker_self_update_skips_then_follows_the_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    origin, clone, state = _recorded_clone(tmp_path, monkeypatch)
    loaded = _git(clone, "rev-parse", "HEAD")
    refused = _advance_origin(origin, "print('v2')\n")
    (state / rom.ROLLBACK_MARKER).write_text(refused, "utf-8")
    rows: list[str] = []
    with (
        patch.object(
            rom, "write_refusal_status", lambda e, reason, env, *a: rows.append(reason)
        ),
        patch(SYNC, side_effect=AssertionError("synced a refused revision")),
        patch("os.execv") as execv,
    ):
        for _ in range(2):
            DeployExecutor().self_update(boundary=EnumSelfUpdateBoundary.POST_TERMINAL)
        assert _git(clone, "rev-parse", "HEAD") == loaded
        assert len(rows) == 2 and all(refused in row for row in rows)
        execv.assert_not_called()
    # The remote moves past the refused revision: the marker is dropped and it is pulled.
    newer = _advance_origin(origin, "print('v3')\n")
    with (
        patch(SYNC, return_value=None),
        patch("deploy_agent.executor._uv_sync_after_pull", return_value=None),
        patch("os.execv"),
    ):
        DeployExecutor().self_update(boundary=EnumSelfUpdateBoundary.POST_TERMINAL)
    assert _git(clone, "rev-parse", "HEAD") == newer
    assert not (state / rom.ROLLBACK_MARKER).exists()


# --- finding 4: self_update with exit 3 ---------------------------------------------


def _moved_clone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    origin, clone, _ = _recorded_clone(tmp_path, monkeypatch)
    _advance_origin(origin, "print('v2')\n")
    _git(clone, "fetch", "origin", "dev")
    _git(clone, "reset", "--hard", "origin/dev")


def test_restart_on_merge_self_update_exit_3_raises_agent_left_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _moved_clone(tmp_path, monkeypatch)
    with (
        patch(SYNC, return_value=UNIT),
        patch("subprocess.run", side_effect=_restart_ends_with(rom.LEFT_STOPPED)),
        patch("os.execv") as execv,
        pytest.raises(rom.AgentLeftStoppedError, match="operator action needed"),
    ):
        DeployExecutor().self_update(boundary=EnumSelfUpdateBoundary.POST_TERMINAL)
    execv.assert_not_called()


def test_restart_on_merge_self_update_exit_2_ends_the_update_without_reexec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _moved_clone(tmp_path, monkeypatch)
    with (
        patch(SYNC, return_value=UNIT),
        patch("subprocess.run", side_effect=_restart_ends_with(rom.ROLLED_BACK)),
        patch("os.execv") as execv,
    ):
        DeployExecutor().self_update(boundary=EnumSelfUpdateBoundary.POST_TERMINAL)
    execv.assert_not_called()


# --- finding 6: the re-exec refusal -------------------------------------------------


def test_restart_on_merge_reexec_refusal_status_is_written_once_and_cleared_on_a_pass(
    tmp_path: Path,
) -> None:
    ran: list[list[str]] = []
    execution = _status_execution(tmp_path, ran)
    store = tmp_path / "store.env"
    store.write_text("DEPLOY_AGENT_REPO_DIR=\n", "utf-8")
    env = {
        "DEPLOY_AGENT_ENV_FILE": str(store),
        "DEPLOY_AGENT_ALLOWED_LANES": "dev",
        "DEPLOY_AGENT_STATE_DIR": str(tmp_path),
        "ONEX_LEDGER_HOST_NAME": "lab-host",
    }
    assert rom.reexec_blocked_by_settings(execution, env) is True
    assert rom.reexec_blocked_by_settings(execution, env) is True
    assert len(ran) == 1
    assert "not re-execing into the pulled revision" in ran[0][-1]
    assert (tmp_path / rom.REEXEC_REFUSAL_MARKER).is_file()
    store.write_text("".join(f"{k}={v}\n" for k, v in SETTINGS.items()), "utf-8")
    assert rom.reexec_blocked_by_settings(execution, env) is False
    assert not (tmp_path / rom.REEXEC_REFUSAL_MARKER).exists()
    store.write_text("DEPLOY_AGENT_REPO_DIR=\n", "utf-8")
    assert rom.reexec_blocked_by_settings(execution, env) is True
    assert len(ran) == 2


def test_restart_on_merge_reexec_refusal_status_the_unit_sync_pass_clears_its_marker(
    tmp_path: Path,
) -> None:
    repo = _world(tmp_path, store=dict(SETTINGS))
    agent_dir = repo / "scripts" / "deploy-agent"
    agent_dir.mkdir(parents=True)
    state = tmp_path / "state"
    state.mkdir()
    (state / rom.REFUSAL_MARKER).write_text("an earlier refusal", "utf-8")
    execution = SimpleNamespace(
        unit_drift=SimpleNamespace(
            own_unit_name_from_cgroup=lambda text: "agent.service",
            load_manifest=lambda path: _entries(repo),
            sync_own_unit=lambda *args, **kwargs: False,
        ),
        _run=lambda argv, timeout: SimpleNamespace(returncode=0),
    )
    env = {"HOME": str(tmp_path / "home"), "DEPLOY_AGENT_STATE_DIR": str(state)}
    with (
        patch.object(rom.sys, "platform", "linux"),
        patch.object(rom, "read_own_cgroup", return_value=""),
    ):
        assert rom.sync_own_unit_after_pull(execution, str(agent_dir), 60, env) is None
    assert not (state / rom.REFUSAL_MARKER).exists()


# --- finding 8: the earlier LOW items ------------------------------------------------


def test_deploy_source_clone_refuses_the_agents_own_code_checkout() -> None:
    env = {
        "DEPLOY_AGENT_REPO_DIR": "/host/omnibase_infra",
        "DEPLOY_AGENT_CANONICAL_CLONE": "/registry/omnibase_infra",
        "DEPLOY_AGENT_TRACKING_REF": "dev",
        "DEPLOY_AGENT_ALLOWED_LANES": "prod",
        "DEPLOY_AGENT_DIR": "/host/omnibase_infra/scripts/deploy-agent",
    }
    with pytest.raises(dsc.DeploySourceIsCanonicalCloneError, match="code checkout"):
        dsc.settings(env)
    elsewhere = {**env, "DEPLOY_AGENT_REPO_DIR": "/state/deploy-source/omnibase_infra"}
    assert dsc.settings(elsewhere).code_checkout == "/host/omnibase_infra"


def test_deploy_source_clone_creation_runs_under_the_declared_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _World(tmp_path)
    seen: dict[str, object] = {}
    real_run = subprocess.run

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        timeout = kwargs.get("timeout")
        if "clone" in argv:
            seen["clone"] = timeout
        return real_run(
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout if isinstance(timeout, float) else None,
        )

    monkeypatch.setattr(dsc.subprocess, "run", run)
    assert _ensure(world.settings()).created is True
    assert seen["clone"] == 120.0


def test_deploy_source_clone_without_a_declared_timeout_is_refused(
    tmp_path: Path,
) -> None:
    world = _World(tmp_path)
    unbounded = world.settings().model_copy(update={"timeout_seconds": 0.0})
    with pytest.raises(dsc.DeploySourceCloneError, match="declares no timeout"):
        _ensure(unbounded)
    assert not world.agent_clone.exists()


def test_deploy_source_clone_the_executor_declares_the_bound_on_the_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor_mod.deploy_source.cache_clear()
    try:
        bound = executor_mod.deploy_source().timeout_seconds
    finally:
        executor_mod.deploy_source.cache_clear()
    assert bound == 5 * executor_mod.PHASE_TIMEOUTS[executor_mod.Phase.GIT]


def test_deploy_source_clone_that_times_out_is_refused_not_hung(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _World(tmp_path)

    def hang(argv: list[str], **kwargs: object) -> object:
        raise subprocess.TimeoutExpired(argv, 1.0)

    monkeypatch.setattr(dsc.subprocess, "run", hang)
    with pytest.raises(dsc.DeploySourceCloneError, match="timed out"):
        _ensure(world.settings())


def test_deploy_source_clone_test_settings_replace_an_ambient_clone_path() -> None:
    """``tests/conftest.py`` assigns the clone settings: an ambient value is replaced.

    The repository's own block runs in a clean interpreter that starts with a real
    host path in the variable.
    """
    tree = ast.parse((REPO_ROOT / "tests" / "conftest.py").read_text("utf-8"))
    block = next(
        node
        for node in tree.body
        if isinstance(node, ast.For) and "DEPLOY_AGENT_REPO_DIR" in ast.unparse(node)
    )
    code = (
        f"import os\n{ast.unparse(block)}\nprint(os.environ['DEPLOY_AGENT_REPO_DIR'])"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-c", code],
        env={"DEPLOY_AGENT_REPO_DIR": "/host/real/deploy-source"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert done.stdout.strip().startswith("/nonexistent/")


def test_deploy_source_clone_the_tracked_units_do_not_misdescribe_the_seed() -> None:
    for name in ("deploy-agent.service", "deploy-agent-dev.service"):
        text = (REPO_ROOT / "scripts" / "deploy-agent" / "deploy" / name).read_text(
            "utf-8"
        )
        assert "from the canonical clone's HEAD" not in text, name
        assert "never a commit" in text, name
