# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The findings of the omnibase_infra#6027 review at c9f626af2538 (OMN-24761).

* rollback_reset_failed: after the new revision died under the unit's start limit, a
  restart is refused until ``reset-failed``; the transient script resets the unit's
  failed state before each restart so the rollback can start the previous revision.
* check_host_environment_file: a name an ``EnvironmentFile=`` source assigns with
  ``export`` is dropped by systemd, so only a bare ``NAME=value`` counts there; the
  store a launcher sources still counts ``export`` lines.
* refusal_append_failure: a ledger append that raises or times out is logged and
  leaves the marker unwritten; it never raises out of ``self_update``.
* check_host_contract: nothing calls ``check_host`` for the merge, so the contract
  does not claim that a gap blocks it.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from .test_restart_on_merge_review_omn24677 import SETTINGS, rom

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
CONTRACT = (
    REPO_ROOT / "src/omnibase_infra/nodes/node_restart_on_merge_effect/contract.yaml"
)


# --- finding 2: the rollback restart under the start limit ---------------------------


def _start_limited_systemctl(bin_dir: Path, marker: Path) -> Path:
    """A ``systemctl`` whose unit enters the failed state after its first restart.

    The new revision dies at start until the start limit trips; from then on a
    ``restart`` exits 1 until ``reset-failed`` runs (measured with systemd 255.4 under
    StartLimitBurst=5), and a failed unit is not active.
    """
    log = bin_dir.parent / "limited.log"
    failed = bin_dir.parent / "failed"
    started = bin_dir.parent / "started"
    systemctl = bin_dir / "systemctl"
    systemctl.write_text(
        "#!/bin/bash\n"
        f'echo "systemctl $*" >> {log}\n'
        'case "$*" in\n'
        f"  *reset-failed*) rm -f {failed}; exit 0;;\n"
        f"  *is-active*) [ -f {failed} ] && exit 3; exit 0;;\n"
        "  *NRestarts*) echo 0; exit 0;;\n"
        "  *restart*)\n"
        f'    if [ -f {failed} ]; then echo "restart refused: start limit" >&2; exit 1; fi\n'
        f'    echo "marker=$([ -f {marker} ] && echo yes || echo no)" >> {log}\n'
        f"    if [ ! -f {started} ]; then touch {started} {failed}; fi\n"
        "    exit 0;;\n"
        "esac\nexit 0\n",
        "utf-8",
    )
    systemctl.chmod(0o755)
    return log


def _run_script(fake_systemd: SimpleNamespace, tmp_path: Path, marker: Path) -> int:
    script = rom.transient_restart_script(
        "agent.service",
        installed=fake_systemd.installed,
        agent_dir=str(tmp_path / "agent"),
        loaded_sha="a" * 40,
        settle=0,
        refused_marker=marker,
    )
    env = {
        **os.environ,
        "PATH": f"{fake_systemd.bin_dir}{os.pathsep}{os.environ['PATH']}",
    }
    return subprocess.run(["bash", "-c", script], env=env, check=False).returncode


def test_restart_on_merge_rollback_reset_failed_starts_the_previous_revision(
    fake_systemd: SimpleNamespace, tmp_path: Path
) -> None:
    marker = tmp_path / "state" / rom.ROLLBACK_MARKER
    log = _start_limited_systemctl(fake_systemd.bin_dir, marker)
    assert _run_script(fake_systemd, tmp_path, marker) == rom.ROLLED_BACK
    calls = log.read_text("utf-8").splitlines()
    assert "systemctl --user reset-failed agent.service" in calls
    assert marker.read_text("utf-8") == "b" * 40


def test_restart_on_merge_rollback_reset_failed_runs_before_every_restart(
    fake_systemd: SimpleNamespace, tmp_path: Path
) -> None:
    marker = tmp_path / "state" / rom.ROLLBACK_MARKER
    log = _start_limited_systemctl(fake_systemd.bin_dir, marker)
    _run_script(fake_systemd, tmp_path, marker)
    calls = [
        c
        for c in log.read_text("utf-8").splitlines()
        if "restart" in c or "reset-failed" in c
    ]
    restarts = [i for i, c in enumerate(calls) if c.endswith("restart agent.service")]
    assert len(restarts) == 2
    for index in restarts:
        assert calls[index - 1] == "systemctl --user reset-failed agent.service"


# --- finding 3: EnvironmentFile= drops an exported name -------------------------------


def _unit_world(tmp_path: Path, *, launcher_store: bool) -> tuple[Path, Path]:
    """One agent unit whose settings come from one env file, read by ``source``."""
    repo = tmp_path / "repo"
    (repo / "deploy").mkdir(parents=True)
    (repo / "units").mkdir()
    store = tmp_path / "store.env"
    source = (
        f"Environment=DEPLOY_AGENT_ENV_FILE={store}\n"
        if launcher_store
        else f"EnvironmentFile={store}\n"
    )
    (repo / "units" / "agent.service").write_text(
        "Environment=DEPLOY_AGENT_TRACKING_REF=dev\n"
        "Environment=DEPLOY_AGENT_ALLOWED_LANES=prod\n" + source,
        "utf-8",
    )
    (repo / "deploy" / "unit-drift-manifest.yaml").write_text(
        "units:\n  - name: agent.service\n    tracked: units/agent.service\n"
        "    installed: ~/.config/systemd/user/agent.service\n    hosts: [h]\n",
        "utf-8",
    )
    return repo, store


def _gap(repo: Path, home: Path) -> object:
    entries = rom.as_agent_units(
        [
            {"name": "agent.service", "tracked": "units/agent.service"},
        ]
    )
    return rom.settings_gap(entries[0], repo, home)


def test_restart_on_merge_check_host_environment_file_drops_exported_names(
    tmp_path: Path,
) -> None:
    repo, store = _unit_world(tmp_path, launcher_store=False)
    store.write_text(
        "".join(f"export {k}='{v}'\n" for k, v in SETTINGS.items()), "utf-8"
    )
    gap = _gap(repo, tmp_path)
    assert set(gap.missing) == {"DEPLOY_AGENT_REPO_DIR", "DEPLOY_AGENT_CANONICAL_CLONE"}


def test_restart_on_merge_check_host_environment_file_counts_bare_assignments(
    tmp_path: Path,
) -> None:
    repo, store = _unit_world(tmp_path, launcher_store=False)
    store.write_text("".join(f"{k}={v}\n" for k, v in SETTINGS.items()), "utf-8")
    assert _gap(repo, tmp_path).missing == ()


def test_restart_on_merge_check_host_launcher_store_still_counts_exported_names(
    tmp_path: Path,
) -> None:
    repo, store = _unit_world(tmp_path, launcher_store=True)
    store.write_text(
        "".join(f"export {k}='{v}'\n" for k, v in SETTINGS.items()), "utf-8"
    )
    assert _gap(repo, tmp_path).missing == ()


# --- finding 5: a failed ledger append never raises out of self_update ---------------


def _raising_execution(tmp_path: Path, error: BaseException) -> SimpleNamespace:
    def run(argv: list[str], timeout: int) -> object:
        raise error

    return SimpleNamespace(
        dev_boot_report_config=lambda: SimpleNamespace(
            plugin_dir=tmp_path / "plugin",
            ledger_via_bus=True,
            registry_root=tmp_path / "registry",
        ),
        _run=run,
    )


@pytest.mark.parametrize(
    "error",
    [
        subprocess.TimeoutExpired(cmd="onex_ledger.py", timeout=60),
        FileNotFoundError("interpreter missing"),
    ],
    ids=["timeout", "missing-interpreter"],
)
def test_restart_on_merge_refusal_append_failure_is_logged_and_leaves_no_marker(
    tmp_path: Path, error: BaseException, caplog: pytest.LogCaptureFixture
) -> None:
    execution = _raising_execution(tmp_path, error)
    env = {"DEPLOY_AGENT_STATE_DIR": str(tmp_path), "ONEX_LEDGER_HOST_NAME": "h"}
    rom.write_refusal_status(execution, "a reason", env)
    assert not (tmp_path / rom.REFUSAL_MARKER).exists()
    assert "refusal STATUS append failed" in caplog.text


def test_restart_on_merge_refusal_append_failure_does_not_stop_the_reexec_check(
    tmp_path: Path,
) -> None:
    execution = _raising_execution(
        tmp_path, subprocess.TimeoutExpired(cmd="onex_ledger.py", timeout=60)
    )
    env = {
        "DEPLOY_AGENT_ALLOWED_LANES": "dev",
        "DEPLOY_AGENT_STATE_DIR": str(tmp_path),
    }
    assert rom.reexec_blocked_by_settings(execution, env) is True
    assert not (tmp_path / rom.REEXEC_REFUSAL_MARKER).exists()


def test_restart_on_merge_refusal_append_failure_does_not_stop_the_rollback_skip(
    tmp_path: Path,
) -> None:
    """``self_update`` appends its rollback-skip row outside its try."""
    execution = _raising_execution(
        tmp_path, subprocess.TimeoutExpired(cmd="onex_ledger.py", timeout=60)
    )
    env = {"DEPLOY_AGENT_STATE_DIR": str(tmp_path)}
    marker = tmp_path / rom.ROLLBACK_MARKER
    marker.write_text("c" * 40, "utf-8")
    assert rom.skip_refused_revision(execution, "c" * 40, env) is True
    assert not (tmp_path / rom.ROLLBACK_STATUS_MARKER).exists()


# --- finding 4: no caller, so the contract does not say a gap blocks the merge -------


def test_restart_on_merge_check_host_contract_does_not_claim_a_blocking_caller() -> (
    None
):
    contract = yaml.safe_load(CONTRACT.read_text("utf-8"))
    text = " ".join(
        [contract["description"], *(c["description"] for c in contract["capabilities"])]
    )
    assert "blocks the merge" not in text
    assert "blocking per-host check" not in text
    assert "run by the operator" in text
