# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-15525 -- the host-maintenance drift detector must actually fail.

WHAT IS UNDER TEST
    ``deploy/maintenance/omninode-host-maintenance-sync.sh`` -- the artifact
    root runs hourly from ``/etc/cron.d/omninode-host-maintenance-sync`` on
    ``.201``. These tests drive that file itself, not a re-implementation
    (memory ``feedback_test_the_artifact_that_runs``).

WHY IT EXISTS
    ``omnibase_infra#2572`` merged a fix for the system-health reporter and
    changed nothing about what the platform alarmed on, because nothing
    installed or checked the copy at ``/data/maintenance/bin/``. Per CLAUDE.md
    rule 5 and ``feedback_a_rule_is_not_a_mechanism``, a runbook step is not
    enforcement -- the check has to fail something. So the load-bearing
    assertion in every test below is the EXIT CODE, not the printed text.

HERMETICITY
    Each test builds a throwaway git repo as the "infra clone" and points the
    manifest at temp paths via ``OMNINODE_MAINTENANCE_SYNC_MANIFEST``, so no
    test ever reads or writes a real ``/data/maintenance`` or ``/etc/cron.d``
    path. ``OMNINODE_MAINTENANCE_SYNC_SKIP_FETCH=1`` keeps the network out.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SYNC_SCRIPT = REPO_ROOT / "deploy" / "maintenance" / "omninode-host-maintenance-sync.sh"
SYNC_CRON = (
    REPO_ROOT / "deploy" / "maintenance" / "cron.d" / "omninode-host-maintenance-sync"
)

TRACKED_REL = "deploy/maintenance/omninode-system-slack-report.sh"
TRACKED_BODY = "#!/usr/bin/env bash\necho canonical\n"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
        },
    )


@pytest.fixture
def fake_clone(tmp_path: Path) -> Path:
    """A git repo with an `origin/dev` ref carrying the tracked artifact."""
    repo = tmp_path / "infra-clone"
    (repo / "deploy" / "maintenance").mkdir(parents=True)
    _git(repo.parent, "init", "--quiet", "-b", "dev", str(repo))
    _git(repo, "config", "user.email", "test@omninode.ai")
    _git(repo, "config", "user.name", "test")
    (repo / TRACKED_REL).write_text(TRACKED_BODY)
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "--no-gpg-sign", "-m", "seed")
    # The script compares against `origin/dev`; create that remote-tracking ref
    # locally so no network is required.
    _git(repo, "update-ref", "refs/remotes/origin/dev", "HEAD")
    return repo


def _run_check(
    clone: Path, manifest: Path, tmp_path: Path
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.update(
        {
            "OMNINODE_INFRA_REPO_ROOT": str(clone),
            "OMNINODE_MAINTENANCE_SYNC_MANIFEST": str(manifest),
            "OMNINODE_MAINTENANCE_SYNC_SKIP_FETCH": "1",
            "OMNINODE_ALERT_ENV_FILE": str(tmp_path / "absent.env"),
        }
    )
    return subprocess.run(
        ["bash", str(SYNC_SCRIPT), "--check"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )


def _manifest(tmp_path: Path, hostpath: Path) -> Path:
    path = tmp_path / "manifest.txt"
    path.write_text(f"{TRACKED_REL}|{hostpath}|0755\n")
    return path


def test_in_sync_host_file_passes(tmp_path: Path, fake_clone: Path) -> None:
    """Control: identical content must exit 0, or every RED below is meaningless."""
    hostpath = tmp_path / "installed.sh"
    hostpath.write_text(TRACKED_BODY)
    proc = _run_check(fake_clone, _manifest(tmp_path, hostpath), tmp_path)

    assert proc.returncode == 0, f"in-sync check failed: {proc.stdout}{proc.stderr}"
    assert "drifted=0" in proc.stdout, proc.stdout
    assert f"OK|{hostpath}" in proc.stdout, proc.stdout


def test_drifted_host_file_is_red(tmp_path: Path, fake_clone: Path) -> None:
    """AC5: a host copy that differs from origin/dev must FAIL, not just log."""
    hostpath = tmp_path / "installed.sh"
    hostpath.write_text(TRACKED_BODY + "# hand-edited on the host\n")
    proc = _run_check(fake_clone, _manifest(tmp_path, hostpath), tmp_path)

    assert proc.returncode == 1, (
        f"drift did not fail the check -- a detector that exits 0 enforces "
        f"nothing (rule 5):\n{proc.stdout}{proc.stderr}"
    )
    assert "DRIFT" in proc.stdout, proc.stdout
    assert "drifted=1" in proc.stdout, proc.stdout


def test_missing_host_file_is_red(tmp_path: Path, fake_clone: Path) -> None:
    """Fail-closed: never-installed is a failure, not an absence of evidence."""
    hostpath = tmp_path / "never-installed.sh"
    proc = _run_check(fake_clone, _manifest(tmp_path, hostpath), tmp_path)

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "NOT INSTALLED" in proc.stdout, proc.stdout
    assert "missing=1" in proc.stdout, proc.stdout


def test_artifact_absent_from_the_ref_is_red(tmp_path: Path, fake_clone: Path) -> None:
    """A manifest entry with no blob at origin/dev must fail, not be skipped."""
    hostpath = tmp_path / "installed.sh"
    hostpath.write_text(TRACKED_BODY)
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(f"deploy/maintenance/does-not-exist.sh|{hostpath}|0755\n")
    proc = _run_check(fake_clone, manifest, tmp_path)

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "absent at" in proc.stdout, proc.stdout


def test_unresolvable_repo_root_is_fatal(tmp_path: Path) -> None:
    """No clone means the check CANNOT know -- that is fatal, never green."""
    hostpath = tmp_path / "installed.sh"
    hostpath.write_text(TRACKED_BODY)
    proc = _run_check(
        tmp_path / "no-such-clone", _manifest(tmp_path, hostpath), tmp_path
    )

    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "no git clone" in proc.stderr, proc.stderr


def test_live_report_script_is_governed_by_the_manifest() -> None:
    """The artifact this whole ticket is about must be in the default manifest.

    OMN-15525's root cause was an artifact that no install path covered. An
    entry missing from the built-in manifest reproduces exactly that.
    """
    source = SYNC_SCRIPT.read_text()
    for hostpath in (
        "/data/maintenance/bin/omninode-system-slack-report.sh",
        "/etc/cron.d/omninode-system-slack-report",
        # The detector must govern itself, or it can silently rot too.
        "/data/maintenance/bin/omninode-host-maintenance-sync.sh",
        "/etc/cron.d/omninode-host-maintenance-sync",
    ):
        assert hostpath in source, f"{hostpath} is not governed by the sync manifest"


def test_cron_unit_runs_the_check_and_can_alert() -> None:
    unit = SYNC_CRON.read_text()
    assert "omninode-host-maintenance-sync.sh" in unit, unit
    assert "--check" in unit, unit
    assert "--slack" in unit, unit
    # --install from cron would silently overwrite host state on every tick.
    assert "--install" not in unit, (
        "the scheduled unit must DETECT drift, not auto-overwrite host artifacts"
    )
