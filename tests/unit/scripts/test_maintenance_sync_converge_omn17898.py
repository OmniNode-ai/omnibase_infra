# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17898 -- the host-maintenance sync must CONVERGE, not only detect.

WHAT IS UNDER TEST
    ``deploy/maintenance/omninode-host-maintenance-sync.sh --converge`` and the
    cron unit that invokes it. These tests drive the artifact that actually
    runs on the host, not a re-implementation (memory
    ``feedback_test_the_artifact_that_runs``).

WHY IT EXISTS
    OMN-15525 built the detector and OMN-17311 scheduled it, but the scheduled
    invocation was ``--check --slack``: drift was detected and alarmed on
    forever and never repaired, because ``--install`` is reachable only by an
    operator typing it. Measured on 2026-09-16: ``omnibase_infra#3629``
    (``26e734f6``) fixed a false ``runtime-prod-28085`` CRITICAL, merged, and
    the live host kept posting the false alert because the installed copy of
    the monitor was never refreshed -- a live ``--check`` on ``.201`` reported
    ``drifted=1 missing=0 checked=7``.

    Per CLAUDE.md rule 5, detection that is not wired to a repair is advisory.
    ``--converge`` is the repair, and the load-bearing assertions below are the
    EXIT CODE and the resulting on-disk bytes, never the printed prose.

WHY NOT JUST SCHEDULE ``--install``
    ``--install`` writes every manifest entry on every tick whether or not it
    drifted, and reports nothing about what it changed -- so a scheduled
    ``--install`` overwrites host state hourly with no receipt, which is the
    objection the previous revision of this unit's test recorded and was right
    about. ``--converge`` writes ONLY entries that differ from the ref, reads
    each written file back, and receipts every entry with its before and after
    sha256-12. A converge that writes nothing prints the same receipt as a
    converge that writes everything, so "nothing drifted" and "nothing ran"
    are distinguishable from the output alone.

HERMETICITY
    Each test builds a throwaway git repo as the "infra clone" and points the
    manifest at temp paths via ``OMNINODE_MAINTENANCE_SYNC_MANIFEST``, so no
    test reads or writes a real ``/data/maintenance`` or ``/etc/cron.d`` path.
    ``OMNINODE_MAINTENANCE_SYNC_SKIP_FETCH=1`` keeps the network out.
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
SECOND_REL = "deploy/maintenance/omninode-workspace-reconcile.sh"
SECOND_BODY = "#!/usr/bin/env bash\necho second\n"

pytestmark = pytest.mark.unit


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
    """A git repo with an `origin/dev` ref carrying two tracked artifacts."""
    repo = tmp_path / "infra-clone"
    (repo / "deploy" / "maintenance").mkdir(parents=True)
    _git(repo.parent, "init", "--quiet", "-b", "dev", str(repo))
    _git(repo, "config", "user.email", "test@omninode.ai")
    _git(repo, "config", "user.name", "test")
    (repo / TRACKED_REL).write_text(TRACKED_BODY)
    (repo / SECOND_REL).write_text(SECOND_BODY)
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "--no-gpg-sign", "-m", "seed")
    _git(repo, "update-ref", "refs/remotes/origin/dev", "HEAD")
    return repo


def _slack_free_env() -> dict[str, str]:
    """The ambient environment MUST NOT reach the script under test.

    A developer shell and a CI runner both commonly export SLACK_BOT_TOKEN and
    SLACK_DEFAULT_CHANNEL. The script falls back to the process environment
    when its env file is absent, so a `--slack` test that inherits os.environ
    posts a real message to a real channel -- measured once, on
    2026-09-16T11:54:57Z, while these very tests were being written: the
    failure-path case paged #omninode-notifications with a pytest tmp path in
    the body. Scrubbing here is what makes the alert assertions below provable
    from stderr instead of from a channel nobody wants paged by a test run.
    """
    return {k: v for k, v in os.environ.items() if not k.startswith("SLACK_")}


def _run(
    clone: Path,
    manifest: Path,
    tmp_path: Path,
    *args: str,
    env_file: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = _slack_free_env()
    env.update(
        {
            "OMNINODE_INFRA_REPO_ROOT": str(clone),
            "OMNINODE_MAINTENANCE_SYNC_MANIFEST": str(manifest),
            "OMNINODE_MAINTENANCE_SYNC_SKIP_FETCH": "1",
            "OMNINODE_ALERT_ENV_FILE": str(env_file or (tmp_path / "absent.env")),
        }
    )
    return subprocess.run(
        ["bash", str(SYNC_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )


def _manifest(tmp_path: Path, *entries: tuple[str, Path]) -> Path:
    path = tmp_path / "manifest.txt"
    path.write_text("".join(f"{rel}|{host}|0755\n" for rel, host in entries))
    return path


# --------------------------------------------------------------------------
# The drift -> install path
# --------------------------------------------------------------------------


def test_converge_repairs_a_drifted_artifact(tmp_path: Path, fake_clone: Path) -> None:
    """The whole point: drift is repaired, not merely reported.

    RED before the fix -- `--converge` is an unknown argument and exits 2.
    """
    hostpath = tmp_path / "installed.sh"
    hostpath.write_text(TRACKED_BODY + "# hand-edited on the host\n")

    proc = _run(
        fake_clone, _manifest(tmp_path, (TRACKED_REL, hostpath)), tmp_path, "--converge"
    )

    assert proc.returncode == 0, (
        f"a converge that repaired the drift must exit 0:\n{proc.stdout}{proc.stderr}"
    )
    assert hostpath.read_text() == TRACKED_BODY, (
        "the host copy still differs from origin/dev after --converge -- "
        "convergence that does not converge is the OMN-15525 condition again"
    )
    assert "converged=1" in proc.stdout, proc.stdout
    assert "failed=0" in proc.stdout, proc.stdout


def test_converge_receipts_the_before_and_after_sha(
    tmp_path: Path, fake_clone: Path
) -> None:
    """Every write names what it replaced and what it wrote (sha256-12).

    Without this the cron log records that something changed but not what, so
    an operator cannot tell a good repair from a bad one after the fact.
    """
    import hashlib

    drifted = TRACKED_BODY + "# hand-edited on the host\n"
    hostpath = tmp_path / "installed.sh"
    hostpath.write_text(drifted)
    before = hashlib.sha256(drifted.encode()).hexdigest()[:12]
    after = hashlib.sha256(TRACKED_BODY.encode()).hexdigest()[:12]

    proc = _run(
        fake_clone, _manifest(tmp_path, (TRACKED_REL, hostpath)), tmp_path, "--converge"
    )

    assert f"before={before}" in proc.stdout, (
        f"receipt does not name the replaced sha {before}:\n{proc.stdout}"
    )
    assert f"after={after}" in proc.stdout, (
        f"receipt does not name the written sha {after}:\n{proc.stdout}"
    )
    assert f"CONVERGED|{hostpath}|" in proc.stdout, proc.stdout


def test_converge_installs_a_missing_artifact(tmp_path: Path, fake_clone: Path) -> None:
    """Never-installed converges too, and its receipt says so rather than
    printing a sha it does not have."""
    hostpath = tmp_path / "never-installed.sh"

    proc = _run(
        fake_clone, _manifest(tmp_path, (TRACKED_REL, hostpath)), tmp_path, "--converge"
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert hostpath.read_text() == TRACKED_BODY, proc.stdout
    assert "before=absent" in proc.stdout, proc.stdout
    assert "converged=1" in proc.stdout, proc.stdout


def test_converge_does_not_rewrite_an_in_sync_artifact(
    tmp_path: Path, fake_clone: Path
) -> None:
    """An in-sync entry is left ALONE -- this is what separates --converge from
    a scheduled --install, which rewrites every entry on every tick."""
    hostpath = tmp_path / "installed.sh"
    hostpath.write_text(TRACKED_BODY)
    before_inode = hostpath.stat().st_ino

    proc = _run(
        fake_clone, _manifest(tmp_path, (TRACKED_REL, hostpath)), tmp_path, "--converge"
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert hostpath.stat().st_ino == before_inode, (
        "an in-sync artifact was replaced anyway; --converge must not churn "
        "host state on a tick with nothing to do"
    )
    assert "converged=0" in proc.stdout, proc.stdout
    assert f"OK|{hostpath}" in proc.stdout, proc.stdout


def test_converge_mode_is_named_in_the_summary(
    tmp_path: Path, fake_clone: Path
) -> None:
    """A converge tick and a check tick must be distinguishable in the log."""
    hostpath = tmp_path / "installed.sh"
    hostpath.write_text(TRACKED_BODY)

    proc = _run(
        fake_clone, _manifest(tmp_path, (TRACKED_REL, hostpath)), tmp_path, "--converge"
    )

    assert "mode=converge" in proc.stdout, proc.stdout


# --------------------------------------------------------------------------
# The failure path
# --------------------------------------------------------------------------


def test_converge_failure_is_red_and_receipted(
    tmp_path: Path, fake_clone: Path
) -> None:
    """A write that cannot land must FAIL the run and say so per file.

    The host path's parent does not exist, so the write is refused for root
    and non-root alike -- the test does not depend on who runs it.
    """
    unwritable = tmp_path / "no-such-dir" / "installed.sh"

    proc = _run(
        fake_clone,
        _manifest(tmp_path, (TRACKED_REL, unwritable)),
        tmp_path,
        "--converge",
    )

    assert proc.returncode == 1, (
        f"a failed converge exited {proc.returncode}; a repair that silently "
        f"does not repair is worse than no repair:\n{proc.stdout}{proc.stderr}"
    )
    assert "CONVERGE FAILED" in proc.stdout, proc.stdout
    assert "failed=1" in proc.stdout, proc.stdout


def test_a_failed_entry_does_not_abort_the_remaining_entries(
    tmp_path: Path, fake_clone: Path
) -> None:
    """Fail-closed, but with a COMPLETE receipt.

    Dying on the first unwritable path would leave the operator without any
    statement about the other artifacts -- which is the ambiguity ("did it
    fail, or did nobody run it?") this whole surface exists to remove.
    """
    unwritable = tmp_path / "no-such-dir" / "first.sh"
    writable = tmp_path / "second.sh"

    proc = _run(
        fake_clone,
        _manifest(tmp_path, (TRACKED_REL, unwritable), (SECOND_REL, writable)),
        tmp_path,
        "--converge",
    )

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert writable.read_text() == SECOND_BODY, (
        "the second entry was never attempted; the run aborted on the first "
        f"failure:\n{proc.stdout}{proc.stderr}"
    )
    assert "failed=1" in proc.stdout, proc.stdout
    assert "converged=1" in proc.stdout, proc.stdout


def test_converge_refuses_to_write_when_the_blob_is_absent_at_the_ref(
    tmp_path: Path, fake_clone: Path
) -> None:
    """ "Could not determine" is never "fine", and it is never a reason to write.

    A manifest entry with no blob at the ref must redden and leave the host
    copy exactly as it found it.
    """
    hostpath = tmp_path / "installed.sh"
    hostpath.write_text("# a local file that must survive\n")

    proc = _run(
        fake_clone,
        _manifest(tmp_path, ("deploy/maintenance/does-not-exist.sh", hostpath)),
        tmp_path,
        "--converge",
    )

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert hostpath.read_text() == "# a local file that must survive\n", (
        "the host copy was touched for an entry the ref does not carry"
    )
    assert "absent at" in proc.stdout, proc.stdout
    assert "failed=1" in proc.stdout, proc.stdout


def test_unresolvable_repo_root_is_fatal_in_converge_mode(tmp_path: Path) -> None:
    """No clone means no known-good bytes -- fatal, and nothing is written."""
    hostpath = tmp_path / "installed.sh"
    hostpath.write_text("# untouched\n")

    proc = _run(
        tmp_path / "no-such-clone",
        _manifest(tmp_path, (TRACKED_REL, hostpath)),
        tmp_path,
        "--converge",
    )

    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert hostpath.read_text() == "# untouched\n", proc.stdout


# --------------------------------------------------------------------------
# The alert is for failures only
# --------------------------------------------------------------------------
#
# The Slack block is reached or it is not; with no token configured it says so
# on stderr, and that warning is the observable proof the block was entered.
# This keeps the assertion behavioural without a network call.

_SLACK_ATTEMPTED = "--slack requested but no SLACK_BOT_TOKEN"


def test_a_successful_converge_does_not_alert(tmp_path: Path, fake_clone: Path) -> None:
    """Repairing drift is the job, not an incident.

    Alerting on every successful self-heal is how a channel becomes noise and
    a real failure goes unread.
    """
    hostpath = tmp_path / "installed.sh"
    hostpath.write_text(TRACKED_BODY + "# drifted\n")

    proc = _run(
        fake_clone,
        _manifest(tmp_path, (TRACKED_REL, hostpath)),
        tmp_path,
        "--converge",
        "--slack",
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _SLACK_ATTEMPTED not in proc.stderr, (
        f"a successful converge tried to post to Slack:\n{proc.stderr}"
    )


def test_a_failed_converge_alerts(tmp_path: Path, fake_clone: Path) -> None:
    """A converge that could not repair is exactly what a human must see."""
    unwritable = tmp_path / "no-such-dir" / "installed.sh"

    proc = _run(
        fake_clone,
        _manifest(tmp_path, (TRACKED_REL, unwritable)),
        tmp_path,
        "--converge",
        "--slack",
    )

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert _SLACK_ATTEMPTED in proc.stderr, (
        f"a failed converge did not reach the Slack block:\n{proc.stderr}"
    )


# --------------------------------------------------------------------------
# The scheduled unit
# --------------------------------------------------------------------------


def _cron_command_lines(unit: str) -> list[str]:
    """The schedule lines only.

    Asserting against the whole file would make the unit's own comments part
    of the contract: a comment EXPLAINING why `--install` is not scheduled
    would fail a naive "--install not in unit" check. Per CLAUDE.md rule 15,
    prose that merely names a token must never be what a matcher fires on.
    """
    return [
        line
        for line in unit.splitlines()
        if line.strip() and not line.startswith(("#", "SHELL=", "PATH="))
    ]


def test_cron_unit_converges_rather_than_only_checking() -> None:
    """The scheduled tick must repair. Detection alone is what OMN-17898 is."""
    commands = _cron_command_lines(SYNC_CRON.read_text())

    assert len(commands) == 1, commands
    command = commands[0]

    assert "omninode-host-maintenance-sync.sh" in command, command
    assert "--converge" in command, (
        "the hourly unit still only detects; drift is then alarmed on forever "
        f"and never repaired (OMN-17898): {command}"
    )
    assert "--slack" in command, command
    assert "--install" not in command, (
        "--install rewrites every manifest entry on every tick with no "
        f"before/after receipt; --converge writes only what drifted: {command}"
    )


def test_the_unit_runs_as_root_and_the_script_never_sudos() -> None:
    """Privilege is declared by the scheduler, never taken by the script.

    /data/maintenance/bin and /etc/cron.d are root-owned, so the writes need
    root -- and the cron unit already supplies it in its user field. A `sudo`
    inside the script would be a second, invisible privilege rule that works
    from cron and prompts from a terminal.
    """
    unit = SYNC_CRON.read_text()
    command_lines = _cron_command_lines(unit)
    assert command_lines, unit
    for line in command_lines:
        assert line.split()[5] == "root", (
            f"the converge writes root-owned paths; this line does not run as "
            f"root: {line}"
        )

    source = SYNC_SCRIPT.read_text()
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert "sudo " not in stripped, (
            f"the script escalates its own privilege: {stripped}"
        )
