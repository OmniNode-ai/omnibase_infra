# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The reconcile-host lock records its holder and reclaims a dead one (OMN-18608).

A bare ``mkdir`` lock released only by a shell trap is leaked permanently by a
``SIGKILL``, and a directory with nothing in it cannot tell a live holder from a
corpse. Measured on the workstation on 2026-09-17: a session died at about
15:25Z holding a lock it had taken at 15:23Z, and every tick for the next 35
minutes printed ``nothing to do`` and exited 0 while ``ps`` showed no reconcile
process anywhere. The tick's own receipt line read ``verdict="in sync"`` on runs
that did no work.

Every test here decides the outcome by SEEDING the lock and reading what the
script then does, never by asserting on the script's text. The pairs matter more
than the individual cases: a reclaim test with no "live holder is respected"
control would pass just as well against a lock that had been deleted outright,
which would trade a stuck host for a corrupted one.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

import pytest

from tests.scripts.test_reconcile_host_omn17307 import (
    EXIT_OK,
    Workspace,
    _lock,
    _run,
    _stub,
    _write_dist,
    build_workspace,
)

pytestmark = pytest.mark.unit

_NOTHING_TO_DO = "nothing to do"
_RECLAIMING = "RECLAIMING a stale reconcile-host lock"


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    return build_workspace(tmp_path)


def _ready(ws: Workspace, *, body: str = "") -> None:
    """Stub both delegates and give the run a coherent surface to report on.

    The property under test is which side of the lock decision the script takes,
    so the delegates are stubbed and their behaviour is deliberately not part of
    any assertion here.
    """
    _lock(ws, **{"omnibase-core": "0.46.9"})
    _write_dist(ws.site_packages, "omnibase_core", "0.46.9")
    _stub(
        ws.scripts / "runtime_build" / "reconcile_deploy_clones.sh",
        ws.delegate_witness,
    )
    _stub(ws.scripts / "reconcile-workspace-venvs.sh", ws.delegate_witness, body=body)


def _seed_lock(
    ws: Workspace,
    *,
    pid: int | None = None,
    host: str | None = None,
    age_seconds: int = 0,
    holder_text: str | None = None,
) -> Path:
    """Put a lock on disk as some other process would have left it."""
    lock_dir = ws.root / ".onex-reconcile-host.lock"
    lock_dir.mkdir(parents=True)
    holder = lock_dir / "holder"
    if holder_text is not None:
        holder.write_text(holder_text, encoding="utf-8")
    elif pid is not None and host is not None:
        holder.write_text(
            f"pid={pid}\nhost={host}\nstarted_at=2026-09-17T15:23:00Z\n",
            encoding="utf-8",
        )
    if age_seconds:
        old = time.time() - age_seconds
        target = holder if holder.exists() else lock_dir
        os.utime(target, (old, old))
    return lock_dir


def _dead_pid() -> int:
    """A pid that certainly is not running: spawned, exited, reaped."""
    proc = subprocess.Popen(["true"], start_new_session=True)
    proc.wait()
    return proc.pid


# --------------------------------------------------------------------------- #
# AC1 -- the lock says who holds it and since when
# --------------------------------------------------------------------------- #
def test_the_lock_records_its_holder_while_it_is_held(ws: Workspace) -> None:
    """Read the holder DURING the run, which is the only time it exists.

    The trap removes the lock on exit, so asserting after the process returns
    would assert about an absence. The venv delegate stub copies the holder out
    while the script is standing on it.
    """
    snapshot = ws.root.parent / "holder-snapshot"
    _ready(ws, body=f'cp "$OMNI_HOME/.onex-reconcile-host.lock/holder" {snapshot}')

    proc = _run(ws)

    assert snapshot.exists(), (
        f"no holder record existed while the lock was held: {proc.stdout}{proc.stderr}"
    )
    recorded = dict(
        line.split("=", 1)
        for line in snapshot.read_text(encoding="utf-8").splitlines()
        if "=" in line
    )
    assert recorded["pid"].isdigit()
    assert recorded["host"] == os.uname().nodename
    # ISO-8601 to the second, UTC. A start time nobody can parse is not a start
    # time, and the age bound below is meaningless without it.
    time.strptime(recorded["started_at"], "%Y-%m-%dT%H:%M:%SZ")


def test_the_lock_is_removed_when_the_run_finishes(ws: Workspace) -> None:
    """Control for the test above: the holder file must not defeat the cleanup.

    The lock gained a file inside it, so a `rmdir` cleanup would now silently
    fail and leave exactly the stale lock this ticket is about.
    """
    _ready(ws)

    _run(ws)

    assert not (ws.root / ".onex-reconcile-host.lock").exists()


# --------------------------------------------------------------------------- #
# AC3 -- a dead holder is reclaimed, a live one is respected
# --------------------------------------------------------------------------- #
def test_a_lock_left_by_a_dead_pid_is_reclaimed(ws: Workspace) -> None:
    dead = _dead_pid()
    _seed_lock(ws, pid=dead, host=os.uname().nodename)
    _ready(ws)

    proc = _run(ws)

    assert _NOTHING_TO_DO not in proc.stderr, (
        "a lock whose holder is not running still stopped the tick, which is "
        f"the 35-minute outage this ticket exists to end: {proc.stderr}"
    )
    assert ws.delegate_witness.exists(), "the reconcile did not actually proceed"


def test_a_lock_held_by_a_live_pid_is_respected(ws: Workspace) -> None:
    """The positive control, and the more important half.

    Without it, "reclaims a dead lock" would pass against a script that simply
    deleted any lock it found, turning a stuck host into two concurrent writers
    -- which is the OMN-15590 stall shape the lock exists to prevent.
    """
    live = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        _seed_lock(ws, pid=live.pid, host=os.uname().nodename)
        _ready(ws)

        proc = _run(ws)

        assert proc.returncode == EXIT_OK
        assert _NOTHING_TO_DO in proc.stderr
        assert not ws.delegate_witness.exists(), (
            "the tick reconciled while a live peer held the lock"
        )
    finally:
        live.terminate()
        live.wait()


# --------------------------------------------------------------------------- #
# AC4 -- a reclaim is reported, never silent
# --------------------------------------------------------------------------- #
def test_a_reclaim_names_the_pid_it_overrode_and_the_lock_age(ws: Workspace) -> None:
    """A silent reclaim would hide a genuine concurrency bug behind self-healing.

    The whole failure this ticket fixes was invisible because a tick reported
    success while doing nothing; a reclaim that said nothing would be the same
    mistake pointed the other way.
    """
    dead = _dead_pid()
    _seed_lock(ws, pid=dead, host=os.uname().nodename, age_seconds=120)
    _ready(ws)

    proc = _run(ws)

    assert _RECLAIMING in proc.stderr
    assert str(dead) in proc.stderr, (
        f"the reclaim did not name the pid it overrode: {proc.stderr}"
    )
    assert re.search(r"\b12[0-5]s old\b", proc.stderr), (
        f"the reclaim did not name the age of the lock it broke: {proc.stderr}"
    )


# --------------------------------------------------------------------------- #
# AC2 -- a holder record nobody can read is bounded by age, not trusted forever
# --------------------------------------------------------------------------- #
def test_a_malformed_holder_record_is_respected_inside_the_age_bound(
    ws: Workspace,
) -> None:
    """The `mkdir`-to-holder window of a peer that is very much alive.

    A lock taken microseconds ago has no holder file yet. Treating that as
    stale would race every concurrent tick straight past the lock.
    """
    _seed_lock(ws, holder_text="this is not a holder record\n")
    _ready(ws)

    proc = _run(ws)

    assert proc.returncode == EXIT_OK
    assert _NOTHING_TO_DO in proc.stderr
    assert not ws.delegate_witness.exists()


def test_a_malformed_holder_record_is_reclaimed_past_the_age_bound(
    ws: Workspace,
) -> None:
    """Paired with the test above: unreadable must not mean untouchable.

    Without the bound, a lock whose holder record failed to write would pin the
    host permanently with nothing on disk to explain why.
    """
    _seed_lock(ws, holder_text="this is not a holder record\n", age_seconds=7200)
    _ready(ws)

    proc = _run(ws)

    assert _RECLAIMING in proc.stderr
    assert "no readable holder record" in proc.stderr
    assert ws.delegate_witness.exists()


def test_an_empty_lock_directory_is_reclaimed_past_the_age_bound(
    ws: Workspace,
) -> None:
    """The exact shape of the 2026-09-17 leak: a lock with nothing in it at all.

    Every lock taken before this change looks like this, so an upgrade that
    could not reclaim one would leave the defect in place on the hosts that
    already have it.
    """
    _seed_lock(ws, age_seconds=7200)
    _ready(ws)

    proc = _run(ws)

    assert _RECLAIMING in proc.stderr
    assert ws.delegate_witness.exists()


# --------------------------------------------------------------------------- #
# AC5 -- a foreign holder is never resolved by a local pid probe
# --------------------------------------------------------------------------- #
def test_a_foreign_host_holder_is_decided_by_age_not_by_a_local_pid(
    ws: Workspace,
) -> None:
    """Seeded with a pid that IS live locally, under a foreign hostname.

    If the script probed that pid with `kill -0` it would read the lock as held
    and refuse. It must instead notice the holder is on another machine, where a
    local pid number means nothing, and fall through to the age bound -- which
    this lock is past. A run that refuses here is a run that asked the wrong
    machine about the wrong process.
    """
    live = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        _seed_lock(ws, pid=live.pid, host="some-other-host", age_seconds=7200)
        _ready(ws)

        proc = _run(ws)

        assert _RECLAIMING in proc.stderr
        assert "some-other-host" in proc.stderr
        assert ws.delegate_witness.exists()
    finally:
        live.terminate()
        live.wait()


def test_a_fresh_foreign_host_holder_is_respected(ws: Workspace) -> None:
    """Control for the test above.

    The age bound is what decides a foreign lock, so a foreign lock INSIDE the
    bound must still be respected. Otherwise "ignore foreign holders" would be
    indistinguishable from "ignore the lock", and two hosts sharing a workspace
    would write at once.
    """
    _seed_lock(ws, pid=_dead_pid(), host="some-other-host")
    _ready(ws)

    proc = _run(ws)

    assert proc.returncode == EXIT_OK
    assert _NOTHING_TO_DO in proc.stderr
    assert not ws.delegate_witness.exists()


# --------------------------------------------------------------------------- #
# The refusal has to be diagnosable, which is the point of recording a holder
# --------------------------------------------------------------------------- #
def test_the_refusal_names_the_holder_it_is_deferring_to(ws: Workspace) -> None:
    """ "Another reconcile-host is running" with no holder named is a dead end.

    That message is exactly what the workstation printed for 35 minutes while
    nothing held the lock. Naming the pid, the host and the start time is what
    turns the next occurrence into a diagnosis instead of a puzzle.
    """
    live = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        _seed_lock(ws, pid=live.pid, host=os.uname().nodename)
        _ready(ws)

        proc = _run(ws)

        assert str(live.pid) in proc.stderr
        assert "2026-09-17T15:23:00Z" in proc.stderr
    finally:
        live.terminate()
        live.wait()


# --------------------------------------------------------------------------- #
# The mtime probe must survive GNU coreutils, where `-f` is not a format flag
# --------------------------------------------------------------------------- #
def _gnu_stat_shim(tmp_path: Path) -> Path:
    """A ``stat`` that behaves like GNU coreutils rather than BSD.

    The difference is not cosmetic. On GNU, ``-f`` means "report on the
    FILESYSTEM", so ``stat -f %m <path>`` SUCCEEDS and prints a dump beginning
    ``File:`` instead of failing the way a BSD-only reader assumes it will.

    ``-c %Y`` answers truthfully, via python rather than the real ``stat``. A shim that
    returned a canned number would also distort every OTHER ``stat`` call this
    script makes, and would then be testing the shim rather than the script.

    Scoped to the mtime probe (``-f %m`` / ``-c %Y``) and passing everything
    else straight through, for the same reason: the script's other ``stat``
    calls are BSD-spelled and are not what this regression is about. A blanket
    GNU shim broke those instead and made the test fail for a reason that had
    nothing to do with the bug.
    """
    shim_dir = tmp_path / "gnu-stat-bin"
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim = shim_dir / "stat"
    shim.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "-f" && "$2" == "%m" ]]; then\n'
        '  printf "  File: \\"%s\\"\\n  ID: 0 Namelen: 255 Type: apfs\\n" "$3"\n'
        "  exit 0\n"
        "fi\n"
        'if [[ "$1" == "-c" && "$2" == "%Y" ]]; then\n'
        # Via python, NOT via `stat -f %m`. That spelling is the mtime on BSD
        # and the FILESYSTEM flag on GNU, so a shim using it would answer
        # correctly on the workstation and garbage on the Linux runner -- which
        # is the very confusion this test exists to pin, reintroduced inside the
        # test's own scaffolding. CI caught exactly that.
        "  exec python3 -c 'import os, sys; print(int(os.path.getmtime(sys.argv[1])))' \"$3\"\n"
        "fi\n"
        'exec /usr/bin/stat "$@"\n',
        encoding="utf-8",
    )
    shim.chmod(0o755)
    return shim_dir


def test_a_gnu_stat_does_not_poison_the_age_arithmetic(
    ws: Workspace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression pin for a bug that macOS cannot surface at all.

    A version of this file proven green on the workstation died on the Linux CI
    runner with ``line 250: File: unbound variable``: GNU's ``stat -f`` exited 0
    with a filesystem dump, the fallback never fired, and bash read the bare
    word ``File`` in the ``(( ))`` as a variable name under ``set -u``.

    The probe must therefore validate the VALUE, not the exit status. Here the
    lock is two hours old and its holder is unrecorded, so the run can only
    reclaim it if the GNU spelling was reached and parsed.
    """
    monkeypatch.setenv("PATH", f"{_gnu_stat_shim(tmp_path)}:{os.environ['PATH']}")
    _seed_lock(ws, age_seconds=7200)
    _ready(ws)

    proc = _run(ws)

    assert "unbound variable" not in proc.stderr, (
        f"the filesystem dump reached the age arithmetic: {proc.stderr}"
    )
    assert _RECLAIMING in proc.stderr, (
        f"the GNU mtime spelling was never reached or never parsed: {proc.stderr}"
    )
    assert ws.delegate_witness.exists()
