# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19262: the ledger lock admits exactly one writer.

The lock was a directory published by rename, with a stale-lock break that
deleted whatever occupied the lock path once a waiter had decided the holder
was dead, a ``release()`` that deleted the lock path without checking it was
still its own, and a lock root that ``LEDGER_LOCK_ROOT`` could move -- so the
ledger roll (which exports that variable) and lane appends (which do not) took
two different locks on one file. A replay drove the real ``LedgerLock`` to two,
then three, simultaneous holders.

The operator ruling of 2026-09-23T14:20:26Z (item 3) replaced it with an fcntl
lock on one fixed lock-file path derived from the ledger's resolved path. The
kernel drops that lock when its holder dies, so there is no stale break to
race, a release can only unlock the releasing process's own descriptor, and
every writer resolves the same file whatever the environment says.

Each test below drives the real ``LedgerLock`` or the real CLI:

* AC1 -- two waiters race a dead holder, one of them paused right after its
  liveness probe (the replay's hook). At most one may hold.
* AC2 (``release_foreign``) -- the loser of that race releases; a third writer
  with a zero timeout must still be refused while the winner holds.
* AC3 (``roll_and_append``) -- a lane append and the roll, with the roll's
  exported ``LEDGER_LOCK_ROOT`` set, exclude each other.
* AC4 (``killed_holder``) -- a holder killed with SIGKILL mid-append leaves
  nothing that blocks the next writer.

The helpers accept the pre-change constructor too (it took a positional
``stale_after``), so this file runs unmodified against the pre-change script
and fails there for the reason each criterion names, not with a TypeError.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "ledger_lock.py"

# The roll driver exports this before it runs the roll (the omni_home ledger
# roll configuration); lane appends run without it.
_ROLL_ENV = "LEDGER_LOCK_ROOT"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("ledger_lock_single_holder", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _load_module()

_PRE_CHANGE_API = "stale_after" in inspect.signature(MOD.LedgerLock).parameters


def _lock(ledger: Path, timeout: float) -> Any:
    if _PRE_CHANGE_API:
        return MOD.LedgerLock(ledger, timeout, None, None)
    return MOD.LedgerLock(ledger, timeout)


def _release_quietly(lock: Any) -> None:
    # The pre-change release() deleted the lock path without ignoring a
    # missing one, so after a takeover the second release raised. That is the
    # defect under test, not a reason for cleanup to fail.
    with suppress(OSError):
        lock.release()


def _row() -> str:
    # A date-only row opens a markdown row and carries no self-stamp for the
    # clock guard to judge, so nothing but the lock decides the exit code.
    return "- 2026-09-24 single-holder probe row"


def _run_cli(
    ledger: Path, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), str(ledger), *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env=env,
    )


def _spawn_holder(ledger: Path, child_code: str) -> tuple[subprocess.Popen[str], int]:
    """Start the real CLI holding the lock around `child_code`.

    `child_code` runs as the wrapped command, prints its own pid first, and
    then sleeps, so the caller knows the lock is held and which pids exist.
    """
    proc = subprocess.Popen(
        [
            sys.executable,
            str(_SCRIPT),
            str(ledger),
            "--timeout",
            "10s",
            "--",
            sys.executable,
            "-c",
            child_code,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert proc.stdout is not None
    line = proc.stdout.readline().strip()
    assert line.isdigit(), (
        f"the holder never started its command: stdout={line!r} "
        f"stderr={proc.stderr.read() if proc.stderr else ''}"
    )
    return proc, int(line)


def _kill(pid: int) -> None:
    with suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)


def _kill_group(pgid: int) -> None:
    """Reap everything a group-spawned holder started, its command included.

    The holders are spawned with start_new_session=True, so the group id is
    the holder's pid and outlives the holder while its command runs.
    """
    with suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGKILL)


def _plant_dead_holder(ledger: Path) -> None:
    """A real holder takes the lock and is killed with SIGKILL while holding it."""
    proc, _child = _spawn_holder(
        ledger,
        "import os, time; print(os.getpid(), flush=True); time.sleep(120)",
    )
    _kill(proc.pid)
    proc.wait(timeout=10)
    _kill_group(proc.pid)


def _race_two_waiters_at_a_dead_holder(
    ledger: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, bool, Any, bool]:
    """W1 is paused right after its liveness probe answers; W2 races it.

    Returns (w1, w1_held, w2, w2_held) with W1 resumed and finished.
    """
    _plant_dead_holder(ledger)

    probed = threading.Event()
    resume = threading.Event()
    real_probe = getattr(MOD, "process_is_alive", None)
    if real_probe is not None:

        def paused_probe(pid: Any) -> bool:
            answer: bool = real_probe(pid)
            if threading.current_thread().name == "W1" and not probed.is_set():
                probed.set()
                resume.wait(10)
            return answer

        monkeypatch.setattr(MOD, "process_is_alive", paused_probe)

    w1 = _lock(ledger, 10.0)
    w1_outcome: dict[str, bool] = {}

    def run_w1() -> None:
        try:
            w1.acquire()
            w1_outcome["held"] = True
        except TimeoutError:
            w1_outcome["held"] = False

    thread = threading.Thread(target=run_w1, name="W1")
    thread.start()
    deadline = time.monotonic() + 10
    while not probed.is_set() and "held" not in w1_outcome:
        assert time.monotonic() < deadline, "W1 neither probed nor acquired"
        time.sleep(0.01)

    w2 = _lock(ledger, 1.0)
    try:
        w2.acquire()
        w2_held = True
    except TimeoutError:
        w2_held = False
    finally:
        resume.set()
        thread.join(15)
    assert not thread.is_alive(), "W1 never finished acquiring"
    return w1, w1_outcome.get("held", False), w2, w2_held


def test_two_waiters_racing_a_dead_holder_admit_one_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC1."""
    ledger = tmp_path / "ledger.md"
    ledger.write_text("", encoding="utf-8")
    w1, w1_held, w2, w2_held = _race_two_waiters_at_a_dead_holder(ledger, monkeypatch)
    try:
        holders = [name for name, held in (("W1", w1_held), ("W2", w2_held)) if held]
        assert len(holders) == 1, (
            f"{holders} all hold the ledger lock at once; exactly one may"
        )
    finally:
        _release_quietly(w2)
        _release_quietly(w1)


def test_release_foreign_cannot_free_the_current_holder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC2: the loser of the race releases, and the winner still holds."""
    ledger = tmp_path / "ledger.md"
    ledger.write_text("", encoding="utf-8")
    w1, w1_held, w2, w2_held = _race_two_waiters_at_a_dead_holder(ledger, monkeypatch)
    try:
        assert w1_held or w2_held, "neither waiter acquired a lock whose holder died"
        loser, winner = (w2, w1) if w1_held else (w1, w2)
        _release_quietly(loser)
        w3 = _lock(ledger, 0.0)
        try:
            with pytest.raises(TimeoutError):
                w3.acquire()
        finally:
            _release_quietly(w3)
        _release_quietly(winner)
        # Positive control: once the real holder releases, a writer gets in.
        w4 = _lock(ledger, 0.0)
        w4.acquire()
        _release_quietly(w4)
    finally:
        _release_quietly(w2)
        _release_quietly(w1)


def test_release_foreign_by_a_writer_that_never_held_is_a_no_op(
    tmp_path: Path,
) -> None:
    """AC2: releasing a lock this writer does not hold frees nobody."""
    ledger = tmp_path / "ledger.md"
    holder = _lock(ledger, 0.0)
    holder.acquire()
    try:
        bystander = _lock(ledger, 0.0)
        with pytest.raises(TimeoutError):
            bystander.acquire()
        bystander.release()
        third = _lock(ledger, 0.0)
        with pytest.raises(TimeoutError):
            third.acquire()
    finally:
        _release_quietly(holder)


def test_roll_and_append_exclude_each_other_with_the_roll_root_exported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC3, in-process: a lane holds; the roll, with its root exported, waits."""
    ledger = tmp_path / "docs" / "tracking" / "ledger.md"
    ledger.parent.mkdir(parents=True)
    ledger.write_text("", encoding="utf-8")
    monkeypatch.delenv(_ROLL_ENV, raising=False)
    lane = _lock(ledger, 0.0)
    lane.acquire()
    try:
        monkeypatch.setenv(_ROLL_ENV, str(tmp_path / ".onex_state" / "ledger_locks"))
        roll = _lock(ledger, 0.0)
        try:
            with pytest.raises(TimeoutError):
                roll.acquire()
        finally:
            _release_quietly(roll)
    finally:
        _release_quietly(lane)


def test_roll_and_append_exclude_each_other_through_the_cli(tmp_path: Path) -> None:
    """AC3, as deployed: the roll holds with its root exported; a lane append,
    run with that variable unset, is refused with exit 75 and writes nothing."""
    ledger = tmp_path / "docs" / "tracking" / "ledger.md"
    ledger.parent.mkdir(parents=True)
    ledger.write_text("", encoding="utf-8")
    roll_env = {**os.environ, _ROLL_ENV: str(tmp_path / ".onex_state" / "ledger_locks")}
    lane_env = {k: v for k, v in os.environ.items() if k != _ROLL_ENV}

    roll = subprocess.Popen(
        [
            sys.executable,
            str(_SCRIPT),
            str(ledger),
            "--timeout",
            "10s",
            "--",
            sys.executable,
            "-c",
            "import os, time; print(os.getpid(), flush=True); time.sleep(120)",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=roll_env,
        start_new_session=True,
    )
    assert roll.stdout is not None
    child = int(roll.stdout.readline().strip())
    try:
        result = _run_cli(ledger, "--timeout", "0", "--append", _row(), env=lane_env)
        assert result.returncode == 75, (
            f"a lane append landed while the roll held the ledger: rc={result.returncode}"
            f"\n{result.stderr}"
        )
        assert ledger.read_text(encoding="utf-8") == ""
    finally:
        _kill_group(roll.pid)
        roll.wait(timeout=30)

    # Positive control: with the roll gone, the same append lands.
    after = _run_cli(ledger, "--timeout", "5s", "--append", _row(), env=lane_env)
    assert after.returncode == 0, after.stderr
    assert ledger.read_text(encoding="utf-8") == _row() + "\n"


def test_killed_holder_mid_append_leaves_no_lock_that_blocks_the_next_writer(
    tmp_path: Path,
) -> None:
    """AC4: the holder has written part of a row when it is SIGKILLed."""
    ledger = tmp_path / "ledger.md"
    ledger.write_text("", encoding="utf-8")
    partial = (
        "import os, time\n"
        f"with open({str(ledger)!r}, 'a', encoding='utf-8') as fh:\n"
        "    fh.write('- 2026-09-24 half a ro')\n"
        "print(os.getpid(), flush=True)\n"
        "time.sleep(120)\n"
    )
    proc, _child = _spawn_holder(ledger, partial)
    _kill(proc.pid)
    proc.wait(timeout=10)
    _kill_group(proc.pid)

    result = _run_cli(ledger, "--timeout", "2s", "--append", _row())
    assert result.returncode == 0, (
        f"the next writer was blocked by a killed holder: rc={result.returncode}"
        f"\n{result.stderr}"
    )
    assert ledger.read_text(encoding="utf-8").splitlines() == [
        "- 2026-09-24 half a ro",
        _row(),
    ]


def test_killed_holder_whose_command_outlives_it_does_not_keep_the_lock(
    tmp_path: Path,
) -> None:
    """AC4: the wrapped command survives its SIGKILLed parent; the lock does not."""
    ledger = tmp_path / "ledger.md"
    ledger.write_text("", encoding="utf-8")
    proc, child = _spawn_holder(
        ledger,
        "import os, time; print(os.getpid(), flush=True); time.sleep(120)",
    )
    try:
        _kill(proc.pid)
        proc.wait(timeout=10)
        os.kill(child, 0)  # the orphaned command is still running
        result = _run_cli(ledger, "--timeout", "2s", "--append", _row())
        assert result.returncode == 0, (
            f"an orphaned command kept the ledger locked: rc={result.returncode}"
            f"\n{result.stderr}"
        )
    finally:
        _kill_group(proc.pid)
