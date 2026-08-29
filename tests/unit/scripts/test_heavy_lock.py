# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for scripts/heavy_lock.py (OMN-16822).

The defect this file locks down: the sprint machine-load protocol told every
lane to serialize heavy runs with util-linux ``flock(1)``, which does not
exist on the macOS host. The shell returned 127 ``command not found`` WITHOUT
running the wrapped command, so a lane that did not check the exit code
believed it held a lock it never took.

The regression contract proven here:

* the "flock-127 class" cannot recur silently -- the helper's own failure to
  start a wrapped command is a DISTINCT exit code (69) carrying a marker line
  on stderr, so it is mechanically distinguishable from the helper itself
  being absent (shell 127, no marker);
* two near-simultaneous invocations on the same lock path observably
  serialize -- exactly one is in the critical section at a time;
* an unavailable lock FAILS CLOSED after the bounded wait (exit 75, holder
  and elapsed named) and the wrapped command is never run unlocked;
* the lock is ``fcntl.flock(2)`` on the lock path, i.e. the same kernel lock
  ``flock(1)`` takes, so a util-linux holder blocks the helper and vice versa.
"""

from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "heavy_lock.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("heavy_lock", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _load_module()


def _run_cli(args: list[str], timeout: float = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


# --------------------------------------------------------------------------
# The script exists and is a real, runnable helper
# --------------------------------------------------------------------------


def test_script_is_committed_and_executable() -> None:
    """AC1: the helper is committed under omnibase_infra/scripts/."""
    assert _SCRIPT.is_file()
    assert os.access(_SCRIPT, os.X_OK)


def test_header_documents_the_flock1_interop_choice() -> None:
    """AC1: the primitive choice and its flock(1) interop story are stated."""
    header = _SCRIPT.read_text(encoding="utf-8")[:4000]
    assert "fcntl.flock" in header
    assert "flock(1)" in header


def test_default_lock_path_is_the_protocol_lock() -> None:
    assert Path("/tmp/omninode-heavy-suite.lock") == MOD.DEFAULT_LOCK_PATH  # noqa: S108


# --------------------------------------------------------------------------
# AC3 -- the flock-127 class cannot recur
# --------------------------------------------------------------------------


def test_missing_wrapped_command_is_distinguishable_from_missing_helper(
    tmp_path: Path,
) -> None:
    """A command that does not exist must NOT look like the helper's own 127.

    ``flock(1)`` being absent produced a bare shell 127 with no output from
    the wrapper. This helper reserves a distinct exit code (69,
    EX_UNAVAILABLE) plus a ``heavy_lock:`` marker line for "I ran, I took the
    lock, and YOUR command could not be started". Anything without that
    marker is the helper itself being missing.
    """
    lock = tmp_path / "hl.lock"
    result = _run_cli(["--lock", str(lock), "--", "definitely-not-a-real-binary-xyz"])

    assert result.returncode == MOD.EXIT_COMMAND_NOT_STARTED == 69
    assert result.returncode != 127
    assert "heavy_lock:" in result.stderr
    assert "definitely-not-a-real-binary-xyz" in result.stderr

    # Negative control: the helper being ABSENT is a different, markerless
    # failure, which is precisely the flock(1) situation the protocol hit.
    absent = subprocess.run(
        [sys.executable, str(tmp_path / "no_such_helper.py"), "--", "true"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert absent.returncode != 0
    assert "heavy_lock:" not in absent.stderr


def test_wrapped_command_exit_code_passes_through(tmp_path: Path) -> None:
    lock = tmp_path / "hl.lock"
    assert _run_cli(["--lock", str(lock), "--", "true"]).returncode == 0
    assert _run_cli(["--lock", str(lock), "--", "false"]).returncode == 1
    # A shell that itself returns 127 passes through untouched -- the helper
    # does not launder the wrapped command's own exit codes.
    shelled = _run_cli(["--lock", str(lock), "--", "sh", "-c", "exit 127"])
    assert shelled.returncode == 127


def test_two_near_simultaneous_invocations_serialize(tmp_path: Path) -> None:
    """Exactly one invocation is in the critical section at a time.

    Each invocation appends ENTER, sleeps, then appends EXIT. If the lock
    worked, the trace is strictly ENTER/EXIT/ENTER/EXIT with no interleave.
    """
    lock = tmp_path / "hl.lock"
    trace = tmp_path / "trace.txt"
    script = f"echo ENTER >> {trace}; sleep 1.5; echo EXIT >> {trace}"

    procs = [
        subprocess.Popen(
            [
                sys.executable,
                str(_SCRIPT),
                "--lock",
                str(lock),
                "--timeout",
                "60s",
                "--",
                "sh",
                "-c",
                script,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    for proc in procs:
        assert proc.wait(timeout=90) == 0

    lines = trace.read_text(encoding="utf-8").split()
    assert lines == ["ENTER", "EXIT", "ENTER", "EXIT"], (
        f"critical section interleaved: {lines}"
    )


# --------------------------------------------------------------------------
# AC2 -- fail closed and loudly, never fall through unlocked
# --------------------------------------------------------------------------


def test_timeout_fails_closed_without_running_the_command(tmp_path: Path) -> None:
    """The peer here is another heavy_lock, i.e. the real contention shape."""
    lock = tmp_path / "hl.lock"
    sentinel = tmp_path / "ran.txt"
    started = tmp_path / "started.txt"

    holder = subprocess.Popen(
        [
            sys.executable,
            str(_SCRIPT),
            "--lock",
            str(lock),
            "--label",
            "peer heavy suite",
            "--",
            "sh",
            "-c",
            f"touch {started}; sleep 20",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 20
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert started.exists(), "peer never entered the critical section"

        result = _run_cli(
            [
                "--lock",
                str(lock),
                "--timeout",
                "1s",
                "--",
                "sh",
                "-c",
                f"touch {sentinel}",
            ]
        )
    finally:
        holder.kill()
        holder.wait(timeout=30)

    assert result.returncode == MOD.EXIT_LOCK_TIMEOUT == 75
    assert not sentinel.exists(), "FELL THROUGH: command ran without the lock"
    assert "heavy_lock:" in result.stderr
    # Actionable: names the holder (pid + label) and the elapsed wait.
    assert str(holder.pid) in result.stderr
    assert "peer heavy suite" in result.stderr
    assert "waited" in result.stderr.lower()
    assert "NOT run" in result.stderr


def test_timeout_zero_is_nonblocking_and_still_fails_closed(tmp_path: Path) -> None:
    lock = tmp_path / "hl.lock"
    sentinel = tmp_path / "ran.txt"
    fd = os.open(str(lock), os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        result = _run_cli(
            [
                "--lock",
                str(lock),
                "--timeout",
                "0",
                "--",
                "sh",
                "-c",
                f"touch {sentinel}",
            ]
        )
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    assert result.returncode == MOD.EXIT_LOCK_TIMEOUT
    assert not sentinel.exists()


def test_helper_never_kills_the_peer_holding_the_lock(tmp_path: Path) -> None:
    """The helper waits or fails; it must never signal the holder.

    A peer lane's heavy suite being killed to free the lock would be worse
    than the unserialized run this ticket exists to prevent.
    """
    lock = tmp_path / "hl.lock"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import fcntl,os,sys,time\n"
                f"fd=os.open({str(lock)!r}, os.O_CREAT|os.O_RDWR, 0o666)\n"
                "fcntl.flock(fd, fcntl.LOCK_EX)\n"
                "sys.stderr.write('HELD\\n'); sys.stderr.flush()\n"
                "time.sleep(5)\n"
            ),
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stderr is not None
        assert holder.stderr.readline().strip() == "HELD"
        result = _run_cli(["--lock", str(lock), "--timeout", "1s", "--", "true"])
        assert result.returncode == MOD.EXIT_LOCK_TIMEOUT
        # The peer is still alive and unsignalled after the helper gave up.
        assert holder.poll() is None
    finally:
        holder.kill()
        holder.wait(timeout=10)


# --------------------------------------------------------------------------
# AC4 -- flock(1)-compatible kernel lock
# --------------------------------------------------------------------------


def test_helper_takes_fcntl_flock_on_the_lock_path(tmp_path: Path) -> None:
    """While the helper holds the lock, an independent LOCK_EX|LOCK_NB on the
    same path must fail -- i.e. the helper took the SAME kernel lock that
    util-linux ``flock(1)`` takes, so the two interoperate on any host.
    """
    lock = tmp_path / "hl.lock"
    started = tmp_path / "started.txt"
    proc = subprocess.Popen(
        [
            sys.executable,
            str(_SCRIPT),
            "--lock",
            str(lock),
            "--",
            "sh",
            "-c",
            f"touch {started}; sleep 3",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 20
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert started.exists(), "helper never entered the critical section"

        fd = os.open(str(lock), os.O_CREAT | os.O_RDWR, 0o666)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
    finally:
        proc.wait(timeout=30)

    # Released on exit: the same non-blocking take now succeeds.
    fd = os.open(str(lock), os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def test_lock_is_released_when_the_holder_dies(tmp_path: Path) -> None:
    """Kernel-released on death -- a crashed lane cannot wedge every peer.

    This is the property the directory-mutex alternative does not have, and
    the reason no stale-breaking (which would have to guess about liveness)
    is needed.
    """
    lock = tmp_path / "hl.lock"
    started = tmp_path / "started.txt"
    proc = subprocess.Popen(
        [
            sys.executable,
            str(_SCRIPT),
            "--lock",
            str(lock),
            "--",
            "sh",
            "-c",
            f"touch {started}; sleep 30",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 20
    while not started.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert started.exists()
    proc.kill()
    proc.wait(timeout=30)

    result = _run_cli(["--lock", str(lock), "--timeout", "20s", "--", "true"])
    assert result.returncode == 0


# --------------------------------------------------------------------------
# CLI surface
# --------------------------------------------------------------------------


def test_no_command_is_a_usage_error(tmp_path: Path) -> None:
    result = _run_cli(["--lock", str(tmp_path / "hl.lock")])
    assert result.returncode == 2


def test_waiting_is_announced_on_stderr(tmp_path: Path) -> None:
    """A silent wait is how a lane concludes nothing is happening."""
    lock = tmp_path / "hl.lock"
    fd = os.open(str(lock), os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        result = _run_cli(
            [
                "--lock",
                str(lock),
                "--timeout",
                "3s",
                "--notice-every",
                "1s",
                "--",
                "true",
            ]
        )
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    assert "waiting" in result.stderr.lower()


# ==========================================================================
# OMN-16869 -- a holder cannot starve the host indefinitely, and a waiter can
# tell a progressing holder from a wedged one.
#
# Motivating incident (2026-08-28): a lane held /tmp/omninode-heavy-suite.lock
# for ~2h wrapping a single `git push` that had been wedged on the network for
# 1h33m. It consumed no CPU -- the contention the lock exists to prevent was
# not even occurring -- yet it blocked five other lanes, one of which waited
# 2200s+ across two attempts and saw a byte-identical holder line the entire
# time. Two properties were missing and are proven here:
#
#   AC1 -- a max-hold ceiling enforced by the HOLDER on its own wrapped
#          command. Never a waiter stealing or breaking the lock: acquisition
#          stays fcntl.flock-only, and no peer is ever signalled.
#   AC2 -- waiter-visible PROGRESS in the advisory sidecar (a heartbeat plus
#          the wrapped command's tree CPU), so "peer is 20 minutes into a
#          suite" and "peer is wedged on a socket" no longer read identically.
#          Informational only -- never an input to an acquisition decision.
#
# Compatibility: heavy_lock.py is in live use by concurrent lanes, so an old
# and a new invocation must interoperate on the same lock path and the same
# sidecar file. The sidecar contract is additive-only, proven below.
# ==========================================================================


def _read_sidecar(lock: Path) -> dict[str, Any]:
    return MOD.read_holder(lock)


def _wait_for(predicate: Any, timeout: float = 20.0, poll: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll)
    return False


# --------------------------------------------------------------------------
# AC1 -- holder-side max-hold cap
# --------------------------------------------------------------------------


def test_max_hold_cap_aborts_a_wedged_holder(tmp_path: Path) -> None:
    """The 2026-08-28 shape: a wrapped command that never finishes.

    Past the cap the HOLDER aborts its own wrapped command and exits with a
    distinct, marker-carrying code. It does not silently keep holding, and it
    does not report success.
    """
    lock = tmp_path / "hl.lock"
    started = time.monotonic()
    result = _run_cli(
        ["--lock", str(lock), "--max-hold", "3s", "--", "sleep", "120"], timeout=90
    )
    elapsed = time.monotonic() - started

    assert result.returncode == MOD.EXIT_MAX_HOLD_EXCEEDED
    assert MOD.EXIT_MAX_HOLD_EXCEEDED not in (0, 69, 75, 127)
    assert "heavy_lock:" in result.stderr
    assert "max-hold" in result.stderr.lower()
    assert elapsed < 60, f"cap did not fire promptly: {elapsed:.0f}s"


def test_max_hold_abort_releases_the_lock_for_a_waiter(tmp_path: Path) -> None:
    """AC1's operative claim: a capped holder CANNOT indefinitely block peers.

    A waiter whose whole budget is far shorter than the wedged command still
    gets the lock, because the holder let go at its cap.
    """
    lock = tmp_path / "hl.lock"
    holder = subprocess.Popen(
        [
            sys.executable,
            str(_SCRIPT),
            "--lock",
            str(lock),
            "--max-hold",
            "3s",
            "--label",
            "wedged peer",
            "--",
            "sleep",
            "300",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert _wait_for(lambda: bool(_read_sidecar(lock)), timeout=20)
        waiter = _run_cli(
            [
                "--lock",
                str(lock),
                "--timeout",
                "60s",
                "--notice-every",
                "1s",
                "--",
                "true",
            ],
            timeout=90,
        )
        assert waiter.returncode == 0, waiter.stderr
    finally:
        holder.kill()
        holder.wait(timeout=30)


def test_max_hold_kills_a_command_that_ignores_sigterm(tmp_path: Path) -> None:
    """A wedged command does not get to outlast the cap by ignoring TERM."""
    lock = tmp_path / "hl.lock"
    result = _run_cli(
        [
            "--lock",
            str(lock),
            "--max-hold",
            "2s",
            "--",
            "sh",
            "-c",
            'trap "" TERM; sleep 300',
        ],
        timeout=120,
    )
    assert result.returncode == MOD.EXIT_MAX_HOLD_EXCEEDED
    assert "heavy_lock:" in result.stderr


def test_a_command_finishing_inside_the_cap_is_untouched(tmp_path: Path) -> None:
    """The cap must not perturb the overwhelmingly common case."""
    lock = tmp_path / "hl.lock"
    result = _run_cli(
        ["--lock", str(lock), "--max-hold", "60s", "--", "sh", "-c", "exit 3"]
    )
    assert result.returncode == 3
    assert "max-hold" not in result.stderr.lower()


def test_max_hold_zero_disables_the_cap(tmp_path: Path) -> None:
    """An explicit opt-out stays available for a genuinely long release run."""
    lock = tmp_path / "hl.lock"
    result = _run_cli(
        ["--lock", str(lock), "--max-hold", "0", "--", "sh", "-c", "sleep 2; exit 5"],
        timeout=60,
    )
    assert result.returncode == 5


def test_max_hold_has_a_bounded_default(tmp_path: Path) -> None:
    """The cap is ON by default -- an opt-in cap is the defect, not the fix.

    The ticket's own struck premise was that a control existed but was opt-in
    per caller. A default of "no cap" would reproduce exactly that.
    """
    assert MOD.DEFAULT_MAX_HOLD_SECONDS > 0
    args = MOD.build_parser().parse_args([])
    assert args.max_hold == MOD.DEFAULT_MAX_HOLD_SECONDS


def test_a_waiter_never_signals_the_peer_holding_the_lock(tmp_path: Path) -> None:
    """The file's design philosophy, held: waiters wait, they never reclaim.

    The cap is enforced by the holder ON ITSELF. A waiter that times out must
    leave the peer's process completely untouched.
    """
    lock = tmp_path / "hl.lock"
    holder = subprocess.Popen(
        [
            sys.executable,
            str(_SCRIPT),
            "--lock",
            str(lock),
            "--max-hold",
            "0",
            "--",
            "sleep",
            "30",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert _wait_for(lambda: bool(_read_sidecar(lock)), timeout=20)
        waiter = _run_cli(
            ["--lock", str(lock), "--timeout", "2s", "--", "true"], timeout=60
        )
        assert waiter.returncode == MOD.EXIT_LOCK_TIMEOUT
        assert holder.poll() is None, "the waiter killed the peer holding the lock"
    finally:
        holder.kill()
        holder.wait(timeout=30)


# --------------------------------------------------------------------------
# AC2 -- waiter-visible progress, informational only
# --------------------------------------------------------------------------


def test_sidecar_heartbeat_advances_while_the_holder_runs(tmp_path: Path) -> None:
    """The sidecar is refreshed, not written once and left to go stale."""
    lock = tmp_path / "hl.lock"
    holder = subprocess.Popen(
        [
            sys.executable,
            str(_SCRIPT),
            "--lock",
            str(lock),
            "--heartbeat-every",
            "1s",
            "--max-hold",
            "0",
            "--",
            "sleep",
            "30",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert _wait_for(lambda: "heartbeat_at" in _read_sidecar(lock), timeout=25), (
            f"no heartbeat ever appeared: {_read_sidecar(lock)}"
        )
        first = _read_sidecar(lock)
        assert _wait_for(
            lambda: _read_sidecar(lock).get("held_seconds", -1)
            > first.get("held_seconds", 0),
            timeout=25,
        ), "held_seconds never advanced"
    finally:
        holder.kill()
        holder.wait(timeout=30)


def test_a_wedged_holder_is_distinguishable_from_a_working_one(tmp_path: Path) -> None:
    """AC2's whole point, and the exact 2026-08-28 confusion.

    Both holders are alive, both hold the lock, both have a growing
    held_seconds. Only the working one burns CPU. From the sidecar ALONE --
    no `pgrep -P` on the peer's process tree by hand -- the two must read
    differently.
    """
    wedged_lock = tmp_path / "wedged.lock"
    busy_lock = tmp_path / "busy.lock"

    def _spawn(lock: Path, script: str) -> subprocess.Popen[str]:
        return subprocess.Popen(
            [
                sys.executable,
                str(_SCRIPT),
                "--lock",
                str(lock),
                "--heartbeat-every",
                "1s",
                "--max-hold",
                "0",
                "--",
                "sh",
                "-c",
                script,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    wedged = _spawn(wedged_lock, "sleep 60")  # network-wedged shape: no CPU
    busy = _spawn(busy_lock, "while :; do :; done")  # real work: burns CPU
    try:
        for lock in (wedged_lock, busy_lock):
            assert _wait_for(
                lambda lock=lock: "child_cpu_seconds" in _read_sidecar(lock), timeout=25
            ), f"no progress field for {lock}"
        time.sleep(4)
        wedged_first = _read_sidecar(wedged_lock)
        busy_first = _read_sidecar(busy_lock)
        time.sleep(4)
        wedged_second = _read_sidecar(wedged_lock)
        busy_second = _read_sidecar(busy_lock)

        # Both holders are demonstrably still holding.
        assert wedged_second["held_seconds"] > wedged_first["held_seconds"]
        assert busy_second["held_seconds"] > busy_first["held_seconds"]

        # Only the working one shows CPU progress.
        busy_delta = busy_second["child_cpu_seconds"] - busy_first["child_cpu_seconds"]
        wedged_delta = (
            wedged_second["child_cpu_seconds"] - wedged_first["child_cpu_seconds"]
        )
        assert busy_delta > 0.5, (
            f"a CPU-burning holder showed no progress: {busy_delta}"
        )
        assert wedged_delta < 0.5, f"a sleeping holder showed progress: {wedged_delta}"
    finally:
        for proc in (wedged, busy):
            proc.kill()
            proc.wait(timeout=30)


def test_waiter_notice_names_the_holder_and_how_long_it_has_held(
    tmp_path: Path,
) -> None:
    """A waiter learns identity AND held-duration without touching the peer."""
    lock = tmp_path / "hl.lock"
    holder = subprocess.Popen(
        [
            sys.executable,
            str(_SCRIPT),
            "--lock",
            str(lock),
            "--heartbeat-every",
            "1s",
            "--max-hold",
            "0",
            "--label",
            "OMN-16680 scaffold-validate push",
            "--",
            "sleep",
            "60",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert _wait_for(lambda: "heartbeat_at" in _read_sidecar(lock), timeout=25)
        time.sleep(2)
        waiter = _run_cli(
            [
                "--lock",
                str(lock),
                "--timeout",
                "3s",
                "--notice-every",
                "1s",
                "--",
                "true",
            ],
            timeout=60,
        )
        assert waiter.returncode == MOD.EXIT_LOCK_TIMEOUT
        assert "OMN-16680 scaffold-validate push" in waiter.stderr
        assert "held" in waiter.stderr.lower()
        assert "cpu" in waiter.stderr.lower()
    finally:
        holder.kill()
        holder.wait(timeout=30)


def test_the_sidecar_never_decides_acquisition(tmp_path: Path) -> None:
    """Progress data is advisory. A sidecar claiming a live, busy holder must
    not stop acquisition when no process actually holds the kernel lock."""
    lock = tmp_path / "hl.lock"
    MOD.sidecar_path_for(lock).parent.mkdir(parents=True, exist_ok=True)
    MOD.sidecar_path_for(lock).write_text(
        json.dumps(
            {
                "pid": 1,
                "host": "somewhere-else",
                "acquired_at": "2026-08-28T14:49:15Z",
                "heartbeat_at": "2999-01-01T00:00:00Z",
                "held_seconds": 999999.0,
                "child_cpu_seconds": 999999.0,
                "label": "a lie",
                "command": ["sleep", "inf"],
            }
        ),
        encoding="utf-8",
    )
    assert (
        _run_cli(["--lock", str(lock), "--timeout", "5s", "--", "true"]).returncode == 0
    )


# --------------------------------------------------------------------------
# Backward compatibility with concurrently-running pre-OMN-16869 invocations
# --------------------------------------------------------------------------


def test_sidecar_schema_is_additive_only(tmp_path: Path) -> None:
    """An old invocation must still parse a new holder's sidecar.

    The pre-OMN-16869 reader consumes exactly these keys. They must keep
    their names and types; everything new is additional.
    """
    lock = tmp_path / "hl.lock"
    holder = subprocess.Popen(
        [
            sys.executable,
            str(_SCRIPT),
            "--lock",
            str(lock),
            "--heartbeat-every",
            "1s",
            "--max-hold",
            "0",
            "--label",
            "compat",
            "--",
            "sleep",
            "30",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert _wait_for(lambda: "heartbeat_at" in _read_sidecar(lock), timeout=25)
        holder_json = _read_sidecar(lock)
        for key in ("pid", "host", "cwd", "command", "label", "acquired_at"):
            assert key in holder_json, f"pre-OMN-16869 sidecar key {key} was dropped"
        assert isinstance(holder_json["pid"], int)
        assert isinstance(holder_json["host"], str)
        assert isinstance(holder_json["cwd"], str)
        assert isinstance(holder_json["command"], list)
        assert holder_json["label"] == "compat"
        assert isinstance(holder_json["acquired_at"], str)
    finally:
        holder.kill()
        holder.wait(timeout=30)


def test_describe_holder_tolerates_a_pre_omn16869_sidecar(tmp_path: Path) -> None:
    """A NEW waiter blocked by an OLD holder must not crash, and should still
    surface a held-duration derived from the one timestamp an old holder wrote."""
    lock = tmp_path / "hl.lock"
    MOD.sidecar_path_for(lock).write_text(
        json.dumps(
            {
                "pid": 55890,
                "host": "old-host",
                "cwd": "/somewhere",
                "command": ["git", "push"],
                "label": "OMN-16680 scaffold-validate push",
                "acquired_at": "2026-08-28T14:49:15Z",
            }
        ),
        encoding="utf-8",
    )
    described = MOD.describe_holder(lock)
    assert "55890" in described
    assert "OMN-16680 scaffold-validate push" in described
    assert "held" in described.lower()


def test_describe_holder_tolerates_a_missing_or_corrupt_sidecar(tmp_path: Path) -> None:
    lock = tmp_path / "hl.lock"
    assert "unidentified" in MOD.describe_holder(lock)
    MOD.sidecar_path_for(lock).write_text("{not json", encoding="utf-8")
    assert "unidentified" in MOD.describe_holder(lock)


def test_existing_call_sites_need_no_new_flags() -> None:
    """The DISPATCH_LANE_BRIEF item 10 recipe must keep working verbatim."""
    args = MOD.build_parser().parse_args(["--timeout", "20m", "--label", "OMN-1 push"])
    assert args.timeout == 1200.0
    assert args.max_hold == MOD.DEFAULT_MAX_HOLD_SECONDS
    assert args.heartbeat_every == MOD.DEFAULT_HEARTBEAT_SECONDS


def test_cpu_time_parser_handles_both_ps_renderings() -> None:
    """macOS renders `MMM:SS.ss`; Linux renders `HH:MM:SS` and `DD-HH:MM:SS`."""
    assert MOD.parse_ps_cpu_time("0:00.07") == pytest.approx(0.07)
    assert MOD.parse_ps_cpu_time("109:46.51") == pytest.approx(109 * 60 + 46.51)
    assert MOD.parse_ps_cpu_time("01:33:46") == pytest.approx(3600 + 33 * 60 + 46)
    assert MOD.parse_ps_cpu_time("2-01:00:00") == pytest.approx(2 * 86400 + 3600)
    assert MOD.parse_ps_cpu_time("garbage") is None
    assert MOD.parse_ps_cpu_time("") is None
