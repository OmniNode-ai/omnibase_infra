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
