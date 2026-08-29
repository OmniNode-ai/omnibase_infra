#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Serialize a heavy command on a named machine-wide lock (OMN-16822).

Heavy local runs -- ``pre-commit run --all-files``, a full ``pytest`` suite,
``mypy`` over ``src/``, ``git push`` behind the pre-push gate -- contend for
the whole machine. Two lanes running them concurrently do not merely go
slower: they read each other's load as flakiness and report false failures.
So the sprint machine-load protocol requires one such run at a time, and this
script is the mechanism.

WHY THIS EXISTS RATHER THAN ``flock(1)``
----------------------------------------
The protocol used to be written as the util-linux idiom::

    flock /tmp/omninode-heavy-suite.lock -c '<command>'

**macOS ships no ``flock(1)``** (util-linux is not installed; only
``/usr/bin/shlock`` and ``/usr/bin/lockf`` exist). The shell therefore
returned 127 ``command not found`` **without ever running the wrapped
command**, which is the dangerous direction: a caller that does not check the
exit code sees no output, assumes its heavy run happened under a lock, and
continues. Six lanes in the 2026-08-28 wave-2 fan-out each hit this and each
wrote a private throwaway shim. This is the one committed copy.

PRIMITIVE AND ITS ``flock(1)`` INTEROP STORY
--------------------------------------------
This helper takes ``fcntl.flock(2)`` (``LOCK_EX``) on the lock path. That is
the *same kernel lock* util-linux ``flock(1)`` takes, so the two interoperate
exactly: a ``flock(1)`` holder on a Linux CI runner or the ``.201`` host
blocks this helper, and a holder of this helper blocks ``flock(1)``. Mixed
fleets need no coordination beyond agreeing on the lock path.

``shlock`` (PID-file advisory) and ``lockf`` were rejected: ``shlock`` is a
different, non-interoperating protocol, and neither is released by the kernel
when the holder dies.

Because ``fcntl.flock`` is released by the kernel on process death, there is
no stale-lock class to break and therefore no liveness guessing. **This helper
never signals or kills the peer holding the lock** -- killing a peer's heavy
suite to reclaim the lock would be worse than the unserialized run the lock
exists to prevent. It waits, then fails closed.

MAX-HOLD CAP, ENFORCED BY THE HOLDER ON ITSELF (OMN-16869)
-----------------------------------------------------------
Kernel-released-on-death bounds the *crashed* holder, not the *wedged* one.
On 2026-08-28 a lane held the protocol lock for ~2h wrapping a single ``git
push`` that had been stalled on the network for 1h33m. Its pid was alive the
whole time, so nothing above applies -- and it consumed no CPU, so the
contention this lock exists to prevent was not even occurring while it
starved five other lanes.

So the holder now bounds *itself*: past ``--max-hold`` it aborts its own
wrapped command (SIGTERM to the command's process tree, then SIGKILL after a
grace period), logs a loud marker line, and exits ``70``. This is **not** a
waiter reclaiming a lock. Acquisition is still decided by ``fcntl.flock``
alone, no peer is ever signalled, and the only process this helper signals is
the one it started itself. ``--max-hold 0`` opts out for a genuinely long
release run.

WAITER-VISIBLE PROGRESS, INFORMATIONAL ONLY (OMN-16869)
--------------------------------------------------------
The same incident had a second half: every waiter saw a byte-identical holder
line for two hours, because the sidecar was written once at acquisition and
never refreshed. "Peer is 20 minutes into a suite" and "peer is wedged on a
socket" read the same.

The holder now refreshes the sidecar every ``--heartbeat-every`` with
``heartbeat_at``, ``held_seconds`` and ``child_cpu_seconds`` (the CPU time of
the wrapped command's whole process tree, from one ``ps`` sample), and
waiters render those on their notice interval. A holder whose ``held_seconds``
climbs while ``child_cpu_seconds`` does not is the wedged shape.

**This is reporting, not a control input.** Nothing read from the sidecar can
grant, deny, break, or steal the lock -- see ``sidecar_path_for``. A waiter
that dislikes what it reads still only ever waits or fails closed.

FAIL-CLOSED CONTRACT
--------------------
If the lock cannot be acquired within the bounded wait, the wrapped command
is **not run at all** and the helper exits non-zero with a message naming the
holder and the elapsed wait. It never falls through to an unlocked run, and
it never exits 0 without having run the command.

EXIT CODES
----------
  0     the wrapped command ran under the lock and succeeded
  2     usage error (argparse)
  70    EX_SOFTWARE -- the lock WAS acquired, the wrapped command ran, and it
        was ABORTED by this helper for exceeding --max-hold. The command's own
        result is unknown and must not be treated as a pass. A wrapped command
        that itself exits 70 is distinguished by the absence of the
        ``heavy_lock:`` max-hold marker line on stderr.
  69    EX_UNAVAILABLE -- the lock WAS acquired but the wrapped command could
        not be started (e.g. the executable does not exist). Deliberately not
        127: 127 is what the shell reports when *this helper itself* is
        missing, and those two must stay mechanically distinguishable, which
        is the exact confusion ``flock(1)``'s absence created. Every failure
        this helper originates also carries a ``heavy_lock:`` marker line on
        stderr; a markerless failure means the helper never ran.
  75    EX_TEMPFAIL -- timed out waiting for the lock. Fail-closed: the
        command did NOT run.
  <n>   whatever the wrapped command itself exited with, passed through
        untouched (including a genuine 127 from a shell it invoked).

USAGE
-----
    scripts/heavy_lock.py -- pre-commit run --all-files
    scripts/heavy_lock.py --timeout 20m -- uv run pytest tests/ -q
    scripts/heavy_lock.py --label "OMN-1234 push" -- git push

The default lock path is ``/tmp/omninode-heavy-suite.lock`` -- the machine-load
protocol's lock. Pass ``--lock`` for an independent critical section.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_LOCK_PATH = Path("/tmp/omninode-heavy-suite.lock")  # noqa: S108
DEFAULT_TIMEOUT_SECONDS = 600.0
DEFAULT_NOTICE_SECONDS = 60.0
POLL_SECONDS = 0.25

# OMN-16869. The cap is ON by default: a cap the caller must opt into is the
# defect, not the fix -- the starving lane is precisely the one that did not
# think to pass a flag. One hour is far above any legitimate run this lock
# serializes (the item-10 recipe's own budget is `--timeout 20m`) and far
# below the ~2h starvation that motivated the ticket.
DEFAULT_MAX_HOLD_SECONDS = 3600.0
DEFAULT_HEARTBEAT_SECONDS = 15.0
# How long an aborted command gets between SIGTERM and SIGKILL.
TERM_GRACE_SECONDS = 10.0

EXIT_COMMAND_NOT_STARTED = 69  # EX_UNAVAILABLE
EXIT_MAX_HOLD_EXCEEDED = 70  # EX_SOFTWARE -- aborted at the max-hold cap
EXIT_LOCK_TIMEOUT = 75  # EX_TEMPFAIL

_MARKER = "heavy_lock:"


def _log(message: str) -> None:
    """Every line this helper originates carries the ``heavy_lock:`` marker.

    That marker is what makes "the helper ran and reports a problem"
    distinguishable from "the helper was not there at all" -- the failure mode
    that made the ``flock(1)`` protocol silently no-op.
    """
    print(f"{_MARKER} {message}", file=sys.stderr, flush=True)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_duration(value: str) -> float:
    """A duration like ``30s``, ``5m``, ``1h``, ``250ms``, or bare seconds."""
    raw = value.strip().lower()
    if not raw:
        raise argparse.ArgumentTypeError("duration cannot be empty")
    multiplier = 1.0
    if raw.endswith("ms"):
        multiplier, raw = 0.001, raw[:-2]
    elif raw.endswith("s"):
        raw = raw[:-1]
    elif raw.endswith("m"):
        multiplier, raw = 60.0, raw[:-1]
    elif raw.endswith("h"):
        multiplier, raw = 3600.0, raw[:-1]
    try:
        seconds = float(raw) * multiplier
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid duration: {value}") from exc
    if seconds < 0:
        raise argparse.ArgumentTypeError("duration must be non-negative")
    return seconds


def parse_utc(value: Any) -> datetime | None:
    """Parse one of our own ``utc_now()`` stamps back, tolerantly.

    Anything unparseable yields ``None`` rather than raising: this only ever
    feeds a human-readable message, and a corrupt sidecar must never be able
    to crash a waiter.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_ps_cpu_time(value: str) -> float | None:
    """Seconds from a ``ps -o time=`` cell, across both renderings.

    macOS renders ``MMM:SS.ss`` (``109:46.51`` is 109 *minutes*); Linux
    renders ``HH:MM:SS`` and ``DD-HH:MM:SS`` for long-lived processes. Parsing
    right-to-left -- seconds, minutes, hours -- handles every form without
    branching on platform. Unparseable input yields ``None``.
    """
    raw = value.strip()
    if not raw:
        return None
    days = 0.0
    if "-" in raw:
        day_part, _, raw = raw.partition("-")
        try:
            days = float(day_part)
        except ValueError:
            return None
    total = days * 86400.0
    try:
        for index, part in enumerate(reversed(raw.split(":"))):
            total += float(part) * (60.0**index)
    except ValueError:
        return None
    return total


def _ps_process_table() -> list[tuple[int, int, float]]:
    """One ``ps`` sample as ``(pid, ppid, cpu_seconds)`` rows.

    A single sample of the whole table, rather than one ``ps`` per pid, keeps
    the heartbeat cheap enough to run under a lock we are trying not to hold
    any longer than necessary.
    """
    try:
        completed = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,time="],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    rows: list[tuple[int, int, float]] = []
    for line in completed.stdout.splitlines():
        fields = line.split(None, 2)
        if len(fields) != 3:
            continue
        try:
            pid, ppid = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        cpu = parse_ps_cpu_time(fields[2])
        if cpu is None:
            continue
        rows.append((pid, ppid, cpu))
    return rows


def process_tree_pids(root_pid: int, table: list[tuple[int, int, float]]) -> list[int]:
    """``root_pid`` and every descendant, shallowest first."""
    children: dict[int, list[int]] = {}
    for pid, ppid, _cpu in table:
        children.setdefault(ppid, []).append(pid)
    ordered: list[int] = []
    frontier = [root_pid]
    seen = {root_pid}
    while frontier:
        pid = frontier.pop(0)
        ordered.append(pid)
        for child in children.get(pid, ()):
            if child not in seen:
                seen.add(child)
                frontier.append(child)
    return ordered


def tree_cpu_seconds(root_pid: int) -> float | None:
    """CPU seconds burned by the wrapped command's whole process tree.

    The tree, not just the direct child: the item-10 recipe routinely wraps a
    shell (``sh -c 'uv run pytest ...'``) whose own CPU is ~0. Charging only
    the direct child would report a busy suite as making no progress -- the
    exact confusion this field exists to remove.

    ``None`` means "could not sample" (no ``ps``, or the tree exited between
    the poll and the sample), never "no progress".
    """
    table = _ps_process_table()
    if not table:
        return None
    wanted = set(process_tree_pids(root_pid, table))
    total = 0.0
    matched = False
    for pid, _ppid, cpu in table:
        if pid in wanted:
            total += cpu
            matched = True
    return total if matched else None


def sidecar_path_for(lock: Path) -> Path:
    """Where the current holder advertises who it is.

    The kernel lock itself carries no identity, so an *advisory* sidecar makes
    the timeout message actionable ("held by pid=N since T running X") instead
    of merely "still locked". It is advisory only: acquisition is decided by
    ``fcntl.flock`` alone, never by anything read from here, so a stale or
    corrupt sidecar can never grant or deny the lock.
    """
    return lock.with_name(lock.name + ".holder.json")


def read_holder(lock: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(sidecar_path_for(lock).read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _describe_progress(holder: dict[str, Any]) -> str:
    """The OMN-16869 half: how long, and with how much actual work done.

    Renders whatever the sidecar carries and says so plainly when a field is
    absent. A pre-OMN-16869 holder wrote only ``acquired_at``, so held-duration
    is derived from that timestamp and the progress clause degrades to a
    statement of what is missing -- never to a guess about the holder.
    """
    parts: list[str] = []

    held = holder.get("held_seconds")
    if not isinstance(held, int | float):
        acquired = parse_utc(holder.get("acquired_at"))
        if acquired is not None:
            held = (datetime.now(UTC) - acquired).total_seconds()
    if isinstance(held, int | float):
        parts.append(f"held {float(held):.0f}s")

    beat = parse_utc(holder.get("heartbeat_at"))
    if beat is None:
        parts.append(
            "cpu unknown (no progress heartbeat: the holder predates OMN-16869 "
            "or has not reached its first beat)"
        )
    else:
        cpu = holder.get("child_cpu_seconds")
        if isinstance(cpu, int | float):
            parts.append(f"cpu {float(cpu):.1f}s")
        else:
            parts.append("cpu unavailable")
        parts.append(f"last beat {(datetime.now(UTC) - beat).total_seconds():.0f}s ago")

    return "; ".join(parts)


def describe_holder(lock: Path) -> str:
    holder = read_holder(lock)
    if not holder:
        return "held by an unidentified process (no holder sidecar)"
    return (
        f"held by pid={holder.get('pid', 'unknown')} "
        f"host={holder.get('host', 'unknown-host')} "
        f"since {holder.get('acquired_at', 'unknown-time')} "
        f"running {holder.get('label') or holder.get('command')!r} "
        f"[{_describe_progress(holder)}]"
    )


def _write_sidecar(lock: Path, payload: dict[str, Any]) -> None:
    """Publish the sidecar atomically.

    Atomic because OMN-16869 turned this from a single write at acquisition
    into a repeated one: a waiter polling the file must never observe a
    half-written record. ``read_holder`` tolerates a corrupt file anyway, but
    tolerating it and never producing it are different guarantees.

    Advisory only -- a failure here can never affect the lock.
    """
    target = sidecar_path_for(lock)
    tmp = target.with_name(target.name + f".{os.getpid()}.tmp")
    try:
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        tmp.replace(target)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass


def write_holder(
    lock: Path,
    command: list[str],
    label: str | None,
    *,
    acquired_at: str | None = None,
    child_pid: int | None = None,
    held_seconds: float | None = None,
    max_hold_seconds: float | None = None,
) -> str:
    """Advertise who holds the lock; returns the ``acquired_at`` stamp used.

    Every key the pre-OMN-16869 revision wrote (``pid``, ``host``, ``cwd``,
    ``command``, ``label``, ``acquired_at``) is still written under the same
    name and type. The progress keys are strictly ADDITIVE, so a concurrently
    running old invocation reading this file sees exactly what it saw before
    and ignores the rest.
    """
    stamp = acquired_at or utc_now()
    payload: dict[str, Any] = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "cwd": str(Path.cwd()),
        "command": command,
        "label": label,
        "acquired_at": stamp,
    }
    if child_pid is not None:
        payload["heartbeat_at"] = utc_now()
        payload["held_seconds"] = round(float(held_seconds or 0.0), 1)
        payload["child_pid"] = child_pid
        cpu = tree_cpu_seconds(child_pid)
        payload["child_cpu_seconds"] = None if cpu is None else round(cpu, 2)
        payload["max_hold_seconds"] = max_hold_seconds
    _write_sidecar(lock, payload)
    return stamp


def clear_holder(lock: Path) -> None:
    try:
        sidecar_path_for(lock).unlink()
    except OSError:
        pass


class LockTimeoutError(Exception):
    """The lock was not acquired within the bounded wait. Fail closed."""


def acquire(
    lock: Path,
    timeout: float,
    notice_every: float,
) -> int:
    """Block until ``lock`` is held, or raise ``LockTimeout``.

    Returns the held file descriptor. The caller owns closing it; closing the
    descriptor (or the process dying) releases the kernel lock.
    """
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock), os.O_CREAT | os.O_RDWR, 0o666)
    start = time.monotonic()
    deadline = start + timeout
    next_notice = start + notice_every
    announced = False
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                waited = time.monotonic() - start
                if announced:
                    _log(f"acquired {lock} after waiting {waited:.0f}s")
                return fd
            except OSError as exc:
                if exc.errno not in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                    raise
                now = time.monotonic()
                if timeout == 0 or now >= deadline:
                    raise LockTimeoutError(
                        f"FAILED CLOSED: could not acquire {lock}; "
                        f"waited {now - start:.0f}s of a {timeout:.0f}s budget; "
                        f"{describe_holder(lock)}. The command was NOT run -- "
                        f"retry when the peer finishes, or raise --timeout. "
                        f"Do NOT drop this wrapper and run unserialized."
                    )
                if now >= next_notice:
                    _log(
                        f"waiting for {lock} ({now - start:.0f}s elapsed of "
                        f"{timeout:.0f}s); {describe_holder(lock)}"
                    )
                    next_notice = now + notice_every
                    announced = True
                time.sleep(POLL_SECONDS)
    except BaseException:
        os.close(fd)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="heavy_lock.py",
        description=(
            "Run a heavy command while holding a machine-wide fcntl.flock(2) "
            "lock. Replaces the util-linux `flock(1)` idiom, which does not "
            "exist on macOS and silently exits 127 without running anything."
        ),
        epilog=(
            "Examples:\n"
            "  scripts/heavy_lock.py -- pre-commit run --all-files\n"
            "  scripts/heavy_lock.py --timeout 20m -- uv run pytest tests/ -q\n"
            "  scripts/heavy_lock.py --label 'OMN-1234 push' -- git push\n"
            "  scripts/heavy_lock.py --max-hold 2h -- uv run pytest tests/\n"
            "\n"
            "Exit codes: 0 ok | 2 usage | 69 command could not start (lock WAS "
            "held) | 70 command ABORTED at --max-hold, result unknown | 75 lock "
            "timeout, command NOT run | <n> the command's own code.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=DEFAULT_LOCK_PATH,
        help=f"lock file path (default: {DEFAULT_LOCK_PATH})",
    )
    parser.add_argument(
        "--timeout",
        type=parse_duration,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=(
            "how long to wait for the lock, e.g. 30s, 10m, 1h; 0 means do not "
            f"wait at all (default: {DEFAULT_TIMEOUT_SECONDS:.0f}s)"
        ),
    )
    parser.add_argument(
        "--notice-every",
        type=parse_duration,
        default=DEFAULT_NOTICE_SECONDS,
        help=(
            "print a still-waiting notice on stderr this often, so a long wait "
            f"never looks like a hang (default: {DEFAULT_NOTICE_SECONDS:.0f}s)"
        ),
    )
    parser.add_argument(
        "--max-hold",
        type=parse_duration,
        default=DEFAULT_MAX_HOLD_SECONDS,
        help=(
            "abort the wrapped command and release the lock once it has been "
            "held this long, e.g. 45m, 2h; 0 disables the cap. The HOLDER "
            "enforces this on itself -- a waiter never reclaims a lock "
            f"(default: {DEFAULT_MAX_HOLD_SECONDS:.0f}s)"
        ),
    )
    parser.add_argument(
        "--heartbeat-every",
        type=parse_duration,
        default=DEFAULT_HEARTBEAT_SECONDS,
        help=(
            "refresh the holder sidecar with held-duration and the wrapped "
            "command's tree CPU this often, so waiters can tell a progressing "
            f"holder from a wedged one; 0 disables (default: "
            f"{DEFAULT_HEARTBEAT_SECONDS:.0f}s)"
        ),
    )
    parser.add_argument(
        "--label",
        default=None,
        help="short description recorded for peers to see in the holder sidecar",
    )
    return parser


def _signal_pids(pids: list[int], sig: int) -> None:
    """Signal our own descendants, deepest first, tolerating races.

    These are processes THIS helper started. No peer -- no other holder, no
    waiter -- is ever a member of this set, which is what keeps the "never
    signals the peer holding the lock" contract intact while still bounding
    the hold.
    """
    for pid in reversed(pids):
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            continue


def abort_wrapped_command(proc: subprocess.Popen[bytes]) -> None:
    """SIGTERM the wrapped command's tree, then SIGKILL what survives.

    The tree, not just the direct child: a wrapped ``sh -c '...'`` that
    ignores TERM would otherwise leave its real work orphaned and running
    while we released the lock, which defeats the point of the cap.
    """
    tree = process_tree_pids(proc.pid, _ps_process_table()) or [proc.pid]
    _signal_pids(tree, signal.SIGTERM)
    deadline = time.monotonic() + TERM_GRACE_SECONDS
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        time.sleep(POLL_SECONDS)
    _signal_pids(tree, signal.SIGKILL)
    try:
        proc.wait(timeout=TERM_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        _log(f"the wrapped command (pid {proc.pid}) survived SIGKILL; releasing anyway")


def supervise(
    proc: subprocess.Popen[bytes],
    lock: Path,
    command: list[str],
    label: str | None,
    acquired_at: str,
    started: float,
    max_hold: float,
    heartbeat_every: float,
) -> int:
    """Wait out the wrapped command, publishing progress and honouring the cap.

    Returns the command's own exit code, or ``EXIT_MAX_HOLD_EXCEEDED`` when
    the cap fired. The cap is checked against this holder's OWN elapsed hold
    time -- nothing here reads the sidecar, and nothing here can be influenced
    by a peer.
    """
    next_beat = started + heartbeat_every if heartbeat_every > 0 else None
    while True:
        returncode = proc.poll()
        if returncode is not None:
            return returncode
        now = time.monotonic()
        held = now - started
        if max_hold > 0 and held >= max_hold:
            _log(
                f"MAX-HOLD EXCEEDED: the wrapped command {command!r} has held "
                f"{lock} for {held:.0f}s (cap {max_hold:.0f}s) and is being "
                f"ABORTED so it cannot keep starving peers. Its result is "
                f"UNKNOWN -- this is exit {EXIT_MAX_HOLD_EXCEEDED}, NOT a pass. "
                f"A command that legitimately runs this long needs an explicit "
                f"--max-hold, not a dropped wrapper."
            )
            abort_wrapped_command(proc)
            return EXIT_MAX_HOLD_EXCEEDED
        if next_beat is not None and now >= next_beat:
            write_holder(
                lock,
                command,
                label,
                acquired_at=acquired_at,
                child_pid=proc.pid,
                held_seconds=held,
                max_hold_seconds=max_hold or None,
            )
            next_beat = now + heartbeat_every
        time.sleep(POLL_SECONDS)


def split_command(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" not in argv:
        return argv, []
    delimiter = argv.index("--")
    return argv[:delimiter], argv[delimiter + 1 :]


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser_argv, command = split_command(raw_argv)
    parser = build_parser()
    args = parser.parse_args(parser_argv)

    if not command:
        parser.error("provide the command to serialize after `--`")

    lock: Path = args.lock
    try:
        fd = acquire(lock, args.timeout, args.notice_every)
    except LockTimeoutError as exc:
        _log(str(exc))
        return EXIT_LOCK_TIMEOUT
    except OSError as exc:
        _log(f"FAILED CLOSED: cannot open lock {lock}: {exc}. The command was NOT run.")
        return EXIT_LOCK_TIMEOUT

    acquired_at = write_holder(lock, command, args.label)
    started = time.monotonic()
    try:
        try:
            proc = subprocess.Popen(command)
        except (FileNotFoundError, NotADirectoryError, PermissionError) as exc:
            _log(
                f"the lock was acquired but the command could not be started: "
                f"{command[0]!r}: {exc}. This is exit {EXIT_COMMAND_NOT_STARTED} "
                f"(EX_UNAVAILABLE), NOT 127 -- a bare 127 with no {_MARKER!r} line "
                f"means this helper itself is missing."
            )
            return EXIT_COMMAND_NOT_STARTED
        except OSError as exc:
            _log(f"the lock was acquired but the command could not be started: {exc}")
            return EXIT_COMMAND_NOT_STARTED
        return supervise(
            proc,
            lock,
            command,
            args.label,
            acquired_at,
            started,
            args.max_hold,
            args.heartbeat_every,
        )
    finally:
        clear_holder(lock)
        # Releasing the descriptor releases the kernel lock. The kernel would
        # do this for us if we died, which is why no peer is ever signalled.
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
