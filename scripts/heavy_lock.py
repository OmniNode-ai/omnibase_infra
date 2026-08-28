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

EXIT_COMMAND_NOT_STARTED = 69  # EX_UNAVAILABLE
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


def describe_holder(lock: Path) -> str:
    holder = read_holder(lock)
    if not holder:
        return "held by an unidentified process (no holder sidecar)"
    return (
        f"held by pid={holder.get('pid', 'unknown')} "
        f"host={holder.get('host', 'unknown-host')} "
        f"since {holder.get('acquired_at', 'unknown-time')} "
        f"running {holder.get('label') or holder.get('command')!r}"
    )


def write_holder(lock: Path, command: list[str], label: str | None) -> None:
    payload = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "cwd": str(Path.cwd()),
        "command": command,
        "label": label,
        "acquired_at": utc_now(),
    }
    try:
        sidecar_path_for(lock).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except OSError:
        # Advisory only -- never let bookkeeping failure affect the lock.
        pass


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
            "\n"
            "Exit codes: 0 ok | 2 usage | 69 command could not start (lock WAS "
            "held) | 75 lock timeout, command NOT run | <n> the command's own code.\n"
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
        "--label",
        default=None,
        help="short description recorded for peers to see in the holder sidecar",
    )
    return parser


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

    write_holder(lock, command, args.label)
    try:
        try:
            completed = subprocess.run(command, check=False)
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
        return completed.returncode
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
