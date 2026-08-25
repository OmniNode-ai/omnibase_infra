#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Serialize edits/appends to a shared, append-only ledger file.

Multiple agents or processes claiming work in the same ledger file need a
mutex so a "claim, then mutate" write from one writer never interleaves with
another writer's. This script acquires a per-ledger-path lock (an atomically
created lock directory, which works on macOS without flock(1) and needs no
third-party dependencies) and then performs exactly one of:

  * ``--append TEXT`` / ``--append-file PATH`` (``-`` for stdin): append the
    text, durably (fsync'd), to the ledger. Before writing, the payload is
    compared against the last ``--dedup-window`` lines already on disk; an
    identical retry (e.g. re-running after an exit-75 lock timeout without
    knowing whether the prior attempt actually landed) is skipped rather than
    duplicated.
  * ``-- COMMAND ...``: run an arbitrary command (e.g. an editor) while
    holding the lock, for a caller that needs to make an arbitrary edit
    rather than a pure append.

Row convention (documented, not enforced by this script): callers typically
write three row shapes into a ledger of this kind — a CLAIM row ("I am about
to do X"), zero or more PROGRESS rows, and a TERMINAL row ("X is done, here
is the evidence") — so that concurrent agents can grep the ledger to see who
is doing what before claiming new work themselves. This script is agnostic
to row shape; it only serializes and durably persists whatever text a caller
appends.

Exit codes:
  0    success (including an --append skipped as a duplicate of the tail --
       see --dedup-window)
  2    usage error (argparse)
  75   timed out waiting for the lock (EX_TEMPFAIL in sysexits(3)) -- the
       lock is held by someone else; retry is expected to be safe because of
       the dedup-window check above
  127  the -- COMMAND could not be started
  <n>  whatever -- COMMAND itself exited with, when it started and ran
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_TIMEOUT_SECONDS = 300.0
POLL_SECONDS = 0.5
DEFAULT_DEDUP_WINDOW = 20

# Where lock directories live for a given ledger. By default, co-located
# with the ledger file itself (so no shared root needs to be agreed on
# up-front); set LEDGER_LOCK_ROOT to point every writer at one shared
# directory instead (e.g. a network/shared filesystem location), which is
# only necessary if a ledger's own parent directory is not writable by every
# writer.
LOCK_ROOT_ENV = "LEDGER_LOCK_ROOT"
DEFAULT_LOCK_DIRNAME = ".ledger_locks"

# A leading ISO-8601 UTC timestamp token ("2026-08-09T13:59:51Z") immediately
# after an optional bullet, stripped before dedup comparison so a retry whose
# `$(date -u +%Y-%m-%dT%H:%M:%SZ)` shifted by a few seconds still matches on
# tag+body rather than failing a byte-exact comparison.
DEDUP_LEADING_TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\s*"
)
DEDUP_BULLET_PATTERN = re.compile(r"^[-*]\s+")


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_duration(value: str) -> float:
    raw = value.strip().lower()
    if not raw:
        raise argparse.ArgumentTypeError("duration cannot be empty")
    multiplier = 1.0
    if raw.endswith("ms"):
        multiplier = 0.001
        raw = raw[:-2]
    elif raw.endswith("s"):
        raw = raw[:-1]
    elif raw.endswith("m"):
        multiplier = 60.0
        raw = raw[:-1]
    elif raw.endswith("h"):
        multiplier = 3600.0
        raw = raw[:-1]
    try:
        seconds = float(raw) * multiplier
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid duration: {value}") from exc
    if seconds < 0:
        raise argparse.ArgumentTypeError("duration must be non-negative")
    return seconds


def ledger_path(value: str) -> Path:
    """Resolve a ledger file argument to an absolute path.

    A relative argument resolves against the current working directory --
    this is a generic, caller-owned ledger file path, not tied to any
    particular repo layout.
    """
    return Path(value).resolve()


def lock_root_for(ledger: Path) -> Path:
    override = os.environ.get(LOCK_ROOT_ENV)
    if override:
        return Path(override)
    return ledger.parent / DEFAULT_LOCK_DIRNAME


def lock_path_for(ledger: Path) -> Path:
    digest = hashlib.sha256(str(ledger).encode("utf-8")).hexdigest()[:24]
    return lock_root_for(ledger) / f"{ledger.name}.{digest}.lock"


def read_json(path: Path) -> dict[str, Any]:
    try:
        loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return loaded
    except (FileNotFoundError, ValueError, OSError):
        return {}


def write_metadata(lock_dir: Path, ledger: Path, command: list[str] | None) -> None:
    metadata = {
        "ledger": str(ledger),
        "lock_dir": str(lock_dir),
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "cwd": str(Path.cwd()),
        "command": command,
        "acquired_at": utc_now(),
    }
    (lock_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def process_is_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def maybe_break_stale_lock(lock_dir: Path, stale_after: float | None) -> str | None:
    """Break `lock_dir` if it is provably stale, returning a message if so.

    Two independent conditions can make a lock stale:
      * it was written by a process on THIS host that has since died --
        always broken, regardless of `stale_after`.
      * it is older than `stale_after` (if given) -- broken regardless of
        which host wrote it, since a cross-host liveness check is not
        possible.
    """
    metadata = read_json(lock_dir / "metadata.json")
    host = metadata.get("host")
    pid = metadata.get("pid")
    current_host = socket.gethostname()

    if host == current_host and not process_is_alive(pid):
        shutil.rmtree(lock_dir)
        return f"removed dead same-host lock pid={pid}"

    if stale_after is None:
        return None

    try:
        age = time.time() - lock_dir.stat().st_mtime
    except OSError:
        return None
    if age >= stale_after:
        shutil.rmtree(lock_dir)
        return f"removed stale lock age={age:.0f}s"
    return None


class LedgerLock:
    def __init__(
        self,
        ledger: Path,
        timeout: float,
        stale_after: float | None,
        command: list[str] | None,
    ) -> None:
        self.ledger = ledger
        self.timeout = timeout
        self.stale_after = stale_after
        self.command = command
        self.lock_dir = lock_path_for(ledger)
        self.acquired = False

    def acquire(self) -> None:
        self.lock_dir.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self.lock_dir.mkdir()
                self.acquired = True
                write_metadata(self.lock_dir, self.ledger, self.command)
                return
            except FileExistsError:
                message = maybe_break_stale_lock(self.lock_dir, self.stale_after)
                if message:
                    print(f"ledger_lock: {message}: {self.lock_dir}", file=sys.stderr)
                    continue
                if self.timeout == 0 or time.monotonic() >= deadline:
                    metadata = read_json(self.lock_dir / "metadata.json")
                    holder = metadata.get("pid", "unknown")
                    host = metadata.get("host", "unknown-host")
                    acquired_at = metadata.get("acquired_at", "unknown-time")
                    raise TimeoutError(
                        f"timed out waiting for {self.ledger}; held by pid={holder} "
                        f"host={host} since {acquired_at}; lock={self.lock_dir}"
                    )
                time.sleep(POLL_SECONDS)

    def release(self) -> None:
        if self.acquired:
            shutil.rmtree(self.lock_dir)
            self.acquired = False

    def __enter__(self) -> LedgerLock:
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.release()


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if text and not text.endswith("\n"):
        text += "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _normalize_dedup_line(line: str) -> str:
    """One line of an append payload/tail, with its bullet and leading UTC
    timestamp token stripped, for tag+body comparison. Returns "" for a
    blank line (callers filter those out before comparing)."""
    stripped = line.strip()
    stripped = DEDUP_BULLET_PATTERN.sub("", stripped, count=1)
    stripped = DEDUP_LEADING_TIMESTAMP_PATTERN.sub("", stripped, count=1)
    return stripped.strip()


def _normalize_dedup_block(text: str) -> list[str]:
    """`text` as a list of normalized, non-blank lines -- the unit this
    module compares for duplication. Blank lines are dropped so incidental
    leading/trailing newlines in a payload never affect the comparison."""
    return [
        normalized
        for line in text.splitlines()
        if (normalized := _normalize_dedup_line(line))
    ]


def is_duplicate_of_recent_tail(payload: str, existing_tail: str) -> bool:
    """True if `payload` (the full block about to be appended, normalized
    line-by-line) already appears verbatim, as a contiguous run of lines,
    somewhere in `existing_tail` (normalized the same way). False for an
    empty payload or an empty tail -- there is nothing to be a duplicate of.

    A sliding-window match over normalized LINES (not a raw substring check)
    is deliberate: two different rows that happen to share a tag/prefix but
    diverge in body text never falsely match, regardless of where the
    divergence falls relative to line boundaries.
    """
    payload_lines = _normalize_dedup_block(payload)
    if not payload_lines:
        return False
    tail_lines = _normalize_dedup_block(existing_tail)
    window = len(payload_lines)
    if window > len(tail_lines):
        return False
    return any(
        tail_lines[start : start + window] == payload_lines
        for start in range(len(tail_lines) - window + 1)
    )


def read_ledger_tail(path: Path, n: int) -> str:
    """The last `n` lines already on disk at `path`, joined with newlines.
    Empty string if the ledger does not exist yet (first-ever append) or is
    unreadable. Callers hold the ledger lock while calling this, so the read
    is race-free against other writers."""
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return ""
    lines = text.splitlines()
    if n <= 0:
        return ""
    return "\n".join(lines[-n:])


def read_append_payload(args: argparse.Namespace) -> str | None:
    if args.append is not None:
        return str(args.append)
    if args.append_file is None:
        return None
    if args.append_file == "-":
        return sys.stdin.read()
    return Path(args.append_file).read_text(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Acquire a per-ledger mutex before appending to or editing a shared ledger.",
        epilog=(
            "Examples:\n"
            "  scripts/ledger_lock.py path/to/ledger.md --append '- 2026-...: event'\n"
            "  scripts/ledger_lock.py path/to/ledger.md -- ${EDITOR:-vi} path/to/ledger.md\n"
            "  git diff -- path/to/ledger.md | scripts/ledger_lock.py path/to/ledger.md --append-file -\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("ledger", type=ledger_path, help="ledger file to protect")
    parser.add_argument(
        "--timeout",
        type=parse_duration,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="how long to wait for the lock, e.g. 30s, 5m, 1h (default: 5m)",
    )
    parser.add_argument(
        "--stale-after",
        type=parse_duration,
        default=None,
        help="break locks older than this duration; dead same-host pids are always cleaned up",
    )
    parser.add_argument("--append", help="append this text while holding the lock")
    parser.add_argument(
        "--append-file",
        help="append file contents while holding the lock; use '-' for stdin",
    )
    parser.add_argument(
        "--dedup-window",
        type=int,
        default=DEFAULT_DEDUP_WINDOW,
        metavar="N",
        help=(
            "before appending, compare the payload (timestamp-normalized) against the last N "
            "lines already on disk; an identical retry (e.g. after an exit-75 lock timeout) is "
            f"skipped rather than duplicated (default: {DEFAULT_DEDUP_WINDOW})"
        ),
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
    payload = read_append_payload(args)

    requested_actions = sum(1 for item in (payload, command) if item)
    if requested_actions != 1:
        parser.error(
            "provide exactly one action: --append, --append-file, or -- COMMAND"
        )

    try:
        with LedgerLock(args.ledger, args.timeout, args.stale_after, command or None):
            if payload is not None:
                # Dedup check runs inside the held lock, against whatever is
                # actually on disk right now -- race-free against other
                # writers, and against our own prior attempt if this is a
                # retry after exit-75 lock contention.
                tail = read_ledger_tail(args.ledger, args.dedup_window)
                if is_duplicate_of_recent_tail(payload, tail):
                    print(
                        "ledger_lock: DEDUP -- identical row already present in the last "
                        f"{args.dedup_window} lines, skip",
                        file=sys.stderr,
                    )
                    return 0
                append_text(args.ledger, payload)
                return 0
            rc = subprocess.call(command)
            return rc
    except TimeoutError as exc:
        print(f"ledger_lock: {exc}", file=sys.stderr)
        return 75
    except OSError as exc:
        print(f"ledger_lock: command failed to start: {exc}", file=sys.stderr)
        return 127


if __name__ == "__main__":
    raise SystemExit(main())
