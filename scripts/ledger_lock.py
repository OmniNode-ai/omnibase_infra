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

Row convention: callers typically write three row shapes into a ledger of
this kind — a CLAIM row ("I am about to do X"), zero or more PROGRESS rows,
and a TERMINAL row ("X is done, here is the evidence") — so that concurrent
agents can grep the ledger to see who is doing what before claiming new work
themselves. Row shape is otherwise the caller's business; the one shape this
script recognizes is the CLAIM row, for the claim-token support below.

Claim tokens (OMN-16400)
------------------------
The protocol requires a CLAIM row to be appended BEFORE the mutation it
authorizes. A row's own timestamp cannot establish that: it is a string the
writer typed, and it can be (and repeatedly has been) typed after the fact.
The lock-protected byte offset the row landed at can, because the append
assigns it and it only ever increases.

So ``--append`` of a claim-shaped row prints one extra line on stdout::

    ledger_lock: CLAIM-TOKEN LCT1-<offset>-<line>-<digest>-<appended_at>

Usage, end to end::

    # 1. Claim, and keep the token.
    TOKEN=$(ledger_lock.py LEDGER --append "$CLAIM_ROW" \
              | sed -n 's/^ledger_lock: CLAIM-TOKEN //p')

    # 2. Do the mutation, and record when it happened (from the mutated
    #    system, not from your own clock -- e.g. a PR's mergedAt).
    MUTATED_AT=$(gh pr view "$PR" --json mergedAt --jq .mergedAt)

    # 3. Prove the claim preceded it. Exit 0 = ok, 1 = post-hoc claim or
    #    token/ledger mismatch, 2 = malformed token.
    ledger_lock.py LEDGER --verify-claim-token "$TOKEN" \
        --mutation-at "$MUTATED_AT"

Retrying step 1 after an exit-75 lock timeout is safe and token-stable: the
retry is deduped (see ``--dedup-window``) and hands back the FIRST attempt's
token, so the token cited in steps 2-3 does not depend on how many attempts
the append took.

What this does and does not prove. Verification re-reads the ledger at the
token's offset and re-hashes the row, so a token naming a claim that was
never appended, or whose row was later rewritten, fails — a caller cannot
mint a token for a claim it did not make. Ordering two tokens from the same
ledger by ``offset`` needs no clock and is the strongest signal available.
The ``appended_at`` field used by ``--verify-claim-token`` is tool-observed
rather than caller-supplied, but it does trust the host clock; it is not a
defense against a writer who deliberately moves that clock.

Exit codes:
  0    success (including an --append skipped as a duplicate of the tail --
       see --dedup-window -- and a --verify-claim-token that passed)
  1    --verify-claim-token: the claim does not precede the cited mutation,
       or the token does not match any row on disk
  2    usage error (argparse), including a malformed claim token
  74   the append would cross a section cap, or a --roll-section left the
       section over its cap; nothing was written
  75   timed out waiting for the lock (EX_TEMPFAIL in sysexits(3)) -- the
       lock is held by someone else; retry is expected to be safe because of
       the dedup-window check above
  76   the payload does not open a row in the capped section named by
       --section-heading, so appending it would extend the row above it
       instead of starting its own; nothing was written
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
from collections import Counter
from contextlib import suppress
from datetime import UTC, date, datetime, timedelta, timezone
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

# OMN-16400. The bullet+leading-timestamp strip above only fires when the
# self-stamp is the first thing on the line. The rows real callers write are
# frequently a markdown TABLE ROW whose first cell is the self-stamp
# ("| 2026-08-28T18:10:00Z | <handle> | <ticket> | CLAIM | <body> |"), or a
# BOLD-wrapped stamp ("- **2026-08-28T18:10:00Z** ..."). Neither matched, so
# an exit-75 retry whose `$(date -u ...)` had advanced landed a SECOND row
# for the same claim -- the duplicate/ghost-claim defect this closes.
#
# Both patterns are anchored at the START of the (bullet-stripped) line on
# purpose: a timestamp quoted inside a row's BODY is evidence about some
# other event, not this row's self-stamp, and two rows citing different
# mutation instants are genuinely different rows. Normalizing a mid-body
# timestamp away would silently swallow a real second row.
DEDUP_LEADING_TABLE_CELL_TIMESTAMP_PATTERN = re.compile(
    r"^\|\s*\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\s*(?=\|)"
)
DEDUP_LEADING_BOLD_TIMESTAMP_PATTERN = re.compile(
    r"^\*\*\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\*\*\s*"
)

# Claim-row recognition (OMN-16400). A claim row is the row that says "I am
# about to mutate something"; it is the row whose append ORDER relative to
# that mutation the protocol actually cares about. Recognized shapes, all
# drawn from rows real lanes write:
#   * table row with a CLAIM status cell:  "| ts | handle | OMN-1 | CLAIM | ..."
#   * prose row with a CLAIM verb token:   "- ts [handle] OMN-1 - CLAIM: ..."
#   * an explicit status line:             "- **Status:** CLAIM+TERMINAL"
# The token must stand alone in a status/verb position -- the word "claim"
# inside ordinary prose must never promote a row to claim-shaped.
CLAIM_TABLE_CELL_PATTERN = re.compile(r"\|\s*CLAIM(?:\+[A-Z-]+)?\s*\|")
CLAIM_STATUS_PATTERN = re.compile(r"\bStatus:?\*{0,2}\s*:?\s*CLAIM(?:\+[A-Z-]+)?\b")
CLAIM_VERB_PATTERN = re.compile(r"(?:^|[\s\-—*\[(])CLAIM(?:\+[A-Z-]+)?\b[:\s\]).—-]")

# Claim-token wire format (OMN-16400):
#   LCT1-<byte offset>-<line number>-<sha256/12 of the normalized row>-<appended_at>
# `LCT1` is the format version so a future field addition is detectable
# rather than silently mis-parsed.
CLAIM_TOKEN_VERSION = "LCT1"
CLAIM_TOKEN_PREFIX = "ledger_lock: CLAIM-TOKEN "
CLAIM_TOKEN_PATTERN = re.compile(
    rf"^{CLAIM_TOKEN_PREFIX}({CLAIM_TOKEN_VERSION}-\d+-\d+-[0-9a-f]{{12}}-\S+)\s*$",
    re.MULTILINE,
)


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
    self-stamp stripped, for tag+body comparison. Returns "" for a blank
    line (callers filter those out before comparing).

    Three leading-self-stamp shapes are normalized away (OMN-15787 for the
    first, OMN-16400 for the other two): a bare timestamp, a bold-wrapped
    timestamp, and a leading markdown table CELL holding a timestamp. Only
    the LEADING position is stripped -- see the pattern definitions for why
    a mid-body timestamp is left alone deliberately.
    """
    stripped = line.strip()
    stripped = DEDUP_BULLET_PATTERN.sub("", stripped, count=1)
    stripped = DEDUP_LEADING_TIMESTAMP_PATTERN.sub("", stripped, count=1)
    stripped = DEDUP_LEADING_BOLD_TIMESTAMP_PATTERN.sub("", stripped, count=1)
    stripped = DEDUP_LEADING_TABLE_CELL_TIMESTAMP_PATTERN.sub("|", stripped, count=1)
    return stripped.strip()


def is_claim_row(text: str) -> bool:
    """True if `text` (one row, or a block whose first non-blank line is the
    row) is claim-shaped -- i.e. a row asserting that the writer is about to
    perform the mutation it names.

    Claim shape matters because the claim row is the only row whose position
    relative to a mutation the ledger protocol constrains ("claim before you
    mutate"). Non-claim rows get no token; there is nothing to order them
    against.
    """
    for line in text.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        if CLAIM_TABLE_CELL_PATTERN.search(candidate):
            return True
        if CLAIM_STATUS_PATTERN.search(candidate):
            return True
        if CLAIM_VERB_PATTERN.search(candidate + " "):
            return True
    return False


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


class ClaimToken:
    """A receipt for one claim row, minted under the ledger lock (OMN-16400).

    `offset` is the byte position the row was appended at, observed while
    holding the lock. That is the ONLY append-order signal on a ledger that
    is un-forgeable by a writer: file position is assigned by the append
    itself and increases monotonically, whereas the timestamp in the row is
    a string the writer typed and can type at any time. The ghost-collision
    incident this closes came from reading two rows' self-stamps as append
    order when they disagreed with file position; comparing `offset` is the
    reading that cannot invert.

    `digest` binds the token to the row's normalized text, so a token is
    only honoured while that exact row is still sitting at that offset.

    `appended_at` is the instant THIS TOOL observed the append, not a
    caller-supplied string. It is what claim-before-mutation comparison uses.
    Honest limit, stated rather than implied: a caller who never runs the
    tool cannot mint a token at all (verification re-reads and re-hashes the
    file), but the tool trusts the host clock, so `appended_at` is only as
    good as that clock. Ordering two tokens from the same ledger by `offset`
    needs no clock at all and is the stronger check of the two.

    Deliberately a plain class, not a dataclass: this module is loaded by
    path (`spec_from_file_location`) by several callers, and under
    `from __future__ import annotations` the dataclass decorator resolves
    annotations through `sys.modules[cls.__module__]`, which is absent for a
    path-loaded module. A plain `__init__` keeps import-by-path working.
    """

    __slots__ = ("appended_at", "digest", "line_no", "offset")

    def __init__(
        self, offset: int, line_no: int, digest: str, appended_at: str
    ) -> None:
        self.offset = offset
        self.line_no = line_no
        self.digest = digest
        self.appended_at = appended_at

    def __repr__(self) -> str:
        return (
            f"ClaimToken(offset={self.offset}, line_no={self.line_no}, "
            f"digest={self.digest!r}, appended_at={self.appended_at!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ClaimToken):
            return NotImplemented
        return self.render() == other.render()

    def __hash__(self) -> int:
        return hash(self.render())

    def render(self) -> str:
        return "-".join(
            (
                CLAIM_TOKEN_VERSION,
                str(self.offset),
                str(self.line_no),
                self.digest,
                self.appended_at,
            )
        )

    @classmethod
    def parse(cls, raw: str) -> ClaimToken | None:
        parts = raw.strip().split("-", 4)
        if len(parts) != 5 or parts[0] != CLAIM_TOKEN_VERSION:
            return None
        _, offset, line_no, digest, appended_at = parts
        if not (offset.isdigit() and line_no.isdigit()):
            return None
        if len(digest) != 12 or any(c not in "0123456789abcdef" for c in digest):
            return None
        if not appended_at:
            return None
        return cls(
            offset=int(offset),
            line_no=int(line_no),
            digest=digest,
            appended_at=appended_at,
        )


def claim_row_digest(row: str) -> str:
    """A short digest of one row's NORMALIZED text.

    Normalized, not raw, so a retry whose self-stamp shifted resolves to the
    same digest as the row already on disk -- which is what lets a deduped
    retry hand back the original row's token instead of a new one.
    """
    normalized = "\n".join(_normalize_dedup_block(row))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def parse_claim_token_line(stream_text: str) -> ClaimToken | None:
    """The claim token emitted on stdout by `--append`, or None if the run
    emitted no token (the appended row was not claim-shaped)."""
    match = CLAIM_TOKEN_PATTERN.search(stream_text)
    if match is None:
        return None
    return ClaimToken.parse(match.group(1))


def _offsets_and_lines(path: Path) -> list[tuple[int, int, str]]:
    """(byte offset, 1-based line number, text) for every line in `path`.

    Offsets are computed from the encoded bytes so a non-ASCII row (an em
    dash in a body, which these ledgers are full of) does not shift every
    later offset.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return []
    result: list[tuple[int, int, str]] = []
    offset = 0
    for index, line in enumerate(text.splitlines(keepends=True), start=1):
        result.append((offset, index, line.rstrip("\n")))
        offset += len(line.encode("utf-8"))
    return result


def find_existing_claim_token(path: Path, payload: str) -> ClaimToken | None:
    """The token of the row already on disk that `payload` duplicates.

    Called on the dedup path so a retried claim append returns the FIRST
    attempt's token rather than nothing. Token stability across retries is
    the property that makes retry-after-exit-75 safe to script: the caller
    cites one token in its mutation no matter how many attempts it took.

    `appended_at` cannot be recovered for a row written by an earlier
    process, so it is reported as the file's last-modified instant -- an
    upper bound on when the row landed, which is the conservative direction
    for a claim-before-mutation check (it can only make a claim look later,
    never earlier, so it never manufactures a passing verdict).
    """
    digest = claim_row_digest(payload)
    fallback_time = _mtime_iso(path)
    for offset, line_no, line in reversed(_offsets_and_lines(path)):
        if not line.strip():
            continue
        if claim_row_digest(line) == digest:
            return ClaimToken(
                offset=offset,
                line_no=line_no,
                digest=digest,
                appended_at=fallback_time,
            )
    return None


def _ledger_size(path: Path) -> int:
    """Byte size of the ledger, i.e. the offset the next append lands at.

    Zero for a ledger that does not exist yet, and for a ledger whose final
    line has no trailing newline the append helper adds one, so the offset
    reported here is still where the appended block begins.
    """
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _mtime_iso(path: Path) -> str:
    try:
        stamp = path.stat().st_mtime
    except OSError:
        return utc_now()
    return (
        datetime.fromtimestamp(stamp, UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def verify_claim_token(
    path: Path, token: ClaimToken, mutation_at: str
) -> tuple[int, str]:
    """Check that `token`'s claim row is real and that it predates a cited
    mutation. Returns (exit code, message).

    Two independent checks, in order:

    1. INTEGRITY -- the row at the token's recorded offset still hashes to
       the token's digest. This is what stops a caller inventing a token for
       a claim it never appended: the row has to actually be in the file, at
       that position.
    2. ORDERING -- the tool-observed append instant is strictly before the
       cited mutation instant.
    """
    lines = _offsets_and_lines(path)
    matched = next(
        (
            line
            for offset, _line_no, line in lines
            if offset == token.offset and claim_row_digest(line) == token.digest
        ),
        None,
    )
    if matched is None:
        return 1, (
            f"TOKEN DOES NOT MATCH the ledger: no row at byte offset {token.offset} "
            f"of {path} hashes to {token.digest}. The claim this token names was "
            "never appended, or the row was rewritten after it was."
        )
    try:
        claimed_at = _parse_iso_utc(token.appended_at)
        mutated_at = _parse_iso_utc(mutation_at)
    except ValueError as exc:
        return 2, f"unparseable timestamp: {exc}"
    if claimed_at >= mutated_at:
        return 1, (
            f"POST-HOC CLAIM: the claim row was appended at {token.appended_at}, "
            f"which is not before the cited mutation at {mutation_at}. The ledger "
            "protocol requires the claim to precede the mutation it authorizes."
        )
    return 0, (
        f"CLAIM-BEFORE-MUTATION OK: claim appended at {token.appended_at} "
        f"(byte offset {token.offset}, line {token.line_no}) precedes the cited "
        f"mutation at {mutation_at}."
    )


def _parse_iso_utc(value: str) -> datetime:
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


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


# --------------------------------------------------------------------------
# Section caps and the archive roll (OMN-17023)
# --------------------------------------------------------------------------
#
# An append-only ledger section grows forever unless something bounds it. The
# rolling work ledger reached 21,752 lines before a human noticed and split it
# by hand; the split then invalidated every line-number watermark pointing
# into it. Both halves of that failure are addressed here: a cap that refuses
# or rolls at append time, and a roll that emits a receipt naming the boundary
# row so a reader can re-anchor without guessing.
#
# The capped section is the TAIL of the file: from `--section-heading` to EOF.
# That is the shape an append-only section has, and it is the only shape where
# "the oldest N rows can be moved out" is well defined. The heading must occur
# exactly once -- zero or two occurrences means the caller is pointing at
# something other than the section it thinks it is, and that fails closed
# rather than capping the wrong bytes.

EXIT_SECTION_CAP = 74
EXIT_ROW_SHAPE = 76
ROLL_RECEIPT_SCHEMA = "ledger-roll/1"
ROLL_RECEIPT_PREFIX = "ledger_lock: ROLL "
ROLL_POINTER_MARKER = "<!-- ledger-roll:"
# The line-shift banner (OMN-18620). A roll removes rows from the TOP of the
# section, so every `<path>:<line>` citation written before it resolves to the
# wrong row afterwards -- measured on 2026-09-17, where a roll moved the
# operator's qwen hold ruling from :4311 to :3385 and a consent row from :4367
# to :3441. Two gates resolve those citations by line number
# (omniclaude credential_rotation_guard.py, omnibase_infra
# apply_application_database_acl.py), so a roll can make a pending consent
# unresolvable or, worse, point it at a different row.
#
# One banner is kept, not a chain: it is stripped and rewritten on each roll,
# exactly like the pointer block above. It bridges the MOST RECENT roll, which
# is the one a live citation is most likely to have been written before. The
# durable fix for older citations is the timestamp citation form, because a row
# timestamp travels with the row into the archive; the roll chain itself stays
# recoverable from the archive headers.
ROLL_SHIFT_BANNER_PREFIX = "ROLL:"
# A SENTENCE ON THE EXISTING POINTER LINE, not a line of its own. A separate
# line looked tidier and was wrong: `parse_section` counts the pointer block
# toward the section's line budget, so one extra line changed the roll
# arithmetic for every caller -- an existing cap test went from a roll that fit
# in 10 lines to one refused at 11. The offset is roll metadata exactly like the
# pointer it sits beside, and metadata must not consume the budget meant for
# rows.
ROLL_SHIFT_PLACEHOLDER = "@@ROLL_SHIFT@@"
ROLL_POINTER_PROSE_PREFIX = "> Older rows live in"

# A row starts where an append starts (OMN-17403).
#
# This used to read `^#{2,6} \S` -- a row was a markdown heading. `append_text`
# adds no heading, and lanes append timestamp-led lines, so the parser and the
# writer disagreed about what a row is. Measured on the live rolling work
# ledger on 2026-09-16: §5 held 8,219 lines that parsed as 29 rows. Two chronic
# failures fell out of that one mismatch -- `ledger_watermark.py --advance`
# anchored on a tail row whose body stayed open, so the next ordinary append
# rewrote its digest and `--resolve` exited 3 for eleven consecutive days; and
# `--roll-section` counts ROWS, so keeping 40 of 29 rolled nothing while the
# section sat at twice its cap.
#
# The shapes below are the ones counted on that section: a bare UTC timestamp
# (6,918 lines), a pipe-table row opening with one (436), a bullet opening with
# one (715), and a markdown heading (29). Anything else is a continuation line
# of the row above it, which is what wrapped prose in a row body actually is.
ROW_START_PATTERN = re.compile(
    r"^(?:"
    r"#{2,6} \S"  # a markdown heading
    r"|(?:[-*] +)?\|? *\d{4}-\d{2}-\d{2}"  # optional bullet, optional pipe, a date
    r")"
)


def opens_a_row(payload: str) -> bool:
    """Whether `payload` would start a new row rather than extend the last one.

    Judged on the FIRST non-blank line only: a row is routinely several lines
    long, and only its first line opens it.
    """
    for line in payload.splitlines():
        if line.strip():
            return ROW_START_PATTERN.match(line) is not None
    return False


class SectionError(ValueError):
    """The named section cannot be resolved unambiguously."""


class SectionEntry:
    """One row of a capped section: its heading line and its full text.

    `text` is the verbatim bytes of the row including its trailing blank
    lines, so a roll can move it without reformatting it. Rows are the unit a
    roll moves; lines are the unit a cap counts.
    """

    __slots__ = ("end_line", "heading", "start_line", "text")

    def __init__(self, heading: str, text: str, start_line: int, end_line: int) -> None:
        self.heading = heading
        self.text = text
        self.start_line = start_line
        self.end_line = end_line

    def digest(self) -> str:
        """Stable identity for this row -- heading AND body.

        Body is included on purpose: two rows can share a heading, and a row
        that was edited after being read is not the row that was read. A
        reader keyed on this digest detects both cases instead of silently
        resuming from the wrong place.
        """
        normalized = "\n".join(line.rstrip() for line in self.text.strip().splitlines())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


class ParsedSection:
    """A ledger split into everything-before-the-section and the section."""

    __slots__ = ("entries", "head", "heading_index", "heading_line", "preamble")

    def __init__(
        self,
        head: str,
        heading_line: str,
        preamble: str,
        entries: list[SectionEntry],
        heading_index: int,
    ) -> None:
        self.head = head
        self.heading_line = heading_line
        self.preamble = preamble
        self.entries = entries
        self.heading_index = heading_index

    def section_text(self) -> str:
        return self.heading_line + self.preamble + "".join(e.text for e in self.entries)

    def line_count(self) -> int:
        return len(self.section_text().splitlines())

    def byte_count(self) -> int:
        return len(self.section_text().encode("utf-8"))


def parse_section(text: str, heading: str) -> ParsedSection:
    target = heading.strip()
    lines = text.splitlines(keepends=True)
    hits = [i for i, line in enumerate(lines) if line.strip() == target]
    if not hits:
        raise SectionError(f"section heading not found in ledger: {target!r}")
    if len(hits) > 1:
        raise SectionError(
            f"section heading appears {len(hits)} times in ledger "
            f"(lines {[i + 1 for i in hits]}): {target!r} -- refusing to guess which one bounds the section"
        )
    index = hits[0]
    body = lines[index + 1 :]
    starts = [i for i, line in enumerate(body) if ROW_START_PATTERN.match(line)]
    preamble = "".join(body[: starts[0]]) if starts else "".join(body)
    entries: list[SectionEntry] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(body)
        entries.append(
            SectionEntry(
                heading=body[start].rstrip("\n"),
                text="".join(body[start:end]),
                start_line=index + 1 + start + 1,
                end_line=index + 1 + end,
            )
        )
    return ParsedSection(
        head="".join(lines[:index]),
        heading_line=lines[index],
        preamble=preamble,
        entries=entries,
        heading_index=index,
    )


def parse_section_file(path: Path, heading: str) -> ParsedSection:
    return parse_section(path.read_text(encoding="utf-8"), heading)


def section_entries(path: Path, heading: str) -> list[SectionEntry]:
    return parse_section_file(path, heading).entries


def section_line_count(path: Path, heading: str) -> int:
    return parse_section_file(path, heading).line_count()


def section_byte_count(path: Path, heading: str) -> int:
    return parse_section_file(path, heading).byte_count()


def _strip_pointer_block(preamble: str) -> str:
    kept = [
        line
        for line in preamble.splitlines(keepends=True)
        if ROLL_POINTER_MARKER not in line
        and not line.lstrip().startswith(ROLL_POINTER_PROSE_PREFIX)
    ]
    return "".join(kept)


def _repo_root_above(start: Path) -> Path | None:
    """The nearest ancestor of `start` holding a `.git` entry, or None.

    Filesystem-only on purpose: no `git` subprocess, so this answers the same
    way inside a worktree (where `.git` is a FILE, not a directory), on a
    machine with no git installed, and inside a test fixture that only needs a
    marker. `Path.parents` is finite, so there is no walk to bound.
    """
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _portable_path(path: Path, ledger: Path) -> str:
    """Render `path` for writing INTO a file that gets committed (OMN-17403).

    The roll used to write `str(path)` -- the absolute path of whatever
    machine and whatever worktree it happened to run in. On 2026-09-16 that
    put `/Users/<user>/Code/omni_home/omni_worktrees/OMN-17403/omni_home/...`
    into two committed lines of the rolling work ledger on `origin/main`: a
    path naming a worktree that is deleted when its ticket closes, on one
    machine, so the pointer a reader is meant to follow resolves for nobody.
    It is also a plain omni_home CLAUDE.md rule 6 violation.

    Preference order, and each fallback is still relative:

    1. Relative to the repository root above the ledger -- the form every
       citation in this fleet uses (`<ledger dir>/archive/...`, spelled from the
       repo root), and the form that survives being read from a clone, a
       worktree or GitHub. The example is written generically on purpose: this
       tool is TOLD which ledger to protect and must name no particular one,
       which is asserted by
       omni_home tests/test_ledger_lock_path_parametrization.py::
       test_ledger_path_is_a_positional_argument -- a test that greps this
       source, so an illustrative literal fails it exactly as a real hardcoded
       path would.
    2. Relative to the ledger's own directory, when there is no repository
       above it, or when the target sits outside that repository.

    The roll RECEIPT deliberately keeps absolute paths: it is operational
    stdout consumed by the trigger on the machine that produced it, it is
    never committed, and the absolute form is the useful one there.
    """
    root = _repo_root_above(ledger.parent)
    if root is not None:
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            pass
    return Path(os.path.relpath(path, ledger.parent)).as_posix()


def _pointer_block(
    archive: Path, rolled: int, rolled_at: str, first_kept: str, ledger: Path
) -> str:
    archive_ref = _portable_path(archive, ledger)
    marker = json.dumps(
        {
            "rolled_at": rolled_at,
            "archive": archive_ref,
            "entries_rolled": rolled,
            "first_kept_heading": first_kept,
        },
        sort_keys=True,
    )
    return (
        f"{ROLL_POINTER_MARKER} {marker} -->\n"
        f"{ROLL_POINTER_PROSE_PREFIX} `{archive_ref}` -- {rolled} rows rolled at {rolled_at}. "
        f"Rows older than {first_kept!r} are not in this file. "
        f"{ROLL_SHIFT_PLACEHOLDER}\n\n"
    )


def _first_line_of(text: str, needle: str) -> int:
    """The 1-based line number where `needle` starts in `text`.

    Used to measure the shift a roll introduces against the file's own bytes
    rather than by counting what the roll intended to remove -- the same
    readback posture OMN-17307 put on the reconcilers.
    """
    return text[: text.index(needle)].count("\n") + 1


def _shift_sentence(first_kept_line_before: int, shift: int) -> str:
    """The offset, as a sentence for the pointer line."""
    return (
        f"{ROLL_SHIFT_BANNER_PREFIX} lines above {first_kept_line_before} "
        f"shifted by -{shift}, so a `<path>:<line>` citation written before this "
        f"roll resolves at (line - {shift}); cite `<path>@<row timestamp>` "
        "instead, which a roll cannot move."
    )


class RollPlan:
    """A computed-but-not-yet-written roll.

    Planning and writing are separate so a caller can ask "would a roll make
    room for this append?" and refuse WITHOUT having moved anything. A roll
    that fires and then still refuses the append would leave the operator with
    a split file and no row, which is worse than either outcome alone.
    """

    __slots__ = ("archive_path", "archive_text", "live_text", "receipt", "rolled")

    def __init__(
        self,
        live_text: str,
        archive_text: str,
        archive_path: Path,
        receipt: dict[str, Any],
        rolled: int,
    ) -> None:
        self.live_text = live_text
        self.archive_text = archive_text
        self.archive_path = archive_path
        self.receipt = receipt
        self.rolled = rolled


def plan_roll(
    ledger: Path,
    heading: str,
    archive_dir: Path,
    keep_entries: int,
    rolled_at: str,
) -> RollPlan:
    if keep_entries < 1:
        raise SectionError("--roll-keep-entries must be at least 1")
    text = ledger.read_text(encoding="utf-8")
    parsed = parse_section(text, heading)
    lines_before = parsed.line_count()
    bytes_before = parsed.byte_count()
    archive_path = archive_dir / f"{ledger.stem}_{rolled_at[:10]}-split.md"

    if len(parsed.entries) <= keep_entries:
        receipt = {
            "schema": ROLL_RECEIPT_SCHEMA,
            "ledger": str(ledger),
            "section_heading": heading.strip(),
            "archive": str(archive_path),
            "rolled_at": rolled_at,
            "entries_rolled": 0,
            "entries_kept": len(parsed.entries),
            "first_kept_heading": parsed.entries[0].heading if parsed.entries else None,
            "last_rolled_heading": None,
            "section_lines_before": lines_before,
            "section_lines_after": lines_before,
            "section_bytes_before": bytes_before,
            "section_bytes_after": bytes_before,
        }
        return RollPlan(text, "", archive_path, receipt, 0)

    rolled = parsed.entries[:-keep_entries]
    kept = parsed.entries[-keep_entries:]
    first_kept = kept[0].heading

    preamble = _strip_pointer_block(parsed.preamble)
    if preamble and not preamble.endswith("\n"):
        preamble += "\n"
    # The offset is measured off the FINAL text, with the sentence standing in
    # as an inline placeholder. Inline is what makes this exact: substituting a
    # placeholder inside an existing line cannot change any line number, so the
    # figure recorded is true of the bytes finally written.
    live_text = (
        parsed.head
        + parsed.heading_line
        + preamble
        + _pointer_block(archive_path, len(rolled), rolled_at, first_kept, ledger)
        + "".join(entry.text for entry in kept)
    )
    anchor = kept[0].text
    first_kept_line_before = _first_line_of(text, anchor)
    shift = first_kept_line_before - _first_line_of(live_text, anchor)
    live_text = live_text.replace(
        ROLL_SHIFT_PLACEHOLDER, _shift_sentence(first_kept_line_before, shift), 1
    )

    archive_header = (
        "<!-- ledger-roll-archive "
        + json.dumps(
            {
                # Portable for the same reason the pointer block is: the
                # archive is a committed file, and an absolute path in it
                # names one machine's worktree (OMN-17403).
                "source": _portable_path(ledger, ledger),
                "section_heading": heading.strip(),
                "rolled_at": rolled_at,
                "entries": len(rolled),
                "first_rolled_heading": rolled[0].heading,
                "last_rolled_heading": rolled[-1].heading,
            },
            sort_keys=True,
        )
        + " -->\n"
        f"## Rolled from {ledger.name} at {rolled_at} -- {len(rolled)} rows\n\n"
    )
    archive_text = archive_header + "".join(entry.text for entry in rolled)

    after = parse_section(live_text, heading)
    receipt = {
        "schema": ROLL_RECEIPT_SCHEMA,
        "ledger": str(ledger),
        "section_heading": heading.strip(),
        "archive": str(archive_path),
        "rolled_at": rolled_at,
        "entries_rolled": len(rolled),
        "entries_kept": len(kept),
        "first_kept_heading": first_kept,
        "last_rolled_heading": rolled[-1].heading,
        "section_lines_before": lines_before,
        "section_lines_after": after.line_count(),
        "section_bytes_before": bytes_before,
        "section_bytes_after": after.byte_count(),
        "first_kept_line_before": first_kept_line_before,
        "line_shift": shift,
    }
    return RollPlan(live_text, archive_text, archive_path, receipt, len(rolled))


def _scrubbed_git_env() -> dict[str, str]:
    """`os.environ` without the git location variables (OMN-14891).

    This tool runs from lane shells and from git hooks. Git exports GIT_DIR and
    friends into every hook environment and they OVERRIDE `-C`, so an unscrubbed
    `git add` here would stage into whatever worktree invoked the hook rather
    than the one holding the ledger.
    """
    env = dict(os.environ)
    for key in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_COMMON_DIR",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    ):
        env.pop(key, None)
    return env


def stage_rolled_paths(ledger: Path, archive: Path) -> str:
    """Stage the archive and the rewritten ledger. Returns "" or a reason.

    THE ROLL MUST NOT LEAVE ROWS OUTSIDE GIT (OMN-18620). It writes the rolled
    rows into a NEW archive file, and on 2026-09-17 that file was created
    untracked: a lane then committed exactly the one path rule 19 tells it to
    commit, the ledger, and produced `1 file changed, 8 insertions(+), 928
    deletions(-)` -- a commit recording the removal of 928 rows with no record
    of where they went. The rows existed only as an untracked file, one
    clean-untracked away from being gone. Nothing in this tool's output said a
    roll had happened.

    Staging rather than committing, deliberately. Committing is `commit_lock.py`'s
    job and it holds a DIFFERENT lock; taking a commit here would mean two tools
    committing the same shared tree under two locks. Staging is enough to close
    the hole: a staged file is inside git, so it cannot be lost to a clean, and
    `commit_lock.py` refuses (exit 4) when the index carries paths the caller did
    not list -- which turns the next commit into a prompt naming the archive
    instead of a silent 928-row deletion.
    """
    try:
        top = subprocess.run(
            ["git", "-C", str(ledger.parent), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
            env=_scrubbed_git_env(),
        )
    except OSError as exc:
        return f"git could not be run ({exc})"
    if top.returncode != 0:
        return f"{ledger.parent} is not inside a git work tree"
    added = subprocess.run(
        ["git", "-C", str(ledger.parent), "add", "--", str(archive), str(ledger)],
        capture_output=True,
        text=True,
        check=False,
        env=_scrubbed_git_env(),
    )
    if added.returncode != 0:
        return f"git add failed: {(added.stderr or added.stdout).strip()}"
    return ""


def apply_roll(ledger: Path, plan: RollPlan) -> str:
    """Write the roll, stage both paths, and report what could not be staged.

    The return value is a REASON, empty when both paths are staged. It is not
    raised: the bytes are already on disk durably by then, so failing here would
    leave the operator with a completed roll and a traceback instead of a
    completed roll and an instruction.
    """
    if plan.rolled == 0:
        return ""
    plan.archive_path.parent.mkdir(parents=True, exist_ok=True)
    existing = ""
    if plan.archive_path.exists():
        existing = plan.archive_path.read_text(encoding="utf-8")
        if existing and not existing.endswith("\n"):
            existing += "\n"
        existing += "\n"
    _write_text_durably(plan.archive_path, existing + plan.archive_text)
    _write_text_durably(ledger, plan.live_text)
    return stage_rolled_paths(ledger, plan.archive_path)


def _write_text_durably(path: Path, text: str) -> None:
    """Replace `path` atomically, fsync'ing both the file and its directory.

    A roll rewrites the whole ledger. A partial write here loses rows, so the
    new bytes land in a sibling temp file that is fsync'd before the rename,
    and the rename itself is made durable by fsync'ing the directory.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.ledger_lock.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def announce_roll(plan: RollPlan, unstaged_reason: str) -> None:
    """Say on stderr that a roll happened, and whether it is inside git.

    A ROLL CANNOT BE SILENT (OMN-18620). Before this, a cap-crossing append
    printed its ordinary output and a JSON receipt line; nothing said in words
    that 926 lines had just left the file. The lane that hit it caught the loss
    only by reading `git commit`'s stat afterwards and noticing that 928
    deletions is an odd result for a four-row append.

    A plan that rolled nothing says nothing. `plan_roll` returns such a plan
    whenever the section is shorter than `--roll-keep-entries`, and its receipt
    carries no offset fields to report.
    """
    if plan.rolled == 0:
        return
    print(
        f"ledger_lock: ROLLED {plan.rolled} row(s) out of this section into "
        f"{plan.archive_path} -- the live file lost "
        f"{plan.receipt['line_shift']} lines, so every `<path>:<line>` citation "
        f"above line {plan.receipt['first_kept_line_before']} has moved. The "
        "pointer line at the top of the section records the offset.",
        file=sys.stderr,
    )
    if unstaged_reason:
        print(
            "ledger_lock: WARNING -- the archive is NOT staged "
            f"({unstaged_reason}). Commit it TOGETHER with the ledger, or the "
            "rolled rows exist only as an untracked file and a commit of the "
            "ledger alone records their deletion with no record of where they "
            f"went:\n  {plan.archive_path}",
            file=sys.stderr,
        )
    else:
        print(
            "ledger_lock: the archive and the ledger are both STAGED. Commit "
            "them together; commit_lock.py will refuse a commit that lists only "
            "one of them.",
            file=sys.stderr,
        )


def read_append_payload(args: argparse.Namespace) -> str | None:
    if args.append is not None:
        return str(args.append)
    if args.append_file is None:
        return None
    if args.append_file == "-":
        return sys.stdin.read()
    return Path(args.append_file).read_text(encoding="utf-8")


# --- OMN-18554: resolving the omni_home registry from inside this repo -----
#
# Three of the guards ported here (OMN-18258 ruling, OMN-18274 friction,
# OMN-18433 stranded-clone) load a COMMITTED helper module out of
# ``omni_home/docs/workflows/_shared/``. In omni_home's own copy of this script
# that directory was reachable as ``Path(__file__).parents[1]``; here it is not,
# because ``parents[1]`` is this repository's root.
#
# The helpers are deliberately NOT vendored into this repo. They are 995 lines
# of already-committed omni_home code, and copying them would create the exact
# two-divergent-copies condition this port exists to end. They are resolved
# instead, and every one of the three loaders already fails SOFT on absence --
# it announces on stderr and falls back to its inline constants, so a machine
# without the registry degrades loudly rather than silently.
#
# Resolution order, fail-soft by design (this is not a rule-8 required-env site:
# a missing registry must not break a ledger append on a CI runner):
#   1. $OMNI_HOME, the variable every lane already exports;
#   2. a walk up from this file for a directory that actually carries
#      docs/workflows/_shared -- proof by structure, not by name;
#   3. this repository's parent, which is where the registry sits by layout.
def _omni_home_root() -> Path:
    override = os.environ.get("OMNI_HOME")
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "docs" / "workflows" / "_shared").is_dir():
            return candidate
    return here.parents[2]


def _omni_home_shared() -> Path:
    return _omni_home_root() / "docs" / "workflows" / "_shared"


# --- OMN-15649: rule-4 cost-sentence enforcement at claim-append time -----
#
# Scope: the act-of-claiming row class (the rolling work ledger's
# rule 1a claim rows). Status/TERMINAL/merge-sweep/ruling rows are a different
# shape entirely and are never touched. Enforcement is scoped to whether the
# freshness window declared by the goal file's front matter is currently open
# (calendar-day scoped against that declared source) — not a
# permanent blanket check and not a hardcoded date literal in this script.
#
# RELAND (operator ruling R-0802-7, 2026-08-02): the first build (e1f0a3f49 +
# abfd293f5) was reverted after an adversarial verifier found six live
# defects (ticket comment 421adb65) plus an AC-7 gap discovered the same
# night (comment 0f63064a). This version folds all seven fixes in:
#   1. DISPLACES_PATTERN was backtracking-vulnerable: `displac\w*` with no
#      leading \b let the regex engine give back trailing characters of the
#      trigger word itself (e.g. "...; displaces" alone would backtrack to
#      "es"/"s" and satisfy the non-empty capture) — an EMPTY displaces
#      clause read as present. Fixed by anchoring the trigger word with \b on
#      both sides via a closed alternation (`\bdisplac(?:es|ing|ed|ement)\b`)
#      so there is nothing left to backtrack into.
#   2. The row-class matcher is grounded in a full read-only replay of the
#      live ledger (see tests/test_ledger_lock_cost_sentence.py), made
#      case-insensitive, and widened to the claim-verb family
#      (claim/claims/claimed/claiming) covering every live-cited shape:
#      the "[claim]"/"[claim+x]" tag, "[session] CLAIM"/"CLAIMED"/
#      "CLAIMING"/"claimed" (mixed case, all four verb forms live in the
#      real ledger), the rule-1a dot-separated shape ("ticket · session ·
#      claimed ·", also with em-dash separators), and a piped Work Claims
#      table row (including the "claimed — <status>" em-dash shape already
#      in live use for R-0802 rows). An earlier draft of this fix added a
#      fifth "unpiped table row" pattern; it was dropped after it was read
#      as producing a live false positive on Codex's §2b merge/landing
#      table.
#      [SUPERSEDED BY REMEDIATION r4 — that drop was wrong on the facts and
#      cost a whole row class. The one row it was justified by (L5241,
#      "Merge/Landing claim | OCC batch core/claude | claiming evidence
#      batch ...") is not a false positive: read against its own neighbours
#      L5236-L5244, that column is the Codex merge/landing state machine's
#      EVENT TYPE, whose live values are intent / rerun / verify / merged /
#      claim, and the body is textbook dedupe-then-claim ("no existing OCC
#      PR found"). Dropping the pattern left all 8 live bullet-less
#      pipe-field claim rows invisible. CLAIM_PIPE_FIELD_PATTERN below is
#      that pattern, re-added with the verb-adjacency discriminator every
#      other pattern here uses.]
#   3. No CI workflow or pre-commit hook ships with this change. The first
#      build's CI workflow could never fire (this branch takes no PRs/pushes
#      to main) and its pre-commit hook was itself rejected by this repo's
#      own no-functional-code hook — both were dead surfaces. The CLI
#      (scripts/ledger_lock.py, the one path every session already calls
#      per CLAUDE.md rule 1a) is the sole enforcement point.
#   4. The `-- COMMAND` write verb (used for the canonical §2 Work Claims
#      table edit) cannot be validated ahead of time — the command is
#      arbitrary and may rewrite the whole file. It is validated AFTER the
#      command runs, by diffing ledger content before/after and running the
#      same row-class + cost-sentence checks against every genuinely new
#      line.
#      [SUPERSEDED BY REMEDIATION r5 (2026-08-03, aim-judge corrective-f) —
#      this bullet originally ended "any violation prints a loud stderr
#      warning (the command's own exit code is still returned — reverting an
#      arbitrary multi-purpose editor session is a worse failure mode than a
#      loud warning)". That warn-only resolution left the canonical §2 edit
#      verb as a live bypass of the very gate `--append` enforces: the same
#      row, on the same clock, was rc=65-and-not-written via `--append` and
#      rc=0-and-written via `-- COMMAND`. A gate one verb honours and the
#      other narrates is not a gate. The editor path now HARD-REFUSES:
#      enforce_command_claims() restores the ledger's pre-command bytes and
#      main() returns 65. The stated objection — destroying an arbitrary
#      editor session — is answered by preserving the rejected content
#      verbatim in a `<ledger>.rejected-<UTC>` sidecar whose path is printed,
#      so a refusal costs the author a re-run and never their edits. Window
#      scoping and the grace state machine are byte-for-byte the predicates
#      `--append` uses, so the two verbs cannot drift apart again; the r5
#      test block in tests/test_ledger_lock_cost_sentence.py drives the real
#      editor path for each branch, and one test asserts the warn-only
#      helper is GONE from source rather than merely unreferenced.]
#   5. The three caller templates that emit ledger claims
#      (.claude/workflows/build-lane.js, ticket-dedupe-then-claim.js,
#      plan-governor-update.js) are updated in this same commit to emit a
#      conforming cost sentence and to handle rc=65.
#   6. --cost-unknown is now repeatable (one flag per offending row, matched
#      positionally to the order offending claim rows appear in the
#      payload) instead of one reason silently misapplied to every
#      offending row in a multi-row payload; a whitespace-only reason is
#      rejected at the argument-parsing layer, before any payload is read.
#   7. AC-7 (comment 0f63064a): a conforming cost sentence must contain the
#      OMN-XXXX ticket ID it prices. A price sentence that is well-formed
#      but keyed to a DIFFERENT ticket than the one the row claims (a
#      batch-claim artifact) or to a bare PR number is exactly the
#      blindness mode a ticket-id grep can never catch — closed at
#      claim-append time, the only point that sees both the row's subject
#      and its price sentence together.
#
# GRACE TRANSITION (operator ruling R-0802-7, non-negotiable, 2026-08-02):
# peer sessions are claiming right now with pre-gate templates that have
# never heard of a cost sentence. A LEGACY-SHAPE row — one that makes no
# attempt at a cost sentence at all (no "est ~N" marker, no "displac..."
# word anywhere) — gets a LOUD stderr warning and lands anyway (rc=0) until
# GRACE_DEADLINE_UTC below, then hard-refuses (rc=65) the same as any other
# violation. A NEW-SHAPE VIOLATION — a row that DID attempt a cost sentence
# but got it wrong (unfilled placeholder, empty displaces clause, or a
# sentence missing the ticket ID it prices) — refuses immediately regardless
# of the deadline: a session already emitting the new template gets no free
# pass on emitting it incorrectly. This makes bricking a live peer session
# impossible during the fleet-wide template rollout while still closing the
# gap for sessions that have already updated.
#
# GRACE COLLAPSED TO IMMEDIATE (operator FIRST ACTION, 2026-08-02): the
# deadline below is now a PAST instant, so a claim row that makes no attempt
# at a cost sentence hard-refuses (rc=65) NOW rather than warn-and-landing
# until 2026-08-03T12:00Z as originally staged.
#
# Grounding — this is a MEASURED collapse, not an assertion. The
# occ-seams-0802 measurement recorded in the ledger at 2026-08-02T15:19:17Z
# replayed the live ledger tail and found 8/8 same-day claim rows already
# conform to the new template; the only rows that would refuse are 2
# pre-written rows belonging to dead/parked handles. Zero live writers break.
# The check was re-run against the live ledger tail after this change (see the
# lane report) with the same result.
#
# The grace STATE MACHINE is deliberately retained rather than deleted: the
# legacy-shape (no attempt -> grace-eligible) vs new-shape-violation
# (attempted but wrong -> always immediate) split is load-bearing, and
# re-arming grace for a future fleet-wide template change must stay a
# one-line data change. Two tests keep the collapse honest rather than
# depending on this comment: tests/test_ledger_lock_cost_sentence.py's
# test_grace_deadline_is_in_the_past (asserts the constant is in the past
# against REAL wall clock, so bumping it forward fails loudly) and
# test_unpriced_claim_row_refuses_at_real_wall_clock_with_no_now_override
# (drives the real CLI with no LEDGER_LOCK_NOW override at all).
GRACE_DEADLINE_UTC = datetime(2026, 8, 2, 0, 0, 0, tzinfo=UTC)

# REMEDIATION r1 (2026-08-02, this pass): an independent adversarial replay
# of the reland (commit eaaaa4a38f8808b672f4785656547aa29a57e27a) against the
# LIVE ledger found the AC-7 binding itself was the next brick — the exact
# failure class the reland was built to close, now against correctly-priced
# rows. Folded in:
#   (a) CRITICAL/kill-criterion — AC-7 required the ticket ID to appear at or
#       AFTER the lane-hour token ("sentence_start = lane_hour_match.start()"
#       scoping). The dominant LIVE shape ("claimed OMN-XXXX ...; est ~N
#       lane-hours; displaces <what>") names the ticket BEFORE the price and
#       never repeats it — 15 of 19 real priced claim rows were hard-refused
#       by this alone. Replaced with claimed_ticket_id()'s structural
#       extraction (below) plus PRICE_SUBJECT_PATTERNS, which only rejects
#       when an EXPLICIT, different item-identifier (a second OMN-XXXX or a
#       bare PR shorthand like "core#1536") is bound to the price — the live
#       batch-claim blindness mode AC-7 exists to catch. A row whose own
#       ticket appears anywhere with no such conflicting subject is
#       unambiguously bound and never required to repeat it. [r2: the
#       original single PRICE_SUBJECT_PATTERN only recognized the subject
#       sitting in literal whitespace adjacency immediately before "est" —
#       see REMEDIATION r2 (b) below for the binding-syntax holes this
#       missed and the broadened PRICE_SUBJECT_PATTERNS that replaced it.]
#   (b) HIGH — row_ticket_id() picked the first "OMN-\\d+" ANYWHERE in the
#       row, so a row that merely mentions an unrelated ticket in passing
#       (live specimens: "...does NOT touch the OMN-15435 worktree..."; "...
#       comment OMN-15639 with terminal chain.") was forced to price a
#       ticket it was never about. claimed_ticket_id() only recognizes a
#       ticket sitting in one of the positions CLAIM_ROW_PATTERNS structurally
#       expects one (right after a "[claim...]" tag, right after the claim
#       verb, or the table row's first column); a row with no ticket in any
#       of those positions is AC-7-exempt, same as a genuinely ticket-less
#       plan-governor claim.
#   (c) HIGH (test coverage) — the peer-safety replay test asserted `rc` only
#       for the has_cost_sentence_attempt()==False branch, so the exact
#       population defect-(a) breaks (priced rows) was never checked by any
#       test; 46 tests passed with the gate broken. Both the existing replay
#       loop and a new dedicated full-population regression test now assert
#       `rc` against cost_sentence_reason()'s own prediction for every priced
#       claim row too.
#   (d) MEDIUM — AC-4 was calendar-scoped only: every claim row was checked
#       merely because "today" fell inside the plan's declared **Window:**,
#       regardless of whether the CLAIMED TICKET itself is part of the dated
#       chain. read_declared_chain_tickets() parses the plan's own "0-CHAIN"
#       dated-milestone-chain section (rule 1 TARGET-FIRST: the plan's first
#       content section while a chain is open) for the declared in-window
#       ticket set; a claim naming a ticket outside that set (the
#       out-of-window "Backlog ... non-displacing" shape) now lands unpriced
#       even inside the calendar window. No chain section declared (as in
#       every test fixture here) falls back to the prior calendar-only scope
#       — never silently narrower than before.
#       [SUPERSEDED BY REMEDIATION r2 (a) below — this bullet is kept only
#       as the historical record of why the (now-removed) chain-scoping
#       code existed. AC-4's own text ("The refusal fires only for claims
#       naming a ticket inside the open dated window ... The window boundary
#       is read from a declared source ... never hardcoded to a date
#       literal") is calendar-only; it says nothing about per-ticket
#       dated-chain membership. Narrowing to the "0-CHAIN" prose section was
#       an unrequested reading that regressed live enforcement to 11 of 206
#       ticket-bearing claim rows (5.3%) — a CRITICAL kill-criterion, the
#       exact class of silent gate inertness OMN-15649 exists to close.
#       Removed; enforcement is calendar-window-only again, matching AC-4 as
#       written and the pre-r1 baseline.]
#   (e) MEDIUM — CLAIM_DOT_FIELD_PATTERN's unbounded `.*` let backtracking
#       find a "· claimed ·" fragment QUOTED far inside an unrelated
#       status/audit row's free text (live specimen: an [aim-tick-...]
#       RULE-4 audit row that quotes another session's claim verbatim ~1200
#       characters in) and misclassify it as a claim row. Bounded to the
#       first 300 characters after the timestamp — verified against the full
#       live ledger to drop exactly that one false positive and zero
#       genuine dot-shape claim rows (the longest genuine gap measured was
#       276 characters).
#   (f) LOW — resolve_now() crashed with an uncaught
#       "can't compare offset-naive and offset-aware datetimes" TypeError
#       (raw traceback, rc=1) on a timezone-naive LEDGER_LOCK_NOW. A naive
#       override is now treated as UTC, same as the documented "Z" form.
#   (g) LOW — matcher recall holes probed post-grace: leading whitespace
#       before the bullet, a "*" bullet instead of "-", the claim-verb
#       synonym "taking", and a piped table row whose claim verb leads a
#       column other than exactly column 3. All four are now recognized;
#       verified against the full live ledger to add no false positives
#       beyond two already-real historical claim-shaped rows.
#   (h) INFO — "est 0 lane-hours" is numerically well-formed but a zero-cost
#       estimate is the same ritual-string problem AC-5 exists to reject;
#       now rejected alongside the unfilled-placeholder case.
#
# REMEDIATION r2 (2026-08-02, this pass): a third adversarial replay of the
# r1 remediation against the LIVE ledger AND the LIVE rolling plan found r1
# (d)'s AC-4 chain-scoping was itself a regression — the exact "gate goes
# silently inert" failure class OMN-15649 exists to close, now against
# almost the whole live claim population instead of a handful of rows.
# Folded in:
#   (a) CRITICAL/kill-criterion — read_declared_chain_tickets() (r1 (d))
#       narrowed enforcement to only the OMN-XXXX tokens textually present
#       in the plan's "## 0-CHAIN." section (23 live tickets); of 206
#       ticket-bearing claim rows in the live ledger, only 11 (5.3%) stayed
#       enforced and 195 (94.7%) silently landed unpriced even inside the
#       open calendar window and even for the exact recurrence population
#       (OMN-15610/11/12/13/14, 15622/24/25/26, 15600/01, 15621/34/37/42/45)
#       this ticket's own body names. AC-4's text is calendar-only (see the
#       superseded-bullet note on r1 (d) above) — the chain-membership
#       reading was never grounded in the AC. read_declared_chain_tickets(),
#       CHAIN_SECTION_HEADING_PATTERN, and NEXT_SECTION_HEADING_PATTERN are
#       removed outright; validate_claim_payload() is calendar-window-only
#       again, matching the pre-r1 baseline and AC-4 as written.
#   (b) HIGH — AC-7's PRICE_SUBJECT_PATTERN (r1 (a)) only recognized a
#       conflicting subject in literal whitespace adjacency immediately
#       before "est" ("OMN-XXXX est ..."). Three further live binding
#       shapes named in the ticket's own recorded blindness mode were
#       missed entirely (rc=0 on all of them, on a row keyed to a DIFFERENT
#       ticket than the one it claims): a colon/dash delimiter before "est"
#       ("OMN-XXXX: est ...", "OMN-XXXX — est ..."), the price naming its
#       subject via a trailing "for" clause ("est ~N lane-hours for
#       OMN-XXXX"), and — the shape the caller templates themselves emit —
#       a trailing parenthetical binding ("est ~N lane-hours; displaces
#       <what>; (OMN-XXXX)"). PRICE_SUBJECT_PATTERNS (plural) now covers all
#       four binding syntaxes; cost_sentence_reason() flags a conflict if
#       ANY of them names a subject that disagrees with the row's own
#       ticket, not just the first pattern's match. The correct positive
#       case (a trailing paren naming the row's OWN ticket) is unaffected —
#       it was already accepted before r2 and stays accepted.
#   (c) MEDIUM — warn_on_unvalidated_command_claims() (the `-- COMMAND`
#       write-verb's post-hoc check) evaluated every added claim row
#       unconditionally, never checking whether the declared window was
#       even open — while validate_claim_payload() (the `--append` write
#       verb) always did. Same row, same clock: `--append` outside the
#       window landed silently (correct per AC-4); the equivalent write via
#       `-- COMMAND` printed a loud "UNVALIDATED CLAIM ROW" false alarm.
#       Both write verbs now share one window-resolution helper
#       (resolve_enforcement_window()) so they apply identical scoping.
#   (d) LOW — GRACE_DEADLINE_UTC was defined twice with identical values (r1
#       inserted its comment block between the reland's declaration and a
#       second, redundant re-declaration). The duplicate is removed; the
#       sole surviving definition is the original one, immediately above
#       this REMEDIATION r2 comment block (see the "GRACE TRANSITION"
#       paragraph earlier in this file).
#   (e) LOW — the two caller templates that emit the documented trailing
#       "(<ticket_id>)" cost-sentence shape (.claude/workflows/build-lane.js,
#       ticket-dedupe-then-claim.js) stated the trailing ticket mention was
#       unconditionally "load-bearing" / "required" — false since r1 (a)
#       (a row whose own ticket is already named earlier, unrepeated, is
#       unambiguously bound and accepted). Corrected in the same commit to
#       describe the real rule: still emit the trailing "(<ticket_id>)" (it
#       is the documented conforming shape and never wrong to include), but
#       if a sentence explicitly names a *different* subject in one of the
#       PRICE_SUBJECT_PATTERNS binding positions, it must match the row's
#       own ticket or AC-7 rejects it.
#   (f) MEDIUM/LOW (test coverage) — no test drove the gate against the REAL
#       rolling seven-day plan then under docs/plans/ (retired 2026-09-18,
#       OMN-18751); every existing full-ledger-replay test monkeypatched the
#       plan-path override to a synthetic
#       no-chain fixture, so 65 tests passed while the live plan's declared
#       chain section silently exempted 94.7% of live claim rows. A new test
#       drives ledger_lock.main() against the unmodified DEFAULT_PLAN_PATH.
#       Separately, two live-ledger replay tests asserted aggregate
#       properties (row counts, landed-vs-refused balance) directly against
#       the live, peer-mutable ledger file — a concurrent session appending
#       a malformed claim row could flip those assertions with no code
#       change. Both now replay a frozen, verbatim snapshot of real ledger
#       specimens instead of re-reading the live file on every run.

# --- OMN-18554: the window source moved, and an unresolvable window now
# --- refuses instead of going silently inert -------------------------------
#
# Measured 2026-09-17: DEFAULT_PLAN_PATH used to be the rolling seven-day plan
# under OMNI_HOME/"docs"/"plans"/ (that plan is itself retired as of
# 2026-09-18, OMN-18751; see the block below). omni_home#341
# (17bbbac6d67048fb1aa9b92d149518e8dd01cc55, 2026-09-16T23:46:23Z) deleted that
# file when the plans corpus migrated to knowledge-base-internal under
# OMN-16978, and no successor was written at that path. read_declared_window()
# therefore returned None on EVERY invocation, and resolve_enforcement_window()
# treated "no window could be resolved" as identical to "we are outside an open
# window" -- print a warning, return False, accept the row. 143 of the 146
# CLAIM rows in the 24h before OMN-18554 was filed carry no cost sentence.
#
# Two changes close that, and they are separate:
#
#   (a) SOURCE. The plan is resolved from the knowledge-base-internal clone
#       (rule 20's destination for plans), via KNOWLEDGE_BASE_INTERNAL_PATH,
#       fail-fast on unset -- the same resolution the morning workflows use
#       (.claude/workflows/morning-ground-state.js KB_ROOT). There is NO
#       default path and NO omni_home fallback (rule 8): a silent default that
#       resolves to a path nobody maintains is exactly how this gate went
#       inert, so the failure mode is refused rather than re-created. The old
#       constant is DELETED, not repointed, so nothing can reach for it.
#
#   (b) STATES. The boolean resolve_enforcement_window() collapsed four
#       distinguishable conditions into one `return False`. They are now four:
#
#         WINDOW_OPEN       today inside the declared window -> ENFORCE
#         WINDOW_STALE      the window's end date is in the past -> ENFORCE,
#                           loudly. The rule-4 requirement is that a claim row
#                           carries its price; the declared window supplies the
#                           DENOMINATOR that price is read against. A lapsed
#                           denominator is a reason to shout for a re-cut, never
#                           a reason to stop asking for the price -- and a gate
#                           that silently switches itself off seven days after
#                           the last re-cut is the same defect this ticket is
#                           closing, on a timer.
#         WINDOW_PENDING    the window has not started yet -> do not enforce.
#                           Nothing is "inside" a window that has not opened,
#                           and this is the state that keeps "outside an open
#                           window" observably different from "unresolvable"
#                           (OMN-18554 AC4).
#         WINDOW_UNRESOLVED the declared source could not be read, or carries no
#                           parseable date line -> REFUSE claim rows, naming the
#                           path and the cause. CLAUDE.md rule 16: a gate that
#                           cannot read its own input has not passed, it has
#                           not run. Non-claim rows (TERMINAL/NOTE/RULING/
#                           PROGRESS) are untouched -- they were never in this
#                           gate's scope and blocking them would take the
#                           fleet's coordination surface down rather than price
#                           it. LEDGER_LOCK_ALLOW_INERT=1 is the explicit, loud
#                           escape for a machine that genuinely has no clone.
#
# DIVERGENCE FROM OMN-18554 AC4, recorded rather than buried: AC4's
# parenthetical asks for a CLOSED (past) window to land unenforced. That is the
# WINDOW_STALE branch, and it enforces here instead, per the build direction
# this change was dispatched under. AC4's binding first clause -- unresolvable
# and out-of-window must be distinguishable -- is satisfied by WINDOW_PENDING,
# which lands unenforced and is pinned by
# tests/test_ledger_lock_cost_sentence.py::test_unresolvable_and_out_of_window_
# are_not_the_same_outcome. The divergence is stated on the ticket; AC4 is not
# ticked on this build.
# --- OMN-18751: the window source is the goal file, and the plan it replaced
# --- is retired rather than re-cut ----------------------------------------
#
# Operator ruling, 2026-09-18T18:32:57Z, verbatim "retire it", recorded at
# ROLLING_WORK_LEDGER.md:4469 in omni_home's tracking directory. That
# directory is named here in prose rather than spelled as a path, because
# tests/test_ledger_lock_path_parametrization.py greps this whole file for
# the literal and a citation is not a hardcoded ledger location -- but a
# grep cannot tell the two apart, and the check is right to be blunt about a
# tool whose entire premise is that it is TOLD which ledger to protect.
# OMN-18757 restored the green: the literal arrived in a comment with
# OMN-18751 (43045ce9a) and left that test red on dev. The hand-maintained rolling
# seven-day plan OMN-18554 pointed this gate at is retired. It was 43 days
# stale when the ruling landed, and its staleness was load-bearing: this gate
# reads it for the DENOMINATOR every rule-4 price is measured against, so
# every append on the fleet carried a staleness banner for six weeks and,
# before OMN-18554, the gate was inert entirely.
#
# Re-cutting it a second time was the alternative and was rejected on the
# ruling. A hand-maintained week is a document somebody has to remember to
# rewrite, and this gate is the proof of what happens when nobody does. The
# standing surfaces are `beta/GOAL.md` -- rows, rungs, falsifiers and a
# `state_as_of:` stamp, rewritten by the morning ground-state workflow rather
# than by hand -- and the Program Board, which carries the day-by-day
# sequencing view as a COLUMN rather than as prose in a third file.
#
# WHAT CHANGES HERE, precisely: the source moves and the parse moves with it.
# Nothing else does. The five window states keep their meanings, the refusal
# and announcement branches keep their shapes, and the enforcement behaviour of
# every state is byte-for-byte what OMN-18554 shipped:
#
#   * SOURCE: `beta/GOAL.md` in the knowledge-base-internal clone, still via
#     KNOWLEDGE_BASE_INTERNAL_PATH, still fail-fast on unset, still no default
#     and no omni_home fallback (rule 8).
#   * PARSE: a `state_as_of: YYYY-MM-DD` line, not a `**Window:** a → b` line.
#     One date, not two.
#   * HORIZON: the second date is SYNTHESIZED -- `state_as_of` plus
#     FRESHNESS_HORIZON_DAYS. A goal file re-measured within the horizon is
#     WINDOW_OPEN; past it, WINDOW_STALE, which still enforces and now says so
#     in the vocabulary of a re-measurement rather than a re-cut. A stamp dated
#     in the FUTURE is WINDOW_PENDING and lands unenforced, which is what keeps
#     "not current" observably different from "unresolvable" (OMN-18554 AC4).
#
# Synthesizing the horizon rather than removing the window tuple is deliberate:
# the states, their tests and their messages are all expressed over a (start,
# end) pair, and re-expressing them over a single date would rewrite five
# branches to change one input. The freshness question -- "was this measured
# recently enough for a price read against it to mean anything" -- is exactly
# the question the window asked.
GOAL_PATH_ENV = "LEDGER_LOCK_GOAL_PATH"
KB_INTERNAL_ROOT_ENV = "KNOWLEDGE_BASE_INTERNAL_PATH"
ALLOW_INERT_ENV = "LEDGER_LOCK_ALLOW_INERT"
NOW_OVERRIDE_ENV = "LEDGER_LOCK_NOW"
GOAL_RELATIVE_PARTS = ("beta", "GOAL.md")

# How recently the goal file must have been re-measured for the denominator it
# supplies to mean anything. Seven days because that is the cadence the
# retired plan declared and the ruling preserved; the number is named here and
# pinned at its boundary by a test, so it cannot drift by one in a refactor.
FRESHNESS_HORIZON_DAYS = 7

WINDOW_OPEN = "open"
WINDOW_STALE = "stale"
WINDOW_PENDING = "pending"
WINDOW_UNRESOLVED = "unresolved"
# A FIFTH state, added after CI caught what a developer machine cannot: a GitHub
# runner has no knowledge-base-internal clone and no KNOWLEDGE_BASE_INTERNAL_PATH,
# so the plan is not merely unreadable -- the environment has no way to obtain it.
#
# That is a different condition from WINDOW_UNRESOLVED and must not share its
# refusal. The defect this gate closes was lanes on machines that DO have the
# registry landing unpriced rows; refusing on a runner that never could have read
# the plan does not price a single one of those rows, it just stops CI appending.
#
# The split is: an explicitly-pointed plan that does not resolve is a MISCONFIGURED
# lane machine and still REFUSES; an absent registry is an environment that cannot
# host the gate and lands with a loud announcement. The announcement matters --
# unsetting the variable would otherwise be a quieter bypass than the sanctioned
# LEDGER_LOCK_ALLOW_INERT one, which is the shape of hole this whole ticket is about.
WINDOW_NO_REGISTRY = "no_registry"


class GoalSourceUnresolvedError(RuntimeError):
    """The goal file this gate reads its freshness from cannot be located.

    Raised only by goal_path_for_window(); every call site inside this module
    converts it into a WINDOW_NO_REGISTRY resolution carrying the reason, so a
    missing clone is a named announcement rather than a traceback.
    """


# The goal file's front-matter stamp, on its own line at the head of the file:
#   state_as_of: 2026-09-18 (second measurement, lane ...)
# Anchored to the line start so a `state_as_of` discussed in the body -- or
# quoted inside a fenced block further down -- cannot be mistaken for the
# declaration. The trailing parenthetical the live file carries is ignored.
STATE_AS_OF_PATTERN = re.compile(r"^state_as_of:\s*(\d{4}-\d{2}-\d{2})", re.MULTILINE)

# Row-class matchers (AC-3/AC-2). Each is grounded against a cited live
# specimen in tests/test_ledger_lock_cost_sentence.py's full-ledger replay.
# All case-insensitive and cover the claim/claims/claimed/claiming verb
# family — the live ledger uses all four forms depending on author/session.
#
# r1 (g): bullet prefix now tolerates leading whitespace and a "*" bullet in
# addition to "-", CLAIM_SESSION_PATTERN also accepts the "taking" synonym,
# and CLAIM_TABLE_ROW_PATTERN accepts the claim verb leading column 3 OR
# column 4 (not only exactly column 3) — each verified against the full live
# ledger to add zero new false positives (the header row's "Claimed" column
# name and the Codex-owned §2b merge/landing table row both stay excluded).
#
# REMEDIATION r4 (2026-08-02, this pass) — BULLET REQUIREMENT WAS ITSELF A
# WHOLE-CLASS RECALL HOLE. Every pattern below was anchored on a MANDATORY
# "-"/"*" bullet. The live ledger carries 1498 timestamped rows that begin
# with the bare timestamp and NO bullet (the Codex-authored families: the
# "<ts> [handle] CLAIM OMN-XXXX ..." lane rows and the "<ts> | <field> |
# <field> | claimed ..." §2/§2b merge-controller rows). None of them were
# ever inspected by this gate — not warned, not refused, never even reaching
# is_rule4_claim_row's body. Measured live: making the bullet OPTIONAL adds 17
# genuine act-of-claiming rows and exactly one false positive, L12 — a
# markdown ORDERED-LIST item in this ledger's own instructional preamble
# ("1. Read the **Work Claims** table ... (`ticket · your session handle ·
# claimed · timestamp`)") which quotes the rule-1a claim shape as
# documentation. The `(?!\d+[.)]\s)` guard excludes exactly that class (a
# numbered list marker is never a ledger event row) and nothing else.
ROW_PREFIX = r"^\s*(?:[-*]\s+)?(?!\d+[.)]\s)"

# r4: the tag family, not the two literals "[claim]"/"[claim+x]". Live tags
# that are acts of claiming and were previously unrecognized: "[handoff-claim]"
# (worktree ownership transfer to a new owning lane — L11954, a genuine claim
# by a different session than the one that opened the ticket). Deliberately
# EXCLUDED: "[claim-correction]", whose two live specimens (L11920, L11924)
# correct a display timestamp and a worktree-path clause of an ALREADY-PRICED
# claim row — gating those would demand a second, duplicate price for work
# already priced at its real claim row, and would refuse a routine correction.
CLAIM_TAG_PATTERN = re.compile(
    ROW_PREFIX
    + r"\S+\s+\[(?![^\]]*claim-correction)[^\]]*\bclaim(?:s|ed|ing)?\b[^\]]*\]",
    re.IGNORECASE,
)
CLAIM_SESSION_PATTERN = re.compile(
    ROW_PREFIX + r"\S+\s+\[[^\]]+\]\s+(?:claim(?:s|ed|ing)?|taking)\b", re.IGNORECASE
)
# r1 (e): bounded to 300 characters after the timestamp (was unbounded `.*`,
# which let backtracking find a "· claimed ·" fragment QUOTED deep inside an
# unrelated status/audit row's free text — live false positive, see the r1
# comment block above). 300 is well above the longest genuine dot-shape gap
# measured live (276 chars) and well below the false positive's ~1200.
CLAIM_DOT_FIELD_PATTERN = re.compile(
    ROW_PREFIX + r"\S+\s+.{0,300}?[·—]\s*claim(?:s|ed|ing)?\s*(?:[·—]|$)", re.IGNORECASE
)
CLAIM_TABLE_ROW_PATTERN = re.compile(
    r"^\s*\|(?:[^|]*\|){2,3}\s*claim(?:s|ed|ing)?\b", re.IGNORECASE
)

# r4: the bullet-less, timestamp-first PIPE-FIELD row — the Codex §2/§2b
# merge-controller ledger shape ("<ts> | <lane/handle> | <subject> | claimed
# <what> ..."). CLAIM_TABLE_ROW_PATTERN cannot see it: that pattern requires a
# LEADING "|" (a markdown table row), and these rows lead with the timestamp.
# The discriminator is the same verb adjacency every other pattern here uses —
# the claim verb must LEAD a pipe-delimited field, not sit inside one's prose.
# Live census: 8 rows match and all 8 are genuine acts of claiming (L5241,
# L9733, L10461, L10466, L10871, L11166, L11181, L11206); zero narrative.
#
# OMN-18554 PRECISION FIX — `claim=` IS A CITATION FIELD, NOT A CLAIM VERB.
# Measured over the live ledger 2026-09-17: of the 194 rows is_rule4_claim_row
# recognized, 151 were TERMINAL / PROGRESS / NOTE rows carrying a
# `claim=<ledger path>:<line>` field — the citation a
# closing row uses to point AT the claim it answers. The verb-adjacency
# discriminator cannot tell `| claimed <what>` (an act) from `| claim=<path>`
# (a key=value field) because `\b` sits happily before `=`. 52 of those rows
# were written in the three days to 2026-09-17, and every one would have been
# REFUSED as an unpriced claim row the moment enforcement resumed — i.e. the
# closeout path for the whole fleet. That false-positive class was invisible
# while the gate was inert, and turning the gate back on is exactly what
# exposes it, so it is fixed in the same change rather than shipped as a
# regression. The guard is a negative lookahead on the assignment, nothing
# wider: `claim=` and `claims=` stop matching; `claimed <what>`, `claiming
# <what>` and a bare `claim` field are untouched.
# --- OMN-18554 item 2: the DOMINANT live CLAIM shape, on a dated cutover ----
#
# Measured over the live ledger 2026-09-17: 714 rows lead with a timestamp whose
# FIRST pipe field is the claim verb -- `<ts> | CLAIM | lane=...`, the shape every
# orchestrator dispatch brief writes. CLAIM_PIPE_FIELD_PATTERN below cannot see
# one of them: it requires at least one pipe field BEFORE the verb, and here the
# verb IS the first field. So the dominant row class never reached this gate at
# all -- not warned, not refused, never inspected.
#
# Recognizing it and refusing on the same day would have been an outage, not a
# gate: only 6 of the 72 rows written in this shape on 2026-09-17 carry a
# conforming cost sentence, so ~92% of the fleet's live claim appends would have
# started failing in one commit. So recognition lands NOW and enforcement lands
# on a DATE:
#
#   before RULE4_PIPE_LEAD_CUTOVER_UTC -- the row LANDS, and an unpriced one
#       prints a `RULE-4 UNPRICED CLAIM` line naming the lane, so the population
#       is visible and shrinking while the templates roll;
#   at or after it -- the row is refused exactly like every other claim shape.
#
# The date is the ONLY switch. There is deliberately no environment flag and no
# opt-in: a flag is a thing lanes set and forget, and the whole defect this
# ticket closes is a gate that was off while everyone assumed it was on.
RULE4_PIPE_LEAD_CUTOVER_UTC = datetime(2026, 9, 24, 0, 0, 0, tzinfo=UTC)

# The verb leads the FIRST pipe field. Same three precision guards the general
# pipe-field pattern carries: no compound token before it (`closes-CLAIM`), no
# assignment after it (`claim=`, `claim-token`), and the row's own timestamp
# immediately ahead of it so a claim quoted inside another row's prose cannot
# match.
CLAIM_PIPE_LEAD_PATTERN = re.compile(
    r"^\s*\d{4}-\d{2}-\d{2}\S*\s*\|\s*(?<![\w-])(?:claim(?:s|ed|ing)?|taking)\b(?![\s]*[=\-])",
    re.IGNORECASE,
)


def is_pipe_lead_claim_row(line: str) -> bool:
    """The dominant live shape, recognized separately from the settled matchers
    so its dated grace is expressible and so it can never silently widen the
    population the other patterns already refuse."""
    return bool(CLAIM_PIPE_LEAD_PATTERN.match(line))


def pipe_lead_grace_active(now: datetime) -> bool:
    return now < RULE4_PIPE_LEAD_CUTOVER_UTC


CLAIM_PIPE_FIELD_PATTERN = re.compile(
    # Each leading field is bounded to 120 characters, the same defence r1 (e)
    # gave CLAIM_DOT_FIELD_PATTERN against a claim row QUOTED inside another
    # row's prose. Live specimen this drops: the 2026-09-16T00:43:27Z
    # CORRECTION row, which quotes a peer's `| CLAIM | lane=...` verbatim ~255
    # characters in while correcting a mis-cited line number. Measured over the
    # full live ledger: the bound removes that one row and keeps all 6 genuine
    # pipe-field claim rows (40/60/80/120 all drop it; 120 is the loosest bound
    # that does, so it is the one taken).
    r"^\s*\d{4}-\d{2}-\d{2}\S*\s*(?:\|[^|\n]{0,120}){1,5}\|\s*(?<![\w-])(?:claim(?:s|ed|ing)?|taking)\b(?![\s]*[=\-])",
    re.IGNORECASE,
)

# PAREN-HANDLE RECALL HOLE (operator FIRST ACTION, 2026-08-02): every pattern
# above requires a BRACKETED session handle, so the paren-handle claim shape —
# "- <ts> (handle) CLAIM: ..." and "- <ts> (`handle`): CLAIMED OMN-XXXX ..." —
# was not recognized as a claim row AT ALL. Not warned, not refused: never
# even inspected. Live census over the full
# ROLLING_WORK_LEDGER.md (2026-08-02): 48 rows match the shape below and all
# 48 are genuine acts of claiming.
#
# The precision problem this must not create: the same file carries 1401 rows
# of the "(handle): <prose>" event shape, overwhelmingly NARRATIVE, many of
# which contain a claim-family word somewhere in free text — a status row
# reporting that "this lane is already claimed by S-A", a TERMINAL row ending
# "(claims 5/6)", an event row reading "Adopted ledger; claimed the
# colonization drain". Gating those would refuse routine peer writes.
#
# DISCRIMINATOR: verb adjacency, exactly mirroring what the bracketed
# CLAIM_SESSION_PATTERN already requires — the claim verb must be the FIRST
# token after the handle token, across at most a single ":" delimiter. A row
# whose claim verb sits later, inside free text, is narrative and stays
# unrecognized; at that position the word is genuinely indistinguishable from
# a mention of someone else's claim, so recognizing it would be a guess.
# Both polarities are pinned by tests against verbatim live specimens (see
# test_paren_handle_* and test_full_live_ledger_replay_paren_handle_recall_
# and_precision).
CLAIM_PAREN_SESSION_PATTERN = re.compile(
    ROW_PREFIX + r"\S+\s+\([^)\n]+\)\s*:?\s+(?:claim(?:s|ed|ing)?|taking)\b",
    re.IGNORECASE,
)

# REMEDIATION r3 (2026-08-02): verb-adjacency alone left a MEASURED 10-row
# recall hole in the paren shape — including a live, recurring MACHINE
# template (the codex-cloud-delegation-0730 hostile-review row: "- <ts>
# (<handle>/<lane>_review): OMN-15571 exact 111f7cac independent hostile
# review CLAIMED; ..."), which would still land unpriced today. Those rows
# announce the claim with an ALL-CAPS status verb a short way into the body,
# after a ticket id and a few words of subject, instead of immediately after
# the handle.
#
# DISCRIMINATOR (two conditions, both measured against the live ledger, not
# asserted):
#   1. The claim verb is ALL-CAPS. The ledger's convention is to SHOUT the
#      act (CLAIM/CLAIMED/TERMINAL/FILED/VERIFIED) and to lowercase mere
#      mentions of one ("already claimed by S-A", "(claims 5/6)"). Of the 101
#      unrecognized paren rows carrying any claim-family word, 81 are
#      lowercase-only and are dominated by narrative; this pattern is
#      case-SENSITIVE on purpose and does not touch them.
#   2. The verb starts within CAPS_VERB_WINDOW characters of the body.
#      Live census: every one of the 11 paren rows with an ALL-CAPS claim
#      verb inside 200 chars is an act of claiming or a claim RESCISSION
#      (excluded below); beyond 200 the same population is a coin flip — 4
#      genuine deep-embedded claims vs 5 narrative reports ("PEER-CLAIM
#      NOTE:", "CLAIM COLLISION:", "ENFORCEMENT CLAIM NOW SATISFIED") — so
#      recognizing there would be a guess, not a rule. That residual is
#      deliberately left open and pinned by
#      test_paren_caps_verb_beyond_the_window_stays_unrecognized_residual.
#
# RESCISSION CARVE-OUT: a row whose claim verb is immediately qualified by a
# rescinding verb records a claim ENDING, not an act of claiming — live
# specimen L10975 "(fable-day-0731): WAVE-2 CLAIMS RESCINDED BY OPERATOR".
# Gating those would refuse a routine stand-down row. Only RESCINDED is
# live-grounded; the other three are the same act under different words.
#
# REMEDIATION r4 (2026-08-02, this pass): r3 built this discriminator for the
# PAREN handle only. The BRACKETED handle has the identical shape and the
# identical hole — measured live, 29 bracketed rows announce the claim with an
# ALL-CAPS verb a short way into the body ("[claude-omn14960] OMN-14960
# CLAIMED — ...", "[fable-plan-0727] OMN-15226 CLAIMED (A1 of OMN-15027 ...)",
# "[lane sweep-unsatisfiable-checks-0730] **CLAIM OMN-15540** — ...") and none
# of them was ever inspected. The rule is the same rule; it is now applied to
# both handle shapes through ONE helper instead of duplicated per shape.
#
# Widening to the bracketed population surfaced two narrative classes the
# paren population did not contain, both closed by measured guards rather than
# accepted as false positives (a false positive here REFUSES a peer's routine
# write, so the bar is precision-first):
#   * NOUN-PHRASE MENTIONS — the verb is the head of a noun phrase introduced
#     by a determiner or quantifier, i.e. a row talking ABOUT a claim rather
#     than making one: "CONFIRMED, THE LOAD-BEARING CLAIM: ..." (L8685),
#     "(A) THREE CLAIMS IN THE 02:14Z ROW ARE REFUTED" (L9776), "cost
#     sentences OWED AT THE CLAIM ROW" (L11834). CLAIM_NOUN_PHRASE_TAIL
#     rejects the verb when its immediate left context ends in a
#     determiner/quantifier optionally followed by ALL-CAPS modifier words.
#     Bare digits are deliberately NOT in that set: "OMN-14960 CLAIMED" would
#     otherwise read its own ticket number as a quantifier and silently drop 6
#     genuine rows (measured — that was this pass's own first draft).
#   * CLAIM-STATUS ROWS — the verb is followed by a word reporting the STATE
#     of an existing claim rather than making one: "CLAIM STANDS" (L8923),
#     "CLAIM STAYS RELEASED" (L9821), "CLAIM RELEASE" (L9764), "THE CLAIM ROW"
#     (L11834). Folded into the rescission carve-out's word list — same idea:
#     this row is not the act.
# Measured outcome over the full live ledger: 46 rows newly recognized, all 46
# genuine acts of claiming, ZERO previously-recognized rows lost.
CAPS_VERB_WINDOW = 200
CLAIM_HANDLE_PREFIX_PATTERN = re.compile(
    ROW_PREFIX + r"\S+\s+(?:\[[^\]\n]+\]|\([^)\n]+\))\s*:?\s+"
)
CAPS_CLAIM_VERB_PATTERN = re.compile(r"(?<![A-Za-z-])CLAIM(?:S|ED|ING)?(?![A-Za-z-])")
CLAIM_NOUN_PHRASE_TAIL = re.compile(
    r"\b(?:THE|A|AN|THIS|THAT|THESE|THOSE|ITS|OUR|THEIR|ANY|EACH|EVERY|NO|ALL|BOTH|SOME"
    r"|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)"
    r"(?:\s+[A-Z][A-Z0-9-]*)*\s+$"
)
CLAIM_NOT_AN_ACT_PATTERN = re.compile(
    r"^\s+(?:RESCINDED|WITHDRAWN|RELEASED|RELEASE|REVOKED|CLOSED|CLEARED"
    r"|STANDS|STAYS|ROW|ROWS|GAP|GAPS)\b"
)


def caps_verb_claim_row(line: str) -> bool:
    """True if `line` is a handle-prefixed row that announces the claim with
    an ALL-CAPS claim verb within CAPS_VERB_WINDOW characters of the body (r3
    for the paren handle, r4 for the bracketed handle — one rule, both
    shapes). See the comment block above for the two measured guards. A verb
    beyond the window is a named, tested recall residual, not an oversight."""
    prefix = CLAIM_HANDLE_PREFIX_PATTERN.match(line)
    if prefix is None:
        return False
    body = line[prefix.end() :]
    for verb in CAPS_CLAIM_VERB_PATTERN.finditer(body):
        if verb.start() > CAPS_VERB_WINDOW:
            return False
        if CLAIM_NOUN_PHRASE_TAIL.search(body[: verb.start()]):
            continue
        if CLAIM_NOT_AN_ACT_PATTERN.match(body[verb.end() :]):
            continue
        return True
    return False


CLAIM_ROW_PATTERNS = (
    CLAIM_TAG_PATTERN,
    CLAIM_SESSION_PATTERN,
    CLAIM_PAREN_SESSION_PATTERN,
    CLAIM_DOT_FIELD_PATTERN,
    CLAIM_TABLE_ROW_PATTERN,
    CLAIM_PIPE_FIELD_PATTERN,
)


def is_rule4_claim_row(line: str) -> bool:
    """True if `line` records an act of claiming work — any live-grounded
    shape in CLAIM_ROW_PATTERNS, the handle-prefixed ALL-CAPS shape
    caps_verb_claim_row recognizes, or (OMN-18554) the dominant pipe-lead
    shape — false for every other row class.

    The pipe-lead shape is recognized here so it REACHES the gate; whether it
    is refused or merely reported is decided by its dated cutover in
    validate_claim_payload, not by this predicate."""
    return (
        any(pattern.match(line) for pattern in CLAIM_ROW_PATTERNS)
        or caps_verb_claim_row(line)
        or is_pipe_lead_claim_row(line)
    )


LANE_FIELD_PATTERN = re.compile(r"\blane=([^\s|]+)")


def claim_row_lane(line: str) -> str:
    """The lane a claim row names, for the unpriced-claim report. Unknown is
    reported as such rather than guessed — a wrong lane name in a report is
    worse than an absent one, because someone will act on it."""
    match = LANE_FIELD_PATTERN.search(line)
    return match.group(1) if match else "<no lane= field>"


# The EN DASH in the range alternation is load-bearing, not a typo: the live
# ledger carries both "est ~3-4 lane-hours" and "est ~3–4 lane-hours", and
# dropping it would silently stop pricing every row written with the second.
LANE_HOUR_PATTERN = re.compile(
    r"~?\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?\s*(?:[a-zA-Z]+-)?lane-hours?"  # noqa: RUF001
)

# Strict, correctness-checking pattern (round-2 fix: anchored with \b on both
# sides of a closed verb alternation so there is no trailing substring left
# for the engine to backtrack into and misread as a non-empty clause).
DISPLACES_PATTERN = re.compile(r"\bdisplac(?:es|ing|ed|ement)\b\s*[:\-—]?\s*(\S.*)$")

# Loose ATTEMPT detectors — deliberately broader than the strict patterns
# above. Used only to route between the grace-eligible legacy-shape path (no
# attempt at all) and the always-strict new-shape path (attempted but
# wrong); never used to decide correctness.
EST_ATTEMPT_PATTERN = re.compile(r"\best\.?\s*~?\s*(?:\d|<)", re.IGNORECASE)
DISPLACES_ATTEMPT_PATTERN = re.compile(r"\bdisplac\w*\b", re.IGNORECASE)

# AC-7: the Linear ticket ID a row is about — used both to require the cost
# sentence name it, and (--cost-unknown binding is positional, not by this
# id — see build_parser) to report a useful reason.
TICKET_ID_PATTERN = re.compile(r"\bOMN-\d+\b")

# r1 (b): each pattern extracts a ticket ONLY from a position one of
# CLAIM_ROW_PATTERNS structurally expects one — right after a "[claim...]"
# tag or the dot-shape's leading field, right after the claim verb in the
# "[session] claimed/taking OMN-XXXX" shape, or a piped table row's first
# column. This is deliberately NOT "the first OMN-\d+ anywhere in the row"
# (the pre-r1 behavior): a row that merely mentions an unrelated ticket in
# free text (live specimens — "...does NOT touch the OMN-15435 worktree...";
# "...comment OMN-15639 with terminal chain.") has no match here and is
# correctly AC-7-exempt rather than forced to price a ticket it never
# claimed.
OWN_TICKET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(ROW_PREFIX + r"\S+\s+(?:\[[^\]]+\]\s+)?(OMN-\d+)\b"),
    re.compile(
        ROW_PREFIX + r"\S+\s+\[[^\]]+\]\s+(?:claim(?:s|ed|ing)?|taking)\s+(OMN-\d+)\b",
        re.IGNORECASE,
    ),
    # Paren-handle counterpart of the pattern above (2026-08-02). Recognition
    # without ticket binding would only half-close the paren hole: the row
    # would be gated for est/displaces but stay AC-7-exempt, so a price keyed
    # to a DIFFERENT ticket would still land on a paren-shaped claim.
    re.compile(
        ROW_PREFIX
        + r"\S+\s+\([^)\n]+\)\s*:?\s+(?:claim(?:s|ed|ing)?|taking)\s+(OMN-\d+)\b",
        re.IGNORECASE,
    ),
    # r3: counterpart for caps_verb_claim_row's paren shape, where the
    # ticket LEADS the body and the ALL-CAPS verb follows it ("(handle):
    # OMN-15571 exact 111f7cac independent hostile review CLAIMED; ..."). It
    # is the same "ticket in a structurally-expected position" rule as the
    # bracketed leading-ticket pattern at the top of this tuple, for the paren
    # handle. Without it those rows would be recognized-but-AC-7-exempt: gated
    # for est/displaces, yet still able to land a price keyed to a DIFFERENT
    # ticket. Ordered after the verb-adjacent paren pattern so the more
    # specific "verb then ticket" binding still wins where both could match.
    re.compile(ROW_PREFIX + r"\S+\s+\([^)\n]+\)\s*:?\s+(OMN-\d+)\b"),
    re.compile(r"^\s*\|\s*(OMN-\d+)\b"),
    # r4: counterpart for CLAIM_PIPE_FIELD_PATTERN's bullet-less §2/§2b
    # merge-controller shape, where the ticket follows the claim verb leading a
    # pipe field ("<ts> | codex-.../root | CLAIM OMN-14498 recurrence — ...").
    # Same "ticket in a structurally-expected position" rule as the bracketed
    # and paren verb-adjacent patterns above; without it those rows would be
    # recognized-but-AC-7-exempt and could still land a price keyed to a
    # DIFFERENT ticket. Ordered last so no earlier, more specific binding loses.
    re.compile(
        r"^\s*\d{4}-\d{2}-\d{2}\S*\s*(?:\|[^|\n]*){1,5}\|\s*(?:claim(?:s|ed|ing)?|taking)\s+(OMN-\d+)\b",
        re.IGNORECASE,
    ),
)

# r1 (a) / r2 (b): a "priced-for" subject token bound to the price is the
# live batch-claim shape where one row prices multiple items and each price
# is prefixed or suffixed with the identifier it belongs to. Only an
# EXPLICIT-subject shape like these is checked against the row's own ticket;
# a row whose ticket appears earlier with no such binding (the dominant
# "claimed OMN-XXXX ...; est ~N lane-hours" shape) is implicitly,
# unambiguously bound and is never required to repeat the ticket inside the
# sentence itself.
#
# r2 (b): r1 shipped exactly one of these (subject immediately before "est",
# with no separator) and missed three further live binding shapes named in
# the ticket's own recorded blindness mode (comment 0f63064a) — all fired
# rc=0 on a row keyed to a DIFFERENT ticket than the one it claims:
#   - a colon or dash delimiter between the subject and "est"
#     ("OMN-15653: est ~2 lane-hours", "OMN-15653 -- est ~2 lane-hours")
#   - the subject named via a trailing "for" clause after the lane-hour
#     phrase ("est ~2 lane-hours for OMN-15653")
#   - a trailing parenthetical binding — the exact shape the caller
#     templates themselves emit for the POSITIVE case
#     ("est ~2 lane-hours; displaces nothing; (OMN-15653)")
# All four are covered below; _priced_subjects() collects every match from
# every pattern so a conflict anywhere in the sentence is caught, not only
# the first one found.
PRICE_SUBJECT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(OMN-\d+|[A-Za-z][\w./-]*#\d+)\s*[:\-—]?\s*(?=est\.?\s*~?\d)", re.IGNORECASE
    ),
    re.compile(
        r"\blane-hours?\b\s*for\s+(OMN-\d+|[A-Za-z][\w./-]*#\d+)\b", re.IGNORECASE
    ),
    re.compile(r"\((OMN-\d+|[A-Za-z][\w./-]*#\d+)\)", re.IGNORECASE),
)


def _priced_subjects(line: str) -> list[str]:
    """Every explicit item-identifier found in one of PRICE_SUBJECT_PATTERNS'
    known price-binding positions, in the order each pattern's matches occur
    (subject-before-"est", the trailing "for SUBJECT" clause, then any
    trailing "(SUBJECT)" parenthetical). A row with none of these binding
    syntaxes present returns an empty list — nothing for AC-7 to conflict
    with, the dominant implicitly-bound shape."""
    subjects: list[str] = []
    for pattern in PRICE_SUBJECT_PATTERNS:
        subjects.extend(match.group(1) for match in pattern.finditer(line))
    return subjects


def has_cost_sentence_attempt(line: str) -> bool:
    """Loose classifier: did this row attempt to price itself at all?

    True routes the row to the always-strict new-shape-violation path
    (immediate refusal on any defect, no grace). False routes it to the
    legacy-shape path (grace-eligible warn-then-refuse per
    GRACE_DEADLINE_UTC). Deliberately loose so a malformed attempt cannot
    slip through as "legacy" and dodge immediate refusal.
    """
    return bool(EST_ATTEMPT_PATTERN.search(line)) or bool(
        DISPLACES_ATTEMPT_PATTERN.search(line)
    )


def claimed_ticket_id(line: str) -> str | None:
    """The ticket THIS row is claiming — see OWN_TICKET_PATTERNS. Returns
    None (AC-7-exempt) if no structurally-expected position holds an
    OMN-<number> token, even if one appears elsewhere in the row's free text.

    (The ticket placeholder is written OMN-<number> rather than spelled out
    with four literal X characters: the upstream unimplemented-code detector
    substring-matches a three-X marker inside docstrings, so the spelled form
    reads to it as unfinished code. Same discipline as CLAUDE.md rule 15 --
    never spell a literal that a machine parses.)"""
    for pattern in OWN_TICKET_PATTERNS:
        match = pattern.match(line)
        if match:
            return match.group(1)
    return None


def cost_unknown_reason_type(value: str) -> str:
    """argparse type= validator for --cost-unknown (AC-6): whitespace-only
    reasons are rejected before any payload is even read."""
    if not value.strip():
        raise argparse.ArgumentTypeError(
            "--cost-unknown REASON must not be empty/whitespace-only"
        )
    return value


def write_text_atomic(path: Path, text: str) -> None:
    """Whole-file durable replace, used only by the `-- COMMAND` refusal
    path to restore the ledger's pre-command bytes. Writes to a temp file in
    the same directory and renames, so a crash mid-restore can never leave
    the ledger truncated."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.restore.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    Path(tmp).replace(path)


def goal_path_for_window() -> Path:
    """Resolve the goal file this gate reads its freshness from (OMN-18751).

    ``LEDGER_LOCK_GOAL_PATH`` wins when set. Otherwise the goal file is read
    from the knowledge-base-internal clone named by
    ``KNOWLEDGE_BASE_INTERNAL_PATH`` — rule 20's destination, and the same
    variable the morning workflows and the SessionStart goal hook resolve.
    Unset raises: there is no default and no omni_home fallback (rule 8),
    because an omni_home default is the one that evaporated under OMN-16978
    and took this gate with it.
    """
    override = os.environ.get(GOAL_PATH_ENV)
    if override:
        return Path(override)
    root = os.environ.get(KB_INTERNAL_ROOT_ENV)
    if not root:
        raise GoalSourceUnresolvedError(
            f"{KB_INTERNAL_ROOT_ENV} is unset, so the rule-4 freshness source "
            f"({'/'.join(GOAL_RELATIVE_PARTS)} in the knowledge-base-internal clone) "
            "cannot be located. Export it to the absolute path of that clone, or set "
            f"{GOAL_PATH_ENV} to a file that declares a 'state_as_of:' line. There is no "
            "default path and no fallback — a silent default is how this gate went inert."
        )
    return Path(root).joinpath(*GOAL_RELATIVE_PARTS)


def read_declared_window_detail(
    goal_path: Path,
) -> tuple[tuple[date, date] | None, str | None]:
    """``(window, cause)`` — exactly one is None.

    The window is ``(state_as_of, state_as_of + FRESHNESS_HORIZON_DAYS)``: the
    goal file declares one date, and the horizon supplies the other. See the
    OMN-18751 comment block above for why the pair is synthesized rather than
    removed.

    ``cause`` distinguishes the four ways the window fails to resolve, which
    the bare ``None`` OMN-18554 replaced could not: the refusal has to be able
    to say which one it hit, or a reader cannot tell a missing clone from an
    un-re-measured goal file.
    """
    try:
        text = goal_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, "file missing"
    except OSError as exc:
        return None, f"file unreadable: {exc.strerror or exc}"
    match = STATE_AS_OF_PATTERN.search(text)
    if not match:
        return None, "no 'state_as_of:' line at the head of the goal file"
    try:
        start = datetime.strptime(match.group(1), "%Y-%m-%d").replace(tzinfo=UTC).date()
    except ValueError:
        return None, f"malformed date in the 'state_as_of:' line: {match.group(0)!r}"
    return (start, start + timedelta(days=FRESHNESS_HORIZON_DAYS)), None


def read_declared_window(goal_path: Path) -> tuple[date, date] | None:
    """Read the freshness window from the goal file's ``state_as_of:`` stamp.

    Returns None when no window resolves. Retained as the narrow accessor;
    callers that must act on WHY it failed use read_declared_window_detail.
    Never hardcode window bounds in this script; the start always comes from
    the declared source and the end from FRESHNESS_HORIZON_DAYS.
    """
    window, _cause = read_declared_window_detail(goal_path)
    return window


class WindowResolution:
    """One of WINDOW_OPEN / WINDOW_STALE / WINDOW_PENDING / WINDOW_UNRESOLVED,
    plus the facts a message needs: the goal path, the window if one parsed,
    and the cause if one did not."""

    __slots__ = ("cause", "goal_path", "state", "window")

    def __init__(
        self,
        state: str,
        *,
        window: tuple[date, date] | None = None,
        cause: str | None = None,
        goal_path: Path | None = None,
    ) -> None:
        self.state = state
        self.window = window
        self.cause = cause
        self.goal_path = goal_path

    @property
    def enforces(self) -> bool:
        """WINDOW_UNRESOLVED deliberately answers False here: it does not
        enforce the cost sentence, it refuses the row outright at a separate,
        earlier branch. Reading it as "enforces" would run the priced-row
        checks against a gate that could not read its own input."""
        return self.state in (WINDOW_OPEN, WINDOW_STALE)

    def describe(self) -> str:
        if self.window is None:
            return f"{self.state} ({self.cause})"
        start, end = self.window
        return f"{self.state} ({start.isoformat()} → {end.isoformat()})"


def resolve_window_state(now: datetime) -> WindowResolution:
    """Classify the goal file's freshness against ``now`` (OMN-18554 (b),
    re-sourced by OMN-18751).

    ``start`` is the declared ``state_as_of`` stamp and ``end`` is that stamp
    plus the horizon, so "today > end" reads as "the goal file has not been
    re-measured inside the horizon" and "today < start" as "the stamp is dated
    in the future."
    """
    try:
        goal_path = goal_path_for_window()
    except GoalSourceUnresolvedError as exc:
        return WindowResolution(WINDOW_NO_REGISTRY, cause=str(exc), goal_path=None)
    window, cause = read_declared_window_detail(goal_path)
    if window is None:
        return WindowResolution(WINDOW_UNRESOLVED, cause=cause, goal_path=goal_path)
    start, end = window
    today = now.date()
    if today < start:
        return WindowResolution(WINDOW_PENDING, window=window, goal_path=goal_path)
    if today > end:
        return WindowResolution(WINDOW_STALE, window=window, goal_path=goal_path)
    return WindowResolution(WINDOW_OPEN, window=window, goal_path=goal_path)


def resolve_now() -> datetime:
    """r1 (f): a timezone-naive LEDGER_LOCK_NOW (no 'Z'/offset — e.g.
    '2026-08-05T12:00:00') used to be accepted by `datetime.fromisoformat`
    and then crash deep in window_is_open/_grace_active with an uncaught
    "can't compare offset-naive and offset-aware datetimes" TypeError,
    exiting rc=1 with a raw traceback instead of the documented rc in
    {0, 65, 75} and silently dropping the payload. A naive override is now
    treated as UTC, the same convention the 'Z' suffix already implies."""
    override = os.environ.get(NOW_OVERRIDE_ENV)
    if override:
        try:
            parsed = datetime.fromisoformat(override.replace("Z", "+00:00"))
        except ValueError:
            parsed = None
        if parsed is not None:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed
    return datetime.now(UTC)


def window_is_open(window: tuple[date, date] | None, now: datetime) -> bool:
    if window is None:
        return False
    start, end = window
    return start <= now.date() <= end


def find_claim_lines(payload: str) -> list[str]:
    return [line for line in payload.splitlines() if is_rule4_claim_row(line)]


def _lane_hour_is_nonzero(match: re.Match[str]) -> bool:
    """r1 (h), INFO fix: 'est 0 lane-hours' is numerically well-formed but a
    zero-cost estimate is the same ritual-string problem AC-5 exists to
    reject (no real work costs zero lane-hours) — require at least one
    strictly-positive number in the matched estimate."""
    numbers = re.findall(r"\d+(?:\.\d+)?", match.group(0))
    return any(float(n) > 0 for n in numbers)


def cost_sentence_reason(line: str) -> str | None:
    """Strict shape+binding check (AC-1/AC-5/AC-7). Only meaningful for a
    line that `has_cost_sentence_attempt` — call sites route accordingly.
    Returns None if the line carries a fully conforming rule-4 cost
    sentence, else a named reason identifying every missing/malformed field
    and the exact expected shape."""
    missing: list[str] = []
    lane_hour_match = LANE_HOUR_PATTERN.search(line)
    if lane_hour_match is None or not _lane_hour_is_nonzero(lane_hour_match):
        missing.append(
            "numeric lane-hour estimate (expected shape: 'est <N> lane-hours', "
            "e.g. 'est ~3-4 lane-hours'; a literal unfilled placeholder like "
            "'<n> lane-hours' does not count, and neither does a zero-cost "
            "estimate like 'est 0 lane-hours')"
        )
    displaces_match = DISPLACES_PATTERN.search(line)
    if displaces_match is None or not displaces_match.group(1).strip():
        missing.append(
            "non-empty displaces clause (expected shape: 'displaces <what it "
            "bumps>', e.g. 'displaces nothing'; an empty trailing 'displaces' "
            "does not count)"
        )
    # r1 (a)/(b), r2 (b), AC-7: the row's own ticket (claimed_ticket_id — a
    # structural extraction, not "first OMN-\d+ anywhere") is implicitly,
    # unambiguously bound as long as no CONFLICTING item identifier is bound
    # to the price via any of PRICE_SUBJECT_PATTERNS — it is never required
    # to be repeated inside the sentence. This is the fix for the CRITICAL
    # kill-criterion: the dominant live shape ("claimed OMN-XXXX ...; est ~N
    # lane-hours; displaces <what>") names the ticket BEFORE the price and
    # never repeats it, so requiring the ticket to appear at/after the
    # lane-hour token (the pre-r1 behavior) hard-refused it. A row with no
    # ticket in any structurally-expected position (claimed_ticket_id is
    # None) has nothing for AC-7 to bind to and is exempt, same as a
    # genuinely ticket-less plan-governor claim. r2 (b): checks EVERY subject
    # binding found in the line, not only the first pattern's match, so a
    # conflict via a colon/dash/"for"/trailing-paren binding is caught the
    # same as the plain whitespace-adjacent form.
    ticket = claimed_ticket_id(line)
    if ticket is not None:
        conflicting = next(
            (subject for subject in _priced_subjects(line) if subject != ticket), None
        )
        if conflicting is not None:
            missing.append(
                f"the ticket ID it prices ('{ticket}') — a different item "
                f"identifier ('{conflicting}') is bound to the price instead "
                "(expected shape: 'est ~N lane-hours; displaces <what>; "
                f"({ticket})' or equivalent — a price keyed to a different "
                "ticket, or a bare PR number, instead of the row's own "
                "ticket is rejected)"
            )
    if not missing:
        return None
    return "missing " + " and ".join(missing)


def _grace_active(now: datetime) -> bool:
    return now < GRACE_DEADLINE_UTC


def evaluate_claim_line(line: str, now: datetime) -> str | None:
    """Return None if `line` is acceptable right now, else a rejection
    reason string. Implements the grace-transition state machine:

    - No cost-sentence attempt at all (legacy-shape) -> None (accepted, but
      caller must still emit the loud warning) while grace is active; a
      named refusal reason once GRACE_DEADLINE_UTC has passed.
    - A cost-sentence attempt that fails strict shape/binding checks
      (new-shape violation) -> always a named refusal reason, regardless of
      grace.
    """
    if not has_cost_sentence_attempt(line):
        if _grace_active(now):
            return None
        return (
            "no rule-4 cost sentence at all, and the OMN-15649 grace period "
            f"({GRACE_DEADLINE_UTC.isoformat().replace('+00:00', 'Z')}) has "
            "passed — add one (e.g. 'est ~2 lane-hours; displaces nothing; "
            "(OMN-XXXX)')"
        )
    return cost_sentence_reason(line)


def resolve_enforcement_window(now: datetime, *, announce_inert: bool) -> bool:
    """r2 (c): the ONE shared window-scoping gate for BOTH ledger-write
    verbs. Previously only ``--append``/``--append-file`` (via
    validate_claim_payload) checked whether the declared window was open at
    all — the ``-- COMMAND`` post-hoc warning path
    (warn_on_unvalidated_command_claims) evaluated every added claim row
    unconditionally, so an out-of-window write via ``-- COMMAND`` produced a
    loud false-alarm warning that the equivalent ``--append`` write would
    have silently, correctly skipped. Both call sites now resolve enforcement
    through this one function.

    AC-4: enforcement is calendar-only, read from the goal file's declared
    ``state_as_of:`` stamp — never hardcoded to a date literal (r2 (a): a
    prior per-claim "declared dated-chain ticket set" narrowing was removed
    here; see the REMEDIATION r2 (a) comment block near the top of this file
    for why). If the freshness cannot be determined (goal file unreadable, or
    no declared ``state_as_of:`` line at all), the row is REFUSED at an
    earlier branch rather than landing unenforced.
    """
    resolution = resolve_window_state(now)
    if announce_inert and resolution.state == WINDOW_STALE:
        assert resolution.window is not None
        start, _end = resolution.window
        age = (now.date() - start).days
        print(
            f"ledger_lock: rule-4 cost-sentence check: STALE GOAL — {resolution.goal_path} "
            f"declares state_as_of {start.isoformat()}, re-measured {age} day(s) ago, past "
            f"the {FRESHNESS_HORIZON_DAYS}-day freshness horizon. The cost sentence is "
            "STILL ENFORCED: the goal file supplies the denominator a price is read "
            "against, not the reason to ask for one. Fix it by RE-MEASURING that file "
            "(the morning ground-state workflow writes it) — there is no plan document to "
            "re-cut; the rolling seven-day plan was retired 2026-09-18 (OMN-18751).",
            file=sys.stderr,
        )
    return resolution.enforces


def unresolved_window_refusal(resolution: WindowResolution) -> str | None:
    """The refusal text for a claim row written while the gate cannot read its
    own declared window (OMN-18554 AC1), or None when the override is set.

    Returning None on the override is deliberate: the escape is real, and the
    caller announces it on stderr so taking it is never silent.
    """
    if resolution.state == WINDOW_NO_REGISTRY:
        print(
            "ledger_lock: rule-4 cost-sentence check cannot run in this environment — "
            f"{resolution.cause} Claim rows LAND here, unenforced and unpriced. This is "
            "not a bypass you should reach for on a lane machine: point "
            f"{KB_INTERNAL_ROOT_ENV} at the clone and the gate runs.",
            file=sys.stderr,
        )
        return None
    if resolution.state != WINDOW_UNRESOLVED:
        return None
    if os.environ.get(ALLOW_INERT_ENV, "").strip():
        print(
            f"ledger_lock: rule-4 cost-sentence check is INERT and {ALLOW_INERT_ENV} is set — "
            f"claim row(s) landing UNENFORCED. Cause: {resolution.cause}. This override exists "
            "for a machine with no knowledge-base-internal clone; it is not a way past a "
            "refusal, and every row it lands is unpriced.",
            file=sys.stderr,
        )
        return None
    where = (
        str(resolution.goal_path)
        if resolution.goal_path is not None
        else "<unresolved goal path>"
    )
    return (
        "rule-4 cost-sentence check cannot be enforced: no freshness stamp resolves from "
        f"{where} ({resolution.cause}). A gate that cannot read its own input has not passed; "
        "it has not run (CLAUDE.md rule 16), so the claim row is REFUSED rather than landed "
        "unpriced. Fix by declaring a 'state_as_of:' line at the head of that goal file, "
        f"pointing {GOAL_PATH_ENV} at a file that has one, or — only on a machine with no "
        f"knowledge-base-internal clone — setting {ALLOW_INERT_ENV}=1 to land unenforced. "
        "Non-claim rows (TERMINAL/NOTE/RULING/PROGRESS) are unaffected and still land."
    )


def validate_claim_payload(
    payload: str, *, cost_unknown: list[str], now: datetime
) -> tuple[str, str | None]:
    """Enforce the rule-4 cost-sentence gate (OMN-15649) on claim rows.

    Returns ``(payload_to_write, rejection_reason)``. ``rejection_reason``
    is None on success. ``payload_to_write`` is byte-identical to the input
    unless ``--cost-unknown`` was supplied to attach a visible, verbatim
    escape note to an offending line — the validator otherwise performs no
    rewriting, normalization, or reflow of any row (claim or non-claim).

    Scope: only lines matching one of CLAIM_ROW_PATTERNS (an act-of-claiming
    row) are inspected; every other row class (TERMINAL, merge-sweep,
    ruling, status, etc.) is passed through untouched. Enforcement
    additionally requires the dated milestone window (read from the rolling
    plan's front matter, via resolve_enforcement_window) to be open for
    "now"; outside an open window, claims land without a cost sentence.
    """
    # OMN-18554: an unresolvable window is its own outcome, checked before the
    # enforce/do-not-enforce split. It refuses claim rows outright (rather than
    # running checks a gate that cannot read its own input has no standing to
    # run), and leaves every other row class alone.
    resolution = resolve_window_state(now)
    # Every announcement below is scoped to a payload that actually carries a
    # claim row. A banner on a TERMINAL/NOTE/PROGRESS append is noise about a
    # gate that was never going to inspect the row, and noise on every write is
    # how a real signal stops being read.
    has_claims = bool(find_claim_lines(payload))
    if resolution.state in (WINDOW_UNRESOLVED, WINDOW_NO_REGISTRY):
        if not has_claims:
            return payload, None
        refusal = unresolved_window_refusal(resolution)
        if refusal is not None:
            return payload, refusal
        return payload, None

    if not resolve_enforcement_window(now, announce_inert=has_claims):
        return payload, None

    lines = payload.splitlines(keepends=True)
    offending_indices: list[int] = []
    reasons_by_index: dict[int, str] = {}
    for index, raw_line in enumerate(lines):
        line = raw_line.rstrip("\n")
        if not is_rule4_claim_row(line):
            continue
        reason = evaluate_claim_line(line, now)
        if reason is None:
            if not has_cost_sentence_attempt(line):
                # Accepted-under-grace legacy row: still loud, never silent.
                print(
                    "ledger_lock: WARNING (OMN-15649 grace transition, expires "
                    f"{GRACE_DEADLINE_UTC.isoformat().replace('+00:00', 'Z')}) — "
                    "claim row has NO rule-4 cost sentence; accepted only "
                    f"because the grace period has not expired: {line.strip()}",
                    file=sys.stderr,
                )
            continue
        if is_pipe_lead_claim_row(line) and pipe_lead_grace_active(now):
            # OMN-18554 item 2: recognized, reported, LANDED. This shape was
            # invisible to the gate until this change, so refusing it the same
            # day would refuse ~92% of live claim appends. The row is named and
            # counted instead, until the cutover date below turns it into the
            # refusal the other shapes already get.
            print(
                "ledger_lock: RULE-4 UNPRICED CLAIM (lane="
                f"{claim_row_lane(line)}) — this row carries no conforming rule-4 cost "
                f"sentence ({reason}). It LANDS today, and is REFUSED from "
                f"{RULE4_PIPE_LEAD_CUTOVER_UTC.isoformat().replace('+00:00', 'Z')}. "
                "Add one now: 'est ~2 lane-hours; displaces nothing; (OMN-XXXX)'.",
                file=sys.stderr,
            )
            continue
        offending_indices.append(index)
        reasons_by_index[index] = reason

    if not offending_indices:
        return payload, None

    # --cost-unknown is positional: the Nth flag reason binds to the Nth
    # offending row in the order it appears in the payload (AC-6 — fixes the
    # round-2 defect where one reason silently applied to every offending
    # row in a multi-row payload).
    cost_unknown = list(cost_unknown or [])
    unescaped: list[int] = []
    for position, index in enumerate(offending_indices):
        if position < len(cost_unknown):
            reason_text = cost_unknown[position]
            raw_line = lines[index]
            line = raw_line.rstrip("\n")
            newline = "\n" if raw_line.endswith("\n") else ""
            note = f" [cost-unknown: {reason_text}]"
            lines[index] = line + note + newline
        else:
            unescaped.append(index)

    if unescaped:
        detail = "; ".join(
            f"line {i + 1} ({reasons_by_index[i]}): {lines[i].rstrip(chr(10)).strip()}"
            for i in unescaped
        )
        reason = (
            "rule-4 cost-sentence check failed for claim row(s) — "
            f"{detail}. Fix by adding a conforming cost sentence naming the "
            "ticket it prices (e.g. 'est ~2 lane-hours; displaces nothing; "
            '(OMN-XXXX)\'), or pass one --cost-unknown "<reason>" per '
            "offending row (in order) to record an explicit, visible escape."
        )
        return payload, reason

    return "".join(lines), None


def diff_added_lines(before: str, after: str) -> list[str]:
    """Line-content diff (not position-aware) used only to find genuinely
    NEW lines an external `-- COMMAND` wrote, for the post-hoc warning check
    (defect 4). A line already present the same number of times before is
    never counted as "added" even if its position moved."""
    before_counts = Counter(before.splitlines())
    seen: Counter[str] = Counter()
    added: list[str] = []
    for line in after.splitlines():
        seen[line] += 1
        if seen[line] > before_counts.get(line, 0):
            added.append(line)
    return added


def enforce_command_claims(ledger: Path, before: str, existed_before: bool) -> bool:
    """Post-command enforcement for the `-- COMMAND` write verb (defect 4,
    HARDENED by r5). Returns True when the write was REFUSED and reverted.

    The command is arbitrary, so it cannot be pre-validated — but "cannot be
    pre-validated" was never a reason to let it LAND. This runs the exact
    same row-class, window and grace predicates `--append` runs, against
    every genuinely new line the command wrote, and on a hard failure
    restores the ledger to its pre-command bytes and makes main() return 65.

    Called while the lock is still held, so the revert cannot race a peer
    writer inside the protocol.

    Grace and window scoping are UNCHANGED and shared with `--append`:
      * outside the declared window -> nothing is enforced, silent.
      * legacy-shape row under grace -> loud warning, lands, command rc kept.
      * anything evaluate_claim_line() rejects -> refuse + revert + rc 65.

    r2 (c) parity note, still load-bearing: gating on
    resolve_enforcement_window() is what stops an out-of-window `-- COMMAND`
    write being treated differently from the identical `--append` write."""
    try:
        after = ledger.read_text(encoding="utf-8")
    except OSError:
        return False
    now = resolve_now()

    # OMN-17427: the clock guard is evaluated on EVERY newly added line and
    # is not window/grace-scoped, so it is collected before the window gate
    # below can short-circuit the cost-sentence checks. `-- COMMAND` must
    # enforce exactly what `--append` enforces; there is no editor bypass.
    added_lines = diff_added_lines(before, after)
    refusals: list[str] = []
    override_reason = clock_override_reason(now)
    if override_reason is not None:
        refusals.append(f"(OMN-17427 clock guard) {override_reason}")
    for line in added_lines:
        clock_reason = clock_skew_reason(line, now)
        if clock_reason is not None:
            refusals.append(
                f"(OMN-17427 clock guard: {clock_reason}): {line.strip()[:120]}"
            )

    # OMN-18554: identical window handling to --append, for the same reason r5
    # gave — a gate one verb honours and the other narrates is not a gate. An
    # unresolvable window refuses any claim row this command added and reverts
    # the write; the override, when set, announces itself and lands.
    resolution = resolve_window_state(now)
    if resolution.state in (WINDOW_UNRESOLVED, WINDOW_NO_REGISTRY):
        unresolved_refusal = unresolved_window_refusal(resolution)
        if unresolved_refusal is not None:
            for line in added_lines:
                if is_rule4_claim_row(line):
                    refusals.append(f"({unresolved_refusal}): {line.strip()}")
        added_lines = []

    if not refusals and not resolve_enforcement_window(now, announce_inert=True):
        return False
    if not resolve_enforcement_window(now, announce_inert=False):
        added_lines = []

    for line in added_lines:
        if not is_rule4_claim_row(line):
            continue
        reason = evaluate_claim_line(line, now)
        if reason is None:
            if not has_cost_sentence_attempt(line):
                print(
                    "ledger_lock: WARNING (OMN-15649 grace transition, expires "
                    f"{GRACE_DEADLINE_UTC.isoformat().replace('+00:00', 'Z')}) — "
                    "claim row written via -- COMMAND has NO rule-4 cost "
                    "sentence; accepted only because the grace period has not "
                    f"expired: {line.strip()}",
                    file=sys.stderr,
                )
            continue
        if is_pipe_lead_claim_row(line) and pipe_lead_grace_active(now):
            # Same dated grace as --append, for the same reason r5 gave: the two
            # write verbs cannot differ, or the one that refuses is the one lanes
            # stop using.
            print(
                "ledger_lock: RULE-4 UNPRICED CLAIM (lane="
                f"{claim_row_lane(line)}) written via -- COMMAND — no conforming rule-4 "
                f"cost sentence ({reason}). It LANDS today, and is REFUSED from "
                f"{RULE4_PIPE_LEAD_CUTOVER_UTC.isoformat().replace('+00:00', 'Z')}.",
                file=sys.stderr,
            )
            continue
        refusals.append(f"({reason}): {line.strip()}")

    if not refusals:
        return False

    # Preserve the author's work before reverting. Reverting an arbitrary
    # editor session was the stated reason the original build chose
    # warn-only; the sidecar removes that objection — a refusal costs a
    # re-run, never the edits.
    sidecar = ledger.with_name(
        f"{ledger.name}.rejected-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    )
    suffix = 0
    while sidecar.exists():
        suffix += 1
        sidecar = sidecar.with_name(f"{sidecar.name.rsplit('.', 1)[0]}.{suffix}")
    try:
        sidecar.write_text(after, encoding="utf-8")
        sidecar_note = str(sidecar)
    except OSError as exc:  # pragma: no cover - filesystem failure path
        sidecar_note = f"<could not be written: {exc}>"

    if existed_before:
        write_text_atomic(ledger, before)
    else:
        with suppress(OSError):
            ledger.unlink()

    detail = "\n".join(f"ledger_lock:   {item}" for item in refusals)
    print(
        "ledger_lock: *** CLAIM ROW REFUSED — -- COMMAND WRITE REVERTED ***\n"
        f"{detail}\n"
        "ledger_lock: the ledger has been restored to its pre-command bytes. "
        f"Your full rejected version is preserved verbatim at: {sidecar_note}\n"
        "ledger_lock: fix the refusal above and re-run — a cost-sentence "
        "refusal wants 'est ~2 lane-hours; displaces nothing; (OMN-XXXX)'; an "
        "OMN-17427 clock-guard refusal wants the row re-stamped from the real "
        "clock (`date -u`). The -- COMMAND verb now enforces exactly what "
        "--append enforces; there is no editor bypass.",
        file=sys.stderr,
    )
    return True


# --- OMN-15897: pre-append lint for uncited quantitative claims ------------
#
# Six "integrity register" findings on 2026-08-11 (the 2026-08-11
# friction-register.md § self-reported-tallies-unreliable) were caused by a
# builder lane writing a bare rerun/PR/deploy/merge COUNT into this ledger
# with no citation (a run id, PR/issue reference, commit SHA, or gh/git
# command) proving the figure, later disproven -- sometimes hours later, in
# a dedicated adversarial-verify correction lane -- by an independent
# re-derivation from live sources. Two directly caused a lane STOP:
#   * crossdb-disposition, 2026-08-11T03:59Z (live specimen, still in the
#     ledger): "...degraded into a CI rerun loop — 12 rerun invocations
#     total, 5 within 4.5min at 03:49-03:53Z re-firing runs
#     31455796265/31455687027...". The run ids ARE present in the row, but
#     ~65 characters after the "12 rerun" figure they attach to -- they cite
#     the specific PAIR re-fired at 03:49-03:53Z, not the aggregate "12".
#   * gate-green-chain-0811, 2026-08-11T17:03Z (live specimen, still in the
#     ledger): "...writer-fleet true-cost RE-CORRECTED: 9 rerun invocations
#     across 8 runs (was 7/6)..." -- itself a correction of an EARLIER wrong
#     tally (the 13:54Z [records-fix] cycle it supersedes restated "9
#     invocations across 8 runs, not 7 across 6"), with zero citation
#     tokens anywhere near it, even though the same (long, multi-clause) row
#     cites real PR/OCC/run ids for unrelated facts elsewhere.
#
# Both specimens show why a whole-row "does this line contain ANY citation
# token at all" check would be too weak to catch what actually happened:
# real rows are long and multi-clause, and routinely cite something
# unrelated to the specific count being asserted. CITATION_PROXIMITY_CHARS
# bounds the search to a window around each quantitative-claim match, tuned
# against these same two specimens: the genuinely-cited sibling claims in
# the SAME crossdb row ("5 merges (OCC#6345, ...)", "3 PRs opened
# (omnimarket OMN-15846, ...)") cite within ~10-20 characters, comfortably
# inside the window; both offending claims above have their nearest token
# 60+ characters away, comfortably outside it. See
# tests/test_ledger_lock_quantitative_claims.py for both specimens replayed
# verbatim through the real CLI as RED-before/GREEN-after fixtures.
#
# Scope is deliberately ROW-CLASS-AGNOSTIC (unlike the OMN-15649 cost-
# sentence gate above, which only inspects CLAIM_ROW_PATTERNS rows): a bare
# tally can land in a TERMINAL/STOP/ADJUDICATED row just as easily as a
# claim row, and 5 of the 6 named findings were exactly that. It is also
# NOT window/grace-scoped -- OMN-15649's calendar-window and grace-
# transition machinery exists because that gate changes what shape a claim
# row must already have; this gate only asks for a citation next to a
# number, which was never a burden any live template needed a transition
# period to adopt.
#
# Deliberately out of scope: the `-- COMMAND` write verb. The ticket
# (OMN-15897) scopes this to "the --append path (the sole serialized
# ledger-write hook, CLAUDE.md rule 1a)"; --append/--append-file is the
# path every documented caller (build-lane.js, ticket-dedupe-then-claim.js,
# plan-governor-update.js, and the CLAUDE.md rule 1a recipe itself) uses.
QUANT_CLAIM_PATTERN = re.compile(
    r"(?<![#\w])(?<!OMN-)\b(\d+)\s+(reruns?|retriggers?|PRs?|deploys?|merges?)\b",
    re.IGNORECASE,
)

# A citation token: a gh/git command invocation, a URL, an OMN-XXXX ticket
# id, a repo-shorthand PR/issue reference (infra#855, OCC#6344, ... --
# mirrors PRICE_SUBJECT_PATTERNS' precedent above for this shape), an
# explicit "run <id>" mention, a commit SHA, or any bare 4+-digit number
# (PR/issue numbers are routinely written space-separated rather than
# "#"-prefixed in this ledger -- live specimen: "9 PRs live+CI-checked
# (infra 2601/2602/2603/2605 green, ...)", tests/fixtures/
# ledger_lock_peer_safety_sample.txt L59 -- so a bare-digit citation still
# counts even without a "#"). A citation token is EXCLUDED from the digit
# it is attached to counting as its own quantitative claim by
# QUANT_CLAIM_PATTERN's own "not preceded by '#'/word-char/'OMN-'" guard
# (excludes "infra#858 deploys", a PR NUMBER, not a deploy COUNT; and
# "OMN-15717 merges to dev", a TICKET as the grammatical subject of the verb
# "merges", not a merge COUNT -- a live false positive found replaying this
# gate against the real production ledger, tests/test_ledger_lock_cost_
# sentence.py::test_full_live_ledger_replay_against_the_real_production_plan
# L13's "...until OMN-15717 merges to dev and the refresh runs...").
#
# The bare-digit branch's ``(?!-\d{2}-)`` guard excludes a YYYY-MM-DD year
# component (e.g. the "2026" in the row's OWN leading "2026-08-12T00:00:00Z"
# timestamp, or any other embedded date in free text) from counting as a
# citation -- every row in this ledger starts with one, so without this
# guard a short row's leading timestamp would fall inside
# CITATION_PROXIMITY_CHARS of nearly any early claim and silently
# manufacture a false citation for it.
CITATION_TOKEN_PATTERN = re.compile(
    r"\bgh\s+\S|\bgit\s+\S|https?://\S+|\bOMN-\d+\b|\b[A-Za-z][\w.]*#\d+"
    r"|\brun\s+\d+\b|\b[0-9a-f]{7,40}\b|\b\d{4,}\b(?!-\d{2}-)",
    re.IGNORECASE,
)

CITATION_PROXIMITY_CHARS = 40


def bare_quantitative_claims(line: str) -> list[str]:
    """Every QUANT_CLAIM_PATTERN match in `line` with no CITATION_TOKEN_PATTERN
    match within CITATION_PROXIMITY_CHARS characters on either side (the
    match's own span is excluded from the window so the claim text itself
    can never satisfy its own citation requirement). Returns the matched
    substrings ("12 rerun", "9 deploys", ...) in order of appearance; empty
    if `line` has no quantitative claim at all, or every one found is
    adjacently cited."""
    offending: list[str] = []
    for match in QUANT_CLAIM_PATTERN.finditer(line):
        window_start = max(0, match.start() - CITATION_PROXIMITY_CHARS)
        window_end = min(len(line), match.end() + CITATION_PROXIMITY_CHARS)
        window = line[window_start : match.start()] + line[match.end() : window_end]
        if not CITATION_TOKEN_PATTERN.search(window):
            offending.append(match.group(0))
    return offending


def quantitative_claim_reason(line: str) -> str | None:
    """None if `line` carries no bare quantitative claim, else a named
    rejection reason quoting every offending match."""
    offending = bare_quantitative_claims(line)
    if not offending:
        return None
    quoted = ", ".join(f"'{item}'" for item in offending)
    return (
        f"bare quantitative claim(s) with no adjacent citation: {quoted} -- "
        "add a run id, PR/issue reference (e.g. '#1234' or 'infra#1234' or "
        "'infra 1234'), commit SHA, OMN-XXXX ticket, or gh/git command "
        f"within ~{CITATION_PROXIMITY_CHARS} characters proving the figure, "
        "or rewrite the sentence so the citation you already have sits next "
        "to the number it backs (OMN-15897)"
    )


def validate_quantitative_claims_payload(payload: str) -> str | None:
    """Scan every line of `payload` -- not only claim rows, see the scope
    note above -- for a bare quantitative claim (OMN-15897). Returns None if
    the whole payload is clean, else a combined rejection reason naming
    every offending line, 1-indexed within the payload."""
    reasons: list[str] = []
    for index, raw_line in enumerate(payload.splitlines()):
        reason = quantitative_claim_reason(raw_line)
        if reason is not None:
            reasons.append(f"line {index + 1} ({reason}): {raw_line.strip()}")
    if not reasons:
        return None
    return "; ".join(reasons)


# --- OMN-15901: pre-append lint for deferred findings with no follow-up ----
#
# Sibling to OMN-15897 (same --append hook, ticket's own Links section says
# "share the implementation"): a finding *recorded* as deferred ("no ticket
# filed", "still has no ticket", "unticketed", ...) has no mechanism that
# produces the follow-up ticket -- the deferral is durable, the follow-up
# depends on someone remembering. Proven failure: OMN-15567 named the same
# nightly 24f/49e population twice in comments 2026-08-02 as "deferred for
# separate tickets" and it sat NEVER TICKETED for 8 days until OMN-15856
# filed it. The same class recurred three more times on 2026-08-11, each
# written straight into this ledger with no ticket at the time:
#   * L14389 (writer-fleet-build, 13:10Z): "...3 OCC companions (#6354/
#     #6356/#6357, all hand-authored after occ-autobind stalled 3/3 times on
#     this repo/ticket -- flagged, no ticket filed to keep scope tight)..."
#     Only filed ~9h later, as OMN-15887, by the friction-register pass.
#   * L14430 (opus-final-verify-omn15876, 21:40Z), residual (b):
#     "...occ-evidence-source-autobind mutating an already-merged receipt
#     fired TWICE (OCC#6364, #6365) and still has no ticket."
#   * L14431 (projapi-takeover-0811, 21:36Z): "...autobind receipt-mutation
#     fired TWICE today unticketed; test docstring claims counting-RED
#     falsely...; restarted-for annotation temporary-by-comment." -- three
#     further residuals recorded with no ticket at write time.
#
# Design: reuse OMN-15897's proximity mechanism wholesale rather than a
# whole-row "does this row contain ANY OMN-\d+ anywhere" check. The ticket's
# own AC phrase ("unless the same row carries an OMN-\d+ reference") reads
# whole-row on its face, but the L14389 specimen above is the OMN-15897-
# style trap that phrasing would walk into: that row's OWN later clause
# ("1 new Linear ticket (OMN-15868)") cites a *real* ticket -- for the
# root-cause fix, a completely different finding than the "no ticket filed"
# OCC-companion clause 204 characters earlier. A whole-row check would read
# OMN-15868 as satisfying the OCC-companion deferral and wrongly accept the
# row; OMN-15868 does not own that deferral (OMN-15887 does, filed 9h
# later). This is the direct analogue of OMN-15897's "ticket-as-subject"
# false-accept class (an OMN- token present but not actually backing the
# claim next to it) -- so the fix is the same one OMN-15897 already proved:
# require the citation to sit within CITATION_PROXIMITY_CHARS of the
# specific defer-language match, not merely anywhere in the row. Measured
# against all four real specimens this file cites (see
# tests/test_ledger_lock_defer_language.py): the three offending rows'
# nearest OMN-\d+ token sits 204-3094 characters away (rejected, correctly);
# the legitimate OMN-15856/OMN-15567 historical row (L14349, already
# resolved, both tickets present) has its nearest OMN- token 26-78
# characters from its own "unticketed"/"deferred" language (accepted,
# correctly). CITATION_PROXIMITY_CHARS (40, defined above for OMN-15897) is
# reused verbatim -- not re-tuned -- and holds for all four specimens.
#
# OMN-15897's other named false-positive class (a leading YYYY-MM-DD
# timestamp misread as a citation) does not reproduce here: this gate only
# accepts an OMN-\d+ token as a citation (never a bare digit run), and no
# ledger row's leading timestamp is shaped "OMN-\d+".
#
# The ticket's "explicit [defer-ack: OMN-XXXX] token" escape is not a
# separate pattern: any "[defer-ack: OMN-XXXX]" annotation already contains
# a bare OMN-\d+ occurrence, so OMN_REFERENCE_PATTERN alone recognizes both
# accepted forms the AC names.
#
# Scope, like OMN-15897: row-class-agnostic (all four real specimens above
# are TERMINAL/VERDICT/ADJUDICATED rows, not claim rows) and NOT window/
# grace-scoped -- this gate only asks for a citation next to a phrase, which
# was never a burden any live template needed a transition period to adopt.
DEFER_LANGUAGE_PATTERN = re.compile(
    r"\bno ticket filed\b|\bneeds its own ticket\b|\bstill has no ticket\b"
    r"|\bdeferred (?:for|to) (?:a )?separate tickets?\b|\bunticketed\b",
    re.IGNORECASE,
)

OMN_REFERENCE_PATTERN = re.compile(r"\bOMN-\d+\b", re.IGNORECASE)


def bare_defer_language_claims(line: str) -> list[str]:
    """Every DEFER_LANGUAGE_PATTERN match in `line` with no OMN_REFERENCE_
    PATTERN match within CITATION_PROXIMITY_CHARS characters on either side
    (the match's own span is excluded from the window, same convention as
    bare_quantitative_claims). Returns the matched substrings in order of
    appearance; empty if `line` has no defer-language at all, or every one
    found is adjacently backed by an OMN-<number> reference."""
    offending: list[str] = []
    for match in DEFER_LANGUAGE_PATTERN.finditer(line):
        window_start = max(0, match.start() - CITATION_PROXIMITY_CHARS)
        window_end = min(len(line), match.end() + CITATION_PROXIMITY_CHARS)
        window = line[window_start : match.start()] + line[match.end() : window_end]
        if not OMN_REFERENCE_PATTERN.search(window):
            offending.append(match.group(0))
    return offending


def defer_language_reason(line: str) -> str | None:
    """None if `line` carries no bare defer-language phrase, else a named
    rejection reason quoting every offending match."""
    offending = bare_defer_language_claims(line)
    if not offending:
        return None
    quoted = ", ".join(f"'{item}'" for item in offending)
    return (
        f"deferred finding with no adjacent follow-up ticket: {quoted} -- "
        "file the follow-up ticket and cite its OMN-XXXX id within "
        f"~{CITATION_PROXIMITY_CHARS} characters, or add an explicit "
        "'[defer-ack: OMN-XXXX]' token naming the ticket that already owns "
        "this deferral (OMN-15901)"
    )


def validate_defer_language_payload(payload: str) -> str | None:
    """Scan every line of `payload` -- row-class-agnostic, see the scope
    note above -- for a deferred finding with no adjacent follow-up ticket
    (OMN-15901). Returns None if the whole payload is clean, else a combined
    rejection reason naming every offending line, 1-indexed within the
    payload."""
    reasons: list[str] = []
    for index, raw_line in enumerate(payload.splitlines()):
        reason = defer_language_reason(raw_line)
        if reason is not None:
            reasons.append(f"line {index + 1} ({reason}): {raw_line.strip()}")
    if not reasons:
        return None
    return "; ".join(reasons)


# --- OMN-17427: fail-closed wall-clock guard on a row's OWN timestamp -----
#
# Root cause this closes (fact, 2026-09-01/02): an orchestrating session
# passed guessed wall-clock values as a "now" argument into workflow scripts
# from ~2026-09-01T20:20Z onward. Those values ran 30 minutes to ~3 hours
# ahead of the real clock, and lanes copied them verbatim into the leading
# timestamp of ledger rows (and into doc headers). Nothing in this script
# ever looked at the row's own leading timestamp, so every one of those rows
# landed unchallenged.
#
# Why that is not cosmetic: this ledger is read as a state history. A row
# stamped ahead of the clock makes a TERMINAL appear to close a CLAIM that
# has not happened yet, makes two lanes' rows interleave in the wrong order,
# and makes "is this lane still running?" unanswerable from the file.
#
# Scope: deliberately row-class-agnostic and NOT window/grace-scoped, unlike
# the OMN-15649 cost-sentence gate. A mis-stamped TERMINAL row corrupts the
# history exactly as badly as a mis-stamped CLAIM row, and there is no
# transition period during which writing a false time is acceptable. Rows
# with no parseable leading timestamp are untouched (headings, table rows,
# continuation lines) -- this gate never invents a rule for them.
#
# Tolerances:
#   * 5 minutes AHEAD -- covers host clock jitter plus the seconds-to-minutes
#     between an agent composing a row and the append actually landing.
#   * 24 hours BEHIND -- a legitimately delayed or backfilled write of work
#     done earlier the same day still lands; a stale-clock host or a stamp
#     copy-pasted from yesterday's row does not.
CLOCK_GUARD_AHEAD_TOLERANCE_SECONDS = 5 * 60
CLOCK_GUARD_BEHIND_TOLERANCE_SECONDS = 24 * 60 * 60

# Leading timestamp of a ledger row: optional list bullet, optional bracket,
# then an ISO-8601 instant. Minutes-only ("...T21:45Z") and seconds ("...
# T21:45:00Z") shapes are both live in this ledger, as is a "- " bullet and
# a bare start-of-line. A naked offset ("+00:00"/"-0700") is accepted for
# completeness; the ledger convention is "Z".
LEADING_TIMESTAMP_PATTERN = re.compile(
    r"^[ \t]*(?:[-*+][ \t]+)?\[?"
    r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?)"
    r"(Z|z|[+-]\d{2}:?\d{2})"
)


def _format_instant(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _format_skew(seconds: float) -> str:
    seconds = abs(seconds)
    if seconds < 90:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f}m"
    return f"{minutes / 60:.1f}h"


def row_leading_timestamp(line: str) -> datetime | None:
    """The instant a ledger row stamps itself with, or None when the line
    does not open with one. Always returned tz-aware (UTC for the 'Z' form)
    so callers can compare it to resolve_now() without a naive/aware crash
    -- the same trap resolve_now()'s own docstring records."""
    match = LEADING_TIMESTAMP_PATTERN.match(line)
    if match is None:
        return None
    offset = match.group(2)
    normalized = "+00:00" if offset in ("Z", "z") else offset
    if len(normalized) == 5 and ":" not in normalized:  # "+0000" -> "+00:00"
        normalized = normalized[:3] + ":" + normalized[3:]
    try:
        parsed = datetime.fromisoformat(match.group(1) + normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:  # pragma: no cover - normalized above
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def clock_skew_reason(line: str, now: datetime) -> str | None:
    """None when the row's leading timestamp is inside the tolerance window
    around `now` (or absent), else a rejection reason naming BOTH times."""
    stamped = row_leading_timestamp(line)
    if stamped is None:
        return None
    skew_seconds = (stamped - now).total_seconds()
    if skew_seconds > CLOCK_GUARD_AHEAD_TOLERANCE_SECONDS:
        return (
            f"row is stamped {_format_instant(stamped)} but the wall clock is "
            f"{_format_instant(now)} -- {_format_skew(skew_seconds)} AHEAD "
            f"(tolerance {CLOCK_GUARD_AHEAD_TOLERANCE_SECONDS // 60}m)"
        )
    if -skew_seconds > CLOCK_GUARD_BEHIND_TOLERANCE_SECONDS:
        return (
            f"row is stamped {_format_instant(stamped)} but the wall clock is "
            f"{_format_instant(now)} -- {_format_skew(skew_seconds)} BEHIND "
            f"(tolerance {CLOCK_GUARD_BEHIND_TOLERANCE_SECONDS // 3600}h)"
        )
    return None


def clock_override_reason(now: datetime) -> str | None:
    """LEDGER_LOCK_NOW is the one documented way to move this script's idea
    of "now". It is also the exact shape of the OMN-17427 root cause -- a
    guessed wall-clock value handed to the writer -- so an override that is
    itself outside the same tolerances is refused rather than trusted. Without
    this, the guard below would validate a mis-stamped row against an equally
    mis-stamped clock and pass it."""
    if not os.environ.get(NOW_OVERRIDE_ENV):
        return None
    real = datetime.now(UTC)
    skew_seconds = (now - real).total_seconds()
    if (
        skew_seconds > CLOCK_GUARD_AHEAD_TOLERANCE_SECONDS
        or -skew_seconds > CLOCK_GUARD_BEHIND_TOLERANCE_SECONDS
    ):
        return (
            f"{NOW_OVERRIDE_ENV} is set to {_format_instant(now)} but the real "
            f"wall clock is {_format_instant(real)} -- {_format_skew(skew_seconds)} "
            f"{'AHEAD' if skew_seconds > 0 else 'BEHIND'}; the override may not be "
            "used to move the clock past the guard's own tolerances"
        )
    return None


def validate_clock_skew_payload(payload: str, now: datetime) -> str | None:
    """Scan every line of `payload` for a leading timestamp that is ahead of
    or far behind the wall clock. Returns None when the whole payload is
    clean, else a combined rejection reason naming every offending line,
    1-indexed within the payload."""
    override_reason = clock_override_reason(now)
    if override_reason is not None:
        return override_reason
    reasons: list[str] = []
    for index, raw_line in enumerate(payload.splitlines()):
        reason = clock_skew_reason(raw_line, now)
        if reason is not None:
            reasons.append(f"line {index + 1} ({reason}): {raw_line.strip()[:120]}")
    if not reasons:
        return None
    return (
        "; ".join(reasons)
        + " -- read the real clock (`date -u`) and stamp the row with it; never "
        "copy a 'now' value handed in by a caller or carried over from an "
        "earlier row (OMN-17427)"
    )


# --- OMN-18258: refuse a second ruling on an unacknowledged subject -------
#
# The deciding logic is NOT here. It lives in the committed module
# docs/workflows/_shared/ruling_guard.py, where tests/test_ledger_lock_ruling_guard.py
# runs its real bytes in CI. This file is gitignored local tooling, so logic
# that lived here would be tested nowhere; this is the thin caller, and that
# test asserts the caller still calls.
#
# Unlike every other pre-append lint in this file, this one is STATE-DEPENDENT:
# it reads what is already in the ledger. So it runs INSIDE the held lock,
# immediately before the append, where a concurrent writer cannot slip a
# competing ruling in between the check and the write.

_RULING_GUARD_PATH = _omni_home_shared() / "ruling_guard.py"


def load_ruling_guard() -> Any | None:
    """Import the committed guard. Returns None when it is absent -- a checkout
    without docs/ is not a reason to refuse every append, and the absence is
    announced on stderr rather than swallowed."""
    if not _RULING_GUARD_PATH.is_file():
        print(
            f"ledger_lock: OMN-18258 ruling guard not found at {_RULING_GUARD_PATH}; "
            "ruling sequencing is NOT being enforced for this append",
            file=sys.stderr,
        )
        return None
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "onex_ruling_guard", _RULING_GUARD_PATH
    )
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ledger_display_name(ledger: Path) -> str:
    """The name the guard cites in its refusal. Repo-relative where possible so
    a lane copying the citation into a row does not paste a machine-local
    absolute path (CLAUDE.md rule 6)."""
    try:
        return str(ledger.resolve().relative_to(_omni_home_root()))
    except ValueError:
        return ledger.name


def validate_ruling_payload(payload: str, ledger: Path) -> str | None:
    guard = load_ruling_guard()
    if guard is None:
        return None
    existing = ledger.read_text(encoding="utf-8") if ledger.exists() else ""
    # Annotated rather than returned straight through: the guard is imported
    # from a path at runtime, so mypy sees Any and a bare return silently
    # widens this function's declared type at every call site.
    refusal: str | None = guard.refusal_for_payload(
        payload, existing, ledger_display_name(ledger)
    )
    return refusal


# --- OMN-18274: friction recording is mechanical, not remembered ----------
#
# Same arrangement as the OMN-18258 ruling guard immediately above, for the
# same reason: this file is gitignored local tooling, so the DECIDING LOGIC
# lives in the committed module docs/workflows/_shared/friction_guard.py
# where tests/test_ledger_lock_friction_guard.py runs its real bytes in CI.
# This is the thin caller, and that test asserts the caller still calls.
#
# Why it exists: the 2026-09-13 friction trend report measured 231 of 237
# FRICTION rows landing in one backfill -- the row type was not being used,
# and friction was being reported in TERMINAL row prose instead. See that
# module's header for the full grounding.
#
# Like the ruling guard, obligation 3 (a cited friction row must exist) is
# STATE-DEPENDENT, so this runs INSIDE the held lock where a concurrent
# writer cannot slip the row in between the check and the write.
_FRICTION_GUARD_PATH = _omni_home_shared() / "friction_guard.py"


def load_friction_guard() -> Any | None:
    """Import the committed guard. Returns None when it is absent -- a checkout
    without docs/ is not a reason to refuse every append, and the absence is
    announced on stderr rather than swallowed."""
    if not _FRICTION_GUARD_PATH.is_file():
        print(
            f"ledger_lock: OMN-18274 friction guard not found at {_FRICTION_GUARD_PATH}; "
            "friction recording is NOT being enforced for this append",
            file=sys.stderr,
        )
        return None
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "onex_friction_guard", _FRICTION_GUARD_PATH
    )
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_friction_payload(payload: str, ledger: Path) -> str | None:
    guard = load_friction_guard()
    if guard is None:
        return None
    existing = ledger.read_text(encoding="utf-8") if ledger.exists() else ""
    # Annotated rather than returned straight through: the guard is imported
    # from a path at runtime, so mypy sees Any and a bare return silently
    # widens this function's declared type at every call site.
    refusal: str | None = guard.refusal_for_payload(
        payload, existing, ledger_display_name(ledger)
    )
    return refusal


# --- OMN-18433: the stranded-clone signal ---------------------------------
#
# On 2026-09-16 this clone sat on a branch whose pull request had already
# merged. Lanes kept appending here for 1h53m while commit_lock.py refused
# that branch with exit 5 on every attempt to persist the rows. Fifty
# row-blocks reached no committed copy and nothing said so.
#
# The deciding logic is committed at docs/workflows/_shared/stranded_clone_guard.py
# so it is testable in CI. The FALLBACK below is not belt-and-braces: this
# script is gitignored and the module is tracked, so a clone checked out at a
# revision without the module -- which is precisely the condition being
# detected -- would otherwise disarm the guard that exists to detect it.
_STRANDED_GUARD_PATH = _omni_home_shared() / "stranded_clone_guard.py"
_FALLBACK_EXIT_STRANDED_CLONE = 78
_FALLBACK_EXPECTED_BRANCH = "main"
_FALLBACK_REMEDY = (
    "converge the clone before the rows are needed: "
    "git -C $OMNI_HOME fetch origin && git -C $OMNI_HOME status --short, "
    "move any uncommitted work to a worktree off origin/main, "
    "then put the clone back on main"
)


def load_stranded_clone_guard() -> Any | None:
    """Import the committed stranded-clone guard, or None when it is absent.

    Absence is announced, never swallowed, and never silences the signal --
    the caller falls back to its inline constants.
    """
    if not _STRANDED_GUARD_PATH.is_file():
        print(
            f"ledger_lock: OMN-18433 stranded-clone guard not found at {_STRANDED_GUARD_PATH}; "
            "falling back to the inline constants (the signal still fires)",
            file=sys.stderr,
        )
        return None
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "onex_stranded_clone_guard", _STRANDED_GUARD_PATH
    )
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def clone_state(path: Path) -> tuple[str | None, Path | None]:
    """The branch and the repository root of the git tree containing `path`.

    `(None, None)` means "could not be determined" -- not a git tree, git
    absent, or the probe failed -- and is deliberately NOT treated as
    stranded: an unknown must not manufacture a signal an operator acts on.

    The ROOT matters as much as the branch. The throttle state belongs to the
    clone being reported, not to whichever clone happens to hold the tool, or
    one stranded clone's note would suppress another's.
    """
    directory = path.parent if path.parent.exists() else _omni_home_root()
    try:
        proc = subprocess.run(
            [
                "git",
                "-C",
                str(directory),
                "rev-parse",
                "--abbrev-ref",
                "HEAD",
                "--show-toplevel",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    if proc.returncode != 0:
        return None, None
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None, None
    return (lines[0] or None), Path(lines[1])


def signal_stranded_clone(ledger: Path, payload: str) -> int:
    """Called with the lock still held, immediately AFTER the row landed.

    Returns the stranded-clone exit code when this clone will not commit the
    row, 0 otherwise. The row is never withheld: a guard that refused the
    write would turn rows-on-disk-uncommitted into rows-never-written.

    Any UNEXPECTED failure here returns 0 with a warning. The append has
    already succeeded at this point, and taking down every lane's ledger write
    because a signal could not be computed would be a worse defect than the
    one this closes. The one anticipated failure -- the committed module being
    absent -- is handled by the fallback, not by this catch.
    """
    try:
        # Cheap, quiet pre-check FIRST. The overwhelmingly common case is a
        # clone on its integration branch, and that case must cost one git
        # call and print nothing at all -- a guard that chatters on every
        # append of every lane is a guard that gets filtered out.
        branch, root = clone_state(ledger)
        if (
            branch is None
            or root is None
            or branch in ("", "HEAD", _FALLBACK_EXPECTED_BRANCH)
        ):
            return 0

        guard = load_stranded_clone_guard()
        expected = getattr(guard, "EXPECTED_BRANCH", _FALLBACK_EXPECTED_BRANCH)
        if guard is not None and not guard.is_stranded(branch):
            return 0

        code = int(getattr(guard, "EXIT_STRANDED_CLONE", _FALLBACK_EXIT_STRANDED_CLONE))
        name = ledger_display_name(ledger)
        if guard is None:
            print(
                f"ledger_lock: STRANDED CLONE -- the row WAS appended to {name}, but this "
                f"clone is on '{branch}', not {expected}. commit_lock.py refuses this branch, "
                f"so the row reaches no committed copy (OMN-18433, exit {code}). "
                f"REMEDY: {_FALLBACK_REMEDY}",
                file=sys.stderr,
            )
            return code

        print(guard.stderr_warning(branch, name), file=sys.stderr)

        now = resolve_now()
        state_path = root / guard.STATE_RELPATH
        raw = state_path.read_text(encoding="utf-8") if state_path.is_file() else None
        state = guard.parse_state(raw)
        if not guard.note_is_due(state, branch, now):
            return code

        lane = guard.row_lane(
            next((ln for ln in payload.splitlines() if ln.strip()), "")
        )
        append_text(ledger, guard.note_row(now, branch, name, lane))
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps(guard.advance_state(branch, now), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        return code
    except Exception as exc:  # noqa: BLE001 -- see the docstring
        print(
            f"ledger_lock: OMN-18433 stranded-clone signal could not be computed ({exc!r}); "
            "the row WAS appended and this append is not being failed for it",
            file=sys.stderr,
        )
        return 0


# --- OMN-18433 / OMN-18757: the verbatim replay path ---------------------
#
# WHAT IT IS FOR. A row that already existed on some copy of this ledger is
# being restored, byte for byte, after being recovered from a tree that was
# never committed. On 2026-09-16, replaying the 50 rows stranded on
# `jonah/omn-16642-ledger-rows` landed 45 and left 5 refused by the OMN-18274
# mandatory-friction guard. A historical row must not be rewritten to pass a
# present-day guard -- a row edited to satisfy a later reader is no longer
# evidence of anything -- so without this path the only choices were falsify
# the row or lose it.
#
# WHAT IT IS NOT FOR. It is not a way to write a row a guard would refuse
# today. Every refusal below is fail-closed, and the positive control for the
# whole feature is that the same payload WITHOUT the flag is still refused.
#
# WHY THIS LIVES HERE AND THE DECIDING LOGIC DOES NOT. `replay_refusal` and
# `replay_marker_row` are in the committed guard module, which is tracked and
# therefore tested in CI; this file only wires them to argv. That split is
# deliberate and it is the reason there is NO inline fallback for the waiver,
# unlike `signal_stranded_clone` above. The two cases are opposites: there,
# the danger is SILENCE, so a caller that cannot reach the module must still
# fire; here, the danger is a WAIVER, so a caller that cannot reach the module
# must refuse. A waiver granted by a caller that cannot read the rule it is
# waiving is not a waiver, it is a bypass -- and an inline copy of the rule
# would be a second implementation of it, free to drift towards permissive.
#
# WHY IT WAS REBUILT. OMN-18433 shipped these two flags on 2026-09-16 and used
# them: five `| REPLAY |` marker rows stamped 2026-09-16T11:26:21Z are in the
# live ledger, landed by omni_home#326. They existed only in the gitignored
# omni_home copy of this script, so nothing carried them into a commit, and
# the OMN-18554 port of that copy into this committed one (#3688, 80dc408ba)
# carried the six guards and not the flags. Three tests in
# `tests/test_ledger_stranded_clone.py` have been red ever since. The tests in
# `tests/unit/scripts/test_ledger_lock_replay_omn18757.py` exist so that the
# next port of this file cannot drop the flags silently a second time.


def parse_replay_window(parser: argparse.ArgumentParser, raw: str) -> datetime:
    """`--replay-before` as an aware UTC instant, or a usage error.

    Deliberately strict about the form: this flag waives guards, so a value
    the operator mistyped must stop the command rather than be coerced into
    some nearby instant that silently widens the window.
    """
    try:
        return datetime.strptime(raw.strip(), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        parser.error(
            "--replay-before must be an ISO-8601 UTC instant of the exact form "
            f"2026-09-16T10:41:37Z, not {raw!r}"
        )
        raise AssertionError("unreachable: parser.error exits")  # pragma: no cover


def resolve_replay(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    payload: str | None,
) -> tuple[str | None, str | None]:
    """Decide whether this append is a sanctioned verbatim replay.

    Returns ``(None, None)`` when no replay was requested, ``(marker, None)``
    when one is sanctioned, and ``(None, reason)`` when one is refused.

    Pairing and shape problems go through ``parser.error`` (exit 2) rather
    than becoming refusals, because nothing about the payload or the ledger
    was consulted to reach them -- they are the operator holding the tool
    wrongly. A refusal (exit 65) is a judgement about the bytes.
    """
    if args.replay_before is None and args.replay_source is None:
        return None, None
    if args.replay_before is None:
        parser.error(
            "--replay-source requires --replay-before: naming a source does not by "
            "itself declare anything to be a restore"
        )
    if args.replay_source is None or not args.replay_source.strip():
        parser.error(
            "--replay-before requires --replay-source: a restore carrying no named "
            "provenance is indistinguishable from a bypass"
        )
    if payload is None:
        parser.error(
            "--replay-before applies to --append/--append-file only; there is no "
            "payload to restore under -- COMMAND or --roll-section"
        )
    before = parse_replay_window(parser, args.replay_before)

    if not _STRANDED_GUARD_PATH.is_file():
        return None, (
            "OMN-18433 replay REFUSED -- the committed deciding logic is not reachable "
            f"at {_STRANDED_GUARD_PATH}, and this tool holds no inline copy of it on "
            "purpose: a waiver granted by a caller that cannot read the rule it is "
            "waiving is a bypass. Nothing was written. Restore the module, or replay "
            "from a clone that has it"
        )
    guard = load_stranded_clone_guard()
    if guard is None:
        return None, (
            "OMN-18433 replay REFUSED -- the committed deciding logic at "
            f"{_STRANDED_GUARD_PATH} could not be imported, so no waiver can be "
            "granted. Nothing was written"
        )

    now = resolve_now()
    refusal = guard.replay_refusal(payload, before, now)
    if refusal is not None:
        return None, f"OMN-18433 replay REFUSED -- {refusal}. Nothing was written"
    return guard.replay_marker_row(payload, now, args.replay_source.strip()), None


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
    parser.add_argument(
        "--verify-claim-token",
        metavar="TOKEN",
        help=(
            "verify a claim token minted by an earlier --append: confirm its row is "
            "still on disk at the recorded byte offset and that it was appended "
            "before --mutation-at. Exits 0 when the claim precedes the mutation, "
            "1 when it does not (or the token does not match the ledger), 2 on a "
            "malformed token"
        ),
    )
    parser.add_argument(
        "--mutation-at",
        metavar="ISO8601",
        help=(
            "the instant of the mutation the claim is supposed to authorize, as an "
            "ISO-8601 UTC timestamp; required with --verify-claim-token"
        ),
    )
    parser.add_argument(
        "--roll-section",
        action="store_true",
        help=(
            "action: roll the capped section instead of appending -- move its oldest rows "
            "into --archive-dir, leaving --roll-keep-entries behind, and print a roll receipt. "
            "Rolls only when a cap is exceeded unless --force-roll is given"
        ),
    )
    parser.add_argument(
        "--force-roll",
        action="store_true",
        help="with --roll-section: roll even when the section is under its caps",
    )
    parser.add_argument(
        "--section-heading",
        metavar="HEADING",
        help=(
            "the exact heading line opening the append-only section that the caps below "
            "govern; the section runs from that line to EOF and the heading must occur "
            "exactly once"
        ),
    )
    parser.add_argument(
        "--max-section-rows",
        type=int,
        metavar="N",
        help="refuse or roll when the section would exceed N lines (requires --section-heading)",
    )
    parser.add_argument(
        "--max-section-bytes",
        type=int,
        metavar="N",
        help="refuse or roll when the section would exceed N bytes (requires --section-heading)",
    )
    parser.add_argument(
        "--max-append-bytes",
        type=int,
        metavar="N",
        help=(
            "refuse a single append larger than N bytes; a row bigger than the section cap "
            "cannot be made to fit by rolling, so it is refused outright"
        ),
    )
    parser.add_argument(
        "--on-cap",
        choices=("roll", "block"),
        help=(
            "what to do when a cap would be crossed: 'roll' archives the oldest rows first "
            "and then appends; 'block' refuses the append (exit "
            f"{EXIT_SECTION_CAP}) pending a --roll-section. Required whenever a cap is set"
        ),
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        metavar="DIR",
        help="where rolled rows are written (required with --on-cap roll)",
    )
    parser.add_argument(
        "--roll-keep-entries",
        type=int,
        metavar="N",
        help="how many of the newest rows stay in the live section after a roll",
    )
    parser.add_argument(
        "--cost-unknown",
        metavar="REASON",
        action="append",
        default=[],
        type=cost_unknown_reason_type,
        help=(
            "documented per-row escape (OMN-15649 AC-6) for a claim row that genuinely "
            "cannot carry a rule-4 cost sentence; REPEATABLE -- pass one per offending "
            "row, in the order the offending rows appear in the payload (a reason never "
            "silently applies to more than the one row it is matched to). REASON is "
            "recorded verbatim in that row only, visible in the ledger -- this is not a "
            "silent bypass. Whitespace-only REASON is rejected."
        ),
    )
    parser.add_argument(
        "--replay-before",
        metavar="ISO8601",
        help=(
            "restore a row RECOVERED from a tree that was never committed: waive the "
            "append-time guards for this one payload, whose own leading UTC timestamp "
            "must be strictly before ISO8601. The row is written byte for byte and a "
            "REPLAY marker row recording its provenance lands immediately before it. "
            "Requires --replay-source. Not a way to write a row a guard would refuse "
            "today: a payload with no leading timestamp, one stamped at or after "
            "ISO8601, and a future ISO8601 are each refused (exit 65, nothing written)"
        ),
    )
    parser.add_argument(
        "--replay-source",
        metavar="NAME",
        help=(
            "where the recovered bytes came from, recorded verbatim in the REPLAY "
            "marker row; required with --replay-before, because a restore carrying no "
            "provenance is indistinguishable from a bypass"
        ),
    )
    return parser


def validate_section_cap_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Reject a half-configured cap rather than silently not capping.

    Every combination below is a caller who believes a cap is in force. A
    silently inert cap is the failure this whole feature exists to prevent,
    so each one is a usage error at parse time.
    """
    cap_flags = (args.max_section_rows, args.max_section_bytes)
    any_cap = any(cap is not None for cap in cap_flags)
    section_only_flags = (
        args.on_cap,
        args.archive_dir,
        args.roll_keep_entries,
        args.max_append_bytes,
    )
    if args.section_heading is None:
        if (
            any_cap
            or any(flag is not None for flag in section_only_flags)
            or args.roll_section
        ):
            parser.error(
                "--max-section-rows/--max-section-bytes/--max-append-bytes/--on-cap/"
                "--archive-dir/--roll-keep-entries/--roll-section all require --section-heading"
            )
        return
    if not any_cap:
        parser.error(
            "--section-heading requires at least one of --max-section-rows or --max-section-bytes "
            "-- a section heading with no cap does nothing"
        )
    if args.on_cap is None:
        parser.error(
            "--max-section-rows/--max-section-bytes require --on-cap {roll,block}"
        )
    for cap in cap_flags:
        if cap is not None and cap < 1:
            parser.error("--max-section-rows/--max-section-bytes must be at least 1")
    if args.max_append_bytes is not None and args.max_append_bytes < 1:
        parser.error("--max-append-bytes must be at least 1")
    if args.on_cap == "roll":
        if args.archive_dir is None:
            parser.error("--on-cap roll requires --archive-dir")
        if args.roll_keep_entries is None:
            parser.error("--on-cap roll requires --roll-keep-entries")
        if args.roll_keep_entries < 1:
            parser.error("--roll-keep-entries must be at least 1")
    if args.roll_section and args.on_cap != "roll":
        parser.error("--roll-section requires --on-cap roll")
    if args.force_roll and not args.roll_section:
        parser.error("--force-roll requires --roll-section")


def section_cap_exceeded(
    parsed_lines: int, parsed_bytes: int, args: argparse.Namespace
) -> str | None:
    """The name of the cap that `parsed_lines`/`parsed_bytes` crosses, if any."""
    if args.max_section_rows is not None and parsed_lines > args.max_section_rows:
        return f"--max-section-rows ({parsed_lines} > {args.max_section_rows})"
    if args.max_section_bytes is not None and parsed_bytes > args.max_section_bytes:
        return f"--max-section-bytes ({parsed_bytes} > {args.max_section_bytes})"
    return None


def split_command(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" not in argv:
        return argv, []
    delimiter = argv.index("--")
    return argv[:delimiter], argv[delimiter + 1 :]


def _payload_growth(payload: str) -> tuple[int, int]:
    """Lines and bytes `payload` adds to a section, as append_text will write it."""
    normalized = payload if payload.endswith("\n") else payload + "\n"
    return normalized.count("\n"), len(normalized.encode("utf-8"))


def run_roll_section(args: argparse.Namespace) -> int:
    """--roll-section, under the ledger lock."""
    parsed = parse_section_file(args.ledger, args.section_heading)
    over = section_cap_exceeded(parsed.line_count(), parsed.byte_count(), args)
    if over is None and not args.force_roll:
        plan = plan_roll(
            args.ledger,
            args.section_heading,
            args.archive_dir,
            args.roll_keep_entries,
            utc_now(),
        )
        receipt = dict(plan.receipt)
        receipt["entries_rolled"] = 0
        receipt["entries_kept"] = len(parsed.entries)
        receipt["last_rolled_heading"] = None
        receipt["section_lines_after"] = parsed.line_count()
        receipt["section_bytes_after"] = parsed.byte_count()
        print(f"{ROLL_RECEIPT_PREFIX}{json.dumps(receipt, sort_keys=True)}")
        return 0
    plan = plan_roll(
        args.ledger,
        args.section_heading,
        args.archive_dir,
        args.roll_keep_entries,
        utc_now(),
    )
    # A roll that fires and does not get under the cap is a FAILURE, and it has
    # to say so (OMN-17403). It used to print a success receipt and exit 0 --
    # on the live ledger on 2026-09-16 that was `entries_rolled: 0,
    # section_lines_after: 8220` against a cap of 4000, exit 0. A trigger
    # reading that receipt cannot tell it apart from a healthy no-op, so twelve
    # days of daily firings and twelve days of nothing firing at all produce
    # the same evidence. Nothing is written on the refusal.
    still_over = section_cap_exceeded(
        plan.receipt["section_lines_after"],
        plan.receipt["section_bytes_after"],
        args,
    )
    if still_over is not None:
        print(
            f"ledger_lock: ROLL REFUSED -- {args.section_heading!r} still crosses "
            f"{still_over} after rolling {plan.receipt['entries_rolled']} of "
            f"{plan.receipt['entries_rolled'] + plan.receipt['entries_kept']} rows. "
            "Nothing was written. Lower --roll-keep-entries or raise the cap.",
            file=sys.stderr,
        )
        print(f"{ROLL_RECEIPT_PREFIX}{json.dumps(plan.receipt, sort_keys=True)}")
        return EXIT_SECTION_CAP
    announce_roll(plan, apply_roll(args.ledger, plan))
    print(f"{ROLL_RECEIPT_PREFIX}{json.dumps(plan.receipt, sort_keys=True)}")
    return 0


def enforce_row_shape(args: argparse.Namespace, payload: str) -> int | None:
    """Refuse an append that would extend the previous row instead of opening
    its own (OMN-17403).

    Scoped to a caller that named a capped section, because that is the caller
    who has declared the file to be row-structured. Without this the row model
    holds only by the goodwill of every lane in the fleet, and one careless
    payload silently rewrites the digest of the row above it -- which is
    exactly how the 2026-09-04 friction-sweep anchor was invalidated.
    """
    if args.section_heading is None:
        return None
    # Resolve the section BEFORE judging the payload, so a heading that is
    # absent or duplicated still fails closed as the usage error it is rather
    # than being masked by a shape complaint about the payload.
    parse_section_file(args.ledger, args.section_heading)
    if opens_a_row(payload):
        return None
    first = next((line for line in payload.splitlines() if line.strip()), "")
    print(
        "ledger_lock: ROW SHAPE REFUSED -- this payload does not open a row in "
        f"{args.section_heading!r}, so appending it would extend the row above it. "
        f"First line: {first[:120]!r}. A row opens with a UTC date (optionally "
        "behind a '- ' bullet or a '| ' table pipe) or with a markdown heading. "
        "Nothing was written.",
        file=sys.stderr,
    )
    return EXIT_ROW_SHAPE


def enforce_section_caps(args: argparse.Namespace, payload: str) -> int | None:
    """Apply the section caps to a pending append.

    Returns None when the append may proceed (having rolled first if that is
    the configured policy and it makes room), or EXIT_SECTION_CAP when the
    append is refused. A refusal writes nothing at all -- not the row, and not
    a roll -- because a roll that fires and still cannot fit the row leaves a
    split file with the row lost.
    """
    if args.section_heading is None:
        return None
    added_lines, added_bytes = _payload_growth(payload)
    if args.max_append_bytes is not None and added_bytes > args.max_append_bytes:
        print(
            f"ledger_lock: SECTION CAP -- this row is {added_bytes} bytes, over "
            f"--max-append-bytes ({args.max_append_bytes}). Rolling cannot make a single "
            "row smaller; split the row or raise the cap. Nothing was written.",
            file=sys.stderr,
        )
        return EXIT_SECTION_CAP
    parsed = parse_section_file(args.ledger, args.section_heading)
    over = section_cap_exceeded(
        parsed.line_count() + added_lines, parsed.byte_count() + added_bytes, args
    )
    if over is None:
        return None
    if args.on_cap == "block":
        print(
            f"ledger_lock: SECTION CAP -- appending would cross {over} in section "
            f"{args.section_heading.strip()!r}. Nothing was written. Roll the section first: "
            f"--roll-section --section-heading ... --on-cap roll --archive-dir ... "
            "--roll-keep-entries N",
            file=sys.stderr,
        )
        return EXIT_SECTION_CAP
    plan = plan_roll(
        args.ledger,
        args.section_heading,
        args.archive_dir,
        args.roll_keep_entries,
        utc_now(),
    )
    still_over = section_cap_exceeded(
        plan.receipt["section_lines_after"] + added_lines,
        plan.receipt["section_bytes_after"] + added_bytes,
        args,
    )
    if still_over is not None:
        print(
            f"ledger_lock: SECTION CAP -- a roll keeping {args.roll_keep_entries} rows still "
            f"crosses {still_over}. Nothing was written and nothing was rolled; lower "
            "--roll-keep-entries or raise the cap.",
            file=sys.stderr,
        )
        return EXIT_SECTION_CAP
    announce_roll(plan, apply_roll(args.ledger, plan))
    print(f"{ROLL_RECEIPT_PREFIX}{json.dumps(plan.receipt, sort_keys=True)}")
    return None


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser_argv, command = split_command(raw_argv)
    parser = build_parser()
    args = parser.parse_args(parser_argv)
    validate_section_cap_args(parser, args)
    payload = read_append_payload(args)

    # Verification is a read-only query about a token, not one of the three
    # mutating actions, so it is checked (and returns) before the
    # exactly-one-action rule applies.
    if args.verify_claim_token is not None:
        if not args.mutation_at:
            parser.error("--verify-claim-token requires --mutation-at")
        token = ClaimToken.parse(args.verify_claim_token)
        if token is None:
            print(
                f"ledger_lock: malformed claim token: {args.verify_claim_token!r} "
                f"(expected {CLAIM_TOKEN_VERSION}-<offset>-<line>-<digest>-<appended_at>)",
                file=sys.stderr,
            )
            return 2
        code, message = verify_claim_token(args.ledger, token, args.mutation_at)
        stream = sys.stdout if code == 0 else sys.stderr
        print(f"ledger_lock: {message}", file=stream)
        return code

    requested_actions = sum(
        1 for item in (payload, command, args.roll_section or None) if item
    )
    if requested_actions != 1:
        parser.error(
            "provide exactly one action: --append, --append-file, --roll-section, or -- COMMAND"
        )
    if args.cost_unknown and not payload:
        parser.error(
            "--cost-unknown only applies to --append/--append-file, not -- COMMAND"
        )

    # --- OMN-18433: is this a sanctioned verbatim replay? Resolved BEFORE the
    # lint chain below, because a sanctioned replay is precisely the case in
    # which that chain must not run: the chain judges a row being written now,
    # and these bytes were written at their own timestamp on a copy of this
    # ledger that was lost. A refusal here writes nothing at all.
    replay_marker, replay_refusal_reason = resolve_replay(parser, args, payload)
    if replay_refusal_reason is not None:
        print(f"ledger_lock: {replay_refusal_reason}", file=sys.stderr)
        return 65

    # --- OMN-18554: the pre-append lint chain, ported from the omni_home copy.
    # Order is load-bearing and is the order that copy used. The first three are
    # row-class-agnostic and are NOT window- or grace-scoped: a row that miscounts
    # (OMN-15897), defers with no follow-up (OMN-15901), or lies about WHEN it was
    # written (OMN-17427) poisons the history whatever else it says. The rule-4
    # cost-sentence gate (OMN-15649/OMN-18554) runs last because it is the only
    # one of the four that is scoped -- to claim rows, and to a declared window.
    payload_to_write: str | None = payload
    if payload is not None and replay_marker is None:
        quant_reason = validate_quantitative_claims_payload(payload)
        if quant_reason is not None:
            print(
                f"ledger_lock: OMN-15897 quantitative-claim lint rejected append -- {quant_reason}",
                file=sys.stderr,
            )
            return 65
        defer_reason = validate_defer_language_payload(payload)
        if defer_reason is not None:
            print(
                f"ledger_lock: OMN-15901 defer-language lint rejected append -- {defer_reason}",
                file=sys.stderr,
            )
            return 65
        now = resolve_now()
        clock_reason = validate_clock_skew_payload(payload, now)
        if clock_reason is not None:
            print(
                f"ledger_lock: OMN-17427 clock guard rejected append -- {clock_reason}",
                file=sys.stderr,
            )
            return 65
        payload_to_write, rejection = validate_claim_payload(
            payload, cost_unknown=args.cost_unknown, now=now
        )
        if rejection is not None:
            print(f"ledger_lock: {rejection}", file=sys.stderr)
            return 65
        # A --cost-unknown escape rewrites the row it annotates, so the rest of
        # main() must see the bytes that actually land -- otherwise the dedup
        # window, the row digest and the minted claim token would all describe a
        # payload that was never written.
        payload = payload_to_write

    try:
        with LedgerLock(args.ledger, args.timeout, args.stale_after, command or None):
            if args.roll_section:
                return run_roll_section(args)
            if payload is not None:
                # A replay writes TWO rows, so the shape and cap checks judge
                # both: the marker is a row like any other and must not be
                # able to overflow a capped section or extend the row above
                # it just because it rides in beside a restored row.
                projected = (
                    payload if replay_marker is None else f"{replay_marker}\n{payload}"
                )
                shape_rc = enforce_row_shape(args, projected)
                if shape_rc is not None:
                    return shape_rc
                cap_rc = enforce_section_caps(args, projected)
                if cap_rc is not None:
                    return cap_rc
                # A replayed row never mints a claim token. Its claim, if it
                # made one, was made at its own timestamp and whatever it
                # authorized is long settled; a fresh token minted now could
                # be cited to authorize a mutation TODAY, which is exactly the
                # bypass this path must not open.
                claim_shaped = is_claim_row(payload) and replay_marker is None
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
                    if claim_shaped:
                        # Hand back the FIRST attempt's token so a retry is
                        # token-stable: the caller cites one token in its
                        # mutation however many attempts the append took.
                        existing = find_existing_claim_token(args.ledger, payload)
                        if existing is not None:
                            print(f"{CLAIM_TOKEN_PREFIX}{existing.render()}")
                    return 0
                # Offset is read under the lock, immediately before the write
                # that lands at it, so it is the true append position.
                # OMN-18258 / OMN-18274: both are STATE-dependent (they read
                # rows already in this ledger), so they run here inside the held
                # lock rather than in the pre-lock chain above.
                #
                # Both are skipped for a sanctioned replay, for the same
                # reason the pre-lock chain is: the OMN-18274 friction guard
                # is the gate that refused five of the recovered rows in the
                # first place, and a restore that has to satisfy it is a
                # rewrite, not a restore.
                if replay_marker is None:
                    ruling_reason = validate_ruling_payload(payload, args.ledger)
                    if ruling_reason is not None:
                        print(f"ledger_lock: {ruling_reason}", file=sys.stderr)
                        return 65
                    friction_reason = validate_friction_payload(payload, args.ledger)
                    if friction_reason is not None:
                        print(f"ledger_lock: {friction_reason}", file=sys.stderr)
                        return 65
                else:
                    # The marker lands FIRST, so the offset computed below is
                    # the restored row's own, and a reader scanning upwards
                    # from the row finds its provenance on the line above.
                    append_text(args.ledger, replay_marker)
                offset = _ledger_size(args.ledger)
                line_no = len(_offsets_and_lines(args.ledger)) + 1
                appended_at = utc_now()
                append_text(args.ledger, payload)
                if claim_shaped:
                    token = ClaimToken(
                        offset=offset,
                        line_no=line_no,
                        digest=claim_row_digest(payload),
                        appended_at=appended_at,
                    )
                    print(f"{CLAIM_TOKEN_PREFIX}{token.render()}")
                # OMN-18433: after the row, never instead of it.
                return signal_stranded_clone(args.ledger, payload)
            existed_before = args.ledger.exists()
            before = args.ledger.read_text(encoding="utf-8") if existed_before else ""
            rc = subprocess.call(command)
            # The editor verb REFUSES, it does not merely warn: a gate one verb
            # honours and the other narrates is not a gate. A refusal overrides
            # the command's own rc and restores the pre-command bytes.
            if enforce_command_claims(args.ledger, before, existed_before):
                return 65
            return rc
    except SectionError as exc:
        print(f"ledger_lock: {exc}", file=sys.stderr)
        return 2
    except TimeoutError as exc:
        print(f"ledger_lock: {exc}", file=sys.stderr)
        return 75
    except OSError as exc:
        print(f"ledger_lock: command failed to start: {exc}", file=sys.stderr)
        return 127


if __name__ == "__main__":
    raise SystemExit(main())
