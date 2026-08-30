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
ROLL_RECEIPT_SCHEMA = "ledger-roll/1"
ROLL_RECEIPT_PREFIX = "ledger_lock: ROLL "
ROLL_POINTER_MARKER = "<!-- ledger-roll:"
ROLL_POINTER_PROSE_PREFIX = "> Older rows live in"

# A row inside the section starts at a markdown heading. `##` and deeper are
# both in live use in the rolling work ledger, so both open a row.
ENTRY_HEADING_PATTERN = re.compile(r"^#{2,6} \S")


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
    starts = [i for i, line in enumerate(body) if ENTRY_HEADING_PATTERN.match(line)]
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


def _pointer_block(archive: Path, rolled: int, rolled_at: str, first_kept: str) -> str:
    marker = json.dumps(
        {
            "rolled_at": rolled_at,
            "archive": str(archive),
            "entries_rolled": rolled,
            "first_kept_heading": first_kept,
        },
        sort_keys=True,
    )
    return (
        f"{ROLL_POINTER_MARKER} {marker} -->\n"
        f"{ROLL_POINTER_PROSE_PREFIX} `{archive}` -- {rolled} rows rolled at {rolled_at}. "
        f"Rows older than {first_kept!r} are not in this file.\n\n"
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
    live_text = (
        parsed.head
        + parsed.heading_line
        + preamble
        + _pointer_block(archive_path, len(rolled), rolled_at, first_kept)
        + "".join(entry.text for entry in kept)
    )

    archive_header = (
        "<!-- ledger-roll-archive "
        + json.dumps(
            {
                "source": str(ledger),
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
    }
    return RollPlan(live_text, archive_text, archive_path, receipt, len(rolled))


def apply_roll(ledger: Path, plan: RollPlan) -> None:
    if plan.rolled == 0:
        return
    plan.archive_path.parent.mkdir(parents=True, exist_ok=True)
    existing = ""
    if plan.archive_path.exists():
        existing = plan.archive_path.read_text(encoding="utf-8")
        if existing and not existing.endswith("\n"):
            existing += "\n"
        existing += "\n"
    _write_text_durably(plan.archive_path, existing + plan.archive_text)
    _write_text_durably(ledger, plan.live_text)


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
    apply_roll(args.ledger, plan)
    print(f"{ROLL_RECEIPT_PREFIX}{json.dumps(plan.receipt, sort_keys=True)}")
    return 0


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
    apply_roll(args.ledger, plan)
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

    try:
        with LedgerLock(args.ledger, args.timeout, args.stale_after, command or None):
            if args.roll_section:
                return run_roll_section(args)
            if payload is not None:
                cap_rc = enforce_section_caps(args, payload)
                if cap_rc is not None:
                    return cap_rc
                claim_shaped = is_claim_row(payload)
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
                return 0
            rc = subprocess.call(command)
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
