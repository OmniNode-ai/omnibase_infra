# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for scripts/ledger_lock.py.

Covers the locking/append/exit-75-retry contract: the fcntl lock on one
lock file per resolved ledger path (OMN-19262; the single-holder races live in
test_ledger_lock_single_holder.py), durable append, dedup-window idempotent
retry, and the CLI's exit codes (0 success,
75 lock timeout, 127 bad -- COMMAND, argparse usage errors, and the -- COMMAND
verb's own passthrough exit code).
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "ledger_lock.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("ledger_lock", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _load_module()


# --- OMN-18554: fixture rows are stamped from the LIVE clock ----------------
#
# ledger_lock.py in this repo now carries the OMN-17427 wall-clock guard, ported
# from the omni_home copy where it had been living uncommitted. That guard reads
# a row's OWN leading timestamp and refuses one stamped more than 5 minutes ahead
# of the wall clock or more than 24 hours behind it. Every fixture row below was
# written before this repo's script had that guard, so each carried a frozen date
# literal that is now weeks in the past and is correctly refused.
#
# `_live_stamp` rewrites only the LEADING timestamp, at call time, and leaves the
# rest of each row byte-identical. The literal stays in the source as the shape
# documentation it always was; what changes is that the row is honest about when
# it was written, which is the only thing the guard asks.
_LEADING_TS = re.compile(r"(?<![\d])20\d{2}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


def _live_stamp(row: str, *, shift_seconds: int = 0) -> str:
    """Rewrite the row's FIRST timestamp to now (+shift), leaving all else."""
    moment = datetime.now(UTC).replace(microsecond=0) + timedelta(seconds=shift_seconds)
    return _LEADING_TS.sub(moment.strftime("%Y-%m-%dT%H:%M:%SZ"), row, count=1)


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


# --------------------------------------------------------------------------
# parse_duration
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("30s", 30.0),
        ("5m", 300.0),
        ("1h", 3600.0),
        ("250ms", 0.25),
        ("10", 10.0),
    ],
)
def test_parse_duration_units(raw: str, expected: float) -> None:
    assert MOD.parse_duration(raw) == pytest.approx(expected)


def test_parse_duration_rejects_empty() -> None:
    with pytest.raises(Exception):  # argparse.ArgumentTypeError
        MOD.parse_duration("")


def test_parse_duration_rejects_negative() -> None:
    with pytest.raises(Exception):
        MOD.parse_duration("-5s")


# --------------------------------------------------------------------------
# ledger_path / lock_path_for
# --------------------------------------------------------------------------


def test_ledger_path_resolves_relative_against_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    resolved = MOD.ledger_path("some/ledger.md")
    assert resolved == (tmp_path / "some" / "ledger.md").resolve()


def test_lock_path_for_is_stable_and_unique_per_ledger(tmp_path: Path) -> None:
    ledger_a = tmp_path / "a.md"
    ledger_b = tmp_path / "b.md"
    lock_a_first = MOD.lock_path_for(ledger_a)
    lock_a_second = MOD.lock_path_for(ledger_a)
    lock_b = MOD.lock_path_for(ledger_b)
    assert lock_a_first == lock_a_second
    assert lock_a_first != lock_b


def test_lock_file_sits_beside_the_resolved_ledger(tmp_path: Path) -> None:
    ledger = tmp_path / "sub" / "ledger.md"
    lock = MOD.lock_path_for(ledger)
    assert lock.parent == ledger.resolve().parent / MOD.DEFAULT_LOCK_DIRNAME
    assert lock.name.startswith("ledger.md.")
    assert lock.name.endswith(MOD.LOCK_FILE_SUFFIX)


def test_lock_path_ignores_ledger_lock_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OMN-19262: the roll exported LEDGER_LOCK_ROOT and lane appends did not,
    so the two held different locks on one ledger. Nothing moves it now."""
    ledger = tmp_path / "ledger.md"
    monkeypatch.delenv("LEDGER_LOCK_ROOT", raising=False)
    unset = MOD.lock_path_for(ledger)
    monkeypatch.setenv("LEDGER_LOCK_ROOT", str(tmp_path / "shared-locks"))
    assert MOD.lock_path_for(ledger) == unset


def test_lock_path_is_the_same_through_a_symlinked_directory(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    assert MOD.lock_path_for(alias / "ledger.md") == MOD.lock_path_for(
        real / "ledger.md"
    )


# --------------------------------------------------------------------------
# dedup
# --------------------------------------------------------------------------


def test_dedup_detects_identical_row_ignoring_timestamp_drift() -> None:
    tail = "- 2026-08-09T13:59:51Z [handle] claimed OMN-1 doing the thing\n"
    payload = "- 2026-08-09T13:59:57Z [handle] claimed OMN-1 doing the thing\n"
    assert MOD.is_duplicate_of_recent_tail(payload, tail) is True


def test_dedup_rejects_different_body() -> None:
    tail = "- 2026-08-09T13:59:51Z [handle] claimed OMN-1 doing the thing\n"
    payload = "- 2026-08-09T13:59:57Z [handle] claimed OMN-2 doing a different thing\n"
    assert MOD.is_duplicate_of_recent_tail(payload, tail) is False


def test_dedup_empty_payload_is_never_a_duplicate() -> None:
    assert MOD.is_duplicate_of_recent_tail("", "anything\n") is False


def test_dedup_empty_tail_is_never_matched() -> None:
    assert MOD.is_duplicate_of_recent_tail("- some row\n", "") is False


def test_dedup_payload_longer_than_tail_window_is_not_a_duplicate() -> None:
    tail = "line one\n"
    payload = "line one\nline two\n"
    assert MOD.is_duplicate_of_recent_tail(payload, tail) is False


# --------------------------------------------------------------------------
# LedgerLock acquire/release + append_text
# --------------------------------------------------------------------------


def test_ledger_lock_acquire_release_roundtrip(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.md"
    lock = MOD.LedgerLock(ledger, timeout=1.0)
    with lock:
        assert lock.acquired is True
        assert lock.lock_file.is_file()
    assert lock.acquired is False
    # The file outlives the holder: deleting it on release would let a waiter
    # holding the old inode and a writer creating a new one both hold.
    assert lock.lock_file.is_file()
    again = MOD.LedgerLock(ledger, timeout=0.0)
    with again:
        assert again.acquired is True


def test_ledger_lock_times_out_when_already_held(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.md"
    holder = MOD.LedgerLock(ledger, timeout=1.0)
    holder.acquire()
    try:
        waiter = MOD.LedgerLock(ledger, timeout=0.0)
        with pytest.raises(TimeoutError):
            waiter.acquire()
    finally:
        holder.release()


def test_release_twice_and_release_unheld_are_no_ops(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.md"
    never_held = MOD.LedgerLock(ledger, timeout=0.0)
    never_held.release()
    lock = MOD.LedgerLock(ledger, timeout=0.0)
    lock.acquire()
    lock.release()
    lock.release()
    assert lock.acquired is False


def test_lock_directory_ignores_its_own_contents(tmp_path: Path) -> None:
    """The lock file is permanent, so it must never read as untracked in the
    repository that carries the ledger."""
    repo = tmp_path / "repo"
    ledger = repo / "docs" / "ledger.md"
    ledger.parent.mkdir(parents=True)
    subprocess.run(
        ["git", "init", "-q", str(repo)],
        check=True,
        env=scrub_git_location_env(os.environ),
    )
    with MOD.LedgerLock(ledger, timeout=0.0):
        pass
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=True,
        env=scrub_git_location_env(os.environ),
    )
    assert status.stdout == "", status.stdout
    # Positive control: an ordinary file beside the ledger does show up.
    (ledger.parent / "other.md").write_text("x\n", encoding="utf-8")
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=True,
        env=scrub_git_location_env(os.environ),
    )
    assert "docs/other.md" in status.stdout


def test_a_lock_file_replaced_before_the_flock_is_not_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A writer that locked an unlinked file would exclude no one, so the
    identity check sends it back to open the path again."""
    ledger = tmp_path / "ledger.md"
    lock = MOD.LedgerLock(ledger, timeout=1.0)
    real_flock = MOD.fcntl.flock
    swapped: list[bool] = []

    def flock_after_swap(fd: int, operation: int) -> None:
        if not swapped and operation & MOD.fcntl.LOCK_EX:
            swapped.append(True)
            lock.lock_file.unlink()
            lock.lock_file.write_text("", encoding="utf-8")
        real_flock(fd, operation)

    monkeypatch.setattr(MOD.fcntl, "flock", flock_after_swap)
    with lock:
        held = os.fstat(lock._fd)
        on_disk = lock.lock_file.stat()
        assert (held.st_dev, held.st_ino) == (on_disk.st_dev, on_disk.st_ino)
    assert swapped == [True]


def test_append_text_is_durable_and_newline_terminated(tmp_path: Path) -> None:
    ledger = tmp_path / "nested" / "ledger.md"
    MOD.append_text(ledger, "first row")
    MOD.append_text(ledger, "second row\n")
    assert ledger.read_text(encoding="utf-8") == "first row\nsecond row\n"


def test_read_ledger_tail_missing_file_is_empty(tmp_path: Path) -> None:
    assert MOD.read_ledger_tail(tmp_path / "nope.md", 20) == ""


def test_read_ledger_tail_returns_last_n_lines(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.md"
    ledger.write_text("a\nb\nc\nd\n", encoding="utf-8")
    assert MOD.read_ledger_tail(ledger, 2) == "c\nd"


# --------------------------------------------------------------------------
# CLI end-to-end
# --------------------------------------------------------------------------


def test_cli_append_lands_on_disk(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.md"
    row = _live_stamp("- 2026-08-25T00:00:00Z row one")
    result = _run_cli([str(ledger), "--append", row])
    assert result.returncode == 0, result.stderr
    assert ledger.read_text(encoding="utf-8") == row + "\n"


def test_cli_append_retry_is_deduped(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.md"
    first = _run_cli(
        [str(ledger), "--append", _live_stamp("- 2026-08-25T00:00:00Z row one")]
    )
    assert first.returncode == 0, first.stderr
    # Simulate a retry after an exit-75 timeout: same tag/body, a few
    # seconds' worth of timestamp drift.
    second = _run_cli(
        [str(ledger), "--append", _live_stamp("- 2026-08-25T00:00:05Z row one")]
    )
    assert second.returncode == 0, second.stderr
    assert "DEDUP" in second.stderr
    # Only one copy landed.
    assert ledger.read_text(encoding="utf-8").count("row one") == 1


def test_cli_exit_75_on_lock_timeout(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.md"
    holder = MOD.LedgerLock(ledger.resolve(), timeout=0.0)
    holder.acquire()
    try:
        result = _run_cli(
            [str(ledger), "--timeout", "1s", "--append", "should not land"]
        )
        assert result.returncode == 75
        assert "timed out" in result.stderr
        # The timeout names the holder to chase.
        assert f"pid={os.getpid()}" in result.stderr
        assert not ledger.exists()
    finally:
        holder.release()


def test_cli_no_longer_accepts_the_stale_break_flags(tmp_path: Path) -> None:
    """There is no stale lock to break: a dead holder's flock is gone."""
    ledger = tmp_path / "ledger.md"
    for flag in ("--stale-after", "--anonymous-stale-after"):
        result = _run_cli([str(ledger), flag, "1s", "--", "true"])
        assert result.returncode == 2, (flag, result.stderr)


def test_cli_command_verb_passes_through_exit_code(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.md"
    ok = _run_cli([str(ledger), "--", "true"])
    assert ok.returncode == 0
    failing = _run_cli([str(ledger), "--", "false"])
    assert failing.returncode == 1


def test_cli_exit_127_on_missing_command(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.md"
    result = _run_cli([str(ledger), "--", "/no-such-binary-xyz"])
    assert result.returncode == 127
    assert "command failed to start" in result.stderr


def test_cli_requires_exactly_one_action(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.md"
    no_action = _run_cli([str(ledger)])
    assert no_action.returncode != 0

    both_actions = _run_cli([str(ledger), "--append", "x", "--", "true"])
    assert both_actions.returncode != 0


# --------------------------------------------------------------------------
# OMN-16400: claim-row idempotency across a shifted self-stamp
#
# The rows every lane actually writes into ROLLING_WORK_LEDGER.md are
# five-column pipe rows whose FIRST CELL is the caller's self-stamped UTC
# timestamp (see e.g. ledger L19735/L20080). The OMN-15787 normalizer only
# strips a timestamp that leads the line after an optional `- `/`* ` bullet,
# so it never fires on that shape: an exit-75 retry whose `$(date -u ...)`
# advanced a few seconds lands a SECOND claim row for the same claim. That
# is the duplicate/ghost-claim class this ticket exists to close.
# --------------------------------------------------------------------------

# OMN-18554: this repo's ledger_lock.py now carries the OMN-15649 rule-4
# cost-sentence gate, ported from the omni_home copy. A claim row is exactly the
# row class that gate inspects, so the fixture now carries a conforming cost
# sentence. That is not a workaround: these tests assert that a CLAIM row mints a
# claim token, and a claim row that the gate refuses never reaches the minting
# path at all, so an unpriced fixture would be testing the refusal instead.
# OMN-18766: a claim row must also NAME ITS EXECUTOR (actor=/model=), so the
# fixtures below carry actor=. Same reasoning the OMN-18554 note above gives for
# the cost sentence: a row the attribution gate refuses never reaches the
# behaviour under test, so an unattributed fixture would be testing that refusal.
_PIPE_CLAIM = (
    "| {ts} | build-OMN-16400 | OMN-16400 | CLAIM | actor=claude:opus5 | "
    "Claiming the ledger hardening work; est ~2 lane-hours; displaces nothing; (OMN-16400). |"
)


def test_dedup_detects_pipe_row_claim_retry_with_shifted_self_stamp() -> None:
    tail = _PIPE_CLAIM.format(ts="2026-08-28T18:10:00Z") + "\n"
    payload = _PIPE_CLAIM.format(ts="2026-08-28T18:12:31Z") + "\n"
    assert MOD.is_duplicate_of_recent_tail(payload, tail) is True


def test_dedup_still_rejects_two_distinct_pipe_rows() -> None:
    """The leading-cell strip must not collapse genuinely different rows."""
    tail = (
        "| 2026-08-28T18:10:00Z | build-OMN-16400 | OMN-16400 | CLAIM | "
        "Claiming the ledger hardening work. |\n"
    )
    payload = (
        "| 2026-08-28T18:12:31Z | build-OMN-16401 | OMN-16401 | CLAIM | "
        "Claiming a different piece of work. |\n"
    )
    assert MOD.is_duplicate_of_recent_tail(payload, tail) is False


def test_dedup_detects_bold_wrapped_leading_timestamp_retry() -> None:
    tail = "- **2026-08-28T18:10:00Z** build-x OMN-16400 CLAIM body\n"
    payload = "- **2026-08-28T18:12:31Z** build-x OMN-16400 CLAIM body\n"
    assert MOD.is_duplicate_of_recent_tail(payload, tail) is True


def test_dedup_does_not_strip_a_mid_body_timestamp() -> None:
    """Only a LEADING self-stamp is normalized away.

    A timestamp quoted inside the body is evidence, not a self-stamp: two
    rows citing different mutation instants are different rows.
    """
    tail = "- [h] OMN-1 TERMINAL merged at 2026-08-28T18:10:00Z\n"
    payload = "- [h] OMN-1 TERMINAL merged at 2026-08-28T18:12:31Z\n"
    assert MOD.is_duplicate_of_recent_tail(payload, tail) is False


# --------------------------------------------------------------------------
# OMN-16400: claim tokens
# --------------------------------------------------------------------------


def test_is_claim_row_recognizes_the_live_row_shapes() -> None:
    assert MOD.is_claim_row(_PIPE_CLAIM.format(ts="2026-08-28T18:10:00Z"))
    assert MOD.is_claim_row("- 2026-08-28T18:10:00Z [handle] OMN-1 — CLAIM: doing x")
    assert MOD.is_claim_row("- **Status:** CLAIM+TERMINAL")


def test_is_claim_row_rejects_a_non_claim_row() -> None:
    assert not MOD.is_claim_row(
        "| 2026-08-28T18:05:00Z | build-x | OMN-1 | TERMINAL | done |"
    )
    assert not MOD.is_claim_row("- just some prose about a claim of victory")


def test_cli_append_of_a_claim_row_returns_a_claim_token(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.md"
    ledger.write_text("existing header line\n", encoding="utf-8")
    row = _PIPE_CLAIM.format(ts="2026-08-28T18:10:00Z")

    result = _run_cli([str(ledger), "--append", _live_stamp(row)])

    assert result.returncode == 0
    token = MOD.parse_claim_token_line(result.stdout)
    assert token is not None
    # The offset is the lock-protected byte position the row landed at --
    # i.e. the authoritative append-order signal, not a self-declared clock.
    assert token.offset == len("existing header line\n")


def test_cli_append_of_a_non_claim_row_emits_no_token(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.md"
    result = _run_cli(
        [
            str(ledger),
            "--append",
            _live_stamp("| 2026-08-28T18:05:00Z | h | OMN-1 | TERMINAL | x |"),
        ]
    )
    assert result.returncode == 0
    assert MOD.parse_claim_token_line(result.stdout) is None


def test_cli_claim_retry_is_deduped_and_returns_the_same_token(
    tmp_path: Path,
) -> None:
    """An exit-75 retry must be a no-op that hands back the FIRST token.

    Token stability is what makes the retry safe: the caller cites one token
    in its mutation regardless of how many times it retried the append.
    """
    ledger = tmp_path / "ledger.md"
    first = _run_cli(
        [
            str(ledger),
            "--append",
            _live_stamp(_PIPE_CLAIM.format(ts="2026-08-28T18:10:00Z")),
        ]
    )
    retry = _run_cli(
        [
            str(ledger),
            "--append",
            _live_stamp(_PIPE_CLAIM.format(ts="2026-08-28T18:12:31Z"), shift_seconds=5),
        ]
    )

    assert first.returncode == 0
    assert retry.returncode == 0
    assert "DEDUP" in retry.stderr
    assert ledger.read_text(encoding="utf-8").count("| CLAIM |") == 1

    first_token = MOD.parse_claim_token_line(first.stdout)
    retry_token = MOD.parse_claim_token_line(retry.stdout)
    assert first_token is not None and retry_token is not None
    assert retry_token.offset == first_token.offset
    assert retry_token.digest == first_token.digest


# --------------------------------------------------------------------------
# OMN-16400: verifying a claim token predates a cited mutation
# --------------------------------------------------------------------------


def _token_of(result: subprocess.CompletedProcess[str]) -> str:
    token = MOD.parse_claim_token_line(result.stdout)
    assert token is not None
    return token.render()


def test_cli_verify_claim_token_accepts_a_claim_that_predates_the_mutation(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "ledger.md"
    appended = _run_cli(
        [
            str(ledger),
            "--append",
            _live_stamp(_PIPE_CLAIM.format(ts="2026-08-28T18:10:00Z")),
        ]
    )
    token = _token_of(appended)

    verdict = _run_cli(
        [
            str(ledger),
            "--verify-claim-token",
            token,
            "--mutation-at",
            "2099-01-01T00:00:00Z",
        ]
    )
    assert verdict.returncode == 0
    assert "CLAIM-BEFORE-MUTATION OK" in verdict.stdout


def test_cli_verify_claim_token_rejects_a_post_hoc_claim(tmp_path: Path) -> None:
    """The L17421/L17467/L17574 defect class, mechanically caught."""
    ledger = tmp_path / "ledger.md"
    appended = _run_cli(
        [
            str(ledger),
            "--append",
            _live_stamp(_PIPE_CLAIM.format(ts="2026-08-28T18:10:00Z")),
        ]
    )
    token = _token_of(appended)

    verdict = _run_cli(
        [
            str(ledger),
            "--verify-claim-token",
            token,
            "--mutation-at",
            "2000-01-01T00:00:00Z",
        ]
    )
    assert verdict.returncode == 1
    assert "POST-HOC CLAIM" in verdict.stderr


def test_cli_verify_claim_token_rejects_a_token_whose_row_is_not_on_disk(
    tmp_path: Path,
) -> None:
    """A token is only worth as much as the row it points at.

    Verification re-reads the ledger at the recorded offset and re-hashes:
    a token citing a row that was never appended (or was rewritten) fails,
    so a caller cannot mint a claim token for a claim it never made.
    """
    ledger = tmp_path / "ledger.md"
    appended = _run_cli(
        [
            str(ledger),
            "--append",
            _live_stamp(_PIPE_CLAIM.format(ts="2026-08-28T18:10:00Z")),
        ]
    )
    token = _token_of(appended)
    ledger.write_text("something else entirely\n", encoding="utf-8")

    verdict = _run_cli(
        [
            str(ledger),
            "--verify-claim-token",
            token,
            "--mutation-at",
            "2099-01-01T00:00:00Z",
        ]
    )
    assert verdict.returncode == 1
    assert "TOKEN DOES NOT MATCH" in verdict.stderr


def test_cli_verify_claim_token_rejects_a_malformed_token(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.md"
    ledger.write_text("x\n", encoding="utf-8")
    verdict = _run_cli(
        [
            str(ledger),
            "--verify-claim-token",
            "not-a-token",
            "--mutation-at",
            "2099-01-01T00:00:00Z",
        ]
    )
    assert verdict.returncode == 2


def test_claim_tokens_order_by_lock_protected_offset(tmp_path: Path) -> None:
    """Two claims on one ledger order by file position, not by self-stamp.

    This is the ghost-collision fix: the earlier-appended row has the smaller
    offset even when its self-stamped wall clock reads LATER, which is
    exactly the inversion that produced the L17511 VOID ruling and its
    L17523 retraction.
    """
    ledger = tmp_path / "ledger.md"
    earlier_append_later_clock = _run_cli(
        [
            str(ledger),
            "--append",
            # OMN-18554: both rows are stamped live and priced, so they clear the
            # ported clock and rule-4 guards. The INVERSION this test exists for is
            # preserved exactly: the row appended FIRST carries the LATER self-stamp
            # (shift 0 here, -60s below), which is the ghost-collision shape.
            _live_stamp(
                "| 2026-08-22T14:45:00Z | lane-a | OMN-16385 | CLAIM | actor=claude:opus5 | first appended; "
                "est ~2 lane-hours; displaces nothing; (OMN-16385) |"
            ),
        ]
    )
    later_append_earlier_clock = _run_cli(
        [
            str(ledger),
            "--append",
            _live_stamp(
                "| 2026-08-22T14:20:00Z | lane-b | OMN-16386 | CLAIM | actor=claude:opus5 | second appended; "
                "est ~2 lane-hours; displaces nothing; (OMN-16386) |",
                shift_seconds=-60,
            ),
        ]
    )
    first = MOD.parse_claim_token_line(earlier_append_later_clock.stdout)
    second = MOD.parse_claim_token_line(later_append_earlier_clock.stdout)
    assert first is not None and second is not None
    assert first.offset < second.offset


# --------------------------------------------------------------------------
# The holder record (OMN-16729's naming requirement, kept by OMN-19262)
# --------------------------------------------------------------------------
#
# OMN-16729 required that a timeout name someone to chase. Under the fcntl
# lock the holder writes its record into the lock file once it holds the lock.
# The record is a courtesy only: whether the lock is held is the kernel's
# answer, so a record left by a dead holder blocks nobody.


def test_live_holder_is_named_in_the_timeout(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.md"
    holder = MOD.LedgerLock(ledger, timeout=1.0)
    holder.acquire()
    try:
        record = json.loads(holder.lock_file.read_text("utf-8"))
        assert record["pid"] == os.getpid()
        assert record["host"] == socket.gethostname()
        assert record["acquired_at"]

        waiter = MOD.LedgerLock(ledger, timeout=0.0)
        with pytest.raises(TimeoutError) as raised:
            waiter.acquire()
        assert f"pid={os.getpid()}" in str(raised.value)
    finally:
        holder.release()


def test_a_record_left_by_a_dead_holder_blocks_nobody(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.md"
    lock_file = MOD.lock_path_for(ledger)
    lock_file.parent.mkdir(parents=True)
    lock_file.write_text(
        json.dumps({"host": socket.gethostname(), "pid": 999_999_999}),
        encoding="utf-8",
    )
    with MOD.LedgerLock(ledger, timeout=0.0) as lock:
        assert json.loads(lock.lock_file.read_text("utf-8"))["pid"] == os.getpid()


def test_two_concurrent_acquirers_get_exactly_one_lock(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.md"
    workers = 8
    barrier = threading.Barrier(workers)
    tally = threading.Lock()
    winners: list[Any] = []
    losers: list[int] = []

    def contend() -> None:
        lock = MOD.LedgerLock(ledger, timeout=0.0)
        barrier.wait()
        try:
            lock.acquire()
        except TimeoutError:
            with tally:
                losers.append(1)
            return
        with tally:
            winners.append(lock)

    threads = [threading.Thread(target=contend) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert len(winners) == 1, f"expected exactly one holder, got {len(winners)}"
    assert len(losers) == workers - 1

    held = winners[0]
    assert held.acquired is True
    # The lock directory holds the one lock file and its ignore file.
    assert sorted(p.name for p in held.lock_file.parent.iterdir()) == sorted(
        [held.lock_file.name, ".gitignore"]
    )
    held.release()
    with MOD.LedgerLock(ledger, timeout=0.0) as after:
        assert after.acquired is True


def test_acquire_releases_the_lock_when_the_holder_record_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed acquire holds nothing, so the next writer gets straight in."""
    ledger = tmp_path / "ledger.md"

    def explode(*_args: object, **_kwargs: object) -> None:
        raise OSError("no space left on device")

    monkeypatch.setattr(MOD, "holder_record", explode)
    lock = MOD.LedgerLock(ledger, timeout=0.0)
    with pytest.raises(OSError, match="no space left"):
        lock.acquire()
    assert lock.acquired is False
    monkeypatch.undo()
    with MOD.LedgerLock(ledger, timeout=0.0) as after:
        assert after.acquired is True
