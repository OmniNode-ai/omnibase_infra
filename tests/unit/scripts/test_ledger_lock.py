# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for scripts/ledger_lock.py.

Covers the locking/append/exit-75-retry contract: atomic lock acquire/
release, same-host dead-pid and age-based stale-lock breaking, durable
append, dedup-window idempotent retry, and the CLI's exit codes (0 success,
75 lock timeout, 127 bad -- COMMAND, argparse usage errors, and the -- COMMAND
verb's own passthrough exit code).
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

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


def test_lock_root_defaults_beside_the_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(MOD.LOCK_ROOT_ENV, raising=False)
    ledger = tmp_path / "sub" / "ledger.md"
    lock = MOD.lock_path_for(ledger)
    assert lock.parent == ledger.parent / MOD.DEFAULT_LOCK_DIRNAME


def test_lock_root_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = tmp_path / "shared-locks"
    monkeypatch.setenv(MOD.LOCK_ROOT_ENV, str(override))
    ledger = tmp_path / "ledger.md"
    lock = MOD.lock_path_for(ledger)
    assert lock.parent == override


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
# stale lock breaking
# --------------------------------------------------------------------------


def _write_lock_metadata(lock_dir: Path, *, host: str, pid: int) -> None:
    lock_dir.mkdir(parents=True)
    (lock_dir / "metadata.json").write_text(
        json.dumps({"host": host, "pid": pid, "acquired_at": "x"}),
        encoding="utf-8",
    )


def test_maybe_break_stale_lock_removes_dead_same_host_lock(tmp_path: Path) -> None:
    lock_dir = tmp_path / "lock"
    # A pid that (almost certainly) does not exist.
    _write_lock_metadata(lock_dir, host=socket.gethostname(), pid=999_999_999)
    message = MOD.maybe_break_stale_lock(lock_dir, stale_after=None)
    assert message is not None
    assert "dead same-host lock" in message
    assert not lock_dir.exists()


def test_maybe_break_stale_lock_keeps_live_same_host_lock(tmp_path: Path) -> None:
    lock_dir = tmp_path / "lock"
    _write_lock_metadata(lock_dir, host=socket.gethostname(), pid=os.getpid())
    message = MOD.maybe_break_stale_lock(lock_dir, stale_after=None)
    assert message is None
    assert lock_dir.exists()


def test_maybe_break_stale_lock_keeps_other_host_lock_without_stale_after(
    tmp_path: Path,
) -> None:
    lock_dir = tmp_path / "lock"
    _write_lock_metadata(lock_dir, host="some-other-host", pid=123)
    message = MOD.maybe_break_stale_lock(lock_dir, stale_after=None)
    assert message is None
    assert lock_dir.exists()


def test_maybe_break_stale_lock_breaks_other_host_lock_by_age(tmp_path: Path) -> None:
    lock_dir = tmp_path / "lock"
    _write_lock_metadata(lock_dir, host="some-other-host", pid=123)
    message = MOD.maybe_break_stale_lock(lock_dir, stale_after=0.0)
    assert message is not None
    assert "stale lock age=" in message
    assert not lock_dir.exists()


# --------------------------------------------------------------------------
# LedgerLock acquire/release + append_text
# --------------------------------------------------------------------------


def test_ledger_lock_acquire_release_roundtrip(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.md"
    lock = MOD.LedgerLock(ledger, timeout=1.0, stale_after=None, command=None)
    with lock:
        assert lock.acquired is True
        assert lock.lock_dir.exists()
    assert lock.acquired is False
    assert not lock.lock_dir.exists()


def test_ledger_lock_times_out_when_already_held(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.md"
    holder = MOD.LedgerLock(ledger, timeout=1.0, stale_after=None, command=None)
    holder.acquire()
    try:
        waiter = MOD.LedgerLock(ledger, timeout=0.0, stale_after=None, command=None)
        with pytest.raises(TimeoutError):
            waiter.acquire()
    finally:
        holder.release()


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
    result = _run_cli([str(ledger), "--append", "- 2026-08-25T00:00:00Z row one"])
    assert result.returncode == 0, result.stderr
    assert ledger.read_text(encoding="utf-8") == "- 2026-08-25T00:00:00Z row one\n"


def test_cli_append_retry_is_deduped(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.md"
    first = _run_cli([str(ledger), "--append", "- 2026-08-25T00:00:00Z row one"])
    assert first.returncode == 0, first.stderr
    # Simulate a retry after an exit-75 timeout: same tag/body, a few
    # seconds' worth of timestamp drift.
    second = _run_cli([str(ledger), "--append", "- 2026-08-25T00:00:05Z row one"])
    assert second.returncode == 0, second.stderr
    assert "DEDUP" in second.stderr
    # Only one copy landed.
    assert ledger.read_text(encoding="utf-8").count("row one") == 1


def test_cli_exit_75_on_lock_timeout(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.md"
    lock_dir = MOD.lock_path_for(ledger.resolve())
    lock_dir.mkdir(parents=True)
    (lock_dir / "metadata.json").write_text(
        json.dumps({"host": "some-other-host", "pid": 123, "acquired_at": "x"}),
        encoding="utf-8",
    )
    try:
        result = _run_cli(
            [str(ledger), "--timeout", "1s", "--append", "should not land"]
        )
        assert result.returncode == 75
        assert "timed out" in result.stderr
        assert not ledger.exists()
    finally:
        import shutil

        shutil.rmtree(lock_dir, ignore_errors=True)


def test_cli_stale_same_host_lock_is_broken_automatically(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.md"
    lock_dir = MOD.lock_path_for(ledger.resolve())
    lock_dir.mkdir(parents=True)
    (lock_dir / "metadata.json").write_text(
        json.dumps(
            {"host": socket.gethostname(), "pid": 999_999_999, "acquired_at": "x"}
        ),
        encoding="utf-8",
    )
    result = _run_cli(
        [str(ledger), "--timeout", "3s", "--append", "- after stale break"]
    )
    assert result.returncode == 0, result.stderr
    assert "removed dead same-host lock" in result.stderr
    assert ledger.read_text(encoding="utf-8") == "- after stale break\n"


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

_PIPE_CLAIM = (
    "| {ts} | build-OMN-16400 | OMN-16400 | CLAIM | "
    "Claiming the ledger hardening work. |"
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

    result = _run_cli([str(ledger), "--append", row])

    assert result.returncode == 0
    token = MOD.parse_claim_token_line(result.stdout)
    assert token is not None
    # The offset is the lock-protected byte position the row landed at --
    # i.e. the authoritative append-order signal, not a self-declared clock.
    assert token.offset == len("existing header line\n")


def test_cli_append_of_a_non_claim_row_emits_no_token(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.md"
    result = _run_cli(
        [str(ledger), "--append", "| 2026-08-28T18:05:00Z | h | OMN-1 | TERMINAL | x |"]
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
        [str(ledger), "--append", _PIPE_CLAIM.format(ts="2026-08-28T18:10:00Z")]
    )
    retry = _run_cli(
        [str(ledger), "--append", _PIPE_CLAIM.format(ts="2026-08-28T18:12:31Z")]
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
        [str(ledger), "--append", _PIPE_CLAIM.format(ts="2026-08-28T18:10:00Z")]
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
        [str(ledger), "--append", _PIPE_CLAIM.format(ts="2026-08-28T18:10:00Z")]
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
        [str(ledger), "--append", _PIPE_CLAIM.format(ts="2026-08-28T18:10:00Z")]
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
            "| 2026-08-22T14:45:00Z | lane-a | OMN-16385 | CLAIM | first appended |",
        ]
    )
    later_append_earlier_clock = _run_cli(
        [
            str(ledger),
            "--append",
            "| 2026-08-22T14:20:00Z | lane-b | OMN-16386 | CLAIM | second appended |",
        ]
    )
    first = MOD.parse_claim_token_line(earlier_append_later_clock.stdout)
    second = MOD.parse_claim_token_line(later_append_earlier_clock.stdout)
    assert first is not None and second is not None
    assert first.offset < second.offset
