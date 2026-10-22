# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for scripts/ci/check_hostile_review_threads.py (OMN-17492, OMN-17427).

The thread gate is the DETERMINISTIC merge surface for hostile-review
findings. These tests pin its rule: every hostile-reviewer thread needs a
disposition reply (``fixed-in <sha>`` or ``rejected: <reason>``); a thread
resolved without one blocks, an outdated open thread blocks, a ``fixed-in``
commit must belong to the PR and touch the cited lines, and a rejection
clears only when the review bot's verdict, posted under the workflow
identity, accepts it. Other actors' threads are never this gate's business,
and a PR with no threads (the reviewer was skipped or down) passes.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

pytestmark = pytest.mark.unit

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "ci"
    / "check_hostile_review_threads.py"
)
_spec = importlib.util.spec_from_file_location("check_hostile_review_threads", _SCRIPT)
assert _spec is not None and _spec.loader is not None
gate = importlib.util.module_from_spec(_spec)
sys.modules["check_hostile_review_threads"] = gate
_spec.loader.exec_module(gate)

REJ = "rejected: the caller already validates this input"

_FIRST_BODY = (
    f"<!-- {gate.MARKER} fp=abc123def456 -->\n"
    "**[MAJOR] hostile-reviewer (sonnet)**\n\nthe finding"
)


REPLY_ID_BASE = 100


def _reply(i: int, body: str) -> dict[str, Any]:
    return {
        "databaseId": REPLY_ID_BASE + i,
        "body": body,
        "url": f"https://x/reply{i}",
        "author": {"login": "someone"},
        "authorAssociation": "MEMBER",
    }


def _verdict(
    rejection_index: int,
    rejection_body: str,
    verdict: str = "accepted",
    *,
    author: str = "github-actions",
    sentence: str = "the guard on line 9 already rejects that input",
) -> dict[str, Any]:
    """A judge reply bound to the rejection at ``replies[rejection_index]``."""
    return {
        "databaseId": 900 + rejection_index,
        "body": (
            f"<!-- {gate.JUDGE_MARKER} rejection={REPLY_ID_BASE + rejection_index} "
            f"digest={gate.body_digest(rejection_body)} verdict={verdict} "
            "judge=rejection-judge identity=anthropic#id=claude-opus-5-5 -->\n"
            f"**Review bot: rejection {verdict}**\n\n> {sentence}\n\ntail"
        ),
        "url": f"https://x/verdict{rejection_index}",
        "author": {"login": author},
        "authorAssociation": "NONE",
    }


def _thread(
    *,
    resolved: bool = False,
    outdated: bool = False,
    marker: bool = True,
    path: str = "src/foo.py",
    replies: tuple[str | dict[str, Any], ...] = (),
    line: int | None = 10,
    start_line: int | None = None,
    original_commit: str = "",
) -> dict[str, Any]:
    first: dict[str, Any] = {
        "databaseId": 1,
        "body": _FIRST_BODY if marker else "human note",
        "url": "https://x/thread",
        "author": {"login": "github-actions"},
        "originalCommit": {"oid": original_commit} if original_commit else None,
    }
    nodes = [first] + [
        body if isinstance(body, dict) else _reply(i, body)
        for i, body in enumerate(replies)
    ]
    return {
        "id": "T1",
        "isResolved": resolved,
        "isOutdated": outdated,
        "path": path,
        "originalLine": line,
        "originalStartLine": start_line,
        "subjectType": "LINE" if line is not None else "FILE",
        "resolvedBy": {"login": "resolver"} if resolved else None,
        "comments": {"nodes": nodes},
    }


def _never(sha: str, cited: Any) -> tuple[bool, str]:
    raise AssertionError("the verifier must not be called")


def _accept(sha: str, cited: Any) -> tuple[bool, str]:
    return True, "touched"


def _refuse(sha: str, cited: Any) -> tuple[bool, str]:
    return False, "does not touch the cited lines"


class TestClassifyThreads:
    def test_unresolved_marker_thread_blocks(self) -> None:
        blocking, cleared = gate.classify_threads([_thread()], _never)
        assert len(blocking) == 1
        assert cleared == []

    def test_resolved_thread_without_a_disposition_blocks(self) -> None:
        # OMN-17427: before this, a resolved thread was ignored and nothing
        # recorded what was done about the finding (omnibase_infra#5201).
        blocking, cleared = gate.classify_threads([_thread(resolved=True)], _never)
        assert len(blocking) == 1
        assert blocking[0].reason == "resolved with no disposition reply"
        assert blocking[0].resolved_by == "resolver"
        assert cleared == []

    def test_resolved_thread_with_a_non_disposition_reply_blocks(self) -> None:
        blocking, _ = gate.classify_threads(
            [_thread(resolved=True, replies=("done, thanks",))], _never
        )
        assert len(blocking) == 1

    def test_outdated_unresolved_thread_blocks(self) -> None:
        # Before OMN-17427 this was reported and passed, so a finding whose
        # line moved under a later push escaped with no record.
        blocking, cleared = gate.classify_threads([_thread(outdated=True)], _never)
        assert len(blocking) == 1
        assert cleared == []

    def test_a_rejection_alone_no_longer_clears(self) -> None:
        # OMN-17427: every agent replies through the operator's account, so a
        # rejection proves nothing until the review bot accepts it.
        blocking, cleared = gate.classify_threads(
            [_thread(resolved=True, replies=(REJ,))], _never
        )
        assert cleared == []
        assert blocking[0].disposition == "rejected"
        assert blocking[0].judge_verdict == "unjudged"
        assert "the review bot has not judged it" in blocking[0].reason

    def test_a_rejection_the_review_bot_accepted_clears(self) -> None:
        blocking, cleared = gate.classify_threads(
            [_thread(resolved=True, replies=(REJ, _verdict(0, REJ)))], _never
        )
        assert blocking == []
        assert cleared[0].disposition == "rejected"
        assert cleared[0].disposition_author == "someone"
        assert cleared[0].judge_verdict == "accepted"
        assert cleared[0].judge_voter == "rejection-judge"
        assert cleared[0].judge_identity == "anthropic#id=claude-opus-5-5"
        assert cleared[0].judge_sentence == (
            "the guard on line 9 already rejects that input"
        )

    def test_rejected_without_a_real_reason_does_not_count(self) -> None:
        blocking, _ = gate.classify_threads(
            [_thread(resolved=True, replies=("rejected: no",))], _never
        )
        assert len(blocking) == 1

    def test_open_thread_with_a_valid_disposition_is_cleared(self) -> None:
        # The reply is the record; the workflow token cannot resolve a thread
        # (resolveReviewThread is FORBIDDEN for it), so an open thread with an
        # accepted disposition clears as it is.
        rejection = "rejected: false positive, see the guard above"
        blocking, cleared = gate.classify_threads(
            [_thread(replies=(rejection, _verdict(0, rejection)))],
            _never,
        )
        assert blocking == []
        assert not cleared[0].resolved

    def test_fixed_in_is_verified(self) -> None:
        seen: list[tuple[str, Any]] = []

        def verifier(sha: str, cited: Any) -> tuple[bool, str]:
            seen.append((sha, cited))
            return True, "touched"

        _, cleared = gate.classify_threads(
            [
                _thread(
                    resolved=True,
                    replies=("Fixed-in ABCDEF1234",),
                    line=12,
                    start_line=9,
                    original_commit="0" * 40,
                )
            ],
            verifier,
        )
        assert cleared[0].fix_verified is True
        sha, cited = seen[0]
        assert sha == "abcdef1234"
        assert (cited.path, cited.start, cited.end) == ("src/foo.py", 9, 12)
        assert cited.original_commit == "0" * 40

    def test_fixed_in_that_did_not_touch_the_lines_blocks(self) -> None:
        blocking, _ = gate.classify_threads(
            [_thread(resolved=True, replies=("fixed-in abcdef1",))], _refuse
        )
        assert blocking[0].fix_verified is False
        assert "does not touch" in blocking[0].reason

    def test_latest_disposition_wins(self) -> None:
        blocking, cleared = gate.classify_threads(
            [
                _thread(
                    resolved=True,
                    replies=(
                        "fixed-in abcdef1",
                        "rejected: on reflection the model misread the loop bound",
                        _verdict(
                            1,
                            "rejected: on reflection the model misread the loop bound",
                        ),
                    ),
                )
            ],
            _never,
        )
        assert blocking == []
        assert cleared[0].disposition == "rejected"

    def test_file_level_thread_cites_no_lines(self) -> None:
        cited = gate.cited_lines(_thread(line=None))
        assert (cited.start, cited.end) == (None, None)

    def test_non_marker_threads_are_none_of_this_gates_business(self) -> None:
        blocking, cleared = gate.classify_threads(
            [_thread(marker=False), _thread(marker=False, resolved=True)], _never
        )
        assert blocking == []
        assert cleared == []

    def test_thread_with_no_comments_is_ignored(self) -> None:
        thread: dict[str, Any] = {
            "isResolved": False,
            "isOutdated": False,
            "path": "x",
            "comments": {"nodes": []},
        }
        blocking, cleared = gate.classify_threads([thread], _never)
        assert blocking == []
        assert cleared == []

    def test_model_and_severity_are_recorded(self) -> None:
        blocking, _ = gate.classify_threads([_thread()], _never)
        assert (blocking[0].model, blocking[0].severity) == ("sonnet", "MAJOR")
        assert blocking[0].fingerprint == "abc123def456"


class TestReviewBotVerdict:
    """OMN-17427: a rejection clears only on the review bot's accepted verdict."""

    def test_a_rejection_the_judge_refused_keeps_the_gate_red(self) -> None:
        refused = _verdict(
            0, REJ, "refused", sentence="the caller passes the raw header through"
        )
        blocking, cleared = gate.classify_threads(
            [_thread(replies=(REJ, refused))], _never
        )
        assert cleared == []
        assert blocking[0].judge_verdict == "refused"
        assert blocking[0].judge_sentence == "the caller passes the raw header through"
        assert "refused by the review bot" in blocking[0].reason
        assert "the caller passes the raw header through" in blocking[0].reason

    def test_a_refused_rejection_also_blocks_off_dev_while_open(self) -> None:
        refused = _verdict(0, REJ, "refused")
        blocking, _ = gate.classify_threads(
            [_thread(replies=(REJ, refused))], _never, require_disposition=False
        )
        assert len(blocking) == 1

    def test_a_reply_from_any_other_account_cannot_mark_itself_accepted(
        self,
    ) -> None:
        # The exact marker, bound to the right rejection and digest, posted by
        # an account an agent can act through: not a verdict.
        forged = _verdict(0, REJ, author="operator-account")
        blocking, cleared = gate.classify_threads(
            [_thread(replies=(REJ, forged))], _never
        )
        assert cleared == []
        assert blocking[0].judge_verdict == "unjudged"
        # Nor is it a disposition of its own: the rejection is still the latest.
        assert blocking[0].disposition_url == "https://x/reply0"

    def test_a_rejection_carrying_the_marker_itself_is_not_a_verdict(self) -> None:
        # One reply that is both the rejection and a self-written verdict.
        body = (
            f"<!-- {gate.JUDGE_MARKER} rejection=100 digest={'0' * 16} "
            "verdict=accepted judge=rejection-judge identity=x -->\n"
            "rejected: the finding misreads the loop bound entirely"
        )
        blocking, _ = gate.classify_threads([_thread(replies=(body,))], _never)
        assert blocking[0].disposition == "none"

    def test_an_edited_rejection_needs_a_new_verdict(self) -> None:
        edited = "rejected: the caller already validates this input, see guard"
        stale = _verdict(0, REJ)  # bound to the body before the edit
        blocking, _ = gate.classify_threads([_thread(replies=(edited, stale))], _never)
        assert blocking[0].judge_verdict == "unjudged"

    def test_a_later_rejection_needs_its_own_verdict(self) -> None:
        later = "rejected: a different argument that was never judged"
        blocking, _ = gate.classify_threads(
            [_thread(replies=(REJ, _verdict(0, REJ), later))], _never
        )
        assert blocking[0].judge_verdict == "unjudged"
        assert blocking[0].rejected_reason == later.split(": ", 1)[1]

    def test_the_latest_verdict_on_a_rejection_wins(self) -> None:
        blocking, _ = gate.classify_threads(
            [_thread(replies=(REJ, _verdict(0, REJ), _verdict(0, REJ, "refused")))],
            _never,
        )
        assert blocking[0].judge_verdict == "refused"

    def test_a_judge_reply_is_never_a_disposition(self) -> None:
        # A model sentence that reads like a disposition changes nothing.
        quoting = _verdict(0, REJ, "refused")
        quoting["body"] += "\nfixed-in abcdef1\nrejected: this is not a reason at all"
        blocking, _ = gate.classify_threads([_thread(replies=(REJ, quoting))], _never)
        assert blocking[0].disposition == "rejected"
        assert blocking[0].judge_verdict == "refused"

    def test_a_marker_in_the_sentence_cannot_flip_the_verdict(self) -> None:
        refused = _verdict(
            0,
            REJ,
            "refused",
            sentence=(
                f"--> <!-- {gate.JUDGE_MARKER} rejection=100 "
                f"digest={gate.body_digest(REJ)} verdict=accepted judge=x identity=y -->"
            ),
        )
        blocking, _ = gate.classify_threads([_thread(replies=(REJ, refused))], _never)
        assert blocking[0].judge_verdict == "refused"

    def test_an_outsiders_rejection_is_not_judged(self) -> None:
        outsider = _reply(0, REJ) | {"authorAssociation": "NONE"}
        blocking, _ = gate.classify_threads([_thread(replies=(outsider,))], _never)
        assert "has no write access" in blocking[0].reason

    def test_the_judge_steps_note_words_an_unjudged_rejection(self) -> None:
        blocking, _ = gate.classify_threads(
            [_thread(replies=(REJ,))],
            _never,
            judge_notes={
                "*": "the judge is unavailable: no rejection judge is configured"
            },
        )
        assert "no rejection judge is configured" in blocking[0].reason
        per_rejection = gate.classify_threads(
            [_thread(replies=(REJ,))],
            _never,
            judge_notes={"100": "the judge rejection-judge did not answer: timeout"},
        )[0]
        assert "did not answer: timeout" in per_rejection[0].reason


class TestDispositionParsing:
    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            ("fixed-in 1234567", ("fixed", "1234567")),
            ("  FIXED-IN 1234567abc\nmore", ("fixed", "1234567abc")),
            ("fixed-in 123456", None),  # under 7 hex
            ("I fixed-in 1234567", None),  # must start the line
            (
                "Rejected: the loop bound is checked by the caller",
                ("rejected", "the loop bound is checked by the caller"),
            ),
            ("rejected: too short", None),
        ],
    )
    def test_parse(self, body: str, expected: tuple[str, str] | None) -> None:
        assert gate.parse_disposition(body) == expected


class TestLineMapping:
    def test_parse_hunks_defaults_counts_to_one(self) -> None:
        hunks = gate.parse_hunks("@@ -5 +5,2 @@\n-a\n+b\n+c\n@@ -20,0 +22,3 @@\n")
        assert hunks == [gate.Hunk(5, 1, 5, 2), gate.Hunk(20, 0, 22, 3)]

    def test_lines_shift_past_an_insertion_above(self) -> None:
        hunks = [gate.Hunk(2, 0, 3, 4)]  # four lines inserted after line 2
        assert gate.map_range(hunks, 10, 12) == (14, 16)

    def test_lines_inside_a_changed_hunk_map_to_its_span(self) -> None:
        hunks = [gate.Hunk(10, 2, 10, 5)]
        assert gate.map_range(hunks, 11, 11) == (10, 14)

    def test_touched_by_an_overlapping_change(self) -> None:
        assert gate.range_touched([gate.Hunk(8, 3, 8, 1)], 10, 12)

    def test_not_touched_by_a_change_elsewhere(self) -> None:
        assert not gate.range_touched([gate.Hunk(30, 2, 30, 2)], 10, 12)

    def test_touched_by_an_insertion_inside_the_range(self) -> None:
        assert gate.range_touched([gate.Hunk(11, 0, 12, 2)], 10, 12)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env=scrub_git_location_env(os.environ),
    ).stdout.strip()


@pytest.fixture
def pr_repo(tmp_path: Path) -> dict[str, Any]:
    """A base commit, a reviewed PR commit, an unrelated commit and a fix."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "dev")
    _git(repo, "config", "user.email", "ci@example.invalid")
    _git(repo, "config", "user.name", "ci")
    target = repo / "src.py"
    target.write_text("".join(f"line {i}\n" for i in range(1, 31)))
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "-b", "pr")
    lines = target.read_text().splitlines(keepends=True)
    lines[19] = "line 20 has the bug\n"
    target.write_text("".join(lines))
    _git(repo, "commit", "-q", "-am", "reviewed")
    reviewed = _git(repo, "rev-parse", "HEAD")
    # Three lines inserted at the top shift the cited line 20 to line 23.
    target.write_text("new 1\nnew 2\nnew 3\n" + target.read_text())
    _git(repo, "commit", "-q", "-am", "unrelated")
    unrelated = _git(repo, "rev-parse", "HEAD")
    lines = target.read_text().splitlines(keepends=True)
    lines[22] = "line 20 fixed\n"
    target.write_text("".join(lines))
    _git(repo, "commit", "-q", "-am", "fix")
    fix = _git(repo, "rev-parse", "HEAD")
    return {
        "repo": repo,
        "base": base,
        "reviewed": reviewed,
        "unrelated": unrelated,
        "fix": fix,
    }


class TestGitFixVerifier:
    def _verifier(self, pr: dict[str, Any]) -> Any:
        return gate.GitFixVerifier(pr["repo"], pr["fix"], pr["base"])

    def _cited(self, pr: dict[str, Any]) -> Any:
        return gate.CitedLines("src.py", pr["reviewed"], 20, 20)

    def test_the_fix_commit_touches_the_carried_forward_line(
        self, pr_repo: dict[str, Any]
    ) -> None:
        ok, why = self._verifier(pr_repo)(pr_repo["fix"][:10], self._cited(pr_repo))
        assert ok, why
        assert "src.py:23-23" in why

    def test_a_commit_of_the_pr_that_misses_the_lines_is_refused(
        self, pr_repo: dict[str, Any]
    ) -> None:
        ok, why = self._verifier(pr_repo)(pr_repo["unrelated"], self._cited(pr_repo))
        assert not ok
        assert "does not touch the cited lines" in why

    def test_a_base_branch_commit_is_refused(self, pr_repo: dict[str, Any]) -> None:
        ok, why = self._verifier(pr_repo)(pr_repo["base"], self._cited(pr_repo))
        assert not ok
        assert "already on the base branch" in why

    def test_an_unknown_commit_is_refused(self, pr_repo: dict[str, Any]) -> None:
        ok, why = self._verifier(pr_repo)("deadbeef1", self._cited(pr_repo))
        assert not ok
        assert "not in this repository" in why

    def test_a_file_level_thread_needs_the_file_changed(
        self, pr_repo: dict[str, Any]
    ) -> None:
        cited = gate.CitedLines("src.py", pr_repo["reviewed"], None, None)
        assert self._verifier(pr_repo)(pr_repo["unrelated"], cited)[0]
        other = gate.CitedLines("other.py", pr_repo["reviewed"], None, None)
        assert not self._verifier(pr_repo)(pr_repo["unrelated"], other)[0]


class TestRunGate:
    def _env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
        monkeypatch.setenv("GITHUB_TOKEN", "t")
        monkeypatch.setenv("REPO", "OmniNode-ai/omnibase_infra")
        monkeypatch.setenv("PR_NUMBER", "1")
        monkeypatch.setenv("HEAD_SHA", "h" * 40)
        record = tmp_path / "records.jsonl"
        monkeypatch.setenv("DISPOSITION_RECORD_PATH", str(record))
        monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
        monkeypatch.delenv("JUDGE_STATUS_PATH", raising=False)
        return record

    def test_no_threads_passes_so_a_reviewer_outage_never_blocks(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # The reviewer was skipped, degraded or down: it posted no thread.
        self._env(monkeypatch, tmp_path)
        monkeypatch.setattr(gate, "fetch_review_threads", lambda *a: [])
        assert gate.main([]) == 0

    def test_fetch_failure_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._env(monkeypatch, tmp_path)

        def boom(*a: Any) -> list[dict[str, Any]]:
            raise RuntimeError("unreachable")

        monkeypatch.setattr(gate, "fetch_review_threads", boom)
        assert gate.main([]) == 1

    def test_a_valid_open_disposition_passes_and_is_recorded(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        record = self._env(monkeypatch, tmp_path)
        rejection = "rejected: the reviewer misread the retry loop"
        thread = _thread(replies=(rejection, _verdict(0, rejection)))
        monkeypatch.setattr(gate, "fetch_review_threads", lambda *a: [thread])
        assert gate.main([]) == 0
        row = json.loads(record.read_text().splitlines()[0])
        assert row["disposition"] == "rejected"
        assert row["judge_verdict"] == "accepted"
        assert row["judge_identity"] == "anthropic#id=claude-opus-5-5"
        assert row["resolved"] is False
        assert row["model"] == "sonnet"
        assert row["finding"] == "the finding"

    def test_a_resolved_thread_without_a_reply_fails_the_gate(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._env(monkeypatch, tmp_path)
        monkeypatch.setattr(
            gate, "fetch_review_threads", lambda *a: [_thread(resolved=True)]
        )
        assert gate.main([]) == 1

    def test_judge_unavailable_fails_closed_with_the_reason(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._env(monkeypatch, tmp_path)
        status = tmp_path / "judge-status.json"
        status.write_text(
            json.dumps({"unavailable": "no rejection judge is configured", "notes": {}})
        )
        monkeypatch.setenv("JUDGE_STATUS_PATH", str(status))
        monkeypatch.setattr(
            gate, "fetch_review_threads", lambda *a: [_thread(replies=(REJ,))]
        )
        assert gate.main([]) == 1
        out = capsys.readouterr().out
        assert "the judge is unavailable: no rejection judge is configured" in out

    def test_no_threads_passes_even_when_the_judge_is_unavailable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # A PR the reviewer never ran on has nothing to judge: it passes.
        self._env(monkeypatch, tmp_path)
        status = tmp_path / "judge-status.json"
        status.write_text(json.dumps({"unavailable": "no judge", "notes": {}}))
        monkeypatch.setenv("JUDGE_STATUS_PATH", str(status))
        monkeypatch.setattr(gate, "fetch_review_threads", lambda *a: [])
        assert gate.main([]) == 0

    def test_an_unreadable_judge_status_never_reads_as_a_pass(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._env(monkeypatch, tmp_path)
        status = tmp_path / "judge-status.json"
        status.write_text("{not json")
        monkeypatch.setenv("JUDGE_STATUS_PATH", str(status))
        monkeypatch.setattr(
            gate, "fetch_review_threads", lambda *a: [_thread(replies=(REJ,))]
        )
        assert gate.main([]) == 1


class TestAccuracyReport:
    def test_counts_per_model_from_records_and_job_logs(self, tmp_path: Path) -> None:
        rows = [
            {
                "repo": "r",
                "pr": 1,
                "fingerprint": "a",
                "model": "sonnet",
                "disposition": "fixed",
                "fix_verified": True,
            },
            {
                "repo": "r",
                "pr": 1,
                "fingerprint": "b",
                "model": "sonnet",
                "disposition": "rejected",
                "judge_verdict": "accepted",
            },
            {
                "repo": "r",
                "pr": 1,
                "fingerprint": "e",
                "model": "sonnet",
                "disposition": "rejected",
                "judge_verdict": "refused",
            },
            {
                "repo": "r",
                "pr": 1,
                "fingerprint": "f",
                "model": "sonnet",
                "disposition": "rejected",
            },
            {
                "repo": "r",
                "pr": 2,
                "fingerprint": "c",
                "model": "correctness",
                "disposition": "none",
            },
            {
                "repo": "r",
                "pr": 2,
                "fingerprint": "d",
                "model": "correctness",
                "disposition": "fixed",
                "fix_verified": False,
            },
        ]
        records = tmp_path / "records.jsonl"
        records.write_text("".join(json.dumps(r) + "\n" for r in rows[:4]))
        log = tmp_path / "job.log"
        log.write_text(
            "".join(
                f"2026-10-21T13:00:00Z {gate.DISPOSITION_LINE_PREFIX}{json.dumps(r)}\n"
                for r in rows[4:]
            )
            + "2026-10-21T13:00:01Z unrelated line\n"
        )
        table = gate.accuracy(gate._read_records([records, log]))
        # OMN-17427: a bot-accepted rejection is counted apart from a fix, and
        # a refused or unjudged rejection is neither.
        assert table["sonnet"] == {
            "findings": 4,
            "fixed": 1,
            "rejection_accepted": 1,
            "rejection_refused": 1,
            "rejection_unjudged": 1,
            "none": 0,
        }
        assert table["correctness"] == {
            "findings": 2,
            "fixed": 0,
            "rejection_accepted": 0,
            "rejection_refused": 0,
            "rejection_unjudged": 0,
            "none": 2,
        }

    def test_a_rerun_does_not_double_count(self) -> None:
        row = {
            "repo": "r",
            "pr": 1,
            "fingerprint": "a",
            "model": "m",
            "disposition": "none",
        }
        later = dict(row, disposition="rejected", judge_verdict="accepted")
        assert gate.accuracy([row, later])["m"] == {
            "findings": 1,
            "fixed": 0,
            "rejection_accepted": 1,
            "rejection_refused": 0,
            "rejection_unjudged": 0,
            "none": 0,
        }


class TestOffDevBase:
    """PRs into main keep the earlier rule; the disposition rule is dev's."""

    def test_resolved_without_a_reply_does_not_block_off_dev(self) -> None:
        blocking, _ = gate.classify_threads(
            [_thread(resolved=True)], _never, require_disposition=False
        )
        assert blocking == []

    def test_outdated_open_thread_does_not_block_off_dev(self) -> None:
        blocking, cleared = gate.classify_threads(
            [_thread(outdated=True)], _never, require_disposition=False
        )
        assert blocking == []
        assert cleared[0].disposition == "none"

    def test_open_current_thread_still_blocks_off_dev(self) -> None:
        blocking, _ = gate.classify_threads(
            [_thread()], _never, require_disposition=False
        )
        assert len(blocking) == 1

    def test_a_refused_fix_on_an_outdated_thread_does_not_block_off_dev(self) -> None:
        blocking, cleared = gate.classify_threads(
            [_thread(outdated=True, replies=("fixed-in abcdef1",))],
            _refuse,
            require_disposition=False,
        )
        assert blocking == []
        assert cleared[0].fix_verified is False
