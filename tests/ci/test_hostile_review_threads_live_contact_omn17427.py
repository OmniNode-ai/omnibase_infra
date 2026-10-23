# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17427: recorded GitHub evidence for requiring the Hostile Review Thread Gate.

Two recordings, both read through ``onex scm`` on the dev bus:

1. GitHub's 422 to the poster's create-review call on omnibase_infra#5154.
   Every comment in that call's ``comments`` array carried ``subject_type``
   or no ``line``, GitHub refused the whole review, and the poster demoted
   every finding to the review body. Over the 15 reviewed dev PRs merged
   2026-10-19..21, 22 of 60 reviewer runs hit this (75 findings at quorum
   lost to the body) and no other run posted a thread, so not one review
   thread was created and the thread gate passed on every run with nothing
   to read. Requiring that gate without
   this fix would have required a check that could not fail.
2. The thread gate's check-runs on the 20 latest merged dev heads: the
   admission record for registering it in CI Summary's
   EXPECTED_EXTERNAL_CONTEXTS (present and green on every one, including the
   bot PRs the reviewer skips), so requiring it cannot wedge dev.
"""

from __future__ import annotations

import email.message
import importlib.util
import io
import json
import sys
import urllib.error
from pathlib import Path
from typing import Any, cast

import pytest

from scripts.ci.ci_summary_gate import (
    DEV_BASE_ONLY_CONTEXTS,
    EXPECTED_EXTERNAL_CONTEXTS,
    evaluate_external_contexts,
    scoped_to_base,
)

pytestmark = pytest.mark.unit

_POSTER = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "ci"
    / "post_hostile_review_threads.py"
)
_spec = importlib.util.spec_from_file_location("post_hostile_review_threads", _POSTER)
assert _spec is not None and _spec.loader is not None
poster = importlib.util.module_from_spec(_spec)
sys.modules["post_hostile_review_threads"] = poster
_spec.loader.exec_module(poster)

_GATE = _POSTER.parent / "check_hostile_review_threads.py"
_gate_spec = importlib.util.spec_from_file_location(
    "check_hostile_review_threads", _GATE
)
assert _gate_spec is not None and _gate_spec.loader is not None
gate = importlib.util.module_from_spec(_gate_spec)
sys.modules["check_hostile_review_threads"] = gate
_gate_spec.loader.exec_module(gate)

THREAD_GATE = "Hostile Review Thread Gate"


@pytest.mark.live_contact(
    "tests/fixtures/omn17427/infra-pr5154-reviewer-job-118859725505-review-post-422.json"
)
def test_the_review_call_no_longer_carries_what_github_refused(
    recorded_response: dict[str, object],
) -> None:
    response = cast("dict[str, Any]", recorded_response["response"])
    lines = cast("list[str]", response["log_lines"])
    refusal = next(line for line in lines if "review POST failed (422)" in line)
    # GitHub's own words: the create-review comments array has no subjectType,
    # and a comment there needs a position (a line).
    assert (
        "subjectType (Field is not defined on DraftPullRequestReviewComment)" in refusal
    )
    assert "position (Expected value to not be null)" in refusal
    # The poster then claimed threads it never made.
    assert any(
        "Posted hostile review (REQUEST_CHANGES) with 4 thread" in line
        for line in lines
    )

    sent: list[tuple[str, dict[str, Any]]] = []

    def github(method: str, url: str, token: str, body: dict[str, Any]) -> None:
        """Refuse a review exactly where the recorded GitHub response did."""
        sent.append((url, body))
        if url.endswith("/reviews") and any(
            "subject_type" in c or "line" not in c for c in body["comments"]
        ):
            raise urllib.error.HTTPError(
                url,
                422,
                "Unprocessable Entity",
                email.message.Message(),
                io.BytesIO(b"{}"),
            )

    payload = {
        "event": "REQUEST_CHANGES",
        "body": "summary",
        "comments": [
            {"path": "src/a.py", "body": "file finding", "subject_type": "file"},
            {"path": "src/b.py", "body": "line finding", "line": 7, "side": "RIGHT"},
        ],
    }
    code = poster.publish(
        payload,
        base="https://api.example.invalid/repos/o/r",
        pr_number=5154,
        token="t",
        head_sha="e" * 40,
        api=github,
    )
    assert code == 0
    urls = [url for url, _ in sent]
    # One accepted review (the line thread), one file thread, no demotion.
    assert urls == [
        "https://api.example.invalid/repos/o/r/pulls/5154/reviews",
        "https://api.example.invalid/repos/o/r/pulls/5154/comments",
    ]


@pytest.mark.live_contact(
    "tests/fixtures/omn17427/infra-dev-merged-20-thread-gate-check-runs-20261021.json"
)
def test_every_recent_merged_dev_head_carried_a_green_thread_gate(
    recorded_response: dict[str, object],
) -> None:
    heads = cast(
        "list[dict[str, Any]]",
        cast("dict[str, Any]", recorded_response["response"])["heads"],
    )
    assert len(heads) == 20
    assert frozenset({THREAD_GATE}) == DEV_BASE_ONLY_CONTEXTS
    dev_contexts = scoped_to_base(EXPECTED_EXTERNAL_CONTEXTS, "dev")
    assert THREAD_GATE in dev_contexts
    for head in heads:
        runs = head["thread_gate_check_runs"]
        assert runs, f"#{head['pr']} has no {THREAD_GATE} check-run"
        rows = [
            {
                "id": run["id"],
                "name": THREAD_GATE,
                "status": "completed",
                "conclusion": run["conclusion"],
                "completed_at": run["completed_at"],
                "head_sha": head["head_sha"],
            }
            for run in runs
        ]
        failures, unresolved = evaluate_external_contexts(rows, (THREAD_GATE,))
        assert (failures, unresolved) == ([], []), head["pr"]
    skipped_reviewer = [
        h["pr"]
        for h in heads
        if all(r["conclusion"] == "skipped" for r in h["reviewer_check_runs"])
    ]
    # The bot PRs: the reviewer skipped, the thread gate still ran and passed.
    assert sorted(skipped_reviewer) == [5184, 5198, 5203, 5209, 5219]


@pytest.mark.live_contact(
    "tests/fixtures/omn17427/infra-pr5234-run-39621551864-first-threads-and-gate.json"
)
def test_the_fixed_poster_made_threads_and_the_gate_held_them(
    recorded_response: dict[str, object],
) -> None:
    """The first live run of this change, on its own PR, read back from the job logs."""
    response = cast("dict[str, Any]", recorded_response["response"])
    reviewer = cast("list[str]", response["reviewer_log_lines"])
    assert any('"posted_threads": 2' in line for line in reviewer)
    assert any("with 2 finding thread(s); 0 demoted" in line for line in reviewer)

    gate_lines = cast("list[str]", response["thread_gate_log_lines"])
    rows = [
        json.loads(line.split(gate.DISPOSITION_LINE_PREFIX, 1)[1])
        for line in gate_lines
        if gate.DISPOSITION_LINE_PREFIX in line
    ]
    assert [row["reason"] for row in rows] == ["unresolved, no disposition reply"] * 2
    assert all(row["blocking"] for row in rows)
    # Accuracy is read per voter model from exactly these lines.
    assert gate.accuracy(rows) == {
        "correctness": {"findings": 1, "fixed": 0, "rejected": 0, "none": 1},
        "sonnet": {"findings": 1, "fixed": 0, "rejected": 0, "none": 1},
    }


@pytest.mark.live_contact(
    "tests/fixtures/omn17427/infra-pr5234-job-119316797666-resolve-thread-forbidden.json"
)
def test_the_gate_does_not_depend_on_resolving_a_thread(
    recorded_response: dict[str, object],
) -> None:
    """GitHub refuses resolveReviewThread to the workflow token, so the reply is the record."""
    response = cast("dict[str, Any]", recorded_response["response"])
    lines = cast("list[str]", response["log_lines"])
    assert len(lines) == 2
    for line in lines:
        assert "'type': 'FORBIDDEN', 'path': ['resolveReviewThread']" in line
        assert "Resource not accessible by integration" in line
    # The gate sends no GraphQL mutation at all any more.
    assert "mutation" not in _GATE.read_text(encoding="utf-8")
    # The same two threads, open, with the rejected reply they carried: cleared.
    thread = {
        "id": "PRRT_kwDOPuAjtM6u01K4",
        "isResolved": False,
        "isOutdated": False,
        "path": ".github/workflows/hostile-review-thread-gate.yml",
        "originalLine": None,
        "subjectType": "FILE",
        "comments": {
            "nodes": [
                {"body": f"<!-- {gate.MARKER} fp=0b75b492dc2a -->\nfinding"},
                {"body": "rejected: the fixed-in check needs this PR's commit history"},
            ]
        },
    }

    def no_fix_check(sha: str, cited: object) -> tuple[bool, str]:
        raise AssertionError("a rejected disposition runs no fix check")

    blocking, cleared = gate.classify_threads([thread], no_fix_check)
    assert blocking == []
    assert cleared[0].disposition == "rejected"
