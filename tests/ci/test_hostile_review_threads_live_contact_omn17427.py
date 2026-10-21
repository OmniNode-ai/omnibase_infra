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
