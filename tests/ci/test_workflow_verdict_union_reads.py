# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20277 -- the workflow-verdict reader unions several listing reads.

On 2026-09-30/10-01 the Delegation Health Check went red 14 times because one
read of the omnimarket nightly's run listing returned a page without the newest
runs: it named run 35320416369 (2026-09-18, failure) as newest while run
36832264088 (2026-10-01T07:46:16Z, success) existed. The reader now unions the
runs of every read by id, so a page that hides the newest run no longer decides
the verdict, and a newer red that any read returns still refuses.
"""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from typing import Any

import pytest

from scripts.ci.lab_pass_receipt import evaluate_workflow_verdict

pytestmark = pytest.mark.unit

REPO = "OmniNode-ai/omnimarket"
WORKFLOW = "delegation-regression-nightly.yml"
NOW = datetime(2026, 10, 1, 9, 56, 56, tzinfo=UTC)


def _run(
    run_id: int, *, conclusion: str, started: str, attempt: int = 1
) -> dict[str, Any]:
    return {
        "id": run_id,
        "event": "schedule",
        "status": "completed",
        "conclusion": conclusion,
        "head_branch": "dev",
        "head_sha": "2b591abfd3aae2485777111905e249112126b06c",
        "run_attempt": attempt,
        "created_at": started,
        "run_started_at": started,
        "display_title": "Delegation Regression (nightly) lane=stability-test",
        "html_url": f"https://github.com/{REPO}/actions/runs/{run_id}",
    }


OLD_RED = _run(35320416369, conclusion="failure", started="2026-09-18T07:38:43Z")
NEWEST_GREEN = _run(36832264088, conclusion="success", started="2026-10-01T07:46:16Z")


class _Pages:
    """Answers each read with the next page in order (the last repeats)."""

    def __init__(self, *pages: list[dict[str, Any]]) -> None:
        self.pages = pages
        self.calls = 0

    def __call__(self, _path: str) -> bytes:
        page = self.pages[min(self.calls, len(self.pages) - 1)]
        self.calls += 1
        return json.dumps({"workflow_runs": page}).encode()


def _verdict(pages: _Pages, monkeypatch: Any) -> tuple[int, str]:
    monkeypatch.setattr("scripts.ci.lab_pass_receipt._gh_api", pages)
    out = io.StringIO()
    code = evaluate_workflow_verdict(
        REPO,
        WORKFLOW,
        "dev",
        26.0,
        ("schedule", "workflow_dispatch"),
        out,
        now=NOW,
        dispatch_title_contains="lane=stability-test",
    )
    return code, out.getvalue()


def test_a_page_that_hides_the_newest_green_does_not_decide(monkeypatch: Any) -> None:
    code, output = _verdict(
        _Pages([OLD_RED], [NEWEST_GREEN, OLD_RED], [OLD_RED]), monkeypatch
    )
    assert code == 0, output
    assert "36832264088" in output
    assert "3 listing read(s) unioned" in output


def test_the_single_bad_page_alone_still_refuses(monkeypatch: Any) -> None:
    """Positive control: if every read hides the newest run, the old red decides."""
    code, output = _verdict(_Pages([OLD_RED]), monkeypatch)
    assert code == 1
    assert "35320416369" in output


def test_a_newer_red_in_any_read_still_refuses(monkeypatch: Any) -> None:
    newer_red = _run(36900000000, conclusion="failure", started="2026-10-01T09:00:00Z")
    code, output = _verdict(
        _Pages([NEWEST_GREEN], [newer_red, NEWEST_GREEN], [NEWEST_GREEN]), monkeypatch
    )
    assert code == 1
    assert "36900000000" in output


def test_a_red_rerun_seen_by_one_read_supersedes_its_green_first_attempt(
    monkeypatch: Any,
) -> None:
    red_rerun = _run(
        36832264088, conclusion="failure", started="2026-10-01T09:30:00Z", attempt=2
    )
    code, output = _verdict(
        _Pages([NEWEST_GREEN], [red_rerun], [NEWEST_GREEN]), monkeypatch
    )
    assert code == 1
    assert "attempt 2" in output


def test_a_stale_green_from_every_read_still_refuses(monkeypatch: Any) -> None:
    stale = _run(36538536307, conclusion="success", started="2026-09-29T07:45:26Z")
    code, output = _verdict(_Pages([stale]), monkeypatch)
    assert code == 1
    assert "stale green" in output


def test_any_unreadable_read_refuses(monkeypatch: Any) -> None:
    calls = {"n": 0}

    def flaky(_path: str) -> bytes:
        calls["n"] += 1
        if calls["n"] == 2:
            msg = "`gh api` exited 1: HTTP 502"
            raise RuntimeError(msg)
        return json.dumps({"workflow_runs": [NEWEST_GREEN]}).encode()

    monkeypatch.setattr("scripts.ci.lab_pass_receipt._gh_api", flaky)
    out = io.StringIO()
    code = evaluate_workflow_verdict(
        REPO, WORKFLOW, "dev", 26.0, ("schedule",), out, now=NOW
    )
    assert code == 1
    assert "unreadable" in out.getvalue()
