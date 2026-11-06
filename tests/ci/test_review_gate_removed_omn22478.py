# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22478, operator RULING 2026-11-05T20:27:56Z: no automated review holds a merge.

CI Summary is the only required context on dev, so it is the only surface here
that could hold a merge on the Hostile Reviewer. A head with the reviewer's rows
red or absent must pass it, the reviewer must start no run on a pull request, and
the manifest must not call its context required.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci.ci_summary_gate import (
    DEV_BASE_ONLY_CONTEXTS,
    EXIT_SUCCESS,
    EXPECTED_EXTERNAL_CONTEXTS,
    SKIPPABLE_GATE_JOBS,
    STRICT_GATE_JOBS,
    evaluate,
    scoped_to_base,
)

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 11, 5, 21, 0, 0, tzinfo=UTC)
REVIEW_CONTEXTS = (
    "Hostile Reviewer (adversarial gate)",
    "Hostile Review Gate",
    "Hostile Review Thread Gate",
    "Hostile Review Judge",
)


def _row(name: str, conclusion: str) -> dict[str, Any]:
    return {
        "id": abs(hash(name)) % 10_000_000,
        "name": name,
        "status": "completed",
        "conclusion": conclusion,
        "started_at": "2026-11-05T20:00:00Z",
        "completed_at": "2026-11-05T20:05:00Z",
        "head_sha": "a" * 40,
    }


def _gates() -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "status": "completed",
            "conclusion": "success",
            "started_at": "2026-11-05T20:00:00Z",
            "completed_at": "2026-11-05T20:05:00Z",
        }
        for name in (*STRICT_GATE_JOBS, *SKIPPABLE_GATE_JOBS)
    ]


def _triggers(workflow: str) -> dict[str, Any]:
    data = yaml.safe_load((REPO / ".github/workflows" / workflow).read_text())
    triggers = data.get("on", data.get(True))
    assert isinstance(triggers, dict)
    return triggers


@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "timed_out", None])
def test_ci_summary_passes_with_the_reviewer_rows_red_or_absent(
    conclusion: str | None,
) -> None:
    rows = [_row(context, "success") for context in EXPECTED_EXTERNAL_CONTEXTS]
    if conclusion is not None:
        rows += [_row(context, conclusion) for context in REVIEW_CONTEXTS]
    code, report = evaluate(
        _gates(),
        check_runs=rows,
        external_contexts=scoped_to_base(EXPECTED_EXTERNAL_CONTEXTS, "dev"),
        now=NOW,
    )
    assert code == EXIT_SUCCESS, report


def test_no_review_context_is_asserted_on_any_base() -> None:
    assert not set(REVIEW_CONTEXTS) & set(EXPECTED_EXTERNAL_CONTEXTS)
    assert not set(REVIEW_CONTEXTS) & DEV_BASE_ONLY_CONTEXTS


def test_the_manifest_does_not_call_a_review_context_required() -> None:
    manifest = yaml.safe_load((REPO / ".github/required-checks.yaml").read_text())
    required = {g["name"] for g in manifest["gates"] if g.get("mode") == "REQUIRED"}
    assert not set(REVIEW_CONTEXTS) & required


def test_the_reviewer_starts_no_run_on_a_pull_request_and_no_reply_posts_a_row() -> (
    None
):
    branches = _triggers("hostile-reviewer.yml")["pull_request"]["branches"]
    assert not {"dev", "main"} & set(branches)
    on_reply = yaml.safe_load(
        (REPO / ".github/workflows/hostile-review-thread-gate.yml").read_text()
    )
    assert on_reply["jobs"]["hostile-review-thread-gate"]["if"] == "${{ false }}"
