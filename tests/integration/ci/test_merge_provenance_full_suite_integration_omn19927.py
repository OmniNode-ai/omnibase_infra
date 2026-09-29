# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exercise GitHub HTTP provenance through full-suite selection (OMN-19927)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from omnibase_infra.nodes.node_merge_provenance_observe_effect.handlers.handler_merge_group_run_read_github import (
    HandlerMergeGroupRunReadGithub,
)
from scripts.ci import detect_test_paths, merge_provenance
from scripts.ci.test_selection_models import EnumFullSuiteReason, ModelTestSelection

if TYPE_CHECKING:
    from pytest_httpserver import HTTPServer

pytest.importorskip("pytest_httpserver")

pytestmark = pytest.mark.integration

REPO = "OmniNode-ai/omnibase_infra"
SHA_UNVALIDATED = "e263eee036a8e338e6b39ea746d7e59bf91602cb"
SHA_VALIDATED = "7b1f6fb690d65e86c804e011ef80b8485a4d070e"
RUN_ID = 36409772437
ADJ = Path(__file__).resolve().parents[3] / "scripts/ci/test_selection_adjacency.yaml"


def _runs_query(sha: str) -> dict[str, str]:
    return {"head_sha": sha, "event": "merge_group", "per_page": "100", "page": "1"}


def _evaluate_and_select(
    httpserver: HTTPServer, tmp_path: Path, sha: str
) -> tuple[dict[str, str], ModelTestSelection]:
    output = tmp_path / "github_output"
    result_json = tmp_path / "provenance.json"
    result = merge_provenance.evaluate_and_write(
        repository=REPO,
        sha=sha,
        reader=HandlerMergeGroupRunReadGithub(
            token="test-token", api_url=httpserver.url_for("")
        ),
        github_output=output,
        result_json=result_json,
    )
    httpserver.check_assertions()
    outputs = dict(
        line.split("=", 1) for line in output.read_text().splitlines() if line
    )
    assert json.loads(result_json.read_text()) == result.model_dump(mode="json")
    assert outputs["verdict"] == result.verdict.value
    assert outputs["force_full_suite"] == str(result.forces_full_suite).lower()

    # Match ci.yml: only an explicit false permits normal selection.
    forced_reason = (
        EnumFullSuiteReason.UNVALIDATED_PUSH
        if outputs["force_full_suite"] != "false"
        else None
    )
    selection = detect_test_paths.compute_selection(
        changed_files=["docs/x.md"],
        adjacency_path=ADJ,
        ref_name="dev",
        event_name="push",
        force_full_suite_reason=forced_reason,
    )
    return outputs, selection


def test_unvalidated_push_forces_full_suite_for_docs_only_diff(
    httpserver: HTTPServer, tmp_path: Path
) -> None:
    httpserver.expect_oneshot_request(
        f"/repos/{REPO}/actions/runs",
        method="GET",
        query_string=_runs_query(SHA_UNVALIDATED),
    ).respond_with_json({"total_count": 0, "workflow_runs": []})

    outputs, selection = _evaluate_and_select(httpserver, tmp_path, SHA_UNVALIDATED)

    assert outputs["verdict"] == "UNVALIDATED"
    assert outputs["force_full_suite"] == "true"
    assert selection.is_full_suite is True
    assert selection.full_suite_reason == "unvalidated_push"
    assert selection.selected_paths == ["tests/"]


def test_successful_summary_keeps_normal_docs_only_selection(
    httpserver: HTTPServer, tmp_path: Path
) -> None:
    httpserver.expect_oneshot_request(
        f"/repos/{REPO}/actions/runs",
        method="GET",
        query_string=_runs_query(SHA_VALIDATED),
    ).respond_with_json(
        {
            "total_count": 1,
            "workflow_runs": [
                {
                    "id": RUN_ID,
                    "run_attempt": 1,
                    "head_sha": SHA_VALIDATED,
                    "head_branch": "gh-readonly-queue/dev/pr-4235-31258513",
                    "path": ".github/workflows/ci.yml",
                    "event": "merge_group",
                    "status": "completed",
                    "conclusion": "failure",
                }
            ],
        }
    )
    httpserver.expect_oneshot_request(
        f"/repos/{REPO}/actions/runs/{RUN_ID}/jobs",
        method="GET",
        query_string={"filter": "latest", "per_page": "100", "page": "1"},
    ).respond_with_json(
        {
            "total_count": 1,
            "jobs": [
                {"name": "CI Summary", "status": "completed", "conclusion": "success"}
            ],
        }
    )

    outputs, selection = _evaluate_and_select(httpserver, tmp_path, SHA_VALIDATED)

    assert outputs["verdict"] == "VALIDATED"
    assert outputs["force_full_suite"] == "false"
    assert selection.is_full_suite is False
    assert selection.full_suite_reason is None
    assert selection.selected_paths == []


def test_failed_read_is_undecidable_and_forces_full_suite(
    httpserver: HTTPServer, tmp_path: Path
) -> None:
    httpserver.expect_oneshot_request(
        f"/repos/{REPO}/actions/runs",
        method="GET",
        query_string=_runs_query(SHA_UNVALIDATED),
    ).respond_with_json({"message": "Internal Server Error"}, status=500)

    outputs, selection = _evaluate_and_select(httpserver, tmp_path, SHA_UNVALIDATED)

    assert outputs["verdict"] == "UNDECIDABLE"
    assert outputs["force_full_suite"] == "true"
    assert (
        "HTTP 500"
        in json.loads((tmp_path / "provenance.json").read_text())["reason_detail"]
    )
    assert selection.is_full_suite is True
    assert selection.full_suite_reason == "unvalidated_push"
    assert selection.selected_paths == ["tests/"]
