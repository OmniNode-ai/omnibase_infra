# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Replay of real pull-request facts through the dev rebuild publisher (OMN-26637).

``runtime-rebuild-trigger.yml`` hands the publisher the pull_request event's
``merged`` state and ``head.sha``. The recording holds those facts for one open
pull request (the head of this change) and one merged pull request with its
merge commit on dev, as the PR watcher and git recorded them. Replayed through
the real publisher command, the open head must be refused and the merge commit
must pass the guard, so the refusal is the publisher's decision on recorded
input and not an invocation error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from scripts import trigger_rebuild_on_merge as trigger

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]


def _publish(source_sha: str, merged: str, head_sha: str) -> Any:
    return CliRunner().invoke(
        trigger.main,
        [
            "--changed-files",
            "src/omnibase_infra/runtime/x.py",
            "--labels",
            "runtime_change",
            "--base-branch",
            "dev",
            "--source-sha",
            source_sha,
            "--pr-merged",
            merged,
            "--pr-head-sha",
            head_sha,
            "--runtime-path-validator",
            str(ROOT / "scripts" / "runtime_change_classifier.py"),
            "--dry-run",
        ],
    )


@pytest.mark.live_contact(
    "tests/ci/fixtures/omn26637_pull_request_events_dev_publish.json"
)
def test_the_recorded_open_pull_request_head_is_refused_for_dev(
    recorded_response: dict[str, Any],
) -> None:
    opened = recorded_response["open_pull_request"]
    # positive control: the recording really is an unmerged pull request
    assert opened["merged"] == "false"
    assert opened["state"] == "OPEN"

    result = _publish(opened["head_sha"], opened["merged"], opened["head_sha"])

    assert result.exit_code == 1
    assert "not merged" in result.output


@pytest.mark.live_contact(
    "tests/ci/fixtures/omn26637_pull_request_events_dev_publish.json"
)
def test_the_recorded_merge_commit_is_not_refused_for_dev(
    recorded_response: dict[str, Any],
) -> None:
    merged = recorded_response["merged_pull_request"]
    # positive control: the recording's merge commit differs from its head
    assert merged["merge_commit_sha"] != merged["head_sha"]

    # the guard returns nothing for a merged pull request's merge commit
    assert (
        trigger.assert_source_is_merged_ref(
            source_sha=merged["merge_commit_sha"],
            pr_merged=merged["merged"],
            pr_head_sha=merged["head_sha"],
        )
        is None
    )
    # and the same recorded facts reach past it on the command: the refusal
    # texts of the guard are absent from its output
    result = _publish(merged["merge_commit_sha"], merged["merged"], merged["head_sha"])
    assert "not merged" not in result.output
    assert "head commit" not in result.output
