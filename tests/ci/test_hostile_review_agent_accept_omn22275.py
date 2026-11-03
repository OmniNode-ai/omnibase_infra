# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22275: the Hostile Review job accepts an agent receipt only through ``pass_agent``.

Each test runs the workflow's own review, accept and enforcement steps and its gate job
(see ``_hostile_review_job_omn22275``) with a node fake, and asserts the conclusion of
the two contexts CI Summary's external sweep reads:

* a degraded voter result with a verified passing receipt for the exact head turns
  ``Hostile Review Gate`` green;
* a degraded result in which the voter that answered raised a blocking finding stays red,
  receipt or not (the node answers ``unchanged``);
* a receipt for another head is refused and the degraded verdict stays red;
* with no receipt, or a node that does not run, or a clone without the node, the
  voters' verdict stands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.ci._hostile_review_job_omn22275 import (
    HEAD_SHA,
    OTHER_HEAD_SHA,
    Run,
    reported_contexts,
    simulate,
)

pytestmark = pytest.mark.unit


def _run(tmp_path: Path, case: str, **kwargs: Any) -> Run:
    work = tmp_path / case
    work.mkdir()
    return simulate(work, **kwargs)


def _assert_contexts(run: Run, *, green: bool, case: str) -> None:
    required = reported_contexts()
    assert len(required) == 2
    assert set(required) == set(run.contexts)
    for context in required:
        assert run.green(context) is green, (
            f"{case}: context {context!r} should be "
            f"{'green' if green else 'red'}; steps={run.exits} skipped={run.skipped} "
            f"outputs={run.outputs}"
        )


def test_degraded_voters_with_a_verified_receipt_for_the_exact_head_go_green(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path, "accepted", verdict="degraded", receipt_head=HEAD_SHA)
    _assert_contexts(run, green=True, case="degraded + receipt for the head")
    assert run.outputs["accept"]["decision"] == "pass_agent"
    assert run.outputs["accept"]["receipt_id"] == "rcpt-1"
    assert "enforce" in run.skipped, "pass_agent skips the enforcement step"
    assert len(run.node_calls) == 1, run.node_calls


def test_degraded_voters_without_a_receipt_stay_red(tmp_path: Path) -> None:
    run = _run(tmp_path, "norcpt", verdict="degraded")
    _assert_contexts(run, green=False, case="degraded, no receipt")
    assert run.outputs["accept"]["decision"] == "unchanged"
    assert run.exits["enforce"] == 1, "the enforcement step ran and failed the job"


def test_a_blocking_finding_from_a_voter_that_answered_stays_red_with_a_receipt(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path, "blocking", verdict="degraded_blocking", receipt_head=HEAD_SHA)
    _assert_contexts(run, green=False, case="degraded + blocking finding + receipt")
    assert run.outputs["accept"]["decision"] == "unchanged"
    assert run.exits["enforce"] == 1


def test_a_blocked_quorum_is_left_to_the_thread_gate_and_accept_changes_nothing(
    tmp_path: Path,
) -> None:
    """This workflow reads a blocked quorum as ``passed`` (OMN-17492: the model finds,
    the Hostile Review Thread Gate blocks), so the two contexts are green with or
    without a receipt; the node answers ``unchanged`` and the enforcement step runs.
    """
    for receipt_head in ("", HEAD_SHA):
        run = _run(
            tmp_path,
            f"blocked-{bool(receipt_head)}",
            verdict="blocked",
            receipt_head=receipt_head,
        )
        _assert_contexts(run, green=True, case="blocked quorum")
        assert run.outputs["accept"]["decision"] == "unchanged"
        assert run.exits["enforce"] == 0


def test_a_receipt_for_another_head_is_refused_and_the_gate_stays_red(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path, "otherhead", verdict="degraded", receipt_head=OTHER_HEAD_SHA)
    _assert_contexts(run, green=False, case="degraded + receipt for another head")
    assert run.outputs["accept"]["decision"] == "refuse"
    assert run.exits["enforce"] == 1


def test_passed_voters_stay_green_and_the_enforcement_step_runs(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path, "passed", verdict="passed")
    _assert_contexts(run, green=True, case="passed voters")
    assert run.exits["enforce"] == 0, "the enforcement step still ran and passed"


def test_the_node_is_asked_about_the_pull_request_head_and_the_reviewers_result(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path, "input", verdict="degraded", receipt_head=HEAD_SHA)
    (call,) = run.node_calls
    flags = dict(
        zip(
            call[call.index("--repository") :: 2],
            call[call.index("--repository") + 1 :: 2],
            strict=True,
        )
    )
    assert flags["--head-sha"] == HEAD_SHA
    assert flags["--voter-result-path"].endswith(
        "/workspace/hostile-review-result.json"
    )


def test_a_node_that_does_not_run_leaves_the_voters_verdict_standing(
    tmp_path: Path,
) -> None:
    degraded = _run(
        tmp_path, "crash-d", verdict="degraded", receipt_head=HEAD_SHA, node_exit=7
    )
    _assert_contexts(degraded, green=False, case="degraded, node exit 7")
    assert degraded.outputs["accept"]["decision"] == "unchanged"
    passed = _run(tmp_path, "crash-p", verdict="passed", node_exit=7)
    _assert_contexts(passed, green=True, case="passed, node exit 7")


def test_a_clone_without_the_node_skips_with_a_notice_and_the_verdict_stands(
    tmp_path: Path,
) -> None:
    degraded = _run(
        tmp_path,
        "nonode-d",
        verdict="degraded",
        receipt_head=HEAD_SHA,
        node_in_clone=False,
    )
    _assert_contexts(degraded, green=False, case="degraded, clone without the node")
    assert degraded.outputs["accept"]["decision"] == "unchanged"
    assert degraded.node_calls == []
    passed = _run(tmp_path, "nonode-p", verdict="passed", node_in_clone=False)
    _assert_contexts(passed, green=True, case="passed, clone without the node")
