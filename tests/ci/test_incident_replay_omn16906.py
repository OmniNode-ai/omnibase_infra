# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16906 incident replay — both new guards, driven over the real bytes.

Two guards ship with this ticket and both are newly wired enforcement, so
OMN-15547's default-deny demands a real regression case for each rather than a
baseline exemption.

The incident: ``deliver-dev-candidate-to-staging.yml`` (OMN-15796) returned
GitHub's ``startup_failure`` on every run from the day it landed. That state
creates no job, no step, and no annotation — so nothing in either repo went red
while every omnibase_infra ``dev`` merge shipped merged-but-undelivered and
onex-dev kept serving an older candidate.

Every artifact below is the verbatim thing that failed, fetched from the surface
that failed, and re-fetchable from the locator recorded in
``tests/incident_replays/registry.yaml``. Nothing here is retyped.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci.check_dev_candidate_delivery_liveness import (
    evaluate,
    normalize_commit,
    normalize_run,
)
from scripts.ci.check_workflow_input_default_types import scan_document

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "omn16906"

# The verbatim blob PR #2912 merged — the commit that added the workflow_call
# block, and with it the string default that killed every caller run.
SHIPPED_WORKFLOW = FIXTURES / "build-workspace-candidate-runtime.64ffb5a15.yml.captured"
SHIPPED_WORKFLOW_SHA256 = (
    "f19da8ba0c25751e23e5f9de9bd660c69f07fa2477f836bbeda277de0d1f4b10"
)

# GitHub's own record of the last startup_failure run, run 33169436998, the one
# that dropped dev commit 25a24fae on the floor.
STARTUP_FAILURE_RUN = FIXTURES / "run-33169436998.gh-api.json.captured"
STARTUP_FAILURE_RUN_SHA256 = (
    "832b4d1553ffc575e13d023cdba0b01b47999184c886bf3edf6e2da7500b4d6e"
)

# The merge that produced no run at all: the OMN-16493 migration fence,
# PR #2974, whose non-delivery broke the staging deploy six minutes later.
UNDELIVERED_COMMIT = FIXTURES / "commit-7090f386f.gh-api.json.captured"
# OMN-17288: ONE byte-range in this capture was redacted, and the pin moved with
# it. The captured `files[].patch` quoted a live customer's tenant slug -- this
# repository is PUBLIC -- so that slug reads `t-REDACTED-OMN17288` here.
#
# Stated plainly because a "captured" fixture whose value is fidelity should not
# be edited quietly: the true bytes remain retrievable from the public commit
# this capture is OF (GitHub API `repos/OmniNode-ai/omnibase_infra/commits/
# 7090f386fba00b47b580c182f3fdd934225f6f53`), so nothing is lost, and redacting
# here does not shrink that exposure -- it only stops this working tree from
# being a second copy. What this test actually reads is `sha` and
# `files[].filename`; neither is inside the redacted range, so the replay is
# unaffected. The pin below still does its job for every byte from now on: it
# was `bdb0d64575b42940f7e761aa0e5f80f34cc044d68ed7da7b6b9ad23ed926c12d` before
# the redaction.
UNDELIVERED_COMMIT_SHA256 = (
    "2cada0bad2ed4623b32510bb444e255ea74ab5df7f25cb103212fb38e2daeb28"
)


def _pinned(path: Path, expected: str) -> bytes:
    """Read a fixture and refuse it if the bytes have moved since capture."""
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    assert actual == expected, (
        f"{path.name} has changed since capture ({actual} != {expected}); a "
        "reformatted artifact is no longer the artifact that failed"
    )
    return raw


class TestWorkflowInputDefaultTypeRatchet:
    """`scripts/ci/check_workflow_input_default_types.py` vs. the shipped blob."""

    def test_the_real_shipped_workflow_is_rejected_by_the_real_checker(self) -> None:
        """The false_green this ratchet closes.

        Every gate on PR #2912 was green. `actionlint` reports this file clean —
        it does not model GitHub's caller-side input type check. The only thing
        that said no was GitHub, at run time, in a state that produces no log.
        """
        document = yaml.safe_load(
            _pinned(SHIPPED_WORKFLOW, SHIPPED_WORKFLOW_SHA256).decode("utf-8")
        )
        violations = scan_document(SHIPPED_WORKFLOW.name, document)

        assert violations, (
            "the checker accepted the exact blob that produced four consecutive "
            "startup_failure runs — it would not have caught the incident"
        )
        offending = {(v.trigger, v.input_name) for v in violations}
        assert ("workflow_call", "no-cache") in offending, offending
        # The latent twin in the same file. GitHub tolerates it under
        # workflow_dispatch, which is why it survived long enough for the
        # workflow_call block to be copied from it.
        assert ("workflow_dispatch", "no-cache") in offending, offending

    def test_the_declared_type_and_the_shipped_default_really_do_disagree(
        self,
    ) -> None:
        """Read the defect straight out of the captured bytes, not the verdict."""
        document = yaml.safe_load(
            _pinned(SHIPPED_WORKFLOW, SHIPPED_WORKFLOW_SHA256).decode("utf-8")
        )
        spec = document[True]["workflow_call"]["inputs"]["no-cache"]
        assert spec["type"] == "boolean"
        assert spec["default"] == "false"
        assert isinstance(spec["default"], str)

    def test_the_repaired_file_is_accepted(self) -> None:
        """Accept control.

        Without it a checker hard-wired to reject every workflow would replay
        the incident perfectly and fail every healthy file in the repo.
        """
        repaired = (
            REPO_ROOT
            / ".github"
            / "workflows"
            / "build-workspace-candidate-runtime.yml"
        )
        document = yaml.safe_load(repaired.read_text(encoding="utf-8"))
        assert scan_document(repaired.name, document) == []


class TestDeliveryLivenessGuard:
    """`scripts/ci/check_dev_candidate_delivery_liveness.py` vs. the real API bytes."""

    @staticmethod
    def _now(payload: dict[str, Any], offset: timedelta) -> datetime:
        stamp = payload.get("created_at") or payload["commit"]["committer"]["date"]
        return (
            datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(UTC)
            + offset
        )

    def test_the_real_startup_failure_run_is_reported_not_passed_over(self) -> None:
        """The whole point: this run is what "nothing went red" looked like.

        Driven through the guard's real ``normalize_run`` projection, so the case
        also proves the guard can read the shape GitHub actually returns — a
        hand-built run dict would prove only that the verdict logic branches.
        """
        payload = json.loads(_pinned(STARTUP_FAILURE_RUN, STARTUP_FAILURE_RUN_SHA256))
        assert payload["conclusion"] == "startup_failure"

        verdict = evaluate(
            runs=[normalize_run(payload)],
            candidate_commits=[],
            patterns=["src/**", "docker/**"],
            now=self._now(payload, timedelta(hours=1)),
        )
        assert not verdict.ok, "the guard passed the run at the centre of the outage"
        assert [f.code for f in verdict.findings] == ["STARTUP_FAILURE"]
        assert "33169436998" in verdict.findings[0].detail

    def test_the_real_undelivered_merge_is_reported_though_no_run_exists(self) -> None:
        """The shape a conclusion-only guard cannot see.

        Commit 7090f386f produced zero delivery runs of any conclusion. There is
        no run object to inspect, so the only evidence that anything is wrong is
        the absence itself — which is why the guard reasons from the workflow's
        own path filter over real commits rather than from run history alone.
        """
        payload = json.loads(_pinned(UNDELIVERED_COMMIT, UNDELIVERED_COMMIT_SHA256))
        commit = normalize_commit(payload)
        assert commit["sha"] == "7090f386fba00b47b580c182f3fdd934225f6f53"
        assert any(f.startswith("docker/migrations/") for f in commit["files"]), commit[
            "files"
        ]

        verdict = evaluate(
            # A healthy-looking older run: run history alone says all is well.
            runs=[
                {
                    "id": 1,
                    "head_sha": "0" * 40,
                    "status": "completed",
                    "conclusion": "success",
                    "created_at": "2026-08-20T00:00:00Z",
                }
            ],
            candidate_commits=[commit],
            patterns=["src/**", "docker/**"],
            now=self._now(payload, timedelta(hours=2)),
        )
        assert [f.code for f in verdict.findings] == ["NOT_FIRED"]
        assert "7090f386f" in verdict.findings[0].detail

    def test_a_delivered_commit_of_the_same_shape_is_accepted(self) -> None:
        """Accept control against a guard that just always reports a gap."""
        payload = json.loads(_pinned(UNDELIVERED_COMMIT, UNDELIVERED_COMMIT_SHA256))
        commit = normalize_commit(payload)
        verdict = evaluate(
            runs=[
                {
                    "id": 2,
                    "head_sha": commit["sha"],
                    "status": "completed",
                    "conclusion": "success",
                    "created_at": commit["committed_at"],
                }
            ],
            candidate_commits=[commit],
            patterns=["src/**", "docker/**"],
            now=self._now(payload, timedelta(hours=2)),
        )
        assert verdict.ok, verdict.findings
