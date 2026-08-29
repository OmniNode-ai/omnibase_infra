# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16906 — the delivery liveness guard must catch both observed shapes.

The AC1 fix (a bare boolean default) ends *this* outage. This suite covers the
part the ticket calls the one that matters: making the NEXT one loud.

Two silent shapes were actually measured during the incident, and a guard that
misses either is worthless:

1. ``STARTUP_FAILURE`` — runs 33077062502 / 33091593742 / 33106694579 /
   33169436998. The run exists but never compiled, so there is no job, no step,
   and no annotation to go red.
2. ``NOT_FIRED`` — dev commit ``7090f386f`` (PR #2974, the OMN-16493 migration
   fence) merged 2026-08-28T20:13Z and produced **no run at all**. A guard that
   inspected only run conclusions would have reported clean straight through
   the exact merge whose non-delivery broke the staging deploy six minutes
   later.

Hermetic: the verdict function is driven over fixtures. No network, no ``gh``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci.check_dev_candidate_delivery_liveness import (
    DELIVERY_COMPLETION_GRACE,
    RUN_APPEARANCE_GRACE,
    assert_supported_patterns,
    evaluate,
    matches_filter,
    path_filter,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
DELIVER = WORKFLOWS / "deliver-dev-candidate-to-staging.yml"
GUARD = WORKFLOWS / "dev-candidate-delivery-liveness.yml"

NOW = datetime(2026, 8, 29, 0, 0, tzinfo=UTC)
PATTERNS = ["src/**", "docker/**", "pyproject.toml", "uv.lock"]


def _ts(delta: timedelta) -> str:
    return (NOW - delta).strftime("%Y-%m-%dT%H:%M:%SZ")


def _commit(sha: str, age: timedelta, files: list[str]) -> dict[str, Any]:
    return {"sha": sha, "committed_at": _ts(age), "files": files}


def _run(
    run_id: int,
    sha: str,
    age: timedelta,
    conclusion: str | None,
    status: str = "completed",
) -> dict[str, Any]:
    return {
        "id": run_id,
        "head_sha": sha,
        "status": status,
        "conclusion": conclusion,
        "created_at": _ts(age),
    }


class TestTheObservedIncidentShapes:
    def test_startup_failure_on_the_newest_run_is_a_finding(self) -> None:
        """Run 33169436998's shape. Nothing else in the org goes red on this."""
        verdict = evaluate(
            runs=[
                _run(33169436998, "25a24fae", timedelta(hours=12), "startup_failure")
            ],
            candidate_commits=[],
            patterns=PATTERNS,
            now=NOW,
        )
        assert not verdict.ok
        assert [f.code for f in verdict.findings] == ["STARTUP_FAILURE"]
        assert "33169436998" in verdict.findings[0].detail

    def test_a_trigger_touching_dev_commit_with_no_run_is_a_finding(self) -> None:
        """Commit 7090f386f's shape — zero runs, which run-conclusion checks miss."""
        verdict = evaluate(
            runs=[_run(1, "older", timedelta(days=2), "success")],
            candidate_commits=[
                _commit("7090f386f", timedelta(hours=4), ["src/omnibase_infra/x.py"])
            ],
            patterns=PATTERNS,
            now=NOW,
        )
        assert not verdict.ok
        assert [f.code for f in verdict.findings] == ["NOT_FIRED"]
        assert "7090f386f" in verdict.findings[0].detail

    def test_both_shapes_at_once_are_both_reported(self) -> None:
        """The real incident was both: dead workflow AND an undelivered merge."""
        verdict = evaluate(
            runs=[
                _run(33169436998, "25a24fae", timedelta(hours=12), "startup_failure")
            ],
            candidate_commits=[
                _commit("7090f386f", timedelta(hours=4), ["docker/migrations/0031.sql"])
            ],
            patterns=PATTERNS,
            now=NOW,
        )
        assert {f.code for f in verdict.findings} == {"STARTUP_FAILURE", "NOT_FIRED"}


class TestHealthyStates:
    def test_delivered_commit_is_clean(self) -> None:
        verdict = evaluate(
            runs=[_run(9, "abc123", timedelta(hours=2), "success")],
            candidate_commits=[
                _commit("abc123", timedelta(hours=3), ["src/omnibase_infra/x.py"])
            ],
            patterns=PATTERNS,
            now=NOW,
        )
        assert verdict.ok, verdict.findings

    def test_commit_inside_the_run_appearance_grace_is_not_yet_a_finding(self) -> None:
        verdict = evaluate(
            runs=[_run(9, "older", timedelta(days=1), "success")],
            candidate_commits=[
                _commit("fresh", RUN_APPEARANCE_GRACE / 2, ["src/a.py"])
            ],
            patterns=PATTERNS,
            now=NOW,
        )
        assert verdict.ok, verdict.findings

    def test_in_flight_build_inside_the_completion_grace_is_not_a_finding(self) -> None:
        """The candidate build is allotted 75 minutes; slow is not failed."""
        verdict = evaluate(
            runs=[_run(9, "abc", timedelta(minutes=40), None, status="in_progress")],
            candidate_commits=[_commit("abc", timedelta(minutes=45), ["src/a.py"])],
            patterns=PATTERNS,
            now=NOW,
        )
        assert verdict.ok, verdict.findings

    def test_a_docs_only_dev_commit_does_not_demand_a_run(self) -> None:
        verdict = evaluate(
            runs=[_run(9, "older", timedelta(days=1), "success")],
            candidate_commits=[_commit("docsonly", timedelta(hours=6), ["README.md"])],
            patterns=PATTERNS,
            now=NOW,
        )
        assert verdict.ok, verdict.findings


class TestLateAndFailedStates:
    def test_a_run_stuck_past_the_completion_grace_is_a_finding(self) -> None:
        age = DELIVERY_COMPLETION_GRACE + timedelta(hours=1)
        verdict = evaluate(
            runs=[_run(9, "abc", age, None, status="in_progress")],
            candidate_commits=[_commit("abc", age, ["src/a.py"])],
            patterns=PATTERNS,
            now=NOW,
        )
        assert [f.code for f in verdict.findings] == ["DELIVERY_STALLED"]

    def test_a_completed_failure_is_a_finding(self) -> None:
        verdict = evaluate(
            runs=[_run(9, "abc", timedelta(hours=2), "failure")],
            candidate_commits=[_commit("abc", timedelta(hours=3), ["src/a.py"])],
            patterns=PATTERNS,
            now=NOW,
        )
        assert [f.code for f in verdict.findings] == ["NOT_DELIVERED"]

    def test_a_cancelled_run_is_not_silently_treated_as_delivered(self) -> None:
        """Concurrency cancels supersede OLDER commits, never the newest one."""
        verdict = evaluate(
            runs=[_run(9, "abc", timedelta(hours=2), "cancelled")],
            candidate_commits=[_commit("abc", timedelta(hours=3), ["src/a.py"])],
            patterns=PATTERNS,
            now=NOW,
        )
        assert [f.code for f in verdict.findings] == ["NOT_DELIVERED"]


class TestFailsClosed:
    def test_an_empty_run_list_is_a_finding_not_a_pass(self) -> None:
        """An empty result is not evidence of absence."""
        verdict = evaluate(runs=[], candidate_commits=[], patterns=PATTERNS, now=NOW)
        assert [f.code for f in verdict.findings] == ["NO_RUNS"]

    def test_an_unimplemented_glob_shape_raises_rather_than_approximating(self) -> None:
        with pytest.raises(ValueError, match="does not implement"):
            assert_supported_patterns(["src/**/*.py"])

    def test_a_workflow_without_push_paths_raises(self, tmp_path: Path) -> None:
        broken = tmp_path / "deliver.yml"
        broken.write_text("on:\n  push:\n    branches: [dev]\n", encoding="utf-8")
        with pytest.raises(ValueError, match=r"on\.push\.paths"):
            path_filter(broken)


class TestTriggerIsReadFromTheWorkflowNotRestated:
    def test_guard_reads_the_live_delivery_trigger(self) -> None:
        """Drift safety: the guard must not carry its own copy of the trigger."""
        patterns = path_filter(DELIVER)
        document = yaml.safe_load(DELIVER.read_text(encoding="utf-8"))
        triggers = document.get("on", document.get(True))
        assert patterns == [str(p) for p in triggers["push"]["paths"]]

    def test_live_trigger_shapes_are_all_implemented_exactly(self) -> None:
        assert_supported_patterns(path_filter(DELIVER))

    def test_matching_agrees_with_the_live_trigger_on_real_paths(self) -> None:
        patterns = path_filter(DELIVER)
        assert matches_filter("src/omnibase_infra/runtime/x.py", patterns)
        assert matches_filter("docker/migrations/0031_x.sql", patterns)
        assert matches_filter("uv.lock", patterns)
        assert not matches_filter("README.md", patterns)
        assert not matches_filter("tests/ci/test_x.py", patterns)


class TestGuardWorkflowWiring:
    @pytest.fixture(scope="class")
    def guard(self) -> dict[Any, Any]:
        loaded = yaml.safe_load(GUARD.read_text(encoding="utf-8"))
        assert isinstance(loaded, dict)
        return loaded

    def test_guard_is_scheduled_so_it_notices_without_being_asked(
        self, guard: dict[Any, Any]
    ) -> None:
        triggers = guard.get("on", guard.get(True))
        assert "schedule" in triggers, (
            "a delivery guard that only runs on demand reproduces the failure "
            "mode it exists to close"
        )
        assert triggers["schedule"][0]["cron"] == "11,41 * * * *"

    def test_guard_runs_the_real_checker(self) -> None:
        text = GUARD.read_text(encoding="utf-8")
        assert "scripts/ci/check_dev_candidate_delivery_liveness.py" in text

    def test_guard_does_not_share_fate_with_the_self_hosted_fleet(
        self, guard: dict[Any, Any]
    ) -> None:
        assert guard["jobs"]["delivery-liveness"]["runs-on"] == "ubuntu-latest"

    def test_guard_is_registered_in_the_hosted_runner_allowlist(self) -> None:
        policy = yaml.safe_load(
            (REPO_ROOT / "config" / "runner_routing_policy.yaml").read_text(
                encoding="utf-8"
            )
        )
        allowlisted = {
            entry["path"] for entry in policy.get("hosted_runner_allowlist", [])
        }
        assert ".github/workflows/dev-candidate-delivery-liveness.yml" in allowlisted

    def test_guard_needs_actions_read_to_see_run_history(
        self, guard: dict[Any, Any]
    ) -> None:
        assert guard["jobs"]["delivery-liveness"]["permissions"]["actions"] == "read"

    def test_guard_has_no_skip_or_bypass_tokens(self) -> None:
        text = GUARD.read_text(encoding="utf-8")
        assert "[skip-" not in text
        assert "continue-on-error" not in text
