# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Alert on a scheduled workflow that keeps failing (OMN-18322).

Extends the OMN-18254 non-required-check failure-rate reader
(`scripts/ci/nonrequired_check_failure_rate.py`) to the class of run it never
looked at: `event == schedule` runs have no pull-request head, so nothing in
that reader's original check-run path could ever see them. The 2026-09-13
friction trend report (kb-internal#406/#407, sections 3 and 7) measured
scheduled workflow runs failing at 15.4% over 9,746 runs against 2.9% on pull
requests, with nothing triaging the difference.

THE FIXTURES ARE TWO REAL `actions/workflows/{id}/runs?event=schedule` PAGES
    Fetched live on 2026-09-13 against the trailing 7-day window
    (`created=>=<date>`), for two of the three repos this alert already
    watches (omnibase_infra). Each run object is projected to the fields the
    evaluator reads (id, name, path, workflow_id, event, status, conclusion,
    created_at, run_started_at, html_url) -- real values, reduced shape, to
    keep a 100-run capture small enough to commit; see the module docstring
    of the reader for the re-fetch command. Re-fetchable:
    `gh api "repos/OmniNode-ai/omnibase_infra/actions/workflows/<id>/runs?event=schedule&created=>=<date>&per_page=100"`.

    `dlq-depth-monitor.yml` (workflow id 344267547): 44 of its most recent 100
    scheduled runs failed (44.0%), above the 10% policy default -- the ABOVE
    case, and a real above-threshold defect, not a constructed one.

    `dev-lane-liveness.yml` (workflow id 322618483): 0 of its most recent 100
    scheduled runs failed (0.0%) -- the BELOW case, and a real below-threshold
    control, not a constructed one.

THE BELOW-THRESHOLD CONTROL IS NOT OPTIONAL
    Same argument OMN-18254 already made and the ticket restates: an alerter
    that is always silent and an alerter that is correct look identical from a
    green run. Both cases are driven through the same `evaluate_scheduled`.

THE PR-TRIGGERED PATH IS UNMODIFIED
    `test_nonrequired_check_failure_rate_omn18254.py` and
    `test_incident_replay_omn18254.py` are not touched by this ticket and
    both suites pass unchanged (20 passed) -- the positive control that
    extending the reader for scheduled runs did not alter the check-run path.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "nonrequired_check_failure_rate.py"
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "omn18322"
POLICY = REPO_ROOT / "config" / "runner_routing_policy.yaml"

ABOVE_FIXTURE = (
    FIXTURES / "omnibase_infra-dlq-depth-monitor-scheduled-runs.json.captured"
)
BELOW_FIXTURE = (
    FIXTURES / "omnibase_infra-dev-lane-liveness-scheduled-runs.json.captured"
)

ABOVE_WORKFLOW = ".github/workflows/dlq-depth-monitor.yml"
BELOW_WORKFLOW = ".github/workflows/dev-lane-liveness.yml"

REPO = "OmniNode-ai/omnibase_infra"


def _module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "nonrequired_check_failure_rate_scheduled", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _runs(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_both_captures_are_present_and_parse() -> None:
    for path in (ABOVE_FIXTURE, BELOW_FIXTURE):
        assert path.is_file(), f"{path} is missing"
        runs = json.loads(path.read_text(encoding="utf-8"))
        assert len(runs) == 100

    for run in _runs(ABOVE_FIXTURE):
        assert run["path"] == ABOVE_WORKFLOW
        assert run["event"] == "schedule"
    for run in _runs(BELOW_FIXTURE):
        assert run["path"] == BELOW_WORKFLOW
        assert run["event"] == "schedule"


def test_the_captures_record_what_this_ticket_claims() -> None:
    """Prove the premise from the bytes before testing anything about the code."""
    above = _runs(ABOVE_FIXTURE)
    failing = [r for r in above if r["conclusion"] == "failure"]
    assert len(failing) == 44, "the captured above-threshold rate has drifted"

    below = _runs(BELOW_FIXTURE)
    failing_below = [
        r
        for r in below
        if r["conclusion"] in {"failure", "timed_out", "action_required"}
    ]
    assert failing_below == [], "the captured below-threshold control has drifted"


def test_a_scheduled_workflow_above_threshold_raises_one_alert() -> None:
    """AC1, first half: names the workflow, the rate, the count, the URL."""
    module = _module()
    alert = module.evaluate_scheduled(REPO, ABOVE_WORKFLOW, _runs(ABOVE_FIXTURE), 10.0)
    assert alert is not None
    assert alert.repo == REPO
    assert alert.workflow == ABOVE_WORKFLOW
    assert alert.failures == 44
    assert alert.observed == 100
    assert alert.rate_pct == pytest.approx(44.0)
    assert alert.last_failure_url.startswith(
        "https://github.com/OmniNode-ai/omnibase_infra/actions/runs/"
    )
    assert ABOVE_WORKFLOW in alert.detail
    assert "44.0%" in alert.detail
    assert alert.last_failure_url in alert.detail


def test_the_below_threshold_control_fires_nothing() -> None:
    """AC1, second half, and the reason it is mandatory (see module docstring)."""
    module = _module()
    assert (
        module.evaluate_scheduled(REPO, BELOW_WORKFLOW, _runs(BELOW_FIXTURE), 10.0)
        is None
    )


def test_a_workflow_with_zero_observed_runs_is_silent_not_zero_percent() -> None:
    """A rate with no denominator is not evidence of anything.

    Distinguishes "nothing to report" from "reported and clean" -- the same
    distinction OMN-18254's report schema draws for an absent alerts file.
    """
    module = _module()
    assert module.evaluate_scheduled(REPO, "unused.yml", [], 10.0) is None


def test_a_workflow_at_exactly_the_threshold_does_not_alert() -> None:
    """Alerting is "more than" the threshold, not "at or above" it.

    Ten runs, exactly one failing, is exactly 10% -- the policy default -- and
    must not page: paging on the threshold itself would make the declared
    number a lie about where the line actually is.
    """
    module = _module()
    runs = [{"conclusion": "success"} for _ in range(9)] + [
        {"conclusion": "failure", "created_at": "2026-09-01T00:00:00Z", "html_url": "u"}
    ]
    assert module.evaluate_scheduled(REPO, "ten-runs.yml", runs, 10.0) is None


@pytest.mark.parametrize("conclusion", ["failure", "timed_out", "action_required"])
def test_the_failing_conclusions_match_the_check_run_path(conclusion: str) -> None:
    """Same FAILING_CONCLUSIONS set as the pull-request path, for one reason:
    a check and a workflow run should not disagree about what "failed" means.

    Two of ten (20%) rather than one of ten (10%): the exactly-at-threshold
    case is its own test above and must not alert, so this case has to clear
    the threshold rather than sit on it.
    """
    module = _module()
    runs = [{"conclusion": "success"} for _ in range(8)] + [
        {
            "conclusion": conclusion,
            "created_at": "2026-09-01T00:00:00Z",
            "html_url": "https://example.invalid/run/1",
        },
        {
            "conclusion": conclusion,
            "created_at": "2026-09-02T00:00:00Z",
            "html_url": "https://example.invalid/run/2",
        },
    ]
    alert = module.evaluate_scheduled(REPO, "two-in-ten.yml", runs, 10.0)
    assert alert is not None
    assert alert.rate_pct == pytest.approx(20.0)


@pytest.mark.parametrize("conclusion", ["success", "skipped", "neutral", "cancelled"])
def test_a_cancelled_or_skipped_run_is_not_a_failure(conclusion: str) -> None:
    module = _module()
    runs = [{"conclusion": conclusion} for _ in range(5)]
    assert module.evaluate_scheduled(REPO, "quiet.yml", runs, 10.0) is None


def test_the_last_failure_url_is_the_most_recent_failing_run_not_the_first() -> None:
    module = _module()
    runs = [
        {
            "conclusion": "failure",
            "created_at": "2026-09-01T00:00:00Z",
            "html_url": "https://example.invalid/run/oldest",
        },
        {"conclusion": "success", "created_at": "2026-09-05T00:00:00Z"},
        {
            "conclusion": "failure",
            "created_at": "2026-09-10T00:00:00Z",
            "html_url": "https://example.invalid/run/newest",
        },
    ]
    alert = module.evaluate_scheduled(REPO, "w.yml", runs, 10.0)
    assert alert is not None
    assert alert.last_failure_url == "https://example.invalid/run/newest"


def test_the_policy_declares_the_scheduled_threshold_and_window() -> None:
    """Rule 8: no silent default for either new key."""
    module = _module()
    block = module.load_policy(POLICY)
    assert block["scheduled_failure_threshold_pct"] == 10.0
    assert block["scheduled_window_days"] == 7


def test_the_policy_fails_loudly_without_the_scheduled_keys(tmp_path: Path) -> None:
    module = _module()
    partial = tmp_path / "policy.yaml"
    partial.write_text(
        "route:\n  nonrequired_check_alert:\n    failure_threshold: 3\n"
        "    heads_observed: 10\n",
        encoding="utf-8",
    )
    with pytest.raises(KeyError, match="scheduled_failure_threshold_pct"):
        module.load_policy(partial)
