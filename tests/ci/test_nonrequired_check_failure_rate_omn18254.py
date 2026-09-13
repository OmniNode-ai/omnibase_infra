# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A non-required check failing repeatedly raises one alert (OMN-18254).

Phase 3 item 14 of the process-friction remediation plan (epic OMN-18232), and
the last of its five.

WHY THE BELOW-THRESHOLD CONTROL IS NOT OPTIONAL
    The ticket says so in as many words, and it is the same argument the rest of
    this phase rests on: an alerter that is always silent and an alerter that is
    correct look identical from a green run. Every test that requires an alert
    here is paired with one that requires silence, driven over the same real
    bytes with only the input changed.

THE FIXTURES ARE FOUR REAL CHECK-RUN RESPONSES
    The web repository's `occ-companion-effect / Publish occ-companion-effect
    command` check failed on the heads of three consecutive pull requests, and
    that repository's `dev` requires exactly one context, `merge-hold-gate /
    evaluate` -- so the failing check blocked nothing and nobody learned. The
    fourth capture is the head of the pull request that fixed it, where the same
    check concludes `success`. That is the below-threshold control, and it is a
    real measurement rather than a constructed one.
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
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "omn18254"
POLICY = REPO_ROOT / "config" / "runner_routing_policy.yaml"

# The three heads whose companion-effect check failed, and the one that fixed it.
FAILING_HEADS = (
    "omniweb-checkruns-8024bb78.json.captured",
    "omniweb-checkruns-87adf60a.json.captured",
    "omniweb-checkruns-930fcc5d.json.captured",
)
FIXED_HEAD = "omniweb-checkruns-1ea89df5.json.captured"

CHECK = "occ-companion-effect / Publish occ-companion-effect command"
# The single context omniweb's `dev` actually required, read live 2026-09-13.
OMNIWEB_REQUIRED = {"merge-hold-gate / evaluate"}


def _module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "nonrequired_check_failure_rate", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pages(*names: str) -> list[dict[str, Any]]:
    return [json.loads((FIXTURES / n).read_text(encoding="utf-8")) for n in names]


def test_every_capture_is_present_and_parses() -> None:
    """A missing fixture must fail loudly rather than silently weaken a case."""
    for name in (*FAILING_HEADS, FIXED_HEAD):
        path = FIXTURES / name
        assert path.is_file(), f"{path} is missing"
        assert json.loads(path.read_text(encoding="utf-8"))["check_runs"]


def test_the_captures_record_what_this_ticket_claims() -> None:
    """Prove the premise from the bytes before testing anything about the code."""
    for name in FAILING_HEADS:
        page = _pages(name)[0]
        verdicts = [r["conclusion"] for r in page["check_runs"] if r["name"] == CHECK]
        assert verdicts == ["failure"], f"{name} no longer records the failure"
    fixed = _pages(FIXED_HEAD)[0]
    assert [r["conclusion"] for r in fixed["check_runs"] if r["name"] == CHECK] == [
        "success"
    ]


def test_a_non_required_check_above_threshold_raises_one_alert() -> None:
    """AC1, first half: one alert naming the check and its count."""
    module = _module()
    alerts = module.evaluate(
        "OmniNode-ai/omniweb", _pages(*FAILING_HEADS), OMNIWEB_REQUIRED, 3
    )
    assert len(alerts) == 1, [a.check for a in alerts]
    alert = alerts[0]
    assert alert.check == CHECK
    assert alert.failures == 3
    assert alert.observed == 3
    assert CHECK in alert.detail
    assert "3 of 3" in alert.detail


def test_the_below_threshold_control_fires_nothing() -> None:
    """AC1, second half, and the reason it is mandatory.

    Two of the three failing heads plus the head that fixed it: two failures
    against a threshold of three. An alerter that fired here would be firing on
    ordinary red, and would be muted within a week.
    """
    module = _module()
    pages = _pages(FAILING_HEADS[0], FAILING_HEADS[1], FIXED_HEAD)
    assert module.evaluate("OmniNode-ai/omniweb", pages, OMNIWEB_REQUIRED, 3) == []


def test_a_required_check_is_never_alerted_on() -> None:
    """If the same check were required, its failure would already block a merge.

    Driven over the identical failing bytes with only the required set changed,
    so the difference in verdict is attributable to that and nothing else.
    """
    module = _module()
    required = OMNIWEB_REQUIRED | {CHECK}
    assert (
        module.evaluate("OmniNode-ai/omniweb", _pages(*FAILING_HEADS), required, 3)
        == []
    )


def test_an_umbrella_enforced_context_is_treated_as_required() -> None:
    """The naive reading -- absent from branch protection means unwatched -- would
    report dozens of omnibase_infra checks that do block a merge.

    The Receipt Gate is the sharpest example: it surfaces as `verify / verify`,
    it is nowhere in branch protection, and it blocks every merge.
    """
    module = _module()
    declared = module.load_policy(POLICY)["umbrella_enforced_contexts"][
        "omnibase_infra"
    ]
    for enforced in ("CI Summary", "verify / verify", "occ-preflight / eligibility"):
        assert enforced in declared, f"{enforced} would be reported as unwatched"


def test_a_rerun_storm_on_one_head_cannot_reach_the_threshold() -> None:
    """One head contributes at most one failure per check name.

    Without this, a single pull request re-run three times would page as though
    three separate changes had hit the same defect.
    """
    module = _module()
    one_head = _pages(FAILING_HEADS[0])[0]
    storm = {
        "check_runs": [r for r in one_head["check_runs"] if r["name"] == CHECK] * 5
    }
    assert module.evaluate("OmniNode-ai/omniweb", [storm], OMNIWEB_REQUIRED, 3) == []


@pytest.mark.parametrize("conclusion", ["failure", "timed_out", "action_required"])
def test_the_failing_conclusions_are_the_ones_that_mean_something_broke(
    conclusion: str,
) -> None:
    module = _module()
    pages = [
        {"check_runs": [{"name": CHECK, "conclusion": conclusion}]} for _ in range(3)
    ]
    assert len(module.evaluate("r", pages, set(), 3)) == 1


@pytest.mark.parametrize("conclusion", ["success", "skipped", "neutral", "cancelled"])
def test_a_cancelled_or_skipped_check_is_not_a_failure(conclusion: str) -> None:
    """`cancelled` is deliberately not counted.

    `gh pr checks` renders a cancellation as a failure, which is a known local
    trap; counting it here would turn every concurrency-group cancel into an
    alert about a check that never ran.
    """
    module = _module()
    pages = [
        {"check_runs": [{"name": CHECK, "conclusion": conclusion}]} for _ in range(5)
    ]
    assert module.evaluate("r", pages, set(), 3) == []


def test_the_policy_declares_a_threshold_and_fails_loudly_without_one(
    tmp_path: Path,
) -> None:
    """Rule 8: a silently-defaulted threshold is how an alerter ends up either
    permanently silent or permanently noisy."""
    module = _module()
    assert module.load_policy(POLICY)["failure_threshold"] == 3
    empty = tmp_path / "policy.yaml"
    empty.write_text("route: {}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="nonrequired_check_alert"):
        module.load_policy(empty)


def test_required_contexts_reads_both_branch_protection_shapes() -> None:
    """GitHub returns `contexts` on one shape and `checks` on the other."""
    module = _module()
    assert module.required_contexts({"contexts": ["a", "b"]}) == {"a", "b"}
    assert module.required_contexts({"checks": [{"context": "a"}]}) == {"a"}


def test_the_umbrella_list_matches_the_summary_gate() -> None:
    """The policy's umbrella list is a snapshot of another module's constants.

    It will drift. This reads those constants live and requires the policy to
    declare every one of them, so drift surfaces as a red test instead of as a
    page about a check that does block a merge.

    Measured on the first live read-only run, with only the umbrella name
    declared: four findings on `omnibase_infra`, two of them enforced checks --
    `verify / verify`, the Receipt Gate, and `occ-preflight / eligibility`. Two
    false positives out of four on run one is how an alerter gets muted before
    it ever catches anything.
    """
    import sys as _sys

    _sys.path.insert(0, str(REPO_ROOT / "scripts" / "ci"))
    import ci_summary_gate as gate

    enforced = (
        {gate.SELF_JOB_NAME}
        | set(gate.STRICT_GATE_JOBS)
        | set(getattr(gate, "SKIPPABLE_GATE_JOBS", ()))
        | set(gate.EXPECTED_EXTERNAL_CONTEXTS)
    )
    module = _module()
    declared = set(
        module.load_policy(POLICY)["umbrella_enforced_contexts"]["omnibase_infra"]
    )
    missing = sorted(enforced - declared)
    assert not missing, (
        "these contexts are enforced by the CI Summary poller but are absent from "
        "route.nonrequired_check_alert.umbrella_enforced_contexts, so the alerter "
        "would page about checks that DO block a merge:\n  " + "\n  ".join(missing)
    )
