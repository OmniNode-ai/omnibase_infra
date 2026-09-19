# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A merge to ``dev`` must prove the union it just created (OMN-18835).

Why this module exists
----------------------
``ci.yml`` used to trigger on ``push`` to ``main`` only. Every job in it
therefore ran against a pull request's own merge-base preview and never against
``dev``'s post-merge head, so a break formed by the UNION of two individually
green pull requests was structurally invisible in this repository. It surfaced
only as collateral damage on some unrelated open pull request that happened to
pick the new head up as its merge base.

Measured 2026-09-19, twice in two hours. ``#3818`` + ``#3822`` merged at
12:58:39Z and broke ``Deploy Agent Tests``; the first red run anywhere was at
13:00:24Z on ``#3795``, a pull request with nothing to do with either of them;
the first human notice was 13:29:21Z, 30 minutes 57 seconds after the break went
live; five of six open pull requests were blocked until ``#3833`` merged at
14:14:13Z. ``#3827`` + ``#3829`` repeated the shape the same afternoon. No
automated surface reported either one.

Three invariants, and the second is the one that is easy to lose
----------------------------------------------------------------
1. The trigger itself. Trivially checkable, trivially deletable.

2. **The cascade.** Adding the trigger alone would have produced a
   guaranteed-red run on every merge, which is worse than no signal at all,
   because ``contract-sync-gate`` was gated to ``pull_request``/``merge_group``
   and so reported ``skipped`` on a push -- and ``detect-changes`` ``needs:`` it
   with no ``if:`` of its own, so GitHub skipped ``detect-changes``, then
   ``test-parallel``, and then the STRICT ``CI Tests Gate`` FAILED with "Test
   matrix skipped despite admitted PR / non-PR event". That is why all five of
   the most recent push runs on ``main`` concluded ``failure`` while every job
   that actually ran was green.

   Registering the job in ``SKIPPABLE_GATE_JOBS`` does not help and never did:
   that governs how the POLLER reads a conclusion, and GitHub resolves a
   ``needs:`` cascade before the poller sees the run at all.

   :func:`test_no_job_detect_changes_needs_is_event_gated` is therefore the
   load-bearing test here. It does not check the one job that was wrong; it
   checks the PROPERTY that made it wrong, over every one of the sixteen
   dependencies, so re-gating any of them on an event is a red test rather than
   a silent return to a permanently red push lane.

3. No pull-request-evaluated semantics move. The required contexts on this
   repository's ``dev`` are evaluated on ``pull_request`` runs, and this change
   must not touch what they mean.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
CI_SUMMARY_GATE = REPO_ROOT / "scripts" / "ci" / "ci_summary_gate.py"


def _load_workflow(path: Path) -> dict[Any, Any]:
    """Parse a workflow file.

    PyYAML's YAML 1.1 resolver parses the bare ``on:`` key as the boolean
    ``True`` rather than the string ``"on"``. Callers below read through
    :func:`_triggers`, which handles both.
    """
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    assert isinstance(loaded, dict)
    return loaded


def _triggers(workflow: dict[Any, Any]) -> dict[str, Any]:
    triggers = workflow.get(True, workflow.get("on"))
    assert isinstance(triggers, dict), "ci.yml has no parsable `on:` block"
    return triggers


class TestPushToDevTrigger:
    def test_push_trigger_includes_dev(self) -> None:
        """A merge to `dev` must produce a run on `dev`'s own head (AC1)."""
        branches = _triggers(_load_workflow(CI_WORKFLOW))["push"]["branches"]
        assert "dev" in branches, (
            "ci.yml does not trigger on a push to `dev`, so a merge to `dev` "
            "never runs this suite against `dev`'s own post-merge head. A "
            "break formed by the union of two individually green PRs is then "
            "invisible until an unrelated PR picks the new head up as its "
            "merge base -- measured twice on 2026-09-19, 30m57s and ~1h to "
            f"notice. Got: {branches!r}"
        )

    def test_push_trigger_still_includes_main(self) -> None:
        """The `main` push lane is widened, never traded away."""
        branches = _triggers(_load_workflow(CI_WORKFLOW))["push"]["branches"]
        assert "main" in branches, f"ci.yml stopped running on `main`: {branches!r}"


class TestPushRunIsGreenCapable:
    """The push run must be able to pass, or the signal gets muted.

    A trigger that produces a guaranteed-red run on every merge is not a
    monitoring surface. It is an alarm nobody reads, and then it is a reverted
    commit.
    """

    def test_contract_sync_gate_carries_no_event_gate(self) -> None:
        """AC2 -- the job that was cascading.

        Its own step resolves an empty PR number on a non-PR event and exits 0
        by its own guard, so the event gate bought nothing and cost the whole
        push lane.
        """
        condition = str(_load_workflow(CI_WORKFLOW)["jobs"]["contract-sync-gate"]["if"])
        assert "github.event_name" not in condition, (
            "contract-sync-gate is event-gated again. On a push event it will "
            "report `skipped`, and `detect-changes` needs it with no `if:`, so "
            "the skip cascades to `test-parallel` and the STRICT `CI Tests "
            f"Gate` FAILS. Got: {condition!r}"
        )

    def test_no_job_detect_changes_needs_is_event_gated(self) -> None:
        """The cascade property itself, over all sixteen dependencies.

        This is the general form of the defect, not the instance of it. A
        ``needs:`` edge from ``detect-changes`` to any job whose ``if:``
        excludes the running event makes GitHub skip ``detect-changes`` — and
        the poller never gets a say, because ``needs:`` resolution happens
        first.
        """
        jobs = _load_workflow(CI_WORKFLOW)["jobs"]
        needs = jobs["detect-changes"]["needs"]
        assert len(needs) >= 10, (
            "detect-changes lost most of its dependencies; this test's premise "
            f"no longer holds. Got: {needs!r}"
        )
        offenders = {
            name: str(jobs[name]["if"])
            for name in needs
            if "github.event_name" in str(jobs[name].get("if", ""))
        }
        assert not offenders, (
            "a job `detect-changes` depends on is gated on `github.event_name`. "
            "On an event its condition excludes, that job reports `skipped`, "
            "`detect-changes` skips with it, `test-parallel` skips, and the "
            "STRICT `CI Tests Gate` fails -- which is exactly how every push "
            "run on `main` concluded `failure` before OMN-18835. Gate on the "
            "event INSIDE the job's own steps, the way contract-sync-gate "
            f"already does. Offenders: {offenders!r}"
        )

    def test_detect_changes_handles_a_non_pull_request_event(self) -> None:
        """Its diff step must have a non-PR arm, or the push run cannot select."""
        steps = _load_workflow(CI_WORKFLOW)["jobs"]["detect-changes"]["steps"]
        diff = next(s for s in steps if s.get("id") == "diff")
        body = str(diff["run"])
        assert "HEAD~1" in body, (
            "detect-changes' diff step has no non-pull_request arm, so a push "
            f"run cannot compute a change set. Got: {body!r}"
        )


class TestRuntimeBootSmokeStaysDormantOnDev:
    """AC5 -- the one job deliberately NOT widened.

    It has concluded ``skipped`` on every observable run and has never once
    executed, so a dev-push trigger would wake an unproven Docker-compose boot
    on roughly fifty merges a day with no evidence it can pass. A monitor that
    is red for a reason unrelated to the break it exists to catch gets muted.
    """

    def test_does_not_fire_on_a_dev_push(self) -> None:
        condition = str(_load_workflow(CI_WORKFLOW)["jobs"]["runtime-boot-smoke"]["if"])
        assert "refs/heads/main" in condition, (
            "runtime-boot-smoke no longer pins its push arm to `main`, so it "
            "now fires on every merge to `dev` -- a compose boot this "
            "repository has never once seen execute. Widening it is fine, but "
            "it owes one green run as evidence first. Got: "
            f"{condition!r}"
        )
        assert "github.event_name != 'pull_request'" not in condition, (
            "runtime-boot-smoke is back on the blanket non-PR condition, which "
            "now includes a push to `dev`. Got: "
            f"{condition!r}"
        )


class TestPullRequestSemanticsAreUnchanged:
    """AC4 -- widening the push lane must not narrow the PR lane."""

    def test_pull_request_branches_unchanged(self) -> None:
        branches = _triggers(_load_workflow(CI_WORKFLOW))["pull_request"]["branches"]
        assert sorted(branches) == ["dev", "main"], (
            f"ci.yml's pull_request branches changed: {branches!r}"
        )

    def test_pull_request_types_unchanged(self) -> None:
        """OMN-16171 keeps `edited` out; OMN-16216 keeps three others in."""
        types = _triggers(_load_workflow(CI_WORKFLOW))["pull_request"]["types"]
        assert "edited" not in types, (
            "`edited` is back in ci.yml's pull_request types; OMN-16171 removed "
            "it because every PR-body edit restarted the whole ~48-job matrix"
        )
        for required in ("labeled", "unlabeled", "ready_for_review"):
            assert required in types, (
                f"ci.yml pull_request.types lost {required!r} (OMN-16216)"
            )

    def test_external_contexts_are_asserted_on_pull_request_only(self) -> None:
        """The reason a push run cannot acquire the OMN-17181 shape.

        ``omniweb``'s push CI was permanently red because ``CI Summary``
        asserted two external contexts that can only ever mint on a
        ``pull_request`` event. This repository's gate passes that tuple only
        when ``--event-name`` is ``pull_request``, which is what makes a push
        run green-capable at all — so the property is pinned here rather than
        left as a paragraph in a PR body.
        """
        source = CI_SUMMARY_GATE.read_text(encoding="utf-8")
        assert (
            'EXPECTED_EXTERNAL_CONTEXTS if args.event_name == "pull_request" else ()'
            in source
        ), (
            "ci_summary_gate.py no longer restricts EXPECTED_EXTERNAL_CONTEXTS "
            "to pull_request events. Every push run on `dev` will now fail "
            "closed on contexts that can never mint on a push -- the OMN-17181 "
            "shape that made omniweb's push CI red on 40 consecutive runs."
        )
