# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Fail-closed resolution of the PR whose DoD ``check_values`` this repo gates on.

OMN-16508. ``ci.yml``'s ``contract-compliance`` job runs
``run_contract_compliance_check.py --pr <n>``. That ``<n>`` comes from
``github.event.pull_request.number``, which only the ``pull_request`` event
populates -- but the job's ``if:`` also admits ``push`` and ``merge_group``.
The pre-fix step handled the empty case with ``exit 0``: a **vacuous pass**.

Why the vacuous pass mattered: ``"Contract Compliance Check"`` is a
``GATE_JOBS`` entry in :mod:`scripts.ci.ci_summary_gate`, and ``CI Summary`` is
the sole required branch-protection context on this repo. An ``exit 0`` that
evaluated zero ``check_values`` was therefore counted by the umbrella poller as
a *provable* pass. The failure mode is invisible -- a green gate that checked
nothing never prompts investigation, unlike a red one.

Blast radius, stated precisely (the ticket body overstated it as "every dev
push"): this workflow's ``on.push.branches`` is ``[main]``, so it never runs on
a ``dev`` commit at all. The two live fail-open events are

* ``push`` to ``main`` -- the release-synced fast-forward commits, and
* ``merge_group`` -- currently unexercised (no repo in the registry has a queue
  on ``dev`` as of 2026-08-24) but a latent bypass of the repo's only required
  context the instant a queue is re-enabled, with zero further code change.

The fix is resolution, not narrowing. A job-level ``if:`` that excluded
``push``/``merge_group`` would publish a ``skipped`` check run, which branch
protection counts as passing -- the same skip-vector class the
``reject-required-check-skip-vector`` pre-commit hook (OMN-14863) exists to
catch, and which ``ci_summary_gate``'s external-context assertion treats as a
hard failure. So the job keeps running on all three events and instead resolves
which PR's ``check_values`` to evaluate:

1. ``pull_request`` -- use ``github.event.pull_request.number`` verbatim
   (byte-identical to the pre-fix path; the resolver never shells out).
2. ``merge_group`` -- parse the queue ref. GitHub names it
   ``refs/heads/gh-readonly-queue/<base>/pr-<n>-<sha>``, so the PR number is
   carried in the ref itself and needs no API call.
3. ``push`` -- ask ``gh api repos/{repo}/commits/{sha}/pulls``; GitHub
   associates a squash-merge commit with its source PR. Only a **merged** PR
   counts, and a PR whose ``merge_commit_sha`` equals the pushed SHA is
   preferred over a merely-associated one.
4. Anything else -- ``exit 1``. Fail closed, never a pass on zero checks.

Every unresolvable state is an exit-1 here, including API failure: a transient
``gh`` error must not become a green gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci.resolve_contract_compliance_pr import (
    EXIT_FAIL,
    EXIT_PASS,
    main,
    parse_merge_queue_ref,
    resolve_pr_number,
)

pytestmark = pytest.mark.unit

REPO = "OmniNode-ai/omnibase_infra"
REPO_ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


class FakeCommitPulls:
    """Deterministic stand-in for the ``gh api .../commits/{sha}/pulls`` call."""

    def __init__(self, payload: list[dict[str, Any]] | None, *, fails: bool = False):
        self._payload = payload
        self._fails = fails
        self.calls: list[tuple[str, str]] = []

    def __call__(self, repo: str, sha: str) -> list[dict[str, Any]]:
        self.calls.append((repo, sha))
        if self._fails:
            raise RuntimeError("gh api exploded")
        assert self._payload is not None
        return self._payload


def _merged(number: int, merge_commit_sha: str = "") -> dict[str, Any]:
    return {
        "number": number,
        "merged_at": "2026-08-25T00:00:00Z",
        "merge_commit_sha": merge_commit_sha,
    }


def _open(number: int) -> dict[str, Any]:
    return {"number": number, "merged_at": None, "merge_commit_sha": ""}


# ---------------------------------------------------------------------------
# pull_request: unchanged path, no API call
# ---------------------------------------------------------------------------


def test_pull_request_number_is_used_verbatim_without_any_api_call() -> None:
    fetch = FakeCommitPulls([])
    assert (
        resolve_pr_number(
            repo=REPO,
            pr_number="2915",
            merge_group_ref="",
            sha="deadbeef",
            fetch_commit_pulls=fetch,
        )
        == 2915
    )
    assert fetch.calls == [], (
        "the pull_request path must stay byte-identical in behaviour to pre-fix: "
        "PR_NUMBER is already populated, so no network call may be introduced"
    )


# ---------------------------------------------------------------------------
# merge_group: the queue ref carries the PR number
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("refs/heads/gh-readonly-queue/dev/pr-2915-8ce11d5181289cbee6a9", 2915),
        ("gh-readonly-queue/dev/pr-17-abc0123", 17),
        ("refs/heads/gh-readonly-queue/main/pr-1-0000000", 1),
    ],
)
def test_merge_queue_ref_yields_the_source_pr_number(ref: str, expected: int) -> None:
    assert parse_merge_queue_ref(ref) == expected


@pytest.mark.parametrize(
    "ref",
    [
        "",
        "refs/heads/dev",
        "refs/heads/gh-readonly-queue/dev/nope-2915-abc",
        "refs/heads/gh-readonly-queue/dev/pr--abc",
    ],
)
def test_non_queue_refs_resolve_to_nothing(ref: str) -> None:
    assert parse_merge_queue_ref(ref) is None


def test_merge_group_resolves_from_the_ref_without_an_api_call() -> None:
    fetch = FakeCommitPulls([])
    assert (
        resolve_pr_number(
            repo=REPO,
            pr_number="",
            merge_group_ref="refs/heads/gh-readonly-queue/dev/pr-2915-8ce11d5",
            sha="8ce11d5",
            fetch_commit_pulls=fetch,
        )
        == 2915
    )
    assert fetch.calls == []


# ---------------------------------------------------------------------------
# push: resolve the merge commit's originating PR
# ---------------------------------------------------------------------------


def test_push_resolves_the_merged_source_pr_for_the_commit() -> None:
    fetch = FakeCommitPulls([_merged(2923)])
    assert (
        resolve_pr_number(
            repo=REPO,
            pr_number="",
            merge_group_ref="",
            sha="8ce11d5181289cbee6a9",
            fetch_commit_pulls=fetch,
        )
        == 2923
    )
    assert fetch.calls == [(REPO, "8ce11d5181289cbee6a9")]


def test_push_prefers_the_pr_whose_merge_commit_is_this_sha() -> None:
    """A commit can be *associated* with several PRs; only one produced it."""
    sha = "8ce11d5181289cbee6a9"
    fetch = FakeCommitPulls([_merged(100), _merged(2923, merge_commit_sha=sha)])
    assert (
        resolve_pr_number(
            repo=REPO,
            pr_number="",
            merge_group_ref="",
            sha=sha,
            fetch_commit_pulls=fetch,
        )
        == 2923
    )


def test_push_ignores_unmerged_associations() -> None:
    """An OPEN PR containing the commit never evaluated it as a merge result."""
    fetch = FakeCommitPulls([_open(3000), _merged(2923)])
    assert (
        resolve_pr_number(
            repo=REPO,
            pr_number="",
            merge_group_ref="",
            sha="abc",
            fetch_commit_pulls=fetch,
        )
        == 2923
    )


# ---------------------------------------------------------------------------
# fail-closed: every unresolvable state, including transient API failure
# ---------------------------------------------------------------------------


def test_push_with_no_merged_pr_fails_closed() -> None:
    fetch = FakeCommitPulls([_open(3000)])
    assert (
        resolve_pr_number(
            repo=REPO,
            pr_number="",
            merge_group_ref="",
            sha="abc",
            fetch_commit_pulls=fetch,
        )
        is None
    )


def test_push_with_no_associated_pr_at_all_fails_closed() -> None:
    fetch = FakeCommitPulls([])
    assert (
        resolve_pr_number(
            repo=REPO,
            pr_number="",
            merge_group_ref="",
            sha="abc",
            fetch_commit_pulls=fetch,
        )
        is None
    )


def test_api_failure_fails_closed_rather_than_passing() -> None:
    """A transient `gh` error must never be indistinguishable from a green gate."""
    fetch = FakeCommitPulls(None, fails=True)
    assert (
        resolve_pr_number(
            repo=REPO,
            pr_number="",
            merge_group_ref="",
            sha="abc",
            fetch_commit_pulls=fetch,
        )
        is None
    )


def test_no_sha_and_no_pr_number_fails_closed() -> None:
    fetch = FakeCommitPulls([])
    assert (
        resolve_pr_number(
            repo=REPO,
            pr_number="",
            merge_group_ref="",
            sha="",
            fetch_commit_pulls=fetch,
        )
        is None
    )
    assert fetch.calls == []


# ---------------------------------------------------------------------------
# CLI contract: stdout carries the number, exit status carries the verdict
# ---------------------------------------------------------------------------


def test_main_prints_the_number_and_exits_zero_when_resolved(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        ["--repo", REPO, "--pr-number", "2915", "--sha", "abc"],
        fetch_commit_pulls=FakeCommitPulls([]),
    )
    assert code == EXIT_PASS
    assert capsys.readouterr().out.strip() == "2915"


def test_main_exits_nonzero_and_prints_nothing_when_unresolved(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        ["--repo", REPO, "--pr-number", "", "--sha", "abc", "--event-name", "push"],
        fetch_commit_pulls=FakeCommitPulls([_open(3000)]),
    )
    assert code == EXIT_FAIL
    captured = capsys.readouterr()
    assert captured.out.strip() == "", (
        "an unresolved run must emit no number on stdout -- the caller reads "
        "stdout as the --pr argument and would otherwise pass an empty string"
    )
    assert "::error::" in captured.err


# ---------------------------------------------------------------------------
# Workflow shape: the vacuous exit-0 must not come back
# ---------------------------------------------------------------------------


def _contract_compliance_steps() -> list[dict[str, Any]]:
    data = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    return [dict(step) for step in data["jobs"]["contract-compliance"]["steps"]]


def _only_step_running(marker: str) -> dict[str, Any]:
    steps = [
        step
        for step in _contract_compliance_steps()
        if marker in str(step.get("run", ""))
    ]
    assert len(steps) == 1, (
        f"expected exactly one contract-compliance step invoking {marker}, "
        f"found {len(steps)}"
    )
    return steps[0]


def _contract_compliance_dod_step() -> dict[str, Any]:
    return _only_step_running("run_contract_compliance_check.py")


def test_dod_step_no_longer_vacuously_passes_on_an_empty_pr_number() -> None:
    run = str(_contract_compliance_dod_step()["run"])
    assert "exit 0" not in run, (
        "the contract-compliance DoD step regained an `exit 0` (OMN-16508). "
        "This step's verdict is counted by ci_summary_gate.GATE_JOBS as a "
        "provable pass, so exiting 0 without running a single check_value "
        "reports a green required gate that evaluated nothing. Resolve the "
        "governing PR via scripts/ci/resolve_contract_compliance_pr.py and "
        "exit 1 when it cannot be resolved."
    )
    assert "exit 1" in run, "the unresolved branch must fail closed"


def test_job_resolves_the_pr_rather_than_reading_only_the_event() -> None:
    resolver = _only_step_running("resolve_contract_compliance_pr.py")
    env = resolver.get("env", {})
    assert "merge_group" in str(env.get("MERGE_GROUP_REF", "")), (
        "the resolver needs github.event.merge_group.head_ref to recover the "
        "PR number on merge_group without an API call"
    )
    assert "pull_request" in str(env.get("PR_NUMBER", "")), (
        "the pull_request path must keep reading the event's own number"
    )
    assert "GH_TOKEN" in env, "the resolver's gh api call needs a token"


def test_dod_step_consumes_the_resolved_pr_not_the_raw_event_number() -> None:
    """The whole point: the executor runs against the RESOLVED scope."""
    resolver_id = _only_step_running("resolve_contract_compliance_pr.py").get("id", "")
    assert resolver_id, "the resolver step needs an `id:` so its output is addressable"
    dod_pr = str(_contract_compliance_dod_step().get("env", {}).get("PR_NUMBER", ""))
    assert f"steps.{resolver_id}.outputs.pr_number" in dod_pr, (
        f"the DoD step still reads PR_NUMBER from the raw event ({dod_pr!r}). It "
        f"must consume steps.{resolver_id}.outputs.pr_number, or push and "
        f"merge_group runs go back to having no scope to evaluate."
    )
    assert "github.event.pull_request.number" not in dod_pr


def test_contract_compliance_job_still_runs_on_all_three_events() -> None:
    """Narrowing the `if:` would publish a `skipped` check run, which passes."""
    data = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    condition = str(data["jobs"]["contract-compliance"]["if"])
    for event in ("pull_request", "merge_group", "push"):
        assert event in condition, (
            f"contract-compliance no longer admits {event}. A job-level `if:` "
            f"that evaluates false publishes conclusion `skipped`, which branch "
            f"protection counts as passing -- converting this fix's fail-closed "
            f"exit 1 back into a silent bypass via the skip path (OMN-14863)."
        )
