# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18352 — the occ-preflight heal must not race the CI matrix.

The defect, live on ``omnibase_infra#3503`` and reproduced on ``#3505``
-----------------------------------------------------------------------
``occ-preflight-heal.yml`` fires on ``pull_request: edited`` and calls
``gh run rerun --failed`` **only if** the target ``CI`` run's own ``status``
already reads ``completed``. The OCC autobind stamp lands on the PR body
roughly a minute after the initiating push, while the ~48-job matrix is still
``in_progress``. So on #3503 the heal job ran at 20:53:03-20:53:08Z, evaluated
its guard cleanly, printed "nothing to do", and concluded ``success`` — a
success that means "the guard ran", not "the preflight was healed".

Nothing re-invoked it afterwards: a ``gh run rerun`` does not emit ``edited``,
and the CI run reaching ``completed`` was not an event this workflow listened
to. ``occ-preflight / eligibility`` and therefore ``CI Summary`` stayed frozen
on the pre-stamp ``FAILURE`` until a human ran ``gh run rerun --failed`` on run
``34782083407`` at 21:21:21Z (``run_attempt: 2``, ``triggering_actor:
jonahgabriel``) — 28 minutes after the run it targeted had finished.

The fix, at cause
-----------------
"Run not yet completed" is not a terminal state, it is a *not yet*. The heal is
therefore invoked at the moment its precondition becomes true — on the CI run's
own ``workflow_run: completed`` event — **in addition to** the body edit, which
still covers a stamp that lands after the run has already finished. Between
them the two invocation points cover every ordering of (stamp lands) and (run
completes), which is what makes this a removal of the race rather than a
narrowing of the window. A bounded in-job poll was rejected: the CI matrix runs
for tens of minutes and the heal job has a 5-minute timeout, so a poll would
hold a runner for the whole matrix and *still* lose the race whenever the
matrix outran the ceiling.

Being re-invoked on run completion makes the guard recursion-exposed for the
first time — a heal-issued rerun completes, which is itself a
``workflow_run: completed``. Two independent conditions bound it:

* **staleness**: rerun only when the failed preflight job STARTED BEFORE the
  PR's ``updated_at``, i.e. the recorded verdict predates the PR state it is
  supposed to describe. After a heal the new preflight job starts later than
  ``updated_at``, so the next completion evaluates to "not stale" and stops.
  This also stops the heal from re-queueing a preflight that failed for a
  reason no body edit touched.
* **attempt ceiling**: a hard cap on ``run_attempt`` as a backstop, so an
  unforeseen re-trigger loop terminates even if the timestamp reasoning above
  is ever wrong.

The guard is extracted out of workflow YAML into ``scripts/ci`` so it is
testable at all, the same shape ``ci_summary_gate.py`` already uses.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from scripts.ci.occ_preflight_heal import (
    CiRunSnapshot,
    EnumHealOutcome,
    HealDecision,
    decide_heal,
    main,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
HEAL_WORKFLOW = WORKFLOWS_DIR / "occ-preflight-heal.yml"

REPO = "OmniNode-ai/omnibase_infra"
HEAD_SHA = "fe5db3856fe17fd96297be60732f9eb9d8742ef1"
RUN_ID = 34782083407
PR_NUMBER = 3503


def _ts(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)


# The #3503 timeline, to the second, from the Actions API.
PUSH_AT = _ts("2026-09-13T20:52:05Z")
PREFLIGHT_STARTED_AT = _ts("2026-09-13T20:52:09Z")
STAMP_EDIT_AT = _ts("2026-09-13T20:53:00Z")
HEAL_FIRED_AT = _ts("2026-09-13T20:53:03Z")
CI_COMPLETED_AT = _ts("2026-09-13T20:53:40Z")


class FakeGh:
    """Deterministic stand-in for the real ``gh`` calls.

    Records every rerun it is asked to issue so idempotency is asserted on
    observed calls, not on a return value the module could get wrong.
    """

    def __init__(
        self,
        *,
        runs: dict[int, CiRunSnapshot] | None = None,
        latest_by_sha: dict[str, CiRunSnapshot | None] | None = None,
        preflight_started: dict[int, datetime | None] | None = None,
        pr_updated: dict[int, datetime | None] | None = None,
        pr_numbers: dict[str, int | None] | None = None,
    ) -> None:
        self._runs = runs or {}
        self._latest_by_sha = latest_by_sha or {}
        self._preflight_started = preflight_started or {}
        self._pr_updated = pr_updated or {}
        self._pr_numbers = pr_numbers or {}
        self.reruns: list[tuple[str, int]] = []

    def latest_ci_run(
        self, *, repo: str, head_sha: str, workflow_name: str
    ) -> CiRunSnapshot | None:
        assert repo == REPO
        assert workflow_name == "CI"
        return self._latest_by_sha.get(head_sha)

    def run_snapshot(self, *, repo: str, run_id: int) -> CiRunSnapshot | None:
        assert repo == REPO
        return self._runs.get(run_id)

    def earliest_failed_preflight_started_at(
        self, *, repo: str, run_id: int, job_prefix: str
    ) -> datetime | None:
        assert repo == REPO
        assert job_prefix == "occ-preflight"
        return self._preflight_started.get(run_id)

    def pr_updated_at(self, *, repo: str, pr_number: int) -> datetime | None:
        assert repo == REPO
        return self._pr_updated.get(pr_number)

    def resolve_pr_number(self, *, repo: str, head_sha: str) -> int | None:
        assert repo == REPO
        return self._pr_numbers.get(head_sha)

    def rerun_failed(self, *, repo: str, run_id: int) -> None:
        self.reruns.append((repo, run_id))


# ---------------------------------------------------------------------------
# The race itself
# ---------------------------------------------------------------------------


def test_stamp_edit_while_the_run_is_in_progress_cannot_heal_from_the_edit() -> None:
    """#3503, at 20:53:03Z: the edit fires the heal, the run is ``in_progress``.

    This is the no-op the ticket reports, and it stays a no-op — no rerun of a
    single failed job is even possible inside a run that has not finished. The
    test pins that the guard SAYS SO with a distinguishable outcome rather than
    reporting the same "nothing to do" it reports when there was genuinely
    nothing to heal. The fix is not to make this case rerun; it is that this
    case is no longer the only invocation point (see the next test).
    """
    decision = decide_heal(
        run=CiRunSnapshot(
            run_id=RUN_ID, status="in_progress", run_attempt=1, head_sha=HEAD_SHA
        ),
        earliest_failed_preflight_started_at=PREFLIGHT_STARTED_AT,
        pr_updated_at=STAMP_EDIT_AT,
    )
    assert decision.outcome is EnumHealOutcome.RUN_NOT_COMPLETED
    assert decision.rerun is False
    assert decision.run_id == RUN_ID


def test_the_same_run_completing_heals_without_a_second_human_action() -> None:
    """OMN-18352 AC1, the whole point: the completion event picks it up.

    Same PR, same run, same stale verdict — only the invocation point differs.
    On #3503 nothing happened here and a human ran ``gh run rerun --failed`` 28
    minutes later.
    """
    decision = decide_heal(
        run=CiRunSnapshot(
            run_id=RUN_ID, status="completed", run_attempt=1, head_sha=HEAD_SHA
        ),
        earliest_failed_preflight_started_at=PREFLIGHT_STARTED_AT,
        pr_updated_at=STAMP_EDIT_AT,
    )
    assert decision.outcome is EnumHealOutcome.RERUN_REQUIRED
    assert decision.rerun is True
    assert decision.run_id == RUN_ID


def test_main_issues_exactly_one_rerun_for_the_3503_timeline() -> None:
    """End-to-end through ``main`` on the run-id path the completion event uses."""
    gh = FakeGh(
        runs={
            RUN_ID: CiRunSnapshot(
                run_id=RUN_ID, status="completed", run_attempt=1, head_sha=HEAD_SHA
            )
        },
        preflight_started={RUN_ID: PREFLIGHT_STARTED_AT},
        pr_updated={PR_NUMBER: STAMP_EDIT_AT},
    )
    exit_code = main(
        [
            "--repo",
            REPO,
            "--run-id",
            str(RUN_ID),
            "--pr-number",
            str(PR_NUMBER),
        ],
        gh=gh,
    )
    assert exit_code == 0
    assert gh.reruns == [(REPO, RUN_ID)]


def test_main_resolves_the_pr_from_the_head_sha_when_none_is_supplied() -> None:
    """``workflow_run`` payloads carry ``pull_requests`` only sometimes.

    The run-completion path must be able to find the PR from the head SHA, or
    it silently degrades to the unresolved branch on exactly the events it
    exists to serve.
    """
    gh = FakeGh(
        runs={
            RUN_ID: CiRunSnapshot(
                run_id=RUN_ID, status="completed", run_attempt=1, head_sha=HEAD_SHA
            )
        },
        preflight_started={RUN_ID: PREFLIGHT_STARTED_AT},
        pr_updated={PR_NUMBER: STAMP_EDIT_AT},
        pr_numbers={HEAD_SHA: PR_NUMBER},
    )
    assert (
        main(["--repo", REPO, "--run-id", str(RUN_ID), "--head-sha", HEAD_SHA], gh=gh)
        == 0
    )
    assert gh.reruns == [(REPO, RUN_ID)]


def test_main_on_the_edited_path_finds_the_run_by_head_sha() -> None:
    """The ``edited`` invocation still works, and still heals a completed run."""
    gh = FakeGh(
        latest_by_sha={
            HEAD_SHA: CiRunSnapshot(
                run_id=RUN_ID, status="completed", run_attempt=1, head_sha=HEAD_SHA
            )
        },
        preflight_started={RUN_ID: PREFLIGHT_STARTED_AT},
        pr_updated={PR_NUMBER: STAMP_EDIT_AT},
    )
    assert (
        main(
            [
                "--repo",
                REPO,
                "--head-sha",
                HEAD_SHA,
                "--pr-number",
                str(PR_NUMBER),
            ],
            gh=gh,
        )
        == 0
    )
    assert gh.reruns == [(REPO, RUN_ID)]


def test_a_run_id_alone_resolves_its_own_pr() -> None:
    """The completion and dispatch paths may know only the run id.

    ``workflow_run`` payloads carry ``pull_requests`` only sometimes, and a
    ``workflow_dispatch`` of this heal carries no PR at all. Falling back to the
    run's OWN head SHA is what stops the guard reporting "no pull request" on
    exactly the events it exists to serve.
    """
    gh = FakeGh(
        runs={
            RUN_ID: CiRunSnapshot(
                run_id=RUN_ID, status="completed", run_attempt=1, head_sha=HEAD_SHA
            )
        },
        preflight_started={RUN_ID: PREFLIGHT_STARTED_AT},
        pr_updated={PR_NUMBER: STAMP_EDIT_AT},
        pr_numbers={HEAD_SHA: PR_NUMBER},
    )
    assert main(["--repo", REPO, "--run-id", str(RUN_ID)], gh=gh) == 0
    assert gh.reruns == [(REPO, RUN_ID)]


# ---------------------------------------------------------------------------
# Idempotency and recursion bounds
# ---------------------------------------------------------------------------


def test_a_second_pass_after_the_heal_does_not_rerun_again() -> None:
    """The heal-issued rerun completes, which is another ``workflow_run``.

    Attempt 2's preflight job started AFTER the stamp edit, so its verdict
    describes the current body and there is nothing stale left to clear. This
    is what stops the completion trigger from feeding itself.
    """
    gh = FakeGh(
        runs={
            RUN_ID: CiRunSnapshot(
                run_id=RUN_ID, status="completed", run_attempt=2, head_sha=HEAD_SHA
            )
        },
        preflight_started={RUN_ID: _ts("2026-09-13T21:22:00Z")},
        pr_updated={PR_NUMBER: STAMP_EDIT_AT},
    )
    assert (
        main(
            ["--repo", REPO, "--run-id", str(RUN_ID), "--pr-number", str(PR_NUMBER)],
            gh=gh,
        )
        == 0
    )
    assert gh.reruns == []


def test_a_verdict_newer_than_the_pr_state_is_a_real_failure_not_a_stale_one() -> None:
    """No edit ever happened: the preflight read the body it is failing on.

    Re-queueing it would burn CI to reproduce the same verdict, which is the
    behaviour the original ``edited``-only trigger got for free and which the
    completion trigger would otherwise lose.
    """
    decision = decide_heal(
        run=CiRunSnapshot(
            run_id=RUN_ID, status="completed", run_attempt=1, head_sha=HEAD_SHA
        ),
        earliest_failed_preflight_started_at=PREFLIGHT_STARTED_AT,
        pr_updated_at=PUSH_AT,
    )
    assert decision.outcome is EnumHealOutcome.VERDICT_NOT_STALE
    assert decision.rerun is False


def test_the_attempt_ceiling_terminates_an_unforeseen_loop() -> None:
    """Backstop, independent of the timestamp reasoning.

    If the staleness condition were ever wrong in the direction that keeps
    saying "stale", this is what stops the heal re-triggering itself forever.
    """
    over_ceiling = decide_heal(
        run=CiRunSnapshot(
            run_id=RUN_ID, status="completed", run_attempt=99, head_sha=HEAD_SHA
        ),
        earliest_failed_preflight_started_at=PREFLIGHT_STARTED_AT,
        pr_updated_at=STAMP_EDIT_AT,
    )
    assert over_ceiling.outcome is EnumHealOutcome.ATTEMPT_CEILING
    assert over_ceiling.rerun is False

    # Positive control for the same input one attempt below the ceiling: the
    # ceiling must be what refused it, not some other condition in the chain.
    at_ceiling = decide_heal(
        run=CiRunSnapshot(
            run_id=RUN_ID, status="completed", run_attempt=1, head_sha=HEAD_SHA
        ),
        earliest_failed_preflight_started_at=PREFLIGHT_STARTED_AT,
        pr_updated_at=STAMP_EDIT_AT,
        max_run_attempt=1,
    )
    assert at_ceiling.outcome is EnumHealOutcome.RERUN_REQUIRED


# ---------------------------------------------------------------------------
# Nothing-to-heal and fail-loud branches
# ---------------------------------------------------------------------------


def test_no_ci_run_for_the_head_sha_is_a_clean_no_op() -> None:
    gh = FakeGh(latest_by_sha={HEAD_SHA: None})
    assert (
        main(
            ["--repo", REPO, "--head-sha", HEAD_SHA, "--pr-number", str(PR_NUMBER)],
            gh=gh,
        )
        == 0
    )
    assert gh.reruns == []


def test_a_run_with_no_failed_preflight_job_is_not_touched() -> None:
    """An unrelated body edit must not re-queue genuine test failures."""
    gh = FakeGh(
        runs={
            RUN_ID: CiRunSnapshot(
                run_id=RUN_ID, status="completed", run_attempt=1, head_sha=HEAD_SHA
            )
        },
        preflight_started={RUN_ID: None},
        pr_updated={PR_NUMBER: STAMP_EDIT_AT},
    )
    assert (
        main(
            ["--repo", REPO, "--run-id", str(RUN_ID), "--pr-number", str(PR_NUMBER)],
            gh=gh,
        )
        == 0
    )
    assert gh.reruns == []


def test_an_unresolvable_pr_state_fails_loud_rather_than_silently_skipping() -> None:
    """The failure mode this whole ticket is about is SILENCE.

    A heal that cannot read the PR it is deciding about must go red so somebody
    sees it, not conclude ``success`` meaning "the guard ran cleanly" — which is
    exactly how #3503 read as healthy while nothing had been healed.
    """
    gh = FakeGh(
        runs={
            RUN_ID: CiRunSnapshot(
                run_id=RUN_ID, status="completed", run_attempt=1, head_sha=HEAD_SHA
            )
        },
        preflight_started={RUN_ID: PREFLIGHT_STARTED_AT},
        pr_updated={PR_NUMBER: None},
    )
    assert (
        main(
            ["--repo", REPO, "--run-id", str(RUN_ID), "--pr-number", str(PR_NUMBER)],
            gh=gh,
        )
        == 1
    )
    assert gh.reruns == []


def test_dry_run_decides_without_issuing_the_rerun() -> None:
    gh = FakeGh(
        runs={
            RUN_ID: CiRunSnapshot(
                run_id=RUN_ID, status="completed", run_attempt=1, head_sha=HEAD_SHA
            )
        },
        preflight_started={RUN_ID: PREFLIGHT_STARTED_AT},
        pr_updated={PR_NUMBER: STAMP_EDIT_AT},
    )
    assert (
        main(
            [
                "--repo",
                REPO,
                "--run-id",
                str(RUN_ID),
                "--pr-number",
                str(PR_NUMBER),
                "--dry-run",
            ],
            gh=gh,
        )
        == 0
    )
    assert gh.reruns == []


def test_decision_outcomes_are_distinguishable() -> None:
    """A single "nothing to do" string is what hid this defect for a month.

    Every refusal branch must carry its own outcome value so a heal job's log
    says which one fired.
    """
    values = {outcome.value for outcome in EnumHealOutcome}
    assert len(values) == len(list(EnumHealOutcome))
    assert HealDecision(
        outcome=EnumHealOutcome.RERUN_REQUIRED, run_id=RUN_ID, detail="x"
    ).rerun


# ---------------------------------------------------------------------------
# The workflow must actually be wired to the completion event
# ---------------------------------------------------------------------------


def _on_block(path: Path) -> dict[str, object]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict), f"{path} did not parse to a mapping"
    # PyYAML 1.1 resolves an unquoted `on:` key to the boolean True.
    on_block = loaded.get("on")
    if on_block is None:
        on_block = loaded.get(True)
    assert isinstance(on_block, dict), f"{path} has no on: mapping"
    return on_block


def test_healer_listens_for_the_ci_run_completing() -> None:
    """AC1's mechanism, pinned against the live file.

    Without this trigger the ONLY invocation point is the body edit, and a
    stamp landing mid-matrix is unhealable — the #3503 defect verbatim.
    """
    on_block = _on_block(HEAL_WORKFLOW)
    workflow_run = on_block.get("workflow_run")
    assert isinstance(workflow_run, dict), (
        "occ-preflight-heal.yml has no `workflow_run` trigger, so nothing "
        "invokes the heal when the CI run it targets finishes. A stamp that "
        "lands while the matrix is in flight then needs a human "
        "`gh run rerun --failed` (OMN-18352, live on omnibase_infra#3503)."
    )
    assert workflow_run.get("types") == ["completed"], (
        "the heal must fire on the CI run's COMPLETION -- that is the moment "
        "its `status == completed` precondition becomes true"
    )
    ci_name = yaml.safe_load((WORKFLOWS_DIR / "ci.yml").read_text(encoding="utf-8"))[
        "name"
    ]
    assert workflow_run.get("workflows") == [ci_name], (
        f"the heal must listen to the workflow literally named {ci_name!r} "
        f"(ci.yml's `name:`); a mismatch disables it silently"
    )


def test_healer_keeps_the_edited_trigger_for_post_completion_stamps() -> None:
    """Both invocation points are load-bearing; neither replaces the other.

    A stamp that lands AFTER the run finished produces no `workflow_run`
    event, so dropping `edited` would trade one half of the race for the other.
    """
    pull_request = _on_block(HEAL_WORKFLOW).get("pull_request")
    assert isinstance(pull_request, dict)
    assert pull_request.get("types") == ["edited"]


def test_healer_concurrency_key_is_defined_on_the_completion_event() -> None:
    """``github.event.pull_request.number`` is EMPTY on a ``workflow_run``.

    With ``cancel-in-progress: true`` an empty key collapses every concurrent
    completion-triggered heal in the repo into one group, so heals cancel each
    other and the last PR to finish is the only one served.
    """
    loaded = yaml.safe_load(HEAL_WORKFLOW.read_text(encoding="utf-8"))
    group = str(loaded["concurrency"]["group"])
    assert "github.event.workflow_run" in group, (
        "the concurrency group must include a workflow_run-derived "
        f"discriminator; got {group!r}"
    )


def test_healer_delegates_its_guard_to_the_tested_script() -> None:
    """The guard is only testable because it is not YAML any more.

    Pin the delegation so a future edit cannot quietly re-inline shell that no
    test can reach.
    """
    body = HEAL_WORKFLOW.read_text(encoding="utf-8")
    assert "scripts/ci/occ_preflight_heal.py" in body, (
        "occ-preflight-heal.yml must call scripts/ci/occ_preflight_heal.py; "
        "an inlined shell guard is untestable, which is how the race shipped"
    )
