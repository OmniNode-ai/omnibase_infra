# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16808 — the sweep must not say the same thing twice on the same ticket.

The autoclose sweep enumerates OCC companions by a bare ``now - lookback_hours``
window and keeps no cross-run state: ``seen_tickets`` is a local discarded when
``handle()`` returns, and every gap path called ``create_comment``
unconditionally under ``apply=True`` through a bare ``commentCreate`` mutation
that never read what was already on the ticket.

Under the shipped schedule (``cron: '*/30'``, lookback 2h) one merged companion
sits inside four consecutive windows, so a ticket that stays gapped accrues an
identical comment per run, forever. Proven empirically before this fix: OMN-16037
/ OMN-16373 / OMN-16757 / OMN-16759 were classified gap in BOTH run 33098307405
and run 33128661860.

This suite pins the read-before-write rule and its fail-closed direction:

* the same window scanned twice writes exactly ONE comment per (ticket, gap class,
  verdict) — the second run records SKIPPED_DUPLICATE_COMMENT;
* dedup is keyed on ticket + gap class + verdict fingerprint, NOT on the companion
  PR, so a second companion binding the same ticket with the same verdict is also
  suppressed;
* a CHANGED verdict is new information and still gets a comment;
* an unreadable comment history is an ERROR outcome and writes nothing — the sweep
  never guesses that it has not already commented;
* DRY-RUN stays zero-write on every one of those paths.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    HandlerEvidenceAutocloseSweep,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_evidence_autoclose_decision import (
    EnumEvidenceAutocloseDecision,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_evidence_autoclose_sweep_request import (
    ModelEvidenceAutocloseSweepRequest,
)

pytestmark = pytest.mark.unit

_OCC_REPO = "OmniNode-ai/onex_change_control"
_TICKET = "OMN-9999"
_ISSUE_ID = "issue-uuid-1"


def _merged_pr(number: int) -> dict[str, object]:
    recent = (datetime.now(tz=UTC) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "number": number,
        "html_url": f"https://github.com/{_OCC_REPO}/pull/{number}",
        "title": f"evidence({_TICKET}): OCC companion",
        "updated_at": recent,
        "merged_at": recent,
    }


_RECEIPT_SUMMARY_MODEL = (
    "omnibase_infra.cli.model_receipt_runtime_summary.ModelReceiptRuntimeSummary"
)
_DOD_VERIFY_STATE_MODEL = (
    "omnimarket.nodes.node_dod_verify.models.model_dod_verify_state.ModelDodVerifyState"
)


def _skill_result(
    *,
    total: int,
    verified: int,
    failed: int,
    behavior_proving: int = 1,
) -> dict[str, object]:
    """A ModelSkillResult shaped like ``onex skill dod_verify`` prints.

    OMN-16961: the CLI prints two arms and picks between them on the run's own
    outcome — a verified verdict lands FLAT on ``result`` with
    ``result_model: ModelDodVerifyState``; anything else is nested at
    ``result.terminal_payload`` under ``ModelReceiptRuntimeSummary``. Both are
    derived here from the verdict, exactly as ``receipt_mode`` derives them.
    """
    verdict: dict[str, object] = {
        "correlation_id": str(uuid4()),
        "ticket_id": _TICKET,
        "status": "verified" if failed == 0 else "failed",
        "dry_run": False,
        "checks": [],
        "total_checks": total,
        "verified_count": verified,
        "failed_count": failed,
        "skipped_count": 0,
        "superseded_count": 0,
        "behavior_proving_count": behavior_proving,
        "error_message": None,
    }
    if failed == 0:
        return {
            "skill_name": "dod_verify",
            "node_name": "node_dod_verify",
            "status": "success",
            "exit_code": 0,
            "result": verdict,
            "result_model": _DOD_VERIFY_STATE_MODEL,
        }
    return {
        "skill_name": "dod_verify",
        "node_name": "node_dod_verify",
        "status": "failed",
        "exit_code": 1,
        "result": {
            "workflow_result": "failed",
            "exit_code": 1,
            "terminal_payload": verdict,
        },
        "result_model": _RECEIPT_SUMMARY_MODEL,
    }


class _StatefulLinear:
    """A Linear double that REMEMBERS what the sweep posted.

    The production defect is invisible to a double that forgets: the old fake
    only appended to a write log and served no comment history back, so two
    runs looked identical to one. This one serves ``create_comment`` writes
    back through ``fetch_comment_bodies``, which is what makes a second run
    distinguishable from a first.
    """

    def __init__(
        self,
        *,
        description: str | None = None,
        comment_history_readable: bool = True,
    ) -> None:
        self.comments: list[tuple[str, str]] = []
        self.state_updates: list[tuple[str, str]] = []
        self.comment_history_readable = comment_history_readable
        self.fetch_comment_calls: list[str] = []
        self._description = description

    async def fetch_issue(self, ticket_id: str) -> dict[str, object] | None:
        return {
            "id": _ISSUE_ID,
            "identifier": _TICKET,
            "state": {"id": "s1", "name": "In Progress", "type": "started"},
            "labels": {"nodes": []},
            "team": {"id": "team-1"},
            "description": self._description,
        }

    async def fetch_done_state_id(self, team_id: str) -> str | None:
        return "state-done-id"

    async def update_issue_state(self, issue_id: str, state_id: str) -> bool:
        self.state_updates.append((issue_id, state_id))
        return True

    async def create_comment(self, issue_id: str, body: str) -> bool:
        self.comments.append((issue_id, body))
        return True

    async def fetch_comment_bodies(self, issue_id: str) -> tuple[str, ...] | None:
        self.fetch_comment_calls.append(issue_id)
        if not self.comment_history_readable:
            return None
        return tuple(body for target, body in self.comments if target == issue_id)


def _gh_fake(companions: list[dict[str, object]]):
    async def fake_run_gh_command(args: list[str], timeout: float):
        path = args[2]
        if "/files" in path:
            return [{"filename": f"contracts/{_TICKET}.yaml"}], ""
        if "page=1" in path:
            return companions, ""
        return [], ""

    return fake_run_gh_command


def _handler(
    skill_results: list[dict[str, object]],
    linear: _StatefulLinear,
    companions: list[dict[str, object]],
) -> HandlerEvidenceAutocloseSweep:
    """Handler whose dod_verify double serves ``skill_results`` in order.

    The last entry is reused once exhausted, so a single-verdict test passes a
    one-element list and both runs see the same verdict.
    """
    calls = {"n": 0}

    async def fake_dod_verify(ticket_id: str, cwd: str, timeout: float):
        index = min(calls["n"], len(skill_results) - 1)
        calls["n"] += 1
        return skill_results[index], 0, ""

    return HandlerEvidenceAutocloseSweep(
        linear_client=linear,  # type: ignore[arg-type]
        autoclose_disabled=False,
        run_gh_command=_gh_fake(companions),
        run_dod_verify_command=fake_dod_verify,
    )


def _request(*, apply: bool) -> ModelEvidenceAutocloseSweepRequest:
    return ModelEvidenceAutocloseSweepRequest(
        correlation_id=uuid4(),
        occ_repo=_OCC_REPO,
        lookback_hours=2,
        apply=apply,
    )


@pytest.mark.asyncio
class TestOmn16808CommentIdempotency:
    async def test_same_window_twice_posts_exactly_one_gap_comment(self) -> None:
        """The headline AC: one fixture window, two runs, one comment."""
        linear = _StatefulLinear()
        companions = [_merged_pr(7001)]
        handler = _handler(
            [_skill_result(total=6, verified=3, failed=3)], linear, companions
        )

        first = await handler.handle(_request(apply=True))
        second = await handler.handle(_request(apply=True))

        assert first.outcomes[0].decision == EnumEvidenceAutocloseDecision.GAP_POSTED
        assert first.outcomes[0].linear_comment_posted is True
        assert (
            second.outcomes[0].decision
            == EnumEvidenceAutocloseDecision.SKIPPED_DUPLICATE_COMMENT
        )
        assert second.outcomes[0].linear_comment_posted is False
        assert second.outcomes[0].applied is False
        assert len(linear.comments) == 1, linear.comments

    async def test_posted_comment_carries_a_stable_marker(self) -> None:
        linear = _StatefulLinear()
        handler = _handler(
            [_skill_result(total=6, verified=3, failed=3)], linear, [_merged_pr(7001)]
        )

        await handler.handle(_request(apply=True))

        body = linear.comments[0][1]
        assert "onex-autoclose-sweep" in body
        assert "class=gap_posted" in body
        assert "fingerprint=" in body

    async def test_dedup_holds_across_a_different_companion(self) -> None:
        """A second companion binding the same ticket must not re-say it.

        ``seen_tickets`` already suppresses this WITHIN one run. The failure
        mode is across runs: window N sees companion #7001, window N+1 sees
        #7002 bound to the same ticket with the same verdict. The dedup key is
        deliberately (ticket, gap class, verdict) and excludes the companion.
        """
        linear = _StatefulLinear()
        verdict = _skill_result(total=6, verified=3, failed=3)

        first = await _handler([verdict], linear, [_merged_pr(7001)]).handle(
            _request(apply=True)
        )
        second = await _handler([verdict], linear, [_merged_pr(7002)]).handle(
            _request(apply=True)
        )

        assert first.outcomes[0].decision == EnumEvidenceAutocloseDecision.GAP_POSTED
        assert (
            second.outcomes[0].decision
            == EnumEvidenceAutocloseDecision.SKIPPED_DUPLICATE_COMMENT
        )
        assert len(linear.comments) == 1, linear.comments

    async def test_changed_verdict_still_gets_a_comment(self) -> None:
        """3/6 -> 5/6 is new information and must not be swallowed."""
        linear = _StatefulLinear()
        handler = _handler(
            [
                _skill_result(total=6, verified=3, failed=3),
                _skill_result(total=6, verified=5, failed=1),
            ],
            linear,
            [_merged_pr(7001)],
        )

        first = await handler.handle(_request(apply=True))
        second = await handler.handle(_request(apply=True))

        assert first.outcomes[0].decision == EnumEvidenceAutocloseDecision.GAP_POSTED
        assert second.outcomes[0].decision == EnumEvidenceAutocloseDecision.GAP_POSTED
        assert len(linear.comments) == 2, linear.comments

    async def test_unreadable_comment_history_fails_closed(self) -> None:
        """Cannot prove it has not commented => must not comment."""
        linear = _StatefulLinear(comment_history_readable=False)
        handler = _handler(
            [_skill_result(total=6, verified=3, failed=3)], linear, [_merged_pr(7001)]
        )

        result = await handler.handle(_request(apply=True))

        assert (
            result.outcomes[0].decision
            == EnumEvidenceAutocloseDecision.ERROR_LINEAR_API
        )
        assert result.tickets_errored == 1
        assert linear.comments == []

    async def test_behavior_proof_gap_dedupes(self) -> None:
        linear = _StatefulLinear()
        handler = _handler(
            [_skill_result(total=6, verified=6, failed=0, behavior_proving=0)],
            linear,
            [_merged_pr(7001)],
        )

        first = await handler.handle(_request(apply=True))
        second = await handler.handle(_request(apply=True))

        assert (
            first.outcomes[0].decision
            == EnumEvidenceAutocloseDecision.GAP_NO_BEHAVIOR_PROOF
        )
        assert (
            second.outcomes[0].decision
            == EnumEvidenceAutocloseDecision.SKIPPED_DUPLICATE_COMMENT
        )
        assert len(linear.comments) == 1
        assert "class=gap_no_behavior_proof" in linear.comments[0][1]

    async def test_ac_coverage_gap_dedupes(self) -> None:
        linear = _StatefulLinear(
            description="## Acceptance criteria\n\n- [ ] AC1 not done yet\n"
        )
        handler = _handler(
            [_skill_result(total=6, verified=6, failed=0, behavior_proving=2)],
            linear,
            [_merged_pr(7001)],
        )

        first = await handler.handle(_request(apply=True))
        second = await handler.handle(_request(apply=True))

        assert (
            first.outcomes[0].decision == EnumEvidenceAutocloseDecision.GAP_AC_COVERAGE
        )
        assert (
            second.outcomes[0].decision
            == EnumEvidenceAutocloseDecision.SKIPPED_DUPLICATE_COMMENT
        )
        assert len(linear.comments) == 1
        assert "class=gap_ac_coverage" in linear.comments[0][1]

    async def test_dry_run_writes_nothing_on_any_dedup_path(self) -> None:
        """DRY-RUN reads the history (honest preview) and writes nothing."""
        linear = _StatefulLinear()
        handler = _handler(
            [_skill_result(total=6, verified=3, failed=3)], linear, [_merged_pr(7001)]
        )

        first = await handler.handle(_request(apply=False))
        second = await handler.handle(_request(apply=False))

        # Nothing was written, so nothing is a duplicate: both dry-runs report
        # the gap they would post.
        assert first.outcomes[0].decision == EnumEvidenceAutocloseDecision.GAP_POSTED
        assert second.outcomes[0].decision == EnumEvidenceAutocloseDecision.GAP_POSTED
        assert linear.comments == []
        assert linear.state_updates == []

    async def test_dry_run_previews_the_suppression_after_a_real_post(self) -> None:
        """Once a marker exists, DRY-RUN reports the suppression, not a gap."""
        linear = _StatefulLinear()
        verdict = _skill_result(total=6, verified=3, failed=3)
        await _handler([verdict], linear, [_merged_pr(7001)]).handle(
            _request(apply=True)
        )

        preview = await _handler([verdict], linear, [_merged_pr(7001)]).handle(
            _request(apply=False)
        )

        assert (
            preview.outcomes[0].decision
            == EnumEvidenceAutocloseDecision.SKIPPED_DUPLICATE_COMMENT
        )
        assert len(linear.comments) == 1

    async def test_duplicate_suppression_counts_as_skipped_not_gap(self) -> None:
        linear = _StatefulLinear()
        handler = _handler(
            [_skill_result(total=6, verified=3, failed=3)], linear, [_merged_pr(7001)]
        )

        await handler.handle(_request(apply=True))
        second = await handler.handle(_request(apply=True))

        assert second.tickets_skipped == 1
        assert second.tickets_gap_posted == 0
        assert second.tickets_errored == 0
