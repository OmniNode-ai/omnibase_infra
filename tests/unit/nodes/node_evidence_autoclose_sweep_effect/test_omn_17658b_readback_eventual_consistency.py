# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17658 follow-up — the bound readback read a connection that lags.

Measured, not theorised. The FIRST scheduled run under the fences
(``33958237006``, ``f8b623672``, mode ``apply_scheduled``, 2026-09-05T09:33Z)
produced this outcome, verbatim:

    {"ticket_id":"OMN-17658", "companion_pr_number":8270,
     "decision":"error_readback_unconfirmed",
     "reason":"issueUpdate(stateId) reported success, but the post-write read
       of this ticket's state history shows no completed segment newer than
       d75cf137-a39c-4929-99bd-d072a2a09817. ...",
     "dod_verify_verified_count":6, "dod_verify_failed_count":0,
     "dod_verify_behavior_proving_count":1, "applied":false}

The write **landed**: OMN-17658's own `stateHistory` carries `In Progress →
Done` at ``2026-09-05T09:34:43.990Z``, `completedAt`
``2026-09-05T09:34:43.951Z``, authored by that run. Linear's `history`
connection simply had not caught up when the readback fired microseconds
later.

Three separate defects follow from that, and each has a test below.

1. **The readback does not retry.** A single immediate read of an eventually
   consistent connection is not a proof of absence, it is a race. Every flip
   would have been recorded ``error_readback_unconfirmed`` forever, so
   ``tickets_flipped`` could never leave 0 — the closer would have been
   silently reduced to a mechanism that writes Done and reports that it did
   not.

2. **``applied`` said ``false`` on a run that mutated Linear.** The field's own
   contract is "True only when a real Linear mutation was made". A receipt
   that under-reports a write is worse than one that over-reports it: the
   over-report is caught by the next reader of the ticket, the under-report is
   invisible.

3. **No audit comment was posted**, so the ticket carried a closer-written Done
   with nothing on it saying who wrote it or why — and no ``class=flipped``
   marker, which is the anchor the OMN-17934 prior-revert fence is meant to
   grow into.
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
_DOD_VERIFY_STATE_MODEL = (
    "omnimarket.nodes.node_dod_verify.models.model_dod_verify_state.ModelDodVerifyState"
)


def _companion(number: int, ticket: str) -> dict[str, object]:
    recent = (datetime.now(tz=UTC) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "number": number,
        "html_url": f"https://github.com/{_OCC_REPO}/pull/{number}",
        "title": f"evidence({ticket}): OCC companion",
        "updated_at": recent,
        "merged_at": recent,
    }


def _receipt() -> dict[str, object]:
    verdict: dict[str, object] = {
        "correlation_id": str(uuid4()),
        "ticket_id": "OMN-17658",
        "status": "verified",
        "dry_run": False,
        "checks": [],
        "total_checks": 8,
        "verified_count": 6,
        "failed_count": 0,
        "skipped_count": 0,
        "superseded_count": 0,
        "non_probative_count": 2,
        "behavior_proving_count": 1,
        "error_message": None,
    }
    return {
        "skill_name": "dod_verify",
        "node_name": "node_dod_verify",
        "status": "success",
        "correlation_id": str(uuid4()),
        "run_id": str(uuid4()),
        "exit_code": 0,
        "duration_ms": 1,
        "result": verdict,
        "result_model": _DOD_VERIFY_STATE_MODEL,
    }


def _issue() -> dict[str, object]:
    return {
        "id": "issue-1",
        "identifier": "OMN-17658",
        "state": {"id": "s1", "name": "In Progress", "type": "started"},
        "labels": {"nodes": []},
        "team": {"id": "team-1"},
        "description": None,
        "children": {"nodes": []},
    }


class LaggingLinear:
    """A Linear whose `history` connection catches up N reads after the write.

    ``lag_reads=0`` models a connection that is already consistent;
    ``lag_reads=2`` models the measured race, where the entry appears only on
    the third post-write read.
    """

    def __init__(self, lag_reads: int, *, ever_consistent: bool = True) -> None:
        self._lag = lag_reads
        self._ever = ever_consistent
        self._written = False
        self._reads_since_write = 0
        self.state_updates: list[tuple[str, str]] = []
        self.comments: list[tuple[str, str]] = []
        self.sleeps: list[float] = []

    async def fetch_issue(self, ticket_id: str) -> dict[str, object] | None:
        return _issue() if ticket_id == "OMN-17658" else None

    async def fetch_done_state_id(self, team_id: str) -> str | None:
        return "state-done"

    async def update_issue_state(self, issue_id: str, state_id: str) -> bool:
        self.state_updates.append((issue_id, state_id))
        self._written = True
        return True

    async def create_comment(self, issue_id: str, body: str) -> bool:
        self.comments.append((issue_id, body))
        return True

    async def fetch_comment_bodies(self, issue_id: str) -> tuple[str, ...] | None:
        return tuple(body for target, body in self.comments if target == issue_id)

    async def fetch_issue_history(
        self, issue_id: str, page_size: int, max_pages: int
    ) -> tuple[list[dict[str, object]] | None, str]:
        base: list[dict[str, object]] = [
            {
                "id": "d75cf137-a39c-4929-99bd-d072a2a09817",
                "createdAt": "2026-09-05T08:10:52Z",
                "actorId": "human",
                "fromState": {"type": "backlog"},
                "toState": {"type": "started"},
            }
        ]
        if not self._written:
            return base, ""
        self._reads_since_write += 1
        if self._ever and self._reads_since_write > self._lag:
            return [
                {
                    "id": "flip-entry",
                    "createdAt": "2026-09-05T09:34:43Z",
                    "actorId": None,
                    "fromState": {"type": "started"},
                    "toState": {"type": "completed"},
                },
                *base,
            ], ""
        return base, ""


def _gh(companions: list[dict[str, object]], files: dict[int, list[str]]):
    async def run_gh(args: list[str], timeout: float):
        path = args[2]
        if "/files" in path:
            number = int(path.split("/pulls/")[1].split("/files")[0])
            return [{"filename": f} for f in files.get(number, [])], ""
        page = int(path.rsplit("page=", 1)[1])
        return (companions, "") if page == 1 else ([], "")

    return run_gh


def _dod():
    async def run_dod(ticket_id: str, cwd: str, timeout: int):
        return _receipt(), 0, ""

    return run_dod


def _handler(linear: LaggingLinear) -> HandlerEvidenceAutocloseSweep:
    return HandlerEvidenceAutocloseSweep(
        linear_client=linear,  # type: ignore[arg-type]
        autoclose_disabled=False,
        run_gh_command=_gh(
            [_companion(8270, "OMN-17658")], {8270: ["contracts/OMN-17658.yaml"]}
        ),
        run_dod_verify_command=_dod(),
    )


def _request(**overrides: object) -> ModelEvidenceAutocloseSweepRequest:
    payload: dict[str, object] = {
        "correlation_id": uuid4(),
        "occ_repo": _OCC_REPO,
        "lookback_hours": 24,
        "apply": True,
        "backfill_lookback_hours": 0,
        # Keep the test fast; the retry delay is contract-exposed precisely so
        # a test does not have to wait out a production backoff.
        "readback_delay_seconds": 0,
    }
    payload.update(overrides)
    return ModelEvidenceAutocloseSweepRequest(**payload)


@pytest.mark.asyncio
class TestReadbackSurvivesAConnectionThatLags:
    async def test_the_measured_race_now_confirms_instead_of_erroring(self) -> None:
        """Two stale reads then the entry — the shape run 33958237006 hit."""
        linear = LaggingLinear(lag_reads=2)
        result = await _handler(linear).handle(_request(readback_max_attempts=4))

        outcome = result.outcomes[0]
        assert outcome.decision is EnumEvidenceAutocloseDecision.FLIPPED
        assert outcome.readback_entry_id == "flip-entry"
        assert outcome.readback_entry_id != outcome.pre_write_head_entry_id
        assert outcome.applied is True
        assert result.tickets_flipped == 1
        assert len(linear.state_updates) == 1

    async def test_an_already_consistent_connection_needs_no_extra_read(self) -> None:
        """The retry must not become a mandatory delay on the common path."""
        linear = LaggingLinear(lag_reads=0)
        result = await _handler(linear).handle(_request(readback_max_attempts=4))
        assert result.outcomes[0].decision is EnumEvidenceAutocloseDecision.FLIPPED
        assert linear._reads_since_write == 1

    async def test_a_permanently_unconfirmed_readback_still_reports_the_write(
        self,
    ) -> None:
        """The under-report defect, which is the worse half of the incident.

        `applied` means "a real Linear mutation was made" — and one was. The
        run that produced this shape recorded `applied: false` on a ticket it
        had just moved to Done, which is a receipt that disagrees with the
        board in the direction nobody checks.
        """
        linear = LaggingLinear(lag_reads=99, ever_consistent=False)
        result = await _handler(linear).handle(_request(readback_max_attempts=2))

        outcome = result.outcomes[0]
        assert (
            outcome.decision is EnumEvidenceAutocloseDecision.ERROR_READBACK_UNCONFIRMED
        )
        assert outcome.applied is True, (
            "a mutation was made; a receipt that denies it is worse than one "
            "that overstates it, because nothing downstream checks for a "
            "write that was never claimed"
        )
        assert outcome.readback_entry_id == ""
        assert result.tickets_flipped == 0
        # The reason must not read as "nothing happened".
        assert "was written" in outcome.reason.lower()

    async def test_an_unconfirmed_flip_still_leaves_the_audit_trail(self) -> None:
        """A Done nobody can attribute is the worst of both outcomes.

        The measured run left OMN-17658 in Done with no comment at all: no
        counters, no companion, and no `class=flipped` marker — which is the
        anchor the OMN-17934 prior-revert fence is meant to grow into. The
        comment is posted either way; what changes is that it states the
        readback did not confirm rather than quoting an entry id it does not
        have.
        """
        linear = LaggingLinear(lag_reads=99, ever_consistent=False)
        await _handler(linear).handle(_request(readback_max_attempts=2))

        assert len(linear.comments) == 1
        body = linear.comments[0][1]
        assert "class=flipped" in body
        assert "UNCONFIRMED" in body
        assert "8270" in body

    async def test_the_retry_is_bounded_by_the_contract(self) -> None:
        linear = LaggingLinear(lag_reads=99, ever_consistent=False)
        await _handler(linear).handle(_request(readback_max_attempts=3))
        assert linear._reads_since_write == 3
