# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18238 — a proposed binding is not proof until somebody accepts it.

`binds_ac` lets an evidence item declare which acceptance criterion it proves,
and the closer discharges a criterion when a VERIFIED check declares it. That is
the right rule for a binding a person wrote. It is the wrong rule the moment a
machine can write one: the obvious way to make bindings cheap is to let the
autobinder guess a criterion from a matching check name, and a passing check
with a matching name is not proof of the criterion it names. Adopting that would
convert the whole mechanism into an expensive way of restating check names.

So a machine may PROPOSE. A proposal is a binding record awaiting acceptance,
and the closer does not count it.
It becomes proof when the evidence author or a reviewer accepts it, and the
acceptance is recorded on the item.

The distinction this module pins, and the reason it does not break the corpus:

* a criterion whose only declaring check marks it a DRAFT is UNBOUND, and the
  hold says the binding is a proposal awaiting acceptance rather than "declared
  by nothing", because those are different repairs;
* a criterion declared by a check that does NOT mark it a draft binds exactly as
  it did before. Sixty-seven live contracts declare `binds_ac` with no binding
  record at all; every one of them was hand-authored, which IS the acceptance
  this rule asks for, and none of them changes behaviour here.

RED before the change: `draft_binds_ac` was read by nothing, so a draft
discharged its criterion and the closer flipped on a binding nobody had agreed
to.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    HandlerEvidenceAutocloseSweep,
    _ac_binding_gap,
    _declared_ac_bindings,
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

DESCRIPTION = (
    "## Acceptance criteria\n"
    "\n"
    "- AC1: the behaviour this fixture exercises is proven by a verified "
    "probative check.\n"
)

MULTI_DESCRIPTION = "## Acceptance criteria\n\n" + "\n".join(
    f"- AC{idx}: behaviour {idx}." for idx in range(1, 23)
)


def _check(*, draft: bool) -> dict[str, object]:
    """One verified probative check declaring AC1, as a draft or as a binding."""
    record: dict[str, object] = {
        "evidence_id": "omn18238-check",
        "status": "verified",
        "proof_class": "behavior",
        "binds_ac": ["AC1"],
    }
    if draft:
        record["draft_binds_ac"] = ["AC1"]
    return record


def _verdict(*, draft: bool) -> dict[str, object]:
    return {
        "correlation_id": str(uuid4()),
        "ticket_id": "OMN-0000",
        "status": "verified",
        "dry_run": False,
        "checks": [_check(draft=draft)],
        "total_checks": 2,
        "verified_count": 2,
        "failed_count": 0,
        "skipped_count": 0,
        "superseded_count": 0,
        "non_probative_count": 0,
        "behavior_proving_count": 1,
        "error_message": None,
    }


def _receipt(*, draft: bool) -> dict[str, object]:
    return {
        "skill_name": "dod_verify",
        "node_name": "node_dod_verify",
        "status": "success",
        "correlation_id": str(uuid4()),
        "run_id": str(uuid4()),
        "exit_code": 0,
        "duration_ms": 1,
        "result": _verdict(draft=draft),
        "result_model": _DOD_VERIFY_STATE_MODEL,
    }


# ------------------------------------------------------------ the predicate --


class TestDeclaredBindings:
    def test_a_draft_label_is_not_a_declared_binding(self) -> None:
        _present, bindings, drafts = _declared_ac_bindings(_verdict(draft=True))

        assert bindings == {}
        assert "AC1" in drafts

    def test_a_non_draft_label_binds_exactly_as_before(self) -> None:
        _present, bindings, drafts = _declared_ac_bindings(_verdict(draft=False))

        assert "AC1" in bindings
        assert drafts == {}

    def test_the_field_still_reads_as_present_when_only_drafts_are_declared(
        self,
    ) -> None:
        """An absent `binds_ac` and an all-draft one are different facts.

        The first means the verifier cannot report bindings at all; the second
        means it read the contract and everything in it is a proposal. Both
        hold, and the hold reason must be able to tell them apart.
        """
        present, _bindings, _drafts = _declared_ac_bindings(_verdict(draft=True))

        assert present is True

    def test_a_draft_label_the_check_does_not_claim_is_ignored(self) -> None:
        """`draft_binds_ac` narrows `binds_ac`; it cannot add to it."""
        verdict = _verdict(draft=False)
        checks = verdict["checks"]
        assert isinstance(checks, list)
        checks[0]["draft_binds_ac"] = ["AC9"]

        _present, bindings, drafts = _declared_ac_bindings(verdict)

        assert "AC1" in bindings
        assert "AC9" not in drafts

    def test_a_draft_on_one_check_does_not_demote_an_accepted_binding_on_another(
        self,
    ) -> None:
        verdict = _verdict(draft=False)
        checks = verdict["checks"]
        assert isinstance(checks, list)
        checks.append(
            {
                "evidence_id": "omn18238-draft-sibling",
                "status": "verified",
                "proof_class": "behavior",
                "binds_ac": ["AC1"],
                "draft_binds_ac": ["AC1"],
            }
        )

        _present, bindings, drafts = _declared_ac_bindings(verdict)

        assert bindings["AC1"] == (("omn18238-check", "verified", "behavior"),)
        assert drafts["AC1"] == ("omn18238-draft-sibling",)


class TestAcBindingGap:
    def test_a_criterion_whose_only_declaration_is_a_draft_is_unbound(self) -> None:
        reason, uncovered, _rows = _ac_binding_gap(
            DESCRIPTION, _verdict(draft=True), "OMN-0000"
        )

        assert reason
        assert uncovered

    def test_the_hold_says_the_binding_is_a_proposal(self) -> None:
        """ "Declared by nothing" and "proposed but not accepted" are different
        repairs, and sending an author to write a binding that already exists
        is how a gate stops being read."""
        reason, _uncovered, _rows = _ac_binding_gap(
            DESCRIPTION, _verdict(draft=True), "OMN-0000"
        )

        assert "proposal" in reason.lower() or "accept" in reason.lower()
        assert "AC1" in reason

    def test_an_accepted_binding_discharges_the_criterion(self) -> None:
        reason, uncovered, _rows = _ac_binding_gap(
            DESCRIPTION, _verdict(draft=False), "OMN-0000"
        )

        assert reason == ""
        assert uncovered == ()

    def test_proposal_note_deduplicates_and_reports_elided_labels(self) -> None:
        verdict = _verdict(draft=True)
        checks = verdict["checks"]
        assert isinstance(checks, list)
        checks[0]["binds_ac"] = [f"AC{idx}" for idx in range(1, 23)]
        checks[0]["draft_binds_ac"] = [f"AC{idx}" for idx in range(1, 23)]

        reason, uncovered, _rows = _ac_binding_gap(
            MULTI_DESCRIPTION, verdict, "OMN-0000"
        )

        assert uncovered
        proposal_note = reason.split("OMN-18238: ", 1)[1]
        assert proposal_note.startswith(
            "AC1, AC2, AC3, AC4, AC5, AC6, AC7, AC8, AC9, AC10, "
            "AC11, AC12, AC13, AC14, AC15, AC16, AC17, AC18, AC19, AC20 "
        )
        assert "and 2 more" in reason


# ------------------------------------------------- wired into the closer -----


def _issue() -> dict[str, object]:
    return {
        "id": "issue-1",
        "identifier": "OMN-0000",
        "state": {"id": "s1", "name": "In Progress", "type": "started"},
        "labels": {"nodes": []},
        "team": {"id": "team-1"},
        "description": DESCRIPTION,
        "children": {"nodes": []},
    }


class FakeLinear:
    def __init__(self) -> None:
        self._issue = _issue()
        self.state_updates: list[tuple[str, str]] = []
        self.comments: list[tuple[str, str]] = []

    async def fetch_issue(self, ticket_id: str) -> dict[str, object] | None:
        return self._issue if ticket_id == "OMN-0000" else None

    async def fetch_done_state_id(self, team_id: str) -> str | None:
        return "state-done"

    async def update_issue_state(self, issue_id: str, state_id: str) -> bool:
        self.state_updates.append((issue_id, state_id))
        return True

    async def create_comment(self, issue_id: str, body: str) -> bool:
        self.comments.append((issue_id, body))
        return True

    async def fetch_comment_bodies(self, issue_id: str) -> tuple[str, ...] | None:
        return tuple(body for target, body in self.comments if target == issue_id)

    async def fetch_issue_history(
        self, issue_id: str, page_size: int, max_pages: int
    ) -> tuple[list[dict[str, object]] | None, str]:
        if any(target == issue_id for target, _state in self.state_updates):
            return [
                {
                    "id": "entry-flip",
                    "createdAt": datetime.now(tz=UTC).isoformat(),
                    "actorId": "actor-1",
                    "botActor": None,
                    "fromState": {"type": "started"},
                    "toState": {"type": "completed"},
                }
            ], ""
        return [], ""


def _gh_fake():
    recent = (datetime.now(tz=UTC) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    companions = [
        {
            "number": 9001,
            "html_url": f"https://github.com/{_OCC_REPO}/pull/9001",
            "title": "evidence(OMN-0000): OCC companion for OmniNode-ai/omnibase_infra#3194",
            "updated_at": recent,
            "merged_at": recent,
        }
    ]

    async def run_gh(args: list[str], timeout: float):
        path = args[2]
        if "/files" in path:
            return [{"filename": "contracts/OMN-0000.yaml"}], ""
        if "/pulls/" in path and "state=closed" not in path:
            return {
                "title": "fix(OMN-0000): a real product change",
                "user": {"login": "a-human-author", "type": "User"},
                "merged_at": recent,
            }, ""
        page = int(path.rsplit("page=", 1)[1])
        return (companions, "") if page == 1 else ([], "")

    return run_gh


def _dod_fake(receipt: dict[str, object]):
    async def run_dod(ticket_id: str, cwd: str, timeout: int):
        return receipt, 0, ""

    return run_dod


def _request() -> ModelEvidenceAutocloseSweepRequest:
    return ModelEvidenceAutocloseSweepRequest(
        correlation_id=uuid4(), occ_repo=_OCC_REPO, lookback_hours=24, apply=True
    )


def _handler(linear: FakeLinear, receipt: dict[str, object]):
    return HandlerEvidenceAutocloseSweep(
        linear_client=linear,  # type: ignore[arg-type]
        autoclose_disabled=False,
        run_gh_command=_gh_fake(),
        run_dod_verify_command=_dod_fake(receipt),
    )


@pytest.mark.asyncio
class TestCloserIntegration:
    """AC1 on the whole closer: the same contract, held then flipped."""

    async def test_an_unaccepted_draft_does_not_satisfy_the_closer(self) -> None:
        linear = FakeLinear()
        handler = _handler(linear, _receipt(draft=True))

        result = await handler.handle(_request())

        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.GAP_AC_UNBOUND
        ]
        assert result.tickets_flipped == 0
        # The receipt says the binding is a proposal, not that nothing
        # declares the criterion.
        assert "PROPOSAL" in result.outcomes[0].reason
        assert linear.state_updates == []

    async def test_the_same_contract_satisfies_it_once_accepted(self) -> None:
        """Byte-identical except that the binding is no longer a draft."""
        linear = FakeLinear()
        handler = _handler(linear, _receipt(draft=False))

        armed = await handler.handle(_request())
        assert [o.decision for o in armed.outcomes] == [
            EnumEvidenceAutocloseDecision.SKIPPED_REDRAW_PENDING
        ]

        result = await handler.handle(_request())

        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.FLIPPED
        ]
        assert result.tickets_flipped == 1

    async def test_a_verdict_with_no_draft_key_at_all_is_unchanged(self) -> None:
        """The corpus control.

        Sixty-seven live contracts declare `binds_ac` and carry no binding
        record, so their verdicts carry no draft key. Not one of them may
        change behaviour because this rule exists.
        """
        linear = FakeLinear()
        receipt = _receipt(draft=False)
        result_payload = receipt["result"]
        assert isinstance(result_payload, dict)
        checks = result_payload["checks"]
        assert isinstance(checks, list)
        assert "draft_binds_ac" not in checks[0]

        handler = _handler(linear, receipt)
        await handler.handle(_request())
        result = await handler.handle(_request())

        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.FLIPPED
        ]
