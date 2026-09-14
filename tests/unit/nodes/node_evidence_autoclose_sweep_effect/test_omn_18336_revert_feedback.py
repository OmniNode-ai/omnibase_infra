# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18336 — a revert becomes a correction instead of being absorbed.

The revert fence exists and has fired sixteen times. Nothing turned a revert
into feedback. Four of forty-seven closer flips were reverted by hand — an 8.5%
false-positive rate being absorbed rather than measured — and all four shared
one failure class: a guard was proven and its remediation was not. Establishing
that took an archaeology exercise across receipts, because nothing recorded what
the closer had BELIEVED at the moment of a wrong flip.

Two surfaces end that, and neither touches the flip predicate:

* the post-revert comment names each acceptance criterion the flip counted as
  discharged, PAIRED with the check id that discharged it, and carries a
  verbatim countable marker line;
* the receipt carries the revert as a boolean per outcome and a count per run,
  which is the series value a process panel reads.

RED before the change: the exact-fingerprint branch returned silently — the
clearest revert of this mechanism's own judgement there is produced no comment
at all — and no receipt field anywhere distinguished a reverted flip from any
other hold, so the rate could not be counted.

What this module ALSO pins is the boundary. A reverted HAND flip is somebody
else's judgement being undone. It gets no feedback section and no marker, and it
does not enter the count, because a rate inflated with errors this mechanism did
not make is worse than no rate.

THE FLIP PREDICATE IS UNCHANGED. OMN-18056 already prevents the failure class
going forward; widening the hold further is explicitly not proposed here.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    _CLOSER_FLIP_REVERTED_MARKER,
    HandlerEvidenceAutocloseSweep,
    _counted_ac_bindings,
    _format_post_revert_feedback,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_ac_binding_check_status import (
    EnumAcBindingCheckStatus,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_evidence_autoclose_decision import (
    EnumEvidenceAutocloseDecision,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_ac_binding_row import (
    ModelAcBindingRow,
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
    "- AC1: the guard exists and refuses the shape it was written to refuse.\n"
    "- AC2: the exposures the guard was meant to fix are declared.\n"
)

#: The OMN-18043 failure class, as a fixture: the guard was proven and its
#: remediation was not, so AC1 discharged and AC2 did not.
_BOUND_ROWS = (
    ModelAcBindingRow(
        acceptance_criterion="AC1: the guard exists and refuses the shape it "
        "was written to refuse.",
        label="AC1",
        evidence_check="ac1-ratchet-refuses-the-shape",
        status=EnumAcBindingCheckStatus.VERIFIED,
        proof_class="behavior",
        bound=True,
    ),
    ModelAcBindingRow(
        acceptance_criterion="AC2: the exposures the guard was meant to fix "
        "are declared.",
        label="AC2",
        evidence_check="ac2-exposures-declared",
        status=EnumAcBindingCheckStatus.VERIFIED,
        proof_class="behavior",
        bound=True,
    ),
)


# ------------------------------------------------------ the pairing itself --


class TestCountedBindings:
    def test_every_bound_criterion_is_paired_with_its_check_id(self) -> None:
        """AC2. A bare list of labels leaves the other half to archaeology."""
        assert _counted_ac_bindings(_BOUND_ROWS) == (
            "AC1 -> ac1-ratchet-refuses-the-shape",
            "AC2 -> ac2-exposures-declared",
        )

    def test_an_unbound_row_is_not_counted(self) -> None:
        """The question is what the flip COUNTED, not what it looked at."""
        rows = (
            *_BOUND_ROWS,
            ModelAcBindingRow(
                acceptance_criterion="AC3: something nothing proved.",
                label="AC3",
                bound=False,
            ),
        )

        assert not any("AC3" in pair for pair in _counted_ac_bindings(rows))

    def test_a_bound_row_with_no_check_id_is_rendered_not_dropped(self) -> None:
        """A criterion counted as bound by nothing nameable is the single most
        interesting row a revert could produce, so it is never silently gone."""
        rows = (
            ModelAcBindingRow(acceptance_criterion="AC1: x", label="AC1", bound=True),
        )

        assert _counted_ac_bindings(rows) == ("AC1 -> <no check id recorded>",)

    def test_the_feedback_section_carries_the_countable_marker(self) -> None:
        """AC4. Prose is not a series value; a verbatim line is."""
        rendered = _format_post_revert_feedback(_BOUND_ROWS)

        assert _CLOSER_FLIP_REVERTED_MARKER in rendered
        assert "AC1 -> ac1-ratchet-refuses-the-shape" in rendered
        assert "AC2 -> ac2-exposures-declared" in rendered

    def test_a_flip_that_counted_nothing_says_so_rather_than_rendering_empty(
        self,
    ) -> None:
        rendered = _format_post_revert_feedback(())

        assert _CLOSER_FLIP_REVERTED_MARKER in rendered
        assert "no labelled criterion" in rendered


# ------------------------------------------------- wired into the closer -----


def _issue(state_type: str) -> dict[str, object]:
    return {
        "id": "issue-1",
        "identifier": "OMN-0000",
        "state": {"id": "s1", "name": "In Progress", "type": state_type},
        "labels": {"nodes": []},
        "team": {"id": "team-1"},
        "description": DESCRIPTION,
        "children": {"nodes": []},
    }


_REVERT_AT = (datetime.now(tz=UTC) - timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ")


class FakeLinear:
    """A ticket this closer flipped and somebody moved back.

    ``seed_flip_comment`` is the whole boundary: with it, the closer's own flip
    comment is on the ticket and the revert is this mechanism's error. Without
    it, the identical history is a reverted HAND flip.
    """

    def __init__(
        self, *, seed_flip_comment: bool, seed_fingerprint: str = "deadbeefdeadbeef"
    ) -> None:
        self._issue = _issue("started")
        self.state_updates: list[tuple[str, str]] = []
        self.comments: list[tuple[str, str]] = []
        self._seeded: tuple[str, ...] = ()
        if seed_flip_comment:
            # The shape the closer's own flip audit comment carries. The fence
            # identifies its own flips from this class marker and from nothing
            # else, which is exactly what makes a reverted HAND flip separable.
            self._seeded = (
                "Auto-closed Done (evidence autoclose sweep).\n\n"
                f"Verdict fingerprint {seed_fingerprint}\n\n"
                "<!-- onex-autoclose class=flipped -->",
            )

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
        return self._seeded + tuple(
            body for target, body in self.comments if target == issue_id
        )

    async def fetch_issue_history(
        self, issue_id: str, page_size: int, max_pages: int
    ) -> tuple[list[dict[str, object]] | None, str]:
        return [
            {
                "id": "entry-revert",
                "createdAt": _REVERT_AT,
                "actorId": "a-person",
                "botActor": None,
                "fromState": {"type": "completed"},
                "toState": {"type": "started"},
            },
            {
                "id": "entry-flip",
                "createdAt": (datetime.now(tz=UTC) - timedelta(hours=8)).isoformat(),
                "actorId": "actor-1",
                "botActor": None,
                "fromState": {"type": "started"},
                "toState": {"type": "completed"},
            },
        ], ""


def _verdict() -> dict[str, object]:
    return {
        "correlation_id": str(uuid4()),
        "ticket_id": "OMN-0000",
        "status": "verified",
        "dry_run": False,
        "checks": [
            {
                "evidence_id": "ac1-ratchet-refuses-the-shape",
                "status": "verified",
                "proof_class": "behavior",
                "binds_ac": ["AC1"],
            },
            {
                "evidence_id": "ac2-exposures-declared",
                "status": "verified",
                "proof_class": "behavior",
                "binds_ac": ["AC2"],
            },
        ],
        "total_checks": 2,
        "verified_count": 2,
        "failed_count": 0,
        "skipped_count": 0,
        "superseded_count": 0,
        "non_probative_count": 0,
        "behavior_proving_count": 2,
        "error_message": None,
    }


def _receipt() -> dict[str, object]:
    return {
        "skill_name": "dod_verify",
        "node_name": "node_dod_verify",
        "status": "success",
        "correlation_id": str(uuid4()),
        "run_id": str(uuid4()),
        "exit_code": 0,
        "duration_ms": 1,
        "result": _verdict(),
        "result_model": _DOD_VERIFY_STATE_MODEL,
    }


def _gh_fake():
    # Every piece of evidence predates the reversal, so the OMN-18106 ordering
    # release cannot fire and the post-revert hold is the branch under test.
    stale = (datetime.now(tz=UTC) - timedelta(hours=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    companions = [
        {
            "number": 9001,
            "html_url": f"https://github.com/{_OCC_REPO}/pull/9001",
            "title": (
                "evidence(OMN-0000): OCC companion for OmniNode-ai/omnibase_infra#3194"
            ),
            "updated_at": stale,
            "merged_at": stale,
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
                "merged_at": stale,
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


def _handler(linear: FakeLinear):
    return HandlerEvidenceAutocloseSweep(
        linear_client=linear,  # type: ignore[arg-type]
        autoclose_disabled=False,
        run_gh_command=_gh_fake(),
        run_dod_verify_command=_dod_fake(_receipt()),
    )


@pytest.mark.asyncio
class TestSyntheticRevert:
    async def test_the_comment_names_the_criteria_and_the_check_ids(self) -> None:
        """AC1 and AC2 on the whole closer, against a synthetic revert."""
        linear = FakeLinear(seed_flip_comment=True)

        result = await _handler(linear).handle(_request())

        assert linear.state_updates == [], "a reverted ticket is not re-flipped"
        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.SKIPPED_PRIOR_REVERT
        ]
        body = "\n".join(body for _target, body in linear.comments)
        assert body, "a revert of the closer's own flip must produce a comment"
        assert "AC1 -> ac1-ratchet-refuses-the-shape" in body
        assert "AC2 -> ac2-exposures-declared" in body

    async def test_the_revert_is_countable_on_the_receipt(self) -> None:
        """AC4. The series value the process panel reads."""
        linear = FakeLinear(seed_flip_comment=True)

        result = await _handler(linear).handle(_request())

        assert result.tickets_closer_flip_reverted == 1
        outcome = result.outcomes[0]
        assert outcome.closer_flip_reverted is True
        assert outcome.counted_ac_bindings == (
            "AC1 -> ac1-ratchet-refuses-the-shape",
            "AC2 -> ac2-exposures-declared",
        )

    async def test_the_comment_carries_the_countable_marker(self) -> None:
        """AC4, the ticket-surface half: countable by grep as well as receipt."""
        linear = FakeLinear(seed_flip_comment=True)

        await _handler(linear).handle(_request())

        body = "\n".join(body for _target, body in linear.comments)
        assert _CLOSER_FLIP_REVERTED_MARKER in body

    async def test_a_reverted_hand_flip_produces_no_feedback_and_no_count(
        self,
    ) -> None:
        """AC3. The boundary.

        The identical history, with no flip comment of this closer's on the
        ticket. That is somebody else's judgement being undone; claiming it
        would inflate this mechanism's error rate with an error it did not
        make. The ticket is still held — that is the OMN-18056 fence and it is
        unchanged — but nothing here calls the hold a correction.
        """
        linear = FakeLinear(seed_flip_comment=False)

        result = await _handler(linear).handle(_request())

        assert result.tickets_closer_flip_reverted == 0
        assert all(not o.closer_flip_reverted for o in result.outcomes)
        body = "\n".join(body for _target, body in linear.comments)
        assert _CLOSER_FLIP_REVERTED_MARKER not in body
        assert "-> ac1-ratchet-refuses-the-shape" not in body

    async def test_the_exact_fingerprint_branch_also_reports_instead_of_returning(
        self,
    ) -> None:
        """The branch that used to return SILENTLY.

        Re-offering the IDENTICAL verdict this closer already wrote and
        somebody already undid is the clearest revert of this mechanism's own
        judgement there is, and it produced no record at all. The fingerprint
        is read from a first run rather than hard-coded, so a change to the
        digest inputs cannot make this test quietly exercise the other branch.
        """
        probe = await _handler(FakeLinear(seed_flip_comment=False)).handle(_request())
        fingerprint = probe.outcomes[0].verdict_fingerprint
        assert fingerprint

        linear = FakeLinear(seed_flip_comment=True, seed_fingerprint=fingerprint)
        result = await _handler(linear).handle(_request())

        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.SKIPPED_PRIOR_REVERT
        ]
        assert result.tickets_closer_flip_reverted == 1
        body = "\n".join(body for _target, body in linear.comments)
        assert _CLOSER_FLIP_REVERTED_MARKER in body
        assert "AC1 -> ac1-ratchet-refuses-the-shape" in body

    async def test_the_flip_predicate_is_not_widened_by_this_change(self) -> None:
        """Out of scope, asserted rather than assumed.

        A ticket with no reversal in its history is untouched: same two
        verified bound criteria, and it still flips on the second tick.
        """
        linear = FakeLinear(seed_flip_comment=False)

        async def no_revert(
            issue_id: str, page_size: int, max_pages: int
        ) -> tuple[list[dict[str, object]] | None, str]:
            if any(target == issue_id for target, _s in linear.state_updates):
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

        linear.fetch_issue_history = no_revert  # type: ignore[method-assign]
        handler = _handler(linear)

        await handler.handle(_request())
        result = await handler.handle(_request())

        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.FLIPPED
        ]
        assert result.tickets_closer_flip_reverted == 0
