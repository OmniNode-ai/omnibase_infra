# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18330 — a binding proves the criterion it was accepted against, and no other.

OMN-18236 gave a binding record a ``criterion_hash``: the sha256 of the criterion
text the binding was derived or accepted against. The closer never read it. So
an author could accept a binding against one sentence, the sentence could be
rewritten into something the bound check does not prove, and the closer still
counted the binding and flipped the ticket Done. That is a false-close path, and
nothing else in the epic closes it.

RED before the change: ``_ac_binding_gap`` took no pins at all, so
``test_a_binding_pinned_to_a_rewritten_criterion_does_not_discharge_it`` could
not even be expressed against the old signature — the predicate had no way to
learn that the criterion had moved, and the same fixture flipped.

What this module pins, and what it deliberately does NOT:

* a pinned label whose hash no longer matches the criterion's current text is
  UNBOUND, and the hold names the label and says the text changed after
  acceptance;
* a pinned label whose hash DOES match is untouched and still flips, so the
  change is proven not to be a blanket widening;
* a binding record carrying no readable hash is unvalidated rather than stale,
  is held, and is named apart from the stale case because the repair differs;
* a label with NO binding record at all binds exactly as it did. Sixty-seven of
  the sixty-eight live contracts that declare ``binds_ac`` carry no record, every
  one hand-authored. Holding those on a pin that does not exist yet is named Out
  of scope on the ticket, and this module has a test asserting it does not happen.

The hash itself is a port of ``onex_change_control``
``src/onex_change_control/validation/ac_criteria.py``. The vectors in
``TestTheHashIsTheChangeControlHash`` are the ones that repository's own tests
pin, so a change on either side fails with a test naming the other.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    HandlerEvidenceAutocloseSweep,
    _ac_binding_gap,
    _criterion_pin_hash,
    _normalise_criterion,
    _pinned_criterion_hashes,
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

#: The criterion as the binding was accepted against it.
_ACCEPTED_CRITERION = (
    "AC1: the sweep refuses to flip a ticket whose criterion text changed "
    "after its binding was accepted."
)
#: The same label, rewritten afterwards into a materially different demand.
_REWRITTEN_CRITERION = (
    "AC1: the sweep flips a ticket whose criterion text changed after its "
    "binding was accepted."
)


def _description(criterion: str) -> str:
    return f"## Acceptance criteria\n\n- {criterion}\n"


#: The item text `_acceptance_criteria_items` returns strips the list bullet, so
#: the pin is taken over the criterion itself and not over its markdown.
_ACCEPTED_PIN = _criterion_pin_hash(_ACCEPTED_CRITERION)


def _verdict(pins: dict[str, str] | None = None) -> dict[str, object]:
    """One verified behaviour-proving check declaring AC1, nothing drafted.

    ``pins`` is the per-check ``ac_binding_hashes`` map. ``None`` omits the key
    entirely, which is the corpus shape: 67 of the 68 contracts that declare a
    binding record no pin at all.
    """
    check: dict[str, object] = {
        "evidence_id": "omn18330-check",
        "status": "verified",
        "proof_class": "behavior",
        "binds_ac": ["AC1"],
    }
    if pins is not None:
        check["ac_binding_hashes"] = pins
    return {
        "correlation_id": str(uuid4()),
        "ticket_id": "OMN-0000",
        "status": "verified",
        "dry_run": False,
        "checks": [check],
        "total_checks": 2,
        "verified_count": 2,
        "failed_count": 0,
        "skipped_count": 0,
        "superseded_count": 0,
        "non_probative_count": 0,
        "behavior_proving_count": 1,
        "error_message": None,
    }


# ---------------------------------------------------------------- the hash --


class TestTheHashIsTheChangeControlHash:
    """The digest must equal `onex_change_control`'s `criterion_hash`.

    These are that module's own vectors. They are duplicated here rather than
    imported because the package is a dev-group dependency pinned to a rev that
    predates the module; the duplication is the coupling, and this class is what
    makes it fail loudly instead of drifting quietly.
    """

    def test_the_digest_is_sha256_of_the_normalised_text(self) -> None:
        assert (
            _criterion_pin_hash("AC1: anything")
            == hashlib.sha256(b"AC1: anything").hexdigest()
        )

    def test_rewrapping_a_paragraph_is_not_a_rewrite(self) -> None:
        wrapped = "AC1: the lane is\ngreen and stays\n  green."
        flowed = "AC1: the lane is green and stays green."

        assert _criterion_pin_hash(wrapped) == _criterion_pin_hash(flowed)

    def test_a_negation_is_a_rewrite(self) -> None:
        assert _criterion_pin_hash("AC1: the gate refuses it.") != _criterion_pin_hash(
            "AC1: the gate does not refuse it."
        )

    def test_a_case_change_is_a_rewrite(self) -> None:
        assert _criterion_pin_hash("AC1: the lane is green.") != _criterion_pin_hash(
            "AC1: the lane MUST be green."
        )

    def test_the_digest_is_a_full_lowercase_sha256(self) -> None:
        value = _criterion_pin_hash("AC1: anything")

        assert len(value) == 64
        assert value == value.lower()

    def test_the_input_ceiling_matches_change_control(self) -> None:
        """4000 chars, truncated deterministically — OCC's own ceiling."""
        assert len(_normalise_criterion("x" * 5000)) == 4000


# ------------------------------------------------------- reading the pins ----


class TestPinnedCriterionHashes:
    def test_a_verdict_with_no_pin_key_pins_nothing(self) -> None:
        assert _pinned_criterion_hashes(_verdict()) == {}

    def test_a_pin_is_read_under_its_canonical_label(self) -> None:
        pins = _pinned_criterion_hashes(_verdict({"ac-1": _ACCEPTED_PIN}))

        assert pins == {"AC1": (_ACCEPTED_PIN,)}

    def test_pins_from_several_checks_for_one_label_all_count(self) -> None:
        """A re-acceptance is APPENDED beside the original -- the OCC
        append-only validator permits no other shape -- and two evidence items
        may each bind one criterion. Either pin matching means the criterion's
        current text was accepted."""
        verdict = _verdict({"AC1": _ACCEPTED_PIN})
        checks = verdict["checks"]
        assert isinstance(checks, list)
        checks.append(
            {
                "evidence_id": "omn18330-sibling",
                "status": "verified",
                "proof_class": "behavior",
                "binds_ac": ["AC1"],
                "ac_binding_hashes": {"AC1": "b" * 64},
            }
        )

        assert _pinned_criterion_hashes(verdict) == {"AC1": (_ACCEPTED_PIN, "b" * 64)}

    def test_a_pin_with_no_hash_leaves_the_label_present_and_empty(self) -> None:
        """Present-and-empty is the "unvalidated" fact, and it is NOT the same
        fact as absent. Absent means no record exists; empty means one exists
        and declines to say which revision it pinned."""
        assert _pinned_criterion_hashes(_verdict({"AC1": ""})) == {"AC1": ()}

    def test_an_unlabelled_pin_is_ignored_rather_than_invented(self) -> None:
        assert _pinned_criterion_hashes(_verdict({"not a label": "c" * 64})) == {}

    def test_a_verdict_with_no_checks_list_pins_nothing(self) -> None:
        assert _pinned_criterion_hashes({"checks": "not-a-list"}) == {}


# ------------------------------------------------------------ the predicate --


class TestTheFlipPredicate:
    def test_a_binding_pinned_to_a_rewritten_criterion_does_not_discharge_it(
        self,
    ) -> None:
        """AC1. The falsifier. RED before this change: the same fixture flipped."""
        reason, uncovered, rows = _ac_binding_gap(
            _description(_REWRITTEN_CRITERION),
            _verdict({"AC1": _ACCEPTED_PIN}),
            "OMN-0000",
        )

        assert reason, "a stale pin must hold the flip"
        assert uncovered
        assert "AC1" in reason
        assert not any(row.bound for row in rows)

    def test_the_hold_says_the_criterion_text_changed_after_acceptance(self) -> None:
        """AC4. The hold has to be legible on the ticket, not only in a receipt."""
        reason, _uncovered, _rows = _ac_binding_gap(
            _description(_REWRITTEN_CRITERION),
            _verdict({"AC1": _ACCEPTED_PIN}),
            "OMN-0000",
        )

        assert "OMN-18330" in reason
        assert "AC1" in reason
        assert "changed" in reason.lower()
        assert "re-accept" in reason.lower()

    def test_a_matching_pin_still_flips(self) -> None:
        """AC2. The positive control: this is not a blanket widening."""
        reason, uncovered, rows = _ac_binding_gap(
            _description(_ACCEPTED_CRITERION),
            _verdict({"AC1": _ACCEPTED_PIN}),
            "OMN-0000",
        )

        assert reason == ""
        assert uncovered == ()
        assert any(row.bound for row in rows)

    def test_a_pin_carrying_no_hash_does_not_discharge(self) -> None:
        """AC3, first half."""
        reason, uncovered, _rows = _ac_binding_gap(
            _description(_ACCEPTED_CRITERION), _verdict({"AC1": ""}), "OMN-0000"
        )

        assert reason
        assert uncovered

    def test_the_hold_tells_no_pin_apart_from_a_stale_pin(self) -> None:
        """AC3, second half. Two different repairs must not read alike."""
        stale, _u1, _r1 = _ac_binding_gap(
            _description(_REWRITTEN_CRITERION),
            _verdict({"AC1": _ACCEPTED_PIN}),
            "OMN-0000",
        )
        unpinned, _u2, _r2 = _ac_binding_gap(
            _description(_ACCEPTED_CRITERION), _verdict({"AC1": ""}), "OMN-0000"
        )

        assert "NO readable `criterion_hash`" in unpinned
        assert "NO readable `criterion_hash`" not in stale
        assert "DIFFERENT revision" in stale
        assert "DIFFERENT revision" not in unpinned

    def test_a_label_with_no_pin_is_untouched(self) -> None:
        """Out of scope, asserted rather than assumed.

        Sixty-seven live contracts declare `binds_ac` and carry no binding
        record. Holding them would be a corpus-wide widening on a pin that does
        not exist yet, and it is exactly what this test refutes.
        """
        reason, uncovered, _rows = _ac_binding_gap(
            _description(_ACCEPTED_CRITERION), _verdict(), "OMN-0000"
        )

        assert reason == ""
        assert uncovered == ()

    def test_reindenting_the_criterion_does_not_make_the_pin_stale(self) -> None:
        """Whitespace is not a rewrite.

        Scoped to whitespace WITHIN the criterion's own line, because the
        criteria reader is line-based on both sides of the join: a criterion
        re-flowed across two lines parses as a shorter item here AND in the
        producer that pinned it, so the two still agree. Spacing inside the line
        is the case a pin must survive, and this is it.
        """
        respaced = _ACCEPTED_CRITERION.replace(" after", "   after") + "   "
        reason, _uncovered, _rows = _ac_binding_gap(
            _description(respaced), _verdict({"AC1": _ACCEPTED_PIN}), "OMN-0000"
        )

        assert reason == ""

    def test_the_pin_is_resolved_inside_the_predicate_with_no_caller_switch(
        self,
    ) -> None:
        """A validation a caller can forget to pass is one that gets skipped.

        The predicate takes no pin parameter and no opt-out, so every path
        through the closer -- scheduled, dispatched, dry-run -- validates.
        """
        import inspect

        parameters = inspect.signature(_ac_binding_gap).parameters

        assert list(parameters) == ["description", "verdict", "ticket_id"]


# ------------------------------------------------- wired into the closer -----


def _issue(criterion: str) -> dict[str, object]:
    return {
        "id": "issue-1",
        "identifier": "OMN-0000",
        "state": {"id": "s1", "name": "In Progress", "type": "started"},
        "labels": {"nodes": []},
        "team": {"id": "team-1"},
        "description": _description(criterion),
        "children": {"nodes": []},
    }


class FakeLinear:
    def __init__(self, criterion: str) -> None:
        self._issue = _issue(criterion)
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
            "title": (
                "evidence(OMN-0000): OCC companion for OmniNode-ai/omnibase_infra#3194"
            ),
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


def _receipt(pins: dict[str, str] | None) -> dict[str, object]:
    return {
        "skill_name": "dod_verify",
        "node_name": "node_dod_verify",
        "status": "success",
        "correlation_id": str(uuid4()),
        "run_id": str(uuid4()),
        "exit_code": 0,
        "duration_ms": 1,
        "result": _verdict(pins),
        "result_model": _DOD_VERIFY_STATE_MODEL,
    }


def _request() -> ModelEvidenceAutocloseSweepRequest:
    return ModelEvidenceAutocloseSweepRequest(
        correlation_id=uuid4(), occ_repo=_OCC_REPO, lookback_hours=24, apply=True
    )


def _handler(linear: FakeLinear, pins: dict[str, str] | None):
    return HandlerEvidenceAutocloseSweep(
        linear_client=linear,  # type: ignore[arg-type]
        autoclose_disabled=False,
        run_gh_command=_gh_fake(),
        run_dod_verify_command=_dod_fake(_receipt(pins)),
    )


@pytest.mark.asyncio
class TestCloserIntegration:
    """The same accepted pin, held after a rewrite and flipped before it."""

    async def test_a_rewritten_criterion_is_held_and_the_comment_names_it(
        self,
    ) -> None:
        """AC1 and AC4 on the whole closer, not only on the predicate."""
        linear = FakeLinear(_REWRITTEN_CRITERION)
        handler = _handler(linear, {"AC1": _ACCEPTED_PIN})

        result = await handler.handle(_request())

        assert linear.state_updates == [], "a stale pin must not flip the ticket"
        assert [outcome.decision for outcome in result.outcomes] == [
            EnumEvidenceAutocloseDecision.GAP_AC_UNBOUND
        ]
        assert result.tickets_flipped == 0
        body = "\n".join(body for _target, body in linear.comments)
        assert "AC1" in body
        assert "OMN-18330" in body
        assert "changed" in body.lower()

    async def test_the_same_pin_still_flips_when_the_text_is_untouched(self) -> None:
        """AC2 on the whole closer. Same pin, original wording."""
        linear = FakeLinear(_ACCEPTED_CRITERION)
        handler = _handler(linear, {"AC1": _ACCEPTED_PIN})

        # The re-draw arm never flips on first observation of a verdict
        # fingerprint, so the flip is the SECOND tick.
        armed = await handler.handle(_request())
        assert [outcome.decision for outcome in armed.outcomes] == [
            EnumEvidenceAutocloseDecision.SKIPPED_REDRAW_PENDING
        ]

        result = await handler.handle(_request())

        assert linear.state_updates == [("issue-1", "state-done")]
        assert [outcome.decision for outcome in result.outcomes] == [
            EnumEvidenceAutocloseDecision.FLIPPED
        ]

    async def test_a_verdict_carrying_no_pin_at_all_is_unchanged(self) -> None:
        """The corpus control.

        Sixty-seven live contracts declare `binds_ac` and carry no binding
        record, so their verdicts carry no pin key. Not one of them may change
        behaviour because this rule exists.
        """
        linear = FakeLinear(_ACCEPTED_CRITERION)
        handler = _handler(linear, None)

        await handler.handle(_request())
        result = await handler.handle(_request())

        assert [outcome.decision for outcome in result.outcomes] == [
            EnumEvidenceAutocloseDecision.FLIPPED
        ]
