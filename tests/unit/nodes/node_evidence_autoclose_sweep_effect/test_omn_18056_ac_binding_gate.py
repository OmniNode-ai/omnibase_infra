# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18056 — a counter that never asks WHICH criterion a check covers.

The measured defect (closeout sweep run 2, 2026-09-08): 8 of 9 adjudicated
sprint tickets satisfied the closer's full flip predicate and 7 of those 8
were not done. In every held case the acceptance criterion that decides the
ticket is bound to no check in its OCC contract — and nothing in the closer
ever asked.

Reproduced live on OMN-15660 before this suite existed: dod_verify returned
``verified``, 8 total, 6 verified, 0 failed, 2 non-probative, 1
behaviour-proving. Every conjunct of the flip predicate passes and all three
:func:`_ac_coverage_gap` counting rules RELEASE (3 parsed criteria against 6
verified checks, 2 non-probative against 6 verified, zero checkboxes). Its
AC3 — "both call sites are covered; a fix that scopes only one path is
incomplete" — is demonstrably unmet in the product tree, and the word
``handler group`` does not appear anywhere in its contract.

The counters were green, the criteria were readable, and the two were never
joined. This suite is that join:

* every parseable acceptance criterion must bind to at least one VERIFIED
  probative check that DECLARES it (``binds_ac`` on the contract's evidence
  item, surfaced per check on the dod_verify verdict);
* a body with no parseable criteria is a HOLD, inverting the ``if not items:
  return "", ()`` early exit that releases today;
* a flip-eligible verdict is RE-DRAWN once — the same fingerprint has to come
  back on a later tick before it counts — so a pass manufactured by a
  one-run input change (the measured case: fast-forwarding two stale product
  clones moved OMN-16025 FAIL -> PASS with no criterion moving) cannot flip a
  ticket in the run that manufactured it.
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
_DOD_VERIFY_STATE_MODEL = (
    "omnimarket.nodes.node_dod_verify.models.model_dod_verify_state.ModelDodVerifyState"
)

# The OMN-15660 shape, rewritten with stable AC labels. Three criteria, no
# checkbox anywhere, no `Gate:` line, no PR citation — so every guard that
# exists today releases and only the binding leg can hold it.
_DESCRIPTION_THREE_LABELLED_ACS = """\
Terminal-event isolation for concurrent RuntimeLocal invocations.

## Acceptance criteria

- **AC1** RED-first: a test drives two concurrent invocations and fails on the
  current code.
- **AC2** GREEN: each invocation accepts only its own correlation-scoped
  terminal event.
- **AC3** Both call sites are covered -- a fix that scopes only one path is
  incomplete.
"""


def _merged_pr(number: int) -> dict[str, object]:
    recent = (datetime.now(tz=UTC) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "number": number,
        "html_url": f"https://github.com/{_OCC_REPO}/pull/{number}",
        "title": f"evidence({_TICKET}): OCC companion",
        "updated_at": recent,
        "merged_at": recent,
    }


def _check(
    evidence_id: str,
    status: str,
    proof_class: str,
    binds_ac: tuple[str, ...] = (),
) -> dict[str, object]:
    """One per-check record as the closer sees it on the verdict JSON.

    ``binds_ac`` is the contract's own declaration of which acceptance
    criteria this evidence item claims to cover, carried through dod_verify.
    An empty tuple is the corpus default and means the item claims none.
    """
    return {
        "evidence_id": evidence_id,
        "description": evidence_id,
        "status": status,
        "message": "OK (1ms)",
        "proof_class": proof_class,
        "binds_ac": list(binds_ac),
    }


def _skill_result(checks: list[dict[str, object]]) -> dict[str, object]:
    """A green ``onex skill dod_verify`` receipt over ``checks``.

    Counters are DERIVED from the checks exactly as omnimarket's
    ``HandlerDodVerify`` derives them, so this double cannot state a tally its
    own check list contradicts.
    """
    verified = sum(1 for c in checks if c["status"] == "verified")
    non_probative = sum(1 for c in checks if c["status"] == "non_probative")
    behavior = sum(
        1
        for c in checks
        if c["status"] == "verified" and c["proof_class"] == "behavior"
    )
    terminal: dict[str, object] = {
        "correlation_id": str(uuid4()),
        "ticket_id": _TICKET,
        "status": "verified",
        "dry_run": False,
        "checks": checks,
        "total_checks": len(checks),
        "verified_count": verified,
        "failed_count": 0,
        "skipped_count": 0,
        "superseded_count": 0,
        "non_probative_count": non_probative,
        "behavior_proving_count": behavior,
        "error_message": None,
    }
    return {
        "skill_name": "dod_verify",
        "node_name": "node_dod_verify",
        "status": "success",
        "exit_code": 0,
        "result": terminal,
        "result_model": _DOD_VERIFY_STATE_MODEL,
    }


def _omn_15660_checks(
    *,
    bound: tuple[str, ...] = ("AC1", "AC2"),
) -> list[dict[str, object]]:
    """OMN-15660's live verdict shape: 6 verified + 2 non-probative = 8.

    ``bound`` names the criteria the contract's evidence items DECLARE. The
    live contract declares none at all; the default here declares two of the
    three, which is the strictly harder case for the gate — a leg that only
    noticed a wholly unannotated contract would pass on this and still let
    the measured defect through.
    """
    checks: list[dict[str, object]] = [
        _check("dod-pr-1645-grep", "verified", "surrogate", bound[:1]),
        _check("dod-pr-1645-files", "verified", "merge-state"),
        _check("dod-deploy-assessment", "verified", "surrogate"),
        _check("dod-tests", "verified", "behavior", bound),
        _check("dod-occ-self-bind", "verified", "merge-state"),
        _check("dod-occ-admissibility", "verified", "surrogate"),
        _check("dod-pr-view", "non_probative", "merge-state"),
        _check("dod-foreign-suite", "non_probative", "surrogate"),
    ]
    return checks


class _FakeLinear:
    def __init__(self, description: str) -> None:
        self._description = description
        self.state_updates: list[tuple[str, str]] = []
        self.comments: list[tuple[str, str]] = []

    async def fetch_issue(self, ticket_id: str) -> dict[str, object]:
        return {
            "id": "issue-uuid-1",
            "identifier": _TICKET,
            "state": {"id": "s1", "name": "In Progress", "type": "started"},
            "labels": {"nodes": []},
            "children": {"nodes": []},
            "team": {"id": "team-1"},
            "description": self._description,
        }

    async def fetch_done_state_id(self, team_id: str) -> str:
        return "state-done-id"

    async def update_issue_state(self, issue_id: str, state_id: str) -> bool:
        self.state_updates.append((issue_id, state_id))
        return True

    async def fetch_issue_history(
        self, issue_id: str, page_size: int, max_pages: int
    ) -> tuple[list[dict[str, object]] | None, str]:
        return [
            {
                "id": f"entry-{index}",
                "createdAt": f"2026-09-08T00:00:{index:02d}Z",
                "actorId": None,
                "fromState": {"type": "started"},
                "toState": {"type": "completed"},
            }
            for index, (target, _state_id) in enumerate(self.state_updates, start=1)
            if target == issue_id
        ], ""

    async def create_comment(self, issue_id: str, body: str) -> bool:
        self.comments.append((issue_id, body))
        return True

    async def fetch_comment_bodies(self, issue_id: str) -> tuple[str, ...] | None:
        return tuple(body for target, body in self.comments if target == issue_id)


def _handler(
    skill_result: dict[str, object], linear: _FakeLinear
) -> HandlerEvidenceAutocloseSweep:
    async def fake_gh(args: list[str], timeout: float):
        path = args[2]
        if "/files" in path:
            return [{"filename": f"contracts/{_TICKET}.yaml"}], ""
        page = int(path.rsplit("page=", 1)[1])
        return ([_merged_pr(7000)], "") if page == 1 else ([], "")

    async def fake_dod_verify(ticket_id: str, cwd: str, timeout: float):
        return skill_result, 0, ""

    return HandlerEvidenceAutocloseSweep(
        linear_client=linear,  # type: ignore[arg-type]
        autoclose_disabled=False,
        run_gh_command=fake_gh,
        run_dod_verify_command=fake_dod_verify,
    )


def _request(**overrides: object) -> ModelEvidenceAutocloseSweepRequest:
    defaults: dict[str, object] = {
        "correlation_id": uuid4(),
        "occ_repo": _OCC_REPO,
        "lookback_hours": 24,
        "apply": False,
    }
    defaults.update(overrides)
    return ModelEvidenceAutocloseSweepRequest(**defaults)


# -- the RED control -------------------------------------------------------


async def test_an_unbound_acceptance_criterion_withholds_the_flip() -> None:
    """THE RECORDED DEFECT. AC3 is declared by no check and the closer flips.

    Every existing conjunct passes on this input: terminal ``verified``, 8
    total, 6 verified, 0 failed, 2 non-probative, 1 behaviour-proving,
    ``6 + 2 == 8``; no unchecked box; 3 parsed criteria against 6 verified
    checks; 2 non-probative against 6 verified; no ``Gate:`` line; no cited
    PR. Only AC3's absence from every ``binds_ac`` distinguishes it, and
    before this ticket nothing read that.
    """
    linear = _FakeLinear(_DESCRIPTION_THREE_LABELLED_ACS)
    handler = _handler(_skill_result(_omn_15660_checks()), linear)

    result = await handler.handle(_request(apply=True))

    outcome = result.outcomes[0]
    assert outcome.decision is EnumEvidenceAutocloseDecision.GAP_AC_UNBOUND
    assert result.tickets_flipped == 0
    assert linear.state_updates == []
    # The hold NAMES the criterion, not the arithmetic.
    assert any("AC3" in item for item in outcome.uncovered_acceptance_criteria)
    assert "AC3" in outcome.reason


# -- the GREEN counterpart: a fully bound ticket still closes ---------------


async def test_a_fully_bound_ticket_flips_on_the_second_draw() -> None:
    """The leg is a BINDING requirement, not a blanket hold.

    Same shape, with every criterion declared by a verified check. The first
    run arms the re-draw and writes no Done; the second run, recomputing the
    identical fingerprint, flips. Both halves are asserted here so a change
    that turned the re-draw into a permanent hold fails rather than reading as
    "the gate is strict".
    """
    linear = _FakeLinear(_DESCRIPTION_THREE_LABELLED_ACS)
    handler = _handler(
        _skill_result(_omn_15660_checks(bound=("AC1", "AC2", "AC3"))), linear
    )

    first = (await handler.handle(_request(apply=True))).outcomes[0]
    assert first.decision is EnumEvidenceAutocloseDecision.SKIPPED_REDRAW_PENDING
    assert linear.state_updates == []

    second_result = await handler.handle(_request(apply=True))
    second = second_result.outcomes[0]
    assert second.decision is EnumEvidenceAutocloseDecision.FLIPPED
    assert second_result.tickets_flipped == 1
    assert linear.state_updates == [("issue-uuid-1", "state-done-id")]
    # The flip STATES its bindings rather than its arithmetic.
    assert {row.label for row in second.ac_binding_rows} == {"AC1", "AC2", "AC3"}
    assert all(row.bound for row in second.ac_binding_rows)


async def test_a_changed_verdict_re_arms_the_re_draw() -> None:
    """A different fingerprint on the second tick is a NEW observation.

    The re-draw is keyed on the verdict, not on the ticket: two runs that
    disagree have observed two different things, and neither is a second
    sighting of the other.
    """
    linear = _FakeLinear(_DESCRIPTION_THREE_LABELLED_ACS)
    bound_all = _omn_15660_checks(bound=("AC1", "AC2", "AC3"))
    first = (
        await _handler(_skill_result(bound_all), linear).handle(_request(apply=True))
    ).outcomes[0]
    assert first.decision is EnumEvidenceAutocloseDecision.SKIPPED_REDRAW_PENDING

    # One more verified check -> different counters -> different fingerprint.
    widened = [*bound_all, _check("dod-extra", "verified", "behavior", ("AC3",))]
    second = (
        await _handler(_skill_result(widened), linear).handle(_request(apply=True))
    ).outcomes[0]
    assert second.decision is EnumEvidenceAutocloseDecision.SKIPPED_REDRAW_PENDING
    assert second.verdict_fingerprint != first.verdict_fingerprint
    assert linear.state_updates == []


# -- no parseable criteria is a HOLD, not a release ------------------------


async def test_a_body_with_no_parseable_criteria_holds() -> None:
    """Today's counting rules RELEASE this through `if not items`.

    "Nothing written down" is not "nothing to prove". A ticket the sweep can
    read no criteria from is one it can say nothing about, so it holds and the
    hold says what to write.
    """
    linear = _FakeLinear("Ship the thing. It is important.\n")
    handler = _handler(
        _skill_result(_omn_15660_checks(bound=("AC1", "AC2", "AC3"))), linear
    )

    outcome = (await handler.handle(_request(apply=True))).outcomes[0]

    assert outcome.decision is EnumEvidenceAutocloseDecision.GAP_AC_UNBOUND
    assert linear.state_updates == []
    assert "No acceptance criterion could be parsed" in outcome.reason


async def test_an_unlabelled_criterion_is_unbindable_and_says_so() -> None:
    """OMN-15660's REAL body: three prose bullets with no `AC<n>` label.

    `binds_ac: ["AC3"]` needs something stable to point at. The hold names the
    labelling gap explicitly, because the repair is a ticket-body change and a
    reason that only said "unbound" would send the author to the contract.
    """
    linear = _FakeLinear(
        "## Acceptance criteria\n\n"
        "- RED-first: a test drives two concurrent invocations.\n"
        "- GREEN: each invocation accepts only its own terminal event.\n"
        "- Both call sites are covered.\n"
    )
    handler = _handler(
        _skill_result(_omn_15660_checks(bound=("AC1", "AC2", "AC3"))), linear
    )

    outcome = (await handler.handle(_request(apply=True))).outcomes[0]

    assert outcome.decision is EnumEvidenceAutocloseDecision.GAP_AC_UNBOUND
    assert linear.state_updates == []
    assert "carry no `AC<n>`/`DoD<n>` label" in outcome.reason


async def test_a_verifier_that_cannot_report_bindings_is_named_as_such() -> None:
    """An ABSENT `binds_ac` and an EMPTY one are different facts.

    Absent means the verifier predates this change and cannot report a
    binding; empty means the contract was read and declares none. Only the
    second is something a ticket author can fix, so the hold distinguishes
    them instead of reporting one repair for both.
    """
    checks = [
        {k: v for k, v in check.items() if k != "binds_ac"}
        for check in _omn_15660_checks()
    ]
    linear = _FakeLinear(_DESCRIPTION_THREE_LABELLED_ACS)
    outcome = (
        await _handler(_skill_result(checks), linear).handle(_request(apply=True))
    ).outcomes[0]

    assert outcome.decision is EnumEvidenceAutocloseDecision.GAP_AC_UNBOUND
    assert linear.state_updates == []
    assert "carries no `binds_ac` on any check" in outcome.reason


async def test_a_declared_but_non_probative_binding_does_not_discharge() -> None:
    """`binds_ac` is a claim; only a VERIFIED check is a proof.

    A contract could otherwise discharge every criterion by naming it on a
    provenance entry that could not have failed.
    """
    checks = [
        _check("dod-tests", "verified", "behavior", ("AC1", "AC2")),
        _check("dod-a", "verified", "merge-state"),
        _check("dod-b", "verified", "surrogate"),
        _check("dod-c", "verified", "surrogate"),
        _check("dod-d", "verified", "merge-state"),
        _check("dod-e", "verified", "surrogate"),
        _check("dod-pr-view", "non_probative", "merge-state", ("AC3",)),
        _check("dod-foreign", "non_probative", "surrogate"),
    ]
    linear = _FakeLinear(_DESCRIPTION_THREE_LABELLED_ACS)
    outcome = (
        await _handler(_skill_result(checks), linear).handle(_request(apply=True))
    ).outcomes[0]

    assert outcome.decision is EnumEvidenceAutocloseDecision.GAP_AC_UNBOUND
    assert linear.state_updates == []
    # The near miss is legible: AC3 IS declared, by a check that proved nothing.
    ac3 = [row for row in outcome.ac_binding_rows if row.label == "AC3"]
    assert ac3 and all(not row.bound for row in ac3)
    assert ac3[0].evidence_check == "dod-pr-view"
    assert ac3[0].status.value == "non_probative"


# -- item 7: the prior-revert fence, made POSITIVE -------------------------


class _RevertedLinear(_FakeLinear):
    """A ticket that was closed and then moved back out of a completed state.

    The OMN-17298 shape: flipped 2026-09-08T21:21:29Z on a hand probe whose
    search term exists in no spelling anywhere, reverted 22:03:49Z once live
    state showed the projection node running with topic traffic. Nothing in
    that sequence carries a fingerprint THIS closer wrote, which is exactly
    why the negative fence could not see it.

    OMN-18106 made the ORDERING of that shape load-bearing, and this fixture
    did not have one. Its flip/revert pair was two fixed 2026-09-08 literals
    while ``_merged_pr`` merges its companion at ``now - 1h`` — so as the wall
    clock moved past 2026-09-08 the fixture quietly came to describe a ticket
    whose evidence landed AFTER the revert, which is the OMN-15542 shape and
    not the one the docstring names. The timestamps are therefore derived from
    the same clock the companion is, preserving the incident's real sequence:
    companion merges, somebody flips, somebody takes it back, nothing new
    lands. Every assertion below is unchanged.
    """

    async def fetch_issue_history(
        self, issue_id: str, page_size: int, max_pages: int
    ) -> tuple[list[dict[str, object]] | None, str]:
        now = datetime.now(tz=UTC)
        # Newest first, exactly as Linear's `orderBy: createdAt` returns it:
        # any completed segment this sweep has since written, then the human
        # flip/revert pair that predates the sweep entirely.
        later = [
            {
                "id": f"entry-sweep-{index}",
                "createdAt": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "actorId": None,
                "fromState": {"type": "started"},
                "toState": {"type": "completed"},
            }
            for index, (target, _state_id) in enumerate(
                reversed(self.state_updates), start=1
            )
            if target == issue_id
        ]
        return [
            *later,
            {
                "id": "entry-revert",
                "createdAt": (now - timedelta(minutes=30)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "actorId": "a-person",
                "fromState": {"type": "completed"},
                "toState": {"type": "started"},
            },
            {
                "id": "entry-flip",
                "createdAt": (now - timedelta(minutes=45)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "actorId": "a-person",
                "fromState": {"type": "started"},
                "toState": {"type": "completed"},
            },
        ], ""


async def test_a_reverted_ticket_holds_until_a_check_outcome_changes() -> None:
    """THE SECOND RECORDED DEFECT — OMN-17298-second-pass.

    The fence this replaces is NEGATIVE: it fires only when the current
    fingerprint matches a verdict this closer itself wrote. A revert of
    anybody else's flip matches nothing, so the identical evidence re-clears
    the predicate on the next tick.

    Positive form: hold, record the verdict as the post-revert baseline, and
    release only on a verdict that DIFFERS from it — the only observable this
    sweep has that a check outcome moved.
    """
    linear = _RevertedLinear(_DESCRIPTION_THREE_LABELLED_ACS)
    bound_all = _omn_15660_checks(bound=("AC1", "AC2", "AC3"))

    first = (
        await _handler(_skill_result(bound_all), linear).handle(_request(apply=True))
    ).outcomes[0]
    assert first.decision is EnumEvidenceAutocloseDecision.SKIPPED_PRIOR_REVERT
    assert "post-revert baseline" in first.reason
    assert linear.state_updates == []

    # Same evidence, next tick: still held. The decision reads
    # SKIPPED_DUPLICATE_COMMENT rather than SKIPPED_PRIOR_REVERT because the
    # OMN-16808 idempotency layer recognises that this sweep has already made
    # exactly this statement on this ticket and declines to repeat it — the
    # hold is unchanged, the comment is not duplicated, and nothing is written.
    second = (
        await _handler(_skill_result(bound_all), linear).handle(_request(apply=True))
    ).outcomes[0]
    assert second.decision is EnumEvidenceAutocloseDecision.SKIPPED_DUPLICATE_COMMENT
    assert second.applied is False
    assert linear.state_updates == []

    # A check outcome CHANGES -> the hold lifts (into the ordinary re-draw).
    widened = [*bound_all, _check("dod-new", "verified", "behavior", ("AC3",))]
    third = (
        await _handler(_skill_result(widened), linear).handle(_request(apply=True))
    ).outcomes[0]
    assert third.decision is EnumEvidenceAutocloseDecision.SKIPPED_REDRAW_PENDING
    assert linear.state_updates == []


async def test_the_revert_hold_bootstraps_and_does_not_deadlock() -> None:
    """A positive gate with no baseline to compare against must not be a wall.

    The hold that records the baseline is the same hold a later, different
    verdict is released against — so a reverted ticket whose work is genuinely
    redone can still close, in three ticks, with no human unlocking anything.
    """
    linear = _RevertedLinear(_DESCRIPTION_THREE_LABELLED_ACS)
    bound_all = _omn_15660_checks(bound=("AC1", "AC2", "AC3"))
    await _handler(_skill_result(bound_all), linear).handle(_request(apply=True))

    widened = [*bound_all, _check("dod-new", "verified", "behavior", ("AC3",))]
    await _handler(_skill_result(widened), linear).handle(_request(apply=True))
    final = (
        await _handler(_skill_result(widened), linear).handle(_request(apply=True))
    ).outcomes[0]

    assert final.decision is EnumEvidenceAutocloseDecision.FLIPPED
    assert linear.state_updates == [("issue-uuid-1", "state-done-id")]


# -- the wiring: the counter-only predicate is not reachable ---------------


async def test_the_gate_consumes_the_legs_verdict_not_merely_calls_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BEHAVIOURAL control on the source gate below, which it can defeat.

    The AST assertion that follows proves `_ac_binding_gap` is CALLED and that
    its call precedes the counting rules. It cannot prove the RESULT is acted
    on. Measured during this ticket's own verify phase: wrapping the branch as
    ``if False and ac_binding_reason:`` leaves the call in place, leaves its
    position ahead of `_ac_coverage_gap` unchanged, leaves every name the
    source gate greps for absent -- and the guard is dead. The AST test stays
    green through it.

    So this drives the seam from the outside. The leg is replaced with one
    that reports a gap on a ticket whose evidence is otherwise flawless and
    whose criteria are all genuinely bound -- a fixture that flips in
    `test_a_fully_bound_ticket_flips_on_the_second_draw` above. If the
    decision is anything but GAP_AC_UNBOUND, the run reached a verdict without
    consulting the leg it just called, whatever the source says.

    The stub takes and returns the production signature, so a change to that
    signature reddens this rather than silently passing a mismatched double.
    """
    from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers import (
        handler_evidence_autoclose_sweep as module,
    )

    sentinel = "STUBBED BINDING GAP: this criterion is bound to nothing."

    def _always_a_gap(
        description: str,
        verdict: dict[str, object],
        ticket_id: str,
    ) -> tuple[str, tuple[str, ...], tuple[object, ...]]:
        return sentinel, ("AC-STUB",), ()

    monkeypatch.setattr(module, "_ac_binding_gap", _always_a_gap)

    linear = _FakeLinear(_DESCRIPTION_THREE_LABELLED_ACS)
    handler = _handler(
        _skill_result(_omn_15660_checks(bound=("AC1", "AC2", "AC3"))), linear
    )

    result = await handler.handle(_request(apply=True))

    outcome = result.outcomes[0]
    assert outcome.decision is EnumEvidenceAutocloseDecision.GAP_AC_UNBOUND, (
        "the sweep reached a decision that ignored the binding leg's verdict "
        "— the leg is called but its result is not consumed, which is the "
        "neutering the AST gate below cannot see"
    )
    assert outcome.reason == sentinel
    assert outcome.uncovered_acceptance_criteria == ("AC-STUB",)
    assert result.tickets_flipped == 0
    assert linear.state_updates == []


async def test_the_leg_releasing_is_what_lets_the_same_ticket_flip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The positive control for the test above.

    Without it, "GAP_AC_UNBOUND when the leg reports a gap" is also satisfied
    by a sweep that holds this fixture for some other reason entirely. Same
    ticket, same evidence, a stub that RELEASES -- and it closes.
    """
    from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers import (
        handler_evidence_autoclose_sweep as module,
    )

    def _never_a_gap(
        description: str,
        verdict: dict[str, object],
        ticket_id: str,
    ) -> tuple[str, tuple[str, ...], tuple[object, ...]]:
        return "", (), ()

    monkeypatch.setattr(module, "_ac_binding_gap", _never_a_gap)

    linear = _FakeLinear(_DESCRIPTION_THREE_LABELLED_ACS)
    handler = _handler(
        _skill_result(_omn_15660_checks(bound=("AC1", "AC2", "AC3"))), linear
    )

    await handler.handle(_request(apply=True))  # arms the re-draw
    result = await handler.handle(_request(apply=True))

    assert result.outcomes[0].decision is EnumEvidenceAutocloseDecision.FLIPPED
    assert result.tickets_flipped == 1


def test_the_binding_leg_is_on_the_only_flip_path_and_has_no_off_switch() -> None:
    """A source gate, so removing the leg is a RED TEST, not a review catch.

    The closer has exactly one decision site. The scheduled workflow passes no
    arming flag and the skill mapping exposes no skip argument; this asserts
    the same posture for the binding leg — it is called inside the single
    `if all_verified:` block every trigger traverses, and no request field,
    workflow input or CLI argument can route around it.
    """
    import ast
    import inspect

    from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers import (
        handler_evidence_autoclose_sweep as module,
    )

    source = inspect.getsource(module)
    tree = ast.parse(source)

    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_ac_binding_gap" in called, (
        "_ac_binding_gap is defined but never called — the binding leg is "
        "dead code and the counter-only predicate is reachable again."
    )

    # It is called BEFORE the counting rules, so the reported hold names a
    # criterion rather than a ratio.
    assert source.index(
        "_ac_binding_gap(description, verdict, ticket_id)"
    ) < source.index("ac_gap_reason, uncovered = _ac_coverage_gap(")

    # No skip surface anywhere: not a request field, not a CLI flag.
    for forbidden in (
        "skip_ac_binding",
        "--skip-ac-binding",
        "ignore_ac_binding",
        "ac_binding_optional",
    ):
        assert forbidden not in source, (
            f"{forbidden!r} would let a caller opt out of the binding leg; "
            "the only supported way to stop this sweep is the kill switch, "
            "which stops it entirely rather than loosening what counts as "
            "proven."
        )
