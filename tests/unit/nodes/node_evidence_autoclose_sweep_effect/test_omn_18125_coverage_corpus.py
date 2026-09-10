# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18125 — the counting rules must judge the checks that CLAIM coverage.

THE DEFECT, MEASURED
--------------------
OMN-17478, closer run ``34431989659``, comment ``e1cb7e79`` at
2026-09-10T03:10:44Z. Its OCC contract at ``70ae3b98`` binds all six
acceptance criteria to VERIFIED probative checks, so the OMN-18056 binding
leg RELEASES — the recorded class moved from ``gap_ac_unbound`` to
``gap_ac_coverage``. ``dod_verify`` read 12/34 verified, 0 failed.

The closer held anyway, on one counting rule, quoted from the comment it
wrote:

    dod_verify returned 22 non-probative check(s) against 12 verified
    probative one(s), so at least half of the corpus that was supposed to
    prove this ticket's 6 acceptance criterion(s) proved nothing.

**Those 22 entries were never that corpus.** They are auto-minted
Evidence-Source provenance items from four earlier companion PRs — the
``-ci`` product-diff-scope items, the ``occ-self-bind-pr-<n>`` items and the
bare PR-state items — each contributing itself plus a derived live-state
overlay. Every one is a ``gh pr view --json ...`` that exits 0 for any
visible PR, which is exactly why omnimarket's ``_non_probative_reason``
demotes them. **None declares** ``binds_ac``, so none claims to cover any
acceptance criterion.

So the rule refuted a claim about coverage with a population that is mostly
outside coverage. The population grows with every companion a ticket ever
attracts, which makes this fleet-wide and monotone: the longer a ticket's
autobind history, the harder its own evidence has to work to out-vote it.

WHAT CHANGES, AND WHAT DELIBERATELY DOES NOT
--------------------------------------------
The rule's ARITHMETIC is correct and untouched; it was being handed the
wrong numbers. Both terms of the bounded-minority rule now come from
:func:`_coverage_corpus_counts` — the checks whose ``binds_ac`` names a
criterion parsed from this ticket's body.

The other counting rule, "the section lists more items than there are
verified probative checks", deliberately keeps the verdict-wide count. It
asks a different question, and narrowing it would hold a contract whose one
integration proof legitimately declares three criteria — measured on the
OMN-18056 fixture ``test_a_fully_bound_ticket_flips_on_the_second_draw``,
which that narrowing turned red. Authorial over-claim is a real question and
it is not this one.

The anti-padding property is preserved BY CONSTRUCTION and is asserted here
rather than assumed:

* a non-probative check can never join the numerator, because
  :func:`_ac_binding_gap` binds only on ``status == verified`` and
  omnimarket demotes exactly the greens. A bare ``occ-self-bind-pr-<n>``
  cannot raise coverage whether or not it declares ``binds_ac``;
* a non-probative check that DOES declare ``binds_ac`` lands in the
  DENOMINATOR, so declaring one hurts the contract that declares it;
* an unbound check — provenance or otherwise — moves neither term.

This is NOT OMN-18075. That ticket stops the minting path emitting the
self-bind at all, on the 2026-09-09 ruling that the fix belongs at the mint
and not at the read. Its own Out of scope names this change as the separate
question it is: *"changing the strict-majority rule to per-criterion
coverage ... Separate question, separate ticket if wanted."* Nothing here
special-cases a self-bind by name or shape, and nothing here removes the
reason OMN-18075 exists.
"""

from __future__ import annotations

from typing import Any
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
from tests.unit.nodes.node_evidence_autoclose_sweep_effect._ac_binding_support import (
    redraw_marker_comment,
)

pytestmark = pytest.mark.unit

_OCC_REPO = "OmniNode-ai/onex_change_control"
_TICKET = "OMN-17478"
_DOD_VERIFY_STATE_MODEL = (
    "omnimarket.nodes.node_dod_verify.models.model_dod_verify_state.ModelDodVerifyState"
)

#: OMN-17478's six labelled criteria, shortened to their first clause. The
#: LABELS are what matter — `binds_ac` joins on the canonical `AC<n>` form,
#: so a fixture whose criteria were unlabelled would exercise a shape the
#: binding leg refuses before this rule is ever reached.
_SIX_AC_DESCRIPTION = (
    "## Acceptance criteria\n"
    "\n"
    "- **AC1 (RED-first).** A check that fails today against the live plane.\n"
    "- **AC2.** Root-cause and fix the staging failure that precedes the repin.\n"
    "- **AC3 (live).** A staging run completes green on a fresh digest.\n"
    "- **AC4 (live).** The tenant-registry writer is Ready and its mirror fills.\n"
    "- **AC5.** Fix the concurrency-group collision.\n"
    "- **AC6.** A fail-closed gate refuses to report a stale plane green.\n"
)

_SIX_LABELS = ("AC1", "AC2", "AC3", "AC4", "AC5", "AC6")


def _check(
    evidence_id: str,
    status: str,
    proof_class: str,
    *,
    binds_ac: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """One check record on the terminal payload.

    ``binds_ac`` is written only when a caller names it: an ABSENT key models
    an item that declares no coverage, which is precisely what every
    auto-minted provenance item does.
    """
    check: dict[str, Any] = {
        "evidence_id": evidence_id,
        "description": evidence_id,
        "status": status,
        "message": "OK (1ms)",
        "proof_class": proof_class,
    }
    if binds_ac is not None:
        check["binds_ac"] = list(binds_ac)
    return check


def _omn_17478_checks() -> list[dict[str, Any]]:
    """The measured OMN-17478 verdict shape: 34 checks, 12 verified, 22 not.

    Six hand-authored items, one per criterion, two verified checks each —
    that is the 12. Eleven auto-minted provenance items, each contributing
    itself plus its derived ``::pr-live-state`` overlay — that is the 22.
    None of the 22 declares a binding, because the autobind path does not
    write one.
    """
    checks: list[dict[str, Any]] = []
    for index, label in enumerate(_SIX_LABELS, start=1):
        checks.append(
            _check(
                f"dod-omn17478-ac{index}-proof",
                "verified",
                "behavior",
                binds_ac=(label,),
            )
        )
        checks.append(
            _check(
                f"dod-omn17478-ac{index}-readback",
                "verified",
                "behavior",
                binds_ac=(label,),
            )
        )
    for index in range(1, 12):
        checks.append(
            _check(f"dod-autobind-pr-{index}", "non_probative", "merge-state")
        )
        checks.append(
            _check(
                f"dod-autobind-pr-{index}::pr-live-state",
                "non_probative",
                "merge-state",
            )
        )
    return checks


def _skill_result(checks: list[dict[str, Any]]) -> dict[str, Any]:
    """A ModelSkillResult shaped like ``onex skill dod_verify`` really prints.

    Every counter is DERIVED from ``checks`` exactly as omnimarket's
    ``HandlerDodVerify._summarize`` derives it, so this double cannot drift
    into agreeing with the reader while disagreeing with the CLI.
    """
    verified = sum(1 for c in checks if c["status"] == "verified")
    failed = sum(1 for c in checks if c["status"] == "failed")
    skipped = sum(1 for c in checks if c["status"] == "skipped")
    non_probative = sum(1 for c in checks if c["status"] == "non_probative")
    superseded = sum(1 for c in checks if c["status"] == "superseded")
    behavior = sum(
        1
        for c in checks
        if c["status"] == "verified" and c["proof_class"] == "behavior"
    )
    if failed > 0:
        status = "failed"
    elif verified == 0:
        status = "skipped"
    else:
        status = "verified"
    terminal: dict[str, Any] = {
        "correlation_id": str(uuid4()),
        "ticket_id": _TICKET,
        "status": status,
        "dry_run": False,
        "checks": checks,
        "total_checks": len(checks) - superseded,
        "verified_count": verified,
        "failed_count": failed,
        "skipped_count": skipped,
        "superseded_count": superseded,
        "non_probative_count": non_probative,
        "behavior_proving_count": behavior,
        "error_message": None,
    }
    if status == "verified":
        return {
            "skill_name": "dod_verify",
            "node_name": "node_dod_verify",
            "status": "success",
            "exit_code": 0,
            "result": terminal,
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
            "terminal_payload": terminal,
        },
        "result_model": "omnibase_infra.cli.model_receipt_runtime_summary.ModelReceiptRuntimeSummary",
    }


class _FakeLinear:
    def __init__(self, description: str = _SIX_AC_DESCRIPTION) -> None:
        self.description = description
        self.state_updates: list[tuple[str, str]] = []
        self.comments: list[tuple[str, str]] = []

    async def fetch_issue(self, ticket_id: str) -> dict[str, Any]:
        return {
            "id": "issue-uuid-1",
            "identifier": _TICKET,
            "state": {"id": "s1", "name": "In Progress", "type": "started"},
            "labels": {"nodes": []},
            "children": {"nodes": []},
            "team": {"id": "team-1"},
            "description": self.description,
        }

    async def fetch_done_state_id(self, team_id: str) -> str:
        return "state-done-id"

    async def update_issue_state(self, issue_id: str, state_id: str) -> bool:
        self.state_updates.append((issue_id, state_id))
        return True

    async def fetch_issue_history(
        self, issue_id: str, page_size: int, max_pages: int
    ) -> tuple[list[dict[str, Any]] | None, str]:
        return [
            {
                "id": f"entry-{index}",
                "createdAt": f"2026-09-10T00:00:{index:02d}Z",
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


def _merged_pr(number: int) -> dict[str, Any]:
    recent = "2026-09-10T02:00:00Z"
    return {
        "number": number,
        "html_url": f"https://github.com/{_OCC_REPO}/pull/{number}",
        "title": f"evidence({_TICKET}): OCC companion",
        "updated_at": recent,
        "merged_at": recent,
    }


def _handler(
    skill_result: dict[str, Any], linear: _FakeLinear
) -> HandlerEvidenceAutocloseSweep:
    async def fake_gh(args: list[str], timeout: float):
        path = args[2]
        if "/files" in path:
            return [{"filename": f"contracts/{_TICKET}.yaml"}], ""
        page = int(path.rsplit("page=", 1)[1])
        return ([_merged_pr(8859)], "") if page == 1 else ([], "")

    async def fake_dod_verify(ticket_id: str, cwd: str, timeout: float):
        return skill_result, 0, ""

    return HandlerEvidenceAutocloseSweep(
        linear_client=linear,  # type: ignore[arg-type]
        autoclose_disabled=False,
        run_gh_command=fake_gh,
        run_dod_verify_command=fake_dod_verify,
    )


def _request(**overrides: Any) -> ModelEvidenceAutocloseSweepRequest:
    defaults: dict[str, Any] = {
        "correlation_id": uuid4(),
        "occ_repo": _OCC_REPO,
        "lookback_hours": 24,
        "apply": False,
    }
    defaults.update(overrides)
    return ModelEvidenceAutocloseSweepRequest(**defaults)


def _seed_redraw(linear: _FakeLinear, checks: list[dict[str, Any]]) -> None:
    """Arm the OMN-18056 re-draw for this verdict.

    A DRY-RUN writes nothing, so it can never arm its own re-draw; the marker
    an APPLY tick would have left is seeded instead. Without it every fixture
    below would stop at ``skipped_redraw_pending`` and could say nothing about
    the coverage rule.
    """
    terminal = _skill_result(checks)
    body = terminal["result"]
    if "terminal_payload" in body:
        body = body["terminal_payload"]
    linear.comments.append(
        (
            "issue-uuid-1",
            redraw_marker_comment(
                total_checks=body["total_checks"],
                verified_count=body["verified_count"],
                failed_count=body["failed_count"],
                non_probative_count=body["non_probative_count"],
                behavior_proving_count=body["behavior_proving_count"],
            ),
        )
    )


# --------------------------------------------------------------------------
# AC1 / AC2 — the measured shape must stop being held.
# --------------------------------------------------------------------------


async def test_the_omn_17478_shape_is_not_held_by_the_coverage_rule() -> None:
    """RED. 12 verified bound, 22 non-probative bound to nothing.

    The whole finding in one assertion: every criterion resolves to a
    verified probative check, and the only thing standing between this
    verdict and a flip is a ratio computed over items that claim to cover
    nothing.
    """
    checks = _omn_17478_checks()
    # Pin the double against the numbers the closer actually reported before
    # trusting anything it says.
    assert len(checks) == 34
    assert sum(1 for c in checks if c["status"] == "verified") == 12
    assert sum(1 for c in checks if c["status"] == "non_probative") == 22
    assert all("binds_ac" not in c for c in checks if c["status"] == "non_probative")

    linear = _FakeLinear()
    _seed_redraw(linear, checks)
    result = await _handler(_skill_result(checks), linear).handle(_request())
    outcome = result.outcomes[0]

    assert outcome.decision is not EnumEvidenceAutocloseDecision.GAP_AC_COVERAGE
    assert outcome.uncovered_acceptance_criteria == ()
    # The counters the comment renders are still the verdict's own, unchanged:
    # only the RULE's population narrowed, and a receipt that hid the 22
    # would be less honest, not more.
    assert outcome.dod_verify_verified_count == 12
    assert outcome.dod_verify_non_probative_count == 22


async def test_the_omn_17478_shape_reaches_the_flip() -> None:
    """Positive control for the RED above.

    Without this, the RED is also satisfied by a change that swapped one hold
    for another, which would look like progress on the board and be none.
    """
    checks = _omn_17478_checks()
    linear = _FakeLinear()
    _seed_redraw(linear, checks)
    result = await _handler(_skill_result(checks), linear).handle(_request())
    assert result.outcomes[0].decision is EnumEvidenceAutocloseDecision.FLIPPED


# --------------------------------------------------------------------------
# AC3 — every direction that must still HOLD.
# --------------------------------------------------------------------------


async def test_a_majority_non_probative_coverage_corpus_still_holds() -> None:
    """The rule keeps its teeth where its premise is actually true.

    Same six criteria, but the checks that CLAIM to cover them are mostly
    non-probative: 6 verified against 12 that carry no product verdict. That
    is the corpus the rule's own sentence describes, and it must refuse.
    """
    checks: list[dict[str, Any]] = []
    for index, label in enumerate(_SIX_LABELS, start=1):
        checks.append(
            _check(f"dod-ac{index}-proof", "verified", "behavior", binds_ac=(label,))
        )
        checks.append(
            _check(
                f"dod-ac{index}-surrogate-a",
                "non_probative",
                "merge-state",
                binds_ac=(label,),
            )
        )
        checks.append(
            _check(
                f"dod-ac{index}-surrogate-b",
                "non_probative",
                "merge-state",
                binds_ac=(label,),
            )
        )
    linear = _FakeLinear()
    _seed_redraw(linear, checks)
    result = await _handler(_skill_result(checks), linear).handle(_request())
    outcome = result.outcomes[0]

    assert outcome.decision is EnumEvidenceAutocloseDecision.GAP_AC_COVERAGE
    assert "12 non-probative check(s) against 6 verified probative one(s)" in (
        outcome.reason
    )
    assert result.tickets_flipped == 0


async def test_more_criteria_than_covering_checks_still_holds() -> None:
    """One check declaring it covers all six criteria is an over-claim.

    `binds_ac` is the author's CLAIM, and OMN-18056 states that honest limit
    outright: it can prove a declared check ran and was probative, never that
    the check proves the criterion. The counting bound is the cheap refusal
    that survives — six criteria cannot be covered by fewer than six checks.
    """
    checks = [
        _check(
            "dod-one-check-covers-everything",
            "verified",
            "behavior",
            binds_ac=_SIX_LABELS,
        )
    ]
    linear = _FakeLinear()
    _seed_redraw(linear, checks)
    result = await _handler(_skill_result(checks), linear).handle(_request())
    outcome = result.outcomes[0]

    assert outcome.decision is EnumEvidenceAutocloseDecision.GAP_AC_COVERAGE
    assert "lists 6 item(s)" in outcome.reason
    assert result.tickets_flipped == 0


async def test_an_unchecked_checkbox_still_holds() -> None:
    """Rule 1 is a different fact and is not touched by any of this."""
    linear = _FakeLinear(
        description="## Acceptance criteria\n\n- [ ] AC1 — the live lane is green\n"
    )
    checks = [_check("dod-ac1-proof", "verified", "behavior", binds_ac=("AC1",))]
    _seed_redraw(linear, checks)
    result = await _handler(_skill_result(checks), linear).handle(_request())
    assert result.outcomes[0].decision is EnumEvidenceAutocloseDecision.GAP_AC_COVERAGE


# --------------------------------------------------------------------------
# AC4 — the OMN-18075 padding loophole stays closed.
# --------------------------------------------------------------------------


async def test_minting_more_provenance_does_not_release_a_real_hold() -> None:
    """Adding companions cannot move a verdict the rule correctly refuses.

    The failure this guards against is the mirror of the one being fixed: a
    narrowed population that a contract could pad its way OUT of. It cannot —
    an unbound check is in neither term, so 40 more of them change nothing.
    """
    base: list[dict[str, Any]] = []
    for index, label in enumerate(_SIX_LABELS, start=1):
        base.append(
            _check(f"dod-ac{index}-proof", "verified", "behavior", binds_ac=(label,))
        )
        base.append(
            _check(
                f"dod-ac{index}-surrogate-a",
                "non_probative",
                "merge-state",
                binds_ac=(label,),
            )
        )
        base.append(
            _check(
                f"dod-ac{index}-surrogate-b",
                "non_probative",
                "merge-state",
                binds_ac=(label,),
            )
        )
    padded = base + [
        _check(f"occ-self-bind-pr-{n}", "non_probative", "surrogate")
        for n in range(9000, 9020)
    ]

    linear = _FakeLinear()
    _seed_redraw(linear, padded)
    result = await _handler(_skill_result(padded), linear).handle(_request())
    outcome = result.outcomes[0]

    assert outcome.decision is EnumEvidenceAutocloseDecision.GAP_AC_COVERAGE
    # The stated arithmetic is identical to the un-padded case: the 20 added
    # entries are in neither term.
    assert "12 non-probative check(s) against 6 verified probative one(s)" in (
        outcome.reason
    )


async def test_a_self_bind_that_declares_a_criterion_hurts_its_own_contract() -> None:
    """A non-probative check naming a criterion lands in the DENOMINATOR.

    So the read-time shape the 2026-09-09 ruling rejected is not reachable
    here by declaring a binding either: the self-bind cannot bind (it never
    verifies) and it cannot hide (it is counted against the contract that
    declared it).
    """
    covered = [
        _check("dod-ac1-proof", "verified", "behavior", binds_ac=("AC1",)),
        _check("dod-ac2-proof", "verified", "behavior", binds_ac=("AC2",)),
    ]
    description = (
        "## Acceptance criteria\n\n- **AC1** the first thing.\n- **AC2** the second.\n"
    )

    clean = _FakeLinear(description=description)
    _seed_redraw(clean, covered)
    clean_result = await _handler(_skill_result(covered), clean).handle(_request())
    assert clean_result.outcomes[0].decision is EnumEvidenceAutocloseDecision.FLIPPED

    declared = covered + [
        _check(
            f"occ-self-bind-pr-{n}",
            "non_probative",
            "surrogate",
            binds_ac=("AC1", "AC2"),
        )
        for n in (9001, 9002)
    ]
    dirty = _FakeLinear(description=description)
    _seed_redraw(dirty, declared)
    dirty_result = await _handler(_skill_result(declared), dirty).handle(_request())
    assert (
        dirty_result.outcomes[0].decision
        is EnumEvidenceAutocloseDecision.GAP_AC_COVERAGE
    )
    assert "2 non-probative check(s) against 2 verified probative one(s)" in (
        dirty_result.outcomes[0].reason
    )


async def test_a_check_bound_to_a_criterion_this_ticket_does_not_list_is_excluded() -> (
    None
):
    """The corpus is joined on THIS ticket's criteria, not on any label.

    A contract that declared ``binds_ac: ["AC9"]`` on a pile of verified
    checks would otherwise inflate the numerator of a six-criterion ticket
    with checks covering a criterion nobody wrote down.
    """
    checks: list[dict[str, Any]] = [
        _check(f"dod-ac{i}-proof", "verified", "behavior", binds_ac=(f"AC{i}",))
        for i in range(1, 7)
    ]
    checks += [
        _check(
            f"dod-ac9-elsewhere-{n}",
            "verified",
            "behavior",
            binds_ac=("AC9",),
        )
        for n in range(1, 9)
    ]
    checks += [
        _check(
            f"dod-ac{i}-surrogate",
            "non_probative",
            "merge-state",
            binds_ac=(f"AC{i}",),
        )
        for i in range(1, 7)
    ]
    linear = _FakeLinear()
    _seed_redraw(linear, checks)
    result = await _handler(_skill_result(checks), linear).handle(_request())
    outcome = result.outcomes[0]

    # 6 verified against 6 non-probative INSIDE the corpus is a tie, and
    # OMN-16106 settled that a tie refutes nothing and therefore holds. The
    # eight AC9 checks do not rescue it.
    assert outcome.decision is EnumEvidenceAutocloseDecision.GAP_AC_COVERAGE
    assert "6 non-probative check(s) against 6 verified probative one(s)" in (
        outcome.reason
    )


# --------------------------------------------------------------------------
# The wiring, so a later refactor cannot silently revert this.
# --------------------------------------------------------------------------


def test_the_call_site_feeds_the_coverage_corpus_not_the_verdict_totals() -> None:
    """`_ac_coverage_gap` is reached with corpus counts, by construction.

    Rule 5 — enforcement, not detection. A change that restored the global
    counters here would pass every behavioural test above only by accident of
    fixture arithmetic; this reads the call site itself.
    """
    from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers import (
        handler_evidence_autoclose_sweep as module,
    )

    source = module.__file__
    assert source is not None
    with open(source, encoding="utf-8") as handle:
        text = handle.read()

    corpus_call = text.index("coverage_verified, coverage_non_probative = ")
    gap_call = text.index("ac_gap_reason, uncovered = _ac_coverage_gap(")
    assert corpus_call < gap_call, "the corpus must be computed before the rule runs"
    call = text[gap_call : text.index("\n            )", gap_call)]
    # Rule 3's two terms are the corpus.
    assert "coverage_verified," in call
    assert "coverage_non_probative," in call
    # Rule 2's term is deliberately still the verdict-wide count, and
    # `non_probative_count` — the verdict-wide term rule 3 used to read — is
    # gone from this call entirely.
    assert "verified_count," in call
    assert "non_probative_count," not in call
