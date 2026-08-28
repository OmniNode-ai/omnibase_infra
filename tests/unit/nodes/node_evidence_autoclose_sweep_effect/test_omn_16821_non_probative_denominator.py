# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16821 — the flip predicate must compare against a verdict-bearing denominator.

The blocker, stated exactly. ``_process_ticket`` released a flip only when
``verified_count == total_checks``. ``total_checks`` counts every check that
carries a verdict, and ``non_probative`` (OMN-15391 — executed, exited 0, and
its exit status cannot depend on the product change) is one of those verdicts.
A ``non_probative`` entry therefore counted in the DENOMINATOR and never in the
NUMERATOR, so the equality was unsatisfiable for any contract containing a
single one — regardless of how strong the rest of the evidence was.

``gh pr view --json state`` surrogates are the most common check shape in the
autobound OCC corpus, so this was not an edge case: measured against the merged
verifier, OMN-16260 read ``verified``/12 total/6 verified/0 failed/1
behavior-proving and still could not flip, purely on the arithmetic. That is
why the flip path had never fired once, and why a ticket in that shape would
have been told, under ``--apply``, that "not all ACs are receipt-proven" by a
verifier that had just said ``verified`` with zero failures.

The fix, and what this suite pins:

* ``verified_count + non_probative_count == total_checks`` is the honest
  equality — "every check that could carry a verdict did, and none of them
  failed". It is exactly the OMN-15390 precedent applied one axis over:
  ``total_checks`` already excludes ``superseded`` upstream on the reasoning
  that an entry carrying no product-dependent verdict does not belong in a
  verdict-bearing denominator.
* Strictness is unchanged in every other direction, and each is asserted here
  rather than argued: a real ``failed`` still gaps; a ``skipped`` check still
  gaps (it proves nothing, and it is neither verified nor non-probative);
  an ALL-non-probative contract still gaps, both by dod_verify's own terminal
  status and, independently, by the arithmetic; the OMN-15911 behavior
  conjunct and the OMN-16736 AC-coverage guard still withhold the flip on a
  verdict that now clears the denominator.
* AC4: when the verifier returned ``verified`` with zero failures, the gap
  comment must not claim the ACs are unproven. It has to name the real
  shortfall.
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


def _merged_pr(number: int) -> dict[str, object]:
    recent = (datetime.now(tz=UTC) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "number": number,
        "html_url": f"https://github.com/{_OCC_REPO}/pull/{number}",
        "title": f"evidence({_TICKET}): OCC companion",
        "updated_at": recent,
        "merged_at": recent,
    }


def _check(evidence_id: str, status: str, proof_class: str) -> dict[str, object]:
    return {
        "evidence_id": evidence_id,
        "description": evidence_id,
        "status": status,
        "message": "OK (1ms)",
        "proof_class": proof_class,
    }


def _skill_result(
    *,
    checks: list[dict[str, object]],
    include_non_probative_count: bool = True,
) -> dict[str, object]:
    """A ModelSkillResult shaped like `onex skill dod_verify` really prints.

    Every count is DERIVED from ``checks`` exactly as
    ``HandlerDodVerify._summarize`` derives it in omnimarket, including the
    terminal-status chain, so this double cannot drift into agreeing with the
    reader while disagreeing with the CLI — the OMN-16736 failure mode.

    ``total_checks`` is ``len(checks) - superseded``: the verdict-bearing
    denominator (OMN-15390). ``non_probative`` entries stay IN it, which is the
    whole subject of this suite.

    ``include_non_probative_count=False`` models a verifier predating
    OMN-15391, whose payload omits the key entirely.
    """
    verified = sum(1 for c in checks if c["status"] == "verified")
    failed = sum(1 for c in checks if c["status"] == "failed")
    skipped = sum(1 for c in checks if c["status"] == "skipped")
    non_probative = sum(1 for c in checks if c["status"] == "non_probative")
    superseded = sum(1 for c in checks if c["status"] == "superseded")
    behavior_proving = sum(
        1
        for c in checks
        if c["status"] == "verified" and c["proof_class"] == "behavior"
    )

    # omnimarket's own chain, in order (handler_dod_verify._summarize).
    if failed > 0:
        status = "failed"
    elif verified == 0 and non_probative > 0:
        # OMN-15391: nothing went wrong and nothing was proven.
        status = "skipped"
    elif superseded > 0 and verified == 0:
        status = "failed"
    elif verified == 0:
        status = "skipped"
    else:
        status = "verified"

    terminal: dict[str, object] = {
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
        "behavior_proving_count": behavior_proving,
        "error_message": None,
    }
    if include_non_probative_count:
        terminal["non_probative_count"] = non_probative
    return {
        "skill_name": "dod_verify",
        "node_name": "node_dod_verify",
        "status": "success" if failed == 0 else "failed",
        "exit_code": 0 if failed == 0 else 1,
        "result": {
            "workflow_result": "completed" if failed == 0 else "failed",
            "exit_code": 0 if failed == 0 else 1,
            "terminal_payload": terminal,
        },
    }


class _FakeLinear:
    def __init__(self, description: str | None = None) -> None:
        self.description = description
        self.state_updates: list[tuple[str, str]] = []
        self.comments: list[tuple[str, str]] = []

    async def fetch_issue(self, ticket_id: str) -> dict[str, object]:
        return {
            "id": "issue-uuid-1",
            "identifier": _TICKET,
            "state": {"id": "s1", "name": "In Progress", "type": "started"},
            "labels": {"nodes": []},
            "team": {"id": "team-1"},
            "description": self.description,
        }

    async def fetch_done_state_id(self, team_id: str) -> str:
        return "state-done-id"

    async def update_issue_state(self, issue_id: str, state_id: str) -> bool:
        self.state_updates.append((issue_id, state_id))
        return True

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
        return ([_merged_pr(7100)], "") if page == 1 else ([], "")

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


def _omn_16260_shape() -> list[dict[str, object]]:
    """The measured OMN-16260 verdict, reproduced as checks.

    ``verified`` terminal status, 12 verdict-bearing checks, 6 verified, 0
    failed, 1 behavior-proving, and the remaining 6 non-probative
    (``NON_PROBATIVE[pr_state_surrogate]``). The clean example from the ticket:
    it clears the OMN-15911 proof-class guard outright and could still not
    flip.
    """
    checks: list[dict[str, object]] = [
        _check("dod-ac1-pytest", "verified", "behavior"),
    ]
    checks += [
        _check(f"dod-ac{i}-artifact", "verified", "artifact") for i in range(2, 7)
    ]
    checks += [
        _check(f"dod-pr-{i}-state", "non_probative", "merge-state") for i in range(1, 7)
    ]
    return checks


# --------------------------------------------------------------------------
# AC2 — the shape that could never flip, now flips.
# --------------------------------------------------------------------------


async def test_omn_16260_shape_reaches_a_flip() -> None:
    """verified / 0 failed / behavior-proven, with non-probative siblings."""
    linear = _FakeLinear()
    payload = _skill_result(checks=_omn_16260_shape())
    terminal = payload["result"]["terminal_payload"]  # type: ignore[index]

    # Pin the double against the measured numbers before trusting the verdict.
    assert terminal["status"] == "verified"
    assert terminal["total_checks"] == 12
    assert terminal["verified_count"] == 6
    assert terminal["failed_count"] == 0
    assert terminal["non_probative_count"] == 6
    assert terminal["behavior_proving_count"] == 1

    result = await _handler(payload, linear).handle(_request())
    outcome = result.outcomes[0]

    assert outcome.decision is EnumEvidenceAutocloseDecision.FLIPPED
    assert result.tickets_flipped == 1
    assert outcome.dod_verify_total_checks == 12
    assert outcome.dod_verify_verified_count == 6
    assert outcome.dod_verify_behavior_proving_count == 1
    # The structured record and the stated reason must BOTH show the term that
    # released the flip, or "6/12 ACs verified" reads as an unexplained
    # shortfall to whoever audits the first automatic close-out.
    assert outcome.dod_verify_non_probative_count == 6
    assert "6/12 ACs verified (6 non-probative)" in outcome.reason
    # DRY-RUN: the decision is reached, nothing is written.
    assert outcome.applied is False
    assert linear.state_updates == []
    assert linear.comments == []


async def test_no_non_probative_entries_still_flips() -> None:
    """The pre-existing all-verified shape is unaffected by the new arithmetic."""
    linear = _FakeLinear()
    payload = _skill_result(
        checks=[
            _check("dod-ac1-pytest", "verified", "behavior"),
            _check("dod-ac2-pytest", "verified", "behavior"),
        ]
    )
    result = await _handler(payload, linear).handle(_request())
    assert result.outcomes[0].decision is EnumEvidenceAutocloseDecision.FLIPPED


# --------------------------------------------------------------------------
# AC3 — every direction that must still NOT flip.
# --------------------------------------------------------------------------


async def test_a_real_failure_still_gaps() -> None:
    linear = _FakeLinear()
    payload = _skill_result(
        checks=[
            _check("dod-ac1-pytest", "verified", "behavior"),
            _check("dod-ac2-pytest", "failed", "behavior"),
            _check("dod-pr-state", "non_probative", "merge-state"),
        ]
    )
    result = await _handler(payload, linear).handle(_request())
    outcome = result.outcomes[0]
    assert outcome.decision is EnumEvidenceAutocloseDecision.GAP_POSTED
    assert result.tickets_flipped == 0


async def test_an_all_non_probative_contract_still_gaps() -> None:
    """The merge-state-only corpus stays un-flippable.

    Two independent guards hold here and both are load-bearing: dod_verify's
    own terminal status is ``skipped`` (OMN-15391), and the arithmetic requires
    ``verified_count > 0`` so the equality cannot be satisfied by a denominator
    made entirely of non-probative entries.
    """
    linear = _FakeLinear()
    payload = _skill_result(
        checks=[
            _check(f"dod-pr-{i}-state", "non_probative", "merge-state")
            for i in range(1, 5)
        ]
    )
    terminal = payload["result"]["terminal_payload"]  # type: ignore[index]
    assert terminal["status"] == "skipped"
    assert terminal["verified_count"] == 0
    assert terminal["non_probative_count"] == terminal["total_checks"] == 4

    result = await _handler(payload, linear).handle(_request())
    assert result.outcomes[0].decision is EnumEvidenceAutocloseDecision.GAP_POSTED
    assert result.tickets_flipped == 0


async def test_a_forged_verified_status_over_all_non_probative_still_gaps() -> None:
    """The arithmetic alone must refuse it, not only dod_verify's status.

    A payload asserting ``verified`` while every check is non-probative is not
    a shape the current omnimarket produces; it is asserted here so the
    ``verified_count > 0`` conjunct cannot be dropped as redundant by someone
    reading only the status chain. If that conjunct is removed this test goes
    red and the one above stays green.
    """
    linear = _FakeLinear()
    payload = _skill_result(
        checks=[
            _check(f"dod-pr-{i}-state", "non_probative", "merge-state")
            for i in range(1, 5)
        ]
    )
    payload["result"]["terminal_payload"]["status"] = "verified"  # type: ignore[index]

    result = await _handler(payload, linear).handle(_request())
    assert result.outcomes[0].decision is EnumEvidenceAutocloseDecision.GAP_POSTED
    assert result.tickets_flipped == 0


async def test_a_skipped_check_still_gaps() -> None:
    """A skip proves nothing and is neither verified nor non-probative.

    This is the conjunct that keeps the new denominator honest: without it
    ``verified + non_probative == total`` would be satisfiable while a check
    that never ran sat in the contract.
    """
    linear = _FakeLinear()
    payload = _skill_result(
        checks=[
            _check("dod-ac1-pytest", "verified", "behavior"),
            _check("dod-pr-state", "non_probative", "merge-state"),
            _check("dod-ac3-live", "skipped", "unknown"),
        ]
    )
    terminal = payload["result"]["terminal_payload"]  # type: ignore[index]
    assert terminal["status"] == "verified"
    assert terminal["failed_count"] == 0
    assert terminal["verified_count"] + terminal["non_probative_count"] == 2
    assert terminal["total_checks"] == 3

    result = await _handler(payload, linear).handle(_request())
    assert result.outcomes[0].decision is EnumEvidenceAutocloseDecision.GAP_POSTED


async def test_the_omn_15911_behavior_conjunct_is_untouched() -> None:
    """Clearing the new denominator does not clear the proof-class guard."""
    linear = _FakeLinear()
    payload = _skill_result(
        checks=[
            _check("dod-pr-state-assert", "verified", "merge-state"),
            _check("dod-occ-admissibility", "verified", "surrogate"),
            _check("dod-pr-2-state", "non_probative", "merge-state"),
        ]
    )
    terminal = payload["result"]["terminal_payload"]  # type: ignore[index]
    assert (
        terminal["verified_count"] + terminal["non_probative_count"]
        == (terminal["total_checks"])
    )
    assert terminal["behavior_proving_count"] == 0

    result = await _handler(payload, linear).handle(_request())
    assert (
        result.outcomes[0].decision
        is EnumEvidenceAutocloseDecision.GAP_NO_BEHAVIOR_PROOF
    )


async def test_the_omn_16736_ac_coverage_guard_is_untouched() -> None:
    """An unchecked Linear-only criterion still withholds the flip."""
    linear = _FakeLinear(
        description="## Acceptance criteria\n\n- [ ] AC1 — the live lane is green\n"
    )
    result = await _handler(_skill_result(checks=_omn_16260_shape()), linear).handle(
        _request()
    )
    assert result.outcomes[0].decision is EnumEvidenceAutocloseDecision.GAP_AC_COVERAGE


async def test_a_verifier_without_the_non_probative_key_degrades_to_the_old_rule() -> (
    None
):
    """Absence is not read as zero non-probative entries in the permissive direction.

    A payload with no ``non_probative_count`` cannot distinguish "there were
    none" from "this verifier never counted them", so the term contributes 0
    and the predicate collapses back to ``verified_count == total_checks`` —
    strictly stricter than the new rule, so the fallback can only withhold a
    flip, never grant one.
    """
    linear = _FakeLinear()
    payload = _skill_result(
        checks=_omn_16260_shape(), include_non_probative_count=False
    )
    assert "non_probative_count" not in payload["result"]["terminal_payload"]  # type: ignore[operator,index]

    result = await _handler(payload, linear).handle(_request())
    assert result.outcomes[0].decision is EnumEvidenceAutocloseDecision.GAP_POSTED


# --------------------------------------------------------------------------
# AC4 — the gap comment must state the shortfall it actually found.
# --------------------------------------------------------------------------


async def test_gap_wording_does_not_claim_unproven_acs_when_nothing_failed() -> None:
    """The AC4 defect, executed.

    A ``verified`` verdict with zero failures that gaps on an unexecuted check
    must not be reported as "not all ACs are receipt-proven" — the verifier
    just said the opposite about every check it ran.
    """
    linear = _FakeLinear()
    payload = _skill_result(
        checks=[
            _check("dod-ac1-pytest", "verified", "behavior"),
            _check("dod-ac2-live", "skipped", "unknown"),
        ]
    )
    result = await _handler(payload, linear).handle(_request(apply=True))
    outcome = result.outcomes[0]

    assert outcome.decision is EnumEvidenceAutocloseDecision.GAP_POSTED
    assert "not all ACs are receipt-proven" not in outcome.reason
    assert "1 check(s) reached no verdict" in outcome.reason
    assert len(linear.comments) == 1
    _, body = linear.comments[0]
    assert "not all ACs are receipt-proven" not in body
    assert "1 check(s) reached no verdict" in body


async def test_gap_wording_still_names_a_real_failure() -> None:
    linear = _FakeLinear()
    payload = _skill_result(
        checks=[
            _check("dod-ac1-pytest", "verified", "behavior"),
            _check("dod-ac2-pytest", "failed", "behavior"),
        ]
    )
    result = await _handler(payload, linear).handle(_request(apply=True))
    outcome = result.outcomes[0]
    assert outcome.decision is EnumEvidenceAutocloseDecision.GAP_POSTED
    assert "not all ACs are receipt-proven" in outcome.reason
    assert "1 failed" in outcome.reason


async def test_gap_wording_names_the_no_probative_evidence_refusal() -> None:
    linear = _FakeLinear()
    payload = _skill_result(
        checks=[
            _check(f"dod-pr-{i}-state", "non_probative", "merge-state")
            for i in range(1, 4)
        ]
    )
    result = await _handler(payload, linear).handle(_request(apply=True))
    outcome = result.outcomes[0]
    assert outcome.decision is EnumEvidenceAutocloseDecision.GAP_POSTED
    assert "not all ACs are receipt-proven" not in outcome.reason
    assert "3 non-probative" in outcome.reason
    assert "'skipped'" in outcome.reason


async def test_gap_fingerprint_tracks_the_non_probative_count() -> None:
    """OMN-16808 dedup must key on the statement actually made.

    Two verdicts sharing (total, verified, failed) but differing in
    non-probative count now produce different gap wording, so they must not
    dedup against each other — otherwise the corrected statement is suppressed
    by the stale one.
    """
    linear = _FakeLinear()
    first = _skill_result(
        checks=[
            _check("dod-ac1-pytest", "verified", "behavior"),
            _check("dod-ac2-live", "skipped", "unknown"),
            _check("dod-ac3-live", "skipped", "unknown"),
        ]
    )
    await _handler(first, linear).handle(_request(apply=True))
    assert len(linear.comments) == 1

    second = _skill_result(
        checks=[
            _check("dod-ac1-pytest", "verified", "behavior"),
            _check("dod-ac2-live", "skipped", "unknown"),
            _check("dod-pr-state", "non_probative", "merge-state"),
        ]
    )
    second_terminal = second["result"]["terminal_payload"]  # type: ignore[index]
    first_terminal = first["result"]["terminal_payload"]  # type: ignore[index]
    assert (
        second_terminal["total_checks"],
        second_terminal["verified_count"],
        second_terminal["failed_count"],
    ) == (
        first_terminal["total_checks"],
        first_terminal["verified_count"],
        first_terminal["failed_count"],
    )

    await _handler(second, linear).handle(_request(apply=True))
    assert len(linear.comments) == 2


async def test_an_unchanged_gap_verdict_still_dedups() -> None:
    """The other direction: the OMN-16808 gate is not defeated by the new key."""
    linear = _FakeLinear()
    checks = [
        _check("dod-ac1-pytest", "verified", "behavior"),
        _check("dod-ac2-live", "skipped", "unknown"),
        _check("dod-pr-state", "non_probative", "merge-state"),
    ]
    await _handler(_skill_result(checks=checks), linear).handle(_request(apply=True))
    assert len(linear.comments) == 1
    await _handler(_skill_result(checks=checks), linear).handle(_request(apply=True))
    assert len(linear.comments) == 1
