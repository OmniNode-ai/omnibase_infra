# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16106 class (d): a check that never ran is HELD, not reported as a gap.

The class-(c) hold that shipped in `omnibase_infra#3214` fired ZERO times in
the three runs after it merged (33986056683, 33991898262, 33993316390). Its
classifier reads only checks whose `status` is `failed` or `unverifiable`
AND whose own message carries an unreachability string. Two things are wrong
with that against the real population:

* `unverifiable` is not a status any producer emits. `EnumEvidenceCheckStatus`
  is `verified | failed | skipped | superseded | non_probative`; a check the
  verifier could not evaluate is SKIPPED carrying an `unverifiable_cause`
  (`ModelEvidenceCheckResult._cause_requires_skipped` enforces exactly that).
  So half the eligible set was addressed by a name nothing answers to.
* The staging-blocked population does not terminate with a failed check at
  all. OMN-17201 — writer at replicas 0, four MSK wire topics absent — came
  back from run 33993316390 as `total=30, verified=3, failed=0,
  non_probative=26, behavior_proving=0`, terminal status `skipped`. Nothing
  FAILED; one check simply never ran. The closer read that as GAP_POSTED and
  wrote *"your acceptance criterion is not met"* onto the ticket, twice
  (21:06:39Z and 21:35:19Z), which is a false statement: the run learned
  nothing about the criterion.

Class (d) is the sibling hold for that shape, and it is deliberately a
DIFFERENT decision from class (c). "The surface was unreachable" and "the
check never executed" are different facts and an enum that spelled them the
same would make the receipt unreadable. The safety argument is identical and
is the one these tests pin: both holds sit after every flip path has already
returned, so neither is reachable from a write.
"""

from __future__ import annotations

import inspect
from typing import Any
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    HandlerEvidenceAutocloseSweep,
    _live_check_not_executed,
    _live_surface_unavailable,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_evidence_autoclose_decision import (
    EnumEvidenceAutocloseDecision,
)

from .test_handler_evidence_autoclose_sweep import (
    FakeLinearClient,
    _issue,
    _make_dod_verify_fake,
    _make_gh_fake,
    _merged_pr,
    _request,
)

pytestmark = pytest.mark.unit

_RECEIPT_SUMMARY_MODEL = (
    "omnibase_infra.cli.model_receipt_runtime_summary.ModelReceiptRuntimeSummary"
)


def _check(
    evidence_id: str,
    status: str,
    *,
    message: str | None = None,
    unverifiable_cause: str | None = None,
    unbindable_derived_overlay: bool = False,
    proof_class: str = "indeterminate",
    binds_ac: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """One `ModelEvidenceCheckResult` as it appears on the terminal payload.

    OMN-18056 added ``binds_ac``, and it is emitted ONLY when a caller names
    it. An absent key and an empty list are different facts to the binding
    gate -- absent models a verifier that predates the field and cannot report
    bindings at all -- so a default that always wrote the key would erase the
    distinction from every fixture in this file.
    """
    check: dict[str, object] = {
        "evidence_id": evidence_id,
        "description": evidence_id,
        "status": status,
        "message": message,
        "unverifiable_cause": unverifiable_cause,
        "unbindable_derived_overlay": unbindable_derived_overlay,
        "proof_class": proof_class,
    }
    if binds_ac is not None:
        check["binds_ac"] = list(binds_ac)
    return check


def _verdict(*checks: dict[str, object]) -> dict[str, object]:
    return {"checks": list(checks)}


def _receipt(
    *,
    checks: list[dict[str, object]],
    verdict_status: str,
    behavior_proving: int,
) -> dict[str, object]:
    """A dod_verify receipt whose counters are DERIVED from its own checks.

    Counters and per-check records that disagree are a payload no verifier
    can emit, and a test built on one proves nothing about production. Every
    count below is computed from `checks`, so a case cannot drift into an
    impossible shape.
    """
    statuses = [str(check["status"]) for check in checks]
    verdict: dict[str, object] = {
        "correlation_id": str(uuid4()),
        "ticket_id": "OMN-9999",
        "status": verdict_status,
        "dry_run": False,
        "checks": checks,
        "total_checks": sum(1 for s in statuses if s != "superseded"),
        "verified_count": statuses.count("verified"),
        "failed_count": statuses.count("failed"),
        "skipped_count": statuses.count("skipped"),
        "superseded_count": statuses.count("superseded"),
        "non_probative_count": statuses.count("non_probative"),
        "behavior_proving_count": behavior_proving,
        "error_message": None,
    }
    return {
        "skill_name": "dod_verify",
        "node_name": "node_dod_verify",
        "status": "failed",
        "correlation_id": str(uuid4()),
        "run_id": str(uuid4()),
        "exit_code": 1,
        "duration_ms": 1,
        "result": {
            "workflow_result": "failed",
            "exit_code": 1,
            "workflow": "<OMNI_HOME>/omnimarket/nodes/node_dod_verify/contract.yaml",
            "terminal_payload": verdict,
        },
        "result_model": _RECEIPT_SUMMARY_MODEL,
    }


def _omn_17201_run_33993316390() -> dict[str, object]:
    """The 2026-09-05T21:35Z terminal payload, by its published counters.

    `total=30, verified=3, failed=0, non_probative=26, behavior_proving=0`,
    terminal status `skipped` — read from run 33993316390's own receipt row
    for OMN-17201. The remaining check is the one that never ran: the live
    readback against the writer that sits at replicas 0.
    """
    checks: list[dict[str, object]] = [
        _check(f"dod-verified-{index}", "verified", proof_class="merge-state")
        for index in range(3)
    ]
    checks += [
        _check(
            f"dod-{index}::pr-live-state",
            "non_probative",
            message="gh pr view --json state",
            proof_class="surrogate",
        )
        for index in range(26)
    ]
    checks.append(
        _check(
            "dod-msk-wire-topics-readback",
            "skipped",
            message="0 of 4 wire topics present; writer deployment at replicas 0",
        )
    )
    return _receipt(checks=checks, verdict_status="skipped", behavior_proving=0)


def _omn_17201_run_33991898262() -> dict[str, object]:
    """The 21:06Z payload of the same ticket: `failed=2, non_probative=24`.

    Nothing about the ticket changed between the two runs — two checks moved
    from `failed` to `non_probative`. That is what minted a second comment
    under the counter-keyed fingerprint.
    """
    checks: list[dict[str, object]] = [
        _check(f"dod-verified-{index}", "verified", proof_class="merge-state")
        for index in range(3)
    ]
    checks += [
        _check(
            f"dod-{index}::pr-live-state",
            "non_probative",
            message="gh pr view --json state",
            proof_class="surrogate",
        )
        for index in range(24)
    ]
    checks += [
        _check("dod-24::pr-live-state", "failed", message="no consistent pr binding"),
        _check("dod-25::pr-live-state", "failed", message="no consistent pr binding"),
    ]
    checks.append(
        _check(
            "dod-msk-wire-topics-readback",
            "skipped",
            message="0 of 4 wire topics present; writer deployment at replicas 0",
        )
    )
    return _receipt(checks=checks, verdict_status="failed", behavior_proving=0)


# --------------------------------------------------------------------------
# The classifier
# --------------------------------------------------------------------------


def test_a_skipped_check_is_a_check_that_did_not_execute() -> None:
    evidence_id, why = _live_check_not_executed(
        _verdict(
            _check("dod-a", "verified"),
            _check(
                "dod-msk-wire-topics-readback",
                "skipped",
                message="writer deployment at replicas 0",
            ),
        )
    )
    assert evidence_id == "dod-msk-wire-topics-readback"
    assert why


def test_an_unverifiable_cause_is_named_as_the_reason() -> None:
    """The cause is a typed field, so the reason quotes it rather than prose."""
    _, why = _live_check_not_executed(
        _verdict(
            _check(
                "dod-branch-protection",
                "skipped",
                unverifiable_cause="credential_cannot_read_branch_protection",
            )
        )
    )
    assert why == "credential_cannot_read_branch_protection"


def test_an_unbindable_derived_overlay_never_executed_either() -> None:
    """OMN-17323: a synthetic overlay the binder could not bind can never pass."""
    evidence_id, why = _live_check_not_executed(
        _verdict(
            _check(
                "dod-occ-diff-derived-behavior-proof::pr-live-state",
                "non_probative",
                unbindable_derived_overlay=True,
            )
        )
    )
    assert evidence_id == "dod-occ-diff-derived-behavior-proof::pr-live-state"
    assert why


def test_a_check_that_ran_is_never_the_attribution() -> None:
    """`non_probative` RAN and exited 0 — it is not a check that could not run.

    This is the boundary that keeps the merge-state-only corpus reportable:
    a contract whose every check is a `gh pr view` surrogate still earns its
    gap comment, because nothing about it went unexecuted.
    """
    assert _live_check_not_executed(
        _verdict(
            _check("dod-a", "verified"),
            _check("dod-b", "non_probative", proof_class="surrogate"),
            _check("dod-c", "failed", message="assert 3 == 4"),
        )
    ) == ("", "")


def test_a_superseded_check_is_not_a_check_that_could_not_run() -> None:
    """OMN-15382: superseded is resolved by a LATER item, not left unproven."""
    assert _live_check_not_executed(_verdict(_check("dod-old", "superseded"))) == (
        "",
        "",
    )


@pytest.mark.parametrize(
    "verdict",
    [
        {},
        {"checks": None},
        {"checks": "not a list"},
        {"checks": []},
        {"checks": [None, 7, "x"]},
    ],
)
def test_a_payload_this_classifier_cannot_read_holds_nothing(
    verdict: dict[str, object],
) -> None:
    """Unreadable is not held — the same direction rule as class (c)."""
    assert _live_check_not_executed(verdict) == ("", "")


def test_the_two_classifiers_do_not_answer_for_each_other() -> None:
    """Class (c) and class (d) are different facts and stay separable.

    A plain skipped check naming an unreachable surface is class (d): the
    check never ran. Only a skip the verifier itself marked unverifiable, or
    a real failure, is class (c) — the surface was read and was dead.
    """
    unreachable_but_skipped = _verdict(
        _check("dod-x", "skipped", message="connection refused")
    )
    assert _live_surface_unavailable(unreachable_but_skipped) == ("", "")
    assert _live_check_not_executed(unreachable_but_skipped)[0] == "dod-x"


def test_a_skip_the_verifier_marked_unverifiable_is_class_c() -> None:
    """The REAL shape of an unverifiable check: SKIPPED plus a typed cause.

    `status: "unverifiable"` does not exist in `EnumEvidenceCheckStatus`, so
    the class-(c) classifier addressed half its eligible set by a name that
    nothing emits.
    """
    verdict = _verdict(
        _check(
            "dod-readback",
            "skipped",
            message="Unable to connect to the server: dial tcp: i/o timeout",
            unverifiable_cause="check_budget_exceeded",
        )
    )
    evidence_id, signal = _live_surface_unavailable(verdict)
    assert evidence_id == "dod-readback"
    assert signal


# --------------------------------------------------------------------------
# The decision, end to end
# --------------------------------------------------------------------------


async def _sweep(
    payloads: list[dict[str, object]],
    *,
    ticket: str = "OMN-9999",
    apply: bool = True,
) -> tuple[list[Any], FakeLinearClient]:
    """Run the sweep once per payload against the same fake board."""
    linear = FakeLinearClient(issues={ticket: _issue()})
    outcomes: list[Any] = []
    for index, payload in enumerate(payloads, start=1):
        gh_fake = _make_gh_fake(
            companions=[_merged_pr(index, f"evidence({ticket}): x", ticket)],
            files_by_pr={index: [f"contracts/{ticket}.yaml"]},
        )
        handler = HandlerEvidenceAutocloseSweep(
            linear_client=linear,
            run_gh_command=gh_fake,
            run_dod_verify_command=_make_dod_verify_fake({ticket: (payload, 0, "")}),
        )
        result = await handler.handle(_request(apply=apply))
        outcomes.append(result.outcomes[0])
    return outcomes, linear


async def test_omn_17201_is_held_and_no_comment_is_written() -> None:
    """The measured defect, stated as the run recorded it.

    Run 33993316390 wrote a gap comment on OMN-17201 asserting its acceptance
    criterion is not met. The run learned nothing: the only check that could
    have said anything never executed.
    """
    outcomes, linear = await _sweep([_omn_17201_run_33993316390()])

    assert (
        outcomes[0].decision
        == EnumEvidenceAutocloseDecision.SKIPPED_LIVE_CHECK_NOT_EXECUTED
    )
    assert linear.comments == []
    assert linear.state_updates == []
    assert outcomes[0].applied is False
    assert outcomes[0].linear_comment_posted is False
    assert "dod-msk-wire-topics-readback" in outcomes[0].reason


async def test_the_two_measured_omn_17201_payloads_produce_one_comment() -> None:
    """21:06Z then 21:35Z — the exact sequence that wrote two.

    The first payload carries real failures and is a gap. The second learned
    nothing and is held. One comment, not two.
    """
    outcomes, linear = await _sweep(
        [_omn_17201_run_33991898262(), _omn_17201_run_33993316390()]
    )

    assert outcomes[0].decision == EnumEvidenceAutocloseDecision.GAP_POSTED
    assert (
        outcomes[1].decision
        == EnumEvidenceAutocloseDecision.SKIPPED_LIVE_CHECK_NOT_EXECUTED
    )
    assert len(linear.comments) == 1


async def test_a_genuine_unmet_ac_on_a_reachable_surface_is_still_a_gap() -> None:
    """The direction that matters. An executed check that FAILED is a gap."""
    payload = _receipt(
        checks=[
            _check("dod-a", "verified", proof_class="behavior"),
            _check("dod-b", "failed", message="assert 3 == 4; 3 rows, expected 4"),
        ],
        verdict_status="failed",
        behavior_proving=1,
    )
    outcomes, linear = await _sweep([payload])

    assert outcomes[0].decision == EnumEvidenceAutocloseDecision.GAP_POSTED
    assert len(linear.comments) == 1


async def test_a_behaviour_proving_run_is_not_held_by_an_unrelated_skip() -> None:
    """The hold requires `behavior_proving == 0`.

    A run that DID execute the claimed behaviour learned something, so its
    shortfall is a real statement about the ticket even with a skip in it.
    """
    payload = _receipt(
        checks=[
            _check("dod-a", "verified", proof_class="behavior"),
            _check("dod-b", "skipped", message="never ran"),
        ],
        verdict_status="skipped",
        behavior_proving=1,
    )
    outcomes, _ = await _sweep([payload])

    assert outcomes[0].decision == EnumEvidenceAutocloseDecision.GAP_POSTED


async def test_the_merge_state_only_corpus_still_reports_its_gap() -> None:
    """No silent hold for a contract that simply proves nothing.

    Every check ran; none proved behaviour. That is a fact about the ticket
    and it keeps its comment (`gap_no_behavior_proof`).
    """
    payload = _receipt(
        checks=[
            _check(f"dod-{index}", "verified", proof_class="merge-state")
            for index in range(4)
        ],
        verdict_status="verified",
        behavior_proving=0,
    )
    outcomes, linear = await _sweep([payload])

    assert outcomes[0].decision == EnumEvidenceAutocloseDecision.GAP_NO_BEHAVIOR_PROOF
    assert len(linear.comments) == 1


async def test_a_held_candidate_is_re_offered_and_flips_when_the_check_runs() -> None:
    """The point of a hold: no human launch when the surface comes back."""
    held = _omn_17201_run_33993316390()
    # OMN-18056: the recovered corpus has to DECLARE which criterion it
    # proves, not merely come back green -- the shared `_issue` body carries
    # `AC1`, and a verified check that names no criterion discharges none.
    recovered = _receipt(
        checks=[
            _check("dod-a", "verified", proof_class="behavior", binds_ac=("AC1",)),
            _check("dod-b", "verified", proof_class="merge-state", binds_ac=("AC1",)),
        ],
        verdict_status="verified",
        behavior_proving=1,
    )
    # The third tick is the re-draw: the first eligible observation of the
    # recovered verdict arms it, the second one flips.
    outcomes, linear = await _sweep([held, recovered, recovered])

    assert (
        outcomes[0].decision
        == EnumEvidenceAutocloseDecision.SKIPPED_LIVE_CHECK_NOT_EXECUTED
    )
    assert outcomes[1].decision == EnumEvidenceAutocloseDecision.SKIPPED_REDRAW_PENDING
    assert outcomes[2].decision == EnumEvidenceAutocloseDecision.FLIPPED
    assert len(linear.state_updates) == 1


def test_the_hold_decision_is_a_skip_not_a_flip_or_a_gap() -> None:
    decision = EnumEvidenceAutocloseDecision.SKIPPED_LIVE_CHECK_NOT_EXECUTED
    assert decision.value.startswith("skipped_")
    assert decision is not EnumEvidenceAutocloseDecision.GAP_POSTED
    assert decision is not EnumEvidenceAutocloseDecision.FLIPPED


def test_both_holds_are_unreachable_from_any_write_path() -> None:
    """The placement invariant, extended to class (d).

    Each classifier has exactly one call site in `_process_ticket` and both
    sit after every `FLIPPED` return. If a later edit moves either above one,
    this reddens — the invariant is placement, not the classifier.
    """
    from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers import (
        handler_evidence_autoclose_sweep as sweep_mod,
    )

    lines = inspect.getsource(
        sweep_mod.HandlerEvidenceAutocloseSweep._process_ticket
    ).splitlines()
    flip_returns = [
        index
        for index, line in enumerate(lines)
        if "EnumEvidenceAutocloseDecision.FLIPPED" in line
    ]
    assert flip_returns, "expected at least one flip return to order against"

    for classifier in ("_live_surface_unavailable(", "_live_check_not_executed("):
        call_sites = [index for index, line in enumerate(lines) if classifier in line]
        assert len(call_sites) == 1, f"{classifier} must have exactly one call site"
        assert call_sites[0] > max(flip_returns), f"{classifier} sits above a flip"


def test_every_decision_is_tallied_in_exactly_one_bucket() -> None:
    """The receipt's four counters must partition the decision enum.

    `SKIPPED_LIVE_SURFACE_UNAVAILABLE` shipped in `#3214` without a bucket, so
    a run that held a candidate reported `flipped + gap_posted + skipped +
    errored` one short of the outcomes it carried — an unbalanced receipt with
    nothing to notice it. This reads the tally source and asserts the
    partition, so the next decision cannot be added without one.
    """
    from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers import (
        handler_evidence_autoclose_sweep as sweep_mod,
    )

    handle_source = inspect.getsource(sweep_mod.HandlerEvidenceAutocloseSweep.handle)
    # The four tally expressions only, so a decision NAMED elsewhere in
    # `handle` (the pre-read exclusion refusal, say) is not read as a bucket.
    start = handle_source.index("flipped = sum(")
    end = handle_source.index("errored = sum(")
    end = handle_source.index("        )", end) + len("        )")
    tally_source = handle_source[start:end]
    for decision in EnumEvidenceAutocloseDecision:
        occurrences = tally_source.count(
            f"EnumEvidenceAutocloseDecision.{decision.name},"
        ) + tally_source.count(f"EnumEvidenceAutocloseDecision.{decision.name}\n")
        assert occurrences == 1, (
            f"{decision.name} appears in {occurrences} tally buckets; every "
            "decision belongs to exactly one of flipped/gap_posted/skipped/"
            "errored"
        )

    # OMN-16106 D3. The partition above says every decision is COUNTED once.
    # This says no decision also carries a run-level effect it was never
    # supposed to have. The auto-disarm used to key on
    # `SKIPPED_PRIOR_REVERT` — a member of the `skipped` bucket — so one
    # correctly-refused ticket silently converted every remaining candidate in
    # the run into `SKIPPED_DISARMED`. A decision is a statement about ONE
    # candidate; the only thing that may stop the run is a flip this run wrote
    # and then watched a person undo.
    disarm_start = handle_source.index("if outcome.flip_reverted_during_run")
    disarm_source = handle_source[
        disarm_start : handle_source.index("flipped = sum(", disarm_start)
    ]
    named = [
        decision.name
        for decision in EnumEvidenceAutocloseDecision
        if f"EnumEvidenceAutocloseDecision.{decision.name}" in disarm_source
    ]
    assert named == [], (
        f"the run-disarm trigger reads decision(s) {named}. A per-candidate "
        "decision must never disarm the run: the OMN-17556 refusal was "
        "recomputed from unchanged evidence on every tick, so keying the "
        "disarm on it stopped the fleet indefinitely. The trigger is "
        "`outcome.flip_reverted_during_run` and nothing else."
    )
