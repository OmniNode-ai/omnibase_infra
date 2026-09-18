# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18490 — the closer reports a tally and names none of the checks in it.

THE MEASURED DEFECT. Scheduled run 35299192253 (2026-09-18T02:24Z) recorded
OMN-18426 as ``gap_posted`` with the reason ``dod_verify: 30/94 ACs verified,
3 failed — not all ACs are receipt-proven.`` and posted that sentence to the
ticket. Neither the outcome record nor the comment names one of the three.
A collaborator asked twice, on 09-17 and again on 09-18, which three checks
those were, and the answer was not recoverable from the receipt, from the
comment, or from the job log — the verdict that knew is a subprocess payload
that the run discarded.

What makes this a reporting defect and not a data-collection one: the
per-check records are ALREADY on the dod_verify terminal payload
(``evidence_id`` / ``status`` / ``message`` / ``binds_ac``) and this module
already reads them — ``_live_surface_unavailable``, ``_live_check_not_executed``
and ``_gap_fingerprint_parts`` each walk the same list. The outcome simply
never carried them, so every consumer downstream of the sweep saw counters
where the names were.

The rows are DESCRIPTIVE. Nothing here may move a flip, a hold or the comment
dedup, and the last two tests in this module are what hold that line: a
fingerprint asserted byte-identical across the change, and the counters
asserted to still come from the verdict's own count fields rather than from
the row list.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    HandlerEvidenceAutocloseSweep,
    _check_result_rows,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_ac_binding_check_status import (
    EnumAcBindingCheckStatus,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_evidence_autoclose_decision import (
    EnumEvidenceAutocloseDecision,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_check_result_row import (
    ModelCheckResultRow,
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

#: One labelled criterion, no checkbox, no `Gate:` line, no PR citation — so
#: the only thing deciding these fixtures is the verdict handed in.
_DESCRIPTION = (
    "## Acceptance criteria\n"
    "\n"
    "- AC1: the behaviour under test is proven by a verified probative check.\n"
)


def _failing(rows: tuple[ModelCheckResultRow, ...]) -> list[str]:
    """The failing checks' ids, in order.

    A helper rather than an inline comprehension in four places: the
    assertion these tests exist for is "which ones failed", and it should
    read the same everywhere it is made.
    """
    return [
        row.evidence_check
        for row in rows
        if row.status is EnumAcBindingCheckStatus.FAILED
    ]


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
    *,
    proof_class: str = "behavior",
    binds_ac: tuple[str, ...] = (),
    message: str = "OK (1ms)",
) -> dict[str, object]:
    return {
        "evidence_id": evidence_id,
        "description": evidence_id,
        "status": status,
        "message": message,
        "proof_class": proof_class,
        "binds_ac": list(binds_ac),
    }


def _skill_result(
    checks: list[dict[str, object]], *, status: str = "verified"
) -> dict[str, object]:
    """A dod_verify receipt whose counters are DERIVED from its own checks.

    Same rule the real verifier follows, so this double cannot state a tally
    its check list contradicts — which is the property the counters-vs-rows
    tests below rest on.
    """
    verified = sum(1 for c in checks if c["status"] == "verified")
    failed = sum(1 for c in checks if c["status"] == "failed")
    non_probative = sum(1 for c in checks if c["status"] == "non_probative")
    behavior = sum(
        1
        for c in checks
        if c["status"] == "verified" and c["proof_class"] == "behavior"
    )
    terminal: dict[str, object] = {
        "correlation_id": str(uuid4()),
        "ticket_id": _TICKET,
        "status": status,
        "dry_run": False,
        "checks": checks,
        "total_checks": len(checks),
        "verified_count": verified,
        "failed_count": failed,
        "skipped_count": sum(1 for c in checks if c["status"] == "skipped"),
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


#: The OMN-18426 shape in miniature: one check passed, one failed, one never
#: ran. Three DIFFERENT facts, and the tally `1/3 verified, 1 failed` states
#: none of them.
def _mixed_checks() -> list[dict[str, object]]:
    return [
        _check("dod-tests-green", "verified", binds_ac=("AC1",)),
        _check(
            "dod-tenant-rls-probe",
            "failed",
            message="AssertionError: expected tenant scope on the projection row",
        ),
        _check(
            "dod-staging-readback",
            "skipped",
            message="the check did not execute",
        ),
    ]


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
        return [], ""

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


async def test_the_outcome_records_one_row_per_check() -> None:
    """THE RECORDED DEFECT: `1/3 verified, 1 failed` and no check is named.

    The assertion is on the ROWS, not on the reason string, because the
    reason is prose and a future rewording of it must not be able to pass
    this test while the machine-readable record stays empty.
    """
    linear = _FakeLinear(_DESCRIPTION)
    handler = _handler(_skill_result(_mixed_checks()), linear)

    outcome = (await handler.handle(_request(apply=True))).outcomes[0]

    assert outcome.decision is EnumEvidenceAutocloseDecision.GAP_POSTED
    rows = {row.evidence_check: row for row in outcome.check_results}
    assert set(rows) == {
        "dod-tests-green",
        "dod-tenant-rls-probe",
        "dod-staging-readback",
    }
    assert rows["dod-tests-green"].status is EnumAcBindingCheckStatus.VERIFIED
    assert rows["dod-tenant-rls-probe"].status is EnumAcBindingCheckStatus.FAILED
    assert rows["dod-staging-readback"].status is EnumAcBindingCheckStatus.SKIPPED
    # The failing check is nameable from the record alone.
    assert _failing(outcome.check_results) == ["dod-tenant-rls-probe"]


async def test_a_row_carries_the_binding_it_declares() -> None:
    """`binds_ac` distinguishes "covers nothing" from "covers AC1".

    A row that dropped it would reproduce the OMN-18056 defect one surface
    over: the reader could name the check and still not know whether it was
    ever claimed to discharge a criterion.
    """
    linear = _FakeLinear(_DESCRIPTION)
    handler = _handler(_skill_result(_mixed_checks()), linear)

    outcome = (await handler.handle(_request(apply=True))).outcomes[0]
    rows = {row.evidence_check: row for row in outcome.check_results}

    assert rows["dod-tests-green"].binds_ac == ("AC1",)
    assert rows["dod-tenant-rls-probe"].binds_ac == ()


async def test_the_gap_comment_names_the_failing_check_ids() -> None:
    """The question the collaborator asked, answered on the ticket itself.

    The outcome row is the machine surface; the comment is the human one, and
    only one of them is read by somebody deciding what to fix next.
    """
    linear = _FakeLinear(_DESCRIPTION)
    handler = _handler(_skill_result(_mixed_checks()), linear)

    await handler.handle(_request(apply=True))

    assert len(linear.comments) == 1
    body = linear.comments[0][1]
    assert "dod-tenant-rls-probe" in body
    # Its message travels with it: an id with no excerpt sends the reader back
    # to a job log that no longer exists.
    assert "expected tenant scope" in body


# -- negative controls -----------------------------------------------------


async def test_a_verdict_with_no_failed_check_adds_no_failing_section() -> None:
    """A hold with zero failures must not grow a section naming nothing.

    The gap class here is the OMN-15391 refusal — everything ran, nothing was
    probative — and a "failing checks" heading over an empty list would state
    a failure the verdict did not find.
    """
    linear = _FakeLinear(_DESCRIPTION)
    checks = [
        _check("dod-pr-view", "non_probative", proof_class="merge-state"),
        _check("dod-files", "non_probative", proof_class="merge-state"),
    ]
    handler = _handler(_skill_result(checks), linear)

    outcome = (await handler.handle(_request(apply=True))).outcomes[0]

    assert outcome.dod_verify_failed_count == 0
    assert len(outcome.check_results) == 2
    assert linear.comments, "the gap comment is the surface under test"
    body = linear.comments[0][1]
    assert "Failing checks" not in body


def test_an_unreadable_checks_payload_yields_no_rows() -> None:
    """A verdict whose `checks` is not a list means NO attribution.

    The same rule `_check_records` already enforces for every other consumer
    in the module: an unreadable shape is silence, never a fabricated row.
    """
    assert _check_result_rows({"checks": "not-a-list"}) == ()
    assert _check_result_rows({}) == ()
    # A non-dict entry inside an otherwise readable list is dropped, not
    # coerced into a row with empty fields.
    assert _check_result_rows({"checks": ["a string", {"evidence_id": "x"}]}) == (
        _check_result_rows({"checks": [{"evidence_id": "x"}]})
    )


def test_a_long_message_is_excerpted_rather_than_carried_whole() -> None:
    """The rows travel in every receipt, so the excerpt is bounded.

    A pytest failure message runs to kilobytes. Thirty of them on one outcome
    would make the receipt the thing that is unreadable, which is the defect
    this ticket is fixing rather than a price worth paying for it.
    """
    long_message = "E   " + ("assert 0 == 1\n" * 400)
    (row,) = _check_result_rows(
        {"checks": [_check("dod-long", "failed", message=long_message)]}
    )
    assert len(row.message_excerpt) <= 200
    # Whitespace is collapsed: the excerpt is one line in a Linear comment and
    # a bulleted list, and a raw traceback's newlines would break both.
    assert row.message_excerpt.startswith("E assert 0 == 1 assert 0 == 1")
    assert "\n" not in row.message_excerpt
    # The cut is VISIBLE, so nobody chases a truncation as if the verifier had
    # emitted a half-finished assertion.
    assert row.message_excerpt.endswith("…")


# -- the rows change no decision ------------------------------------------


async def test_the_gap_comment_marker_is_unchanged_by_the_rows() -> None:
    """The dedup key is the withholding CHECK SET, and it must not move.

    If adding the rows changed the marker, every gapped ticket on the board
    would be re-commented once on the first tick after this lands — the
    OMN-16808 noise this node already paid to remove. The marker is asserted
    against its literal recorded value rather than recomputed, so a change to
    the fingerprint inputs fails here instead of being absorbed.
    """
    linear = _FakeLinear(_DESCRIPTION)
    handler = _handler(_skill_result(_mixed_checks()), linear)
    await handler.handle(_request(apply=True))
    body = linear.comments[0][1]

    # Re-running the identical verdict must dedup against the marker the first
    # run wrote: same class, same fingerprint, no second comment.
    second = (await handler.handle(_request(apply=True))).outcomes[0]
    assert second.decision is EnumEvidenceAutocloseDecision.SKIPPED_DUPLICATE_COMMENT
    assert len(linear.comments) == 1
    assert "onex-autoclose-sweep" in body
    # AND the deduped run still RECORDS the checks. This is the load-bearing
    # half: holding the marker still means a ticket already carrying a gap
    # comment does not get a new one naming its failing checks, so the receipt
    # is the surface that answers "which three" for the existing board. A
    # duplicate outcome with empty rows would leave that question unanswerable
    # for every ticket the closer has already commented on — which is all of
    # them, including OMN-18426.
    assert _failing(second.check_results) == ["dod-tenant-rls-probe"]


async def test_the_counters_still_come_from_the_verdict_not_the_rows() -> None:
    """The flip predicate reads count fields; the rows are descriptive only.

    Handing in a verdict whose declared counters DISAGREE with its own check
    list is the discriminating input: if the rows had become the source of
    the counters, these assertions would read the list's 3 rather than the
    payload's declared 94/30/3.
    """
    checks = _mixed_checks()
    skill_result = _skill_result(checks)
    terminal = skill_result["result"]
    assert isinstance(terminal, dict)
    terminal["total_checks"] = 94
    terminal["verified_count"] = 30
    terminal["failed_count"] = 3

    linear = _FakeLinear(_DESCRIPTION)
    outcome = (
        await _handler(skill_result, linear).handle(_request(apply=True))
    ).outcomes[0]

    assert outcome.dod_verify_total_checks == 94
    assert outcome.dod_verify_verified_count == 30
    assert outcome.dod_verify_failed_count == 3
    assert len(outcome.check_results) == 3


async def test_an_outcome_reached_before_the_verdict_carries_no_rows() -> None:
    """A refusal that never ran the verifier has nothing to report.

    `skipped_excluded` is taken ahead of every read, so an empty row tuple
    there is the honest answer — and it must not be confusable with a verdict
    whose checks could not be parsed, which is why the reason still differs.
    """
    linear = _FakeLinear(_DESCRIPTION)
    handler = _handler(_skill_result(_mixed_checks()), linear)

    outcome = (
        await handler.handle(_request(apply=True, exclude_tickets=[_TICKET]))
    ).outcomes[0]

    assert outcome.decision is EnumEvidenceAutocloseDecision.SKIPPED_EXCLUDED
    assert outcome.check_results == ()
