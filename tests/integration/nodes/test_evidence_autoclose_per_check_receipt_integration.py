# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18490 — the failing check ids survive all the way to the receipt JSON.

The unit suite asserts the rows on the in-memory outcome. That is not where
the defect was observed. It was observed on the RECEIPT — the single JSON
document `onex skill evidence_autoclose_sweep` writes to stdout, which is
the only durable record a scheduled run leaves and the thing a reader opens
two days later. Run 35299192253's receipt carried `30/94 ACs verified, 3
failed` for OMN-18426 and no check name anywhere in it.

So this test drives the handler and then serialises its result exactly as
receipt mode does, `model_dump(mode="json")`, and asks the JSON the question
the collaborator asked: which checks failed. A change that populated the
model but dropped the rows on the way out — a serialisation alias, an
exclude, a field the contract's `output_fields` does not carry — would pass
every unit test in the suite and reproduce the defect verbatim.

It also asserts the reverse trip. A receipt is read by parsing it back into
the result model, so a row that serialises and cannot be re-parsed is a
receipt that breaks its own consumer.

The verdict fixture is OMN-18426's real shape in miniature: a tally that
disagrees with nothing, three checks, one of them failed, and counters
declared independently of the list — 94/30/3, the live numbers — so the test
also proves the rows did not quietly become the source of the counters.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    HandlerEvidenceAutocloseSweep,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_ac_binding_check_status import (
    EnumAcBindingCheckStatus,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_evidence_autoclose_sweep_request import (
    ModelEvidenceAutocloseSweepRequest,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_evidence_autoclose_sweep_result import (
    ModelEvidenceAutocloseSweepResult,
)

pytestmark = pytest.mark.integration

_OCC_REPO = "OmniNode-ai/onex_change_control"
_TICKET = "OMN-9999"
_DOD_VERIFY_STATE_MODEL = (
    "omnimarket.nodes.node_dod_verify.models.model_dod_verify_state.ModelDodVerifyState"
)

_FAILING_CHECK = "dod-OmniNode-ai-omnimarket-pr-2588::pr-live-state"
_FAILING_MESSAGE = (
    "OmniNode-ai/omnimarket#2588: required checks not green "
    "(1 required context(s) not green: CI Summary)"
)

_DESCRIPTION = (
    "## Acceptance criteria\n"
    "\n"
    "- AC1: the behaviour under test is proven by a verified probative check.\n"
)


def _merged_pr(number: int) -> dict[str, object]:
    recent = (datetime.now(tz=UTC) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "number": number,
        "html_url": f"https://github.com/{_OCC_REPO}/pull/{number}",
        "title": f"evidence({_TICKET}): OCC companion",
        "updated_at": recent,
        "merged_at": recent,
    }


def _omn_18426_shaped_verdict() -> dict[str, object]:
    """The live counters, over a three-check list that does not derive them."""
    terminal: dict[str, object] = {
        "correlation_id": str(uuid4()),
        "ticket_id": _TICKET,
        "status": "verified",
        "dry_run": False,
        "checks": [
            {
                "evidence_id": "dod-tests-green",
                "status": "verified",
                "proof_class": "behavior",
                "message": "OK (1ms)",
                "binds_ac": ["AC1"],
            },
            {
                "evidence_id": _FAILING_CHECK,
                "status": "failed",
                "proof_class": "merge-state",
                "message": _FAILING_MESSAGE,
                "binds_ac": [],
            },
            {
                "evidence_id": "ac1-ac2-hook-on-all-fifteen-default-branches",
                "status": "skipped",
                "proof_class": "",
                "message": "the check did not execute",
                "binds_ac": [],
            },
        ],
        # Declared, not derived — the live OMN-18426 numbers.
        "total_checks": 94,
        "verified_count": 30,
        "failed_count": 3,
        "skipped_count": 1,
        "superseded_count": 0,
        "non_probative_count": 0,
        "behavior_proving_count": 1,
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


class _FakeLinear:
    def __init__(self) -> None:
        self.comments: list[tuple[str, str]] = []
        self.state_updates: list[tuple[str, str]] = []

    async def fetch_issue(self, ticket_id: str) -> dict[str, object]:
        return {
            "id": "issue-uuid-1",
            "identifier": _TICKET,
            "state": {"id": "s1", "name": "In Progress", "type": "started"},
            "labels": {"nodes": []},
            "children": {"nodes": []},
            "team": {"id": "team-1"},
            "description": _DESCRIPTION,
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


def _handler(linear: _FakeLinear) -> HandlerEvidenceAutocloseSweep:
    skill_result = _omn_18426_shaped_verdict()

    async def fake_gh(args: list[str], timeout: float):
        path = args[2]
        if "/files" in path:
            return [{"filename": f"contracts/{_TICKET}.yaml"}], ""
        page = int(path.rsplit("page=", 1)[1])
        return ([_merged_pr(10125)], "") if page == 1 else ([], "")

    async def fake_dod_verify(ticket_id: str, cwd: str, timeout: float):
        return skill_result, 0, ""

    return HandlerEvidenceAutocloseSweep(
        linear_client=linear,  # type: ignore[arg-type]
        autoclose_disabled=False,
        run_gh_command=fake_gh,
        run_dod_verify_command=fake_dod_verify,
    )


@pytest.mark.asyncio
async def test_the_receipt_json_names_the_failing_check() -> None:
    """THE OBSERVED SURFACE. Serialise the result the way receipt mode does.

    Asserted against the JSON text as well as the parsed structure: the
    question that went unanswered for two days was asked by grepping a
    receipt, and a row that is present in the object graph but absent from
    the document is, for that reader, not present.
    """
    linear = _FakeLinear()
    result = await _handler(linear).handle(
        ModelEvidenceAutocloseSweepRequest(
            correlation_id=uuid4(),
            occ_repo=_OCC_REPO,
            lookback_hours=24,
            apply=True,
        )
    )

    payload = result.model_dump(mode="json")
    document = json.dumps(payload)

    assert _FAILING_CHECK in document, (
        "the failing check id must reach the receipt document; a reader "
        "greps this text, not the object graph"
    )

    (outcome,) = payload["outcomes"]
    failing = [
        row["evidence_check"]
        for row in outcome["check_results"]
        if row["status"] == EnumAcBindingCheckStatus.FAILED.value
    ]
    assert failing == [_FAILING_CHECK]
    # The excerpt travels with it, so the receipt answers "why" as well as
    # "which" without a job log that GitHub may no longer hold.
    (failing_row,) = [
        row
        for row in outcome["check_results"]
        if row["evidence_check"] == _FAILING_CHECK
    ]
    assert "CI Summary" in failing_row["message_excerpt"]


@pytest.mark.asyncio
async def test_the_receipt_round_trips_back_into_the_result_model() -> None:
    """A receipt is read by re-parsing it, so the rows must survive the trip.

    `extra="forbid"` on every model in this graph means a serialised field
    the model cannot accept is a hard parse failure for every consumer, not
    a dropped value — which is the right direction, and is exactly why it
    has to be exercised rather than assumed.
    """
    linear = _FakeLinear()
    result = await _handler(linear).handle(
        ModelEvidenceAutocloseSweepRequest(
            correlation_id=uuid4(),
            occ_repo=_OCC_REPO,
            lookback_hours=24,
            apply=True,
        )
    )

    reparsed = ModelEvidenceAutocloseSweepResult.model_validate(
        json.loads(json.dumps(result.model_dump(mode="json")))
    )

    assert reparsed.outcomes[0].check_results == result.outcomes[0].check_results
    assert (
        reparsed.outcomes[0].check_results[1].status is EnumAcBindingCheckStatus.FAILED
    )


@pytest.mark.asyncio
async def test_the_declared_counters_survive_a_disagreeing_check_list() -> None:
    """The rows are a report, never a re-derivation of the tally.

    The fixture declares 94/30/3 over a list of three. If a future change
    made the rows the source of the counters, the receipt would quietly say
    3/3/1 and every flip-eligibility argument downstream of it would be
    about a verdict the verifier never reached.
    """
    linear = _FakeLinear()
    result = await _handler(linear).handle(
        ModelEvidenceAutocloseSweepRequest(
            correlation_id=uuid4(),
            occ_repo=_OCC_REPO,
            lookback_hours=24,
            apply=True,
        )
    )

    outcome = result.outcomes[0]
    assert (
        outcome.dod_verify_total_checks,
        outcome.dod_verify_verified_count,
        outcome.dod_verify_failed_count,
    ) == (94, 30, 3)
    assert len(outcome.check_results) == 3


@pytest.mark.asyncio
async def test_the_comment_written_to_linear_names_the_failing_check() -> None:
    """The human surface, end to end through the real write path.

    The outcome is what a machine reads; the comment is what the person
    deciding what to fix reads, and only one of the two was ever going to be
    opened by the collaborator who asked.
    """
    linear = _FakeLinear()
    await _handler(linear).handle(
        ModelEvidenceAutocloseSweepRequest(
            correlation_id=uuid4(),
            occ_repo=_OCC_REPO,
            lookback_hours=24,
            apply=True,
        )
    )

    assert len(linear.comments) == 1
    body = linear.comments[0][1]
    assert _FAILING_CHECK in body
    assert "CI Summary" in body
    # The skipped check is NOT listed as failing: a different fact with a
    # different remedy, and folding them together hands a reader work that
    # is not theirs.
    assert "ac1-ac2-hook-on-all-fifteen-default-branches" not in body
