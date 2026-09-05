# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-15911 — a green dod_verify tally alone must not release a Done flip.

The autoclose sweep flipped on arithmetic: ``total_checks > 0``, zero failed,
``verified_count == total_checks``. Every check in a typical autobound OCC
contract is a ``gh pr view --json state`` read, so that arithmetic is
satisfiable entirely by "the PR merged" — and the sweep would then state, in
Linear, that the ticket's acceptance criteria are receipt-proven.

OMN-15911 gives each check verdict a ``proof_class`` and the verdict a
``behavior_proving_count`` roll-up. This suite pins the consuming rule: at
least one check must have both PASSED and executed the claimed behavior, or
the sweep posts a gap instead of flipping.

Fail-closed direction is asserted explicitly: a verifier too old to report the
field is an ERROR (no verdict can be inferred), never a flip.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
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


_RECEIPT_SUMMARY_MODEL = (
    "omnibase_infra.cli.model_receipt_runtime_summary.ModelReceiptRuntimeSummary"
)
_DOD_VERIFY_STATE_MODEL = (
    "omnimarket.nodes.node_dod_verify.models.model_dod_verify_state.ModelDodVerifyState"
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
    behavior_proving_count: int | None,
) -> dict[str, object]:
    """A ModelSkillResult shaped like `onex skill dod_verify` prints.

    On a NON-success run the counts live under ``result.terminal_payload``
    (OMN-16736), which is ``ModelDodVerifyState.model_dump(mode="json")`` —
    verified against the live capture at tests/fixtures/omn16736/.

    On a success-like run `receipt_mode` puts that same model FLAT on
    ``result`` instead, with no ``terminal_payload`` key (OMN-16961, captures
    at tests/fixtures/omn16961/). The arm is derived from the verdict below
    exactly as the CLI derives it, so this double cannot model a receipt the
    CLI never prints.

    ``behavior_proving_count=None`` models a verifier predating OMN-15911:
    the key is absent from the payload entirely.
    """
    verified = sum(1 for c in checks if c["status"] == "verified")
    failed = sum(1 for c in checks if c["status"] == "failed")
    terminal: dict[str, object] = {
        "correlation_id": str(uuid4()),
        "ticket_id": _TICKET,
        "status": "verified" if failed == 0 else "failed",
        "dry_run": False,
        "checks": checks,
        "total_checks": len(checks),
        "verified_count": verified,
        "failed_count": failed,
        "skipped_count": 0,
        "superseded_count": 0,
        "error_message": None,
    }
    if behavior_proving_count is not None:
        terminal["behavior_proving_count"] = behavior_proving_count
    if failed == 0:
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
        "result_model": _RECEIPT_SUMMARY_MODEL,
    }


class _FakeLinear:
    def __init__(self) -> None:
        self.state_updates: list[tuple[str, str]] = []
        self.comments: list[tuple[str, str]] = []

    async def fetch_issue(self, ticket_id: str) -> dict[str, object]:
        return {
            "id": "issue-uuid-1",
            "identifier": _TICKET,
            "state": {"id": "s1", "name": "In Progress", "type": "started"},
            "labels": {"nodes": []},
            # OMN-17658: the children conjunct reads this connection live on every
            # tick, so the double has to carry it. An absent connection is
            # deliberately ERROR_LINEAR_API in production — a fence that read a
            # missing key as "no children" would retire itself the day the query
            # drifted, so the double must speak the real payload.
            "children": {"nodes": []},
            "team": {"id": "team-1"},
            # No acceptance-criteria section: the OMN-16736 AC-coverage guard
            # must not be what withholds the flip in these cases.
            "description": None,
        }

    async def fetch_done_state_id(self, team_id: str) -> str:
        return "state-done-id"

    async def update_issue_state(self, issue_id: str, state_id: str) -> bool:
        self.state_updates.append((issue_id, state_id))
        return True

    async def fetch_issue_history(
        self, issue_id: str, page_size: int, max_pages: int
    ) -> tuple[list[dict[str, object]] | None, str]:
        """State history that MOVES when the sweep writes (OMN-17658).

        The bound flip readback re-reads this connection and requires a
        completed segment the pre-write read did not have. A double returning a
        constant empty history would model a Linear that never records state
        changes, and would turn every legitimate flip in this file into
        ERROR_READBACK_UNCONFIRMED — hiding the guard rather than exercising it.
        """
        return [
            {
                "id": f"entry-{index}",
                "createdAt": f"2026-09-05T00:00:{index:02d}Z",
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
        # OMN-16808: the sweep reads what it has already said on a ticket
        # before it says anything else. This double serves its own write log
        # back, so a second run over the same window is distinguishable from a
        # first — a fake that always returned () would let the duplicate-post
        # defect pass unnoticed here.
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


async def test_all_verified_but_merge_state_only_posts_a_gap_not_a_flip() -> None:
    """The blocker, executed: 3/3 green, all of it merge state."""
    linear = _FakeLinear()
    handler = _handler(
        _skill_result(
            checks=[
                _check("dod-pr-1559-state", "verified", "merge-state"),
                _check("dod-pr-1559-files", "verified", "merge-state"),
                _check("dod-occ-admissibility", "verified", "surrogate"),
            ],
            behavior_proving_count=0,
        ),
        linear,
    )
    result = await handler.handle(_request())

    assert result.tickets_flipped == 0
    outcome = result.outcomes[0]
    assert outcome.decision is EnumEvidenceAutocloseDecision.GAP_NO_BEHAVIOR_PROOF
    assert outcome.dod_verify_behavior_proving_count == 0
    assert linear.state_updates == []


async def test_one_verified_behavior_check_releases_the_flip() -> None:
    """The negative control: the same tally, with one executed proof."""
    linear = _FakeLinear()
    handler = _handler(
        _skill_result(
            checks=[
                _check("dod-pr-1559-state", "verified", "merge-state"),
                _check("dod-tests", "verified", "behavior"),
            ],
            behavior_proving_count=1,
        ),
        linear,
    )
    result = await handler.handle(_request(apply=True))

    assert result.tickets_flipped == 1
    outcome = result.outcomes[0]
    assert outcome.decision is EnumEvidenceAutocloseDecision.FLIPPED
    assert outcome.dod_verify_behavior_proving_count == 1
    assert linear.state_updates == [("issue-uuid-1", "state-done-id")]


async def test_a_failed_behavior_check_is_not_a_behavior_proof() -> None:
    """A behavior check that FAILED cannot release the flip.

    dod_verify would report ``failed_count > 0`` here anyway, so this asserts
    the two guards agree rather than compete — the sweep must not flip, and it
    must not flip for the behavior-proof reason either.
    """
    linear = _FakeLinear()
    handler = _handler(
        _skill_result(
            checks=[
                _check("dod-pr-1559-state", "verified", "merge-state"),
                _check("dod-tests", "failed", "behavior"),
            ],
            behavior_proving_count=0,
        ),
        linear,
    )
    result = await handler.handle(_request(apply=True))

    assert result.tickets_flipped == 0
    assert linear.state_updates == []


async def test_a_verifier_that_cannot_report_proof_class_fails_closed() -> None:
    """An omnimarket predating OMN-15911 omits the key entirely.

    Absent the field there is no verdict about proof strength to act on, so
    the sweep records an error rather than inferring one. Treating the absence
    as "zero behavior proof" would post a Linear comment stating a finding the
    run never made; treating it as "fine" would flip on the exact arithmetic
    this ticket exists to stop.
    """
    linear = _FakeLinear()
    handler = _handler(
        _skill_result(
            checks=[_check("dod-pr-1559-state", "verified", "merge-state")],
            behavior_proving_count=None,
        ),
        linear,
    )
    result = await handler.handle(_request(apply=True))

    assert result.tickets_flipped == 0
    outcome = result.outcomes[0]
    assert outcome.decision is EnumEvidenceAutocloseDecision.ERROR_VERIFY_UNPARSEABLE
    assert "behavior_proving_count" in outcome.reason
    assert linear.state_updates == []
    assert linear.comments == []


async def test_dry_run_gap_is_an_honest_preview_and_writes_nothing() -> None:
    linear = _FakeLinear()
    handler = _handler(
        _skill_result(
            checks=[_check("dod-pr-1559-state", "verified", "merge-state")],
            behavior_proving_count=0,
        ),
        linear,
    )
    result = await handler.handle(_request(apply=False))

    outcome = result.outcomes[0]
    assert outcome.decision is EnumEvidenceAutocloseDecision.GAP_NO_BEHAVIOR_PROOF
    assert outcome.reason.startswith("[DRY-RUN]")
    assert outcome.applied is False
    assert linear.comments == []
    assert linear.state_updates == []


async def test_no_pre_omn15911_receipt_carries_a_proof_classification() -> None:
    """The committed live capture: every verdict before this ticket is unclassified.

    ``tests/fixtures/omn16736/dod-verify-omn16752.skill-result.json.captured``
    is a real ``onex skill dod_verify`` run (16 checks, 9 verified / 6 failed /
    1 skipped). Neither the verdict nor any individual check carries proof
    classification, because none existed. That is the shape of the entire
    existing receipt corpus, and it is why the sweep's missing-key path is an
    ERROR rather than an inference: replaying any of it must not produce a
    flip. (This particular capture also has failures, so it is stopped by the
    arithmetic first — the isolated proof of the new gate is
    ``test_a_verifier_that_cannot_report_proof_class_fails_closed``.)
    """
    capture = json.loads(
        (
            Path(__file__).parents[3]
            / "fixtures"
            / "omn16736"
            / "dod-verify-omn16752.skill-result.json.captured"
        ).read_text(encoding="utf-8")
    )
    terminal = capture["result"]["terminal_payload"]
    assert "behavior_proving_count" not in terminal
    assert all("proof_class" not in check for check in terminal["checks"])

    linear = _FakeLinear()
    handler = _handler(capture, linear)
    result = await handler.handle(_request(apply=True))

    assert result.tickets_flipped == 0
    assert linear.state_updates == []
