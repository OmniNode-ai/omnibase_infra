# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18056 — the nine adjudicated tickets, driven offline through the closer.

The DoD closeout sweep of 2026-09-08 adjudicated nine sprint tickets, measured
that most of them satisfied the closer's full flip predicate, and concluded it
was correct to flip **none** of them. This module is that conclusion as a
regression: every one of the nine is driven through the real handler against
its real `dod_verify` verdict and its real acceptance-criteria section, and
every one must be HELD.

Fixtures and their provenance — including the one AC text that is reconstructed
rather than read — are documented in `tests/fixtures/omn18056/README.md`. The
verdicts are verbatim captures; `binds_ac` is empty on all 88 checks across the
nine because no contract in the corpus declares one.

The holds are NOT all the same hold, and the table below is the point: the
closer now has three distinct reasons to refuse this population, each naming
what is missing rather than restating a ratio.

Offline by construction: no network, no Linear, no GitHub, no subprocess.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
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

pytestmark = pytest.mark.unit

_FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "omn18056"
_OCC_REPO = "OmniNode-ai/onex_change_control"
_DOD_VERIFY_STATE_MODEL = (
    "omnimarket.nodes.node_dod_verify.models.model_dod_verify_state.ModelDodVerifyState"
)

_UNMET = EnumEvidenceAutocloseDecision.GAP_AC_UNBOUND
_NO_BEHAVIOR = EnumEvidenceAutocloseDecision.GAP_NO_BEHAVIOR_PROOF
_LIVE_NOT_RUN = EnumEvidenceAutocloseDecision.SKIPPED_LIVE_CHECK_NOT_EXECUTED

#: (ticket, expected decision, the criterion the sweep report named as unmet).
#: The third column is documentation of WHY the ticket is not done, carried here
#: so a future change that flips one of these has to delete a named criterion
#: rather than silently relax a bound.
_NINE: tuple[tuple[str, EnumEvidenceAutocloseDecision, str], ...] = (
    ("OMN-17397", _UNMET, "AC2 — the workflow status route returns 401"),
    ("OMN-15660", _UNMET, "AC3 — the handler group is not run-scoped"),
    ("OMN-17304", _UNMET, "AC4 — host_handlers has 0 occurrences"),
    ("OMN-15922", _UNMET, "DoD3 — the live-readback record does not exist"),
    # Held EARLIER than the binding leg and correctly so: its captured verdict
    # is `status=skipped`, 0 verified against 8 non-probative, with two live
    # checks that never executed. The counter predicate refuses it before any
    # question about criteria arises. Recorded with its real class rather than
    # forced into the new one — the report's claim that a fresh-clone re-read
    # made this ticket satisfy the full predicate does NOT reproduce here.
    (
        "OMN-16558",
        _LIVE_NOT_RUN,
        "AC1 — provenance: both Secrets carry ownerReferences=[]",
    ),
    ("OMN-16025", _UNMET, "link 2 — projection_readback_not_configured, 3/5 proven"),
    ("OMN-17295", _UNMET, "AC1 — the --locus flag exists in no repo"),
    ("OMN-16833", _NO_BEHAVIOR, "AC1 — BACKENDS_LOADED(7) vs REFERENCED_BY_TIERS(8)"),
    ("OMN-17298", _UNMET, "AC6b — the projection node IS running with topic traffic"),
)


class _FakeLinear:
    """The ticket as Linear returns it: one AC section, no children, no history.

    Deliberately minimal. Everything this fixture is about happens between the
    description and the verdict, so anything else the closer reads is supplied
    in its most PERMISSIVE form — no open children, no prior comment, no state
    history. A hold produced against that is a hold the fence would have taken
    anyway, never one manufactured by a stub.
    """

    def __init__(self, ticket: str, description: str) -> None:
        self._ticket = ticket
        self._description = description
        self.state_updates: list[tuple[str, str]] = []
        self.comments: list[tuple[str, str]] = []

    async def fetch_issue(self, ticket_id: str) -> dict[str, object]:
        return {
            "id": f"issue-{self._ticket}",
            "identifier": self._ticket,
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


def _merged_pr(ticket: str, number: int) -> dict[str, object]:
    recent = (datetime.now(tz=UTC) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "number": number,
        "html_url": f"https://github.com/{_OCC_REPO}/pull/{number}",
        "title": f"evidence({ticket}): OCC companion",
        "updated_at": recent,
        "merged_at": recent,
    }


def _verdict(ticket: str) -> dict[str, Any]:
    return json.loads((_FIXTURES / f"{ticket}.verdict.json").read_text())


def _description(ticket: str) -> str:
    return (_FIXTURES / f"{ticket}.ac.md").read_text()


def _skill_result(ticket: str) -> dict[str, object]:
    """The captured receipt, re-wrapped in the arm the closer reads.

    OMN-16558's live capture came back on the runtime-summary arm rather than
    the state arm; both are legitimate and declared, and the closer dispatches
    on `result_model`. It is normalised here so the fixture exercises the AC
    machinery on all nine rather than nine-minus-one, and the arm-dispatch
    behaviour keeps its own dedicated test (test_omn_16961_receipt_arm.py).
    """
    captured = _verdict(ticket)
    return {
        "skill_name": "dod_verify",
        "node_name": "node_dod_verify",
        "status": "success",
        "exit_code": 0,
        "result": captured["verdict"],
        "result_model": _DOD_VERIFY_STATE_MODEL,
    }


def _handler(ticket: str, linear: _FakeLinear) -> HandlerEvidenceAutocloseSweep:
    result = _skill_result(ticket)

    async def fake_gh(args: list[str], timeout: float) -> tuple[Any, str]:
        path = args[2]
        if "/files" in path:
            return [{"filename": f"contracts/{ticket}.yaml"}], ""
        page = int(path.rsplit("page=", 1)[1])
        return ([_merged_pr(ticket, 9100)], "") if page == 1 else ([], "")

    async def fake_dod_verify(
        ticket_id: str, cwd: str, timeout: float
    ) -> tuple[dict[str, object], int, str]:
        return result, 0, ""

    return HandlerEvidenceAutocloseSweep(
        linear_client=linear,  # type: ignore[arg-type]
        autoclose_disabled=False,
        run_gh_command=fake_gh,
        run_dod_verify_command=fake_dod_verify,
    )


def _request() -> ModelEvidenceAutocloseSweepRequest:
    return ModelEvidenceAutocloseSweepRequest(
        correlation_id=uuid4(),
        occ_repo=_OCC_REPO,
        lookback_hours=24,
        apply=True,
    )


@pytest.mark.parametrize(
    ("ticket", "expected", "named"), _NINE, ids=[t for t, _, _ in _NINE]
)
async def test_the_closer_holds_every_adjudicated_ticket(
    ticket: str, expected: EnumEvidenceAutocloseDecision, named: str
) -> None:
    """0 flips, 9 holds — the sweep's own conclusion, mechanically pinned.

    `apply=True` deliberately: a dry run cannot demonstrate that no Done was
    written, only that none was previewed. The assertion that matters is
    `state_updates == []` under a mode that WOULD have written one.
    """
    linear = _FakeLinear(ticket, _description(ticket))

    result = await _handler(ticket, linear).handle(_request())

    outcome = result.outcomes[0]
    assert outcome.decision is expected, (
        f"{ticket} ({named}) decided {outcome.decision.value}; "
        f"expected {expected.value}"
    )
    assert result.tickets_flipped == 0
    assert linear.state_updates == [], f"{ticket} was written Done"


async def test_the_population_produces_zero_flips_and_nine_holds() -> None:
    """The aggregate, stated once so the count itself is a regression."""
    decisions: list[EnumEvidenceAutocloseDecision] = []
    for ticket, _expected, _named in _NINE:
        linear = _FakeLinear(ticket, _description(ticket))
        result = await _handler(ticket, linear).handle(_request())
        decisions.append(result.outcomes[0].decision)
        assert linear.state_updates == []

    assert len(decisions) == 9
    assert EnumEvidenceAutocloseDecision.FLIPPED not in decisions


def test_no_captured_contract_declares_an_acceptance_binding() -> None:
    """The corpus measurement, pinned so the fixture cannot drift silently.

    122 checks across nine real contracts, and not one declares `binds_ac`.
    The POSITIVE CONTROL is on the same line: every check does carry a
    `status`, so an empty binding set is a fact about the corpus and not an
    artefact of a capture that dropped the field.
    """
    total = 0
    for ticket, _expected, _named in _NINE:
        checks = _verdict(ticket)["verdict"]["checks"]
        assert checks, f"{ticket} captured no checks at all"
        for check in checks:
            total += 1
            assert check["binds_ac"] == []
            assert check["status"], "positive control: every check has a status"
    assert total == 122
