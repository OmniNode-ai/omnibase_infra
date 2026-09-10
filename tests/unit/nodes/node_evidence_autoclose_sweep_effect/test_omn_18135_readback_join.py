# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18135 AC3 + AC4 — the closer's half of the readback ruling.

THE RULING, and its provenance
------------------------------
Orchestrator ruling under the deterministic-truth doctrine, recorded at
``docs/tracking/ROLLING_WORK_LEDGER.md:6148`` (2026-09-10T17:50:03Z). Not an
operator ruling; stated as it is rather than upgraded:

    an asserted live or shell readback is admissible as its own proof class,
    readback, for a criterion whose text asserts live STATE ...; it never
    proves BEHAVIOUR ..., which stays test-runner/onex-CLI only. The closer
    accepts readback-class checks against state-shaped criteria and holds
    behaviour-shaped criteria bound only to readbacks, naming the bar.

omnimarket carries the CLASS. This carries the JOIN, because deciding whether
a criterion is state-shaped needs the ticket's acceptance text and
``node_dod_verify`` never sees it.

WHAT MOVES IN EACH DIRECTION
----------------------------
This is not purely a loosening, and both halves are asserted below.

**Looser**: a ticket whose criteria are all state-shaped and each discharged
by a verified readback no longer dies on ``behavior_proving_count <= 0``.
That is the case the ruling exists to admit — OMN-17771 is the worked
example.

**Stricter**: a BEHAVIOUR-shaped criterion whose only proving checks are
readbacks now HOLDS. Before this, ``_ac_binding_gap`` bound on check status
alone and never looked at proof class, so a readback silently discharged a
criterion asserting what the code does. That was never right and the ruling
names it.

Deliberately UNTOUCHED: a criterion proved by a merge-state, surrogate or
indeterminate check binds exactly as it did. Narrowing those is a different
argument that nobody has made, and making it here would break tickets that
have already flipped on that basis.

THE HONEST LIMIT
----------------
Criterion shape is read from PROSE. It is a heuristic and it is tight on
purpose: a criterion is state-shaped only when it carries a positive state
marker AND no behaviour marker. Everything else falls to behaviour and holds.
The dangerous direction is a criterion wrongly read as state-shaped, because
that RELEASES; so the vocabulary earns its way in, and the veto wins ties.
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
_TICKET = "OMN-17771"
_DOD_VERIFY_STATE_MODEL = (
    "omnimarket.nodes.node_dod_verify.models.model_dod_verify_state.ModelDodVerifyState"
)

#: Criteria that assert live STATE: a condition, a row, a count, a config
#: value read from the running system. Shaped after OMN-17771's real body.
_STATE_SHAPED_DESCRIPTION = (
    "## Acceptance criteria\n"
    "\n"
    "- **AC1** the published guides contain 0 occurrences of the internal "
    "client ids, read back from the live repository.\n"
    "- **AC2** the registration URL in the guide returns the register form "
    "when requested against the live realm.\n"
    "- **AC3** the same request with an unknown client id does not return "
    "the register form.\n"
)

#: Criteria that assert what the CODE DOES. A readback may never discharge
#: one of these, however green it is.
_BEHAVIOUR_SHAPED_DESCRIPTION = (
    "## Acceptance criteria\n"
    "\n"
    "- **AC1** a red-first test fails today against the unfixed tree and "
    "passes after the change.\n"
    "- **AC2** the handler refuses a malformed envelope instead of raising.\n"
    "- **AC3** the retry path stops after the declared attempt budget.\n"
)


def _check(
    evidence_id: str,
    status: str,
    proof_class: str,
    *,
    binds_ac: tuple[str, ...] | None = None,
) -> dict[str, Any]:
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


def _readback_checks(labels: tuple[str, ...]) -> list[dict[str, Any]]:
    """One verified readback per criterion, and nothing behaviour-proving."""
    return [
        _check(
            f"dod-{label.lower()}-readback", "verified", "readback", binds_ac=(label,)
        )
        for label in labels
    ]


def _skill_result(checks: list[dict[str, Any]]) -> dict[str, Any]:
    """A dod_verify receipt with the OMN-18135 readback counter present."""
    verified = sum(1 for c in checks if c["status"] == "verified")
    failed = sum(1 for c in checks if c["status"] == "failed")
    non_probative = sum(1 for c in checks if c["status"] == "non_probative")
    behavior = sum(
        1
        for c in checks
        if c["status"] == "verified" and c["proof_class"] == "behavior"
    )
    readback = sum(
        1
        for c in checks
        if c["status"] == "verified" and c["proof_class"] == "readback"
    )
    terminal: dict[str, Any] = {
        "correlation_id": str(uuid4()),
        "ticket_id": _TICKET,
        "status": "verified" if verified and not failed else "skipped",
        "dry_run": False,
        "checks": checks,
        "total_checks": len(checks),
        "verified_count": verified,
        "failed_count": failed,
        "skipped_count": 0,
        "superseded_count": 0,
        "non_probative_count": non_probative,
        "behavior_proving_count": behavior,
        "readback_proving_count": readback,
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
    def __init__(self, description: str) -> None:
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
            for index, (target, _s) in enumerate(self.state_updates, start=1)
            if target == issue_id
        ], ""

    async def create_comment(self, issue_id: str, body: str) -> bool:
        self.comments.append((issue_id, body))
        return True

    async def fetch_comment_bodies(self, issue_id: str) -> tuple[str, ...] | None:
        return tuple(body for target, body in self.comments if target == issue_id)


def _handler(
    skill_result: dict[str, Any], linear: _FakeLinear
) -> HandlerEvidenceAutocloseSweep:
    async def fake_gh(args: list[str], timeout: float):
        path = args[2]
        if "/files" in path:
            return [{"filename": f"contracts/{_TICKET}.yaml"}], ""
        page = int(path.rsplit("page=", 1)[1])
        recent = "2026-09-10T16:00:00Z"
        return (
            (
                [
                    {
                        "number": 9100,
                        "html_url": f"https://github.com/{_OCC_REPO}/pull/9100",
                        "title": f"evidence({_TICKET}): OCC companion",
                        "updated_at": recent,
                        "merged_at": recent,
                    }
                ],
                "",
            )
            if page == 1
            else ([], "")
        )

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
    body = _skill_result(checks)["result"]
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
# LOOSER: state-shaped criteria, discharged by readbacks, may now release.
# --------------------------------------------------------------------------


async def test_state_shaped_criteria_proven_by_readbacks_release() -> None:
    """RED. The OMN-17771 shape: every criterion is live state, each read back.

    Before this, `behavior_proving_count <= 0` refused the ticket outright and
    the criteria were never even reached.
    """
    checks = _readback_checks(("AC1", "AC2", "AC3"))
    linear = _FakeLinear(_STATE_SHAPED_DESCRIPTION)
    _seed_redraw(linear, checks)
    result = await _handler(_skill_result(checks), linear).handle(_request())
    outcome = result.outcomes[0]

    assert outcome.decision is not EnumEvidenceAutocloseDecision.GAP_NO_BEHAVIOR_PROOF
    assert outcome.decision is not EnumEvidenceAutocloseDecision.GAP_AC_UNBOUND
    assert outcome.decision is EnumEvidenceAutocloseDecision.FLIPPED


# --------------------------------------------------------------------------
# STRICTER: a behaviour-shaped criterion is NOT discharged by a readback.
# --------------------------------------------------------------------------


async def test_a_behaviour_shaped_criterion_is_not_discharged_by_a_readback() -> None:
    """RED, and this direction is a TIGHTENING.

    Binding used to key on check status alone and never on proof class, so a
    readback silently discharged a criterion asserting what the code does.
    The hold must name the bar rather than report a ratio.
    """
    checks = _readback_checks(("AC1", "AC2", "AC3"))
    linear = _FakeLinear(_BEHAVIOUR_SHAPED_DESCRIPTION)
    _seed_redraw(linear, checks)
    result = await _handler(_skill_result(checks), linear).handle(_request())
    outcome = result.outcomes[0]

    assert result.tickets_flipped == 0
    assert outcome.decision in (
        EnumEvidenceAutocloseDecision.GAP_NO_BEHAVIOR_PROOF,
        EnumEvidenceAutocloseDecision.GAP_AC_UNBOUND,
    )


async def test_one_behaviour_shaped_criterion_holds_the_whole_ticket() -> None:
    """A mixed body is judged by its strictest criterion, not on average."""
    mixed = (
        "## Acceptance criteria\n"
        "\n"
        "- **AC1** the mirror table has 4 rows, read back from the live "
        "database.\n"
        "- **AC2** the handler refuses a malformed envelope instead of "
        "raising.\n"
    )
    checks = _readback_checks(("AC1", "AC2"))
    linear = _FakeLinear(mixed)
    _seed_redraw(linear, checks)
    result = await _handler(_skill_result(checks), linear).handle(_request())

    assert result.tickets_flipped == 0


# --------------------------------------------------------------------------
# AC3: the hold names the bar.
# --------------------------------------------------------------------------


async def test_the_no_behaviour_proof_hold_names_the_bar() -> None:
    """A hold that does not say what would clear it teaches nobody anything.

    Three lanes spent a night discovering this rule by measurement. The
    comment must carry it: which command shapes prove behaviour, and that a
    readback is admissible for a state-shaped criterion.
    """
    checks = _readback_checks(("AC1", "AC2", "AC3"))
    linear = _FakeLinear(_BEHAVIOUR_SHAPED_DESCRIPTION)
    _seed_redraw(linear, checks)
    result = await _handler(_skill_result(checks), linear).handle(_request(apply=True))
    reason = result.outcomes[0].reason

    # Names what DOES prove behaviour.
    assert "pytest" in reason or "test runner" in reason
    assert "onex" in reason
    # Names the readback route and what it is good for.
    assert "readback" in reason.lower()


# --------------------------------------------------------------------------
# UNTOUCHED: every other class binds exactly as it did.
# --------------------------------------------------------------------------


async def test_a_behaviour_check_still_releases_a_behaviour_criterion() -> None:
    """The ordinary case is unmoved."""
    checks = [
        _check("dod-ac1-proof", "verified", "behavior", binds_ac=("AC1",)),
        _check("dod-ac2-proof", "verified", "behavior", binds_ac=("AC2",)),
        _check("dod-ac3-proof", "verified", "behavior", binds_ac=("AC3",)),
    ]
    linear = _FakeLinear(_BEHAVIOUR_SHAPED_DESCRIPTION)
    _seed_redraw(linear, checks)
    result = await _handler(_skill_result(checks), linear).handle(_request())
    assert result.outcomes[0].decision is EnumEvidenceAutocloseDecision.FLIPPED


async def test_a_surrogate_check_binds_exactly_as_it_did() -> None:
    """Deliberately NOT narrowed.

    A criterion proved by a merge-state or surrogate check binds today, and
    tickets have already flipped on that basis. Narrowing it is a different
    argument that nobody has made, and making it here would break them.
    """
    checks = [
        _check("dod-ac1-proof", "verified", "behavior", binds_ac=("AC1",)),
        _check("dod-ac2-artifact", "verified", "surrogate", binds_ac=("AC2",)),
        _check("dod-ac3-merge", "verified", "merge-state", binds_ac=("AC3",)),
    ]
    linear = _FakeLinear(_BEHAVIOUR_SHAPED_DESCRIPTION)
    _seed_redraw(linear, checks)
    result = await _handler(_skill_result(checks), linear).handle(_request())
    assert result.outcomes[0].decision is EnumEvidenceAutocloseDecision.FLIPPED


# --------------------------------------------------------------------------
# The shape predicate itself. Tight on purpose; ambiguity holds.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "criterion",
    [
        "AC1 the mirror table has 4 rows, read back from the live database",
        "AC1 the deployment is Running with 0 restarts on the new digest",
        "AC1 the guide contains 0 occurrences of the internal client id",
        "AC1 the endpoint returns the register form when requested live",
        "AC1 the config value is enabled on the running cluster",
    ],
)
def test_state_markers_are_recognised(criterion: str) -> None:
    from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
        _criterion_is_state_shaped,
    )

    assert _criterion_is_state_shaped(criterion) is True


@pytest.mark.parametrize(
    "criterion",
    [
        # Behaviour language, plainly.
        "AC1 a red-first test fails today and passes after the change",
        "AC1 the handler refuses a malformed envelope instead of raising",
        "AC1 the retry path stops after the declared attempt budget",
        # Ambiguous: no state marker at all. Falls to behaviour and holds.
        "AC1 the thing is done properly",
        "AC1 ship the fix",
        # A state marker VETOED by behaviour language. The veto wins ties,
        # because a criterion wrongly read as state-shaped RELEASES.
        "AC1 a test asserts the mirror table has 4 rows",
    ],
)
def test_anything_else_falls_to_behaviour_and_holds(criterion: str) -> None:
    from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
        _criterion_is_state_shaped,
    )

    assert _criterion_is_state_shaped(criterion) is False


async def test_a_readback_cannot_discharge_a_behaviour_criterion_beside_a_real_proof() -> (
    None
):
    """The case the behaviour conjunct does NOT catch, and the ruling does.

    `behavior_proving_count` is 1, so the conjunct releases. AC1 is genuinely
    proven. AC2 asserts what the code does and is bound ONLY to a readback.
    Before this the binding leg keyed on status alone, never on proof class,
    so AC2 bound and the ticket flipped on a readback standing in for a
    behaviour proof. That is the hole the ruling closes, and it is invisible
    to every test above because they have no behaviour proof at all.
    """
    checks = [
        _check("dod-ac1-proof", "verified", "behavior", binds_ac=("AC1",)),
        _check("dod-ac2-readback", "verified", "readback", binds_ac=("AC2",)),
    ]
    description = (
        "## Acceptance criteria\n"
        "\n"
        "- **AC1** a red-first test fails today and passes after the change.\n"
        "- **AC2** the handler refuses a malformed envelope instead of raising.\n"
    )
    linear = _FakeLinear(description)
    _seed_redraw(linear, checks)
    result = await _handler(_skill_result(checks), linear).handle(_request(apply=True))
    outcome = result.outcomes[0]

    assert result.tickets_flipped == 0
    assert outcome.decision is EnumEvidenceAutocloseDecision.GAP_AC_UNBOUND
    # The hold NAMES the criterion and the bar, not a ratio.
    assert "AC2" in outcome.reason
    assert "readback" in outcome.reason.lower()


async def test_a_readback_does_discharge_a_state_criterion_beside_a_real_proof() -> (
    None
):
    """The positive control for the case above.

    Same shape, but AC2 asserts live state. The readback discharges it, which
    is precisely what the ruling admits. Without this, the test above is also
    satisfied by a change that refuses every readback binding.
    """
    checks = [
        _check("dod-ac1-proof", "verified", "behavior", binds_ac=("AC1",)),
        _check("dod-ac2-readback", "verified", "readback", binds_ac=("AC2",)),
    ]
    description = (
        "## Acceptance criteria\n"
        "\n"
        "- **AC1** a red-first test fails today and passes after the change.\n"
        "- **AC2** the mirror table has 4 rows, read back from the live "
        "database.\n"
    )
    linear = _FakeLinear(description)
    _seed_redraw(linear, checks)
    result = await _handler(_skill_result(checks), linear).handle(_request())
    assert result.outcomes[0].decision is EnumEvidenceAutocloseDecision.FLIPPED
