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

import re
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


# --------------------------------------------------------------------------
# The vocabulary, measured against the criteria it has to read.
# --------------------------------------------------------------------------

#: OMN-17771's acceptance section, verbatim from the ticket. The ruling names
#: this ticket as its worked example, and the first cut of the marker set
#: recognised only 2 of these 5 — so the example would still have held. Each
#: phrase added afterwards is lifted from a criterion below rather than
#: invented, which is the standing rule for this set.
_OMN_17771_ACCEPTANCE = (
    "## Acceptance\n"
    "\n"
    "AC1. Zero `client_id=omnidash-spa` and zero `client_id=omniweb` "
    "occurrences in any customer-facing guide, proven by grep at the merged "
    "tip.\n"
    "AC2. The Step 1a registration URL, pasted verbatim, returns HTTP 200 "
    "with a registration form — probed from a machine with no source checkout "
    "and no org credential, not asserted by inspection.\n"
    "AC3. The Step 1b token command names a client that exists on the plane "
    "the guide names, proven by an `invalid_grant` (user rejected) rather "
    "than `unauthorized_client` / `Client not found` (client rejected) "
    "response against a deliberately invalid user.\n"
    "AC4. `no-private-repo-links` and `beta-layout` checkers green.\n"
    "AC5. Merged to `main` and read back from `origin/main`.\n"
)


def test_the_worked_example_reads_as_state_shaped_end_to_end() -> None:
    """Every one of OMN-17771's five real criteria, not a paraphrase.

    This is the assertion that would have caught the first cut shipping a
    predicate too tight to admit the ticket the ruling names.
    """
    from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
        _acceptance_criteria_items,
        _criterion_is_state_shaped,
        _every_criterion_is_state_shaped,
    )

    items = _acceptance_criteria_items(_OMN_17771_ACCEPTANCE)
    assert len(items) == 5
    assert all(_criterion_is_state_shaped(item) for item in items)
    assert _every_criterion_is_state_shaped(_OMN_17771_ACCEPTANCE) is True


@pytest.mark.parametrize(
    "criterion",
    [
        # The widened phrases must not drag behaviour criteria with them.
        "AC1 the handler exists on the plane and refuses a bad envelope",
        "AC1 a test asserts zero occurrences of the old symbol",
        "AC1 merged to `main`, and the retry path stops after 3 attempts",
        "AC1 the suite is green",
    ],
)
def test_the_widened_vocabulary_did_not_drag_behaviour_along(criterion: str) -> None:
    """Each of these carries a widened state phrase AND behaviour language.

    The veto still wins, which is the property that keeps the widening from
    becoming a general loosening.
    """
    from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
        _criterion_is_state_shaped,
    )

    assert _criterion_is_state_shaped(criterion) is False


@pytest.mark.parametrize(
    ("marker", "criterion"),
    [
        # Each added marker, in ISOLATION, on the shortest criterion that
        # carries it and nothing else. The end-to-end assertion above cannot
        # do this job: OMN-17771's five criteria each match, but a marker that
        # never fires, or one whose semantics are wrong, is invisible there
        # because a sibling marker carries the assertion. Raised by the
        # hostile reviewer on this PR and fixed rather than rejected.
        ("zero ... occurrences", "AC1 zero `client_id=x` occurrences in the guides"),
        ("checkers green", "AC1 the `beta-layout` checkers green"),
        ("exists on the", "AC1 a client that exists on the plane"),
        ("merged to `x", "AC1 merged to `main`"),
        # NOT "returns HTTP 200", which is how the criterion actually reads:
        # `\breturns?\b` is a pre-existing marker, so that phrasing classifies
        # whether or not the HTTP marker fires and asserts nothing about it.
        # Measured, not assumed -- see the mutation control below.
        ("HTTP <status>", "AC1 the URL gives HTTP 200"),
        # And the markers the first cut shipped with, held to the same bar.
        ("read back", "AC1 the value is read back from the running system"),
        ("N rows", "AC1 the mirror holds 4 rows"),
        ("N occurrences", "AC1 the file has 0 occurrences of the old id"),
        ("is running", "AC1 the deployment is Running"),
        ("N restarts", "AC1 the pod shows 0 restarts"),
        ("digest", "AC1 the plane carries the new digest"),
    ],
)
def test_each_state_marker_fires_on_its_own(marker: str, criterion: str) -> None:
    """One marker, one criterion, no sibling to carry it."""
    from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
        _criterion_is_state_shaped,
    )

    assert _criterion_is_state_shaped(criterion) is True, marker


def test_a_criterion_carrying_none_of_the_markers_does_not_fire() -> None:
    """The control that makes the per-marker suite mean something.

    Without it, a predicate that returned True unconditionally would pass
    every case above.
    """
    from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
        _criterion_is_state_shaped,
    )

    assert _criterion_is_state_shaped("AC1 the thing is done properly") is False


@pytest.mark.parametrize(
    ("marker", "near_miss"),
    [
        # Semantics, not just presence. Each of these is the neighbouring
        # string the marker must NOT accept, which is what catches a pattern
        # that is too loose rather than one that never fires.
        ("HTTP <status>", "AC1 the doc mentions HTTP and a status somewhere"),
        ("merged to `x", "AC1 the branch was merged to"),
        ("exists on the", "AC1 the file exists"),
        ("checkers green", "AC1 the checkers ran"),
        ("zero ... occurrences", "AC1 zero is an interesting number"),
    ],
)
def test_each_marker_refuses_its_near_miss(marker: str, near_miss: str) -> None:
    """A marker that accepts its near miss is too loose to be evidence."""
    from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
        _criterion_is_state_shaped,
    )

    assert _criterion_is_state_shaped(near_miss) is False, marker


# --------------------------------------------------------------------------
# Per-marker isolation. Review finding on #3413: the blob test above asserts
# the whole OMN-17771 body classifies as state-shaped, so a marker that never
# fires, or one whose semantics are wrong, is invisible because a neighbouring
# marker carries the assertion. That is not hypothetical on this body: AC2
# ("returns HTTP 200 ...") is already matched by the older `\breturns?\b`
# marker, and AC5 ("Merged to `main` and read back ...") by `\bread back\b`,
# so the two markers added for them contribute nothing to the aggregate and
# could be deleted with every test above still green.
#
# The two tests below close that. Each new marker gets a criterion carrying
# ONLY that marker, and a mutation control that deletes the marker's own
# alternation line from the vocabulary and asserts the same criterion stops
# classifying. A marker that never fires cannot pass its control.
# --------------------------------------------------------------------------


def _vocabulary_without(marker_source: str) -> re.Pattern[str]:
    """`_STATE_MARKER_RE` with one alternation line deleted, nothing else.

    Deleting the marker rather than rewriting the criterion is what makes
    this a control: the criterion is held byte-identical across the pair, so
    a difference in verdict is attributable to that one line and nothing
    else.

    Refuses anything it cannot do exactly: the fragment must identify exactly
    one line, and that line must be an alternation (`| ...`) rather than the
    pattern's first branch or its `(?xi)` header, either of which would leave
    a regex that is broken or silently means something else. A mutation
    control built on a mangled pattern proves nothing.
    """
    from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
        _STATE_MARKER_RE,
    )

    lines = _STATE_MARKER_RE.pattern.split("\n")
    hits = [i for i, line in enumerate(lines) if marker_source in line]
    assert len(hits) == 1, f"{marker_source!r} matched {len(hits)} lines, want 1"
    assert lines[hits[0]].lstrip().startswith("|"), (
        f"{marker_source!r} is not an alternation branch; deleting it would "
        "change the pattern's meaning rather than remove one marker"
    )
    del lines[hits[0]]
    return re.compile("\n".join(lines), _STATE_MARKER_RE.flags)


#: One row per phrase OMN-18135 added: the marker's own source text, and a
#: criterion lifted from the OMN-17771 criterion that phrase was taken from,
#: reduced until that marker is the ONLY thing making it state-shaped.
#:
#: The HTTP row says "gives HTTP 200" where the ticket says "returns HTTP
#: 200", and the merged row drops "and read back from `origin/main`". Those
#: are not paraphrases for convenience — they are the isolation. `returns`
#: and `read back` are pre-existing markers, and leaving either in place
#: would mean the row passed whether or not the new marker fires, which is
#: the exact defect these tests exist to detect.
_ADDED_STATE_MARKERS: tuple[tuple[str, str, str], ...] = (
    (
        "zero-occurrences",
        r"\bzero\b[^.]{0,80}\boccurrences?\b",
        "AC1 Zero `client_id=omnidash-spa` occurrences in any customer-facing guide",
    ),
    (
        "checkers-green",
        r"\bcheckers?\b[^.]{0,24}\bgreen\b",
        "AC4 `no-private-repo-links` and `beta-layout` checkers green",
    ),
    (
        "exists-on-the",
        r"\bexists\s+on\s+the\b",
        "AC3 the Step 1b token command names a client that exists on the plane "
        "the guide names",
    ),
    (
        "merged-to",
        r"\bmerged\s+to\s+`?\w",
        "AC5 Merged to `main`",
    ),
    (
        "http-status",
        r"\bHTTP\s+\d{3}\b",
        "AC2 the Step 1a registration URL, pasted verbatim, gives HTTP 200 with "
        "a registration form",
    ),
)

_ADDED_MARKER_PARAMS = [
    pytest.param(marker, criterion, id=marker_id)
    for marker_id, marker, criterion in _ADDED_STATE_MARKERS
]


@pytest.mark.parametrize(("marker", "criterion"), _ADDED_MARKER_PARAMS)
def test_each_added_marker_recognises_its_own_criterion(
    marker: str, criterion: str
) -> None:
    """Each added phrase carries a criterion on its own, with no neighbour."""
    from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
        _criterion_is_state_shaped,
    )

    assert _criterion_is_state_shaped(criterion) is True


@pytest.mark.parametrize(("marker", "criterion"), _ADDED_MARKER_PARAMS)
def test_each_added_marker_is_the_only_thing_carrying_its_criterion(
    monkeypatch: pytest.MonkeyPatch, marker: str, criterion: str
) -> None:
    """Mutation control. Delete the marker; its criterion must stop matching.

    This is the half the aggregate test cannot do. If the phrase were
    misspelled, over-escaped, or shadowed by a marker that was already there,
    the criterion would still classify with the line gone and this fails.

    The second assertion holds the mutation to one marker: every OTHER added
    marker's criterion must be unaffected. Without it, a helper that returned
    an empty or inverted pattern would satisfy the first assertion for every
    row at once.
    """
    import omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep as _module
    from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
        _criterion_is_state_shaped,
    )

    monkeypatch.setattr(_module, "_STATE_MARKER_RE", _vocabulary_without(marker))

    assert _criterion_is_state_shaped(criterion) is False
    for _id, other_marker, other_criterion in _ADDED_STATE_MARKERS:
        if other_marker == marker:
            continue
        assert _criterion_is_state_shaped(other_criterion) is True, (
            f"deleting {marker!r} also stopped {other_marker!r} from firing"
        )


@pytest.mark.parametrize(("marker", "criterion"), _ADDED_MARKER_PARAMS)
def test_no_added_marker_overrides_the_behaviour_veto(
    marker: str, criterion: str
) -> None:
    """The same criterion, plus behaviour language, holds.

    Per-marker rather than per-phrase-sample: the veto has to win against
    every added marker individually, not just against the four combinations
    the earlier control happens to spell.
    """
    from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
        _criterion_is_state_shaped,
    )

    assert _criterion_is_state_shaped(criterion) is True
    assert _criterion_is_state_shaped(f"{criterion}, asserted by a unit test") is False
