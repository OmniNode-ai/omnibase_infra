# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17942: name every In-Progress ticket no mechanical closer can reach.

A ticket whose OCC contract declares no check is not *failing* the evidence
closer — it is INVISIBLE to it. The closer enumerates merged OCC companions, so
a ticket with none never appears in its outcomes at all, and that absence reads
exactly like "nothing to report". Four tickets in the 2026-08-31 sprint are in
that state.

The two properties these tests exist to hold are: the sweep never changes
state, and it says a thing ONCE.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_in_progress_probe_hygiene_effect.handlers.handler_in_progress_probe_hygiene import (
    _HYGIENE_COMMENT_MARKER,
    HandlerInProgressProbeHygiene,
    _occ_contract_check_count,
    _well_formed_probe_lines,
)
from omnibase_infra.nodes.node_in_progress_probe_hygiene_effect.models.enum_probe_hygiene_decision import (
    EnumProbeHygieneDecision,
)
from omnibase_infra.nodes.node_in_progress_probe_hygiene_effect.models.model_probe_hygiene_request import (
    ModelProbeHygieneRequest,
)

pytestmark = pytest.mark.unit


class _FakeLinear:
    """Records every write so a test can assert none happened."""

    last_error = ""

    def __init__(
        self,
        issues: list[dict[str, object]] | None,
        markers: dict[str, bool | None] | None = None,
        comment_succeeds: bool = True,
    ) -> None:
        self._issues = issues
        self._markers = markers or {}
        self._comment_succeeds = comment_succeeds
        self.comments: list[tuple[str, str]] = []

    async def fetch_in_progress(
        self, project: str, max_tickets: int
    ) -> list[dict[str, object]] | None:
        return self._issues

    async def has_marker(self, issue_id: str, marker: str) -> bool | None:
        return self._markers.get(issue_id, False)

    async def create_comment(self, issue_id: str, body: str) -> bool:
        if self._comment_succeeds:
            self.comments.append((issue_id, body))
        return self._comment_succeeds


def _issue(identifier: str, description: str = "") -> dict[str, object]:
    return {
        "id": f"uuid-{identifier}",
        "identifier": identifier,
        "title": f"{identifier} title",
        "description": description,
        "state": {"name": "In Progress", "type": "started"},
    }


def _request(occ_dir: Path, **overrides: object) -> ModelProbeHygieneRequest:
    payload: dict[str, object] = {
        "correlation_id": uuid4(),
        "occ_repo_dir": str(occ_dir),
        "apply": True,
        "linear_retry_base_delay_seconds": 0.0,
    }
    payload.update(overrides)
    return ModelProbeHygieneRequest(**payload)  # type: ignore[arg-type]


@pytest.fixture
def occ_dir(tmp_path: Path) -> Path:
    (tmp_path / "contracts").mkdir()
    return tmp_path


def _write_contract(occ_dir: Path, ticket: str, checks: int) -> None:
    items = "\n".join(
        f'      - check_type: "command"\n        check_value: "echo {i}"'
        for i in range(checks)
    )
    body = f'---\nticket_id: "{ticket}"\ndod_evidence:\n  - id: "dod-1"\n    checks:\n{items}\n'
    if checks == 0:
        body = f'---\nticket_id: "{ticket}"\ndod_evidence: []\n'
    (occ_dir / "contracts" / f"{ticket}.yaml").write_text(body, encoding="utf-8")


# ---------------------------------------------------------------------------
# Probe-line grammar — the same one the OMN-17942 creation gate admits
# ---------------------------------------------------------------------------


def test_a_well_formed_probe_line_counts() -> None:
    assert _well_formed_probe_lines("Probe: uv run pytest -q => exits 0") == 1


def test_a_probe_missing_its_observation_does_not_count() -> None:
    """A command with no expected observation is adjudicated by a human read.

    Counting it would report the problem this sweep exists to find as solved.
    """
    assert _well_formed_probe_lines("Probe: uv run pytest -q") == 0


def test_a_bulleted_probe_does_not_count() -> None:
    assert _well_formed_probe_lines("- Probe: uv run pytest -q => exits 0") == 0


def test_prose_mentioning_a_probe_does_not_count() -> None:
    assert _well_formed_probe_lines("This has no Probe: line yet, sadly.") == 0


# ---------------------------------------------------------------------------
# OCC contract read — "declares none" vs "could not look"
# ---------------------------------------------------------------------------


def test_contract_checks_are_counted_across_evidence_items(occ_dir: Path) -> None:
    _write_contract(occ_dir, "OMN-1", checks=3)
    assert _occ_contract_check_count(str(occ_dir), "OMN-1") == 3


def test_a_readable_clone_with_no_contract_for_the_ticket_is_zero_not_unknown(
    occ_dir: Path,
) -> None:
    assert _occ_contract_check_count(str(occ_dir), "OMN-404") == 0


def test_no_clone_at_all_is_unknown_not_zero(tmp_path: Path) -> None:
    """The distinction the whole finding rests on.

    Zero and unknown have the same cardinality and opposite meanings:
    collapsing them reports a broken runner as a board-wide finding.
    """
    assert _occ_contract_check_count("", "OMN-1") is None
    assert _occ_contract_check_count(str(tmp_path / "nope"), "OMN-1") is None


def test_a_ticket_id_that_is_not_a_ticket_id_is_refused(occ_dir: Path) -> None:
    """It arrives from the Linear API; a path segment built from remote text."""
    assert _occ_contract_check_count(str(occ_dir), "../../etc/passwd") is None
    assert _occ_contract_check_count(str(occ_dir), "") is None


def test_malformed_contract_yaml_is_unknown(occ_dir: Path) -> None:
    (occ_dir / "contracts" / "OMN-2.yaml").write_text("{[", encoding="utf-8")
    assert _occ_contract_check_count(str(occ_dir), "OMN-2") is None


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_ticket_with_no_probe_anywhere_is_commented_once(
    occ_dir: Path,
) -> None:
    _write_contract(occ_dir, "OMN-17926", checks=0)
    linear = _FakeLinear([_issue("OMN-17926")])
    result = await HandlerInProgressProbeHygiene(linear).handle(_request(occ_dir))
    assert result.tickets_without_probe == 1
    assert result.tickets_commented == 1
    assert len(linear.comments) == 1
    issue_id, body = linear.comments[0]
    assert issue_id == "uuid-OMN-17926"
    assert _HYGIENE_COMMENT_MARKER in body
    assert "contracts/OMN-17926.yaml" in body


@pytest.mark.asyncio
async def test_a_ticket_with_a_contract_check_is_left_alone(occ_dir: Path) -> None:
    _write_contract(occ_dir, "OMN-1", checks=1)
    linear = _FakeLinear([_issue("OMN-1")])
    result = await HandlerInProgressProbeHygiene(linear).handle(_request(occ_dir))
    assert result.tickets_with_probe == 1
    assert result.tickets_without_probe == 0
    assert linear.comments == []


@pytest.mark.asyncio
async def test_a_description_probe_line_is_a_second_chance(occ_dir: Path) -> None:
    """Every ticket filed before the creation gate landed has no probe line.

    So the contract is checked first and the description is the second chance,
    not the only one.
    """
    _write_contract(occ_dir, "OMN-2", checks=0)
    linear = _FakeLinear(
        [_issue("OMN-2", "Probe: uv run pytest tests/x.py -q => exits 0")]
    )
    result = await HandlerInProgressProbeHygiene(linear).handle(_request(occ_dir))
    assert result.tickets_with_probe == 1
    assert linear.comments == []


@pytest.mark.asyncio
async def test_the_sweep_says_it_once(occ_dir: Path) -> None:
    """Marker-line dedup (OMN-16808). Without it: one comment per tick forever."""
    _write_contract(occ_dir, "OMN-3", checks=0)
    linear = _FakeLinear([_issue("OMN-3")], markers={"uuid-OMN-3": True})
    result = await HandlerInProgressProbeHygiene(linear).handle(_request(occ_dir))
    assert linear.comments == []
    assert result.outcomes[0].decision is (
        EnumProbeHygieneDecision.SKIPPED_ALREADY_COMMENTED
    )
    # Still counted as a finding: the standing list must stay visible even
    # though only the first run wrote to the board.
    assert result.tickets_without_probe == 1


@pytest.mark.asyncio
async def test_an_unreadable_comment_history_fails_closed(occ_dir: Path) -> None:
    """'No prior comment' and 'could not look' must not collapse."""
    _write_contract(occ_dir, "OMN-4", checks=0)
    linear = _FakeLinear([_issue("OMN-4")], markers={"uuid-OMN-4": None})
    result = await HandlerInProgressProbeHygiene(linear).handle(_request(occ_dir))
    assert linear.comments == []
    assert result.tickets_errored == 1
    assert result.outcomes[0].decision is EnumProbeHygieneDecision.ERROR_LINEAR_API


@pytest.mark.asyncio
async def test_a_dry_run_reaches_the_verdict_and_writes_nothing(
    occ_dir: Path,
) -> None:
    _write_contract(occ_dir, "OMN-5", checks=0)
    linear = _FakeLinear([_issue("OMN-5")])
    result = await HandlerInProgressProbeHygiene(linear).handle(
        _request(occ_dir, apply=False)
    )
    assert linear.comments == []
    assert result.dry_run is True
    assert result.tickets_without_probe == 1
    assert result.outcomes[0].decision is EnumProbeHygieneDecision.SKIPPED_DRY_RUN


@pytest.mark.asyncio
async def test_the_comment_budget_bounds_a_backlog_without_hiding_it(
    occ_dir: Path,
) -> None:
    """The first run after this lands has a standing backlog to work through.

    Writing all of it at once is a notification storm. A ticket past the budget
    is still REPORTED as a finding — its turn simply has not come round.
    """
    for n in range(5):
        _write_contract(occ_dir, f"OMN-{100 + n}", checks=0)
    linear = _FakeLinear([_issue(f"OMN-{100 + n}") for n in range(5)])
    result = await HandlerInProgressProbeHygiene(linear).handle(
        _request(occ_dir, max_comments_per_run=2)
    )
    assert len(linear.comments) == 2
    assert result.tickets_commented == 2
    assert result.tickets_without_probe == 5
    budget_skips = [
        outcome
        for outcome in result.outcomes
        if outcome.decision is EnumProbeHygieneDecision.SKIPPED_COMMENT_BUDGET_EXHAUSTED
    ]
    assert len(budget_skips) == 3


@pytest.mark.asyncio
async def test_a_fenced_ticket_is_refused_before_any_read(occ_dir: Path) -> None:
    _write_contract(occ_dir, "OMN-6", checks=0)
    linear = _FakeLinear([_issue("OMN-6")])
    result = await HandlerInProgressProbeHygiene(linear).handle(
        _request(occ_dir, exclude_tickets=("omn-6",))
    )
    assert linear.comments == []
    assert result.outcomes[0].decision is EnumProbeHygieneDecision.SKIPPED_EXCLUDED


@pytest.mark.asyncio
async def test_an_unreadable_occ_clone_reports_an_error_not_a_finding(
    tmp_path: Path,
) -> None:
    """A broken runner must never render as 'no ticket has a probe'."""
    linear = _FakeLinear([_issue("OMN-7")])
    result = await HandlerInProgressProbeHygiene(linear).handle(
        ModelProbeHygieneRequest(
            correlation_id=uuid4(),
            occ_repo_dir=str(tmp_path / "missing"),
            apply=True,
            linear_retry_base_delay_seconds=0.0,
        )
    )
    assert linear.comments == []
    assert result.tickets_without_probe == 0
    assert result.tickets_errored == 1
    assert result.outcomes[0].decision is (
        EnumProbeHygieneDecision.ERROR_CONTRACT_UNREADABLE
    )


@pytest.mark.asyncio
async def test_an_unreadable_enumeration_reports_nothing_rather_than_zero(
    occ_dir: Path,
) -> None:
    """An empty sweep and a failed sweep have the same shape, opposite meanings."""
    linear = _FakeLinear(None)
    result = await HandlerInProgressProbeHygiene(linear).handle(_request(occ_dir))
    assert result.success is False
    assert result.tickets_scanned == 0
    assert result.error_message


@pytest.mark.asyncio
async def test_a_failed_comment_write_is_an_error_not_a_success(
    occ_dir: Path,
) -> None:
    _write_contract(occ_dir, "OMN-8", checks=0)
    linear = _FakeLinear([_issue("OMN-8")], comment_succeeds=False)
    result = await HandlerInProgressProbeHygiene(linear).handle(_request(occ_dir))
    assert result.tickets_commented == 0
    assert result.tickets_errored == 1


@pytest.mark.asyncio
async def test_the_sweep_has_no_state_mutation_surface_at_all(occ_dir: Path) -> None:
    """The property that makes this safe to run unattended, read off the class.

    A hygiene sweep that could move a ticket is a closer with no predicate. The
    client exposes no state mutation, so there is nothing for a later edit to
    call by accident.
    """
    from omnibase_infra.nodes.node_in_progress_probe_hygiene_effect.handlers import (
        handler_in_progress_probe_hygiene as mod,
    )

    client_methods = {
        name for name in dir(mod.LinearHygieneTransport) if not name.startswith("_")
    }
    assert client_methods == {
        "apply_retry_policy",
        "create_comment",
        "fetch_in_progress",
        "has_marker",
        "query",
        "required_secrets",
    }
    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert "issueUpdate" not in source
    assert "stateId" not in source
