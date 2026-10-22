# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21272 -- a job that ended not_started is INDETERMINATE, never FAIL.

THE INCIDENT, 2026-10-21. An out-of-band ``refresh_dev_lane.sh --ref`` for a
commit not on dev held the compose lane lock from 10:51:31Z. Deploy-agent job
f19ec60c was accepted at 10:57:06Z and ended ``failed`` with
``not_started=lane_lock_contended`` at 11:12:06Z: it never ran a phase. The
convergence guard counted the lane's 1500 s budget from acceptance and wrote
``deployed_revision`` FAIL at 11:22:25Z for 54ca4ac5f. A FAIL on that check is
never re-emittable, so nothing measured the sha again and the release train
skipped five runs on a verdict about a deploy that never happened.

What is pinned: the agent's record of a not_started end turns that verdict
into INDETERMINATE naming the reason and the correlation id, at any point in
the wait and even once the budget has passed -- and the resulting receipt is
binding-only. A job with no not_started reason still FAILs, byte for byte.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ci.check_dev_lane_staleness import (
    AcceptanceUnresolvedError,
    Ancestry,
    EnumConvergenceOutcome,
    LaneRevision,
    ModelAcceptanceProbe,
    ModelAgentAcceptance,
    ModelConvergenceResult,
    convergence_check_outcome,
    read_agent_acceptance,
    run_convergence_wait,
)
from scripts.ci.lab_pass_receipt import (
    EnumLabLane,
    EnumLabPassCheckOutcome,
    ModelLabPassCheck,
    build_receipt,
    is_binding_only_failure,
)

pytestmark = pytest.mark.unit

MERGE_SHA = "54ca4ac5f" + "0" * 31
STALE_SHA = "843fe808cc7891a1b2c3d4e5f60718293a4b5c6d"
CORRELATION_ID = "f19ec60c-1111-4c2a-9f3b-2a6c8d4e5f60"
SOURCE = f"http://host.docker.internal:8098/job/{CORRELATION_ID}"

# The incident's recorded values (ledger FRICTION 2026-10-21T17:58:33Z).
STEP_START = datetime(2026, 10, 21, 10, 56, 30, tzinfo=UTC)
ACCEPTED_AT = datetime(2026, 10, 21, 10, 57, 6, tzinfo=UTC)
ENDED_AT = datetime(2026, 10, 21, 11, 12, 6, tzinfo=UTC)
DECLARED = timedelta(seconds=1500)
WALL_CLOCK = timedelta(seconds=45 * 60 - 900 - 120)


class _Clock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now = self.now + timedelta(seconds=seconds)


def _lane(revision: str) -> LaneRevision:
    return LaneRevision(
        revision=revision,
        compose_project="omnibase-infra",
        build_source="workspace",
        state="running",
    )


def _wait(
    *,
    not_started: str,
    ended_at: datetime,
    visible_from: datetime | None = None,
) -> ModelConvergenceResult:
    """Drive the real loop: the lane never converges; the job ends at ``ended_at``.

    ``visible_from`` delays when the agent's record shows the end, to pin the
    last look before a FAIL is written.
    """
    clock = _Clock(STEP_START)
    shown_from = visible_from or ended_at

    def resolve_acceptance() -> ModelAcceptanceProbe:
        if clock.now < ACCEPTED_AT:
            return ModelAcceptanceProbe(
                acceptance=None,
                reason=f"the deploy agent reports no job for {CORRELATION_ID} yet",
            )
        ended = clock.now >= shown_from
        return ModelAcceptanceProbe(
            acceptance=ModelAgentAcceptance(
                correlation_id=CORRELATION_ID,
                accepted_at=ACCEPTED_AT,
                source=SOURCE,
                status="failed" if ended else "accepted",
                not_started=not_started if ended else "",
            ),
            reason="",
        )

    return run_convergence_wait(
        expected_revision=MERGE_SHA,
        read_lane=lambda: _lane(STALE_SHA),
        resolve_ancestry=lambda _obs: Ancestry(
            relation="ancestor", commits_ahead=0, observed_on_branch=True, branch="dev"
        ),
        resolve_acceptance=resolve_acceptance,
        declared_budget=DECLARED,
        wall_clock=WALL_CLOCK,
        poll_interval=timedelta(seconds=60),
        clock=clock,
        sleep=clock.sleep,
    )


def test_not_started_is_indeterminate_for_the_incident_values() -> None:
    result = _wait(not_started="lane_lock_contended", ended_at=ENDED_AT)

    assert result.outcome is EnumConvergenceOutcome.INDETERMINATE
    assert "not_started=lane_lock_contended" in result.reason
    assert CORRELATION_ID in result.reason
    # The wait stops on the agent's own end record rather than spending the
    # budget on a deploy that will not happen.
    assert result.finished_at < ACCEPTED_AT + DECLARED
    assert (
        convergence_check_outcome(result.outcome)
        is EnumLabPassCheckOutcome.INDETERMINATE
    )


def test_not_started_is_indeterminate_even_after_the_budget_has_passed() -> None:
    # The agent's record only shows the end once the declared budget from
    # acceptance has run out -- the instant the pre-change loop wrote FAIL.
    result = _wait(
        not_started="lane_lock_contended",
        ended_at=ENDED_AT,
        visible_from=ACCEPTED_AT + DECLARED,
    )

    assert result.finished_at >= ACCEPTED_AT + DECLARED
    assert result.outcome is EnumConvergenceOutcome.INDETERMINATE
    assert "lane_lock_contended" in result.reason
    assert CORRELATION_ID in result.reason


def test_not_started_is_indeterminate_and_the_receipt_is_binding_only() -> None:
    result = _wait(not_started="lane_lock_contended", ended_at=ENDED_AT)
    outcome = convergence_check_outcome(result.outcome)
    receipt = build_receipt(
        sha=MERGE_SHA,
        lane=EnumLabLane.COMPOSE_DEV,
        started_at=STEP_START,
        finished_at=result.finished_at,
        checks=[
            ModelLabPassCheck(
                name="deployed_revision",
                ok=False,
                evidence=f"INDETERMINATE: {result.reason}",
                indeterminate=outcome is EnumLabPassCheckOutcome.INDETERMINATE,
            ),
            ModelLabPassCheck(name="ready_main", ok=True, evidence="read fine"),
        ],
        agent_command_id=None,
    )
    assert is_binding_only_failure(receipt) is True


def test_not_started_is_indeterminate_control_without_reason_still_fails() -> None:
    # Positive control: the same job ending with no not_started reason is the
    # lane's budget spent without convergence -- FAIL, unchanged.
    result = _wait(not_started="", ended_at=ENDED_AT)

    assert result.outcome is EnumConvergenceOutcome.FAIL
    assert result.reason == ""


def test_not_started_is_indeterminate_reads_the_agent_route() -> None:
    def opener(url: str, timeout: float) -> tuple[int, str]:
        return 200, json.dumps(
            {
                "correlation_id": CORRELATION_ID,
                "status": "failed",
                "not_started": "lane_lock_contended",
                "errors": [],
                "accepted_at": ACCEPTED_AT.isoformat(),
                "completed_at": ENDED_AT.isoformat(),
            }
        )

    acceptance = read_agent_acceptance(
        "http://agent:8098", CORRELATION_ID, opener=opener
    )
    assert acceptance.status == "failed"
    assert acceptance.not_started == "lane_lock_contended"


def test_not_started_is_indeterminate_older_agent_reads_as_started() -> None:
    def opener(url: str, timeout: float) -> tuple[int, str]:
        return 200, json.dumps(
            {
                "status": "in_progress",
                "errors": [],
                "accepted_at": ACCEPTED_AT.isoformat(),
            }
        )

    acceptance = read_agent_acceptance(
        "http://agent:8098", CORRELATION_ID, opener=opener
    )
    assert acceptance.not_started == ""


@pytest.mark.live_contact(
    "tests/ci/fixtures/omn21272_lock_contended_job_incident_replay.json"
)
def test_not_started_is_indeterminate_replays_the_incident(
    recorded_response: dict[str, Any],
) -> None:
    """The incident's job record, read through the real route reader and wait loop.

    Before 11:12:06Z the agent served the job as accepted; from then on, the
    recorded terminal body. The pre-change guard wrote FAIL at 11:22:25Z.
    """
    body = recorded_response["response"]
    accepted_at = datetime.fromisoformat(body["accepted_at"])
    ended_at = datetime.fromisoformat(body["completed_at"])
    clock = _Clock(datetime.fromisoformat(recorded_response["guard_started_at"]))
    in_flight = {**body, "status": "accepted", "not_started": None}
    in_flight["completed_at"] = None

    def opener(url: str, timeout: float) -> tuple[int, str]:
        if clock.now < accepted_at:
            return 404, '{"error": "not found"}'
        return 200, json.dumps(body if clock.now >= ended_at else in_flight)

    def resolve_acceptance() -> ModelAcceptanceProbe:
        try:
            acceptance = read_agent_acceptance(
                "http://agent:8098", body["correlation_id"], opener=opener
            )
        except AcceptanceUnresolvedError as exc:
            return ModelAcceptanceProbe(acceptance=None, reason=str(exc))
        return ModelAcceptanceProbe(acceptance=acceptance, reason="")

    result = run_convergence_wait(
        expected_revision=recorded_response["merge_sha"],
        read_lane=lambda: _lane(STALE_SHA),
        resolve_ancestry=lambda _obs: Ancestry(
            relation="ancestor", commits_ahead=0, observed_on_branch=True, branch="dev"
        ),
        resolve_acceptance=resolve_acceptance,
        declared_budget=timedelta(seconds=recorded_response["declared_budget_seconds"]),
        wall_clock=WALL_CLOCK,
        poll_interval=timedelta(seconds=60),
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.outcome is EnumConvergenceOutcome.INDETERMINATE
    assert "not_started=lane_lock_contended" in result.reason
    assert body["correlation_id"] in result.reason
    assert result.finished_at < datetime.fromisoformat(
        recorded_response["fail_written_at"]
    )
