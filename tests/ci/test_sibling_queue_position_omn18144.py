# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18144 AC1 -- the sibling-revision guard reads the deploy agent's queue position.

The infra-revision guard (``check_dev_lane_staleness.py``) already sizes its wait
from the agent's ``/queue`` surface. The sibling guard watched on the clock alone:
a rebuild command queued behind another deploy spent the whole affordable wait
(1680s) on the one verify runner to arrive at an INDETERMINATE that named no
depth. These tests pin the same read in ``run_sibling_convergence_wait``:

* a queue the window cannot outlast -> INDETERMINATE naming the depth, taken
  without waiting, and never a FAIL;
* nothing ahead, an unreadable queue, an already-accepted command and an already
  converged lane -> the wait behaves exactly as before.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from scripts.ci.check_dev_lane_staleness import (
    EnumConvergenceOutcome,
    ModelAcceptanceProbe,
    ModelAgentAcceptance,
    ModelQueueFacts,
)
from scripts.ci.check_lane_sibling_revision import (
    ModelSiblingObservation,
    run_sibling_convergence_wait,
    sibling_convergence_evidence,
)

pytestmark = pytest.mark.unit

MERGE_SHA = "8d52a7ccf56333ceaf82039cdbe82b6f30653a98"
STALE_SHA = "948be2e1a517d04d63f6d0a1f2e3c4b5a6978899"
CORRELATION_ID = "0a3d0f1e-1111-4c2a-9f3b-2a6c8d4e5f60"
T0 = datetime(2026, 9, 18, 17, 0, 0, tzinfo=UTC)
DECLARED = timedelta(minutes=25)
#: The job's 45-minute ceiling less the declared settle budget and the reserved tail.
WALL_CLOCK = timedelta(seconds=45 * 60 - 900 - 120)
POLL = timedelta(seconds=60)
SERVICE_SECONDS = 24 * 60


class _Clock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now = self.now + timedelta(seconds=seconds)


def _queue(commands_ahead: int) -> ModelQueueFacts:
    return ModelQueueFacts(
        commands_ahead=commands_ahead,
        mean_service_time_seconds=SERVICE_SECONDS,
        service_sample_size=6,
        in_flight_correlation_id="7f6e5d4c-2222-4b3a-8c1d-9e0f1a2b3c4d",
        unread_reason="",
        source="http://host.docker.internal:8098/queue",
    )


def _wait(
    *,
    queue: ModelQueueFacts | None,
    converges_after: timedelta | None,
    accepted_at: datetime | None = None,
    queue_reads: list[int] | None = None,
):
    clock = _Clock(T0)

    def observe() -> ModelSiblingObservation:
        if converges_after is not None and clock.now >= T0 + converges_after:
            return ModelSiblingObservation(
                lane_revision=MERGE_SHA, containment="identical", unreadable_reason=""
            )
        return ModelSiblingObservation(
            lane_revision=STALE_SHA, containment="behind", unreadable_reason=""
        )

    def resolve_acceptance() -> ModelAcceptanceProbe:
        if accepted_at is None:
            return ModelAcceptanceProbe(
                acceptance=None,
                reason=f"the deploy agent reports no job for correlation {CORRELATION_ID} yet",
            )
        return ModelAcceptanceProbe(
            acceptance=ModelAgentAcceptance(
                correlation_id=CORRELATION_ID,
                accepted_at=accepted_at,
                source=f"http://host.docker.internal:8098/job/{CORRELATION_ID}",
            ),
            reason="",
        )

    def resolve_queue() -> ModelQueueFacts:
        if queue_reads is not None:
            queue_reads.append(1)
        assert queue is not None
        return queue

    result = run_sibling_convergence_wait(
        observe=observe,
        resolve_acceptance=resolve_acceptance,
        declared_budget=DECLARED,
        wall_clock=WALL_CLOCK,
        poll_interval=POLL,
        clock=clock,
        sleep=clock.sleep,
        resolve_queue=None if queue is None else resolve_queue,
    )
    return result


def test_a_queue_the_window_cannot_outlast_is_indeterminate_naming_the_depth() -> None:
    """Two commands ahead at 24 min each: third in line, refused in zero seconds."""
    result = _wait(queue=_queue(2), converges_after=timedelta(hours=2))
    assert result.outcome is EnumConvergenceOutcome.INDETERMINATE
    assert result.waited == timedelta(0), "the wait was not started"
    assert "number 3 in line" in result.reason
    assert "2 command(s) ahead" in result.reason
    evidence = sibling_convergence_evidence(
        repo="omnimarket", expected_revision=MERGE_SHA, result=result
    )
    assert "queue_position_at_start=3" in evidence
    assert "commands_ahead=2" in evidence
    assert "asserts nothing about the lane" in evidence


def test_nothing_ahead_waits_exactly_as_before() -> None:
    result = _wait(queue=_queue(0), converges_after=timedelta(minutes=10))
    assert result.outcome is EnumConvergenceOutcome.OK
    assert result.waited >= timedelta(minutes=10)


def test_an_unreadable_queue_falls_back_to_the_clock_and_says_so() -> None:
    unread = ModelQueueFacts.unread("http://host.docker.internal:8098/queue timed out")
    result = _wait(queue=unread, converges_after=timedelta(minutes=10))
    assert result.outcome is EnumConvergenceOutcome.OK
    evidence = sibling_convergence_evidence(
        repo="omnimarket", expected_revision=MERGE_SHA, result=result
    )
    assert "queue_position_at_start=UNREAD" in evidence


def test_an_unreadable_queue_that_never_converges_is_not_refused_up_front() -> None:
    unread = ModelQueueFacts.unread("no route")
    result = _wait(queue=unread, converges_after=None)
    assert result.outcome is EnumConvergenceOutcome.INDETERMINATE
    assert result.waited >= WALL_CLOCK - POLL, (
        "the clock bound, not the queue, ended it"
    )


def test_an_accepted_command_ignores_the_queue() -> None:
    """Once the agent has the command, the queue ahead of it is spent."""
    result = _wait(
        queue=_queue(2),
        converges_after=timedelta(minutes=20),
        accepted_at=T0 - timedelta(minutes=1),
    )
    assert result.outcome is EnumConvergenceOutcome.OK


def test_an_already_converged_lane_is_never_refused_for_the_queue() -> None:
    result = _wait(queue=_queue(5), converges_after=timedelta(0))
    assert result.outcome is EnumConvergenceOutcome.OK


def test_no_queue_reader_adds_no_clause_and_changes_nothing() -> None:
    result = _wait(queue=None, converges_after=timedelta(minutes=10))
    assert result.outcome is EnumConvergenceOutcome.OK
    assert result.queue_clause == ""


def test_main_reads_the_agent_queue_and_refuses_a_foregone_wait(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The production wiring: main() hands the agent's /queue read to the wait."""
    from scripts.ci import check_lane_sibling_revision as guard

    asked: list[str] = []

    def fake_queue(agent_url: str, *, request_timeout_seconds: float = 10.0):
        asked.append(agent_url)
        return _queue(2)

    monkeypatch.setattr(guard, "read_agent_queue", fake_queue)
    monkeypatch.setattr(
        guard, "read_compose_project", lambda container: "omnibase-infra"
    )
    monkeypatch.setattr(guard, "read_lane_sibling_revision", lambda *a, **k: STALE_SHA)
    monkeypatch.setattr(guard, "read_containment", lambda *a, **k: "behind")
    monkeypatch.setattr(
        guard,
        "read_agent_acceptance",
        lambda **k: (_ for _ in ()).throw(
            guard.AcceptanceUnresolvedError("the deploy agent reports no job yet")
        ),
    )
    monkeypatch.setattr(guard, "_write_output_generation", lambda container: None)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    code = guard.main(
        [
            "--repo",
            "omnimarket",
            "--repo-slug",
            "OmniNode-ai/omnimarket",
            "--expect-revision",
            MERGE_SHA,
            "--agent-url",
            "http://agent.invalid:8098",
            "--correlation-id",
            CORRELATION_ID,
            "--wall-clock-seconds",
            str(int(WALL_CLOCK.total_seconds())),
        ]
    )
    assert asked == ["http://agent.invalid:8098"]
    assert code == guard.EXIT_INDETERMINATE
    assert "number 3 in line" in capsys.readouterr().out


@pytest.mark.live_contact("tests/ci/fixtures/omn18144_agent_queue_live.json")
def test_recorded_live_agent_queue_is_read_as_unread_and_falls_back_to_the_clock(
    recorded_response: dict[str, object],
) -> None:
    """The deploy agent's real /queue bytes through the guard's own reader.

    Captured from the dev lane's deploy agent: it had committed no offset on the
    control topic yet, so ``commands_ahead`` was null. The reader must report that
    as UNREAD, never as an empty queue, and the sibling wait must then behave as
    the clock-bounded guard did.
    """
    import json

    from scripts.ci.check_dev_lane_staleness import read_agent_queue

    body = json.dumps(recorded_response["response"])
    facts = read_agent_queue(
        "http://agent.invalid:8098",
        opener=lambda url, timeout: (200, body),
    )
    assert facts.commands_ahead is None
    assert "committed no offset" in facts.unread_reason
    result = _wait(queue=facts, converges_after=timedelta(minutes=10))
    assert result.outcome is EnumConvergenceOutcome.OK
    assert "queue_position_at_start=UNREAD" in sibling_convergence_evidence(
        repo="omnimarket", expected_revision=MERGE_SHA, result=result
    )
