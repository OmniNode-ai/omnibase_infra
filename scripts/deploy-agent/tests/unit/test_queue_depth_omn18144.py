# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The agent can say what is ahead of a command, and how fast it drains (OMN-18144).

Every case here is a shape the 2026-09-18 stall went through: a command third
in line behind one in flight and two unconsumed, an agent whose lag cannot be
sampled, and an agent that has completed nothing to take a mean over. The
property under test throughout is that NONE of them is reported as an empty
queue.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from aiohttp import web
from deploy_agent.health import create_health_app
from deploy_agent.job_state import JobState, JobStore
from deploy_agent.queue_depth import (
    NON_TERMINAL_STATUSES,
    LagSampler,
    ModelControlTopicLag,
    ModelQueueSnapshot,
    compute_queue_snapshot,
    mean_service_time,
)

pytestmark = pytest.mark.unit

_T0 = datetime(2026, 9, 18, 17, 0, tzinfo=UTC)


def _store(tmp_path) -> JobStore:  # type: ignore[no-untyped-def]
    return JobStore(state_dir=tmp_path / "jobs")


def _write(
    store: JobStore,
    *,
    correlation_id: UUID | None = None,
    status: str = "success",
    accepted_at: datetime = _T0,
    completed_at: datetime | None = None,
) -> JobState:
    job = JobState(
        correlation_id=correlation_id or uuid4(),
        command={},
        status=status,  # type: ignore[arg-type]
        accepted_at=accepted_at,
        completed_at=completed_at,
    )
    store.state_dir.mkdir(parents=True, exist_ok=True)
    (store.state_dir / f"{job.correlation_id}.json").write_text(job.model_dump_json())
    return job


class TestControlTopicLag:
    def test_a_lag_carries_exactly_one_of_a_value_and_a_reason(self) -> None:
        with pytest.raises(ValueError, match="EXACTLY one"):
            ModelControlTopicLag(value=2, reason="also a reason", basis="committed")
        with pytest.raises(ValueError, match="EXACTLY one"):
            ModelControlTopicLag(value=None, reason="")

    def test_a_valued_lag_must_name_its_basis(self) -> None:
        """An under-reporting basis that does not say so is worse than no number."""
        with pytest.raises(ValueError, match="basis"):
            ModelControlTopicLag(value=3)

    def test_unknown_is_not_zero(self) -> None:
        lag = ModelControlTopicLag.unknown("no assignment yet")
        assert lag.value is None
        assert lag.reason


class TestCommandsAhead:
    def test_the_two_halves_are_summed(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The 2026-09-18 shape: one in flight, two unconsumed, three ahead.

        Either half alone undercounts, and the undercount is the defect: a
        store-only read says 1 and a lag-only read says 2, while the merge
        actually waited behind 3.
        """
        store = _store(tmp_path)
        _write(store, status="in_progress", completed_at=None)
        snapshot = compute_queue_snapshot(
            store,
            ModelControlTopicLag(value=2, basis="committed"),
            now=_T0,
        )
        assert snapshot.store_depth == 1
        assert snapshot.control_topic_lag.value == 2
        assert snapshot.commands_ahead == 3

    def test_an_unreadable_lag_makes_the_total_unknown_not_partial(
        self, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        store = _store(tmp_path)
        _write(store, status="in_progress")
        snapshot = compute_queue_snapshot(
            store, ModelControlTopicLag.unknown("no highwater"), now=_T0
        )
        assert snapshot.store_depth == 1
        assert snapshot.commands_ahead is None, (
            "a store-only count would report 1 command ahead while an unread "
            "lag hid two more; unknown is the only honest total"
        )

    def test_terminal_jobs_are_not_ahead_of_anything(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        store = _store(tmp_path)
        _write(store, status="success", completed_at=_T0 + timedelta(minutes=30))
        _write(store, status="failed", completed_at=_T0 + timedelta(minutes=20))
        snapshot = compute_queue_snapshot(
            store, ModelControlTopicLag(value=0, basis="committed"), now=_T0
        )
        assert snapshot.commands_ahead == 0

    def test_non_terminal_statuses_mirror_the_job_state_literal(self) -> None:
        """A new status must be classified deliberately, not inherited silently."""
        declared = set(JobState.model_fields["status"].annotation.__args__)  # type: ignore[union-attr]
        assert declared > NON_TERMINAL_STATUSES
        assert declared - NON_TERMINAL_STATUSES == {"success", "failed"}


class TestMeanServiceTime:
    def test_the_mean_is_taken_over_completed_jobs_of_either_outcome(
        self, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        """A job that failed after 30 minutes occupied the agent for 30 minutes."""
        store = _store(tmp_path)
        _write(
            store,
            status="success",
            accepted_at=_T0,
            completed_at=_T0 + timedelta(minutes=20),
        )
        _write(
            store,
            status="failed",
            accepted_at=_T0,
            completed_at=_T0 + timedelta(minutes=40),
        )
        snapshot = compute_queue_snapshot(
            store, ModelControlTopicLag(value=0, basis="committed"), now=_T0
        )
        assert snapshot.mean_service_time_seconds == pytest.approx(1800.0)
        assert snapshot.service_sample_size == 2

    def test_no_completed_job_means_no_mean_rather_than_zero(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        store = _store(tmp_path)
        _write(store, status="in_progress")
        snapshot = compute_queue_snapshot(
            store, ModelControlTopicLag(value=1, basis="committed"), now=_T0
        )
        assert snapshot.mean_service_time_seconds is None
        assert snapshot.service_sample_size == 0

    def test_a_backwards_clock_does_not_make_the_agent_look_fast(
        self, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        store = _store(tmp_path)
        _write(
            store,
            status="success",
            accepted_at=_T0,
            completed_at=_T0 - timedelta(minutes=5),
        )
        mean, sample = mean_service_time(
            [
                JobState.model_validate_json(path.read_text())
                for path in store.state_dir.glob("*.json")
            ]
        )
        assert mean is None and sample == 0

    def test_the_sample_window_is_the_most_recent_completions(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        store = _store(tmp_path)
        # An old, slow job and two recent fast ones. A window of 2 must not see
        # the old one, or a host that has since got faster still reads slow.
        _write(
            store,
            accepted_at=_T0 - timedelta(hours=10),
            completed_at=_T0 - timedelta(hours=9),
        )
        for offset in (2, 1):
            _write(
                store,
                accepted_at=_T0 - timedelta(hours=offset),
                completed_at=_T0 - timedelta(hours=offset) + timedelta(minutes=10),
            )
        jobs = [
            JobState.model_validate_json(path.read_text())
            for path in store.state_dir.glob("*.json")
        ]
        mean, sample = mean_service_time(jobs, sample_size=2)
        assert sample == 2
        assert mean == pytest.approx(600.0)

    def test_a_snapshot_refuses_a_mean_without_a_sample(self) -> None:
        with pytest.raises(ValueError, match="stand or fall together"):
            ModelQueueSnapshot(
                observed_at=_T0,
                store_depth=0,
                control_topic_lag=ModelControlTopicLag(value=0, basis="committed"),
                mean_service_time_seconds=900.0,
                service_sample_size=0,
            )


class TestLagSampler:
    def test_an_unsampled_sampler_reports_unknown_not_empty(self) -> None:
        assert LagSampler().latest().value is None

    def test_the_latest_sample_wins(self) -> None:
        sampler = LagSampler()
        sampler.record(ModelControlTopicLag(value=5, basis="committed"))
        sampler.record(ModelControlTopicLag(value=2, basis="committed"))
        assert sampler.latest().value == 2


class TestQueueEndpoint:
    """The route the CI guard reads. Served over the real aiohttp app."""

    async def test_it_serves_the_queue_facts(self, aiohttp_client, tmp_path) -> None:  # type: ignore[no-untyped-def]
        store = _store(tmp_path)
        active = _write(store, status="in_progress")
        _write(
            store,
            status="success",
            accepted_at=_T0,
            completed_at=_T0 + timedelta(minutes=32),
        )
        sampler = LagSampler()
        sampler.record(ModelControlTopicLag(value=2, basis="committed"))
        app = create_health_app(
            job_store=store,
            get_agent_state=lambda: "idle",
            get_control_topic_lag=sampler.latest,
        )
        client = await aiohttp_client(app)
        response = await client.get("/queue")
        assert response.status == 200
        payload = await response.json()
        assert payload["commands_ahead"] == 3
        assert payload["store_depth"] == 1
        assert payload["control_topic_lag"] == 2
        assert payload["in_flight_correlation_id"] == str(active.correlation_id)
        assert payload["mean_service_time_seconds"] == pytest.approx(1920.0)
        assert payload["service_sample_size"] == 1

    async def test_an_agent_with_no_sampler_reports_unknown_with_a_reason(
        self, aiohttp_client, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        """200 with an honest body, never an error status.

        A 5xx here would be indistinguishable to the CI reader from an agent
        too old to serve the route, and that distinction decides whether it
        falls back or retries.
        """
        app = create_health_app(
            job_store=_store(tmp_path), get_agent_state=lambda: "idle"
        )
        client = await aiohttp_client(app)
        response = await client.get("/queue")
        assert response.status == 200
        payload = await response.json()
        assert payload["commands_ahead"] is None
        assert "not zero" in payload["control_topic_lag_reason"]

    async def test_the_endpoint_exposes_no_command_payload(
        self, aiohttp_client, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        """Counts, ids and durations only. A command payload can carry a digest,
        a ref and a service list; none of it is the queue's business."""
        store = _store(tmp_path)
        job = JobState(
            correlation_id=uuid4(),
            command={"secret_looking_field": "must-not-appear"},
            status="in_progress",
            accepted_at=_T0,
        )
        store.state_dir.mkdir(parents=True, exist_ok=True)
        (store.state_dir / f"{job.correlation_id}.json").write_text(
            job.model_dump_json()
        )
        app = create_health_app(
            job_store=store,
            get_agent_state=lambda: "busy",
            get_control_topic_lag=lambda: ModelControlTopicLag(
                value=0, basis="committed"
            ),
        )
        client = await aiohttp_client(app)
        body = await (await client.get("/queue")).text()
        assert "must-not-appear" not in body

    async def test_a_corrupt_record_does_not_take_out_the_surface(
        self, aiohttp_client, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        store = _store(tmp_path)
        _write(store, status="in_progress")
        (store.state_dir / "garbage.json").write_text("{not json")
        app = create_health_app(
            job_store=store,
            get_agent_state=lambda: "busy",
            get_control_topic_lag=lambda: ModelControlTopicLag(
                value=1, basis="committed"
            ),
        )
        client = await aiohttp_client(app)
        response = await client.get("/queue")
        assert response.status == 200
        assert (await response.json())["commands_ahead"] == 2


class TestHealthSurfaceUnchanged:
    """The additive claim, checked rather than asserted in a comment."""

    async def test_health_is_untouched_by_the_queue_route(
        self, aiohttp_client, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        app = create_health_app(
            job_store=_store(tmp_path), get_agent_state=lambda: "idle"
        )
        client = await aiohttp_client(app)
        response = await client.get("/health")
        assert response.status == 200
        payload = await response.json()
        assert set(payload) == {
            "state",
            "version",
            "loaded_code_sha",
            "uptime_seconds",
            "active_job",
            "last_result",
            "pending_publish_count",
            "accept_backlog",
        }

    async def test_the_queue_route_is_get_only(self, aiohttp_client, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """It observes. Nothing about it can cancel, reorder or curtail a job."""
        app = create_health_app(
            job_store=_store(tmp_path), get_agent_state=lambda: "idle"
        )
        client = await aiohttp_client(app)
        for verb in ("post", "delete", "put"):
            response = await getattr(client, verb)("/queue")
            assert response.status == web.HTTPMethodNotAllowed.status_code
