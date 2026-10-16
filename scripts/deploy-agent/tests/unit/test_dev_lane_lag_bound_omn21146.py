# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21146: a merged runtime change reaches the dev lane within a declared bound.

THE INCIDENT, 2026-10-15
------------------------
omnimarket#3964 merged to dev at 17:07Z. The routing table sends omnimarket's
dev rebuilds to another instance, so the default instance's lane (the only one
that runs the gateway) took it only through the idle converge (OMN-19509). That converge may not start
within the converge ceiling plus margin before a C15 or C16 run, nor during
one. C15 runs every two hours at :41 for a declared 65 minutes, so each run
excludes [start-40m, start+65m]: 105 minutes of every 120. The merge landed in
the 17:41Z run's exclusion, the default instance's agent accepted its converge at 18:46:52Z
(the exclusion's end, to the minute), and the runtime was recreated at 19:10Z.
For two hours dod-verify refused merged_pr fleet-wide on an image built at
16:55Z, and nothing said the lane was behind.

THE RULE
--------
Each instance may declare ``flags.max_dev_lag_minutes`` in
``config/deploy_lane_routing.yaml``. A lane that has run behind a followed
repository's dev head for that long is OVERDUE: its converge no longer waits out
the idle period or the probe windows. Every other bound stays: a job in flight,
the FROZEN flag, the load gate, and one converge per head. And an overdue lane
is loud: the agent publishes a ``ModelDevLaneDrift`` naming the running image,
the dev head it lags and by how long, and serves the latest one per repository
at ``GET /dev-lane-drift``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from aiohttp.test_utils import TestClient, TestServer
from deploy_agent import agent as agent_mod
from deploy_agent.events import TOPIC_DEV_LANE_DRIFT, ModelDevLaneDrift
from deploy_agent.health import create_health_app
from deploy_agent.idle_converge import (
    IDLE_AFTER,
    IDLE_CONVERGE_REQUESTER,
    EnumIdleConvergeVerdict,
    ModelIdleConvergeInputs,
    decide,
    dev_lane_drift,
    load_probe_windows,
    next_behind_since,
    probe_blocking,
)
from deploy_agent.job_state import JobStore
from deploy_agent.routing import (
    ModelLaneFlags,
    RoutingTableError,
    lane_flags_for_instance,
    load_routing_table,
    parse_lane_flags,
)
from pydantic import ValidationError

# The agent harness of the OMN-19509 suite, reused so both suites build the
# agent the same way.
from .test_idle_converge_omn19509 import _agent

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
HEAD = "b" * 40
RUNNING = "a" * 40
BOUND = timedelta(minutes=30)

#: The incident: omnimarket#3964 merged at 17:07Z, inside the exclusion of the
#: 17:41Z C15 run.
MERGED = datetime(2026, 10, 15, 17, 7, tzinfo=UTC)
C15_START = datetime(2026, 10, 15, 17, 41, tzinfo=UTC)


def _inputs(**overrides: Any) -> ModelIdleConvergeInputs:
    now = overrides.get("now", MERGED + BOUND + timedelta(minutes=1))
    values: dict[str, Any] = {
        "now": now,
        "repository": "omnimarket",
        "routed_elsewhere": True,
        "job_active": False,
        "last_activity": now - IDLE_AFTER,
        "running_ref": RUNNING,
        "head_ref": HEAD,
        "probe_blocker": None,
        "windows_error": None,
        "attempted_heads": frozenset(),
        "behind_since": MERGED,
        "max_lag": BOUND,
    }
    values.update(overrides)
    return ModelIdleConvergeInputs(**values)


class TestTheIncident:
    def test_the_committed_probe_schedule_blocked_the_merge_for_99_minutes(
        self,
    ) -> None:
        """The cause, read off the committed schedule: a 17:07Z merge could
        not converge before 18:46Z under the probe windows alone."""
        windows = load_probe_windows(REPO_ROOT)
        assert probe_blocking(MERGED, windows) is not None
        first_clear = MERGED
        while probe_blocking(first_clear, windows) is not None:
            first_clear += timedelta(minutes=1)
        assert first_clear == datetime(2026, 10, 15, 18, 47, tzinfo=UTC)

    def test_an_overdue_lane_converges_inside_the_probe_window(self) -> None:
        windows = load_probe_windows(REPO_ROOT)
        now = MERGED + BOUND + timedelta(minutes=1)  # 17:38Z, under C15's lead
        blocker = probe_blocking(now, windows)
        assert blocker is not None

        decision = decide(_inputs(now=now, probe_blocker=blocker))

        assert decision.verdict is EnumIdleConvergeVerdict.CONVERGE
        assert decision.head_ref == HEAD
        assert "overdue" in decision.detail
        assert blocker in decision.detail

    def test_within_the_bound_the_probe_window_still_refuses(self) -> None:
        now = MERGED + BOUND - timedelta(minutes=1)
        decision = decide(_inputs(now=now, probe_blocker="C15 run at 17:41Z"))
        assert decision.verdict is EnumIdleConvergeVerdict.PROBE_WINDOW

    def test_with_no_declared_bound_nothing_changes(self) -> None:
        decision = decide(_inputs(max_lag=None, probe_blocker="C15 run at 17:41Z"))
        assert decision.verdict is EnumIdleConvergeVerdict.PROBE_WINDOW


class TestOverdueKeepsEveryOtherBound:
    def test_overdue_skips_the_idle_wait(self) -> None:
        now = MERGED + BOUND
        decision = decide(_inputs(now=now, last_activity=now))
        assert decision.verdict is EnumIdleConvergeVerdict.CONVERGE

    def test_overdue_skips_an_unreadable_schedule(self) -> None:
        decision = decide(_inputs(windows_error="probe schedule unreadable"))
        assert decision.verdict is EnumIdleConvergeVerdict.CONVERGE

    def test_overdue_never_overlaps_a_job(self) -> None:
        decision = decide(_inputs(job_active=True))
        assert decision.verdict is EnumIdleConvergeVerdict.JOB_ACTIVE

    def test_overdue_respects_the_freeze(self) -> None:
        decision = decide(_inputs(frozen_reason="FROZEN until 2026-10-19T00:00Z"))
        assert decision.verdict is EnumIdleConvergeVerdict.FROZEN

    def test_overdue_converges_on_a_head_once(self) -> None:
        decision = decide(_inputs(attempted_heads=frozenset({HEAD})))
        assert decision.verdict is EnumIdleConvergeVerdict.ALREADY_ATTEMPTED


class TestBehindSince:
    def test_starts_when_first_seen_behind(self) -> None:
        assert next_behind_since(None, RUNNING, HEAD, MERGED) == MERGED

    def test_keeps_the_earliest_while_the_head_moves(self) -> None:
        later = MERGED + timedelta(minutes=20)
        assert next_behind_since(MERGED, RUNNING, "c" * 40, later) == MERGED

    def test_clears_when_the_lane_runs_the_head(self) -> None:
        assert next_behind_since(MERGED, HEAD, HEAD, MERGED) is None

    def test_an_unread_ref_keeps_what_was_known(self) -> None:
        assert next_behind_since(MERGED, None, HEAD, MERGED) == MERGED
        assert next_behind_since(None, RUNNING, None, MERGED) is None


class TestDrift:
    def test_drift_names_the_image_the_head_and_the_lag(self) -> None:
        now = MERGED + timedelta(minutes=45)
        drift = dev_lane_drift(
            instance="lab-instance",
            repository="omnimarket",
            runtime_container="omninode-runtime",
            running_image="omnibase-infra-omninode-runtime@sha256:" + "e" * 64,
            running_ref=RUNNING,
            head_ref=HEAD,
            behind_since=MERGED,
            max_lag=BOUND,
            now=now,
        )
        assert drift is not None
        assert drift.running_image.startswith("omnibase-infra-omninode-runtime@")
        assert drift.running_ref == RUNNING
        assert drift.head_ref == HEAD
        assert drift.lag_seconds == 45 * 60
        assert drift.max_lag_seconds == 30 * 60
        assert drift.overdue is True
        payload = drift.model_dump(mode="json")
        assert payload["behind_since"] == "2026-10-15T17:07:00Z"

    def test_no_drift_on_head_or_without_a_bound(self) -> None:
        kwargs: dict[str, Any] = {
            "instance": "lab-instance",
            "repository": "omnimarket",
            "runtime_container": "omninode-runtime",
            "running_image": "img",
            "running_ref": RUNNING,
            "head_ref": HEAD,
            "behind_since": MERGED,
            "max_lag": BOUND,
            "now": MERGED,
        }
        assert dev_lane_drift(**{**kwargs, "behind_since": None}) is None
        assert dev_lane_drift(**{**kwargs, "max_lag": None}) is None

    def test_the_topic_is_an_event_of_the_deploy_producer(self) -> None:
        assert TOPIC_DEV_LANE_DRIFT == "onex.evt.deploy.dev-lane-drift.v1"


class TestTheBoundIsContractData:
    def test_the_committed_table_bounds_the_dev_lanes(self) -> None:
        table = load_routing_table()
        bounded = {table.default_instance} | {r.instance for r in table.routes}
        for instance in sorted(bounded):
            flags = lane_flags_for_instance(REPO_ROOT, instance)
            bound = flags.max_dev_lag()
            assert bound is not None, f"{instance} declares no max_dev_lag_minutes"
            assert timedelta(0) < bound <= timedelta(minutes=30)

    def test_the_bound_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            ModelLaneFlags(max_dev_lag_minutes=0)

    def test_a_bad_bound_refuses_the_table(self) -> None:
        text = (
            "instances:\n  lab-instance:\n    flags:\n      max_dev_lag_minutes: -5\n"
        )
        with pytest.raises(RoutingTableError):
            parse_lane_flags(text)


class TestInTheAgent:
    def test_an_overdue_agent_converges_and_publishes_drift(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        published: list[dict[str, Any]] = []
        monkeypatch.setattr(
            agent_mod,
            "publish_dev_lane_drift",
            lambda payload, config: published.append(payload) or True,
        )
        agent = _agent(tmp_path, monkeypatch, omnimarket_to_202=True)
        now = MERGED + BOUND + timedelta(minutes=1)
        agent._idle_converge_now = lambda: now
        agent._idle_probe_blocker = lambda at: "C15 run at 2026-10-15T17:41Z"
        agent._idle_read_running_image = lambda: "omnibase-infra-omninode-runtime"
        agent._lane_flags = ModelLaneFlags(max_dev_lag_minutes=30)
        agent._dev_lag_behind_since["omnimarket"] = MERGED

        agent._maybe_idle_converge()

        jobs = [
            JobStore(tmp_path / "jobs").load(UUID(p.stem))
            for p in (tmp_path / "jobs").glob("*.json")
        ]
        assert len(jobs) == 1
        assert jobs[0] is not None
        assert jobs[0].command["requested_by"] == IDLE_CONVERGE_REQUESTER
        assert len(published) == 1
        drift = ModelDevLaneDrift.model_validate(published[0])
        assert drift.overdue is True
        assert drift.repository == "omnimarket"
        assert drift.head_ref == HEAD
        assert drift.lag_seconds == (BOUND + timedelta(minutes=1)).total_seconds()
        assert agent._latest_dev_lane_drift()["omnimarket"]["overdue"] is True

    def test_drift_is_not_republished_every_check(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        published: list[dict[str, Any]] = []
        monkeypatch.setattr(
            agent_mod,
            "publish_dev_lane_drift",
            lambda payload, config: published.append(payload) or True,
        )
        agent = _agent(tmp_path, monkeypatch, omnimarket_to_202=True)
        now = MERGED + BOUND + timedelta(minutes=1)
        agent._idle_converge_now = lambda: now
        agent._idle_read_running_image = lambda: "img"
        agent._lane_flags = ModelLaneFlags(max_dev_lag_minutes=30)
        agent._dev_lag_behind_since["omnimarket"] = MERGED
        agent._idle_converge_attempted.add(HEAD)  # converge already tried

        agent._maybe_idle_converge()
        agent._idle_converge_last_check = None
        agent._maybe_idle_converge()
        assert len(published) == 1

        later = now + agent_mod.DEV_LANE_DRIFT_REPUBLISH_AFTER
        agent._idle_converge_now = lambda: later
        agent._idle_converge_last_check = None
        agent._maybe_idle_converge()
        assert len(published) == 2

    def test_a_cleared_lane_publishes_once_that_it_caught_up(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        published: list[dict[str, Any]] = []
        monkeypatch.setattr(
            agent_mod,
            "publish_dev_lane_drift",
            lambda payload, config: published.append(payload) or True,
        )
        agent = _agent(tmp_path, monkeypatch, omnimarket_to_202=True)
        now = MERGED + BOUND + timedelta(minutes=1)
        agent._idle_converge_now = lambda: now
        agent._idle_read_running_image = lambda: "img"
        agent._lane_flags = ModelLaneFlags(max_dev_lag_minutes=30)
        agent._dev_lag_behind_since["omnimarket"] = MERGED
        agent._idle_converge_attempted.add(HEAD)
        agent._maybe_idle_converge()

        agent._idle_read_running_ref = lambda repo: HEAD
        for _ in range(2):
            agent._idle_converge_last_check = None
            agent._maybe_idle_converge()

        assert len(published) == 2
        cleared = ModelDevLaneDrift.model_validate(published[1])
        assert cleared.overdue is False
        assert cleared.lag_seconds == 0
        assert cleared.running_ref == HEAD
        assert "omnimarket" not in agent._dev_lag_behind_since


async def test_the_health_surface_serves_the_latest_drift(tmp_path: Path) -> None:
    drift = {"omnimarket": {"overdue": True, "lag_seconds": 1860}}
    app = create_health_app(
        job_store=JobStore(tmp_path / "jobs"),
        get_agent_state=lambda: "idle",
        get_dev_lane_drift=lambda: drift,
    )
    async with TestClient(TestServer(app)) as client:
        response = await client.get("/dev-lane-drift")
        assert response.status == 200
        assert await response.json() == {"drift": drift}
