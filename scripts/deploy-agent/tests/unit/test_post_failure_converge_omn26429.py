# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26429: after a failed or recovered dev roll the agent rebuilds the newer dev head once.

Job ca1533c4 (``agent/idle-restore``, dev lane) built omnibase_infra 0f7f2b73, failed at
the runtime phase, and the boot guard rolled the lane back to the two-day-old image. The
fixes for that failure merged while it ran. Nothing started a rebuild afterwards: idle
converge refused omnibase_infra (``not_routed_elsewhere``) and omnimarket
(``probe_window``), and idle restore saw a running runtime. The lane sat on the old
image until a person noticed.

``deploy_agent.post_failure_converge`` is the narrower rule keyed on a terminal failed
dev job. These tests pin its acceptance criteria:

* AC1: one rebuild per head after a failed dev job, none without one, none when the head
  equals the failed job's source, and the need survives a restart of the agent;
* AC2: the C15 and C16 windows are read through idle converge's own function and a
  window delays the rebuild without dropping the need;
* AC3: the recorded ca1533c4 record plus a newer head yields one converge request.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent.agent import DeployAgent
from deploy_agent.events import EnumRuntimeLane
from deploy_agent.executor import load_dev_boot_guard
from deploy_agent.idle_converge import (
    EnumRuntimePresence,
    converge_command,
    load_probe_windows,
    probe_blocking,
)
from deploy_agent.job_state import JobState, JobStore
from deploy_agent.routing import ModelLaneFlags, ModelLaneFreeze

pytestmark = pytest.mark.unit

#: The rule's module, loaded the way the agent loads it.
reading = load_dev_boot_guard(
    "nodes.node_deploy_job_reading_effect.runtime_deploy_job_reading"
)
rule = sys.modules[
    type(reading.post_failure_converge(object(), lambda: None)).__module__
]
POST_FAILURE_CONVERGE_REQUESTER = rule.POST_FAILURE_CONVERGE_REQUESTER
EnumPostFailureVerdict = rule.EnumPostFailureVerdict
ModelFailedDevJob = rule.ModelFailedDevJob
ModelPostFailureInputs = rule.ModelPostFailureInputs
decide = rule.decide
failed_dev_job = rule.failed_dev_job

REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURE = Path(__file__).parent / "fixtures" / "recorded_job_ca1533c4_omn26429.json"

#: What job ca1533c4 built (its record's ``commit=`` token) and dev's head at the read.
FAILED_SOURCE = "0f7f2b730399c250db0222626463381ac99078c8"
NEWER_HEAD = "3f614d4f477576a9575ff716016995eb630151e8"
NEWEST_HEAD = "c" * 40

#: Outside every guarded window (C15 01:41-02:46Z, C16 02:49-04:24Z with the lead).
QUIET = datetime(2026, 11, 22, 4, 50, tzinfo=UTC)
#: Inside C15's run that began 01:41Z, the instant of the incident's refusal.
IN_C15 = datetime(2026, 11, 22, 2, 25, tzinfo=UTC)
#: After C15's run (ends 02:46Z) and before C16's exclusion begins (02:49Z).
BETWEEN_C15_AND_C16 = datetime(2026, 11, 22, 2, 47, tzinfo=UTC)
#: Inside C16's exclusion (C16 starts 03:29Z, lead 40 minutes).
IN_C16 = datetime(2026, 11, 22, 2, 56, tzinfo=UTC)


#: The lanes that are not dev: their failures are never this rule's need.
OTHER_LANES = [
    lane.value for lane in EnumRuntimeLane if lane is not EnumRuntimeLane.DEV
]


def _recorded_job() -> JobState:
    return JobState.model_validate(json.loads(FIXTURE.read_text())["job"])


def _record() -> Any:
    """The recorded job as the rule reads it."""
    return rule.ModelJobRecord.model_validate(json.loads(FIXTURE.read_text())["job"])


def _failed(**overrides: Any) -> ModelFailedDevJob:
    values: dict[str, Any] = {
        "correlation_id": uuid4(),
        "requested_by": "agent/idle-restore",
        "sources": {"omnibase_infra": FAILED_SOURCE},
    }
    values.update(overrides)
    return ModelFailedDevJob(**values)


def _inputs(**overrides: Any) -> ModelPostFailureInputs:
    values: dict[str, Any] = {
        "now": QUIET,
        "job_active": False,
        "failed_job": _failed(),
        "heads": {"omnibase_infra": NEWER_HEAD},
        "probe_blocker": None,
        "windows_error": None,
        "attempted": frozenset(),
    }
    values.update(overrides)
    return ModelPostFailureInputs(**values)


# --------------------------------------------------------------------------- #
# The recorded record: what the rule reads out of ca1533c4                      #
# --------------------------------------------------------------------------- #
class TestRecordedJob:
    def test_the_fixture_is_the_failed_dev_idle_restore(self) -> None:
        job = _recorded_job()
        assert str(job.correlation_id).startswith("ca1533c4")
        assert job.status == "failed"
        assert job.command["requested_by"] == "agent/idle-restore"
        assert job.command["runtime_lane"] == "dev"
        assert job.lineage is None

    def test_the_source_is_the_commit_the_recovery_names(self) -> None:
        failed = failed_dev_job(_record())
        assert failed is not None
        assert failed.sources == {"omnibase_infra": FAILED_SOURCE}

    def test_a_passed_job_is_not_a_failed_job(self) -> None:
        passed = _record().model_copy(update={"status": "success", "errors": ()})
        assert failed_dev_job(passed) is None
        assert failed_dev_job(None) is None

    @pytest.mark.parametrize("lane", OTHER_LANES)
    def test_a_failed_prod_or_stability_job_is_not_a_dev_failure(
        self, lane: str
    ) -> None:
        record = _record()
        record = record.model_copy(
            update={"command": record.command.model_copy(update={"runtime_lane": lane})}
        )
        assert failed_dev_job(record) is None

    def test_a_failure_that_names_no_commit_has_no_known_source(self) -> None:
        record = _record().model_copy(update={"errors": ("preflight refused",)})
        failed = failed_dev_job(record)
        assert failed is not None
        assert failed.sources == {}

    def test_a_lineage_build_ref_is_the_source_before_the_error_text(self) -> None:
        record = _record().model_copy(
            update={"lineage": rule.ModelJobLineage(build_ref=NEWER_HEAD)}
        )
        failed = failed_dev_job(record)
        assert failed is not None
        assert failed.sources == {"omnibase_infra": NEWER_HEAD}

    def test_an_unreadable_record_is_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "broken.json").write_text("{not json")
        recorded = _recorded_job()
        (tmp_path / f"{recorded.correlation_id}.json").write_text(
            recorded.model_dump_json()
        )
        newest = rule.latest_terminal_dev_job(tmp_path)
        assert newest is not None
        assert newest.correlation_id == recorded.correlation_id


# --------------------------------------------------------------------------- #
# The decision                                                                  #
# --------------------------------------------------------------------------- #
class TestDecision:
    def test_a_newer_head_after_a_failed_job_converges_on_that_head(self) -> None:
        decision = decide(_inputs())
        assert decision.verdict is EnumPostFailureVerdict.CONVERGE
        assert decision.heads == {"omnibase_infra": NEWER_HEAD}

    def test_the_head_equal_to_the_failed_source_converges_on_nothing(self) -> None:
        decision = decide(_inputs(heads={"omnibase_infra": FAILED_SOURCE}))
        assert decision.verdict is EnumPostFailureVerdict.UP_TO_DATE

    def test_a_short_recorded_source_matches_the_full_head(self) -> None:
        failed = _failed(sources={"omnimarket": "5b683a9cbadf"})
        decision = decide(
            _inputs(failed_job=failed, heads={"omnimarket": "5b683a9cbadf" + "0" * 28})
        )
        assert decision.verdict is EnumPostFailureVerdict.UP_TO_DATE

    def test_no_failed_job_means_no_rebuild(self) -> None:
        decision = decide(_inputs(failed_job=None))
        assert decision.verdict is EnumPostFailureVerdict.NO_FAILED_JOB

    def test_a_failure_with_an_unknown_source_never_rebuilds(self) -> None:
        decision = decide(_inputs(failed_job=_failed(sources={})))
        assert decision.verdict is EnumPostFailureVerdict.SOURCE_UNKNOWN

    def test_an_unreadable_head_refuses(self) -> None:
        decision = decide(_inputs(heads={"omnibase_infra": None}))
        assert decision.verdict is EnumPostFailureVerdict.HEAD_UNREADABLE

    def test_a_job_in_flight_refuses(self) -> None:
        decision = decide(_inputs(job_active=True))
        assert decision.verdict is EnumPostFailureVerdict.JOB_ACTIVE

    def test_a_frozen_lane_refuses(self) -> None:
        decision = decide(_inputs(frozen_reason="FROZEN until 2026-12-01T00:00Z: x"))
        assert decision.verdict is EnumPostFailureVerdict.FROZEN

    def test_each_head_is_attempted_once(self) -> None:
        first = decide(_inputs())
        assert first.verdict is EnumPostFailureVerdict.CONVERGE
        again = decide(_inputs(attempted=frozenset(first.attempt_keys)))
        assert again.verdict is EnumPostFailureVerdict.ALREADY_ATTEMPTED

    def test_a_head_that_moved_is_a_new_head(self) -> None:
        first = decide(_inputs())
        moved = decide(
            _inputs(
                attempted=frozenset(first.attempt_keys),
                heads={"omnibase_infra": NEWEST_HEAD},
            )
        )
        assert moved.verdict is EnumPostFailureVerdict.CONVERGE
        assert moved.heads == {"omnibase_infra": NEWEST_HEAD}

    def test_the_command_is_an_ordinary_dev_workspace_rebuild(self) -> None:
        cmd = reading.post_failure_converge(object(), converge_command).command()
        assert cmd.requested_by == POST_FAILURE_CONVERGE_REQUESTER
        assert cmd.runtime_lane is EnumRuntimeLane.DEV
        assert cmd.git_ref == "origin/dev"


# --------------------------------------------------------------------------- #
# AC2: the probe windows, read through idle converge's own function            #
# --------------------------------------------------------------------------- #
def _blocker(now: datetime) -> str | None:
    return probe_blocking(now, load_probe_windows(REPO_ROOT))


class TestProbeWindows:
    def test_c15_blocks_at_the_incident_instant_and_the_need_is_kept(self) -> None:
        blocker = _blocker(IN_C15)
        assert blocker is not None and blocker.startswith("C15 run at")
        decision = decide(_inputs(now=IN_C15, probe_blocker=blocker))
        assert decision.verdict is EnumPostFailureVerdict.PROBE_WINDOW
        assert decision.attempt_keys == ()

    def test_c16_blocks_where_c15_does_not(self) -> None:
        assert _blocker(BETWEEN_C15_AND_C16) is None
        blocker = _blocker(IN_C16)
        assert blocker is not None and blocker.startswith("C16 run at")
        decision = decide(_inputs(now=IN_C16, probe_blocker=blocker))
        assert decision.verdict is EnumPostFailureVerdict.PROBE_WINDOW

    def test_the_first_check_after_the_window_converges(self) -> None:
        blocked = decide(_inputs(now=IN_C15, probe_blocker=_blocker(IN_C15)))
        assert blocked.verdict is EnumPostFailureVerdict.PROBE_WINDOW
        clear = decide(
            _inputs(
                now=BETWEEN_C15_AND_C16,
                probe_blocker=_blocker(BETWEEN_C15_AND_C16),
            )
        )
        assert clear.verdict is EnumPostFailureVerdict.CONVERGE

    def test_an_unreadable_schedule_is_a_refusal_never_a_clear_window(self) -> None:
        decision = decide(_inputs(windows_error="probe schedule unreadable: OSError"))
        assert decision.verdict is EnumPostFailureVerdict.WINDOWS_UNREADABLE

    def test_no_decision_converges_inside_a_reserved_window_over_a_day(self) -> None:
        """Falsifier of AC2: sweep every minute of two days through the real schedule.

        A converge verdict inside C15's or C16's exclusion is a job accepted inside a
        reserved window. The sweep also holds both outcomes, so it cannot pass vacuously.
        """
        start = datetime(2026, 11, 22, 0, 0, tzinfo=UTC)
        converged = blocked = 0
        for minute in range(2 * 24 * 60):
            now = start + timedelta(minutes=minute)
            blocker = _blocker(now)
            verdict = decide(_inputs(now=now, probe_blocker=blocker)).verdict
            if blocker is None:
                assert verdict is EnumPostFailureVerdict.CONVERGE, now
                converged += 1
            else:
                assert verdict is EnumPostFailureVerdict.PROBE_WINDOW, now
                blocked += 1
        assert converged > 0 and blocked > 0
        windows = {w.id for w in load_probe_windows(REPO_ROOT)}
        assert {"C15", "C16"} <= windows


# --------------------------------------------------------------------------- #
# The rule in the agent                                                         #
# --------------------------------------------------------------------------- #
class _FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.container_residue: list[object] = []
        self.sibling_source_refs: dict[str, str] = {}
        self.recreate_supervision: list[object] = []
        self.verify_recreate: list[object] = []
        self.oneshot_reruns: list[object] = []
        self.cloud_migration_failure: str = ""
        self.deps_convergence: list[object] = []
        self.compose_invocations: list[object] = []
        self.health_checks: list[object] = []

    def __getattr__(self, name: str) -> Any:
        def record(*args: object, **kwargs: object) -> Any:
            self.calls.append(name)
            if name == "git_pull":
                return "d" * 40
            if name in ("rebuild_scope", "verify"):
                return []
            return None

        return record


def _agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    now: datetime = QUIET,
    head: str = NEWER_HEAD,
    seed_recorded_job: bool = True,
) -> DeployAgent:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "agent-state")
    monkeypatch.setattr(agent_mod, "publish_result", lambda payload, config: False)
    monkeypatch.setattr(agent_mod, "LAB_OVERLAY_ENABLED", False)
    jobs = tmp_path / "jobs"
    jobs.mkdir(exist_ok=True)
    if seed_recorded_job:
        # The record as the agent wrote it, on disk where a restart finds it.
        recorded = _recorded_job()
        (jobs / f"{recorded.correlation_id}.json").write_text(
            recorded.model_dump_json()
        )
    agent = DeployAgent(skip_self_update=True)
    agent.job_store = JobStore(jobs)
    agent.executor = cast("Any", _FakeExecutor())
    agent._idle_converge_started_at = now - timedelta(hours=1)
    agent._idle_converge_now = lambda: now
    agent._idle_read_head_ref = lambda repo: head
    agent._idle_read_runtime_presence = lambda: (
        EnumRuntimePresence.RUNNING,
        "omninode-runtime is running",
    )
    agent._idle_read_dev_tags = lambda repo: None
    monkeypatch.setattr(
        agent_mod, "publish_dev_tag_heartbeat", lambda payload, config: True
    )
    agent._idle_probe_blocker = _blocker
    agent._lane_flags = ModelLaneFlags()
    return agent


def _post_failure_jobs(tmp_path: Path) -> list[JobState]:
    store = JobStore(tmp_path / "jobs")
    jobs = [
        store.load(UUID(p.stem)) for p in sorted((tmp_path / "jobs").glob("*.json"))
    ]
    return [
        job
        for job in jobs
        if job is not None
        and job.command["requested_by"] == POST_FAILURE_CONVERGE_REQUESTER
    ]


def _tick(agent: DeployAgent) -> None:
    """One idle-branch check, past the interval throttle."""
    agent._post_failure.last_check = None
    agent._post_failure.check()


class TestInTheAgent:
    def test_the_recorded_job_plus_a_newer_head_asks_for_exactly_one_rebuild(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC3: this fails when the new step is removed from the agent."""
        agent = _agent(tmp_path, monkeypatch)

        _tick(agent)

        jobs = _post_failure_jobs(tmp_path)
        assert len(jobs) == 1
        assert jobs[0].command["runtime_lane"] == "dev"
        assert jobs[0].command["requested_by"] == "agent/post-failure-converge"
        assert jobs[0].status == "success"
        assert "rebuild_scope" in cast("_FakeExecutor", agent.executor).calls

    def test_the_same_head_on_three_checks_and_after_a_restart_is_one_rebuild(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _agent(tmp_path, monkeypatch)
        for _ in range(3):
            _tick(agent)
        assert len(_post_failure_jobs(tmp_path)) == 1

        # The agent restarts: a new process, an empty in-memory set, the same records.
        restarted = _agent(tmp_path, monkeypatch, seed_recorded_job=False)
        assert restarted._post_failure.attempted == set()
        for _ in range(3):
            _tick(restarted)
        assert len(_post_failure_jobs(tmp_path)) == 1

    def test_a_restart_before_the_rebuild_still_finds_the_need(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The failed record is the durable need: a fresh agent starts the one rebuild."""
        _agent(tmp_path, monkeypatch)  # the agent that restarted before it acted
        restarted = _agent(tmp_path, monkeypatch, seed_recorded_job=False)
        _tick(restarted)
        assert len(_post_failure_jobs(tmp_path)) == 1

    def test_a_head_equal_to_the_failed_source_starts_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _agent(tmp_path, monkeypatch, head=FAILED_SOURCE)
        _tick(agent)
        assert _post_failure_jobs(tmp_path) == []

    def test_no_failed_job_starts_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _agent(tmp_path, monkeypatch, seed_recorded_job=False)
        _tick(agent)
        assert _post_failure_jobs(tmp_path) == []

    def test_a_passed_newest_job_starts_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _agent(tmp_path, monkeypatch)
        recorded = _recorded_job()
        completed = recorded.completed_at
        assert completed is not None
        passed = recorded.model_copy(
            update={
                "correlation_id": uuid4(),
                "status": "success",
                "errors": [],
                "completed_at": completed + timedelta(minutes=30),
            }
        )
        (tmp_path / "jobs" / f"{passed.correlation_id}.json").write_text(
            passed.model_dump_json()
        )
        _tick(agent)
        assert _post_failure_jobs(tmp_path) == []

    @pytest.mark.parametrize("lane", OTHER_LANES)
    def test_a_failed_prod_or_stability_job_starts_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lane: str
    ) -> None:
        agent = _agent(tmp_path, monkeypatch, seed_recorded_job=False)
        recorded = _recorded_job()
        other = recorded.model_copy(
            update={"command": {**recorded.command, "runtime_lane": lane}}
        )
        (tmp_path / "jobs" / f"{other.correlation_id}.json").write_text(
            other.model_dump_json()
        )
        _tick(agent)
        assert _post_failure_jobs(tmp_path) == []

    def test_no_rebuild_while_a_job_is_active(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _agent(tmp_path, monkeypatch)
        agent.job_store.accept(
            uuid4(), {"requested_by": "gha/x", "runtime_lane": "dev"}
        )
        _tick(agent)
        assert _post_failure_jobs(tmp_path) == []

    def test_no_rebuild_on_a_frozen_lane(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _agent(tmp_path, monkeypatch)
        agent._lane_flags = ModelLaneFlags(
            freeze=ModelLaneFreeze(until=QUIET + timedelta(days=1), reason="held")
        )
        _tick(agent)
        assert _post_failure_jobs(tmp_path) == []

    def test_a_closed_load_gate_defers_and_the_need_is_kept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _agent(tmp_path, monkeypatch)
        monkeypatch.setattr(agent, "_idle_converge_gate_open", lambda who: False)
        _tick(agent)
        assert _post_failure_jobs(tmp_path) == []
        assert agent._post_failure.attempted == set()

        monkeypatch.setattr(agent, "_idle_converge_gate_open", lambda who: True)
        _tick(agent)
        assert len(_post_failure_jobs(tmp_path)) == 1

    def test_the_gate_is_asked_for_the_new_requester(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _agent(tmp_path, monkeypatch)
        asked: list[str] = []
        monkeypatch.setattr(
            agent, "_idle_converge_gate_open", lambda who: asked.append(who) or True
        )
        _tick(agent)
        assert asked == [POST_FAILURE_CONVERGE_REQUESTER]

    def test_a_probe_window_delays_and_the_first_clear_check_rebuilds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _agent(tmp_path, monkeypatch, now=IN_C15)
        _tick(agent)
        assert _post_failure_jobs(tmp_path) == []

        agent._idle_converge_now = lambda: IN_C16
        _tick(agent)
        assert _post_failure_jobs(tmp_path) == []

        agent._idle_converge_now = lambda: BETWEEN_C15_AND_C16
        _tick(agent)
        assert len(_post_failure_jobs(tmp_path)) == 1

    def test_an_unreadable_schedule_starts_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _agent(tmp_path, monkeypatch)

        def unreadable(now: datetime) -> str | None:
            raise OSError("no schedule")

        agent._idle_probe_blocker = unreadable
        _tick(agent)
        assert _post_failure_jobs(tmp_path) == []

    def test_a_head_that_moves_while_waiting_gets_one_rebuild_for_the_newest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _agent(tmp_path, monkeypatch, now=IN_C15)
        _tick(agent)
        agent._idle_read_head_ref = lambda repo: "e" * 40
        _tick(agent)
        agent._idle_read_head_ref = lambda repo: NEWEST_HEAD
        agent._idle_converge_now = lambda: QUIET
        _tick(agent)
        assert len(_post_failure_jobs(tmp_path)) == 1
        assert agent._post_failure.attempted == {f"omnibase_infra@{NEWEST_HEAD}"}

    def test_the_check_is_throttled_to_the_idle_interval(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _agent(tmp_path, monkeypatch)
        agent._idle_read_head_ref = lambda repo: None
        agent._post_failure.check()
        reads: list[str] = []
        agent._idle_read_head_ref = lambda repo: reads.append(repo) or NEWER_HEAD
        agent._post_failure.check()
        assert reads == []

    def test_the_rule_runs_only_on_the_poll_loops_idle_branch(self) -> None:
        import inspect

        source = inspect.getsource(DeployAgent.run)
        idle_branch = source[source.index("else:") :]
        assert "_post_failure.check" in idle_branch
        assert source.count("_post_failure.check") == 1
