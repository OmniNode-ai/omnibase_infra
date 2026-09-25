# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19509 (task B7): the .201 instance converges on the omnimarket dev head when idle.

Once omnimarket-requested rebuilds route to dev-202, the .201 lane vendors a newer
omnimarket only when an omnibase_infra rebuild stages omnimarket at dev head, and
the chain canary (C15), which binds staging delivery, grades .201 only. So the
.201 instance starts one rebuild at dev head on its own when it has been idle
for 16 minutes and its running omnimarket revision is not the dev head.

The TLA+ model (DeployRoutingIdle.tla, TLC logs on OMN-19509) fixed four
decisions, each pinned here:

* the converge runs from the poll loop's idle branch, so it never overlaps a
  routed job (mutant: SingleWriter201 violated);
* it starts only when the poll found nothing, so a waiting command runs first
  (mutant: QueuedFirst violated);
* it may not start within the converge ceiling plus the 10-minute margin
  before a C15 or C16 run, nor while one runs. The ticket's 10-minute margin
  ALONE lets a 20-minute converge still be running at the probe (mutant:
  NoConvergeAtProbe violated), and so does a guard that looks only forward;
* it stages each omnimarket head at most once.

The committed routing table routes nothing to dev-202, so omnimarket is not
routed elsewhere and the converge never fires: .201's behaviour is unchanged
(``test_idle_converge_is_inert_under_the_committed_table``).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent.agent import DeployAgent
from deploy_agent.events import BuildSource, EnumRuntimeLane, Scope
from deploy_agent.idle_converge import (
    CONVERGE_CEILING,
    IDLE_AFTER,
    IDLE_CONVERGE_REQUESTER,
    PROBE_MARGIN,
    EnumIdleConvergeVerdict,
    ModelIdleConvergeInputs,
    converge_command,
    decide,
    load_probe_windows,
    probe_blocking,
    read_omnimarket_dev_head,
    read_running_omnimarket_ref,
)
from deploy_agent.job_state import JobStore
from deploy_agent.routing import (
    DeployRouter,
    ModelRoute,
    load_routing_table,
    resolve_instance,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
HEAD = "b" * 40
RUNNING = "a" * 40

#: A C15 run in the committed window file: `41 1,3,...,23 * * *`, 25 minutes.
C15_START = datetime(2026, 9, 25, 5, 41, tzinfo=UTC)
#: Far from every guarded probe (C15 odd hours :41, C16 3/9/15/21 :29).
QUIET = datetime(2026, 9, 25, 4, 30, tzinfo=UTC)


def _inputs(**overrides: Any) -> ModelIdleConvergeInputs:
    values: dict[str, Any] = {
        "now": QUIET,
        "omnimarket_routed_elsewhere": True,
        "job_active": False,
        "last_activity": QUIET - IDLE_AFTER,
        "running_ref": RUNNING,
        "head_ref": HEAD,
        "probe_blocker": None,
        "windows_error": None,
        "attempted_heads": frozenset(),
    }
    values.update(overrides)
    return ModelIdleConvergeInputs(**values)


# --------------------------------------------------------------------------- #
# AC1 -- idle and behind: one rebuild at dev head, as agent/idle-converge       #
# --------------------------------------------------------------------------- #
class TestIdleConvergeDecision:
    def test_idle_converge_fires_when_idle_16_min_and_behind(self) -> None:
        decision = decide(_inputs())
        assert decision.verdict is EnumIdleConvergeVerdict.CONVERGE
        assert decision.head_ref == HEAD

    def test_idle_converge_waits_out_the_16_minutes(self) -> None:
        decision = decide(
            _inputs(last_activity=QUIET - IDLE_AFTER + timedelta(seconds=1))
        )
        assert decision.verdict is EnumIdleConvergeVerdict.RECENTLY_ACTIVE

    def test_idle_converge_does_nothing_when_up_to_date(self) -> None:
        decision = decide(_inputs(running_ref=HEAD))
        assert decision.verdict is EnumIdleConvergeVerdict.UP_TO_DATE

    def test_idle_converge_never_overlaps_a_job(self) -> None:
        """SingleWriter201: the model's out-of-loop mutant ran both at once."""
        decision = decide(_inputs(job_active=True))
        assert decision.verdict is EnumIdleConvergeVerdict.JOB_ACTIVE

    def test_idle_converge_stages_each_head_once(self) -> None:
        """OneConvergePerHead: a converge that failed is not retried in a loop."""
        decision = decide(_inputs(attempted_heads=frozenset({HEAD})))
        assert decision.verdict is EnumIdleConvergeVerdict.ALREADY_ATTEMPTED

    @pytest.mark.parametrize(
        ("field", "verdict"),
        [
            ("head_ref", EnumIdleConvergeVerdict.HEAD_UNREADABLE),
            ("running_ref", EnumIdleConvergeVerdict.RUNNING_UNREADABLE),
        ],
    )
    def test_idle_converge_refuses_on_an_unreadable_revision(
        self, field: str, verdict: EnumIdleConvergeVerdict
    ) -> None:
        assert decide(_inputs(**{field: None})).verdict is verdict

    def test_idle_converge_refuses_when_the_window_file_is_unreadable(self) -> None:
        """An unknown probe schedule is not a clear one."""
        decision = decide(_inputs(windows_error="unreadable"))
        assert decision.verdict is EnumIdleConvergeVerdict.WINDOWS_UNREADABLE

    def test_idle_converge_only_when_omnimarket_is_routed_elsewhere(self) -> None:
        decision = decide(_inputs(omnimarket_routed_elsewhere=False))
        assert decision.verdict is EnumIdleConvergeVerdict.NOT_ROUTED_ELSEWHERE

    def test_idle_converge_command_shape(self) -> None:
        cmd = converge_command()
        assert cmd.requested_by == IDLE_CONVERGE_REQUESTER == "agent/idle-converge"
        assert cmd.runtime_lane is EnumRuntimeLane.DEV
        assert cmd.scope is Scope.RUNTIME
        # Workspace: the build stages omnimarket from origin/dev, which is the
        # whole point, exactly as a CI-published dev command does.
        assert cmd.build_source is BuildSource.WORKSPACE
        assert isinstance(cmd.correlation_id, UUID)


# --------------------------------------------------------------------------- #
# AC2 -- never inside a probe window                                           #
# --------------------------------------------------------------------------- #
class TestIdleConvergeRespectsProbeWindow:
    def _windows(self) -> list[Any]:
        return load_probe_windows(REPO_ROOT)

    def test_idle_converge_respects_probe_window_inside_the_10_minute_margin(
        self,
    ) -> None:
        now = C15_START - timedelta(minutes=10)
        assert probe_blocking(now, self._windows()) is not None
        assert (
            decide(
                _inputs(now=now, probe_blocker=probe_blocking(now, self._windows()))
            ).verdict
            is EnumIdleConvergeVerdict.PROBE_WINDOW
        )

    def test_idle_converge_respects_probe_window_by_the_converge_ceiling(
        self,
    ) -> None:
        """The model's margin-only mutant: 11 minutes out is outside the
        ticket's 10-minute margin, but a converge started there is still
        running when C15 starts. The exclusion is ceiling plus margin."""
        now = C15_START - timedelta(minutes=11)
        assert probe_blocking(now, self._windows()) is not None
        edge = C15_START - CONVERGE_CEILING - PROBE_MARGIN
        assert probe_blocking(edge - timedelta(minutes=1), self._windows()) is None
        assert probe_blocking(edge, self._windows()) is not None

    def test_idle_converge_respects_probe_window_while_the_probe_runs(self) -> None:
        """The model's forward-only mutant: a guard that only looks ahead
        starts a converge under a C15 that is already running."""
        windows = self._windows()
        assert probe_blocking(C15_START + timedelta(minutes=5), windows) is not None
        assert probe_blocking(C15_START + timedelta(minutes=25), windows) is not None
        assert probe_blocking(C15_START + timedelta(minutes=26), windows) is None

    def test_idle_converge_respects_probe_window_for_c16(self) -> None:
        c16 = datetime(2026, 9, 25, 9, 29, tzinfo=UTC)
        blocker = probe_blocking(c16 - timedelta(minutes=5), self._windows())
        assert blocker is not None and blocker.startswith("C16")

    def test_idle_converge_respects_probe_window_quiet_time_is_clear(self) -> None:
        """Positive control: the guard is not simply always closed."""
        assert probe_blocking(QUIET, self._windows()) is None


# --------------------------------------------------------------------------- #
# Readers                                                                      #
# --------------------------------------------------------------------------- #
def _proc(stdout: str, code: int = 0) -> Any:
    import subprocess

    return subprocess.CompletedProcess(
        args=[], returncode=code, stdout=stdout, stderr=""
    )


class TestIdleConvergeReaders:
    def test_idle_converge_reads_the_running_omnimarket_ref(self) -> None:
        manifest = (
            '{"infra_vcs_ref": "' + "c" * 40 + '", "per_repo_vcs_provenance": '
            '{"siblings": {"omnimarket": {"vcs_ref": "' + RUNNING + '"}}}}'
        )
        seen: list[list[str]] = []

        def run(argv: list[str], timeout: int) -> Any:
            seen.append(argv)
            return _proc(manifest)

        assert read_running_omnimarket_ref("omninode-runtime", run=run) == RUNNING
        assert seen[0][:3] == ["docker", "exec", "omninode-runtime"]

    @pytest.mark.parametrize(
        "stdout", ["not json", '{"per_repo_vcs_provenance": {}}', '{"x": 1}']
    )
    def test_idle_converge_an_unreadable_manifest_is_none(self, stdout: str) -> None:
        assert (
            read_running_omnimarket_ref(
                "omninode-runtime", run=lambda argv, timeout: _proc(stdout)
            )
            is None
        )

    def test_idle_converge_reads_the_omnimarket_dev_head(self, tmp_path: Path) -> None:
        def run(argv: list[str], timeout: int) -> Any:
            assert argv[-2:] == ["origin", "refs/heads/dev"]
            return _proc(f"{HEAD}\trefs/heads/dev\n")

        assert read_omnimarket_dev_head(tmp_path, run=run) == HEAD
        assert (
            read_omnimarket_dev_head(tmp_path, run=lambda argv, timeout: _proc("", 128))
            is None
        )


# --------------------------------------------------------------------------- #
# The agent: the idle branch, and inert under the committed table              #
# --------------------------------------------------------------------------- #
class _FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.container_residue: list[object] = []
        self.sibling_source_refs: dict[str, str] = {}
        self.recreate_supervision: list[object] = []
        self.verify_recreate: list[object] = []
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
    omnimarket_to_202: bool,
) -> DeployAgent:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "agent-state")
    monkeypatch.setattr(agent_mod, "publish_result", lambda payload, config: False)
    monkeypatch.setattr(agent_mod, "LAB_OVERLAY_ENABLED", False)
    agent = DeployAgent(skip_self_update=True)
    if omnimarket_to_202:
        table = agent._router.table if agent._router else load_routing_table()
        routed = table.model_copy(
            update={
                "routes": (
                    ModelRoute(
                        runtime_lane=EnumRuntimeLane.DEV,
                        requester_repository="omnimarket",
                        instance="dev-202",
                    ),
                )
            }
        )
        agent._router = DeployRouter(
            routed, resolve_instance(routed), lambda sha: routed
        )
    agent.job_store = JobStore(tmp_path / "jobs")
    agent.executor = _FakeExecutor()  # type: ignore[assignment]
    agent._idle_converge_started_at = QUIET - timedelta(hours=1)
    agent._idle_converge_now = lambda: QUIET
    agent._idle_read_running_ref = lambda: RUNNING
    agent._idle_read_head_ref = lambda: HEAD
    agent._idle_probe_blocker = lambda now: None
    return agent


class TestIdleConvergeInTheAgent:
    def test_idle_converge_runs_one_rebuild_at_dev_head(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _agent(tmp_path, monkeypatch, omnimarket_to_202=True)

        agent._maybe_idle_converge()

        jobs = [
            JobStore(tmp_path / "jobs").load(UUID(p.stem))
            for p in (tmp_path / "jobs").glob("*.json")
        ]
        assert len(jobs) == 1
        job = jobs[0]
        assert job is not None
        assert job.command["requested_by"] == IDLE_CONVERGE_REQUESTER
        assert job.command["runtime_lane"] == "dev"
        assert job.status == "success"
        assert "rebuild_scope" in agent.executor.calls  # type: ignore[attr-defined]

        # The same head is not converged twice.
        agent._idle_converge_last_check = None
        agent._maybe_idle_converge()
        assert len(list((tmp_path / "jobs").glob("*.json"))) == 1

    def test_idle_converge_is_inert_under_the_committed_table(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The committed table routes nothing to dev-202: .201 is unchanged."""
        agent = _agent(tmp_path, monkeypatch, omnimarket_to_202=False)
        agent._maybe_idle_converge()
        assert list((tmp_path / "jobs").glob("*.json")) == []
        assert agent.executor.calls == []  # type: ignore[attr-defined]

    def test_idle_converge_only_on_the_poll_loops_idle_branch(self) -> None:
        """QueuedFirst and SingleWriter201 hold because the converge is called
        from the branch where the poll returned no command, and nowhere else."""
        import inspect

        source = inspect.getsource(DeployAgent.run)
        idle_branch = source[source.index("else:") :]
        assert "_maybe_idle_converge" in idle_branch
        assert source.count("_maybe_idle_converge") == 1

    def test_idle_converge_is_throttled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _agent(tmp_path, monkeypatch, omnimarket_to_202=True)
        agent._idle_read_head_ref = lambda: None  # a refused check
        agent._maybe_idle_converge()
        reads: list[str] = []
        agent._idle_read_head_ref = lambda: reads.append("x") or HEAD
        agent._maybe_idle_converge()
        assert reads == [], "a second check inside the interval must not run"
