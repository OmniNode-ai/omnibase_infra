# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev deploy agent puts declared branch installs back after a recreate (OMN-17427).

The record, the apply, the readback and the release rule are the handler's, and
``tests/unit/handlers/test_handler_dev_lane_branch_install_omn17427.py`` replays
the refresh against a fake Docker. These tests pin the agent's side: it runs the
handler after a dev job that recreated the runtime and on an idle tick, only on
the dev lane, relays the handler's detector line into its own log, never lets a
failed re-apply change a job's verdict, and serves the record at
``/branch-installs``.

The handler is run as a subprocess from the deploy-source clone's ``src`` so it
needs no release of the package the agent's venv holds; the tests replace that
subprocess with a recorder and read the argv, environment and order it was given.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from aiohttp.test_utils import TestClient, TestServer
from deploy_agent import agent as agent_mod
from deploy_agent.agent import BRANCH_INSTALL_HANDLER, DeployAgent
from deploy_agent.events import EnumRuntimeLane, ModelRebuildRequested, Scope
from deploy_agent.health import create_health_app
from deploy_agent.job_state import JobStore

pytestmark = pytest.mark.unit

SERVICE = "omninode-runtime"
WRITER = (
    "/app/.venv/lib/python3.12/site-packages/omnimarket/nodes/"
    "node_projection_consumer_flow/handlers/handler_consumer_flow_runner.py"
)
HEAD = "8020dbdcc9a4961cdc5684d14c63b3bc024eef30"
RUNNING = "c" * 40
MISSING_LINE = (
    "2026-10-22 12:50:00 ERROR branch-installs: BRANCH-INSTALL-MISSING: "
    "id=c28-writer-omnimarket-4312 repo=omnimarket pr=#4312 head=8020dbdcc9a4 "
    f"ticket=OMN-17427 service={SERVICE} container=abc file={WRITER} state=drifted "
    "expected_sha256=17d6b3074259 observed_sha256=778b41873969 detail=sha256 differs "
    "friction_type=branch_install_missing"
)


class _Exec:
    """Stands in for the subprocess that runs the handler; records every call."""

    def __init__(self, timeline: list[str]) -> None:
        self.timeline = timeline
        self.calls: list[tuple[list[str], int, dict[str, str]]] = []
        self.results: dict[str, subprocess.CompletedProcess[str]] = {}

    def answer(self, command: str, code: int = 0, out: str = "", err: str = "") -> None:
        self.results[command] = subprocess.CompletedProcess([], code, out, err)

    def __call__(
        self, argv: list[str], timeout: int, env: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        command = argv[argv.index(BRANCH_INSTALL_HANDLER) + 1]
        self.calls.append((argv, timeout, env))
        self.timeline.append(f"handler {command}")
        return self.results.get(
            command, subprocess.CompletedProcess(argv, 0, "RECEIPT {}", "")
        )

    def commands(self) -> list[str]:
        return [
            argv[argv.index(BRANCH_INSTALL_HANDLER) + 1] for argv, _, _ in self.calls
        ]


class _FakeExecutor:
    """The executor surface ``_run_deploy`` touches on a dev job that passes."""

    def __init__(self, timeline: list[str]) -> None:
        self.timeline = timeline
        self.container_residue: list[object] = []
        self.sibling_source_refs: dict[str, str] = {}
        self.recreate_supervision: list[object] = []
        self.verify_recreate: list[object] = []
        self.oneshot_reruns: list[object] = []
        self.cloud_migration_failure = ""
        self.deps_convergence: list[object] = []
        self.compose_invocations: list[object] = []
        self.health_checks: list[object] = []

    def preflight(self, **kwargs: object) -> None: ...

    def git_pull(self, git_ref: str, **kwargs: object) -> str:
        return "a" * 40

    def compose_gen(self, bundles: list[str], **kwargs: object) -> None: ...

    def seed_infisical(self, **kwargs: object) -> None: ...

    def validate_llm_endpoint_env_contract(self) -> None: ...

    def rebuild_scope(self, *args: object, **kwargs: object) -> list[str]:
        """The refresh: it recreates the runtime, and the layer with the fix goes."""
        self.timeline.append("recreate")
        return [SERVICE]

    def verify(self, **kwargs: object) -> list[object]:
        self.timeline.append("verify")
        return []

    def run_lane_mirror(self, **kwargs: object) -> None:
        return None


REPO_SRC = Path(__file__).resolve().parents[4] / "src"


def _declare(state: Path, tmp_path: Path) -> None:
    """Register through the handler's own command, as a lane does.

    The handler is not in the omnibase-infra release this sub-project's venv holds,
    so it is run from the repository's ``src`` exactly as the agent runs it.
    """
    source = tmp_path / "branch" / "writer.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"# the branch's writer\n")
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            BRANCH_INSTALL_HANDLER,
            "register",
            "--state-dir",
            str(state),
            "--lane",
            "dev",
            "--id",
            "c28-writer-omnimarket-4312",
            "--repo",
            "omnimarket",
            "--pr",
            "4312",
            "--branch",
            "jonah/omn-17427-c28-consumer-flow-writer",
            "--head",
            HEAD,
            "--ticket",
            "OMN-17427",
            "--registered-by",
            "c28-projection-writer-0720",
            "--service",
            SERVICE,
            "--expires-in-hours",
            "48",
            "--file",
            f"{WRITER}={source}",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(REPO_SRC)},
    )
    assert done.returncode == 0, done.stdout + done.stderr


def _agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[DeployAgent, ModelRebuildRequested, JobStore, _Exec, list[str]]:
    timeline: list[str] = []
    cmd = ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by="test",
        scope=Scope.RUNTIME,
        runtime_lane=EnumRuntimeLane.DEV,
    )
    jobs = JobStore(tmp_path / "jobs")
    jobs.accept(cmd.correlation_id, cmd.model_dump(mode="json"))
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    monkeypatch.setenv("OMNI_HOME", str(tmp_path / "registry"))
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(agent_mod, "LAB_OVERLAY_ENABLED", False)
    agent = DeployAgent(skip_self_update=True)
    agent.job_store = jobs
    monkeypatch.setattr(agent, "executor", _FakeExecutor(timeline))
    monkeypatch.setattr(agent_mod, "publish_result", lambda payload, config: False)
    agent._idle_read_running_ref = lambda repo: RUNNING
    runner = _Exec(timeline)
    agent._branch_install_exec = runner
    return agent, cmd, jobs, runner, timeline


# --------------------------------------------------------------------------- #
# the job path                                                                #
# --------------------------------------------------------------------------- #
def test_a_dev_job_applies_the_record_after_the_recreate_and_the_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refresh replayed through the real ``_run_deploy``: recreate, verify, apply."""
    agent, cmd, jobs, runner, timeline = _agent(tmp_path, monkeypatch)
    _declare(tmp_path / "state", tmp_path)

    agent._run_deploy(cmd)

    job = jobs.load(cmd.correlation_id)
    assert job is not None and job.status == "success"
    assert timeline == ["recreate", "verify", "handler apply"]
    argv, timeout, env = runner.calls[0]
    assert argv[:4] == [sys.executable, "-m", BRANCH_INSTALL_HANDLER, "apply"]
    assert argv[argv.index("--state-dir") + 1] == str(tmp_path / "state")
    assert argv[argv.index("--compose-project") + 1] == (
        agent_mod.lane_config_for(EnumRuntimeLane.DEV).compose_project
    )
    assert env["PYTHONPATH"].split(":")[0] == f"{agent_mod.REPO_DIR}/src"
    assert timeout == agent_mod.BRANCH_INSTALL_APPLY_TIMEOUT_SECONDS


def test_the_apply_is_given_each_declared_repositorys_clone_and_running_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, cmd, _, runner, _ = _agent(tmp_path, monkeypatch)
    _declare(tmp_path / "state", tmp_path)

    agent._run_deploy(cmd)

    argv = runner.calls[0][0]
    assert (
        argv[argv.index("--clone") + 1]
        == f"omnimarket={tmp_path / 'registry' / 'omnimarket'}"
    )
    assert argv[argv.index("--running-ref") + 1] == f"omnimarket={RUNNING}"


def test_the_detector_line_reaches_the_agents_job_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    agent, cmd, jobs, runner, _ = _agent(tmp_path, monkeypatch)
    _declare(tmp_path / "state", tmp_path)
    runner.answer("apply", code=4, out="RECEIPT {}", err=MISSING_LINE + "\ninfo line")

    with caplog.at_level(logging.INFO):
        agent._run_deploy(cmd)

    detector = [r for r in caplog.records if "BRANCH-INSTALL-MISSING" in r.getMessage()]
    assert len(detector) == 1
    assert detector[0].levelno == logging.ERROR
    assert "friction_type=branch_install_missing" in detector[0].getMessage()
    assert f"file={WRITER}" in detector[0].getMessage()
    job = jobs.load(cmd.correlation_id)
    assert job is not None and job.status == "success", (
        "a missing install is not a verdict"
    )


def test_a_handler_that_cannot_run_never_changes_the_jobs_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A job built from a ref older than the handler: the module is not in its src."""
    agent, cmd, jobs, runner, _ = _agent(tmp_path, monkeypatch)
    _declare(tmp_path / "state", tmp_path)
    runner.answer("apply", code=1, err="No module named 'omnibase_infra.handlers.x'")

    with caplog.at_level(logging.INFO):
        agent._run_deploy(cmd)

    job = jobs.load(cmd.correlation_id)
    assert job is not None and job.status == "success"
    assert any(
        "friction_type=branch_install_failed" in r.getMessage() for r in caplog.records
    )


def test_a_handler_that_times_out_never_changes_the_jobs_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, cmd, jobs, _, _ = _agent(tmp_path, monkeypatch)
    _declare(tmp_path / "state", tmp_path)

    def hang(
        argv: list[str], timeout: int, env: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(argv, timeout)

    agent._branch_install_exec = hang

    agent._run_deploy(cmd)

    job = jobs.load(cmd.correlation_id)
    assert job is not None and job.status == "success"


def test_a_lane_with_no_declared_install_runs_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, cmd, jobs, runner, timeline = _agent(tmp_path, monkeypatch)

    agent._run_deploy(cmd)
    agent._maybe_reapply_branch_installs()

    assert runner.calls == []
    assert timeline == ["recreate", "verify"]
    job = jobs.load(cmd.correlation_id)
    assert job is not None and job.status == "success"


def test_the_reapply_is_a_dev_lane_step_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, _, _, runner, _ = _agent(tmp_path, monkeypatch)
    _declare(tmp_path / "state", tmp_path)

    for lane in (EnumRuntimeLane.PROD, EnumRuntimeLane.STABILITY_TEST):
        assert agent._reapply_branch_installs(lane) is True

    assert runner.calls == [], "no handler ran for a governed lane"


# --------------------------------------------------------------------------- #
# the idle tick                                                               #
# --------------------------------------------------------------------------- #
class _Lock:
    """Records the lane lock the tick takes."""

    events: list[str] = []
    contended = False

    def __init__(self, project: str, **kwargs: object) -> None:
        self.events.append(f"lock {project} {kwargs['lane']} {kwargs['ref']}")

    def __enter__(self) -> None:
        if self.contended:
            raise agent_mod.LaneLockContendedError("held by a deploy job")
        self.events.append("held")

    def __exit__(self, *exc: object) -> None:
        self.events.append("released")


@pytest.fixture
def lock(monkeypatch: pytest.MonkeyPatch) -> type[_Lock]:
    _Lock.events = []
    _Lock.contended = False
    monkeypatch.setattr(agent_mod, "lane_lock", _Lock)
    return _Lock


def test_the_idle_tick_applies_a_loss_under_the_lane_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lock: type[_Lock]
) -> None:
    """A recreate the job path never saw: the settle worker's pin recreate, a hand refresh."""
    agent, _, _, runner, timeline = _agent(tmp_path, monkeypatch)
    _declare(tmp_path / "state", tmp_path)
    runner.answer("check", code=4, err=MISSING_LINE)

    agent._maybe_reapply_branch_installs()

    assert runner.commands() == ["release-due", "check", "apply"]
    project = agent_mod.lane_config_for(EnumRuntimeLane.DEV).compose_project
    assert lock.events == [f"lock {project} dev branch-installs", "held", "released"]
    assert timeline == ["handler release-due", "handler check", "handler apply"]


def test_the_idle_tick_leaves_a_whole_lane_alone_and_takes_no_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lock: type[_Lock]
) -> None:
    agent, _, _, runner, _ = _agent(tmp_path, monkeypatch)
    _declare(tmp_path / "state", tmp_path)

    agent._maybe_reapply_branch_installs()

    assert runner.commands() == ["release-due", "check"]
    assert lock.events == []


def test_the_idle_tick_is_throttled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lock: type[_Lock]
) -> None:
    agent, _, _, runner, _ = _agent(tmp_path, monkeypatch)
    _declare(tmp_path / "state", tmp_path)

    agent._maybe_reapply_branch_installs()
    agent._maybe_reapply_branch_installs()

    assert runner.commands() == ["release-due", "check"]


def test_the_idle_tick_skips_the_apply_while_a_deploy_holds_the_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lock: type[_Lock]
) -> None:
    agent, _, _, runner, _ = _agent(tmp_path, monkeypatch)
    _declare(tmp_path / "state", tmp_path)
    runner.answer("check", code=4, err=MISSING_LINE)
    lock.contended = True

    agent._maybe_reapply_branch_installs()

    assert runner.commands() == ["release-due", "check"]


def test_the_idle_tick_runs_nothing_on_an_agent_fenced_away_from_dev(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lock: type[_Lock]
) -> None:
    agent, _, _, runner, _ = _agent(tmp_path, monkeypatch)
    _declare(tmp_path / "state", tmp_path)
    agent._allowed_lanes = frozenset({EnumRuntimeLane.PROD})

    agent._maybe_reapply_branch_installs()

    assert runner.calls == []


# --------------------------------------------------------------------------- #
# /branch-installs                                                            #
# --------------------------------------------------------------------------- #
async def test_the_agent_serves_the_declared_installs_and_the_missing_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, _, _, _, _ = _agent(tmp_path, monkeypatch)
    state = tmp_path / "state"
    _declare(state, tmp_path)
    (state / "branch-installs" / "status.json").write_text(
        json.dumps(
            {
                "readbacks": [
                    {"install_key": "c28-writer-omnimarket-4312", "state": "missing"},
                    {"install_key": "other-install", "state": "present"},
                ]
            }
        )
    )

    app = create_health_app(
        JobStore(tmp_path / "jobs"),
        lambda: "idle",
        get_branch_installs=agent._latest_branch_installs,
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get("/branch-installs")
        body = await response.json()
    finally:
        await client.close()

    assert response.status == 200
    assert [i["id"] for i in body["declared"]] == ["c28-writer-omnimarket-4312"]
    assert body["missing"] == ["c28-writer-omnimarket-4312"]
    assert "content" not in json.dumps(body), "no file content is served"


async def test_an_agent_with_no_record_serves_an_empty_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, _, _, _, _ = _agent(tmp_path, monkeypatch)
    body = agent._latest_branch_installs()
    assert body["declared"] == [] and body["missing"] == []


async def test_a_corrupt_record_is_served_as_unreadable_not_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, _, _, _, _ = _agent(tmp_path, monkeypatch)
    root = tmp_path / "state" / "branch-installs"
    root.mkdir(parents=True)
    (root / "records.json").write_text("{not json")
    assert agent._latest_branch_installs()["installs"] is None


async def test_an_agent_without_the_getter_says_indeterminate(tmp_path: Path) -> None:
    app = create_health_app(JobStore(tmp_path / "jobs"), lambda: "idle")
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        body = await (await client.get("/branch-installs")).json()
    finally:
        await client.close()
    assert body["installs"] is None
