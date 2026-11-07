# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22808: the dev agent follows the estimator's declared web repository when idle.

The estimator page image is built only inside a dev-lane rebuild. A merge to the
web repository starts none, so the page served the build from the last
omnimarket or omnibase_infra rebuild (measured 2026-11-06: a merge at 22:32:37Z
and no job for 77 minutes). The follower reads the repository from the
estimator build declaration, and every test names a repository that is not in
the code, so a hard-coded name cannot satisfy them.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent import executor as execution
from deploy_agent.agent import DeployAgent
from deploy_agent.events import EnumRuntimeLane
from deploy_agent.idle_converge import (
    IDLE_AFTER,
    IDLE_CONVERGE_REQUESTER,
    EnumIdleConvergeVerdict,
    EnumRuntimePresence,
)
from deploy_agent.job_state import JobStore
from deploy_agent.routing import ModelLaneFlags

pytestmark = pytest.mark.unit

HEAD = "b" * 40
RUNNING = "a" * 40
QUIET = datetime(2026, 9, 25, 4, 53, tzinfo=UTC)
SOURCE_LABEL = "ai.omninode.image.source-repo"
REVISION_LABEL = "org.opencontainers.image.revision"
WEB_REPOSITORY = "web-front-end"
WEB_URL = "https://example.invalid/web-front-end.git"
CONTAINER = "estimator-page"


def _effect() -> Any:
    return execution.load_dev_boot_guard(
        "nodes.node_setup_local_provision_effect.runtime_cost_estimator_build"
    )


def _follow() -> Any:
    return _effect().CostEstimatorWebFollow(
        repository=WEB_REPOSITORY,
        repository_url=WEB_URL,
        branch="dev",
        container=CONTAINER,
        source_label=SOURCE_LABEL,
        revision_label=REVISION_LABEL,
        read_timeout_seconds=30,
    )


def _proc(stdout: str, code: int = 0) -> Any:
    return subprocess.CompletedProcess(
        args=[], returncode=code, stdout=stdout, stderr=""
    )


def _labels(**labels: str) -> str:
    return json.dumps(labels)


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Every subprocess the reads start, answered from ``state``."""
    state = SimpleNamespace(calls=[], answer=_proc(""), raises=None)

    def fake_run(argv: list[str], **kwargs: object) -> Any:
        state.calls.append((argv, kwargs))
        if state.raises is not None:
            raise state.raises
        return state.answer

    monkeypatch.setattr(
        _effect(),
        "subprocess",
        SimpleNamespace(run=fake_run),
    )
    return state


class TestWebReaders:
    def test_up_to_date_reads_the_running_revision_from_the_image_label(
        self, run: SimpleNamespace
    ) -> None:
        run.answer = _proc(
            _labels(**{SOURCE_LABEL: WEB_REPOSITORY, REVISION_LABEL: RUNNING})
        )

        assert _follow().running_ref() == RUNNING
        argv, kwargs = run.calls[0]
        assert argv[:2] == ["docker", "inspect"] and argv[-1] == CONTAINER
        assert kwargs["timeout"] == 30

    @pytest.mark.parametrize(
        ("stdout", "code"),
        [
            ("not json", 0),
            ("null", 0),
            ("{}", 0),
            (_labels(**{SOURCE_LABEL: WEB_REPOSITORY, REVISION_LABEL: "short"}), 0),
            (_labels(**{SOURCE_LABEL: "another-repo", REVISION_LABEL: RUNNING}), 0),
            (_labels(**{REVISION_LABEL: RUNNING}), 0),
            ("", 1),
        ],
    )
    def test_unreadable_running_revision_is_none(
        self, run: SimpleNamespace, stdout: str, code: int
    ) -> None:
        run.answer = _proc(stdout, code)

        assert _follow().running_ref() is None

    def test_unreadable_running_revision_when_docker_is_missing(
        self, run: SimpleNamespace
    ) -> None:
        run.raises = OSError("no docker")

        assert _follow().running_ref() is None

    def test_head_is_read_on_the_declared_url_and_branch(
        self, run: SimpleNamespace
    ) -> None:
        run.answer = _proc(f"{HEAD}\trefs/heads/dev\n")

        assert _follow().head_ref() == HEAD
        assert run.calls[0][0] == [
            "git",
            "ls-remote",
            "--",
            WEB_URL,
            "refs/heads/dev",
        ]

    @pytest.mark.parametrize(
        ("stdout", "code"), [("", 0), ("  \n", 0), ("zz\trefs/x", 0), ("", 128)]
    )
    def test_unreadable_head_is_none(
        self, run: SimpleNamespace, stdout: str, code: int
    ) -> None:
        run.answer = _proc(stdout, code)

        assert _follow().head_ref() is None


class TestDeclaration:
    @pytest.fixture
    def effect(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        monkeypatch.setattr(
            execution,
            "_compose_env",
            lambda: {
                "COST_ESTIMATOR_WEB_SOURCE_REPOSITORY": WEB_REPOSITORY,
                "COST_ESTIMATOR_WEB_SOURCE_URL": WEB_URL,
            },
        )
        monkeypatch.setattr(
            execution,
            "lane_config_for",
            lambda lane: SimpleNamespace(disabled_services=()),
        )
        return _effect()

    def test_converge_declaration_names_the_repository_from_the_lane_settings(
        self, effect: Any
    ) -> None:
        follow = effect.dev_lane_web_follow(execution, EnumRuntimeLane.DEV)

        assert follow is not None
        assert follow.repository == WEB_REPOSITORY
        assert follow.repository_url == WEB_URL
        assert follow.branch == "dev"
        assert follow.container == "omninode-cost-estimator-web"
        assert follow.source_label == SOURCE_LABEL
        assert follow.revision_label == REVISION_LABEL

    def test_converge_without_the_follow_block_follows_nothing(
        self, effect: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        declared = effect._declaration(execution)
        assert "follow" in declared
        monkeypatch.setattr(
            effect,
            "_declaration",
            lambda module: {k: v for k, v in declared.items() if k != "follow"},
        )

        assert effect.dev_lane_web_follow(execution, EnumRuntimeLane.DEV) is None

    def test_converge_unset_source_settings_follow_nothing(
        self, effect: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(execution, "_compose_env", dict)

        assert effect.dev_lane_web_follow(execution, EnumRuntimeLane.DEV) is None

    def test_converge_an_unreadable_declaration_follows_nothing(
        self, effect: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(module: object) -> dict[str, object]:
            raise OSError("contract is gone")

        monkeypatch.setattr(effect, "_declaration", broken)

        assert effect.dev_lane_web_follow(execution, EnumRuntimeLane.DEV) is None

    def test_converge_a_disabled_estimator_or_another_lane_follows_nothing(
        self, effect: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert effect.dev_lane_web_follow(execution, EnumRuntimeLane.PROD) is None
        monkeypatch.setattr(
            execution,
            "lane_config_for",
            lambda lane: SimpleNamespace(disabled_services=("cost-estimator-web",)),
        )

        assert effect.dev_lane_web_follow(execution, EnumRuntimeLane.DEV) is None

    def test_the_compose_container_name_is_the_declared_one(self) -> None:
        root = Path(__file__).resolve().parents[4]
        compose = (root / "docker/docker-compose.dev-lane.yml").read_text()
        assert "container_name: omninode-cost-estimator-web" in compose


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
    web: Any,
    running: str | None,
    head: str | None,
) -> DeployAgent:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "agent-state")
    monkeypatch.setattr(agent_mod, "publish_result", lambda payload, config: False)
    monkeypatch.setattr(agent_mod, "LAB_OVERLAY_ENABLED", False)
    agent = DeployAgent(skip_self_update=True)
    agent.job_store = JobStore(tmp_path / "jobs")
    monkeypatch.setattr(agent, "executor", _FakeExecutor())
    agent._idle_converge_started_at = QUIET - timedelta(hours=1)
    agent._idle_converge_now = lambda: QUIET
    # The runtime siblings are on their heads, so only the web source can act.
    agent._idle_read_running_ref = lambda repo: (
        agent_mod._read_running_ref(repo) if repo == WEB_REPOSITORY else HEAD
    )
    agent._idle_read_head_ref = lambda repo: (
        agent_mod._read_dev_head(repo) if repo == WEB_REPOSITORY else HEAD
    )
    agent._idle_read_runtime_presence = lambda: (
        EnumRuntimePresence.RUNNING,
        "the runtime is running",
    )
    agent._idle_read_dev_tags = lambda repo: None
    monkeypatch.setattr(
        agent_mod, "publish_dev_tag_heartbeat", lambda payload, config: True
    )
    agent._idle_probe_blocker = lambda now: None
    agent._lane_flags = ModelLaneFlags()
    if web is not None:
        web = SimpleNamespace(
            repository=web.repository,
            running_ref=lambda: running,
            head_ref=lambda: head,
        )
    monkeypatch.setattr(agent_mod, "_web_source", lambda: web)
    return agent


def _jobs(tmp_path: Path) -> list[Any]:
    return [
        JobStore(tmp_path / "jobs").load(UUID(p.stem))
        for p in (tmp_path / "jobs").glob("*.json")
    ]


class TestTheAgentFollowsTheWebRepository:
    def test_converge_starts_one_idle_rebuild_when_the_web_head_moved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _agent(tmp_path, monkeypatch, web=_follow(), running=RUNNING, head=HEAD)

        agent._maybe_idle_converge()

        jobs = _jobs(tmp_path)
        assert len(jobs) == 1
        assert jobs[0].command["requested_by"] == IDLE_CONVERGE_REQUESTER
        assert jobs[0].command["runtime_lane"] == "dev"
        assert (
            agent._idle_converge_last_verdicts[WEB_REPOSITORY]
            is EnumIdleConvergeVerdict.CONVERGE
        )
        # The same web head is not converged on twice.
        agent._idle_converge_last_check = None
        agent._maybe_idle_converge()
        assert len(_jobs(tmp_path)) == 1

    def test_converge_without_a_declaration_makes_no_decision_for_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _agent(tmp_path, monkeypatch, web=None, running=RUNNING, head=HEAD)

        agent._maybe_idle_converge()

        assert WEB_REPOSITORY not in agent._idle_converge_last_verdicts
        assert _jobs(tmp_path) == []

    def test_up_to_date_starts_no_job(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _agent(tmp_path, monkeypatch, web=_follow(), running=HEAD, head=HEAD)

        agent._maybe_idle_converge()

        assert (
            agent._idle_converge_last_verdicts[WEB_REPOSITORY]
            is EnumIdleConvergeVerdict.UP_TO_DATE
        )
        assert _jobs(tmp_path) == []

    @pytest.mark.parametrize(
        ("running", "head", "verdict"),
        [
            (None, HEAD, EnumIdleConvergeVerdict.RUNNING_UNREADABLE),
            (RUNNING, None, EnumIdleConvergeVerdict.HEAD_UNREADABLE),
        ],
    )
    def test_unreadable_revision_refuses_and_starts_nothing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        running: str | None,
        head: str | None,
        verdict: EnumIdleConvergeVerdict,
    ) -> None:
        agent = _agent(tmp_path, monkeypatch, web=_follow(), running=running, head=head)

        agent._maybe_idle_converge()

        assert agent._idle_converge_last_verdicts[WEB_REPOSITORY] is verdict
        assert _jobs(tmp_path) == []

    def test_converge_waits_out_the_idle_period(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _agent(tmp_path, monkeypatch, web=_follow(), running=RUNNING, head=HEAD)
        agent._idle_converge_started_at = QUIET - IDLE_AFTER + timedelta(minutes=1)

        agent._maybe_idle_converge()

        assert (
            agent._idle_converge_last_verdicts[WEB_REPOSITORY]
            is EnumIdleConvergeVerdict.RECENTLY_ACTIVE
        )
        assert _jobs(tmp_path) == []
