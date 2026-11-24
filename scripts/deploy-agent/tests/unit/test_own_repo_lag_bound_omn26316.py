# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26316: the default instance's declared dev lag bound covers its own repository.

THE INCIDENT, 2026-11-23
------------------------
omnibase_infra#6191 merged to dev at 17:46:52Z. CI published its rebuild on
``onex.cmd.omnimarket.redeploy-start.v1`` at 18:01:10Z, but the redeploy
orchestrator, the deploy effect and the dev roll window gate that forward it
all run inside the dev runtime, which read "running but unhealthy ... wedged"
before every hourly idle restore that day. The orchestrator forwarded it at
19:44:57Z and the deploy agent received it at 21:45:23Z, superseded by a build
that already carried it. omnibase_infra#6204 (merged 21:43:39Z) waited behind
the roll window gate until 22:38:36Z.

dev-201 declares ``max_dev_lag_minutes: 20`` (``config/deploy_lane_routing.yaml``),
but the bound never reached omnibase_infra: the agent read the running and head
revisions only for a repository routed to another instance, and
:func:`deploy_agent.idle_converge.decide` refused ``not_routed_elsewhere``
before it looked at the bound. So the agent, the one part of the path that runs
outside the runtime, never acted on an infra merge the runtime failed to
forward, and its ``dev_lane_drift`` warning never fired for one.

THE RULE
--------
When the instance declares a bound, the agent reads its own repository's
revisions as well. An overdue lane converges on it as on any followed
repository (still never inside a reserved probe window, never beside a job,
never on a frozen lane, once per head). A head whose only difference from the
running revision is declared not-runtime paths is not a lag.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent.idle_converge import (
    IDLE_CONVERGE_REQUESTER,
    EnumIdleConvergeVerdict,
    ModelIdleConvergeInputs,
    decide,
)
from deploy_agent.job_state import JobStore
from deploy_agent.routing import ModelLaneFlags

from .test_idle_converge_omn19509 import HEAD, RUNNING, _agent

pytestmark = pytest.mark.unit

#: omnibase_infra#6191 merged at 17:46:52Z; dev-201 declares a 20-minute bound.
MERGED = datetime(2026, 11, 23, 17, 46, 52, tzinfo=UTC)
BOUND = timedelta(minutes=20)
OVERDUE_AT = MERGED + BOUND + timedelta(minutes=1)


def _inputs(**overrides: Any) -> ModelIdleConvergeInputs:
    now = overrides.get("now", OVERDUE_AT)
    values: dict[str, Any] = {
        "now": now,
        "repository": "omnibase_infra",
        "routed_elsewhere": False,
        "job_active": False,
        "last_activity": now,
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


class TestDecide:
    def test_an_overdue_own_repository_converges(self) -> None:
        decision = decide(_inputs())

        assert decision.verdict is EnumIdleConvergeVerdict.CONVERGE
        assert decision.head_ref == HEAD
        assert "overdue" in decision.detail
        assert "sends omnibase_infra rebuilds to this instance" in decision.detail

    def test_within_the_bound_the_own_repository_still_waits_for_its_trigger(
        self,
    ) -> None:
        decision = decide(_inputs(now=MERGED + BOUND - timedelta(minutes=1)))
        assert decision.verdict is EnumIdleConvergeVerdict.NOT_ROUTED_ELSEWHERE

    def test_with_no_declared_bound_nothing_changes(self) -> None:
        decision = decide(_inputs(max_lag=None))
        assert decision.verdict is EnumIdleConvergeVerdict.NOT_ROUTED_ELSEWHERE

    def test_an_overdue_own_repository_still_refuses_a_probe_window(self) -> None:
        decision = decide(_inputs(probe_blocker="C15 run at 2026-11-23T18:41Z"))
        assert decision.verdict is EnumIdleConvergeVerdict.PROBE_WINDOW

    def test_an_overdue_own_repository_never_overlaps_a_job(self) -> None:
        decision = decide(_inputs(job_active=True))
        assert decision.verdict is EnumIdleConvergeVerdict.JOB_ACTIVE

    def test_an_overdue_own_repository_converges_on_a_head_once(self) -> None:
        decision = decide(_inputs(attempted_heads=frozenset({HEAD})))
        assert decision.verdict is EnumIdleConvergeVerdict.ALREADY_ATTEMPTED


def _own_repository_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    max_lag_minutes: int | None,
    runtime_unchanged: bool,
    reads: list[str],
) -> Any:
    agent = _agent(tmp_path, monkeypatch, omnimarket_to_202=True)
    agent._idle_converge_now = lambda: OVERDUE_AT
    # The lane was active a moment ago: only the bound can start a converge.
    agent._idle_converge_started_at = OVERDUE_AT
    agent._idle_read_running_image = lambda: "omnibase-infra-omninode-runtime"

    def running(repository: str) -> str:
        reads.append(repository)
        # omnimarket runs its head, so only omnibase_infra can lag.
        return HEAD if repository == "omnimarket" else RUNNING

    agent._idle_read_running_ref = running
    monkeypatch.setattr(
        _reading(), "runtime_unchanged", lambda repository, head, run: runtime_unchanged
    )
    agent._lane_flags = ModelLaneFlags(max_dev_lag_minutes=max_lag_minutes)
    agent._dev_lag_behind_since["omnibase_infra"] = MERGED
    return agent


def _reading() -> Any:
    return agent_mod.load_dev_boot_guard(agent_mod.JOB_READING)


def _jobs(tmp_path: Path) -> list[Any]:
    store = JobStore(tmp_path / "jobs")
    return [store.load(UUID(p.stem)) for p in (tmp_path / "jobs").glob("*.json")]


class TestInTheAgent:
    def test_an_overdue_own_repository_runs_one_converge(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reads: list[str] = []
        agent = _own_repository_agent(
            tmp_path,
            monkeypatch,
            max_lag_minutes=20,
            runtime_unchanged=False,
            reads=reads,
        )

        agent._maybe_idle_converge()

        jobs = _jobs(tmp_path)
        assert len(jobs) == 1
        assert jobs[0] is not None
        assert jobs[0].command["requested_by"] == IDLE_CONVERGE_REQUESTER
        assert "omnibase_infra" in reads

    def test_a_not_runtime_head_is_not_a_lag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reads: list[str] = []
        agent = _own_repository_agent(
            tmp_path,
            monkeypatch,
            max_lag_minutes=20,
            runtime_unchanged=True,
            reads=reads,
        )

        agent._maybe_idle_converge()

        assert _jobs(tmp_path) == []
        assert "omnibase_infra" not in agent._dev_lag_behind_since

    def test_with_no_declared_bound_the_own_repository_is_not_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reads: list[str] = []
        agent = _own_repository_agent(
            tmp_path,
            monkeypatch,
            max_lag_minutes=None,
            runtime_unchanged=False,
            reads=reads,
        )

        agent._maybe_idle_converge()

        assert _jobs(tmp_path) == []
        assert "omnibase_infra" not in reads


class TestLagRefsReader:
    """The job reading node's reader the agent judges a lag on."""

    class _Host:
        def __init__(self) -> None:
            self._idle_read_running_ref = lambda repository: RUNNING
            self._idle_read_head_ref = lambda repository: HEAD

    def test_a_not_runtime_head_reads_as_the_running_revision(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[str, str, str]] = []

        def unchanged(repository: str, head: str, running: str) -> bool:
            calls.append((repository, head, running))
            return True

        monkeypatch.setattr(_reading(), "runtime_unchanged", unchanged)
        read = _reading().lag_refs_reader(self._Host())

        assert read("omnibase_infra", False) == (RUNNING, RUNNING)
        assert calls == [("omnibase_infra", HEAD, RUNNING)]

    def test_a_repository_routed_elsewhere_reads_its_head(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_reading(), "runtime_unchanged", lambda *a: True)
        read = _reading().lag_refs_reader(self._Host())

        assert read("omnimarket", True) == (RUNNING, HEAD)

    def test_an_unread_classifier_keeps_the_head_a_runtime_head(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unreadable(*args: object) -> bool:
            raise RuntimeError("cannot load the runtime change classifier")

        monkeypatch.setattr(_reading(), "runtime_unchanged", unreadable)
        read = _reading().lag_refs_reader(self._Host())

        assert read("omnibase_infra", False) == (RUNNING, HEAD)
