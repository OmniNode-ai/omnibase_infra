# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26493: idle restore and idle converge do not rebuild a dev head whose last roll failed.

Between 2026-11-16 and 2026-11-23 seven failed dev jobs rebuilt an omnibase_infra head that
an earlier job had already failed to boot (7b25f0ff three times: 5623655f, 8648de6f,
a9ecbe18). Every failed roll recreated the runtime and left it unhealthy until the guard
restored the rollback image. The agent now refuses an idle build of a head whose newest
terminal dev job failed with a recorded verdict class, until the head changes, writes one
ledger row per head, and still runs a request a lane or the operator signed.

Select the acceptance criteria with ``-k idle_skips_failed_head`` (AC1),
``-k idle_failed_head_ledger_row`` (AC2), ``-k signed_request_bypasses_failed_head`` (AC3)
and ``-k failed_head_incident_replay`` (AC4).
"""

from __future__ import annotations

import json
import stat
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent.agent import JOB_READING, DeployAgent
from deploy_agent.events import (
    BuildSource,
    EnumRuntimeLane,
    ModelRebuildRequested,
    Scope,
)
from deploy_agent.executor import load_dev_boot_guard
from deploy_agent.idle_converge import (
    IDLE_CONVERGE_REQUESTER,
    IDLE_RESTORE_REQUESTER,
    EnumRuntimePresence,
)
from deploy_agent.job_state import JobState, JobStore
from deploy_agent.routing import (
    DeployRouter,
    ModelRoute,
    load_routing_table,
    resolve_instance,
)

from .test_idle_converge_omn19509 import _agent as _idle_agent

pytestmark = pytest.mark.unit

reading = load_dev_boot_guard(JOB_READING)

#: A lane that is not dev: its failures are never this rule's.
OTHER_LANE = next(
    lane.value for lane in EnumRuntimeLane if lane is not EnumRuntimeLane.DEV
)

FIXTURE = Path(__file__).parent / "fixtures" / "recorded_jobs_7b25f0ff_omn26493.json"
RECORDED = json.loads(FIXTURE.read_text(encoding="utf-8"))
FAILED_HEAD = RECORDED["failed_head"]
NEWER_HEAD = RECORDED["newer_head"]

#: The replay's first idle check: job 8648de6f started here on the host.
SECOND_RESTORE = datetime(2026, 11, 20, 21, 0, 33, tzinfo=UTC)
#: Before the replay's first job: the agent had run nothing since this time.
AGENT_START = datetime(2026, 11, 20, 18, 0, tzinfo=UTC)


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


def _fake_append(tmp_path: Path) -> tuple[str, Path]:
    """A stand-in for the bus append command: one stdin row per invocation into a file."""
    sink = tmp_path / "rows.txt"
    script = tmp_path / "append.py"
    script.write_text(
        "import json, sys\n"
        f"open({str(sink)!r}, 'a').write(sys.stdin.read())\n"
        "print(json.dumps({'status': 'accepted'}))\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return f"{sys.executable} {script}", sink


@pytest.fixture
def ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The agent's append route and its state directory; the file the rows land in."""
    command, sink = _fake_append(tmp_path)
    settings = reading.load_settings(agent_mod)
    monkeypatch.setenv(settings["append_command_env"], command)
    monkeypatch.setenv(settings["host_name_env"], "lab-host")
    (tmp_path / "agent-state").mkdir(exist_ok=True)
    reading._refs.clear()
    reading._not_routed_read.clear()
    return sink


def _cells(row: str) -> dict[str, str]:
    parts = [part.strip() for part in row.split(" | ")]
    return dict(part.split("=", 1) for part in parts[2:] if "=" in part)


def _failed_head_rows(sink: Path) -> list[dict[str, str]]:
    if not sink.exists():
        return []
    return [
        _cells(line)
        for line in sink.read_text(encoding="utf-8").splitlines()
        if "src=build-decision" in line and "reason=failed_head" in line
    ]


def _seed(tmp_path: Path, *records: dict[str, Any]) -> None:
    """Job records as the agent wrote them, on disk where a restart finds them."""
    jobs = tmp_path / "jobs"
    jobs.mkdir(exist_ok=True)
    for record in records:
        job = JobState.model_validate(record)
        (jobs / f"{job.correlation_id}.json").write_text(job.model_dump_json())


def _failed_record(
    *,
    job_id: UUID | None = None,
    requested_by: str = IDLE_RESTORE_REQUESTER,
    head: str = FAILED_HEAD,
    verdict: str = "FAIL [runtime_degraded] runtime health DEGRADED",
    completed: datetime = datetime(2026, 11, 20, 19, 58, tzinfo=UTC),
    lane: str = "dev",
) -> dict[str, Any]:
    """A failed dev job in the shape of the recorded ones: the commit, then the verdict."""
    template = RECORDED["jobs"][0]["job"]
    job_id = job_id or uuid4()
    return {
        **template,
        "correlation_id": str(job_id),
        "command": {
            **template["command"],
            "correlation_id": str(job_id),
            "requested_by": requested_by,
            "runtime_lane": lane,
        },
        "accepted_at": (completed - timedelta(minutes=58)).isoformat(),
        "completed_at": completed.isoformat(),
        "errors": [f"commit={head}: RuntimeError: Runtime roll stopped: {verdict}"],
    }


def _agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    now: datetime,
    head: str = FAILED_HEAD,
    presence: EnumRuntimePresence = EnumRuntimePresence.DOWN,
) -> DeployAgent:
    """The idle-branch harness of OMN-19509, on a lane whose runtime is ``presence``."""
    agent = _idle_agent(tmp_path, monkeypatch, omnimarket_to_202=False)
    agent.executor = cast("Any", _FakeExecutor())
    agent._idle_converge_started_at = AGENT_START
    agent._idle_converge_now = lambda: now
    agent._idle_read_head_ref = lambda repo: head
    agent._idle_read_running_ref = lambda repo: "a" * 40
    agent._idle_read_runtime_presence = lambda: (presence, f"runtime is {presence}")
    return agent


def _follow_infra_elsewhere(agent: DeployAgent) -> None:
    """The routing table sends omnibase_infra rebuilds to another instance (OMN-20867)."""
    assert agent._router is not None
    table = agent._router.table
    elsewhere = next(n for n in table.instances if n != agent._router.instance.name)
    routed = table.model_copy(
        update={
            "routes": (
                route := ModelRoute(
                    runtime_lane=EnumRuntimeLane.DEV,
                    requester_repository="omnibase_infra",
                    instance=elsewhere,
                ),
            ),
            "instances": {
                **table.instances,
                route.instance: table.instances[route.instance].model_copy(
                    update={"available": True}
                ),
            },
        }
    )
    agent._router = DeployRouter(routed, resolve_instance(routed), lambda sha: routed)


def _tick(agent: DeployAgent, now: datetime) -> None:
    """One idle-branch check at ``now``, past the interval throttle."""
    agent._idle_converge_now = lambda: now
    agent._idle_converge_last_check = None
    agent._maybe_idle_converge()


def _agent_jobs(tmp_path: Path, requester: str) -> list[JobState]:
    store = JobStore(tmp_path / "jobs")
    jobs = [
        store.load(UUID(p.stem)) for p in sorted((tmp_path / "jobs").glob("*.json"))
    ]
    return [j for j in jobs if j is not None and j.command["requested_by"] == requester]


def _signed(requested_by: str) -> ModelRebuildRequested:
    return ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by=requested_by,
        scope=Scope.RUNTIME,
        runtime_lane=EnumRuntimeLane.DEV,
        build_source=BuildSource.WORKSPACE,
    )


# --------------------------------------------------------------------------- #
# AC1                                                                         #
# --------------------------------------------------------------------------- #
class TestIdleSkipsFailedHead:
    def test_idle_skips_failed_head_restore(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ledger: Path
    ) -> None:
        _seed(tmp_path, _failed_record())
        agent = _agent(tmp_path, monkeypatch, now=SECOND_RESTORE)

        _tick(agent, SECOND_RESTORE)

        assert len(_agent_jobs(tmp_path, IDLE_RESTORE_REQUESTER)) == 1, (
            "a restore of the head that just failed started a second job"
        )

    def test_idle_skips_failed_head_restore_accepts_a_newer_head(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ledger: Path
    ) -> None:
        _seed(tmp_path, _failed_record())
        agent = _agent(tmp_path, monkeypatch, now=SECOND_RESTORE, head=NEWER_HEAD)

        _tick(agent, SECOND_RESTORE)

        assert len(_agent_jobs(tmp_path, IDLE_RESTORE_REQUESTER)) == 2
        assert _failed_head_rows(ledger) == []

    def test_idle_skips_failed_head_converge(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ledger: Path
    ) -> None:
        _seed(tmp_path, _failed_record(requested_by=IDLE_CONVERGE_REQUESTER))
        agent = _agent(
            tmp_path,
            monkeypatch,
            now=SECOND_RESTORE,
            presence=EnumRuntimePresence.RUNNING,
        )
        _follow_infra_elsewhere(agent)

        _tick(agent, SECOND_RESTORE)

        assert len(_agent_jobs(tmp_path, IDLE_CONVERGE_REQUESTER)) == 1, (
            "a converge on the head that just failed started a second job"
        )

    def test_idle_skips_failed_head_converge_accepts_a_newer_head(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ledger: Path
    ) -> None:
        _seed(tmp_path, _failed_record(requested_by=IDLE_CONVERGE_REQUESTER))
        agent = _agent(
            tmp_path,
            monkeypatch,
            now=SECOND_RESTORE,
            head=NEWER_HEAD,
            presence=EnumRuntimePresence.RUNNING,
        )
        _follow_infra_elsewhere(agent)

        _tick(agent, SECOND_RESTORE)

        assert len(_agent_jobs(tmp_path, IDLE_CONVERGE_REQUESTER)) == 2

    def test_idle_skips_failed_head_only_for_a_recorded_verdict_class(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ledger: Path
    ) -> None:
        """A failure with no recorded verdict class is not a deterministic one."""
        _seed(tmp_path, _failed_record(verdict="the docker daemon went away"))
        agent = _agent(tmp_path, monkeypatch, now=SECOND_RESTORE)

        _tick(agent, SECOND_RESTORE)

        assert len(_agent_jobs(tmp_path, IDLE_RESTORE_REQUESTER)) == 2

    @pytest.mark.parametrize(
        ("verdict", "recorded"),
        [
            ("FAIL [runtime_degraded] runtime health DEGRADED", "runtime_degraded"),
            ("FAIL [http_error] health endpoint returned HTTP 503", "http_error"),
            (
                "FAIL [probe_unreachable] health endpoint unreachable",
                "probe_unreachable",
            ),
            ("BOOT FAILED 3/3: EvalResultsProjectionWriter", "boot_failed"),
            (
                "Runtime roll stopped at omninode-runtime: compose up timed out",
                "compose_up_timed_out",
            ),
        ],
    )
    def test_idle_skips_failed_head_for_each_recorded_verdict_class(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ledger: Path,
        verdict: str,
        recorded: str,
    ) -> None:
        _seed(tmp_path, _failed_record(verdict=verdict))
        agent = _agent(tmp_path, monkeypatch, now=SECOND_RESTORE)

        _tick(agent, SECOND_RESTORE)

        assert len(_agent_jobs(tmp_path, IDLE_RESTORE_REQUESTER)) == 1
        assert [row["verdict_class"] for row in _failed_head_rows(ledger)] == [recorded]

    def test_idle_skips_failed_head_ignores_a_failed_job_of_another_lane(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ledger: Path
    ) -> None:
        _seed(tmp_path, _failed_record(lane=OTHER_LANE))
        agent = _agent(tmp_path, monkeypatch, now=SECOND_RESTORE)

        _tick(agent, SECOND_RESTORE)

        assert len(_agent_jobs(tmp_path, IDLE_RESTORE_REQUESTER)) == 2

    def test_idle_skips_failed_head_lifts_once_a_later_job_passed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ledger: Path
    ) -> None:
        passed = {
            **_failed_record(completed=datetime(2026, 11, 20, 20, 30, tzinfo=UTC)),
            "status": "success",
            "errors": [],
        }
        _seed(tmp_path, _failed_record(), passed)
        agent = _agent(tmp_path, monkeypatch, now=SECOND_RESTORE)

        _tick(agent, SECOND_RESTORE)

        assert len(_agent_jobs(tmp_path, IDLE_RESTORE_REQUESTER)) == 3

    def test_idle_skips_failed_head_with_the_head_unread_starts_the_restore(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ledger: Path
    ) -> None:
        """An unread head proves nothing about the head: the restore keeps its own rule."""
        _seed(tmp_path, _failed_record())
        agent = _agent(tmp_path, monkeypatch, now=SECOND_RESTORE)
        agent._idle_read_head_ref = lambda repo: None

        _tick(agent, SECOND_RESTORE)

        assert len(_agent_jobs(tmp_path, IDLE_RESTORE_REQUESTER)) == 2


# --------------------------------------------------------------------------- #
# AC2                                                                         #
# --------------------------------------------------------------------------- #
class TestIdleFailedHeadLedgerRow:
    def test_idle_failed_head_ledger_row_names_head_job_and_class(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ledger: Path
    ) -> None:
        failed = _failed_record()
        _seed(tmp_path, failed)
        agent = _agent(tmp_path, monkeypatch, now=SECOND_RESTORE)

        _tick(agent, SECOND_RESTORE)

        (row,) = _failed_head_rows(ledger)
        assert row["trigger_sha"] == FAILED_HEAD
        assert row["job"] == failed["correlation_id"]
        assert row["verdict_class"] == "runtime_degraded"
        assert row["result"] == "refused"
        assert row["repo"] == "omnibase_infra"

    def test_idle_failed_head_ledger_row_is_one_per_head_not_one_per_tick(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ledger: Path
    ) -> None:
        _seed(tmp_path, _failed_record())
        agent = _agent(tmp_path, monkeypatch, now=SECOND_RESTORE)

        for minute in range(30):
            _tick(agent, SECOND_RESTORE + timedelta(minutes=minute))

        assert len(_failed_head_rows(ledger)) == 1

    def test_idle_failed_head_ledger_row_is_not_repeated_after_a_restart(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ledger: Path
    ) -> None:
        """The once-per-head mark lives in the agent's state directory, not in memory."""
        _seed(tmp_path, _failed_record())
        first = _agent(tmp_path, monkeypatch, now=SECOND_RESTORE)
        _tick(first, SECOND_RESTORE)
        assert len(_failed_head_rows(ledger)) == 1

        restarted = _agent(tmp_path, monkeypatch, now=SECOND_RESTORE)
        _tick(restarted, SECOND_RESTORE + timedelta(minutes=5))

        assert len(_failed_head_rows(ledger)) == 1

    def test_idle_failed_head_ledger_row_is_one_per_restore_and_converge_together(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ledger: Path
    ) -> None:
        """Both idle paths refuse the same head; the ledger names it once."""
        _seed(tmp_path, _failed_record())
        agent = _agent(tmp_path, monkeypatch, now=SECOND_RESTORE)
        _follow_infra_elsewhere(agent)
        _tick(agent, SECOND_RESTORE)
        agent._idle_read_runtime_presence = lambda: (
            EnumRuntimePresence.RUNNING,
            "omninode-runtime is running",
        )
        _tick(agent, SECOND_RESTORE + timedelta(minutes=1))

        assert len(_failed_head_rows(ledger)) == 1
        assert len(_agent_jobs(tmp_path, IDLE_CONVERGE_REQUESTER)) == 0

    def test_idle_failed_head_ledger_row_is_one_for_each_head(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ledger: Path
    ) -> None:
        second = "e" * 40
        _seed(tmp_path, _failed_record())
        agent = _agent(tmp_path, monkeypatch, now=SECOND_RESTORE)
        _tick(agent, SECOND_RESTORE)
        _seed(
            tmp_path,
            _failed_record(
                head=second, completed=datetime(2026, 11, 20, 22, 0, tzinfo=UTC)
            ),
        )
        agent._idle_read_head_ref = lambda repo: second
        _tick(agent, SECOND_RESTORE + timedelta(hours=2))

        assert [row["trigger_sha"] for row in _failed_head_rows(ledger)] == [
            FAILED_HEAD,
            second,
        ]


# --------------------------------------------------------------------------- #
# AC3                                                                         #
# --------------------------------------------------------------------------- #
class TestSignedRequestBypassesFailedHead:
    @pytest.mark.parametrize(
        "requester",
        [
            "operator/pinned-fix/devwindow2-2215",
            "lane/idleskip-0030",
            "gha/omnibase_infra/runtime-rebuild",
        ],
    )
    def test_signed_request_bypasses_failed_head(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ledger: Path,
        requester: str,
    ) -> None:
        _seed(tmp_path, _failed_record())
        agent = _agent(tmp_path, monkeypatch, now=SECOND_RESTORE)
        _tick(agent, SECOND_RESTORE)
        assert len(_agent_jobs(tmp_path, IDLE_RESTORE_REQUESTER)) == 1, (
            "the idle restore must be refused for the head that failed"
        )

        cmd = _signed(requester)
        agent.job_store.accept(cmd.correlation_id, cmd.model_dump(mode="json"))
        agent._execute_command(cmd)

        signed = _agent_jobs(tmp_path, requester)
        assert [job.correlation_id for job in signed] == [cmd.correlation_id]
        assert signed[0].status == "success"
        assert "rebuild_scope" in cast("_FakeExecutor", agent.executor).calls

    def test_signed_request_bypasses_failed_head_gate_by_requester(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ledger: Path
    ) -> None:
        _seed(tmp_path, _failed_record())
        agent = _agent(tmp_path, monkeypatch, now=SECOND_RESTORE)

        refused = [
            agent._idle_build_gate.refuses(_signed(requester))
            for requester in (IDLE_RESTORE_REQUESTER, IDLE_CONVERGE_REQUESTER)
        ]
        accepted = [
            agent._idle_build_gate.refuses(_signed(requester))
            for requester in ("operator/pinned-fix/devwindow2-2215", "lane/x")
        ]

        assert refused == [True, True]
        assert accepted == [False, False]

    def test_signed_request_bypasses_failed_head_names_the_agents_idle_requesters(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ledger: Path
    ) -> None:
        """The gate's two spellings are the agent's own, so a rename cannot unguard a path."""
        agent = _agent(tmp_path, monkeypatch, now=SECOND_RESTORE)
        rule = sys.modules[
            "omnibase_infra.nodes.node_deploy_job_reading_effect.runtime_failed_head_gate"
        ]

        assert {IDLE_RESTORE_REQUESTER, IDLE_CONVERGE_REQUESTER} == rule.IDLE_REQUESTERS
        assert rule.INFRA_REPOSITORY == "omnibase_infra"


# --------------------------------------------------------------------------- #
# AC4                                                                         #
# --------------------------------------------------------------------------- #
class TestFailedHeadIncidentReplay:
    def test_the_fixture_is_the_three_jobs_of_head_7b25f0ff(self) -> None:
        jobs = RECORDED["jobs"]
        assert [j["job"]["correlation_id"][:8] for j in jobs] == [
            "5623655f",
            "8648de6f",
            "a9ecbe18",
        ]
        assert FAILED_HEAD.startswith("7b25f0ff") and len(FAILED_HEAD) == 40
        for entry in jobs:
            assert FAILED_HEAD in entry["job"]["errors"][0]
            assert entry["job"]["status"] == "failed"

    def test_failed_head_incident_replay(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ledger: Path
    ) -> None:
        """Replay 19:00Z to 24:00Z on 2026-11-20: the lane is down, dev stays on 7b25f0ff.

        The first job (5623655f) is on disk when the lane's second idle restore comes due
        (21:00:33Z, where 8648de6f started on the host). The agent refuses it and every
        later check; the operator's pinned-fix job (a9ecbe18) is accepted at 23:05:23Z and
        fails; a newer head then lets the next restore through.
        """
        first, _, signed = (entry["job"] for entry in RECORDED["jobs"])
        _seed(tmp_path, first)
        agent = _agent(tmp_path, monkeypatch, now=SECOND_RESTORE)

        # 21:00:33Z to 23:05:22Z, one check a minute: every one refused.
        now = SECOND_RESTORE
        while now < datetime(2026, 11, 20, 23, 5, 23, tzinfo=UTC):
            _tick(agent, now)
            now += timedelta(minutes=1)
        assert [
            j.correlation_id for j in _agent_jobs(tmp_path, IDLE_RESTORE_REQUESTER)
        ] == [UUID(first["correlation_id"])], (
            "the agent rebuilt head 7b25f0ff again (8648de6f on the host)"
        )

        # The operator's signed pinned fix runs for the same head and fails at the health gate.
        _seed(tmp_path, signed)
        for minute in range(0, 180, 3):
            _tick(
                agent,
                datetime(2026, 11, 20, 23, 25, tzinfo=UTC) + timedelta(minutes=minute),
            )
        assert len(_agent_jobs(tmp_path, IDLE_RESTORE_REQUESTER)) == 1

        (row,) = _failed_head_rows(ledger)
        assert row["trigger_sha"] == FAILED_HEAD
        assert row["job"] == first["correlation_id"]
        assert row["verdict_class"] == "runtime_degraded"

        # A newer head is a new head: the next restore runs.
        agent._idle_read_head_ref = lambda repo: NEWER_HEAD
        _tick(agent, datetime(2026, 11, 21, 3, 0, tzinfo=UTC))
        assert len(_agent_jobs(tmp_path, IDLE_RESTORE_REQUESTER)) == 2
