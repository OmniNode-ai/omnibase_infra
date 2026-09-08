# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Self-update fires only at a job boundary, never between deploy phases (OMN-16442).

Live evidence this file encodes, command
``8d0c861a-f91e-4ca2-954e-a073759dd39d`` on the .201 dev lane,
2026-09-08T16:01:27Z-16:01:39Z: ``preflight``/``git``/``compose_gen``/``seed``
all SUCCESS, then ``self_update: behind origin/dev (local=e6e1f2db6b7b
remote=e04eb9adfed8), pulling and re-execing``, then ``Recovered 1 crashed
job(s)`` from the replacement process and the command published as
``status=failed``. ``self_update`` was called as the first statement of
``rebuild_scope``, i.e. mid-deploy after the seed phase; a process that
re-execs there cannot finish the deploy it was executing.

Every test below is red against that ordering and green against the boundary
ordering. The two boundaries are:

``pre_accept``
    inside ``DeployConsumer._process_message``, after the signature, payload,
    lane-fence, busy and dedup checks and BEFORE ``job_store.accept`` -- so a
    re-exec there orphans no job, and the command, whose offset is rewound
    rather than committed, is re-read and processed once by the new image.

``post_terminal``
    inside ``DeployAgent._execute_command``, after the single-flight lock is
    released and the job's terminal status has been published.
"""

from __future__ import annotations

import inspect
import subprocess
import uuid
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from deploy_agent.agent import DeployAgent
from deploy_agent.consumer import DeployConsumer
from deploy_agent.events import (
    EnumRuntimeLane,
    EnumSelfUpdateBoundary,
    ModelRebuildRequested,
    Phase,
    PhaseStatus,
    Scope,
)
from deploy_agent.executor import DeployExecutor
from deploy_agent.job_state import JobStore

SHA_LOCAL = "aaaaaaaabbbbbbbb"
SHA_REMOTE = "ccccccccdddddddd"


def _ok(stdout: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def _git_responses(*, local: str, remote: str):
    """side_effect for executor._run: clean tree, successful fetch/rev-parse/pull."""

    def side_effect(
        cmd: list[str], timeout: int, **kwargs
    ) -> subprocess.CompletedProcess:
        if "status" in cmd and "--porcelain" in cmd:
            return _ok("")
        if "rev-parse" in cmd:
            if any(part.startswith("origin/") for part in cmd):
                return _ok(remote)
            return _ok(local)
        return _ok()

    return side_effect


def _payload(correlation_id: str) -> dict[str, object]:
    return {
        "correlation_id": correlation_id,
        "git_ref": "origin/dev",
        "requested_by": "ci-redeploy",
        "scope": "full",
        "runtime_lane": "dev",
        "services": [],
        "_signature": "a" * 64,
    }


def _message(payload: dict[str, object], *, offset: int = 7) -> SimpleNamespace:
    return SimpleNamespace(
        value=payload,
        topic="onex.cmd.deploy.rebuild-requested.v1",
        partition=0,
        offset=offset,
    )


def _consumer(job_store, hook) -> DeployConsumer:
    """A DeployConsumer with a mocked kafka client and a real boundary hook."""
    consumer = DeployConsumer.__new__(DeployConsumer)
    consumer.consumer = Mock()
    consumer.job_store = job_store
    consumer.allowed_lanes = frozenset({EnumRuntimeLane.DEV})
    consumer.self_update_hook = hook
    return consumer


# ---------------------------------------------------------------------------
# The defect itself: no self-update may originate inside the deploy path.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rebuild_scope_does_not_self_update() -> None:
    """RED against agent.py:219-228 / executor.py rebuild_scope's first statement."""
    executor = DeployExecutor()
    calls: list[object] = []

    def fake_self_update(**kwargs: object) -> None:
        calls.append(kwargs)

    executor.self_update = fake_self_update  # type: ignore[method-assign]
    executor._compose_build = lambda *a, **k: None  # type: ignore[method-assign]
    executor._compose_up = lambda *a, **k: None  # type: ignore[method-assign]

    executor.rebuild_scope(Scope.RUNTIME, [], lambda p, s: None)

    assert calls == [], (
        "rebuild_scope must not self-update: a process that re-execs mid-deploy "
        "cannot complete the deploy it is executing (command 8d0c861a)"
    )


@pytest.mark.unit
def test_rebuild_scope_takes_no_self_update_parameter() -> None:
    """The mid-deploy call site is removed, not merely defaulted off."""
    params = inspect.signature(DeployExecutor.rebuild_scope).parameters
    assert "skip_self_update" not in params, (
        "rebuild_scope still accepts skip_self_update, so the deploy path can "
        "still reach self_update"
    )


@pytest.mark.unit
def test_self_update_requires_its_caller_to_name_the_boundary() -> None:
    """Every call site declares where it fired; there is no default boundary."""
    boundary = inspect.signature(DeployExecutor.self_update).parameters["boundary"]
    assert boundary.default is inspect.Parameter.empty, (
        "boundary must be required so a future mid-deploy caller cannot omit it"
    )


# ---------------------------------------------------------------------------
# Boundary 1: behind at poll -> update, then process the same command once.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_behind_at_poll_updates_before_the_job_is_marked_started(tmp_path) -> None:
    """A command arriving while behind triggers update-then-process.

    ``os.execv`` is patched to raise ``SystemExit`` because that is what the
    real call does to this process image: it never returns. The assertions are
    that nothing was accepted before it fired, and that the offset was rewound
    to this message rather than committed past it.
    """
    job_store = JobStore(state_dir=tmp_path / "jobs")
    executor = DeployExecutor()
    consumer = _consumer(
        job_store,
        lambda rewind: executor.self_update(
            boundary=EnumSelfUpdateBoundary.PRE_ACCEPT, on_before_reexec=rewind
        ),
    )
    msg = _message(_payload(str(uuid.uuid4())), offset=7)

    with (
        patch("deploy_agent.consumer.verify_command", return_value=True),
        patch(
            "deploy_agent.executor._run",
            side_effect=_git_responses(local=SHA_LOCAL, remote=SHA_REMOTE),
        ),
        patch("os.execv", side_effect=SystemExit(0)),
        pytest.raises(SystemExit),
    ):
        consumer._process_message(msg)

    assert job_store.recover_crashed_jobs() == [], (
        "a self-caused re-exec must orphan no job -- 'Recovered N crashed job(s)' "
        "after a self-update is the defect this fix removes"
    )
    consumer.consumer.seek.assert_called_once()
    assert consumer.consumer.seek.call_args.args[1] == msg.offset, (
        "the committed position must point AT the un-processed command so the "
        "replacement process re-reads it"
    )


@pytest.mark.unit
def test_the_same_command_is_processed_exactly_once_across_the_reexec(tmp_path) -> None:
    """Redelivery after the update accepts the command once, not twice."""
    job_store = JobStore(state_dir=tmp_path / "jobs")
    executor = DeployExecutor()
    correlation_id = str(uuid.uuid4())
    msg = _message(_payload(correlation_id), offset=7)

    behind = _consumer(
        job_store,
        lambda rewind: executor.self_update(
            boundary=EnumSelfUpdateBoundary.PRE_ACCEPT, on_before_reexec=rewind
        ),
    )
    with (
        patch("deploy_agent.consumer.verify_command", return_value=True),
        patch(
            "deploy_agent.executor._run",
            side_effect=_git_responses(local=SHA_LOCAL, remote=SHA_REMOTE),
        ),
        patch("os.execv", side_effect=SystemExit(0)),
        pytest.raises(SystemExit),
    ):
        behind._process_message(msg)

    # The replacement process re-reads the same offset, now up to date.
    restarted = _consumer(
        job_store,
        lambda rewind: executor.self_update(
            boundary=EnumSelfUpdateBoundary.PRE_ACCEPT, on_before_reexec=rewind
        ),
    )
    with (
        patch("deploy_agent.consumer.verify_command", return_value=True),
        patch(
            "deploy_agent.executor._run",
            side_effect=_git_responses(local=SHA_REMOTE, remote=SHA_REMOTE),
        ),
        patch("os.execv") as execv_after,
    ):
        cmd, reason = restarted._process_message(msg)

    assert reason is None
    assert cmd is not None
    execv_after.assert_not_called()
    accepted = list((tmp_path / "jobs").glob("*.json"))
    assert len(accepted) == 1, f"expected exactly one accepted job, got {accepted}"
    assert correlation_id in accepted[0].name


@pytest.mark.unit
def test_up_to_date_at_poll_does_not_reexec_and_accepts_normally(tmp_path) -> None:
    job_store = JobStore(state_dir=tmp_path / "jobs")
    executor = DeployExecutor()
    consumer = _consumer(
        job_store,
        lambda rewind: executor.self_update(
            boundary=EnumSelfUpdateBoundary.PRE_ACCEPT, on_before_reexec=rewind
        ),
    )

    with (
        patch("deploy_agent.consumer.verify_command", return_value=True),
        patch(
            "deploy_agent.executor._run",
            side_effect=_git_responses(local=SHA_LOCAL, remote=SHA_LOCAL),
        ),
        patch("os.execv") as execv,
    ):
        cmd, reason = consumer._process_message(_message(_payload(str(uuid.uuid4()))))

    execv.assert_not_called()
    consumer.consumer.seek.assert_not_called()
    assert reason is None
    assert cmd is not None
    consumer.consumer.commit.assert_called_once()


# ---------------------------------------------------------------------------
# Boundary 2: behind during a deploy -> the deploy finishes, then the update.
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_behind_during_deploy_completes_the_deploy_then_updates(
    tmp_path, monkeypatch
) -> None:
    # The broker address is required and has no fallback (there is no
    # localhost default in this package); constructing a DeployAgent needs it
    # declared even though nothing here talks to a broker.
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    monkeypatch.setenv("DEPLOY_AGENT_LOCK_PATH", str(tmp_path / "deploy.lock"))
    monkeypatch.setattr("deploy_agent.lock._LOCK_PATH", tmp_path / "deploy.lock")
    monkeypatch.setattr("deploy_agent.agent.STATE_DIR", tmp_path / "jobs")

    agent = DeployAgent()
    agent.job_store = JobStore(state_dir=tmp_path / "jobs")
    order: list[str] = []

    def record_phase(name: str):
        def phase(*args: object, **kwargs: object):
            order.append(name)
            return [] if name == "verify" else ""

        return phase

    agent.executor.preflight = record_phase("preflight")  # type: ignore[method-assign]
    agent.executor.git_pull = record_phase("git")  # type: ignore[method-assign]
    agent.executor.compose_gen = record_phase("compose_gen")  # type: ignore[method-assign]
    agent.executor.seed_infisical = record_phase("seed")  # type: ignore[method-assign]
    agent.executor.validate_llm_endpoint_env_contract = record_phase("env_contract")  # type: ignore[method-assign]
    agent.executor.rebuild_scope = record_phase("rebuild")  # type: ignore[method-assign]
    agent.executor.verify = record_phase("verify")  # type: ignore[method-assign]

    def fake_self_update(
        *, boundary, skip: bool = False, on_before_reexec=None
    ) -> None:
        order.append(f"self_update:{boundary.value}")

    agent.executor.self_update = fake_self_update  # type: ignore[method-assign]

    cid = uuid.uuid4()
    agent.job_store.accept(correlation_id=cid, command=_payload(str(cid)))
    cmd = ModelRebuildRequested.model_validate(
        {k: v for k, v in _payload(str(cid)).items() if k != "_signature"}
    )

    with patch("deploy_agent.agent.publish_result", return_value=True):
        await agent._execute_command(cmd)

    job = agent.job_store.load(cid)
    assert job is not None
    assert job.status == "success", (
        f"deploy must complete, got {job.status} {job.errors}"
    )
    assert job.phase_results[Phase.PUBLISH] == PhaseStatus.SUCCESS
    assert order[-1] == f"self_update:{EnumSelfUpdateBoundary.POST_TERMINAL.value}", (
        f"the update must fire at the next boundary, after the terminal publish: {order}"
    )
    assert [step for step in order if step.startswith("self_update:")] == [
        f"self_update:{EnumSelfUpdateBoundary.POST_TERMINAL.value}"
    ], f"no self-update may fire between deploy phases: {order}"


# ---------------------------------------------------------------------------
# The journal line names the boundary it fired at.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "boundary",
    [EnumSelfUpdateBoundary.PRE_ACCEPT, EnumSelfUpdateBoundary.POST_TERMINAL],
)
def test_journal_line_names_the_boundary(caplog, boundary) -> None:
    executor = DeployExecutor()
    with (
        caplog.at_level("INFO", logger="deploy_agent.executor"),
        patch(
            "deploy_agent.executor._run",
            side_effect=_git_responses(local=SHA_LOCAL, remote=SHA_REMOTE),
        ),
        patch("os.execv"),
    ):
        executor.self_update(boundary=boundary)

    behind_lines = [
        record.getMessage()
        for record in caplog.records
        if "behind" in record.getMessage()
    ]
    assert behind_lines, "the pulling-and-re-execing line must be journalled"
    assert all(f"boundary={boundary.value}" in line for line in behind_lines), (
        behind_lines
    )
