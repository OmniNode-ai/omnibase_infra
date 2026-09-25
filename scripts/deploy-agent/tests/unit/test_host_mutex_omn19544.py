# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19544 AC1: a deploy-agent job and a prover never hold one lab host at once.

The .105 host's Docker VM holds the dev-105 lane or the prove-105 proof stack,
not both. The two tenants share one lease file on the host
(``deploy_agent/host_slot.py``):

* the lease itself: one owner at a time, an absolute ``until``, never stolen,
  and exactly one winner when several owners race for it;
* the prover's side, the command line, which refuses while the agent holds the
  slot and names the holder;
* the agent's side: the consumer refuses to accept a command while a prover
  holds the slot (reason ``busy``), a job takes the slot before its first phase
  and keeps it for the verify window after its last, and a job that finds the
  slot taken touches nothing;
* the default: an instance without ``DEPLOY_AGENT_HOST_SLOT_DIR`` has no slot
  and behaves exactly as before.

Every test name carries ``host_mutex`` so the ticket's falsifier
(``pytest scripts/deploy-agent/tests/unit -k host_mutex``) selects them.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent import host_slot as host_slot_mod
from deploy_agent.agent import DeployAgent
from deploy_agent.consumer import DeployConsumer
from deploy_agent.events import (
    EnumRejectionReason,
    EnumRuntimeLane,
    ModelRebuildRequested,
    Scope,
)
from deploy_agent.host_slot import (
    ENV_HOST_SLOT_DIR,
    ENV_HOST_SLOT_VERIFY_WINDOW,
    HostSlot,
    HostSlotHeldError,
    host_slot_from_env,
    job_lease,
    verify_window_from_env,
)
from deploy_agent.job_state import JobStore

pytestmark = pytest.mark.unit

T0 = datetime(2026, 9, 25, 11, 0, tzinfo=UTC)
AGENT = "deploy-agent-dev-201"
PROVER = "prove-105"
SHA = "a" * 40
HOST_SLOT_PY = Path(host_slot_mod.__file__)


class _Clock:
    def __init__(self, start: datetime = T0) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now = self.now + timedelta(**kwargs)


def _slot(tmp_path: Path, clock: _Clock | None = None) -> HostSlot:
    return HostSlot(tmp_path / "host-slot", clock=clock or _Clock())


# --- the lease ----------------------------------------------------------------


def test_host_mutex_a_prover_lease_refuses_the_agent(tmp_path: Path) -> None:
    clock = _Clock()
    slot = _slot(tmp_path, clock)
    slot.acquire(PROVER, T0 + timedelta(hours=2), "2026-09-25T10:26:00Z-prove-105")

    with pytest.raises(HostSlotHeldError) as info:
        slot.acquire(AGENT, T0 + timedelta(hours=1))

    assert info.value.holder.owner == PROVER
    assert "2026-09-25T10:26:00Z-prove-105" in str(info.value)
    assert slot.held_by_other(AGENT) is not None
    assert slot.held_by_other(PROVER) is None, "a holder is not blocked by itself"


def test_host_mutex_an_expired_lease_is_not_in_force(tmp_path: Path) -> None:
    clock = _Clock()
    slot = _slot(tmp_path, clock)
    slot.acquire(PROVER, T0 + timedelta(minutes=20))
    clock.advance(minutes=20)

    assert slot.holder() is None
    lease = slot.acquire(AGENT, clock.now + timedelta(hours=1))
    assert lease.owner == AGENT
    assert lease.acquired_at == "2026-09-25T11:20:00Z"


def test_host_mutex_the_same_owner_extends_its_own_lease(tmp_path: Path) -> None:
    clock = _Clock()
    slot = _slot(tmp_path, clock)
    slot.acquire(AGENT, T0 + timedelta(hours=3), "deploy job")
    clock.advance(minutes=40)
    lease = slot.acquire(AGENT, clock.now + timedelta(minutes=30), "verify window")

    assert lease.acquired_at == "2026-09-25T11:00:00Z", "the tenure is kept"
    assert lease.until == "2026-09-25T12:10:00Z"
    assert lease.reason == "verify window"


def test_host_mutex_release_never_removes_another_owner(tmp_path: Path) -> None:
    slot = _slot(tmp_path)
    slot.acquire(PROVER, T0 + timedelta(hours=1))

    assert slot.release(AGENT) is False
    assert slot.holder() is not None and slot.holder().owner == PROVER  # type: ignore[union-attr]
    assert slot.release(PROVER) is True
    assert slot.holder() is None


def test_host_mutex_refuses_an_until_that_is_not_in_the_future(tmp_path: Path) -> None:
    slot = _slot(tmp_path)
    with pytest.raises(ValueError, match="not after now"):
        slot.acquire(PROVER, T0)
    with pytest.raises(ValueError, match="owner"):
        slot.acquire(" ", T0 + timedelta(hours=1))


def test_host_mutex_racing_owners_get_exactly_one_winner(tmp_path: Path) -> None:
    """Each thread opens its own HostSlot, so each flock is its own open file
    description: the same contention two processes on the host would have."""
    directory = tmp_path / "host-slot"
    barrier = threading.Barrier(8)
    winners: list[str] = []
    refused: list[str] = []

    def contend(owner: str) -> None:
        slot = HostSlot(directory, clock=_Clock())
        barrier.wait()
        try:
            slot.acquire(owner, T0 + timedelta(hours=1))
            winners.append(owner)
        except HostSlotHeldError:
            refused.append(owner)

    threads = [threading.Thread(target=contend, args=(f"owner-{i}",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(winners) == 1, winners
    assert len(refused) == 7
    lease = json.loads((directory / "lease.json").read_text(encoding="utf-8"))
    assert lease["owner"] == winners[0]


# --- the agent's job lease ------------------------------------------------------


def test_host_mutex_a_job_holds_the_slot_then_its_verify_window(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    slot = _slot(tmp_path, clock)

    with job_lease(slot, AGENT, verify_window=timedelta(minutes=30)):
        with pytest.raises(HostSlotHeldError):
            slot.acquire(PROVER, T0 + timedelta(hours=2))
        clock.advance(minutes=25)

    lease = slot.holder()
    assert lease is not None and lease.owner == AGENT
    assert lease.until == "2026-09-25T11:55:00Z", "30 minutes after the job ended"
    clock.advance(minutes=29)
    with pytest.raises(HostSlotHeldError):
        slot.acquire(PROVER, clock.now + timedelta(hours=1))
    clock.advance(minutes=1)
    assert slot.acquire(PROVER, clock.now + timedelta(hours=1)).owner == PROVER


def test_host_mutex_a_failed_job_still_keeps_its_verify_window(
    tmp_path: Path,
) -> None:
    slot = _slot(tmp_path)
    with pytest.raises(RuntimeError, match="build"):
        with job_lease(slot, AGENT, verify_window=timedelta(minutes=30)):
            raise RuntimeError("build failed")
    lease = slot.holder()
    assert lease is not None and lease.owner == AGENT


def test_host_mutex_a_zero_verify_window_releases_at_job_end(tmp_path: Path) -> None:
    slot = _slot(tmp_path)
    with job_lease(slot, AGENT, verify_window=timedelta(0)):
        assert slot.holder() is not None
    assert slot.holder() is None


def test_host_mutex_a_job_that_finds_the_slot_taken_runs_nothing(
    tmp_path: Path,
) -> None:
    slot = _slot(tmp_path)
    slot.acquire(PROVER, T0 + timedelta(hours=1))
    ran: list[str] = []

    with pytest.raises(HostSlotHeldError):
        with job_lease(slot, AGENT, verify_window=timedelta(minutes=30)):
            ran.append("phase")

    assert ran == []
    assert slot.holder().owner == PROVER  # type: ignore[union-attr]


def test_host_mutex_no_slot_means_no_lease(tmp_path: Path) -> None:
    with job_lease(None, AGENT, verify_window=timedelta(minutes=30)):
        pass
    assert host_slot_from_env({}) is None
    assert host_slot_from_env({ENV_HOST_SLOT_DIR: "  "}) is None
    slot = host_slot_from_env({ENV_HOST_SLOT_DIR: str(tmp_path)})
    assert slot is not None and slot.directory == tmp_path


def test_host_mutex_verify_window_from_env() -> None:
    assert verify_window_from_env({}) == timedelta(minutes=30)
    assert verify_window_from_env({ENV_HOST_SLOT_VERIFY_WINDOW: "600"}) == timedelta(
        minutes=10
    )
    with pytest.raises(ValueError, match="negative"):
        verify_window_from_env({ENV_HOST_SLOT_VERIFY_WINDOW: "-1"})


# --- the prover's command line ---------------------------------------------------


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HOST_SLOT_PY), *args],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env={"PATH": "/usr/bin:/bin"},
    )


def test_host_mutex_cli_refuses_a_prover_while_the_agent_holds_the_slot(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "host-slot"
    until = (datetime.now(UTC) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    HostSlot(directory).acquire(AGENT, datetime.now(UTC) + timedelta(hours=1))

    refused = _cli(
        "--dir", str(directory), "acquire", "--owner", PROVER, "--until", until
    )
    assert refused.returncode == 2, refused
    assert f"held by {AGENT}" in refused.stderr

    assert HostSlot(directory).release(AGENT)
    taken = _cli(
        "--dir",
        str(directory),
        "acquire",
        "--owner",
        PROVER,
        "--until",
        until,
        "--reason",
        "2026-09-25T10:26:00Z-prove-105",
    )
    assert taken.returncode == 0, taken
    assert json.loads(taken.stdout)["owner"] == PROVER

    shown = _cli("--dir", str(directory), "show")
    assert json.loads(shown.stdout)["reason"] == "2026-09-25T10:26:00Z-prove-105"
    released = _cli("--dir", str(directory), "release", "--owner", PROVER)
    assert released.returncode == 0 and "released" in released.stdout
    assert _cli("--dir", str(directory), "show").stdout.strip() == "host_slot: free"


def test_host_mutex_cli_needs_a_directory_and_an_absolute_utc_until(
    tmp_path: Path,
) -> None:
    missing = _cli("show")
    assert missing.returncode == 3
    assert ENV_HOST_SLOT_DIR in missing.stderr
    naive = _cli(
        "--dir",
        str(tmp_path),
        "acquire",
        "--owner",
        PROVER,
        "--until",
        "2099-01-01T00:00:00",
    )
    assert naive.returncode == 3
    assert "timezone" in naive.stderr


# --- the consumer: refuse to accept while a prover holds the slot ----------------


def _command() -> ModelRebuildRequested:
    return ModelRebuildRequested.model_validate(
        {
            "correlation_id": str(uuid4()),
            "requested_by": "gha/omnimarket/pr-2890",
            "scope": "full",
            "runtime_lane": "dev",
            "build_source": "workspace",
            "git_ref": SHA,
        }
    )


def _message(cmd: ModelRebuildRequested) -> SimpleNamespace:
    payload = cmd.model_dump(mode="json") | {"_signature": "a" * 64}
    return SimpleNamespace(
        value=payload, topic="t", partition=0, offset=100, key=None, timestamp=None
    )


def _consumer(slot: HostSlot | None) -> DeployConsumer:
    consumer = DeployConsumer.__new__(DeployConsumer)
    consumer.consumer = Mock()
    consumer.job_store = Mock()
    consumer.job_store.has_active_job.return_value = False
    consumer.job_store.is_duplicate.return_value = False
    consumer.allowed_lanes = frozenset({EnumRuntimeLane.DEV})
    consumer.self_update_hook = lambda rewind: None
    consumer.notices = []  # type: ignore[attr-defined]
    consumer.on_rejected = consumer.notices.append  # type: ignore[attr-defined]
    consumer.host_slot = slot
    consumer.host_slot_owner = AGENT
    return consumer


def _process(consumer: DeployConsumer, cmd: ModelRebuildRequested) -> tuple:
    with patch("deploy_agent.consumer.verify_command", return_value=True):
        return consumer._process_message(_message(cmd))


def test_host_mutex_consumer_refuses_while_a_prover_holds_the_slot(
    tmp_path: Path,
) -> None:
    slot = HostSlot(tmp_path / "host-slot")
    slot.acquire(PROVER, datetime.now(UTC) + timedelta(hours=1))
    consumer = _consumer(slot)

    accepted, reason = _process(consumer, _command())

    assert accepted is None
    assert reason == EnumRejectionReason.BUSY
    consumer.job_store.accept.assert_not_called()
    assert [n.reason for n in consumer.notices] == [EnumRejectionReason.BUSY]  # type: ignore[attr-defined]
    committed = [
        meta.offset
        for call in consumer.consumer.commit.call_args_list
        for meta in call.args[0].values()
    ]
    assert committed == [101], "a refusal commits past the record, like busy"


def test_host_mutex_consumer_accepts_when_the_slot_is_free_or_its_own(
    tmp_path: Path,
) -> None:
    slot = HostSlot(tmp_path / "host-slot")
    cmd = _command()
    assert _process(_consumer(slot), cmd) == (cmd, None)

    slot.acquire(AGENT, datetime.now(UTC) + timedelta(minutes=30), "verify window")
    cmd2 = _command()
    assert _process(_consumer(slot), cmd2) == (cmd2, None)


def test_host_mutex_consumer_without_a_slot_is_unchanged() -> None:
    cmd = _command()
    assert _process(_consumer(None), cmd) == (cmd, None)
    bare = DeployConsumer.__new__(DeployConsumer)
    assert bare.host_slot is None, "declared on the class, off by default"


# --- the agent: a job takes the slot, or touches nothing ------------------------


class _FakeExecutor:
    def __init__(self, slot: HostSlot | None = None) -> None:
        self.calls: list[str] = []
        self.slot = slot
        self.holder_during_build: str | None = None
        self.container_residue: list[object] = []
        self.sibling_source_refs: dict[str, str] = {}
        self.recreate_supervision: list[object] = []
        self.verify_recreate: list[object] = []
        self.deps_convergence: list[object] = []
        self.compose_invocations: list[object] = []
        self.health_checks: list[object] = []

    def preflight(self, **kwargs: object) -> None:
        self.calls.append("preflight")

    def git_pull(self, git_ref: str, **kwargs: object) -> str:
        self.calls.append("git_pull")
        return SHA

    def compose_gen(self, bundles: list[str], **kwargs: object) -> None:
        self.calls.append("compose_gen")

    def seed_infisical(self, **kwargs: object) -> None:
        self.calls.append("seed_infisical")

    def validate_llm_endpoint_env_contract(self) -> None:
        self.calls.append("validate_llm_endpoint_env_contract")

    def rebuild_scope(self, *args: object, **kwargs: object) -> list[str]:
        self.calls.append("rebuild_scope")
        if self.slot is not None:
            holder = self.slot.holder()
            self.holder_during_build = holder.owner if holder else None
        return ["omninode-runtime"]

    def verify(self, **kwargs: object) -> list[object]:
        self.calls.append("verify")
        return []


def _agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, slot_dir: Path | None
) -> tuple[DeployAgent, JobStore, ModelRebuildRequested]:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    if slot_dir is None:
        monkeypatch.delenv(ENV_HOST_SLOT_DIR, raising=False)
    else:
        monkeypatch.setenv(ENV_HOST_SLOT_DIR, str(slot_dir))
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "agent-state")
    monkeypatch.setattr(agent_mod, "publish_result", lambda payload, config: False)
    monkeypatch.setattr(agent_mod, "LAB_OVERLAY_ENABLED", False)
    cmd = ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by="test",
        scope=Scope.RUNTIME,
        runtime_lane=EnumRuntimeLane.DEV,
    )
    store = JobStore(tmp_path / "jobs")
    store.accept(cmd.correlation_id, cmd.model_dump(mode="json"))
    agent = DeployAgent(skip_self_update=True)
    agent.job_store = store
    return agent, store, cmd


def test_host_mutex_agent_job_holds_the_slot_through_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slot_dir = tmp_path / "host-slot"
    agent, store, cmd = _agent(tmp_path, monkeypatch, slot_dir=slot_dir)
    fake = _FakeExecutor(HostSlot(slot_dir))
    agent.executor = fake  # type: ignore[assignment]

    agent._run_deploy(cmd)

    job = store.load(cmd.correlation_id)
    assert job is not None and job.status == "success"
    assert fake.holder_during_build == "deploy-agent-dev-201"
    after = HostSlot(slot_dir).holder()
    assert after is not None and after.owner == "deploy-agent-dev-201"
    assert after.reason.startswith("verify window"), after


def test_host_mutex_agent_job_touches_nothing_while_a_prover_holds_the_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slot_dir = tmp_path / "host-slot"
    HostSlot(slot_dir).acquire(PROVER, datetime.now(UTC) + timedelta(hours=1))
    agent, store, cmd = _agent(tmp_path, monkeypatch, slot_dir=slot_dir)
    fake = _FakeExecutor()
    agent.executor = fake  # type: ignore[assignment]

    agent._run_deploy(cmd)

    assert fake.calls == [], "no phase runs, not even preflight"
    job = store.load(cmd.correlation_id)
    assert job is not None and job.status == "failed"
    assert any(f"held by {PROVER}" in e for e in job.errors), job.errors
    assert HostSlot(slot_dir).holder().owner == PROVER  # type: ignore[union-attr]


def test_host_mutex_agent_without_a_slot_runs_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, store, cmd = _agent(tmp_path, monkeypatch, slot_dir=None)
    fake = _FakeExecutor()
    agent.executor = fake  # type: ignore[assignment]

    agent._run_deploy(cmd)

    job = store.load(cmd.correlation_id)
    assert job is not None and job.status == "success"
    assert "rebuild_scope" in fake.calls
    assert not (tmp_path / "host-slot").exists()
