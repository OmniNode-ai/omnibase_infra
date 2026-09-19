# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18636 AC5 -- a terminal job that is still being worked on must say so.

THE DEFECT THESE TESTS PIN
--------------------------
``_run_deploy`` writes the job's terminal verdict, and then keeps working on
that same job: the k3s lab-overlay apply (OMN-18200), the onex-api pin delivery
(OMN-18572) and the terminal bus publish all run AFTER
``job_store.complete(...)``. Every one of them is a minutes-long host mutation
belonging to the job whose record already reads ``success``.

Measured 2026-09-17, lane ``deploy-agent-http-hang-diag-2105``: job ``79d743db``
was logged "completed successfully" at 19:56:03.781Z, and the same thread then
ran four ``k3s ctr images import`` invocations to 19:59:35Z, wrote the
lab-overlay record at 19:59:35.523Z, delivered the onex-api pin at 19:59:35.719Z,
published at 19:59:36Z and rejoined an evicted consumer group at 19:59:41Z. For
those three minutes and thirty-eight seconds a reader of the job record saw a
finished job, and a reader of the agent saw a process that answered nothing --
and the two facts could not be reconciled from anything the agent served. The
diagnosis wrote it as the sentence this criterion exists to make false: **"no
job is in progress" per the job store is not "the agent is idle"**.

WHICH ARM OF AC5 THIS TAKES, AND WHY
------------------------------------
AC5 accepts either a terminal write moved after the post-terminal phases, or a
distinct field that exposes "terminal but still settling". This takes the
FIELD arm, deliberately, because the ordering arm would destroy a property the
post-terminal phases were built to have: the compose lane's verdict is settled
on the compose lane's own merits, and a lab-overlay failure must not report a
lane that IS running the merged sha as broken (``_apply_lab_overlay``'s
docstring, OMN-18200 AC5). Moving ``complete`` after the apply would fold the
lab verdict into the compose verdict through the back door of ordering.

The field is written IN THE TERMINAL WRITE ITSELF, not after it. A ``complete``
followed by a separate ``set_settling`` would leave a window -- however short --
in which the record reads exactly as it does today, and a window is what the
19:56:03Z reader fell into.

WHAT IS NOT ASSERTED, DELIBERATELY
----------------------------------
Nothing here reads the source of ``agent.py`` or counts call sites. The surface
is exercised: the agent runs a real deploy, a real post-terminal phase blocks,
and an external thread asks the agent's own ``/job/{correlation_id}`` endpoint
what it sees -- which is the read ``check_dev_lane_staleness.py`` makes.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent.agent import DeployAgent
from deploy_agent.events import (
    EnumRuntimeLane,
    ModelRebuildRequested,
    Phase,
    PhaseStatus,
    Scope,
)
from deploy_agent.job_state import EnumJobSettlingStage, JobStore

pytestmark = pytest.mark.unit

SHA = "b" * 40

#: How long a post-terminal phase holds its thread, standing in for the
#: ``k3s ctr images import`` sequence that held it for 3m38s on 2026-09-17. Long
#: enough that a probe issued at phase entry, and its whole timeout, land
#: strictly inside the phase.
BLOCK_SECONDS = 3.0

PROBE_TIMEOUT_SECONDS = 1.0

PHASE_ENTRY_TIMEOUT_SECONDS = 30.0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _get_json(url: str) -> tuple[int | None, dict[str, Any] | None, str | None]:
    """One bounded GET. No proxy, ever -- see the sibling OMN-18636 test file."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=PROBE_TIMEOUT_SECONDS) as response:
            return int(response.status), json.loads(response.read()), None
    except urllib.error.HTTPError as exc:
        return int(exc.code), None, repr(exc)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return None, None, repr(exc)


class _FastExecutor:
    """A deploy whose phases return immediately.

    This file is about what happens AFTER the phases, so the phases themselves
    are made free: any time spent in them is time not spent in the window under
    test.
    """

    def __init__(self, pin_gate: threading.Event, pin_running: threading.Event):
        self.calls: list[str] = []
        self.container_residue: list[object] = []
        self.sibling_source_refs: dict[str, str] = {}
        # OMN-18692: the agent reads this when it builds the terminal event,
        # so a double that omits it no longer models the object it replaces.
        self.recreate_supervision: list[object] = []
        # OMN-18640: the terminal event now also carries what verification
        # recreated, so a fake executor has to declare it.
        self.verify_recreate: list[object] = []
        # OMN-18640: the terminal event now also carries what the deps leg
        # found before it acted, and the argv of every compose call, so a
        # fake executor has to declare both.
        self.deps_convergence: list[object] = []
        self.compose_invocations: list[object] = []
        # OMN-18640 AC8: the agent publishes the executor's own probe
        # readings when its local list is empty, which is the case on
        # every job that failed verification.
        self.health_checks: list[object] = []
        self._pin_gate = pin_gate
        self._pin_running = pin_running

    def preflight(self, **kwargs: Any) -> None:
        self.calls.append("preflight")
        kwargs["on_phase_update"](Phase.PREFLIGHT, PhaseStatus.SUCCESS)

    def git_pull(self, git_ref: str, **kwargs: Any) -> str:
        self.calls.append("git_pull")
        return SHA

    def compose_gen(self, bundles: list[str], **kwargs: Any) -> None:
        self.calls.append("compose_gen")

    def seed_infisical(self, **kwargs: Any) -> None:
        self.calls.append("seed_infisical")

    def validate_llm_endpoint_env_contract(self) -> None:
        self.calls.append("validate_llm_endpoint_env_contract")

    def rebuild_scope(self, *args: Any, **kwargs: Any) -> list[str]:
        self.calls.append("rebuild_scope")
        return ["omninode-runtime"]

    def verify(self, **kwargs: Any) -> list[object]:
        self.calls.append("verify")
        return []

    def deliver_onex_api_pin(self, **kwargs: Any) -> dict[str, Any]:
        """The SECOND post-terminal phase (OMN-18572), blocking on demand."""
        self.calls.append("deliver_onex_api_pin")
        self._pin_running.set()
        self._pin_gate.wait(timeout=BLOCK_SECONDS * 4)
        self._pin_running.clear()
        return {"result": "delivered", "tag_advanced": True, "recreated": True}

    def self_update(self, *, boundary: Any, **kwargs: Any) -> None:
        pass


class _BlockingApplier:
    """A lab-overlay applier that holds the job thread while the test reads.

    The real one shells out to ``k3s ctr images import`` four times; what
    matters here is only that it takes time and that it runs after the terminal
    write, both of which a gated wait reproduces exactly.
    """

    def __init__(self, gate: threading.Event, running: threading.Event) -> None:
        self.gate = gate
        self.running = running
        self.manifest_sha: str | None = None

    def apply(self, *, sha: str, stamp: str, correlation_id: str) -> str:
        # OMN-18572: the real applier resolves this before anything that can
        # block or fail, and the agent reads it after the apply returns.
        self.manifest_sha = "b" * 40
        self.running.set()
        self.gate.wait(timeout=BLOCK_SECONDS * 4)
        self.running.clear()
        return f"/records/{sha}.json"


class _OneShotConsumer:
    command: ModelRebuildRequested | None = None

    def __init__(self, **kwargs: Any) -> None:
        self.job_store: JobStore = kwargs["job_store"]
        self._served = False

    def poll_and_accept(self) -> tuple[ModelRebuildRequested | None, str | None]:
        command = _OneShotConsumer.command
        if self._served or command is None:
            time.sleep(0.05)
            return None, None
        self._served = True
        self.job_store.accept(command.correlation_id, command.model_dump(mode="json"))
        return command, None

    def close(self) -> None:
        pass


def _dev_command() -> ModelRebuildRequested:
    return ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by="test",
        scope=Scope.RUNTIME,
        runtime_lane=EnumRuntimeLane.DEV,
    )


def _make_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    port: int,
    executor: _FastExecutor,
    applier: _BlockingApplier,
) -> DeployAgent:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    monkeypatch.setattr("deploy_agent.lock._LOCK_PATH", tmp_path / "deploy.lock")
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "jobs")
    monkeypatch.setattr(agent_mod, "HEALTH_PORT", port)
    # ON, unlike the sibling file: the post-terminal phases ARE the subject here.
    monkeypatch.setattr(agent_mod, "LAB_OVERLAY_ENABLED", True)
    monkeypatch.setattr(agent_mod, "DeployConsumer", _OneShotConsumer)
    monkeypatch.setattr(agent_mod, "publish_result", lambda payload, config: True)
    monkeypatch.setattr(agent_mod, "record_loaded_code_sha", lambda _agent_dir: SHA)

    agent = DeployAgent(skip_self_update=True)
    agent.executor = executor  # type: ignore[assignment]
    monkeypatch.setattr(agent, "_lab_overlay_applier", lambda: applier)
    return agent


# --------------------------------------------------------------------------- #
# AC5 -- terminal, and still settling, and saying so                           #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_a_terminal_job_reports_settling_while_post_terminal_work_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 19:56:03Z-versus-19:59:41Z span, reproduced and then made legible.

    RED against the tree where the job record carried no settling field: the
    probe taken while the lab-overlay apply is executing comes back
    ``status: success`` with nothing in the payload that distinguishes it from
    a job the agent finished three minutes ago. That is the exact read the
    receipt reader made on 2026-09-17.

    GREEN once the terminal write carries the stage: the same probe reads
    ``success`` AND ``settling: true`` AND names the phase still running, so
    "terminal" and "done with this job" stop being the same fact.
    """
    overlay_gate = threading.Event()
    overlay_running = threading.Event()
    pin_gate = threading.Event()
    pin_running = threading.Event()
    port = _free_port()
    executor = _FastExecutor(pin_gate, pin_running)
    applier = _BlockingApplier(overlay_gate, overlay_running)
    command = _dev_command()
    monkeypatch.setattr(_OneShotConsumer, "command", command, raising=False)

    agent = _make_agent(
        tmp_path, monkeypatch, port=port, executor=executor, applier=applier
    )
    url = f"http://127.0.0.1:{port}/job/{command.correlation_id}"
    observations: dict[str, tuple[int | None, dict[str, Any] | None, str | None]] = {}

    def _reader() -> None:
        # Read during the lab-overlay apply, release it, then read during the
        # onex-api pin delivery. Both are post-terminal by construction: the
        # agent calls them only after job_store.complete.
        if overlay_running.wait(timeout=PHASE_ENTRY_TIMEOUT_SECONDS):
            observations["overlay"] = _get_json(url)
        overlay_gate.set()
        if pin_running.wait(timeout=PHASE_ENTRY_TIMEOUT_SECONDS):
            observations["pin"] = _get_json(url)
        pin_gate.set()

    reader = threading.Thread(target=_reader, name="omn18636-ac5-reader")
    reader.start()
    run_task = asyncio.create_task(agent.run())
    try:
        await asyncio.to_thread(reader.join, PHASE_ENTRY_TIMEOUT_SECONDS * 2 + 30.0)
        # The job is fully settled only once the publish block has run, which is
        # after the pin delivery returns. Wait for the agent to go idle again.
        deadline = time.monotonic() + 30.0
        while agent._get_state() != "idle" and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
    finally:
        overlay_gate.set()
        pin_gate.set()
        agent._shutdown = True
        await asyncio.wait_for(run_task, timeout=60.0)

    assert "overlay" in observations, (
        "the lab-overlay apply never ran, so nothing post-terminal was observed: "
        f"{executor.calls}"
    )
    status, body, error = observations["overlay"]
    assert error is None and status == 200, (
        f"GET /job during the lab-overlay apply did not answer: {error}"
    )
    assert body is not None
    assert body["status"] == "success", (
        "the premise of this criterion is that the terminal write has already "
        f"happened when the post-terminal work runs: {body}"
    )
    assert body.get("settling") is True, (
        "the job record read `success` while the agent was still running the "
        "lab-overlay apply for that same job, with no field distinguishing the "
        "two. This is the 19:56:03Z-versus-19:59:41Z span: a reader cannot tell "
        f"a settling job from a finished one. payload={body}"
    )
    assert body.get("settling_stage") == EnumJobSettlingStage.LAB_OVERLAY.value, (
        f"the settling field must name WHICH post-terminal phase is running: {body}"
    )

    assert "pin" in observations, "the onex-api pin delivery never ran"
    status, body, error = observations["pin"]
    assert error is None and status == 200, (
        f"GET /job during the onex-api pin delivery did not answer: {error}"
    )
    assert body is not None
    assert body["status"] == "success"
    assert body.get("settling") is True, (
        f"the pin delivery is post-terminal work on this job too: {body}"
    )
    assert body.get("settling_stage") == EnumJobSettlingStage.ONEX_API_PIN.value, (
        f"the second post-terminal phase must be named as itself: {body}"
    )

    # And it must CLEAR. A field that is only ever set turns every finished job
    # into a permanently-settling one, which is as unreadable as no field.
    store = JobStore(state_dir=tmp_path / "jobs")
    settled = store.load(command.correlation_id)
    assert settled is not None
    assert settled.settling_stage is None, (
        "the job never stopped reporting itself as settling, so the field can "
        f"no longer distinguish anything: {settled.settling_stage}"
    )
    assert settled.phase_results[Phase.PUBLISH] == PhaseStatus.SUCCESS, (
        f"the terminal publish must still have happened: {settled.phase_results}"
    )


def test_the_terminal_write_and_the_settling_stage_are_one_write(
    tmp_path: Path,
) -> None:
    """No window between "terminal" and "terminal but settling".

    ``complete`` writes the record once, and the stage is part of that write. A
    ``complete`` followed by a separate ``set_settling`` would be two writes with
    a gap between them, and any reader landing in that gap sees precisely
    today's defect -- so the gap is closed at the store, not by ordering luck in
    the caller.
    """
    store = JobStore(state_dir=tmp_path / "jobs")
    cid = uuid4()
    store.accept(cid, {"scope": "runtime"})

    job = store.complete(
        cid, status="success", settling_stage=EnumJobSettlingStage.LAB_OVERLAY
    )
    assert job.status == "success"
    assert job.settling_stage is EnumJobSettlingStage.LAB_OVERLAY

    # Read it back off disk, not out of the returned object: the reader is
    # another process going through the JSON.
    reread = store.load(cid)
    assert reread is not None
    assert reread.status == "success"
    assert reread.settling_stage is EnumJobSettlingStage.LAB_OVERLAY, (
        "the durable record, which is what /job serves, did not carry the stage"
    )

    store.set_settling(cid, EnumJobSettlingStage.PUBLISH)
    assert store.load(cid).settling_stage is EnumJobSettlingStage.PUBLISH  # type: ignore[union-attr]

    store.clear_settling(cid)
    assert store.load(cid).settling_stage is None  # type: ignore[union-attr]


def test_a_job_with_no_settling_stage_defaults_to_not_settling(
    tmp_path: Path,
) -> None:
    """An existing on-disk record predating the field is not "settling".

    The agent's state directory survives a restart, so records written by the
    previous version are read by the new one. A missing field must mean "not
    settling" -- the opposite default would turn every job already on disk into
    a permanently-settling one at the moment of upgrade.
    """
    state_dir = tmp_path / "jobs"
    state_dir.mkdir(parents=True)
    cid = uuid4()
    (state_dir / f"{cid}.json").write_text(
        json.dumps(
            {
                "correlation_id": str(cid),
                "command": {"scope": "runtime"},
                "accepted_at": "2026-09-17T19:00:00+00:00",
                "current_phase": "verification",
                "phase_results": {},
                "status": "success",
                "errors": [],
                "result_publish_pending": False,
                "completed_at": "2026-09-17T19:56:03+00:00",
            }
        ),
        encoding="utf-8",
    )

    job = JobStore(state_dir=state_dir).load(cid)
    assert job is not None
    assert job.settling_stage is None


def test_a_recovered_crashed_job_is_not_left_settling(tmp_path: Path) -> None:
    """A process that died mid-settle is not still settling.

    ``recover_crashed_jobs`` runs at startup over records the previous process
    left behind. A job whose lab-overlay apply was interrupted by the kill has a
    settling stage on disk and nothing executing it, so the stage must be
    cleared as part of recovery -- otherwise the field asserts that work is in
    flight in a process that no longer exists.
    """
    store = JobStore(state_dir=tmp_path / "jobs")
    cid = uuid4()
    store.accept(cid, {"scope": "runtime"})
    store.update_phase(cid, Phase.RUNTIME, PhaseStatus.IN_PROGRESS)
    store.set_settling(cid, EnumJobSettlingStage.LAB_OVERLAY)

    recovered = store.recover_crashed_jobs()

    assert len(recovered) == 1
    assert recovered[0].status == "failed"
    assert recovered[0].settling_stage is None, (
        "a job recovered from a dead process still claimed post-terminal work "
        "was executing"
    )
