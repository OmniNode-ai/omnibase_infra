# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The deploy agent's error chain: a failed job, a NOT_WORKING reading row (OMN-24674).

``tests/integration/handlers/test_handler_deploy_job_reading_omn24617.py`` runs the reading
handler against an append script the test writes. This module runs the failure path on the
real parts:

1. ``test_error_a_failing_preflight_settles_a_failed_job_and_reads_not_working`` -- the real
   ``DeployAgent._run_deploy`` runs a dev job through the real ``DeployExecutor``. The
   executor's real ``preflight`` shells out to real ``git ls-remote`` in a checkout that has
   no ``origin``, so the job fails in ``preflight`` by design and never reaches docker, a
   compose project or the shared lane. The agent's own terminal publish then runs the real
   reading composition root; the test records what it returned. Hermetic; runs on every
   runner. The environment holds no append route here, so the row is built and reported
   ``unappended``; the live test below holds the append.
2. ``test_error_the_agents_failed_jobs_have_their_not_working_rows`` -- reads the ledger
   rows the live agent's host appended through the real append command, from the live file
   and the archive rolls. Some NOT_WORKING row exists, and every row in the window names a
   job and a failing step the handler can emit, was written within one tick of the job's
   completion and carries the append id the handler derives. Selected with the ``kafka``
   marker: it needs the lab's dev broker path, so the PR test splits deselect it and a lab
   host runs it.

The hermetic test builds the row and reports the append ``unappended`` because this process
holds no append route: the real append command needs a ledger host on a shared bus, which a
test process does not have (``contracts/OMN-24695.yaml`` records the probe). It therefore
proves the row, not the append. The live test proves the append, on rows the agent wrote.
It writes no ledger row: the OMN-19513 guard refuses a test's write to the ledger of
record, and no job is induced on the live agent (a deploy command needs the lane's signing
credential, which a test does not hold).
"""

from __future__ import annotations

import os
import subprocess
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import yaml
from deploy_agent import agent as agent_mod
from deploy_agent import executor as executor_mod
from deploy_agent import publisher as publisher_mod
from deploy_agent.agent import JOB_READING, DeployAgent
from deploy_agent.events import EnumRuntimeLane, ModelRebuildRequested, Scope
from deploy_agent.executor import DeployExecutor, load_dev_boot_guard
from deploy_agent.job_state import JobStore

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from tests.integration.chain_canary.conftest import (
    LiveDeployAgentChain,
    reading_problems,
)

pytestmark = pytest.mark.integration

_CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_deploy_job_reading_effect/contract.yaml"
)
_COVERED_LANES = (EnumRuntimeLane.DEV.value, EnumRuntimeLane.STABILITY_TEST.value)
_SETTINGS = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))["settings"]


def _row_cells(row: str) -> dict[str, str]:
    parts = [part.strip() for part in row.split(" | ")]
    return dict(part.split("=", 1) for part in parts[2:] if "=" in part)


@pytest.mark.parametrize("lane", [EnumRuntimeLane.DEV, EnumRuntimeLane.STABILITY_TEST])
def test_error_a_failing_preflight_settles_a_failed_job_and_reads_not_working(
    lane: EnumRuntimeLane, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "deploy-source"
    checkout.mkdir()
    subprocess.run(
        ["git", "-C", str(checkout), "init", "-q"],
        check=True,
        env=scrub_git_location_env(os.environ),
    )
    # No ``origin``: the real ``git ls-remote --exit-code origin`` exits non-zero.
    monkeypatch.setattr(executor_mod, "REPO_DIR", str(checkout))
    monkeypatch.setenv("ONEX_LANE_LOCK_DIR", str(tmp_path / "lane-locks"))
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    monkeypatch.setenv("DEPLOY_AGENT_TRACKING_REF", "dev")
    monkeypatch.setenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")
    monkeypatch.setenv("DEPLOY_AGENT_ALLOWED_LANES", ",".join(_COVERED_LANES))
    monkeypatch.setenv("DEPLOY_AGENT_INSTANCE", "dev-201")
    monkeypatch.setenv("OMNI_HOME", str(tmp_path))
    # The dev image boot guard inspects docker images; the job fails before any of that is due.
    guard = Mock()
    guard.protect.side_effect = lambda *args: nullcontext()
    monkeypatch.setattr(agent_mod, "DevBootGuard", Mock(return_value=guard))
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "agent-state")
    monkeypatch.setattr(agent_mod, "publish_result", lambda payload, config: False)
    monkeypatch.setattr(agent_mod.socket, "gethostname", lambda: "chain-canary")
    # The dev incident events a failed dev job also publishes go to the agent's producer.
    monkeypatch.setattr(publisher_mod, "_ensure_topic", lambda topic, config: None)
    monkeypatch.setattr(publisher_mod, "_produce", lambda *args, **kwargs: True)

    cmd = ModelRebuildRequested(
        correlation_id=uuid.uuid4(),
        requested_by="chain-canary/omn-24674",
        scope=Scope.RUNTIME,
        runtime_lane=lane,
    )
    agent = DeployAgent(skip_self_update=True)
    agent.job_store = JobStore(tmp_path)
    agent.job_store.accept(cmd.correlation_id, cmd.model_dump(mode="json"))
    agent.executor = DeployExecutor()

    reading = load_dev_boot_guard(JOB_READING)
    results: list[Any] = []
    real_publish = reading.publish

    def spy(job: Any, execution: Any) -> Any:
        result = real_publish(job, execution)
        results.append(result)
        return result

    monkeypatch.setattr(reading, "publish", spy)

    agent._run_deploy(cmd)

    job = agent.job_store.load(cmd.correlation_id)
    assert job is not None and job.status == "failed"
    assert job.phase_results["preflight"] == "failed"
    # The failure is the real ``git ls-remote`` against a checkout with no origin.
    assert job.errors and job.errors[0].startswith("Git remote unreachable")
    assert len(results) == 1, "the terminal publish did not run the reading"
    row = results[0].row
    assert row is not None
    cells = _row_cells(row)
    assert cells["verdict"] == "NOT_WORKING"
    assert cells["reason"] == "job-failed:preflight"
    assert cells["failing_step"] == "preflight"
    assert cells["job"] == str(cmd.correlation_id)
    assert cells["deploy_lane"] == lane.value
    # No append route is exported to a test process, so the row is built and not appended.
    assert results[0].outcome == "unappended"
    assert _SETTINGS["append_command_env"] in results[0].detail


@pytest.mark.kafka
def test_error_the_agents_failed_jobs_have_their_not_working_rows(
    live_deploy_agent_chain: LiveDeployAgentChain,
) -> None:
    chain = live_deploy_agent_chain
    failed = [row for row in chain.rows() if row.get("verdict") == "NOT_WORKING"]
    assert failed, (
        "the live ledger and its archive rolls hold no NOT_WORKING reading row from the "
        "agent's host: no failed job has ever been read, or the append command never wrote one"
    )
    for row in failed:
        uuid.UUID(row["job"])
    # Every row in the window, not the newest alone: the shape the handler writes and one tick
    # between the job's completion and its row.
    problems = reading_problems(chain.window_rows())
    assert not problems, "readings in the window are wrong:\n" + "\n".join(problems)
