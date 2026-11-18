# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The deploy agent's golden chain: new code, a job, a reading row (OMN-24674).

``tests/integration/handlers/test_handler_deploy_job_reading_omn24617.py`` runs the
reading handler against an append script the test writes, so it passes whether or not the
live agent's alarm ever wrote a row. This module runs the chain on the real parts:

1. ``test_golden_a_new_commit_changes_the_loaded_code_sha`` -- the agent's own
   ``deploy_agent.loaded_code`` records the sha of a real git checkout; a new commit and a
   restart (a fresh record) change it. Hermetic; runs on every runner.
2. ``test_golden_the_agents_last_terminal_job_has_its_reading_row`` -- reads the live agent
   and the ledger. The agent's last terminal job, however it ended, has the reading row
   the real append command wrote (``node_deploy_job_reading_effect``), bound to that job by
   id, completion time and the id the handler derives for the append request. The test
   waits while the agent reports the job settling, then fails when the row is missing one
   tick later, which is what removing the append command does. Selected with the ``kafka``
   marker: it needs the lab's dev broker path, so the PR test splits deselect it and a lab
   host runs it.

The live tests read the ledger's archive rolls as well as the live file, check every row in
the window and take the host from ``CHAIN_CANARY_AGENT_LEDGER_HOST`` (OMN-24695). They write no
ledger row: the OMN-19513 guard refuses a test's write to the ledger of record, and the live
agent has already written the row they read. They do not run the append command themselves:
it needs a ledger host on a shared bus (see ``contracts/OMN-24695.yaml``).
"""

from __future__ import annotations

import os
import re
import subprocess
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from deploy_agent import loaded_code
from deploy_agent.events import EnumRuntimeLane

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from tests.integration.chain_canary.conftest import (
    ONE_TICK_SECONDS,
    LiveDeployAgentChain,
    handler_failing_step,
    reading_problems,
)

pytestmark = pytest.mark.integration

_CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_deploy_job_reading_effect/contract.yaml"
)
_SETTINGS = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))["settings"]
_COVERED_LANES = (EnumRuntimeLane.DEV.value, EnumRuntimeLane.STABILITY_TEST.value)
_SHA = re.compile(r"[0-9a-f]{40}")


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env=scrub_git_location_env(os.environ),
    )
    return done.stdout.strip()


def _commit(repo: Path, name: str) -> str:
    (repo / name).write_text(name, encoding="utf-8")
    _git(repo, "add", name)
    _git(
        repo,
        "-c",
        "user.name=chain-canary",
        "-c",
        "user.email=chain-canary@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-m",
        name,
    )
    return _git(repo, "rev-parse", "HEAD")


def test_golden_a_new_commit_changes_the_loaded_code_sha(tmp_path: Path) -> None:
    repo = tmp_path / "agent-code"
    repo.mkdir()
    _git(repo, "init", "-q")
    first = _commit(repo, "first")
    loaded_code.reset_loaded_code_sha()
    try:
        assert loaded_code.record_loaded_code_sha(str(repo)) == first
        second = _commit(repo, "second")
        assert second != first
        # The process has not restarted: it still reports the code it loaded.
        assert loaded_code.loaded_code_sha() == first
        # A restart records the checkout it now loads.
        assert loaded_code.record_loaded_code_sha(str(repo)) == second
        assert loaded_code.loaded_code_sha_if_recorded() == second
    finally:
        loaded_code.reset_loaded_code_sha()


@pytest.mark.kafka
def test_golden_the_newest_reading_row_matches_the_agents_job_record(
    live_deploy_agent_chain: LiveDeployAgentChain,
) -> None:
    """The ledger's newest reading row from the agent's host is the agent's own job, as it settled.

    Every row in the window is also checked for its shape and for being published inside one tick
    of its job's completion.
    """
    chain = live_deploy_agent_chain
    rows = chain.rows()
    assert rows, (
        "the ledger and its archive rolls hold no deploy-agent-job reading row from the agent's "
        "host: the append command never wrote one"
    )
    problems = reading_problems(chain.window_rows())
    assert not problems, "readings in the window are wrong:\n" + "\n".join(problems)
    row = rows[-1]
    job = chain.job(row["job"])
    status = str(job["status"])
    assert row["verdict"] == ("WORKING" if status == "success" else "NOT_WORKING")
    assert status in ("success", "failed"), f"a row was written for a {status} job"
    completed = datetime.fromisoformat(str(job["completed_at"]))
    assert row["completed_at"] == completed.astimezone(UTC).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    if status == "failed":
        phases = job["phase_results"]
        assert isinstance(phases, dict)
        # The handler's own rule: a lane-lock or host-slot job settles every phase skipped, and
        # its row names the first phase that did not succeed.
        assert row["failing_step"] == handler_failing_step(_SETTINGS, phases)
    assert row["deploy_lane"] in _COVERED_LANES
    assert row["piece"] == _SETTINGS["piece"]
    assert row["name"] == _SETTINGS["piece_name"]
    # The append request id is the one the handler derives, so a retry cannot write twice.
    assert row["req"] == str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"onex:deploy-job-reading:{row['job']}")
    )


@pytest.mark.kafka
def test_golden_the_agents_last_terminal_job_has_its_reading_row(
    live_deploy_agent_chain: LiveDeployAgentChain,
) -> None:
    """The agent's last terminal job, however it ended, has its row one tick after it completed."""
    chain = live_deploy_agent_chain
    assert chain.ledger_is_current, (
        "CHAIN_CANARY_LEDGER_IS_CURRENT is not 1: a lane's ledger copy is a snapshot, so a "
        "missing row proves nothing; run this where CHAIN_CANARY_LEDGER_FILE is the ledger of record"
    )
    health = chain.health()
    sha = health["loaded_code_sha"]
    assert isinstance(sha, str)
    assert _SHA.fullmatch(sha), f"the agent reports no loaded code sha: {sha!r}"

    last = health["last_result"]
    assert isinstance(last, dict), "the agent has settled no job to read"
    job_id = str(last["correlation_id"])
    status = str(last["status"])
    completed = datetime.fromisoformat(str(last["completed_at"]))
    # Where the overlay settle runs, completed_at is stamped before it and the row is written
    # after it: the one tick starts when the agent stops reporting the job settling.
    settled_at = chain.wait_while_settling(job_id)
    deadline = max(completed, settled_at) + timedelta(seconds=ONE_TICK_SECONDS)

    row = chain.wait_for_row(job_id, deadline=deadline)

    # /health lists only success and failed jobs, so no other status reaches here.
    assert status in ("success", "failed"), f"unexpected terminal status {status!r}"
    assert row is not None, (
        f"job {job_id} ({status}) completed {completed.isoformat()} and the ledger holds "
        f"no reading row for it {ONE_TICK_SECONDS:.0f}s later; the append command did not "
        "write one"
    )
    assert row["verdict"] == ("WORKING" if status == "success" else "NOT_WORKING")
    assert row["key"] == job_id
