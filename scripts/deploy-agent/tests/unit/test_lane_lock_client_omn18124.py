# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deploy agent takes the lane lock the refresh scripts already take (OMN-18124).

``Executor.git_pull`` runs ``git fetch --all --prune`` and ``git reset --hard``
against the SHARED deploy-source clone under no lock at all. That clone is also
owned by ``scripts/runtime_build/refresh_dev_lane.sh``, which since OMN-16729
holds a per-compose-project host lock across its whole build, gate and readback
critical section -- added because two sanctioned dev-lane refreshes collided on
2026-09-08.

The agent is a third writer and participates in none of it. The collision is
recorded in the clone's own reflog: the refresh checked out ``origin/dev`` at
21:26:37 local and an agent job pulled it back to ``origin/main`` 166 seconds
later; the same pattern repeated at 23:06:50 and 23:31:49. A refresh that had
already captured its pre-state was, for that window, building a tree the agent
was concurrently rewriting.

``reset --hard`` also discards uncommitted state in the shared tree with no
warning, which is the other half of what the lock protects.

WHY THE EXISTING LOCK AND NOT A NEW ONE
---------------------------------------

A second lock file would serialise the agent against itself and nothing else --
the exact non-protection it looks like it provides. The lock has to be the SAME
file the shell front end computes for the same compose project, so the path is
resolved from ``lane_lock.py`` itself rather than rebuilt from a copied rule.
"""

from __future__ import annotations

import os
import subprocess
from unittest.mock import patch

import pytest
from deploy_agent.lane_lock_client import (
    LaneLockContendedError,
    lane_lock,
    lane_lock_path,
    resolve_lane_lock_module,
)


@pytest.fixture(autouse=True)
def isolated_lock_dir(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the tests off the operational lock directory."""
    monkeypatch.setenv("ONEX_LANE_LOCK_DIR", f"{tmp_path}/lane-locks")
    monkeypatch.delenv("ONEX_LANE_LOCK_HELD", raising=False)


class TestPathAgreesWithTheShellHelper:
    """AC1, AC6: the same file, resolved from the same module."""

    def test_the_module_is_the_repo_one_not_a_reimplementation(self) -> None:
        module = resolve_lane_lock_module()
        assert module.__file__ is not None
        assert module.__file__.endswith("scripts/runtime_build/lane_lock.py")

    def test_the_path_matches_what_lane_lock_py_itself_prints(self) -> None:
        """AC6: computed by the helper's own CLI, not by a rule copied here."""
        module = resolve_lane_lock_module()
        printed = subprocess.run(
            ["python3", module.__file__, "path", "--compose-project", "omnibase-infra"],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ},
        ).stdout.strip()

        assert lane_lock_path("omnibase-infra") == printed


class TestMutualExclusion:
    """AC3, AC4: a bounded wait that names the holder and never steals."""

    def test_a_second_acquisition_is_refused_after_the_bounded_wait(self) -> None:
        with lane_lock("omnibase-infra", lane="dev", ref="origin/dev", timeout=0.2):
            with pytest.raises(LaneLockContendedError) as excinfo:
                # A separate process, because an fcntl lock is per open file
                # description: the same process would re-acquire its own lock
                # and prove nothing.
                _acquire_in_a_child(timeout=0.2)

        message = str(excinfo.value)
        assert "omnibase-infra" in message

    def test_the_lock_is_released_when_the_block_exits(self) -> None:
        with lane_lock("omnibase-infra", lane="dev", ref="origin/dev", timeout=0.2):
            pass
        _acquire_in_a_child(timeout=2.0)

    def test_a_failure_inside_the_block_still_releases(self) -> None:
        with pytest.raises(RuntimeError):
            with lane_lock("omnibase-infra", lane="dev", ref="origin/dev", timeout=0.2):
                raise RuntimeError("deploy blew up")
        _acquire_in_a_child(timeout=2.0)

    def test_two_different_lanes_do_not_block_each_other(self) -> None:
        with lane_lock("omnibase-infra", lane="dev", ref="origin/dev", timeout=0.2):
            with lane_lock(
                "omnibase-infra-stability-test",
                lane="stability-test",
                ref="origin/dev",
                timeout=0.2,
            ):
                pass


class TestReentrancy:
    """AC5: an agent-initiated deploy that shells out does not deadlock."""

    def test_an_outer_holder_in_this_ancestry_is_a_no_op(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ONEX_LANE_LOCK_HELD", " omnibase-infra ")
        # No wait, no acquisition: an outer process in this ancestry holds it.
        with lane_lock("omnibase-infra", lane="dev", ref="origin/dev", timeout=0.0):
            pass

    def test_the_token_is_exported_for_children(self) -> None:
        with lane_lock("omnibase-infra", lane="dev", ref="origin/dev", timeout=0.2):
            held = os.environ.get("ONEX_LANE_LOCK_HELD", "")
            assert "omnibase-infra" in held.split()
        assert "omnibase-infra" not in os.environ.get("ONEX_LANE_LOCK_HELD", "").split()

    def test_a_different_project_is_not_covered_by_the_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ONEX_LANE_LOCK_HELD", " omnibase-infra-prod ")
        with lane_lock("omnibase-infra", lane="dev", ref="origin/dev", timeout=0.2):
            assert "omnibase-infra" in os.environ["ONEX_LANE_LOCK_HELD"].split()


class TestExecutorWiring:
    """AC2: the lock is held before anything mutates the tree."""

    def test_git_pull_refuses_when_the_lane_is_contended(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deploy_agent.events import EnumRuntimeLane
        from deploy_agent.executor import DeployExecutor

        monkeypatch.setenv("DEPLOY_AGENT_TRACKING_REF", "dev")
        executor = DeployExecutor()
        issued: list[list[str]] = []

        def fake_run(
            cmd: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess:
            issued.append(cmd)
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="", stderr=""
            )

        holder = _hold_in_a_child()
        try:
            with patch("deploy_agent.executor._run", side_effect=fake_run):
                with pytest.raises(LaneLockContendedError):
                    executor.git_pull(
                        "origin/dev",
                        lane=EnumRuntimeLane.DEV,
                        on_phase_update=_noop_phase_update,
                        lock_timeout=0.2,
                    )
        finally:
            holder.terminate()
            holder.wait(timeout=5)

        assert issued == [], (
            f"the clone was touched while another lane held the lock: {issued}"
        )


def _noop_phase_update(phase: object, status: object) -> None:
    return None


_CHILD_ACQUIRE = """
import sys
sys.path.insert(0, {agent_dir!r})
from deploy_agent.lane_lock_client import LaneLockContendedError, lane_lock
try:
    with lane_lock("omnibase-infra", lane="dev", ref="origin/dev", timeout={timeout}):
        pass
except LaneLockContendedError:
    raise SystemExit(2)
raise SystemExit(0)
"""

_CHILD_HOLD = """
import sys, time
sys.path.insert(0, {agent_dir!r})
from deploy_agent.lane_lock_client import lane_lock
with lane_lock("omnibase-infra", lane="dev", ref="origin/dev", timeout=5.0):
    sys.stderr.write("held\\n")
    sys.stderr.flush()
    time.sleep(120)
"""


def _agent_dir() -> str:
    from pathlib import Path

    import deploy_agent

    return str(Path(deploy_agent.__file__).resolve().parents[1])


def _acquire_in_a_child(*, timeout: float) -> None:
    """Try to take the lane lock from a separate process."""
    result = subprocess.run(
        [
            "python3",
            "-c",
            _CHILD_ACQUIRE.format(agent_dir=_agent_dir(), timeout=timeout),
        ],
        capture_output=True,
        text=True,
        env={k: v for k, v in os.environ.items() if k != "ONEX_LANE_LOCK_HELD"},
        check=False,
    )
    if result.returncode == 2:
        raise LaneLockContendedError(
            f"lane lock for omnibase-infra is held: {result.stderr.strip()}"
        )
    assert result.returncode == 0, result.stderr


def _hold_in_a_child() -> subprocess.Popen[str]:
    process = subprocess.Popen(
        ["python3", "-c", _CHILD_HOLD.format(agent_dir=_agent_dir())],
        stderr=subprocess.PIPE,
        text=True,
        env={k: v for k, v in os.environ.items() if k != "ONEX_LANE_LOCK_HELD"},
    )
    assert process.stderr is not None
    assert process.stderr.readline().strip() == "held"
    return process
