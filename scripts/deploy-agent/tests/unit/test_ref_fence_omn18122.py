# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deploy agent refuses a stale branch alias on a lane that tracks another branch (OMN-18122).

Between 2026-09-10T00:48:53Z and 03:31:48Z five rebuild commands carrying
``git_ref: "origin/main"`` were accepted by the .201 dev agent (jobs
``d62ec83e``, ``68295ceb``, ``1316f524``, ``77343308``, ``ced0ac22``) and each
ran ``git reset --hard origin/main`` on the SHARED deploy-source clone. Measured
at the time: ``origin/main`` was ``276d69383``, ``origin/dev`` was ``bcb3e6460``,
``git merge-base --is-ancestor origin/main origin/dev`` exited 0, and
``git rev-list --count origin/main..origin/dev`` was 77. So the reset moved the
clone 77 commits BACKWARDS, behind the commit that added
``scripts/preflight_required_compose_env.py``, and every subsequent rebuild died.

WHY THE RULE IS "STALE BRANCH ALIAS" AND NOT "THE LITERAL main"
---------------------------------------------------------------

Three candidate rules were considered against the measured facts:

* *Refuse the literal* ``main``. Rejected: it reintroduces the hardcoded branch
  literal OMN-16442 removed from this package, and a repo whose release branch
  is named otherwise slips through.
* *Require the ref to be reachable from the tracking branch.* Rejected on
  evidence: ``origin/main`` **is** an ancestor of ``origin/dev``, so this rule
  does not catch the case that actually fired.
* *Refuse a branch reference that is strictly behind the tracking branch* — no
  commits of its own, some commits missing. This is the one implemented. It
  encodes the actual harm (the deploy clone moves backwards off its lane's own
  lineage) without naming any branch, and it leaves a feature-branch deploy
  alone because a feature branch carries commits the tracking branch does not.

A bare commit SHA is deliberately out of scope: a SHA is an unambiguous pin the
caller chose, and the GitHub Actions publisher uses one on every dev deploy. The
defect class here is a branch ALIAS silently resolving somewhere stale.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest
from deploy_agent.events import Phase, PhaseStatus
from deploy_agent.executor import DeployExecutor
from deploy_agent.job_state import JobState
from deploy_agent.publisher import build_completion_payload
from deploy_agent.ref_fence import (
    ModelRefLineageFacts,
    StaleBranchRefError,
    assert_ref_not_stale_branch,
)


def _noop_phase_update(phase: object, status: object) -> None:
    return None


def _facts(
    *,
    requested_ref: str,
    tracking_ref: str = "origin/dev",
    is_branch_reference: bool = True,
    commits_ahead: int = 0,
    commits_behind: int = 0,
) -> ModelRefLineageFacts:
    return ModelRefLineageFacts(
        requested_ref=requested_ref,
        tracking_ref=tracking_ref,
        is_branch_reference=is_branch_reference,
        commits_ahead_of_tracking=commits_ahead,
        commits_behind_tracking=commits_behind,
    )


class TestDecision:
    """The pure decision, exercised without git."""

    def test_refuses_the_observed_origin_main_case(self) -> None:
        """origin/main: a branch, 0 commits of its own, 77 behind dev (OMN-18122)."""
        with pytest.raises(StaleBranchRefError) as excinfo:
            assert_ref_not_stale_branch(
                _facts(requested_ref="origin/main", commits_behind=77)
            )
        message = str(excinfo.value)
        # AC1: the refusal names the request's ref and the declared tracking ref.
        assert "origin/main" in message
        assert "origin/dev" in message
        assert "77" in message

    def test_refuses_the_bare_branch_form(self) -> None:
        """`main` and `origin/main` are the same defect (AC2)."""
        with pytest.raises(StaleBranchRefError):
            assert_ref_not_stale_branch(_facts(requested_ref="main", commits_behind=77))

    def test_allows_the_tracking_branch_itself(self) -> None:
        assert_ref_not_stale_branch(
            _facts(requested_ref="origin/dev", commits_behind=0)
        )

    def test_allows_a_feature_branch_carrying_its_own_commits(self) -> None:
        """A feature branch is behind dev but has commits of its own — not stale."""
        assert_ref_not_stale_branch(
            _facts(
                requested_ref="origin/jonah/omn-18122-fence",
                commits_ahead=3,
                commits_behind=12,
            )
        )

    def test_allows_a_bare_sha_even_when_strictly_behind(self) -> None:
        """A SHA is an unambiguous pin the caller chose; the GHA publisher uses one."""
        assert_ref_not_stale_branch(
            _facts(
                requested_ref="6c3acf690868687d9f2f2c4f2f6c9f2a1b3c4d5e",
                is_branch_reference=False,
                commits_behind=77,
            )
        )

    def test_allows_a_branch_level_with_the_tracking_branch(self) -> None:
        """Neither ahead nor behind: nothing moves backwards, so nothing to refuse."""
        assert_ref_not_stale_branch(_facts(requested_ref="origin/release-candidate"))


class TestExecutorWiring:
    """The fence runs before the clone is touched (AC3)."""

    def test_git_pull_refuses_before_reset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No `reset --hard` is issued when the fence refuses (AC3, AC5)."""
        monkeypatch.setenv("DEPLOY_AGENT_TRACKING_REF", "dev")
        executor = DeployExecutor()
        issued: list[list[str]] = []

        def fake_run(
            cmd: list[str], timeout: int, **kwargs
        ) -> subprocess.CompletedProcess:
            issued.append(cmd)
            if "--symbolic-full-name" in cmd:
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=0,
                    stdout="refs/remotes/origin/main\n",
                    stderr="",
                )
            if "rev-list" in cmd and "--left-right" in cmd:
                # `<behind>\t<ahead>` for tracking...requested
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout="77\t0\n", stderr=""
                )
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="", stderr=""
            )

        with patch("deploy_agent.executor._run", side_effect=fake_run):
            with pytest.raises(StaleBranchRefError):
                executor.git_pull("origin/main", on_phase_update=_noop_phase_update)

        reset_commands = [cmd for cmd in issued if "reset" in cmd]
        assert reset_commands == [], (
            f"clone was mutated despite refusal: {reset_commands}"
        )

    def test_git_pull_allows_the_tracking_branch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ordinary dev deploy still runs (AC5)."""
        monkeypatch.setenv("DEPLOY_AGENT_TRACKING_REF", "dev")
        executor = DeployExecutor()
        sentinel_sha = "bcb3e646012b"

        def fake_run(
            cmd: list[str], timeout: int, **kwargs
        ) -> subprocess.CompletedProcess:
            if "--symbolic-full-name" in cmd:
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=0,
                    stdout="refs/remotes/origin/dev\n",
                    stderr="",
                )
            if "rev-list" in cmd and "--left-right" in cmd:
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout="0\t0\n", stderr=""
                )
            if "rev-parse" in cmd:
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout=sentinel_sha, stderr=""
                )
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="", stderr=""
            )

        with patch("deploy_agent.executor._run", side_effect=fake_run):
            sha = executor.git_pull("origin/dev", on_phase_update=_noop_phase_update)

        assert sha == sentinel_sha


class TestTerminalEvent:
    """The refusal reaches the wire, not just the log (AC6)."""

    def test_refusal_reason_is_carried_in_the_completion_payload(
        self, tmp_path: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A caller must be able to tell this refusal from a build failure.

        Every one of the five jobs this ticket is about published a terminal
        event, and a consumer reading only ``phase_results`` saw "the git phase
        failed" with no way to learn that the ref itself was the problem.
        """
        monkeypatch.setenv("DEPLOY_AGENT_TRACKING_REF", "dev")

        refusal = StaleBranchRefError(
            "deploy ref 'origin/main' names a branch that is strictly behind "
            "this lane's tracking branch 'origin/dev'"
        )
        job = JobState(
            correlation_id=uuid4(),
            command={
                "correlation_id": "00000000-0000-0000-0000-000000000000",
                "requested_by": "node_redeploy_orchestrator",
                "scope": "full",
                "runtime_lane": "dev",
                "git_ref": "origin/main",
            },
            accepted_at=datetime.now(UTC),
            current_phase=Phase.GIT,
            phase_results={Phase.PREFLIGHT: PhaseStatus.SUCCESS},
            status="failed",
            errors=[str(refusal)],
            completed_at=datetime.now(UTC),
        )

        payload = build_completion_payload(job, "")

        assert payload["errors"] == [str(refusal)]
        assert "strictly behind" in payload["errors"][0]
        assert "origin/main" in payload["errors"][0]


class TestNoEscapeHatch:
    """AC4: the refusal is unconditional for a ref-mode deploy."""

    def test_the_fence_reads_no_environment_variable_of_its_own(self) -> None:
        """An override would make this fence advisory, which is what already failed.

        The five jobs this ticket is about were stopped by nothing, so the
        difference between this fence and the previous state is precisely that
        there is no way to ask it to stand down. The only environment this
        module may consult is the tracking ref the LANE declares -- which
        narrows the fence's input, never disables it.
        """
        import deploy_agent.ref_fence as module

        source = Path(module.__file__).read_text()
        for escape in ("os.environ", "os.getenv", "getenv("):
            assert escape not in source, (
                f"{escape} appears in ref_fence.py; the fence must not be "
                "switchable from the environment"
            )

    def test_the_decision_takes_no_override_argument(self) -> None:
        """The signature is facts in, refusal or nothing out."""
        import inspect

        parameters = inspect.signature(assert_ref_not_stale_branch).parameters
        assert list(parameters) == ["facts"], (
            "assert_ref_not_stale_branch grew a parameter beyond the facts it "
            f"decides on: {list(parameters)}"
        )
        for name in ModelRefLineageFacts.__dataclass_fields__:
            assert not any(
                token in name for token in ("force", "allow", "skip", "override")
            ), f"ModelRefLineageFacts.{name} reads like an override switch"
