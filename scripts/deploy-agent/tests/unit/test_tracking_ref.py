# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deploy agent tracks its declared lane deploy ref, never `main` (OMN-16442).

Operator ruling, in-session 2026-09-08, firm: the deploy agent tracks the lane's
deploy branch, never ``main``; any deploy-path reference that resolves ``main``
is a defect.

These tests pin all four surfaces the literal ``origin/main`` used to occupy:

1. ``deploy_agent.tracking_ref`` itself — required, no default, no remote prefix.
2. ``executor.self_update`` — fetches, rev-parses and pulls the configured ref.
3. ``ModelRebuildRequested.git_ref`` — defaults to the configured ref.
4. ``publisher.build_completion_payload`` — records the configured ref.

Plus the two tracked systemd units, which must declare ``dev``.

The lane-fence and control-bus-transport declarations (OMN-16939, OMN-18012) are
asserted the same way in ``test_lane_policy.py`` / ``test_kafka_config.py``; this
file follows that shape, including deleting the variable in its own autouse
fixture so the permissive conftest default cannot mask a regression.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest
from deploy_agent.events import (
    EnumRuntimeLane,
    ModelRebuildRequested,
    Scope,
)
from deploy_agent.executor import DeployExecutor
from deploy_agent.job_state import JobState
from deploy_agent.publisher import build_completion_payload
from deploy_agent.tracking_ref import (
    ENV_TRACKING_REF,
    load_tracking_ref_from_env,
    load_tracking_remote_ref_from_env,
)

_DEPLOY_DIR = Path(__file__).resolve().parents[2] / "deploy"

SHA_LOCAL = "aaaaaaaabbbbbbbb"
SHA_REMOTE = "ccccccccdddddddd"


@pytest.fixture(autouse=True)
def _own_the_tracking_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    """Delete the conftest-declared value so each test states its own."""
    monkeypatch.delenv(ENV_TRACKING_REF, raising=False)


def _ok(stdout: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


class TestTrackingRefDeclaration:
    def test_unset_raises(self) -> None:
        with pytest.raises(RuntimeError) as exc:
            load_tracking_ref_from_env()
        assert ENV_TRACKING_REF in str(exc.value)

    def test_blank_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_TRACKING_REF, "   ")
        with pytest.raises(RuntimeError):
            load_tracking_ref_from_env()

    def test_declared_value_is_returned_verbatim(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_TRACKING_REF, "dev")
        assert load_tracking_ref_from_env() == "dev"
        assert load_tracking_remote_ref_from_env() == "origin/dev"

    def test_no_hardcoded_fallback_to_main(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A lane on some other branch gets that branch, not a baked-in default."""
        monkeypatch.setenv(ENV_TRACKING_REF, "release-candidate")
        assert load_tracking_remote_ref_from_env() == "origin/release-candidate"

    def test_remote_qualified_value_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`origin/dev` would produce `origin/origin/dev` on the fetch line."""
        monkeypatch.setenv(ENV_TRACKING_REF, "origin/dev")
        with pytest.raises(RuntimeError) as exc:
            load_tracking_ref_from_env()
        assert "bare branch name" in str(exc.value)


class TestSelfUpdateUsesTrackingRef:
    @staticmethod
    def _record(calls: list[list[str]], *, remote: str = SHA_REMOTE):
        def side_effect(
            cmd: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess:
            calls.append(cmd)
            if "status" in cmd and "--porcelain" in cmd:
                return _ok("")
            if "rev-parse" in cmd:
                if any(part.startswith("origin/") for part in cmd):
                    return _ok(remote)
                return _ok(SHA_LOCAL)
            return _ok()

        return side_effect

    def test_fetch_rev_parse_and_pull_all_use_the_declared_branch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_TRACKING_REF, "dev")
        calls: list[list[str]] = []
        executor = DeployExecutor()
        with (
            patch("deploy_agent.executor._run", side_effect=self._record(calls)),
            patch("os.execv") as mock_execv,
        ):
            executor.self_update()

        fetches = [c for c in calls if "fetch" in c]
        rev_parses = [c for c in calls if "rev-parse" in c]
        pulls = [c for c in calls if "pull" in c]

        assert fetches and fetches[0][-2:] == ["origin", "dev"], fetches
        assert any("origin/dev" in c for c in rev_parses), rev_parses
        assert pulls and pulls[0][-2:] == ["origin", "dev"], pulls

        # The whole point: no git invocation may name `main`.
        assert not any("main" in part for c in calls for part in c), calls
        mock_execv.assert_called_once_with(sys.executable, [sys.executable] + sys.argv)

    def test_a_different_declared_branch_is_honoured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_TRACKING_REF, "staging")
        calls: list[list[str]] = []
        executor = DeployExecutor()
        with (
            patch("deploy_agent.executor._run", side_effect=self._record(calls)),
            patch("os.execv"),
        ):
            executor.self_update()
        assert any(c[-2:] == ["origin", "staging"] for c in calls), calls

    def test_already_at_the_tracking_ref_does_not_reexec(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_TRACKING_REF, "dev")
        calls: list[list[str]] = []
        executor = DeployExecutor()
        with (
            patch(
                "deploy_agent.executor._run",
                side_effect=self._record(calls, remote=SHA_LOCAL),
            ),
            patch("os.execv") as mock_execv,
        ):
            executor.self_update()
        mock_execv.assert_not_called()
        assert not any("pull" in c for c in calls), calls

    def test_unset_tracking_ref_raises_rather_than_deploying_a_guess(self) -> None:
        executor = DeployExecutor()
        with (
            patch("deploy_agent.executor._run") as mock_run,
            pytest.raises(RuntimeError) as exc,
        ):
            executor.self_update()
        assert ENV_TRACKING_REF in str(exc.value)
        mock_run.assert_not_called()

    def test_kill_switch_short_circuits_before_the_ref_is_needed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A deliberately disabled self-update must not require the variable."""
        monkeypatch.setenv("DEPLOY_AGENT_NO_SELF_UPDATE", "1")
        executor = DeployExecutor()
        with patch("deploy_agent.executor._run") as mock_run:
            executor.self_update()
        mock_run.assert_not_called()

    def test_skip_flag_short_circuits_before_the_ref_is_needed(self) -> None:
        executor = DeployExecutor()
        with patch("deploy_agent.executor._run") as mock_run:
            executor.self_update(skip=True)
        mock_run.assert_not_called()


class TestCommandDefaultGitRef:
    def _command(self) -> ModelRebuildRequested:
        return ModelRebuildRequested(
            correlation_id=uuid4(),
            requested_by="test",
            scope=Scope.RUNTIME,
            runtime_lane=EnumRuntimeLane.DEV,
        )

    def test_default_git_ref_follows_the_tracking_ref(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_TRACKING_REF, "dev")
        assert self._command().git_ref == "origin/dev"

    def test_default_git_ref_follows_a_non_dev_tracking_ref(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_TRACKING_REF, "staging")
        assert self._command().git_ref == "origin/staging"

    def test_unset_tracking_ref_refuses_to_build_a_command(self) -> None:
        with pytest.raises(RuntimeError) as exc:
            self._command()
        assert ENV_TRACKING_REF in str(exc.value)

    def test_explicit_git_ref_still_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_TRACKING_REF, "dev")
        cmd = ModelRebuildRequested(
            correlation_id=uuid4(),
            requested_by="test",
            scope=Scope.RUNTIME,
            runtime_lane=EnumRuntimeLane.DEV,
            git_ref="origin/jonah/some-branch",
        )
        assert cmd.git_ref == "origin/jonah/some-branch"


class TestCompletionPayloadRequestedRef:
    @staticmethod
    def _job(command: dict[str, object]) -> JobState:
        return JobState(correlation_id=uuid4(), command=command)

    def test_missing_git_ref_records_the_tracking_ref(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_TRACKING_REF, "dev")
        payload = build_completion_payload(
            self._job({"runtime_lane": "dev"}), git_sha="abc123"
        )
        assert payload["requested_git_ref"] == "origin/dev"

    def test_present_git_ref_is_recorded_verbatim(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_TRACKING_REF, "dev")
        payload = build_completion_payload(
            self._job({"runtime_lane": "dev", "git_ref": "origin/release-1"}),
            git_sha="abc123",
        )
        assert payload["requested_git_ref"] == "origin/release-1"


class TestUnitsDeclareTheTrackingRef:
    @pytest.mark.parametrize(
        "unit", ["deploy-agent.service", "deploy-agent-dev.service"]
    )
    def test_unit_declares_dev(self, unit: str) -> None:
        """Both tracked units declare `dev` — the ruling is 'never main'.

        Prod image promotion is governed by the OMN-13418 grant path and by the
        pinned stability-proven digest a prod command must carry, not by which
        branch this agent's own code tracks.
        """
        text = (_DEPLOY_DIR / unit).read_text()
        assert "Environment=DEPLOY_AGENT_TRACKING_REF=dev" in text, unit

    @pytest.mark.parametrize(
        "unit", ["deploy-agent.service", "deploy-agent-dev.service"]
    )
    def test_unit_does_not_declare_main(self, unit: str) -> None:
        text = (_DEPLOY_DIR / unit).read_text()
        assert "Environment=DEPLOY_AGENT_TRACKING_REF=main" not in text, unit
