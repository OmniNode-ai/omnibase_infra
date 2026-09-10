# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``_compose_build`` runs under the derived ceiling, not the flat 300 (OMN-18072).

MEASURED on the dev lane, 2026-09-09, from the deploy agent's own job history:
commands ``79171e79`` and ``2788af33`` both died in ``docker compose --profile
runtime build`` at **exactly** the flat ``PHASE_TIMEOUTS[Phase.RUNTIME] = 300``
(300.0s and 300.3s from the ``_compose_build`` log line to the raised
``TimeoutExpired``), the second with a warm BuildKit cache from the first. The
one build this history completed, command ``6c323639``, took <= 233.9s. The
kill is ``subprocess.run``'s own, at exactly the ``timeout=`` it is handed, so
the value handed to that call IS the mechanism under test here.

Two controls below:

* the ceiling handed to the build subprocess clears 300 with real margin, so a
  build of the length that was killed twice now survives;
* a build that genuinely outruns its ceiling -- a real child process, a real
  ``subprocess.TimeoutExpired`` -- fails with a NAMED verdict that states the
  derivation and states that the lane was not mutated, rather than propagating
  a raw forty-token argv dump into the terminal record.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.events import EnumRuntimeLane, Phase, PhaseStatus, Scope
from deploy_agent.executor import PHASE_TIMEOUTS, DeployExecutor

# The constant that killed both live rebuilds.
FAILED_FLAT_CONSTANT_SECONDS = 300
# The longest build this history completed (command 6c323639).
MEASURED_SUCCESSFUL_BUILD_SECONDS = 234

_REAL_SUBPROCESS_RUN = subprocess.run


def _is_compose_build(cmd: list[str]) -> bool:
    return "build" in cmd and "compose" in cmd


def _noop_phase_update(phase: Phase, status: PhaseStatus) -> None:
    return None


@pytest.fixture
def _no_workspace_staging(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Staging shells out to the host repo; it is not what these tests bound."""
    monkeypatch.setattr(
        DeployExecutor, "_stage_workspace", staticmethod(lambda *a, **k: None)
    )
    monkeypatch.setattr(
        DeployExecutor, "_resolve_plugin_ref", lambda self, path, fallback="": "0" * 40
    )
    monkeypatch.setenv("OMNI_HOME", str(tmp_path))


@pytest.mark.unit
class TestBuildCeilingIsDerived:
    @pytest.mark.usefixtures("_no_workspace_staging")
    def test_build_subprocess_is_handed_a_ceiling_above_the_one_that_failed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A build of the length killed twice on 2026-09-09 now survives.

        subprocess.run kills at exactly the ``timeout=`` it is given -- that is
        why both live failures landed at 300.0s and 300.3s rather than at some
        scattered value -- so asserting the handed ceiling is asserting the
        kill point.
        """
        seen: dict[str, Any] = {}

        def _spy(cmd: list[str], **kwargs: Any) -> Any:
            if _is_compose_build(cmd):
                seen["timeout"] = kwargs.get("timeout")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(executor_mod, "_run", _spy)
        DeployExecutor()._compose_build(
            Scope.RUNTIME,
            "0" * 40,
            _noop_phase_update,
            build_source="workspace",
            runtime_lane=EnumRuntimeLane.DEV,
            git_ref="origin/dev",
        )

        ceiling = seen["timeout"]
        assert ceiling is not None
        assert ceiling != FAILED_FLAT_CONSTANT_SECONDS
        assert ceiling > FAILED_FLAT_CONSTANT_SECONDS
        # Clear of the one build this history completed, with real margin.
        assert ceiling > 2 * MEASURED_SUCCESSFUL_BUILD_SECONDS

    @pytest.mark.usefixtures("_no_workspace_staging")
    def test_derivation_is_logged_verbatim(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The ceiling states its own source, the way the compose-up one does."""
        monkeypatch.setattr(
            executor_mod,
            "_run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "", ""),
        )
        with caplog.at_level("INFO"):
            DeployExecutor()._compose_build(
                Scope.RUNTIME,
                "0" * 40,
                _noop_phase_update,
                build_source="workspace",
                runtime_lane=EnumRuntimeLane.DEV,
                git_ref="origin/dev",
            )
        line = next(
            record.getMessage()
            for record in caplog.records
            if "image-build ceiling" in record.getMessage()
        )
        assert "work steps" in line
        assert "Dockerfile.runtime" in line
        assert "buildable service(s)" in line

    def test_pull_and_preflight_keep_their_own_flat_bound(self) -> None:
        """Widening the build must not widen what is not a build.

        The pinned-digest pull, the image inspects and the migration one-shots
        are not gated on any healthcheck and are not solves; they stay on
        ``PHASE_TIMEOUTS[Phase.RUNTIME]``.
        """
        assert PHASE_TIMEOUTS[Phase.RUNTIME] == FAILED_FLAT_CONSTANT_SECONDS


@pytest.mark.unit
class TestBuildCeilingBreachVerdict:
    @pytest.mark.usefixtures("_no_workspace_staging")
    def test_outrunning_the_ceiling_raises_a_named_verdict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A REAL child process outruns a small injected ceiling.

        The ceiling is injected rather than waited out: the point is the shape
        of the verdict on a genuine ``subprocess.TimeoutExpired``, not a
        three-minute unit test. The stub is a real ``python -c 'sleep'``, so
        the exception is raised by subprocess itself.
        """
        from deploy_agent import build_budget

        tiny = build_budget.ModelBuildBudget(
            timeout_seconds=1,
            floor_seconds=1,
            per_step_seconds=1,
            per_image_seconds=1,
            build_steps=60,
            buildable_services=("omninode-runtime",),
            dockerfile="/repo/docker/Dockerfile.runtime",
            profile="runtime",
            compose_files=("/repo/docker/docker-compose.infra.yml",),
        )
        monkeypatch.setattr(
            executor_mod,
            "runtime_image_build_budget",
            lambda profile, compose_files=(): tiny,
        )

        def _sleeping_build(cmd: list[str], **kwargs: Any) -> Any:
            if _is_compose_build(cmd):
                return _REAL_SUBPROCESS_RUN(
                    [sys.executable, "-c", "import time; time.sleep(30)"],
                    timeout=kwargs.get("timeout"),
                    capture_output=True,
                    text=True,
                )
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(executor_mod, "_run", _sleeping_build)

        with pytest.raises(RuntimeError) as excinfo:
            DeployExecutor()._compose_build(
                Scope.RUNTIME,
                "0" * 40,
                _noop_phase_update,
                build_source="workspace",
                runtime_lane=EnumRuntimeLane.DEV,
                git_ref="origin/dev",
            )

        message = str(excinfo.value)
        # Named, not a raw argv dump.
        assert "runtime image build" in message
        assert "exceeded its 1s ceiling" in message
        # Carries its own derivation, so the terminal record explains itself.
        assert "Dockerfile.runtime" in message
        assert "60 work steps" in message
        # States the lane is intact: a build precedes every compose up, so
        # there is no half-recreated lane for an operator to go and check.
        assert "lane was NOT mutated" in message
        assert not isinstance(excinfo.value, subprocess.TimeoutExpired)
        assert isinstance(excinfo.value.__cause__, subprocess.TimeoutExpired)
