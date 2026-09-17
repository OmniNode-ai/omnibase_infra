# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``_compose_build`` reads the machine, and says what the build DID (OMN-18615).

The unit-level derivation is pinned in
``test_build_ceiling_host_terms_omn18615.py``. This file pins the LIVE PATH:
that the executor actually probes the host rather than deriving a
repository-only ceiling, that a blown ceiling and a broken build carry distinct
machine-readable tokens, that the kill message names observed progress, and
that an adapting ceiling still refuses to touch the lane.

The timeout case drives a REAL child process to a REAL
``subprocess.TimeoutExpired`` -- the child prints BuildKit-shaped progress and
then hangs -- because the whole point of AC2 is that the partial output
``subprocess.run`` hands back on the exception reaches the message. A
hand-constructed exception would pass whether or not that plumbing works.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.build_budget import EnumBuildOutcome
from deploy_agent.events import EnumRuntimeLane, Phase, PhaseStatus, Scope
from deploy_agent.executor import DeployExecutor
from deploy_agent.host_conditions import EnumBuildCacheState, probe_host_conditions
from deploy_agent.job_state import reconcile_terminal_phase_results

LAB_HOST_CPU_COUNT = 32

# A child that prints BuildKit-shaped progress and then outlives its ceiling.
_HANGING_BUILD = (
    "import sys, time\n"
    "sys.stderr.write('#1 [internal] load build definition\\n')\n"
    "sys.stderr.write('#1 DONE 0.1s\\n')\n"
    "sys.stderr.write('#12 [builder 3/9] RUN uv sync\\n')\n"
    "sys.stderr.write('#12 DONE 41.2s\\n')\n"
    "sys.stderr.flush()\n"
    "time.sleep(30)\n"
)


def _is_compose_build(cmd: list[str]) -> bool:
    return "build" in cmd and "compose" in cmd


def _noop_phase_update(phase: Phase, status: PhaseStatus) -> None:
    return None


@pytest.fixture
def _no_workspace_staging(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        DeployExecutor, "_stage_workspace", staticmethod(lambda *a, **k: None)
    )
    monkeypatch.setattr(
        DeployExecutor, "_resolve_plugin_ref", lambda self, path, fallback="": "0" * 40
    )
    monkeypatch.setenv("OMNI_HOME", str(tmp_path))


def _pin_host(
    monkeypatch: pytest.MonkeyPatch, *, load1: float, cache: EnumBuildCacheState
) -> None:
    """Pin the host reading so the ceiling under test is a stated one.

    Patched at ``executor_mod`` rather than inside ``host_conditions``, because
    what is under test here is whether the EXECUTOR consults the probe at all.
    """
    monkeypatch.setattr(
        executor_mod,
        "probe_host_conditions",
        lambda: probe_host_conditions(
            loadavg_reader=lambda: (load1, load1, load1),
            cpu_count_reader=lambda: LAB_HOST_CPU_COUNT,
            builder_cache_reader=lambda: cache,
        ),
    )


def _build_with_spy(monkeypatch: pytest.MonkeyPatch, runner: Any) -> list[list[str]]:
    issued: list[list[str]] = []

    def _spy(cmd: list[str], **kwargs: Any) -> Any:
        issued.append(cmd)
        return runner(cmd, **kwargs)

    monkeypatch.setattr(executor_mod, "_run", _spy)
    return issued


def _ceiling_handed_to_the_build(
    monkeypatch: pytest.MonkeyPatch, *, load1: float, cache: EnumBuildCacheState
) -> int:
    """Run one build under a pinned host and return the ceiling it was handed.

    ``subprocess.run`` kills at exactly the ``timeout=`` it is given -- which is
    why both 2026-09-17 failures landed at 1081s and 1083s rather than at some
    scattered value -- so the handed ceiling IS the kill point.
    """
    _pin_host(monkeypatch, load1=load1, cache=cache)
    seen: dict[str, Any] = {}

    def _runner(cmd: list[str], **kwargs: Any) -> Any:
        if _is_compose_build(cmd):
            seen["timeout"] = kwargs.get("timeout")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    _build_with_spy(monkeypatch, _runner)
    DeployExecutor()._compose_build(
        Scope.RUNTIME,
        "0" * 40,
        _noop_phase_update,
        build_source="workspace",
        runtime_lane=EnumRuntimeLane.DEV,
        git_ref="origin/dev",
    )
    return int(seen["timeout"])


@pytest.mark.unit
class TestTheLivePathReadsTheMachine:
    @pytest.mark.usefixtures("_no_workspace_staging")
    def test_a_contended_cold_host_is_handed_a_wider_ceiling_than_an_idle_warm_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC1's falsifier on the live path: an identical number fails."""
        idle = _ceiling_handed_to_the_build(
            monkeypatch, load1=1.0, cache=EnumBuildCacheState.WARM
        )
        contended = _ceiling_handed_to_the_build(
            monkeypatch, load1=97.0, cache=EnumBuildCacheState.COLD
        )
        assert contended != idle
        assert contended > idle

    @pytest.mark.usefixtures("_no_workspace_staging")
    def test_the_logged_derivation_names_the_host_terms(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A ceiling a reader cannot explain is what produced this ticket."""
        _pin_host(monkeypatch, load1=97.0, cache=EnumBuildCacheState.COLD)
        _build_with_spy(
            monkeypatch, lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "", "")
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
        assert "load1" in line
        assert "cache cold" in line
        assert "hard upper bound" in line


@pytest.mark.unit
class TestTheKillMessageIsActionable:
    @pytest.mark.usefixtures("_no_workspace_staging")
    def test_a_real_timeout_names_the_token_and_the_observed_progress(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC2 + AC4, against a real child process and a real TimeoutExpired."""
        _pin_host(monkeypatch, load1=1.0, cache=EnumBuildCacheState.WARM)

        def _runner(cmd: list[str], **kwargs: Any) -> Any:
            if not _is_compose_build(cmd):
                return subprocess.CompletedProcess(cmd, 0, "", "")
            return subprocess.run(
                [sys.executable, "-c", _HANGING_BUILD],
                timeout=1,
                capture_output=True,
                text=True,
                check=False,
            )

        _build_with_spy(monkeypatch, _runner)
        with pytest.raises(RuntimeError) as raised:
            DeployExecutor()._compose_build(
                Scope.RUNTIME,
                "0" * 40,
                _noop_phase_update,
                build_source="workspace",
                runtime_lane=EnumRuntimeLane.DEV,
                git_ref="origin/dev",
            )

        message = str(raised.value)
        assert EnumBuildOutcome.BUDGET_EXHAUSTED.value in message
        assert EnumBuildOutcome.BUILD_ERRORED.value not in message
        # The two steps the child actually completed, read off its own output.
        assert "2/" in message
        assert "s/step observed" in message
        assert "s/step assumed" in message
        assert "NOT" in message and "mutated" in message

    @pytest.mark.usefixtures("_no_workspace_staging")
    def test_a_nonzero_exit_is_the_other_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A broken build and an exhausted budget lead to opposite next actions."""
        _pin_host(monkeypatch, load1=1.0, cache=EnumBuildCacheState.WARM)
        _build_with_spy(
            monkeypatch,
            lambda cmd, **kw: subprocess.CompletedProcess(
                cmd, 0 if not _is_compose_build(cmd) else 1, "", "ERROR: no such stage"
            ),
        )
        with pytest.raises(RuntimeError) as raised:
            DeployExecutor()._compose_build(
                Scope.RUNTIME,
                "0" * 40,
                _noop_phase_update,
                build_source="workspace",
                runtime_lane=EnumRuntimeLane.DEV,
                git_ref="origin/dev",
            )
        message = str(raised.value)
        assert EnumBuildOutcome.BUILD_ERRORED.value in message
        assert EnumBuildOutcome.BUDGET_EXHAUSTED.value not in message
        assert EnumBuildOutcome.classify(message) is EnumBuildOutcome.BUILD_ERRORED


@pytest.mark.unit
class TestAC5TheLaneIsStillNotMutated:
    @pytest.mark.usefixtures("_no_workspace_staging")
    def test_a_blown_ceiling_issues_no_container_lifecycle_command(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC5: an adapting ceiling must still touch nothing on the lane."""
        _pin_host(monkeypatch, load1=97.0, cache=EnumBuildCacheState.COLD)

        def _runner(cmd: list[str], **kwargs: Any) -> Any:
            if not _is_compose_build(cmd):
                return subprocess.CompletedProcess(cmd, 0, "", "")
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 1))

        issued = _build_with_spy(monkeypatch, _runner)
        with pytest.raises(RuntimeError):
            DeployExecutor()._compose_build(
                Scope.RUNTIME,
                "0" * 40,
                _noop_phase_update,
                build_source="workspace",
                runtime_lane=EnumRuntimeLane.DEV,
                git_ref="origin/dev",
            )

        mutating = {"up", "down", "stop", "rm", "restart", "start", "kill", "create"}
        for cmd in issued:
            assert not mutating.intersection(cmd), (
                f"a blown image-build ceiling issued a lane-mutating command: {cmd}"
            )

    def test_an_unreached_runtime_phase_settles_as_skipped(self) -> None:
        """``runtime: skipped`` is on the record, not a gap in it."""
        settled = reconcile_terminal_phase_results({})
        assert settled[Phase.RUNTIME] is PhaseStatus.SKIPPED
        assert settled[Phase.VERIFICATION] is PhaseStatus.SKIPPED

    @pytest.mark.usefixtures("_no_workspace_staging")
    def test_the_handed_ceiling_never_exceeds_the_hard_upper_bound(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pathological host must not hand the build an unbounded timeout."""
        monkeypatch.setattr(
            executor_mod,
            "probe_host_conditions",
            lambda: probe_host_conditions(
                loadavg_reader=lambda: (1_000_000.0, 1.0, 1.0),
                cpu_count_reader=lambda: 1,
                builder_cache_reader=lambda: EnumBuildCacheState.COLD,
            ),
        )
        seen: dict[str, Any] = {}

        def _runner(cmd: list[str], **kwargs: Any) -> Any:
            if _is_compose_build(cmd):
                seen["timeout"] = kwargs.get("timeout")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        _build_with_spy(monkeypatch, _runner)
        DeployExecutor()._compose_build(
            Scope.RUNTIME,
            "0" * 40,
            _noop_phase_update,
            build_source="workspace",
            runtime_lane=EnumRuntimeLane.DEV,
            git_ref="origin/dev",
        )
        assert seen["timeout"] <= executor_mod.HARD_UPPER_BOUND_SECONDS


@pytest.mark.unit
class TestTheTestHarnessDoesNotSilentlyRevertTheDerivation:
    """The path repoint in ``conftest`` re-implements the production call.

    It therefore has to carry every argument that call passes. Omitting the
    host term would revert every executor test to the OMN-18072 derivation
    while still reporting green -- the exact undetectable wrongness this
    ticket is about, reproduced inside the test harness rather than in the
    agent. Building the fix caught precisely that, so it is pinned here.
    """

    def test_the_repointed_budget_still_carries_host_conditions(self) -> None:
        budget = executor_mod.runtime_image_build_budget("runtime")
        assert budget.host is not None, (
            "the conftest compose-path repoint dropped the OMN-18615 host term; "
            "it is a repoint, not a re-implementation of the derivation"
        )

    def test_the_repointed_budget_reports_the_host_in_its_description(self) -> None:
        text = executor_mod.runtime_image_build_budget("runtime").describe()
        assert "no host conditions" not in text
