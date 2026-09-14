# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18200 -- self-update compares the LOADED code to the clone, not the clone to the remote.

Two defects, both measured on the .201 dev agent on 2026-09-14, both of which
make a fix to the deploy agent unable to reach the deploy agent.

RESIDUAL 1 -- the comparison was against the wrong thing.

``self_update`` decided whether to re-exec by comparing the CLONE's HEAD to
``origin/<tracking ref>``. Nothing in that comparison mentions the code the
RUNNING PROCESS actually imported. An external reconciler resets that same
clone to ``origin/dev`` every hour at :19 past (reflog on the lab host: resets
at 00:19, 01:19, 02:19, 03:19 local), so by the time the agent reaches a job
boundary the clone is already current and the method logs ``already at
origin/dev, nothing to do`` -- literally true, operationally wrong. The process
went on executing pre-#3520 code out of memory while the fixed file sat on disk
beside it, and every automated api build on the lab kept failing
``images_pinned``.

Live proof of the split at 2026-09-14T08:00Z: clone HEAD ``ead1f59b1``
(carrying the #3520 fix), process started 2026-09-09 14:06 EDT, health endpoint
reporting a static ``version: 0.1.0`` that names no commit at all -- so nothing
in the journal, the health payload or the job records could even state which
code was running.

The fix is a loaded-code identity: the sha of the tree the process imported is
RECORDED AT STARTUP (``deploy_agent.loaded_code``), journalled there, and
compared against the clone at every self-update boundary. The clone-vs-remote
comparison stays -- it is what decides whether to PULL -- but it no longer
decides whether to RE-EXEC.

A third boundary is added for the same reason. ``PRE_ACCEPT`` and
``POST_TERMINAL`` are both job-driven, so an agent that nobody sends a job to
can never pick up its own fix: on 2026-09-14 the only change that would have
published a rebuild command was the deploy-agent change itself, which is the
circularity this closes. ``IDLE_HEARTBEAT`` fires on the poll loop's idle
branch, at a bounded cadence, and only when nothing is in flight.

RESIDUAL 2 -- a successful rebuild asserted a failed phase.

``_deploy_gateway_lane`` runs AFTER ``_compose_up`` has already marked
``Phase.RUNTIME`` SUCCESS. It reopens that phase (``IN_PROGRESS``) and never
closes it. ``JobStore.complete`` calls ``reconcile_terminal_phase_results``
unconditionally -- including on success -- and that rewrites IN_PROGRESS to
FAILED. So every successful DEV rebuild recorded ``runtime: failed`` beside
``status: success``: 14 of 14 job records between 2026-09-12T23:41Z and
2026-09-14T03:22Z, with no runtime-phase error anywhere in the journal, and the
terminal redeploy event on the bus asserting the same contradiction.

The fix is at the cause: the step closes the phase it reopens.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from deploy_agent.events import (
    BuildSource,
    EnumRuntimeLane,
    EnumSelfUpdateBoundary,
    Phase,
    PhaseStatus,
    Scope,
)
from deploy_agent.executor import DeployExecutor
from deploy_agent.job_state import JobStore

pytestmark = pytest.mark.unit

TRACKING_BRANCH = "dev"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _make_origin_and_clone(tmp_path: Path) -> tuple[Path, Path]:
    """A real origin on ``dev`` and a clone of it, both on disk.

    A real clone rather than a stubbed ``_run``: the defect is about which of
    two real shas is compared, and a fake that answers ``rev-parse`` from a
    lookup table can be made to agree with either reading.
    """
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "--initial-branch", TRACKING_BRANCH)
    _git(origin, "config", "user.email", "agent@example.invalid")
    _git(origin, "config", "user.name", "deploy agent test")
    (origin / "agent.py").write_text("print('v1')\n", encoding="utf-8")
    _git(origin, "add", "agent.py")
    _git(origin, "commit", "-m", "v1")

    clone = tmp_path / "clone"
    _git(tmp_path, "clone", str(origin), str(clone))
    _git(clone, "config", "user.email", "agent@example.invalid")
    _git(clone, "config", "user.name", "deploy agent test")
    return origin, clone


def _advance_origin(origin: Path, text: str) -> str:
    (origin / "agent.py").write_text(text, encoding="utf-8")
    _git(origin, "add", "agent.py")
    _git(origin, "commit", "-m", text)
    return _git(origin, "rev-parse", "HEAD")


def _ok(stdout: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


# ---------------------------------------------------------------------------
# Residual 1 -- the loaded-code identity exists and is recorded at startup
# ---------------------------------------------------------------------------


class TestTheLoadedCodeIdentityIsRecorded:
    def test_record_returns_the_clone_head_and_is_readable_afterwards(
        self, tmp_path: Path
    ) -> None:
        from deploy_agent import loaded_code

        _origin, clone = _make_origin_and_clone(tmp_path)
        head = _git(clone, "rev-parse", "HEAD")

        recorded = loaded_code.record_loaded_code_sha(str(clone))

        assert recorded == head
        assert loaded_code.loaded_code_sha() == head

    def test_reading_before_recording_raises_rather_than_guessing(self) -> None:
        """Rule 8: an unrecorded identity fails fast, it does not default.

        A silent fallback here would be the defect again -- the comparison
        would quietly revert to clone-versus-remote and read as healthy.
        """
        from deploy_agent import loaded_code

        loaded_code.reset_loaded_code_sha()
        with pytest.raises(loaded_code.LoadedCodeShaNotRecordedError):
            loaded_code.loaded_code_sha()

    def test_a_directory_that_is_not_a_clone_raises(self, tmp_path: Path) -> None:
        from deploy_agent import loaded_code

        not_a_clone = tmp_path / "plain"
        not_a_clone.mkdir()
        with pytest.raises(loaded_code.LoadedCodeShaUnavailableError):
            loaded_code.record_loaded_code_sha(str(not_a_clone))

    def test_the_agent_records_and_journals_the_loaded_sha_at_startup(self) -> None:
        """The journal must name the code the process loaded.

        Nothing did, before this: the health endpoint's ``version`` is a static
        ``0.1.0`` and the startup line named only the state dir, the broker and
        the lane fence -- so 'which code is this process running' was not an
        answerable question on the live host.
        """
        import inspect

        from deploy_agent import agent as agent_mod

        source = inspect.getsource(agent_mod.DeployAgent.run)
        assert "record_loaded_code_sha" in source, (
            "DeployAgent.run must record the loaded-code identity at startup"
        )


# ---------------------------------------------------------------------------
# Residual 1 -- the decision itself
# ---------------------------------------------------------------------------


class TestTheDiskShaMovingUnderARunningInstance:
    """The live case: an external reconciler advances the clone, not a pull."""

    def test_reexecs_when_the_clone_moved_under_the_process(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deploy_agent import loaded_code

        origin, clone = _make_origin_and_clone(tmp_path)
        monkeypatch.setenv("DEPLOY_AGENT_DIR", str(clone))

        # The process started on v1 and recorded that identity.
        loaded_sha = loaded_code.record_loaded_code_sha(str(clone))

        # The hourly reconciler advances origin AND resets the clone onto it,
        # exactly as the :19 tick does on the lab host. No pull is owed: the
        # clone is already current with the remote.
        advanced = _advance_origin(origin, "print('v2')\n")
        _git(clone, "fetch", "origin", TRACKING_BRANCH)
        _git(clone, "reset", "--hard", f"origin/{TRACKING_BRANCH}")
        assert _git(clone, "rev-parse", "HEAD") == advanced
        assert advanced != loaded_sha

        executor = DeployExecutor()
        with patch("os.execv") as mock_execv:
            executor.self_update(boundary=EnumSelfUpdateBoundary.POST_TERMINAL)

        mock_execv.assert_called_once()

    def test_does_not_reexec_when_the_process_is_running_the_clones_code(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The positive control for the test above.

        Same machinery, same real clone, nothing moved -- so a re-exec here
        would mean the new comparison fires on every boundary rather than on a
        real divergence.
        """
        from deploy_agent import loaded_code

        _origin, clone = _make_origin_and_clone(tmp_path)
        monkeypatch.setenv("DEPLOY_AGENT_DIR", str(clone))
        loaded_code.record_loaded_code_sha(str(clone))

        executor = DeployExecutor()
        with patch("os.execv") as mock_execv:
            executor.self_update(boundary=EnumSelfUpdateBoundary.POST_TERMINAL)

        mock_execv.assert_not_called()

    def test_pulls_and_reexecs_when_the_clone_is_behind_the_remote(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pre-existing path still works: clone behind remote, pull, re-exec."""
        from deploy_agent import loaded_code

        origin, clone = _make_origin_and_clone(tmp_path)
        monkeypatch.setenv("DEPLOY_AGENT_DIR", str(clone))
        loaded_code.record_loaded_code_sha(str(clone))
        advanced = _advance_origin(origin, "print('v2')\n")

        executor = DeployExecutor()
        with (
            patch("os.execv") as mock_execv,
            patch("deploy_agent.executor._uv_sync_after_pull", return_value=None),
        ):
            executor.self_update(boundary=EnumSelfUpdateBoundary.POST_TERMINAL)

        assert _git(clone, "rev-parse", "HEAD") == advanced, (
            "the clone must still be brought up to the remote"
        )
        mock_execv.assert_called_once()

    def test_an_unrecorded_identity_refuses_rather_than_falling_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deploy_agent import loaded_code

        _origin, clone = _make_origin_and_clone(tmp_path)
        monkeypatch.setenv("DEPLOY_AGENT_DIR", str(clone))
        loaded_code.reset_loaded_code_sha()

        executor = DeployExecutor()
        with (
            patch("os.execv") as mock_execv,
            pytest.raises(loaded_code.LoadedCodeShaNotRecordedError),
        ):
            executor.self_update(boundary=EnumSelfUpdateBoundary.POST_TERMINAL)

        mock_execv.assert_not_called()


class TestTheJournalNamesBothShas:
    def test_the_reexec_line_carries_loaded_and_clone(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from deploy_agent import loaded_code

        origin, clone = _make_origin_and_clone(tmp_path)
        monkeypatch.setenv("DEPLOY_AGENT_DIR", str(clone))
        loaded_sha = loaded_code.record_loaded_code_sha(str(clone))
        advanced = _advance_origin(origin, "print('v2')\n")
        _git(clone, "fetch", "origin", TRACKING_BRANCH)
        _git(clone, "reset", "--hard", f"origin/{TRACKING_BRANCH}")

        executor = DeployExecutor()
        with (
            caplog.at_level("INFO", logger="deploy_agent.executor"),
            patch("os.execv"),
        ):
            executor.self_update(boundary=EnumSelfUpdateBoundary.POST_TERMINAL)

        assert loaded_sha[:12] in caplog.text
        assert advanced[:12] in caplog.text


# ---------------------------------------------------------------------------
# Residual 1 -- the idle boundary
# ---------------------------------------------------------------------------


class TestTheIdleHeartbeatBoundary:
    def test_the_boundary_is_declared(self) -> None:
        assert EnumSelfUpdateBoundary.IDLE_HEARTBEAT.value == "idle_heartbeat"

    def test_an_idle_agent_self_updates_once_the_interval_has_elapsed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fix to the agent must not need a job to arrive to be picked up.

        This is the circularity the live incident closed on: the only merge
        that would have published a rebuild command was the deploy-agent merge
        itself, and both existing boundaries are job-driven.
        """
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
        monkeypatch.setattr("deploy_agent.agent.STATE_DIR", tmp_path / "jobs")
        from deploy_agent.agent import DeployAgent

        agent = DeployAgent()
        agent.job_store = JobStore(state_dir=tmp_path / "jobs")
        seen: list[EnumSelfUpdateBoundary] = []

        def fake_self_update(*, boundary, skip=False, on_before_reexec=None) -> None:
            seen.append(boundary)

        agent.executor.self_update = fake_self_update  # type: ignore[method-assign]
        # None is the "never checked" sentinel, so the check is due. Not 0.0:
        # time.monotonic()'s zero is the boot instant on Linux, so on a
        # freshly-booted CI runner 0.0 reads as "checked seconds ago" and the
        # check is skipped -- which is how these two cases passed on a
        # workstation and failed on a runner.
        agent._last_idle_self_update = None

        agent._maybe_self_update_idle()

        assert seen == [EnumSelfUpdateBoundary.IDLE_HEARTBEAT]

    def test_the_idle_check_is_rate_limited(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The poll loop idles once a second; the git fetch must not."""
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
        monkeypatch.setattr("deploy_agent.agent.STATE_DIR", tmp_path / "jobs")
        from deploy_agent.agent import DeployAgent

        agent = DeployAgent()
        agent.job_store = JobStore(state_dir=tmp_path / "jobs")
        calls: list[object] = []
        agent.executor.self_update = lambda **kw: calls.append(kw)  # type: ignore[method-assign]
        # None is the "never checked" sentinel, so the check is due. Not 0.0:
        # time.monotonic()'s zero is the boot instant on Linux, so on a
        # freshly-booted CI runner 0.0 reads as "checked seconds ago" and the
        # check is skipped -- which is how these two cases passed on a
        # workstation and failed on a runner.
        agent._last_idle_self_update = None

        agent._maybe_self_update_idle()
        agent._maybe_self_update_idle()
        agent._maybe_self_update_idle()

        assert len(calls) == 1, f"one fetch per interval, not per poll: {calls}"

    def test_a_job_in_flight_blocks_the_idle_reexec(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A re-exec must never happen with work in flight (OMN-16442)."""
        import uuid

        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
        monkeypatch.setattr("deploy_agent.agent.STATE_DIR", tmp_path / "jobs")
        from deploy_agent.agent import DeployAgent

        agent = DeployAgent()
        agent.job_store = JobStore(state_dir=tmp_path / "jobs")
        agent.job_store.accept(correlation_id=uuid.uuid4(), command={})
        calls: list[object] = []
        agent.executor.self_update = lambda **kw: calls.append(kw)  # type: ignore[method-assign]
        # None is the "never checked" sentinel, so the check is due. Not 0.0:
        # time.monotonic()'s zero is the boot instant on Linux, so on a
        # freshly-booted CI runner 0.0 reads as "checked seconds ago" and the
        # check is skipped -- which is how these two cases passed on a
        # workstation and failed on a runner.
        agent._last_idle_self_update = None

        agent._maybe_self_update_idle()

        assert calls == [], "an accepted job was in flight and must have blocked it"

    def test_a_freshly_booted_host_does_not_suppress_the_first_check(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The "never checked" sentinel must not be a value the clock produces.

        ``time.monotonic()``'s zero is an arbitrary reference point -- on Linux
        the boot instant -- so a ``0.0`` sentinel means "checked at boot", and
        on a host that has been up for less than the interval the check is
        suppressed forever. This is not hypothetical: the two cases above passed
        on a workstation with days of uptime and failed on a CI runner minutes
        old, which is the only reason it was found before shipping.
        """
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
        monkeypatch.setattr("deploy_agent.agent.STATE_DIR", tmp_path / "jobs")
        monkeypatch.setattr("deploy_agent.agent.time.monotonic", lambda: 5.0)
        from deploy_agent.agent import DeployAgent

        agent = DeployAgent()
        agent.job_store = JobStore(state_dir=tmp_path / "jobs")
        calls: list[object] = []
        agent.executor.self_update = lambda **kw: calls.append(kw)  # type: ignore[method-assign]

        assert agent._last_idle_self_update is None, (
            "a fresh agent must be due, and must say so with a sentinel the "
            "clock cannot produce"
        )
        agent._maybe_self_update_idle()

        assert len(calls) == 1, (
            "five seconds of host uptime suppressed the first idle check: the "
            "sentinel is being compared as if it were a timestamp"
        )

    def test_the_idle_branch_of_the_poll_loop_calls_it(self) -> None:
        import inspect

        from deploy_agent import agent as agent_mod

        source = inspect.getsource(agent_mod.DeployAgent.run)
        assert "_maybe_self_update_idle" in source, (
            "the poll loop's idle branch must reach the idle boundary"
        )


# ---------------------------------------------------------------------------
# Residual 2 -- a successful rebuild must not record a failed phase
# ---------------------------------------------------------------------------


class TestTheGatewayStepClosesThePhaseItReopens:
    pytestmark = pytest.mark.gateway_lane

    def test_the_runtime_phase_ends_success_after_the_gateway_step(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        executor = DeployExecutor()
        seen: list[tuple[Phase, PhaseStatus]] = []

        with patch("deploy_agent.executor._run", return_value=_ok()):
            executor._deploy_gateway_lane(
                lambda phase, status: seen.append((phase, status)),
                lane=EnumRuntimeLane.DEV,
                build_source=BuildSource.RELEASE,
                targets=["gateway-forwarder"],
                git_ref="origin/dev",
            )

        assert seen, "the step must report its phase at all"
        assert seen[0] == (Phase.RUNTIME, PhaseStatus.IN_PROGRESS)
        assert seen[-1] == (Phase.RUNTIME, PhaseStatus.SUCCESS), (
            "the gateway step reopens Phase.RUNTIME and must close it; leaving "
            "it IN_PROGRESS is what reconcile_terminal_phase_results rewrites "
            "to FAILED on every SUCCESSFUL dev rebuild"
        )

    def test_a_successful_dev_rebuild_record_carries_no_failed_phase(
        self, tmp_path: Path
    ) -> None:
        """The live record shape, reproduced end to end.

        14 of 14 DEV job records between 2026-09-12T23:41Z and
        2026-09-14T03:22Z carried ``runtime: failed`` beside ``status:
        success``.
        """
        import uuid

        executor = DeployExecutor()
        store = JobStore(state_dir=tmp_path / "jobs")
        cid = uuid.uuid4()
        store.accept(correlation_id=cid, command={})

        def on_phase_update(phase: Phase, status: PhaseStatus) -> None:
            store.update_phase(cid, phase, status)

        with (
            patch("deploy_agent.executor._run", return_value=_ok()),
            patch.object(DeployExecutor, "_compose_build", return_value=None),
            patch.object(DeployExecutor, "_compose_up", return_value=None),
            patch.object(
                DeployExecutor, "_build_dev_lane_only_services", return_value=None
            ),
        ):
            # the runtime phase as _compose_up reports it, then the gateway step
            on_phase_update(Phase.RUNTIME, PhaseStatus.SUCCESS)
            executor._deploy_gateway_lane(
                on_phase_update,
                lane=EnumRuntimeLane.DEV,
                build_source=BuildSource.RELEASE,
                targets=["gateway-forwarder"],
                git_ref="origin/dev",
            )

        for phase in (Phase.PREFLIGHT, Phase.GIT, Phase.COMPOSE_GEN, Phase.SEED):
            on_phase_update(phase, PhaseStatus.SUCCESS)
        on_phase_update(Phase.CORE, PhaseStatus.SUCCESS)
        on_phase_update(Phase.VERIFICATION, PhaseStatus.SUCCESS)

        job = store.complete(cid, "success")

        failed = {
            phase.value
            for phase, status in job.phase_results.items()
            if status == PhaseStatus.FAILED
        }
        assert failed == set(), (
            f"a job recorded status=success must assert no failed phase: {failed}"
        )
        assert job.phase_results[Phase.RUNTIME] == PhaseStatus.SUCCESS
