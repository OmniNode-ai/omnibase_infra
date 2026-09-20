# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A failed build must not reach the wire as ``status: success`` (OMN-18861).

LIVE, read off ``onex.evt.deploy.rebuild-completed.v1`` on the .201 dev bus,
2026-09-20. Of the last 120 events on partition 0 (offsets 798-917):

* 76 carried ``status: "success"`` with an empty ``errors`` list -- correct;
* 21 carried ``status: "failed"`` with a non-empty one -- also correct, and the
  positive control for the count below: the derivation DOES reach "failed" when
  the failure raises one step later, out of the compose-up leg, which marks its
  phase before it raises;
* **23 carried** ``status: "success"`` **beside a hard build failure in**
  ``errors``, from 2026-09-16T20:51:10Z to 2026-09-19T23:57:02Z.

Offset 916 is the one reproduced below. Its job record on the host,
``445bdfa9-4c86-4e45-bf68-b8e3ed534a7f``, is durably ``"status": "failed"``, and
the rule-24 compose-dev lab-pass receipt emitted for the sibling job on the same
lane reads ``"result": "FAIL"``. Three surfaces, one liar.

Two composing defects produced it, and both halves are pinned here:

1. ``ModelRebuildCompleted.status`` derived the verdict from phase strings and
   dropped SKIPPED, so it could not see a failure recorded only in ``errors``;
2. ``DeployExecutor._compose_build`` took ``on_phase_update`` and never called
   it, so a build failure left its phase unmarked, ``core``/``runtime`` settled
   to SKIPPED as "never reached", and the derivation had nothing to drop but
   successes.

A fix to either half alone is insufficient, which is why both are tested: the
phase marker only covers the raise sites it is written at, and the errors
premise only covers failures that recorded an error.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from deploy_agent.events import (
    ModelRebuildCompleted,
    Phase,
    PhaseStatus,
    Scope,
)
from deploy_agent.executor import DeployExecutor
from deploy_agent.job_state import JobState
from deploy_agent.publisher import build_completion_payload
from pydantic import ValidationError

pytestmark = pytest.mark.unit


# The exact phase map job 445bdfa9 carried, with PUBLISH stripped as the
# terminal event strips it. Every deploy phase after `seed` reads SKIPPED
# because the runtime image build raised before any of them was marked.
_LIVE_445BDFA9_PHASES: dict[Phase, PhaseStatus] = {
    Phase.PREFLIGHT: PhaseStatus.SUCCESS,
    Phase.GIT: PhaseStatus.SUCCESS,
    Phase.COMPOSE_GEN: PhaseStatus.SUCCESS,
    Phase.SEED: PhaseStatus.SUCCESS,
    Phase.CORE: PhaseStatus.SKIPPED,
    Phase.RUNTIME: PhaseStatus.SKIPPED,
    Phase.VERIFICATION: PhaseStatus.SKIPPED,
}

# Verbatim head of the error that job carried, truncated at the build outcome
# token and the failing target. The token is what a reader keys on.
_LIVE_445BDFA9_ERROR = (
    "runtime_image_build_errored: docker compose build for profile 'runtime' "
    "exited 1 inside its 1095s ceiling -- this is a BROKEN BUILD, not an "
    "exhausted budget, and retrying it unchanged buys nothing: target "
    "skill-lifecycle-consumer: failed to solve: process "
    "/app/.venv/bin/python /workspace/compute_workspace_provenance.py "
    "did not complete successfully: exit code: 1"
)


# What a scope-default FULL deploy on the dev lane actually brings up, trimmed.
# Any non-empty list exercises the third premise identically; these are real
# service names so the fixture cannot be mistaken for a placeholder.
_RESTARTED = ["omninode-runtime", "runtime-effects", "runtime-worker"]


def _completed(
    phase_results: dict[Phase, PhaseStatus],
    errors: list[str],
    services_restarted: list[str] | None = None,
) -> ModelRebuildCompleted:
    now = datetime.now(UTC)
    return ModelRebuildCompleted(
        correlation_id=uuid4(),
        requested_git_ref="cccfec24332723a1ace23afb659091582a8e3b2f",
        git_sha="cccfec24332723a1ace23afb659091582a8e3b2f",
        started_at=now,
        completed_at=now,
        duration_seconds=75.6,
        scope="full",
        runtime_lane="dev",
        phase_results=phase_results,
        errors=errors,
        services_restarted=(
            list(_RESTARTED) if services_restarted is None else services_restarted
        ),
    )


def _job(phase_results: dict[Phase, PhaseStatus], errors: list[str]) -> JobState:
    job = JobState(
        correlation_id=uuid4(),
        command={
            "runtime_lane": "dev",
            "scope": "full",
            "services": [],
            "git_ref": "cccfec24332723a1ace23afb659091582a8e3b2f",
        },
    )
    job.phase_results = dict(phase_results)
    job.completed_at = datetime.now(UTC)
    job.status = "failed"
    job.errors = list(errors)
    return job


class TestTheVerdictFailsClosedOnErrors:
    """AC1 / AC3: a recorded error outranks a phase map of successes."""

    def test_the_live_false_green_now_reads_failed(self) -> None:
        """Offset 916 reproduced exactly, empty services_restarted included."""
        completed = _completed(
            _LIVE_445BDFA9_PHASES, [_LIVE_445BDFA9_ERROR], services_restarted=[]
        )
        assert completed.status == "failed"

    def test_the_wire_payload_agrees_with_the_job_record(self) -> None:
        """AC3. The record said failed, the receipt said FAIL, the event said success."""
        job = _job(_LIVE_445BDFA9_PHASES, [_LIVE_445BDFA9_ERROR])
        payload = build_completion_payload(job, "cccfec24")
        assert job.status == "failed"
        assert payload["status"] == "failed"
        # The phase map is unchanged -- the verdict moved, the record did not.
        assert payload["phase_results"]["runtime"] == "skipped"

    def test_an_error_outranks_a_wholly_successful_phase_map(self) -> None:
        """No phase map, however clean, can carry a recorded failure to success."""
        every_phase_green = dict.fromkeys(_LIVE_445BDFA9_PHASES, PhaseStatus.SUCCESS)
        assert _completed(every_phase_green, ["anything at all"]).status == "failed"

    def test_the_check_is_not_vacuous(self) -> None:
        """The same phase map with no errors is still success (positive control).

        Without this the fix could be `return "failed"` and every test above
        would pass. 76 of the 120 live events are this shape.
        """
        every_phase_green = dict.fromkeys(_LIVE_445BDFA9_PHASES, PhaseStatus.SUCCESS)
        assert _completed(every_phase_green, []).status == "success"

    def test_a_failed_phase_still_fails_with_no_errors_recorded(self) -> None:
        """The phase-string derivation is retained, not replaced."""
        phases = dict(_LIVE_445BDFA9_PHASES)
        phases[Phase.RUNTIME] = PhaseStatus.FAILED
        assert _completed(phases, []).status == "failed"


class TestADeployThatRestartedNothingIsNotASuccess:
    """The third premise: a failure that was never RECORDED is still a failure.

    The errors premise catches a recorded one. This catches the case where a
    future path raises without writing an error, which the phase map alone
    would still read as success.
    """

    def test_an_empty_services_restarted_fails_on_a_clean_phase_map(self) -> None:
        every_phase_green = dict.fromkeys(_LIVE_445BDFA9_PHASES, PhaseStatus.SUCCESS)
        assert _completed(every_phase_green, [], services_restarted=[]).status == (
            "failed"
        )

    def test_the_premise_is_independent_of_errors(self) -> None:
        """No error recorded, nothing restarted, every phase green: still failed."""
        assert _completed(_LIVE_445BDFA9_PHASES, [], services_restarted=[]).status == (
            "failed"
        )

    def test_one_restarted_service_is_enough_to_clear_it(self) -> None:
        """Not vacuous: the premise is emptiness, not a count or a whitelist."""
        every_phase_green = dict.fromkeys(_LIVE_445BDFA9_PHASES, PhaseStatus.SUCCESS)
        completed = _completed(
            every_phase_green, [], services_restarted=["omninode-runtime"]
        )
        assert completed.status == "success"

    def test_a_gateway_only_deploy_is_not_false_redded(self) -> None:
        """The shape a per-scope phase requirement would have broken.

        A command naming only gateway services returns from ``rebuild_scope``
        before ``Phase.CORE`` or ``Phase.RUNTIME`` is ever marked, so both are
        SKIPPED on a wholly successful deploy. It restarted something, so it
        reads success -- which is why the premise is "restarted nothing" and
        not "every leg of the scope ran".
        """
        gateway_shape = dict(_LIVE_445BDFA9_PHASES)
        gateway_shape[Phase.VERIFICATION] = PhaseStatus.SUCCESS
        completed = _completed(
            gateway_shape, [], services_restarted=["onex-gateway-forwarder"]
        )
        assert completed.phase_results[Phase.CORE] == PhaseStatus.SKIPPED
        assert completed.phase_results[Phase.RUNTIME] == PhaseStatus.SKIPPED
        assert completed.status == "success"


class TestTheVerdictStaysDerived:
    """AC4: an emitter cannot assert a verdict, only record the facts behind one."""

    def test_status_is_not_an_input_field(self) -> None:
        assert "status" not in ModelRebuildCompleted.model_fields

    def test_passing_a_status_is_refused(self) -> None:
        now = datetime.now(UTC)
        with pytest.raises(ValidationError):
            ModelRebuildCompleted(
                correlation_id=uuid4(),
                requested_git_ref="origin/dev",
                git_sha="cccfec24",
                started_at=now,
                completed_at=now,
                duration_seconds=1.0,
                scope="full",
                runtime_lane="dev",
                phase_results={Phase.RUNTIME: PhaseStatus.SUCCESS},
                errors=[_LIVE_445BDFA9_ERROR],
                status="success",  # type: ignore[call-arg]
            )

    def test_status_is_serialized_on_the_wire(self) -> None:
        payload = _completed(_LIVE_445BDFA9_PHASES, [_LIVE_445BDFA9_ERROR]).model_dump(
            mode="json"
        )
        assert payload["status"] == "failed"


class _RecordingPhases:
    """Collects every ``on_phase_update`` call the executor makes."""

    def __init__(self) -> None:
        self.calls: list[tuple[Phase, PhaseStatus]] = []

    def __call__(self, phase: Phase, status: PhaseStatus) -> None:
        self.calls.append((phase, status))


def _build_executor_with_result(
    monkeypatch: pytest.MonkeyPatch, result: Any
) -> DeployExecutor:
    """A DeployExecutor whose build subprocess returns ``result``.

    Everything the build does before invoking docker is stubbed out: this test
    is about which phase verdict the failure path records, not about staging,
    provenance or ref resolution.
    """
    executor = DeployExecutor()
    monkeypatch.setattr(
        "deploy_agent.executor.assert_release_build_promoted",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        DeployExecutor, "_resolve_plugin_ref", lambda self, path, fallback: fallback
    )
    monkeypatch.setattr(
        "deploy_agent.executor.load_tracking_ref_from_env", lambda: "dev"
    )

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr("deploy_agent.executor._run", fake_run)
    return executor


class TestABuildFailureNamesItsPhase:
    """AC2: the phase that raised is FAILED on the record, not "never reached"."""

    def test_a_broken_build_marks_its_phase_failed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        executor = _build_executor_with_result(
            monkeypatch,
            subprocess.CompletedProcess(
                args=["docker"], returncode=1, stdout="", stderr="boom"
            ),
        )
        phases = _RecordingPhases()
        with pytest.raises(RuntimeError, match="runtime_image_build_errored"):
            executor._compose_build(Scope.RUNTIME, "cccfec24", phases)
        assert (Phase.RUNTIME, PhaseStatus.FAILED) in phases.calls

    def test_a_killed_build_marks_its_phase_failed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        executor = _build_executor_with_result(
            monkeypatch,
            subprocess.TimeoutExpired(cmd=["docker"], timeout=1095),
        )
        phases = _RecordingPhases()
        with pytest.raises(RuntimeError, match="runtime_image_build_budget_exhausted"):
            executor._compose_build(Scope.RUNTIME, "cccfec24", phases)
        assert (Phase.RUNTIME, PhaseStatus.FAILED) in phases.calls

    def test_the_core_scope_names_the_core_phase(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        executor = _build_executor_with_result(
            monkeypatch,
            subprocess.CompletedProcess(
                args=["docker"], returncode=1, stdout="", stderr="boom"
            ),
        )
        phases = _RecordingPhases()
        with pytest.raises(RuntimeError):
            executor._compose_build(Scope.CORE, "cccfec24", phases)
        assert (Phase.CORE, PhaseStatus.FAILED) in phases.calls
        assert (Phase.RUNTIME, PhaseStatus.FAILED) not in phases.calls

    def test_a_successful_build_marks_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Not vacuous, and deliberately silent.

        ``_compose_up`` owns this phase's SUCCESS verdict. A marker written on
        the way in would report the core phase failed whenever a later runtime
        build raised, which is a false red in place of a false green.
        """
        executor = _build_executor_with_result(
            monkeypatch,
            subprocess.CompletedProcess(
                args=["docker"], returncode=0, stdout="", stderr=""
            ),
        )
        phases = _RecordingPhases()
        executor._compose_build(Scope.RUNTIME, "cccfec24", phases)
        assert phases.calls == []
