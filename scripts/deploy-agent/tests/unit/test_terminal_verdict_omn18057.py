# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A terminal event must carry a verdict, not an unfinished sentence (OMN-18057).

LIVE, read off ``onex.evt.deploy.rebuild-completed.v1`` on the .201 dev bus,
2026-09-08:

* offset 102, command 23edaf62 -- ``completed_at`` set, ``duration_seconds:
  614.0``, and ``phase_results: {..., "runtime": "in_progress", "publish":
  "in_progress"}``, ``services_restarted: []``, ``health_checks: []``. The event
  asserts the deploy is over and simultaneously refuses to say how it ended.
* offset 96, command ddc711ad -- the same shape with ``compose_gen:
  in_progress``. Its job file on the host still reads ``"compose_gen":
  "in_progress"`` beside ``"status": "failed"``.
* no ``status`` key on the wire at all: the payload was a hand-built dict that
  never went through ``ModelRebuildCompleted``, so the model's own verdict never
  reached a consumer.

These tests pin all four halves: reconciliation at the job boundary, refusal at
the model, ``PUBLISH`` stripped from an event that IS the publish, and
``services_restarted`` reporting what the deploy actually restarted.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from deploy_agent.events import (
    DEPLOY_PHASE_ORDER,
    ModelContainerResidue,
    ModelHealthCheck,
    ModelRebuildCompleted,
    Phase,
    PhaseStatus,
)
from deploy_agent.job_state import (
    JobState,
    JobStore,
    reconcile_terminal_phase_results,
)
from deploy_agent.publisher import build_completion_payload
from pydantic import ValidationError

pytestmark = pytest.mark.unit


# The exact phase map command 23edaf62 carried when it raised out of RUNTIME.
_LIVE_23EDAF62_PHASES: dict[Phase, PhaseStatus] = {
    Phase.PREFLIGHT: PhaseStatus.SUCCESS,
    Phase.GIT: PhaseStatus.SUCCESS,
    Phase.COMPOSE_GEN: PhaseStatus.SUCCESS,
    Phase.SEED: PhaseStatus.SUCCESS,
    Phase.CORE: PhaseStatus.SUCCESS,
    Phase.RUNTIME: PhaseStatus.IN_PROGRESS,
}


class TestReconciliation:
    def test_the_phase_that_raised_becomes_failed(self) -> None:
        settled = reconcile_terminal_phase_results(_LIVE_23EDAF62_PHASES)
        assert settled[Phase.RUNTIME] == PhaseStatus.FAILED

    def test_every_unreached_phase_becomes_skipped(self) -> None:
        settled = reconcile_terminal_phase_results(_LIVE_23EDAF62_PHASES)
        assert settled[Phase.VERIFICATION] == PhaseStatus.SKIPPED

    def test_no_deploy_phase_is_left_absent(self) -> None:
        """An absent phase is indistinguishable from a lost result."""
        settled = reconcile_terminal_phase_results(
            {Phase.PREFLIGHT: PhaseStatus.SUCCESS}
        )
        assert set(settled) >= set(DEPLOY_PHASE_ORDER)
        assert all(
            settled[phase] != PhaseStatus.IN_PROGRESS for phase in DEPLOY_PHASE_ORDER
        )

    def test_settled_results_are_untouched(self) -> None:
        settled = reconcile_terminal_phase_results(_LIVE_23EDAF62_PHASES)
        for phase in (Phase.PREFLIGHT, Phase.GIT, Phase.COMPOSE_GEN, Phase.CORE):
            assert settled[phase] == PhaseStatus.SUCCESS

    def test_job_store_complete_settles_the_record(self, tmp_path: Path) -> None:
        store = JobStore(state_dir=tmp_path)
        cid = uuid4()
        store.accept(cid, {"runtime_lane": "dev", "scope": "full"})
        for phase, status in _LIVE_23EDAF62_PHASES.items():
            store.update_phase(cid, phase, status)

        job = store.complete(cid, status="failed", errors=["timed out after 300s"])

        assert job.phase_results[Phase.RUNTIME] == PhaseStatus.FAILED
        assert job.phase_results[Phase.VERIFICATION] == PhaseStatus.SKIPPED
        assert PhaseStatus.IN_PROGRESS not in job.phase_results.values()

    def test_crash_recovery_settles_every_phase_not_only_the_current_one(
        self, tmp_path: Path
    ) -> None:
        store = JobStore(state_dir=tmp_path)
        cid = uuid4()
        store.accept(cid, {"runtime_lane": "dev", "scope": "full"})
        for phase, status in _LIVE_23EDAF62_PHASES.items():
            store.update_phase(cid, phase, status)

        (recovered,) = store.recover_crashed_jobs()

        assert recovered.phase_results[Phase.RUNTIME] == PhaseStatus.FAILED
        assert recovered.phase_results[Phase.VERIFICATION] == PhaseStatus.SKIPPED
        assert recovered.result_publish_pending is True


class TestTheModelRefusesAnUnfinishedVerdict:
    @staticmethod
    def _completed(phase_results: dict[Phase, PhaseStatus]) -> ModelRebuildCompleted:
        now = datetime.now(UTC)
        return ModelRebuildCompleted(
            correlation_id=uuid4(),
            requested_git_ref="origin/dev",
            git_sha="3461e4b0aeae4690fc5bb52c56aef63db1227109",
            started_at=now,
            completed_at=now,
            duration_seconds=614.0,
            scope="full",
            runtime_lane="dev",
            phase_results=phase_results,
        )

    def test_in_progress_is_rejected(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            self._completed(dict(_LIVE_23EDAF62_PHASES))
        assert "runtime" in str(excinfo.value)

    def test_publish_is_rejected(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            self._completed(
                {Phase.RUNTIME: PhaseStatus.SUCCESS, Phase.PUBLISH: PhaseStatus.SUCCESS}
            )
        assert "PUBLISH" in str(excinfo.value) or "publish" in str(excinfo.value)

    def test_settled_results_are_accepted_and_yield_a_verdict(self) -> None:
        settled = reconcile_terminal_phase_results(_LIVE_23EDAF62_PHASES)
        assert self._completed(settled).status == "failed"

    def test_a_clean_deploy_yields_success(self) -> None:
        settled = reconcile_terminal_phase_results(
            dict.fromkeys(DEPLOY_PHASE_ORDER, PhaseStatus.SUCCESS)
        )
        assert self._completed(settled).status == "success"


class TestTheTerminalPayload:
    @staticmethod
    def _job(phase_results: dict[Phase, PhaseStatus]) -> JobState:
        job = JobState(
            correlation_id=uuid4(),
            command={
                "runtime_lane": "dev",
                "scope": "full",
                # A scope-default deploy: the command names NO services, which
                # is why the old payload reported services_restarted=[].
                "services": [],
                "git_ref": "origin/dev",
            },
        )
        job.phase_results = dict(phase_results)
        job.completed_at = datetime.now(UTC)
        job.status = "failed"
        return job

    def test_a_raised_phase_never_reaches_the_wire_as_in_progress(self) -> None:
        job = self._job(reconcile_terminal_phase_results(_LIVE_23EDAF62_PHASES))
        payload = build_completion_payload(job, "3461e4b0")
        assert "in_progress" not in payload["phase_results"].values()
        assert payload["phase_results"]["runtime"] == "failed"
        assert payload["phase_results"]["verification"] == "skipped"

    def test_publish_is_absent_from_the_event_that_is_the_publish(self) -> None:
        settled = reconcile_terminal_phase_results(_LIVE_23EDAF62_PHASES)
        job = self._job(settled)
        job.phase_results[Phase.PUBLISH] = PhaseStatus.IN_PROGRESS
        payload = build_completion_payload(job, "3461e4b0")
        assert "publish" not in payload["phase_results"]

    def test_the_wire_carries_a_verdict(self) -> None:
        """No consumer should have to re-derive a status from phase strings."""
        job = self._job(reconcile_terminal_phase_results(_LIVE_23EDAF62_PHASES))
        payload = build_completion_payload(job, "3461e4b0")
        assert payload["status"] == "failed"

    def test_services_restarted_reports_what_the_deploy_did(self) -> None:
        job = self._job(
            reconcile_terminal_phase_results(
                dict.fromkeys(DEPLOY_PHASE_ORDER, PhaseStatus.SUCCESS)
            )
        )
        payload = build_completion_payload(
            job,
            "3461e4b0",
            [
                ModelHealthCheck(
                    service="omninode-runtime", endpoint=":8085", status="pass"
                )
            ],
            services_restarted=["omninode-runtime", "runtime-effects"],
        )
        assert payload["services_restarted"] == ["omninode-runtime", "runtime-effects"]
        assert payload["health_checks"][0]["service"] == "omninode-runtime"
        assert payload["status"] == "success"

    def test_residue_is_recorded_on_the_event(self) -> None:
        job = self._job(reconcile_terminal_phase_results(_LIVE_23EDAF62_PHASES))
        payload = build_completion_payload(
            job,
            "3461e4b0",
            container_residue=[
                ModelContainerResidue(service="runtime-effects", state="created"),
                ModelContainerResidue(
                    service="omninode-contract-resolver",
                    state="created",
                    recovered=True,
                ),
            ],
        )
        residue = {item["service"]: item for item in payload["container_residue"]}
        assert residue["runtime-effects"]["recovered"] is False
        assert residue["omninode-contract-resolver"]["recovered"] is True
