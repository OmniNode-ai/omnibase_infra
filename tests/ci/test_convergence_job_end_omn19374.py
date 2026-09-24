# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The compose-dev receipt's generation binding survives an in-job recreate (OMN-19374).

MEASURED, receipt artifact 10789633291 (run 35953952476, merge 0e62809004e5):
``probe_generation_bound`` was the only failing check. Convergence had read
``omninode-runtime`` ``0f023e97c811``; the probe read ``c77eb359849b``. Between
the two, deploy-agent job 594a8aad's own post-deploy verification had
force-recreated the runtime once and ended ``success`` -- same image, same
revision, a new container. The same shape failed the first receipt of every
runtime merge on 2026-09-24 that runtime-train recorded.

The convergence guard is satisfied by the running container's revision LABEL,
set at create time, minutes before the agent's verification. So the guard now
also waits for the agent's job to END, and rebinds the generation to the
container the job left running ONLY when the job's own record names a
recovered in-job recreate of that container and the new container carries the
same image and revision. Every other movement keeps the converged binding and
still fails ``probe_generation_bound`` (AC3's positive controls).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts.ci import check_dev_lane_staleness as staleness_module
from scripts.ci.check_dev_lane_staleness import (
    JOB_END_POLL_SECONDS,
    LaneRevision,
    ModelJobEnd,
    bind_generation_to_job_end,
    wait_for_agent_job_end,
)
from scripts.ci.lab_pass_receipt import ModelLaneGeneration, generation_check

pytestmark = pytest.mark.unit

AGENT = "http://agent.invalid:8098"
CID = "594a8aad-1c82-4f10-b66c-f718782f2716"
RUNNER_CID = "0805d076-63be-47b6-b5af-13dd674a0f98"
MERGE_SHA = "0e62809004e5aa11bb22cc33dd44ee55ff667788"
IMAGE = "sha256:" + "1" * 64
OTHER_IMAGE = "sha256:" + "2" * 64
OLD_ID = "0f023e97c811" + "0" * 52
NEW_ID = "c77eb359849b" + "0" * 52
DISPLACED_ID = "d15c1ace0000" + "0" * 52

RECOVERED_RUNTIME: dict[str, Any] = {
    "service": "omninode-runtime",
    "lane": "dev",
    "compose_project": "omnibase-infra",
    "endpoint": "http://localhost:8085/health",
    "outcome": "recovered",
    "recreate_returncode": 0,
    "readiness_wait_seconds": 291.0,
    "readiness_budget_seconds": 600,
    "detail": "",
}


def _generation(
    container_id: str, *, image: str = IMAGE, revision: str = MERGE_SHA
) -> ModelLaneGeneration:
    return ModelLaneGeneration(
        container="omninode-runtime",
        container_id=container_id,
        image=image,
        revision=revision,
    )


def _ended(
    *, status: str = "success", recreates: tuple[dict[str, Any], ...] = ()
) -> ModelJobEnd:
    return ModelJobEnd(
        ended=True,
        correlation_id=CID,
        status=status,
        completed_at="2026-09-24T04:19:00+00:00",
        verify_recreate=recreates,
    )


# --- AC1: the in-job recreate is bound, and the probe agrees -----------------


def test_an_in_job_recreate_of_the_verified_build_rebinds_to_the_new_container() -> (
    None
):
    bound, note = bind_generation_to_job_end(
        _generation(OLD_ID), _generation(NEW_ID), _ended(recreates=(RECOVERED_RUNTIME,))
    )

    assert bound == _generation(NEW_ID)
    assert "rebound" in note
    # The probe then reads the container the job left running, and agrees.
    assert generation_check(bound, _generation(NEW_ID)).ok


def test_before_this_fix_the_same_reads_failed_the_binding() -> None:
    """The measured receipt, reproduced: converged id vs post-recreate id."""
    assert not generation_check(_generation(OLD_ID), _generation(NEW_ID)).ok


# --- AC3: positive controls -- nothing else rebinds ---------------------------


def test_no_recreate_and_no_movement_keeps_the_converged_binding() -> None:
    bound, note = bind_generation_to_job_end(
        _generation(OLD_ID), _generation(OLD_ID), _ended()
    )

    assert bound == _generation(OLD_ID)
    assert note == ""
    assert generation_check(bound, _generation(OLD_ID)).ok


@pytest.mark.parametrize(
    ("after_job", "why"),
    [
        (_generation(NEW_ID, image=OTHER_IMAGE), "a different image"),
        (_generation(NEW_ID, revision="f" * 40), "a different revision"),
    ],
)
def test_a_recreate_that_left_the_wrong_build_running_still_fails(
    after_job: ModelLaneGeneration, why: str
) -> None:
    bound, note = bind_generation_to_job_end(
        _generation(OLD_ID), after_job, _ended(recreates=(RECOVERED_RUNTIME,))
    )

    assert bound == _generation(OLD_ID), why
    assert "not the build convergence verified" in note
    assert not generation_check(bound, after_job).ok, why


@pytest.mark.parametrize(
    "job_end",
    [
        _ended(),  # the job recreated nothing
        _ended(status="failed", recreates=(RECOVERED_RUNTIME,)),
        _ended(recreates=({**RECOVERED_RUNTIME, "outcome": "still_failing"},)),
        _ended(recreates=({**RECOVERED_RUNTIME, "service": "runtime-effects"},)),
        ModelJobEnd(ended=False, correlation_id=CID, reason="still in_progress"),
    ],
)
def test_movement_the_job_record_does_not_explain_keeps_the_converged_binding(
    job_end: ModelJobEnd,
) -> None:
    bound, _note = bind_generation_to_job_end(
        _generation(OLD_ID), _generation(DISPLACED_ID), job_end
    )

    assert bound == _generation(OLD_ID)
    assert not generation_check(bound, _generation(DISPLACED_ID)).ok


# --- AC2: the two cases read differently on the receipt -----------------------


def test_the_in_job_case_and_the_displacement_case_have_different_evidence() -> None:
    in_job = _ended(recreates=(RECOVERED_RUNTIME,))
    _, in_job_note = bind_generation_to_job_end(
        _generation(OLD_ID), _generation(NEW_ID), in_job
    )
    displaced = _ended()
    _, displaced_note = bind_generation_to_job_end(
        _generation(OLD_ID), _generation(DISPLACED_ID), displaced
    )

    assert "in-job recreate" in in_job.evidence_clause()
    assert "omninode-runtime recovered after 291s" in in_job.evidence_clause()
    assert "recreated no runtime container" in displaced.evidence_clause()
    assert "no recovered in-job recreate" in displaced_note
    assert in_job.evidence_clause() != displaced.evidence_clause()
    assert in_job_note != displaced_note


# --- the wait itself ----------------------------------------------------------


class _Agent:
    """A fake ``/job/{cid}`` surface whose answers advance with each sleep."""

    def __init__(self, timeline: dict[str, list[dict[str, Any]]]) -> None:
        self.timeline = timeline
        self.ticks = 0
        self.urls: list[str] = []

    def sleep(self, _seconds: float) -> None:
        self.ticks += 1

    def fetch(self, url: str, _timeout: float) -> tuple[int, str]:
        self.urls.append(url)
        cid = url.rsplit("/", 1)[-1]
        answers = self.timeline.get(cid)
        if not answers:
            return 404, '{"error": "not found"}'
        return 200, json.dumps(answers[min(self.ticks, len(answers) - 1)])


def _now() -> datetime:
    return datetime(2026, 9, 24, 4, 14, tzinfo=UTC)


def test_the_wait_returns_when_the_job_ends_with_its_recreate_record() -> None:
    agent = _Agent(
        {
            CID: [
                {"status": "in_progress"},
                {"status": "in_progress"},
                {
                    "status": "success",
                    "completed_at": "2026-09-24T04:19:00+00:00",
                    "verify_recreate": [RECOVERED_RUNTIME],
                },
            ]
        }
    )

    end = wait_for_agent_job_end(
        agent_url=AGENT,
        correlation_id=CID,
        deadline=_now() + timedelta(minutes=20),
        clock=_now,
        sleep=agent.sleep,
        opener=agent.fetch,
    )

    assert end.ended
    assert end.status == "success"
    assert end.verify_recreate == (RECOVERED_RUNTIME,)
    assert agent.ticks == 2


def test_a_superseded_record_is_followed_to_the_command_that_ran() -> None:
    agent = _Agent(
        {
            CID: [{"status": "superseded", "superseded_by_correlation_id": RUNNER_CID}],
            RUNNER_CID: [
                {"status": "success", "completed_at": "t", "verify_recreate": []}
            ],
        }
    )

    end = wait_for_agent_job_end(
        agent_url=AGENT,
        correlation_id=CID,
        deadline=_now() + timedelta(minutes=20),
        clock=_now,
        sleep=agent.sleep,
        opener=agent.fetch,
    )

    assert end.ended
    assert end.correlation_id == RUNNER_CID


def test_a_job_still_running_at_the_deadline_is_reported_not_raised() -> None:
    agent = _Agent({CID: [{"status": "in_progress"}]})
    moments = iter(
        [_now() + timedelta(seconds=JOB_END_POLL_SECONDS * n) for n in range(10)]
    )

    end = wait_for_agent_job_end(
        agent_url=AGENT,
        correlation_id=CID,
        deadline=_now() + timedelta(seconds=JOB_END_POLL_SECONDS * 3),
        clock=lambda: next(moments),
        sleep=agent.sleep,
        opener=agent.fetch,
    )

    assert not end.ended
    assert "still in_progress" in end.reason
    assert "generation read before" in end.evidence_clause()


def test_an_unreachable_agent_is_reported_not_raised() -> None:
    def refuse(_url: str, _timeout: float) -> tuple[int, str]:
        raise ConnectionRefusedError(111, "Connection refused")

    end = wait_for_agent_job_end(
        agent_url=AGENT,
        correlation_id=CID,
        deadline=_now(),
        clock=_now,
        sleep=lambda _s: None,
        opener=refuse,
    )

    assert not end.ended
    assert "ConnectionRefusedError" in end.reason


def test_no_agent_url_or_correlation_id_does_not_wait() -> None:
    for agent_url, cid in (("", CID), (AGENT, ""), (AGENT, "not-a-uuid")):
        end = wait_for_agent_job_end(
            agent_url=agent_url,
            correlation_id=cid,
            deadline=_now() + timedelta(hours=1),
            clock=_now,
            sleep=lambda _s: pytest.fail("waited without a job to wait on"),
        )
        assert not end.ended


# --- end to end through the guard's convergence mode --------------------------


def test_the_guard_publishes_the_post_recreate_generation_for_an_in_job_recreate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """AC1's falsifier, through the real step: converge, wait, rebind, publish."""
    output = tmp_path / "out"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    agent = _Agent(
        {
            CID: [
                {"status": "in_progress", "accepted_at": "2026-09-24T04:03:11+00:00"},
                {
                    "status": "success",
                    "accepted_at": "2026-09-24T04:03:11+00:00",
                    "completed_at": "2026-09-24T04:19:00+00:00",
                    "verify_recreate": [RECOVERED_RUNTIME],
                },
            ]
        }
    )
    generations = iter([_generation(OLD_ID), _generation(NEW_ID)])

    monkeypatch.setattr(staleness_module, "_http_get_json", agent.fetch)
    monkeypatch.setattr(staleness_module.time, "sleep", agent.sleep)
    monkeypatch.setattr(
        staleness_module, "read_lane_generation", lambda _c: next(generations)
    )
    monkeypatch.setattr(
        staleness_module,
        "_read_lane",
        lambda _args: LaneRevision(
            revision=MERGE_SHA,
            compose_project="omnibase-infra",
            build_source="workspace",
            state="running",
        ),
    )
    monkeypatch.setattr(
        staleness_module._AncestryResolver, "resolve", lambda _self, _observed: None
    )
    monkeypatch.setattr(staleness_module, "resolve_lane_binding", lambda **_k: None)
    monkeypatch.setattr(
        staleness_module, "read_agent_loaded_code_sha", lambda _url: None
    )
    args = type(
        "Args",
        (),
        {
            "expect_revision": MERGE_SHA,
            "wait_timeout": timedelta(minutes=25),
            "poll_interval": timedelta(seconds=60),
            "repo": "repo/name",
            "branch": "dev",
            "container": "omninode-runtime",
            "deployed_revision": "",
            "agent_url": AGENT,
            "correlation_id": CID,
            "agent_timeout_seconds": 10.0,
            "wall_clock_seconds": 1500,
        },
    )()

    assert staleness_module._run_convergence_mode(args) == 0

    outputs = dict(
        line.split("=", 1)
        for line in output.read_text(encoding="utf-8").splitlines()
        if "=" in line and not line.startswith(" ")
    )
    published = ModelLaneGeneration(**json.loads(outputs["generation"]))
    assert published.container_id == NEW_ID
    assert generation_check(published, _generation(NEW_ID)).ok
    assert "in-job recreate" in outputs["evidence"]
    assert "rebound" in outputs["evidence"]
