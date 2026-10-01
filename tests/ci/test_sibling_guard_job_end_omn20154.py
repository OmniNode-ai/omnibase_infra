# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The SIBLING convergence guard binds the generation after the agent's job ends (OMN-20154).

MEASURED, omnimarket run 36788681945 (merge 069f94e1ee6d, lane compose-dev-202):
``probe_generation_bound`` was the only failing check. The sibling guard
(``check_lane_sibling_revision.py``) read ``omninode-dev-202-runtime-effects``
``7b926d92febd`` at convergence, 23:12:36Z. Deploy-agent job ada00fdd's own
post-deploy verification then force-recreated the ``runtime-effects`` compose
service at 23:17:50Z (``/job/ada00fdd`` ``verify_recreate``: outcome
``recovered`` after 250s) and the job ended ``success`` at 23:22:05Z. The probe
read ``e311dd332d91`` -- same image, same revision, the post-recreate container.
The next rebuild command (db368818) was accepted at 23:22:12Z, AFTER the job
ended, so it did not move the lane under the probe.

OMN-19374 fixed exactly this for ``check_dev_lane_staleness.py`` and left the
sibling guard reading the generation at convergence. Two defects, both RED here:

1. the sibling guard never waits for the job to end, so it never rebinds; and
2. ``bind_generation_to_job_end`` matched the job record's compose SERVICE name
   against the container NAME, which only coincide on the .201 dev lane. On
   ``.202`` the service ``runtime-effects`` runs as the container
   ``omninode-dev-202-runtime-effects``, so even the staleness guard could not
   rebind there.

The binding rule is unchanged: a rebind needs a ``success`` job whose own record
names a ``recovered`` recreate of THIS container's service, and a container
with the same image and revision. The positive controls below keep every other
movement failing.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from scripts.ci import check_lane_sibling_revision as sibling
from scripts.ci.check_dev_lane_staleness import ModelJobEnd, bind_generation_to_job_end
from scripts.ci.lab_pass_receipt import ModelLaneGeneration, generation_check

pytestmark = pytest.mark.unit

AGENT = "http://agent.invalid:61098"
CID = "ada00fdd-90ef-430c-9a49-adac38ac7131"
CONTAINER = "omninode-dev-202-runtime-effects"
SERVICE = "runtime-effects"
IMAGE = "sha256:" + "1" * 64
OTHER_IMAGE = "sha256:" + "2" * 64
REVISION = "c0dc1d80ae818b95bbc3dcec8a72dcb0fa4b02ff"
OLD_ID = "7b926d92febd" + "0" * 52
NEW_ID = "e311dd332d91" + "0" * 52

RECOVERED_EFFECTS: dict[str, Any] = {
    "service": SERVICE,
    "lane": "dev",
    "compose_project": "omnibase-infra-dev-202",
    "endpoint": "http://localhost:61086/health",
    "outcome": "recovered",
    "recreate_returncode": 0,
    "readiness_wait_seconds": 250.1,
    "readiness_budget_seconds": 600,
    "detail": "",
}


def _generation(
    container_id: str, *, image: str = IMAGE, revision: str = REVISION
) -> ModelLaneGeneration:
    return ModelLaneGeneration(
        container=CONTAINER, container_id=container_id, image=image, revision=revision
    )


def _ended(
    *, status: str = "success", recreates: tuple[dict[str, Any], ...] = ()
) -> ModelJobEnd:
    return ModelJobEnd(
        ended=True,
        correlation_id=CID,
        status=status,
        completed_at="2026-09-30T23:22:05+00:00",
        verify_recreate=recreates,
    )


# --- defect 2: the service is matched by its compose service name ------------


def test_a_recreate_recorded_by_compose_service_rebinds_a_prefixed_container() -> None:
    bound, note = bind_generation_to_job_end(
        _generation(OLD_ID),
        _generation(NEW_ID),
        _ended(recreates=(RECOVERED_EFFECTS,)),
        service=SERVICE,
    )

    assert bound == _generation(NEW_ID)
    assert "rebound" in note
    assert generation_check(bound, _generation(NEW_ID)).ok


def test_without_the_service_name_the_prefixed_container_is_not_rebound() -> None:
    """The measured .202 shape before this fix: name != service, no rebind."""
    bound, _ = bind_generation_to_job_end(
        _generation(OLD_ID),
        _generation(NEW_ID),
        _ended(recreates=(RECOVERED_EFFECTS,)),
    )

    assert bound == _generation(OLD_ID)
    assert not generation_check(bound, _generation(NEW_ID)).ok


@pytest.mark.parametrize(
    ("record", "after_job", "why"),
    [
        (
            {**RECOVERED_EFFECTS, "service": "omninode-runtime"},
            _generation(NEW_ID),
            "a recreate of a different service",
        ),
        (
            {**RECOVERED_EFFECTS, "outcome": "still_failing"},
            _generation(NEW_ID),
            "a recreate that did not recover",
        ),
        (RECOVERED_EFFECTS, _generation(NEW_ID, image=OTHER_IMAGE), "another image"),
        (RECOVERED_EFFECTS, _generation(NEW_ID, revision="f" * 40), "another rev"),
    ],
)
def test_every_other_movement_still_fails_the_binding(
    record: dict[str, Any], after_job: ModelLaneGeneration, why: str
) -> None:
    bound, _ = bind_generation_to_job_end(
        _generation(OLD_ID), after_job, _ended(recreates=(record,)), service=SERVICE
    )

    assert bound == _generation(OLD_ID), why
    assert not generation_check(bound, after_job).ok, why


# --- defect 1: the sibling guard waits for the job and binds after it --------


def _clock_from(start: datetime) -> tuple[Any, Any]:
    now = [start]

    def clock() -> datetime:
        return now[0]

    def sleep(seconds: float) -> None:
        now[0] += timedelta(seconds=seconds)

    return clock, sleep


def test_the_sibling_guard_binds_to_the_container_the_job_left_running() -> None:
    reads = iter([_generation(OLD_ID), _generation(NEW_ID)])
    bodies = iter(
        [
            (200, '{"status": "in_progress"}'),
            (
                200,
                '{"status": "success", "completed_at": "2026-09-30T23:22:05+00:00",'
                ' "verify_recreate": [' + _json(RECOVERED_EFFECTS) + "]}",
            ),
        ]
    )
    clock, sleep = _clock_from(datetime(2026, 9, 30, 23, 12, 36, tzinfo=UTC))

    bound = sibling.bind_sibling_generation(
        container=CONTAINER,
        agent_url=AGENT,
        correlation_id=CID,
        deadline=datetime(2026, 9, 30, 23, 40, tzinfo=UTC),
        clock=clock,
        sleep=sleep,
        read_generation=lambda _c: next(reads),
        read_service=lambda _c: SERVICE,
        opener=lambda _url, _t: next(bodies),
    )

    assert bound.generation == _generation(NEW_ID)
    assert "rebound" in bound.evidence
    assert "ended success" in bound.evidence


def test_a_job_still_running_at_the_deadline_keeps_the_converged_binding() -> None:
    reads = iter([_generation(OLD_ID), _generation(NEW_ID)])
    clock, sleep = _clock_from(datetime(2026, 9, 30, 23, 12, 36, tzinfo=UTC))

    bound = sibling.bind_sibling_generation(
        container=CONTAINER,
        agent_url=AGENT,
        correlation_id=CID,
        deadline=datetime(2026, 9, 30, 23, 13, tzinfo=UTC),
        clock=clock,
        sleep=sleep,
        read_generation=lambda _c: next(reads),
        read_service=lambda _c: SERVICE,
        opener=lambda _url, _t: (200, '{"status": "in_progress"}'),
    )

    assert bound.generation == _generation(OLD_ID)
    assert "before deploy-agent job" in bound.evidence


def test_an_unreadable_container_publishes_no_generation() -> None:
    def unreadable(_c: str) -> ModelLaneGeneration:
        raise ValueError("docker inspect failed")

    clock, sleep = _clock_from(datetime(2026, 9, 30, 23, 12, 36, tzinfo=UTC))
    bound = sibling.bind_sibling_generation(
        container=CONTAINER,
        agent_url=AGENT,
        correlation_id=CID,
        deadline=datetime(2026, 9, 30, 23, 40, tzinfo=UTC),
        clock=clock,
        sleep=sleep,
        read_generation=unreadable,
        read_service=lambda _c: SERVICE,
        opener=lambda _url, _t: (200, '{"status": "success"}'),
    )

    assert bound.generation is None


def _json(record: dict[str, Any]) -> str:
    import json

    return json.dumps(record)
