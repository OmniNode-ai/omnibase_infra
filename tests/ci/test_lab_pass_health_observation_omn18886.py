# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18886 -- the health dimensions are an OBSERVATION, so the probe waits for one.

THE DEFECT. ``details.runtime_health`` is populated asynchronously and carries
its own ``observed_at`` and ``age_seconds``. The single-sample check read it
once, after readiness but before the first observation completed, found it
absent, and reported "an absent dimension set is not a healthy dimension set".
The sentence is correct and the conclusion was wrong: the dimensions were not
absent, they had not happened yet.

Measured on the omnimarket sibling emitter, runs 35505864089 and 35507742784,
both FAIL with exactly one failing check on a lane where every convergence
check passed. Those receipts BLOCK a staging delivery: the delivery workflow's
``lab-pass-gate`` reads the sibling receipt and ``dispatch-to-staging`` needs
that job.

WHAT THESE TESTS PIN, and the shape of the risk they cover. The fix adds
waiting, and waiting is exactly how a gate gets quietly weakened -- poll long
enough and anything eventually looks fine. So the negative controls outnumber
the positive one, and two of them exist specifically to prove the poll cannot
manufacture a green:

* absent forever still FAILS, and says "never observed" rather than blaming the
  lane;
* unhealthy forever still FAILS, naming the dimensions, after the full budget;
* a single unhealthy observation is NEVER accepted on sight, however long the
  budget is;
* the two failing outcomes carry DIFFERENT evidence, because "the observer ran
  out of time" and "the lane is sick" sent two lanes hunting the wrong thing.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from scripts.ci.lab_pass_receipt import (
    check_health_dimensions,
    check_health_dimensions_observed,
)

pytestmark = pytest.mark.unit

_URL = "http://lane.invalid:8085/health"


def _payload(dimensions: list[dict[str, str]] | None) -> str:
    """A health body shaped like the live one, with or without the block."""
    details: dict[str, Any] = {"healthy": True, "is_running": True}
    if dimensions is not None:
        details["runtime_health"] = {
            "status": "DEGRADED"
            if any(d["status"] != "HEALTHY" for d in dimensions)
            else "HEALTHY",
            "observed_at": "2026-09-20T12:13:22.587823+00:00",
            "age_seconds": 281.45,
            "dimensions": dimensions,
        }
    return json.dumps({"status": "ok", "details": details})


#: The seven dimensions the live lane reported on 2026-09-20, all healthy.
_HEALTHY = [
    {"name": n, "status": "HEALTHY", "detail": "ok"}
    for n in (
        "discovery_errors",
        "empty_consumer_groups",
        "topic_coverage",
        "projection_attachment",
        "projection_dlq_saturation",
        "projection_write_path",
        "consumer_sync",
    )
]

#: The same set with the dimension that was ACTUALLY degraded on the lane while
#: this was written: one projection routing 100% of consumed events to a
#: dead-letter sink. Used rather than an invented name so the unhealthy-path
#: tests exercise a real shape.
_DEGRADED = [
    dict(d, status="DEGRADED", detail="1 projection(s) routed 100% to a DLQ sink")
    if d["name"] == "projection_dlq_saturation"
    else d
    for d in _HEALTHY
]


class FakeHttp:
    """Serves a scripted sequence of bodies, then repeats the last one."""

    def __init__(self, bodies: list[str]) -> None:
        self.bodies = bodies
        self.calls = 0

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _get(url: str, timeout_seconds: float) -> tuple[int, str]:
            body = self.bodies[min(self.calls, len(self.bodies) - 1)]
            self.calls += 1
            return 200, body

        monkeypatch.setattr("scripts.ci.lab_pass_receipt._http_get", _get)


class FakeClock:
    """A clock that only advances when the code under test sleeps."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(seconds, 0.001)


# ---------------------------------------------------------------------------
# the defect, reproduced
# ---------------------------------------------------------------------------


def test_the_single_sample_check_fails_on_a_lane_that_is_merely_slow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE PRODUCTION DEFECT, reproduced against the old code path.

    This is what emitted the two false FAIL receipts. Kept as a test rather
    than only described, so the difference the fix makes is a measurement.
    """
    FakeHttp([_payload(None)]).install(monkeypatch)
    check = check_health_dimensions(_URL, 5.0)
    assert check.ok is False
    assert "runtime_health is absent or not an object" in check.evidence


def test_the_polling_check_passes_once_the_observation_appears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POSITIVE CONTROL: the same lane, waited for rather than sampled once."""
    http = FakeHttp([_payload(None), _payload(None), _payload(_HEALTHY)])
    http.install(monkeypatch)
    clock = FakeClock()
    check = check_health_dimensions_observed(
        _URL, 5.0, 300.0, sleep_fn=clock.sleep, clock=clock
    )
    assert check.ok is True
    assert "all healthy" in check.evidence
    assert "dimensions observed after" in check.evidence
    assert http.calls == 3


# ---------------------------------------------------------------------------
# the fix must not have bought the green with waiting
# ---------------------------------------------------------------------------


def test_absent_forever_still_fails_and_blames_the_observer_not_the_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NEGATIVE CONTROL 1. Absent is still not healthy; the rule is unchanged.

    The evidence must say the observation never completed rather than that the
    lane reported something unhealthy, because those sent two lanes hunting a
    lane defect that did not exist.
    """
    FakeHttp([_payload(None)]).install(monkeypatch)
    clock = FakeClock()
    check = check_health_dimensions_observed(
        _URL, 5.0, 120.0, sleep_fn=clock.sleep, clock=clock
    )
    assert check.ok is False
    assert "never observed" in check.evidence
    assert "NOT that the lane reported an unhealthy dimension" in check.evidence


def test_unhealthy_forever_still_fails_and_names_the_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NEGATIVE CONTROL 2, the one that matters most.

    A real degradation was live on the lane while this was written. Waiting
    must not turn it green, and the failure must still name it.
    """
    FakeHttp([_payload(_DEGRADED)]).install(monkeypatch)
    clock = FakeClock()
    check = check_health_dimensions_observed(
        _URL, 5.0, 120.0, sleep_fn=clock.sleep, clock=clock
    )
    assert check.ok is False
    assert "projection_dlq_saturation" in check.evidence
    assert "STILL" in check.evidence
    assert "never observed" not in check.evidence


def test_an_unhealthy_observation_is_never_accepted_on_sight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The poll may not short-circuit into a pass on the first non-absent read."""
    FakeHttp([_payload(_DEGRADED)]).install(monkeypatch)
    clock = FakeClock()
    check = check_health_dimensions_observed(
        _URL, 5.0, 100.0, sleep_fn=clock.sleep, clock=clock
    )
    assert check.ok is False
    # It kept asking rather than concluding on the first sample.
    assert clock.now >= 100.0


def test_the_two_failing_outcomes_do_not_share_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """They were one string before, and that is what made them undiagnosable."""
    clock_a, clock_b = FakeClock(), FakeClock()
    FakeHttp([_payload(None)]).install(monkeypatch)
    absent = check_health_dimensions_observed(
        _URL, 5.0, 60.0, sleep_fn=clock_a.sleep, clock=clock_a
    )
    FakeHttp([_payload(_DEGRADED)]).install(monkeypatch)
    unhealthy = check_health_dimensions_observed(
        _URL, 5.0, 60.0, sleep_fn=clock_b.sleep, clock=clock_b
    )
    assert absent.ok is False and unhealthy.ok is False
    assert absent.evidence != unhealthy.evidence


def test_a_degradation_that_clears_inside_the_budget_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Why the poll continues past an unhealthy observation rather than stopping.

    Several dimensions are rolling-window measures, so the first observation on
    a freshly recreated lane can be unhealthy from the boot itself. This is the
    case that behaviour exists for, and it is bounded: the test above proves a
    degradation that does NOT clear still fails.
    """
    http = FakeHttp([_payload(_DEGRADED), _payload(_DEGRADED), _payload(_HEALTHY)])
    http.install(monkeypatch)
    clock = FakeClock()
    check = check_health_dimensions_observed(
        _URL, 5.0, 300.0, sleep_fn=clock.sleep, clock=clock
    )
    assert check.ok is True
    assert "all healthy" in check.evidence


def test_a_zero_budget_is_exactly_one_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ad hoc default must not start waiting on somebody's laptop."""
    http = FakeHttp([_payload(None)])
    http.install(monkeypatch)
    clock = FakeClock()
    check = check_health_dimensions_observed(
        _URL, 5.0, 0.0, sleep_fn=clock.sleep, clock=clock
    )
    assert check.ok is False
    assert http.calls == 1
    assert clock.now == 0.0


def test_a_non_200_body_fails_without_waiting_out_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refusing endpoint is not an unfinished observation.

    Without this the probe would spend its whole budget on a lane that is
    plainly down, and the receipt would arrive late saying the wrong thing.
    """

    def _get(url: str, timeout_seconds: float) -> tuple[int, str]:
        return 503, '{"status":"unhealthy"}'

    monkeypatch.setattr("scripts.ci.lab_pass_receipt._http_get", _get)
    clock = FakeClock()
    check = check_health_dimensions_observed(
        _URL, 5.0, 600.0, sleep_fn=clock.sleep, clock=clock
    )
    assert check.ok is False
    assert "503" in check.evidence
    assert "never observed" not in check.evidence
