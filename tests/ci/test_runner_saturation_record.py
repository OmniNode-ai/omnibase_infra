# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Coverage for the runner saturation monitor (OMN-18031 G7).

WHY THIS EXISTS. The operator's question was "make sure we have a way to monitor
if the lab runners start getting slammed". Once routing is live, the dangerous
mode is SILENT: the route job correctly falls back to hosted on every run, every
check is green, and the only symptom is a GitHub Actions bill. Nothing in the
existing surfaces reports it -- the fleet canary watches registrations and
listener liveness, not saturation, and the routing audit is an hourly DRIFT gate
whose red means "a variable changed".

THE TEST THAT MATTERS MOST IS THE POSITIVE CONTROL. An alerter that never fires
and an alerter that is correct are indistinguishable from a green run, so every
"does not alert" assertion here is paired with a fixture proving the same
counter DOES fire on a real breach. ``test_a_clean_window_does_not_alert`` alone
would pass against ``return []``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "ci" / "runner_saturation_record.py"

_spec = importlib.util.spec_from_file_location("runner_saturation_record", MODULE_PATH)
assert _spec is not None and _spec.loader is not None
sat = importlib.util.module_from_spec(_spec)
sys.modules["runner_saturation_record"] = sat
_spec.loader.exec_module(sat)


ALERT_POLICY: dict[str, Any] = {
    "busy_fraction_threshold": 0.85,
    "lab_load_ratio_threshold": 1.5,
    "min_lab_free_mem_mib": 4096,
    "sustained_samples": 4,
    "sustained_min_span_seconds": 900,
}


def _window(
    samples: list[dict[str, Any]], *, spacing_seconds: int = 600
) -> list[dict[str, Any]]:
    """Stamp a NEWEST-FIRST window with real, decreasing timestamps.

    The sustain rule is a duration and not only a count, so a window whose
    samples all share one timestamp spans zero seconds and correctly alerts on
    nothing. Tests must therefore say when their samples were taken.
    """
    base = datetime(2026, 9, 7, 12, 0, 0, tzinfo=UTC)
    stamped = []
    for index, sample in enumerate(samples):
        copy = json.loads(json.dumps(sample))
        copy["sampled_at"] = (
            base - timedelta(seconds=index * spacing_seconds)
        ).isoformat()
        stamped.append(copy)
    return stamped


def _record(
    *,
    sampled_at: str = "2026-09-07T10:00:00Z",
    busy_fraction: float = 0.10,
    lab_ratio: float | None = 0.20,
    lab_probe: str = "ok",
    route_reason: str = "seam_ceiling_hosted",
    free_mem_mib: int = 40000,
) -> dict[str, Any]:
    """One sample of the ``runner_saturation_record/v1`` schema."""
    return {
        "schema": "runner_saturation_record/v1",
        "sampled_at": sampled_at,
        "fleet": {
            "online": 88,
            "busy": int(88 * busy_fraction),
            "busy_fraction": busy_fraction,
        },
        "lab": {
            "probe": lab_probe,
            "hosts": (
                []
                if lab_ratio is None
                else [
                    {"label": "h201", "ratio": lab_ratio, "free_mem_mib": free_mem_mib}
                ]
            ),
        },
        "route": {"recent_reason": route_reason},
    }


# --- schema ---------------------------------------------------------------


def test_the_record_round_trips_through_json() -> None:
    built = sat.build_record(
        fleet={"ok": True, "online": 88, "busy": 12},
        lab={
            "ok": True,
            "hosts": [{"label": "h201", "ratio": 0.2, "free_mem_mib": 40000}],
        },
        route_reason="capacity_available",
        sampled_at="2026-09-07T10:00:00Z",
    )
    payload = json.loads(json.dumps(built))
    assert payload["schema"] == "runner_saturation_record/v1"
    assert payload["fleet"]["busy_fraction"] == pytest.approx(12 / 88)
    assert payload["lab"]["probe"] == "ok"


def test_an_absent_lab_probe_is_recorded_as_unavailable_not_as_healthy() -> None:
    """The self-hosted lab-load probe shares fate with the fleet. Its ABSENCE is
    a signal (the one reason a 3-minute self-hosted job cannot start is that no
    runner is free), so it must never be recorded as a healthy zero.
    """
    built = sat.build_record(
        fleet={"ok": True, "online": 88, "busy": 12},
        lab={"ok": False, "error": "artifact_missing"},
        route_reason="capacity_available",
        sampled_at="2026-09-07T10:00:00Z",
    )
    assert built["lab"]["probe"] == "unavailable"
    assert built["lab"]["hosts"] == []


# --- sustained counting ---------------------------------------------------


def test_a_breach_shorter_than_the_sustain_window_does_not_alert() -> None:
    """3 breaching samples with sustained_samples=4 -> silent. Prevents flapping
    on a signal measured swinging 65 -> 10 busy of 88 inside two minutes.
    """
    window = _window([_record(busy_fraction=0.95)] * 3 + [_record(busy_fraction=0.10)])
    assert sat.evaluate_alerts(window, ALERT_POLICY) == []


def test_the_sustain_counter_counts_consecutive_samples_from_the_newest() -> None:
    """An old breach that has since recovered must not resurrect an alert."""
    window = [_record(busy_fraction=0.10)] + [_record(busy_fraction=0.95)] * 8
    assert sat.evaluate_alerts(window, ALERT_POLICY) == []


# --- the four alert conditions -------------------------------------------


def test_condition_1_sustained_fleet_busy_fraction_alerts() -> None:
    window = _window([_record(busy_fraction=0.90)] * 4)
    alerts = sat.evaluate_alerts(window, ALERT_POLICY)
    assert [a.condition for a in alerts] == ["fleet_busy_sustained"]
    assert "0.9" in alerts[0].detail or "90" in alerts[0].detail


def test_condition_2_sustained_lab_load_alerts() -> None:
    window = _window([_record(lab_ratio=1.9)] * 4)
    alerts = sat.evaluate_alerts(window, ALERT_POLICY)
    assert [a.condition for a in alerts] == ["lab_load_sustained"]


def test_condition_2_also_fires_on_sustained_memory_starvation() -> None:
    window = _window([_record(lab_ratio=0.05, free_mem_mib=256)] * 4)
    alerts = sat.evaluate_alerts(window, ALERT_POLICY)
    assert [a.condition for a in alerts] == ["lab_load_sustained"]


def test_condition_3_sustained_route_fallback_alerts() -> None:
    """THE SILENT ONE. Every run is green; the only symptom is the bill."""
    window = _window([_record(route_reason="fleet_saturated")] * 4)
    alerts = sat.evaluate_alerts(window, ALERT_POLICY)
    assert [a.condition for a in alerts] == ["route_fallback_sustained"]


def test_condition_3_does_not_fire_on_todays_inert_seam_ceiling_state() -> None:
    """CRITICAL. Until the seam is flipped, EVERY run decides
    ``seam_ceiling_hosted``. If that counted as a saturation fallback the
    monitor would alert continuously from the moment it lands, and would be
    muted long before the seam ever flips. Inert is not saturated.
    """
    window = _window([_record(route_reason="seam_ceiling_hosted")] * 12)
    assert sat.evaluate_alerts(window, ALERT_POLICY) == []


def test_condition_3_does_not_fire_on_fork_isolation_fallbacks() -> None:
    """Fork isolation is a correct permanent hosted decision, not saturation."""
    window = _window([_record(route_reason="fork_isolation")] * 12)
    assert sat.evaluate_alerts(window, ALERT_POLICY) == []


def test_condition_4_sustained_lab_probe_unavailability_alerts() -> None:
    window = _window([_record(lab_probe="unavailable", lab_ratio=None)] * 4)
    alerts = sat.evaluate_alerts(window, ALERT_POLICY)
    assert [a.condition for a in alerts] == ["lab_probe_unavailable_sustained"]


# --- the negative, and its mandatory positive control --------------------


def test_a_clean_window_does_not_alert() -> None:
    """Silence on a healthy window. Meaningless alone -- see the control below."""
    window = _window([_record()] * 12)
    assert sat.evaluate_alerts(window, ALERT_POLICY) == []


def test_positive_control_the_same_counter_does_fire_on_a_breach_fixture() -> None:
    """POSITIVE CONTROL for the test above. Same function, same window length,
    same policy: only the values differ. Without this, ``evaluate_alerts``
    could be ``return []`` and every negative assertion in this module would
    still pass green.
    """
    clean = _window([_record()] * 12)
    breach = _window(
        [_record(busy_fraction=0.99, lab_ratio=2.5, route_reason="fleet_saturated")]
        * 12
    )
    assert sat.evaluate_alerts(clean, ALERT_POLICY) == []
    fired = sat.evaluate_alerts(breach, ALERT_POLICY)
    assert len(fired) == 3
    assert {a.condition for a in fired} == {
        "fleet_busy_sustained",
        "lab_load_sustained",
        "route_fallback_sustained",
    }


def test_an_empty_window_does_not_alert_and_does_not_raise() -> None:
    """First run after landing has no history. That is not a breach."""
    assert sat.evaluate_alerts([], ALERT_POLICY) == []


def test_a_malformed_prior_record_is_skipped_rather_than_crashing_the_monitor() -> None:
    """A monitor that dies on one bad artifact stops watching entirely."""
    window = [
        {"schema": "runner_saturation_record/v1"},
        {},
        _record(busy_fraction=0.99),
    ]
    assert sat.evaluate_alerts(window, ALERT_POLICY) == []


def test_the_alert_policy_is_read_from_the_committed_policy_file() -> None:
    """Thresholds live in config/runner_routing_policy.yaml, not in the script."""
    loaded = sat.load_alert_policy(REPO_ROOT / "config" / "runner_routing_policy.yaml")
    assert loaded["busy_fraction_threshold"] == 0.85
    assert loaded["sustained_samples"] == 4
    assert loaded["sustained_min_span_seconds"] == 900


@pytest.mark.parametrize(
    "missing_key",
    [
        "busy_fraction_threshold",
        "lab_load_ratio_threshold",
        "sustained_samples",
        "sustained_min_span_seconds",
    ],
)
def test_a_missing_alert_threshold_raises_rather_than_defaulting(
    missing_key: str, tmp_path: Path
) -> None:
    section = dict(ALERT_POLICY)
    del section[missing_key]
    policy_file = tmp_path / "p.yaml"
    policy_file.write_text(
        json.dumps({"route": {"saturation_alert": section}}), encoding="utf-8"
    )
    with pytest.raises(KeyError):
        sat.load_alert_policy(policy_file)


def test_enough_samples_arriving_too_fast_do_not_alert() -> None:
    """THE MEASURED HAZARD. These records arrive as fast as the GitHub-hosted
    queue drains the canary, not at the cron cadence -- on 2026-09-07 four
    consecutive canary runs sat queued simultaneously. Four breaching samples
    30 seconds apart are a 90-second blip, not a sustained outage, and a
    count-only counter would have paged on them.
    """
    window = _window([_record(busy_fraction=0.99)] * 6, spacing_seconds=30)
    assert sat.evaluate_alerts(window, ALERT_POLICY) == []


def test_the_same_samples_spaced_widely_enough_do_alert() -> None:
    """POSITIVE CONTROL for the span floor: identical values, wider spacing."""
    window = _window([_record(busy_fraction=0.99)] * 6, spacing_seconds=600)
    assert [a.condition for a in sat.evaluate_alerts(window, ALERT_POLICY)] == [
        "fleet_busy_sustained"
    ]


def test_unreadable_timestamps_fail_closed_to_silence() -> None:
    """An unreadable clock is not evidence a breach was sustained."""
    window = [_record(busy_fraction=0.99, sampled_at="not-a-timestamp")] * 6
    assert sat.evaluate_alerts(window, ALERT_POLICY) == []


# --- condition 3 counts an ALLOWLIST, and lab_unknown is not on it ---------


@pytest.mark.parametrize(
    "inert_reason",
    [
        "lab_unknown",
        "probe_error:missing_token",
        "probe_error:timeout",
        "probe_error:internal:ValueError",
        "never_widen_violation",
    ],
)
def test_condition_3_does_not_fire_on_a_reason_that_is_not_lab_pressure(
    inert_reason: str,
) -> None:
    """A hosted decision is not automatically evidence the lab is slammed.

    MEASURED DEFECT, not a hypothetical. Condition 3 first counted a DENYLIST --
    every hosted reason except ``seam_ceiling_hosted``, ``fork_isolation`` and
    ``policy_allowlist``. Run live on 2026-09-07T12:22Z against a fully healthy
    fleet (88 online, 4 busy) and a fully healthy lab host (0.669x load, 66013
    MiB free), it raised ``route_fallback_sustained`` on ``lab_unknown``.

    ``lab_unknown`` is the reason on EVERY run until the monitor has produced
    its first lab record for the route job to read -- which is to say, from the
    day this lands. A denylist therefore reproduces exactly the "fires on every
    sample from the day it lands and is muted long before it matters" failure
    the module docstring claims to have avoided by excluding
    ``seam_ceiling_hosted``, one reason over. ``probe_error:missing_token`` is
    permanent on any run without the fleet-status secret and is a probe fault,
    not lab pressure.

    A denylist fails OPEN: every reason nobody thought of silently becomes an
    alert. The allowlist fails closed.
    """
    window = _window([_record(route_reason=inert_reason)] * 12)
    assert sat.evaluate_alerts(window, ALERT_POLICY) == []


@pytest.mark.parametrize(
    "pressure_reason", ["fleet_saturated", "fleet_degraded", "lab_saturated"]
)
def test_condition_3_positive_control_every_allowlisted_reason_does_fire(
    pressure_reason: str,
) -> None:
    """Mandatory counterpart: the allowlist must not have muted the condition.

    A condition-3 that alerted on NOTHING would satisfy the test above for every
    parameter while leaving the operator's actual question -- "how do we know if
    the lab runners start getting slammed" -- unanswered, and would look
    identical on a green run. Each allowlisted reason is asserted to fire on its
    own, so dropping any one of the three is a failing test rather than a
    quieter monitor.
    """
    window = _window([_record(route_reason=pressure_reason)] * 12)
    alerts = sat.evaluate_alerts(window, ALERT_POLICY)
    assert "route_fallback_sustained" in [a.condition for a in alerts]


# --- OMN-18031 follow-up (2026-09-12): ROUTE_REASON was hardcoded ----------
#
# ``dev-lane-liveness.yml`` hardcoded ``ROUTE_REASON: unknown`` because
# nothing aggregated the route job's own ``runner-route-decision-*``
# artifacts into this workflow -- condition 3 (``route_fallback_sustained``,
# the alert the operator actually asked for) had no input and could never
# fire. ``select_latest_route_artifact`` / ``extract_route_reason`` are the
# pure halves of the fix; the network read (``gh api``) is a workflow step.


def _artifact(
    name: str, created_at: str, *, expired: bool = False, artifact_id: int = 1
) -> dict[str, Any]:
    return {
        "id": artifact_id,
        "name": name,
        "created_at": created_at,
        "expired": expired,
    }


def test_select_latest_route_artifact_picks_the_newest_by_created_at() -> None:
    artifacts = [
        _artifact(
            "runner-route-decision-1-route", "2026-09-12T10:00:00Z", artifact_id=1
        ),
        _artifact(
            "runner-route-decision-2-route", "2026-09-12T12:00:00Z", artifact_id=2
        ),
        _artifact(
            "runner-route-decision-3-route", "2026-09-12T11:00:00Z", artifact_id=3
        ),
    ]
    picked = sat.select_latest_route_artifact(artifacts)
    assert picked is not None
    assert picked["id"] == 2


def test_select_latest_route_artifact_ignores_other_artifact_names() -> None:
    """The listing is REPO-WIDE (`/actions/artifacts`), not scoped to this
    workflow -- `lab-load`, `saturation-record`, and every other artifact in
    the repo's retention window are in the same response and must not match.
    """
    artifacts = [
        _artifact("lab-load", "2026-09-12T12:00:00Z", artifact_id=1),
        _artifact("saturation-record", "2026-09-12T13:00:00Z", artifact_id=2),
        _artifact(
            "runner-route-decision-4-route", "2026-09-12T09:00:00Z", artifact_id=3
        ),
    ]
    picked = sat.select_latest_route_artifact(artifacts)
    assert picked is not None
    assert picked["id"] == 3


def test_select_latest_route_artifact_skips_expired_artifacts() -> None:
    """An expired artifact's download URL 404s; picking it would make the
    step's best-effort download fail for a reason indistinguishable from "no
    artifact exists", so it must be filtered out here instead.
    """
    artifacts = [
        _artifact(
            "runner-route-decision-1-route",
            "2026-09-12T12:00:00Z",
            expired=True,
            artifact_id=1,
        ),
        _artifact(
            "runner-route-decision-2-route", "2026-09-12T10:00:00Z", artifact_id=2
        ),
    ]
    picked = sat.select_latest_route_artifact(artifacts)
    assert picked is not None
    assert picked["id"] == 2


def test_select_latest_route_artifact_returns_none_when_nothing_matches() -> None:
    """The safe fallback: the caller's ``ROUTE_REASON`` stays at ``unknown``,
    the same default this workflow already had before the fix -- not a new
    failure mode.
    """
    assert sat.select_latest_route_artifact([]) is None
    assert (
        sat.select_latest_route_artifact(
            [_artifact("lab-load", "2026-09-12T12:00:00Z")]
        )
        is None
    )
    assert (
        sat.select_latest_route_artifact([{"name": "runner-route-decision-1-route"}])
        is None
    )  # missing created_at


def test_extract_route_reason_reads_the_reason_field() -> None:
    assert sat.extract_route_reason({"reason": "lab_saturated"}) == "lab_saturated"


@pytest.mark.parametrize(
    "payload",
    [
        {"reason": ""},
        {"no_reason_field": True},
        {"reason": 123},
        None,
        "not-a-dict",
        [],
    ],
)
def test_extract_route_reason_is_none_on_any_malformed_shape(payload: Any) -> None:
    """Every malformed shape falls back to ``None`` -- never an empty string
    that would look like a real, distinct value while matching nothing in
    ``SATURATION_FALLBACK_REASONS``.
    """
    assert sat.extract_route_reason(payload) is None
