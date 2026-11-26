# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A runtime merge is built within the declared probe-deferral cap (OMN-26703).

THE FAILURE
-----------
omnimarket 8a5a95e5 (omnimarket#6903) merged at 11:49Z on 2026-11-26. The agent decided
``deferred:probe_window`` on every idle tick until 13:00Z: C15's run at 11:41Z excludes
[11:01Z, 12:46Z] and the next run's lead opens at 13:01Z, so the guarded windows chain. The
OMN-21146 lag bound overrides the idle wait but never a guarded window, so no lag bound could
end the deferral. ``flags.max_probe_deferral_minutes`` is the age past which a lane behind the
head builds between windows.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from deploy_agent.idle_converge import (
    IDLE_AFTER,
    EnumIdleConvergeVerdict,
    ModelIdleConvergeInputs,
    decide,
    load_probe_windows,
    probe_blocking,
)
from deploy_agent.routing import lane_flags_for_instance

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SEEN = datetime(2026, 11, 26, 11, 49, 56, tzinfo=UTC)
DEADLINE = datetime(2026, 11, 26, 12, 49, tzinfo=UTC)
RUNNING = "a" * 40
HEAD = "8a5a95e5" + "0" * 32


def _inputs(now: datetime, **overrides: object) -> ModelIdleConvergeInputs:
    values: dict[str, object] = {
        "now": now,
        "repository": "omnimarket",
        "routed_elsewhere": True,
        "job_active": False,
        "last_activity": now - IDLE_AFTER,
        "running_ref": RUNNING,
        "head_ref": HEAD,
        "probe_blocker": probe_blocking(now, load_probe_windows(REPO_ROOT)),
        "windows_error": None,
        "attempted_heads": frozenset(),
        "behind_since": SEEN,
        "max_lag": timedelta(minutes=20),
        "max_probe_deferral": cap_from_table(),
    }
    values.update(overrides)
    return ModelIdleConvergeInputs(**values)


def cap_from_table() -> timedelta | None:
    return lane_flags_for_instance(REPO_ROOT, "dev-201").max_probe_deferral()


def _first_converge(**overrides: object) -> datetime | None:
    now = SEEN
    while now <= SEEN + timedelta(hours=3):
        if (
            decide(_inputs(now, **overrides)).verdict
            is EnumIdleConvergeVerdict.CONVERGE
        ):
            return now
        now += timedelta(minutes=1)
    return None


def test_the_committed_schedule_defers_past_the_cap_with_no_cap_declared() -> None:
    first = _first_converge(max_probe_deferral=None)
    cap = cap_from_table()
    assert first is not None and cap is not None and first - SEEN > cap


def test_a_merge_seen_at_1149_converges_before_1249() -> None:
    first = _first_converge()
    assert first is not None and first < DEADLINE


def test_the_cap_is_a_setting_of_the_default_instance() -> None:
    assert cap_from_table() is not None


def test_inside_the_cap_the_probe_window_still_refuses() -> None:
    now = SEEN + timedelta(minutes=10)
    decision = decide(_inputs(now))
    assert decision.verdict is EnumIdleConvergeVerdict.PROBE_WINDOW


def test_no_cap_declared_changes_nothing() -> None:
    now = SEEN + timedelta(minutes=50)
    decision = decide(_inputs(now, max_probe_deferral=None))
    assert decision.verdict is EnumIdleConvergeVerdict.PROBE_WINDOW


def test_past_the_cap_a_job_in_flight_still_refuses() -> None:
    now = SEEN + timedelta(minutes=50)
    decision = decide(_inputs(now, job_active=True))
    assert decision.verdict is EnumIdleConvergeVerdict.JOB_ACTIVE


def test_the_override_is_named_in_the_decision() -> None:
    now = SEEN + timedelta(minutes=50)
    decision = decide(_inputs(now))
    assert decision.verdict is EnumIdleConvergeVerdict.CONVERGE
    assert "probe deferral cap" in decision.detail
