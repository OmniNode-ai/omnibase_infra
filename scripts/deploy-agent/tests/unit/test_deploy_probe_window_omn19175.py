# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19175: an overdue lane still refuses the guarded M4 C15/C16 windows.

On 2026-10-22 agent converges inside these windows made C16 and the chain
canary reach their terminals after their probe budgets. The committed schedule
must block those starts without removing the other overdue overrides.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from deploy_agent.idle_converge import (
    IDLE_AFTER,
    EnumIdleConvergeVerdict,
    ModelIdleConvergeInputs,
    decide,
    load_probe_windows,
    probe_blocking,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
INCIDENTS = (
    # C16 run 39858728070, correlation 072852f9.
    pytest.param(
        "C16",
        datetime(2026, 10, 22, 21, 38, 46, tzinfo=UTC),
        datetime(2026, 10, 22, 21, 10, tzinfo=UTC),
        id="c16-21z",
    ),
    # C16 run 39806995189.
    pytest.param(
        "C16",
        datetime(2026, 10, 22, 15, 40, 52, tzinfo=UTC),
        datetime(2026, 10, 22, 15, 10, tzinfo=UTC),
        id="c16-15z",
    ),
    # Chain canary (C15) run 39873781212.
    pytest.param(
        "C15",
        datetime(2026, 10, 22, 23, 45, 48, tzinfo=UTC),
        datetime(2026, 10, 22, 23, 47, tzinfo=UTC),
        id="chain-canary-23z",
    ),
)


def _inputs(now: datetime, **overrides: Any) -> ModelIdleConvergeInputs:
    values: dict[str, Any] = {
        "now": now,
        "repository": "omnimarket",
        "routed_elsewhere": True,
        "job_active": False,
        "last_activity": now - IDLE_AFTER,
        "running_ref": "a" * 40,
        "head_ref": "b" * 40,
        "windows_error": None,
        "probe_blocker": None,
        "attempted_heads": frozenset(),
        "behind_since": now - timedelta(minutes=21),
        "max_lag": timedelta(minutes=20),
    }
    values.update(overrides)
    return ModelIdleConvergeInputs(**values)


@pytest.mark.parametrize(("probe_id", "published", "deploy_started"), INCIDENTS)
def test_overdue_refuses_the_incident_probe_window(
    probe_id: str, published: datetime, deploy_started: datetime
) -> None:
    windows = load_probe_windows(REPO_ROOT)
    assert probe_blocking(published, windows) is not None
    blocker = probe_blocking(deploy_started, windows)
    assert blocker is not None and blocker.startswith(probe_id)

    decision = decide(_inputs(deploy_started, probe_blocker=blocker))

    assert decision.verdict is EnumIdleConvergeVerdict.PROBE_WINDOW
    assert decision.verdict is not EnumIdleConvergeVerdict.CONVERGE
    assert decision.detail == blocker


@pytest.fixture
def clear_minute() -> datetime:
    windows = load_probe_windows(REPO_ROOT)
    start = datetime(2026, 10, 22, 2, tzinfo=UTC)
    for minute in range(24 * 60):
        now = start + timedelta(minutes=minute)
        if probe_blocking(now, windows) is None:
            return now
    pytest.fail("no minute outside the guarded probe windows within 24 hours")


def test_overdue_converges_outside_every_guarded_window(clear_minute: datetime) -> None:
    decision = decide(_inputs(clear_minute))
    assert decision.verdict is EnumIdleConvergeVerdict.CONVERGE
    assert "overdue" in decision.detail


def test_overdue_still_overrides_the_idle_wait(clear_minute: datetime) -> None:
    decision = decide(_inputs(clear_minute, last_activity=clear_minute))
    assert decision.verdict is EnumIdleConvergeVerdict.CONVERGE


def test_overdue_still_overrides_an_unreadable_schedule(clear_minute: datetime) -> None:
    decision = decide(_inputs(clear_minute, windows_error="unreadable"))
    assert decision.verdict is EnumIdleConvergeVerdict.CONVERGE
