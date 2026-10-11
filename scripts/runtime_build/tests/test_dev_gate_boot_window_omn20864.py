# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20864: the dev refresh gate waits out the runtime's declared start budget.

MEASURED on the lab dev lane 2026-10-11: ``refresh_dev_lane.sh --execute``
started ``omninode-runtime`` at 19:30:41Z. The health gate probed ``/health``,
read HTTP 503 "runtime still starting" on every attempt of its 600 s verdict
window, and the receipt ``20261011T191956Z-39781a1da00a`` said FAILED at about
19:37Z. The runtime's health monitor read HEALTHY at 19:41:22Z, inside the
2000 s start budget the container declares (start_period 1800 s + 5 x (30 s +
10 s)). The verdict bound counts from the monitor's first check, and a runtime
that is still starting has not started that clock.

The fix: the refresh script passes the seconds left of that declared budget as
``--boot-window-seconds``; while the runtime says "still starting" inside it, a
probe does not spend one of the verdict bound's attempts. A runtime that never
finishes starting still fails, at the end of the window plus the bound.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import urllib.error
from email.message import Message
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from health_payload import (
    REASON_RUNTIME_STARTING,
    derive_verdict_wait_bound,
)

pytestmark = pytest.mark.unit


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _SCRIPT_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_DEV = _load("verify_dev_refresh")

_STARTING = json.dumps(
    {
        "status": "degraded",
        "details": {
            "healthy": False,
            "is_running": False,
            "runtime_attached": True,
            "startup_in_progress": True,
            "runtime_health": None,
        },
    }
).encode()
_HEALTHY = json.dumps(
    {
        "status": "healthy",
        "details": {"runtime_health": {"status": "HEALTHY", "age_seconds": 1.0}},
    }
).encode()


class _Resp:
    def __init__(self, body: bytes) -> None:
        self._body = body
        self.status = 200

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *_: object) -> None:
        return None


class _Clock:
    """A clock the sleeps advance, so a 2000 s window runs in no real time."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _opener(starting_probes: int) -> tuple[object, dict[str, int]]:
    calls = {"n": 0}

    def _open(url: str, timeout: int = 10) -> _Resp:
        calls["n"] += 1
        if calls["n"] <= starting_probes:
            raise urllib.error.HTTPError(
                url, 503, "Service Unavailable", Message(), io.BytesIO(_STARTING)
            )
        return _Resp(_HEALTHY)

    return _open, calls


BOUND = derive_verdict_wait_bound(
    check_interval_seconds=300.0, boot_grace_seconds=120.0
)
#: 19:30:41Z start, about 19:31Z gate, 19:41:22Z healthy: ~41 probes at 15 s.
STARTING_PROBES_2026_10_11 = 41


def test_replay_2026_10_11_without_the_window_the_gate_fails() -> None:
    """The positive control: the receipt's FAILED, reproduced."""
    clock = _Clock()
    opener, _calls = _opener(STARTING_PROBES_2026_10_11)

    verdict, described = _DEV.check_runtime_health_with_retry(
        "http://x/health",
        label="omninode-runtime",
        opener=opener,
        sleep_fn=clock.sleep,
        clock_fn=clock,
    )

    assert verdict.ok is False
    assert verdict.reason == REASON_RUNTIME_STARTING
    assert "exhausted" in described


def test_replay_2026_10_11_inside_the_declared_window_the_gate_passes() -> None:
    clock = _Clock()
    opener, calls = _opener(STARTING_PROBES_2026_10_11)

    verdict, described = _DEV.check_runtime_health_with_retry(
        "http://x/health",
        label="omninode-runtime",
        opener=opener,
        sleep_fn=clock.sleep,
        clock_fn=clock,
        boot_window_seconds=1940.0,
    )

    assert verdict.ok is True
    assert calls["n"] == STARTING_PROBES_2026_10_11 + 1
    assert f"{STARTING_PROBES_2026_10_11} still-starting probe(s) not counted" in (
        described
    )


def test_a_runtime_that_never_finishes_starting_still_fails() -> None:
    """The window is a ceiling: it ends, then the ordinary bound runs out."""
    clock = _Clock()
    opener, calls = _opener(10_000)

    verdict, described = _DEV.check_runtime_health_with_retry(
        "http://x/health",
        label="omninode-runtime",
        opener=opener,
        sleep_fn=clock.sleep,
        clock_fn=clock,
        boot_window_seconds=300.0,
    )

    assert verdict.ok is False
    assert verdict.reason == REASON_RUNTIME_STARTING
    assert "exhausted" in described
    # 300 s of uncounted probes at the bound's interval, then the whole bound.
    assert calls["n"] == int(300 / BOUND.interval_seconds) + BOUND.attempts


def test_the_refresh_script_passes_the_declared_window_to_both_gates() -> None:
    text = (_SCRIPT_DIR / "refresh_dev_lane.sh").read_text(encoding="utf-8")
    assert 'source "${SCRIPT_DIR}/runtime_health_wait.sh"' in text
    assert text.count("--boot-window-seconds") == 2
    assert "runtime_health_budget_seconds" in text
