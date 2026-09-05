# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17624: the refresh gate must not sign off inside the blind window.

``ServiceRuntimeHealthMonitor._loop`` sleeps one ``check_interval`` BEFORE its
first check, so no ``details.runtime_health`` verdict exists for roughly the
first 300s after a recreate. ``fold_runtime_verdict_into_status`` cannot
degrade ``status`` before that verdict exists, so a gate probing inside the
window reads a genuine ``status: healthy`` and honestly records PASS -- which
is exactly what the ``20260902T145841Z-810c818a10b3`` receipt did.

That makes the stability lane's refresh receipt -- the ``stability-proven``
premise of a live prod-promotion grant -- structurally unable to observe a
DEGRADED lane.

Two design points these tests pin, both of which differ from the ticket's
first-pass framing:

1. The wait bound is DERIVED, never hardcoded. Boot grace suppresses emission
   while ``elapsed < grace``, so with a short interval the first VISIBLE
   verdict is later than one interval. The check landing exactly ON the grace
   boundary is a monotonic-clock race -- ``120 < 120`` is False by a hair --
   so the bound counts the first check strictly AFTER grace and never bets on
   that tie: at interval=60s/grace=120s that is 180s, not 120s. A fixed 390s
   constant is right only for the default 300s/120s pair.
2. Absence of a verdict FAILS, it does not skip. There is no profile flag
   declaring "monitor disabled": ``service_kernel.py`` starts the monitor only
   ``if use_kafka``, and swallows a start failure with
   ``except Exception: ... runtime_health_monitor = None``. A crashed monitor
   and an intentionally absent one are indistinguishable at the /health
   boundary, so skipping on absence would rebuild the blind spot this closes.
   A monitor-less profile must opt out explicitly, and the receipt records it.
"""

from __future__ import annotations

import json

import pytest
from health_payload import (
    HEALTH_POLICY_VERDICT_REQUIRED,
    derive_verdict_wait_bound,
    evaluate_health_body,
)

_PRE_VERDICT_BODY: dict[str, object] = {
    "status": "healthy",
    "details": {"healthy": True},
}


def _body_with_verdict(status: str = "HEALTHY", age_seconds: float = 1.0) -> bytes:
    """A body shaped like the real one.

    ``build_runtime_health_block`` (runtime_health_block.py:93) ALWAYS emits
    ``age_seconds`` beside ``status``. An earlier version of this helper made
    it optional, producing "fresh verdict" fixtures no live lane can emit --
    and an unrealistic fixture is how a guard passes while the property it
    names goes unexercised.
    """
    return json.dumps(
        {
            "status": "healthy",
            "details": {
                "healthy": True,
                "runtime_health": {"status": status, "age_seconds": age_seconds},
            },
        }
    ).encode()


# ── AC1: no sign-off before a verdict exists ────────────────────────────────


def test_gate_refuses_a_body_carrying_no_monitor_verdict() -> None:
    """The 2026-09-02 body. Honest, healthy-looking, and unprovable."""
    verdict = evaluate_health_body(
        json.dumps(_PRE_VERDICT_BODY).encode(), require_verdict=True
    )
    assert verdict.ok is False
    assert verdict.policy == HEALTH_POLICY_VERDICT_REQUIRED
    assert "verdict" in verdict.detail.lower()


def test_gate_accepts_a_body_once_a_fresh_verdict_exists() -> None:
    verdict = evaluate_health_body(_body_with_verdict(), require_verdict=True)
    assert verdict.ok is True
    assert verdict.policy == HEALTH_POLICY_VERDICT_REQUIRED


def test_a_degraded_verdict_still_fails_when_required() -> None:
    """Requiring a verdict must not accidentally accept a bad one."""
    verdict = evaluate_health_body(
        _body_with_verdict(status="DEGRADED"), require_verdict=True
    )
    assert verdict.ok is False


def test_default_call_is_unchanged_for_existing_callers() -> None:
    """require_verdict defaults False: OMN-17563's behaviour is untouched."""
    verdict = evaluate_health_body(json.dumps(_PRE_VERDICT_BODY).encode())
    assert verdict.ok is True


# ── AC2: a stale verdict is stale, not healthy ──────────────────────────────


def test_a_verdict_older_than_the_max_age_is_not_healthy() -> None:
    verdict = evaluate_health_body(
        _body_with_verdict(age_seconds=9_000.0),
        require_verdict=True,
        max_verdict_age_seconds=600.0,
    )
    assert verdict.ok is False


# ── AC3: the bound is derived from the real interval, and recorded ──────────


@pytest.mark.parametrize(
    ("interval", "grace", "at_least"),
    [
        (300.0, 120.0, 300.0),
        (60.0, 120.0, 180.0),
        (300.0, 0.0, 300.0),
        (30.0, 300.0, 300.0),
    ],
)
def test_bound_clears_the_first_visible_verdict(
    interval: float, grace: float, at_least: float
) -> None:
    """Boot grace suppresses EMISSION, so the first visible verdict is the
    first check at or after the grace window -- not simply one interval."""
    bound = derive_verdict_wait_bound(
        check_interval_seconds=interval, boot_grace_seconds=grace
    )
    assert bound.total_seconds >= at_least, (
        f"bound {bound.total_seconds}s cannot observe a first verdict at {at_least}s"
    )
    assert bound.attempts >= 1
    assert bound.interval_seconds > 0


def test_bound_is_recordable_in_the_receipt() -> None:
    bound = derive_verdict_wait_bound(
        check_interval_seconds=300.0, boot_grace_seconds=120.0
    )
    described = bound.describe()
    assert "300" in described
    assert str(int(bound.total_seconds)) in described


def test_bound_is_finite_and_never_unbounded() -> None:
    """AC5: a refresh must fail after a known window, never hang."""
    bound = derive_verdict_wait_bound(
        check_interval_seconds=300.0, boot_grace_seconds=120.0
    )
    assert bound.attempts < 1000
    assert bound.total_seconds < 3600


# ── Both gates, so neither can be fixed without the other ───────────────────
#
# The dev gate is blind in the identical window and had NO gate-level health
# test at all: adding a required join to it broke nothing in the suite. That
# silence is why fixing only the stability lane would have looked complete.

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_SCRIPT_DIR = Path(__file__).resolve().parent.parent


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _SCRIPT_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_STABILITY = _load("verify_stability_refresh")
_DEV = _load("verify_dev_refresh")

_BOTH_GATES = pytest.mark.parametrize(
    "gate", [pytest.param(_STABILITY, id="stability"), pytest.param(_DEV, id="dev")]
)


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


def _opener_for(body: dict[str, object]) -> object:
    def _open(url: str, timeout: int = 10) -> _Resp:
        return _Resp(json.dumps(body).encode())

    return _open


@_BOTH_GATES
def test_gate_refuses_when_no_verdict_ever_arrives(gate: ModuleType) -> None:
    verdict, described = gate.check_health_with_retry(
        "http://x/health",
        opener=_opener_for(_PRE_VERDICT_BODY),
        sleep_fn=lambda _s: None,
    )
    assert verdict.ok is False
    assert "exhausted" in described


@_BOTH_GATES
def test_gate_passes_once_the_verdict_is_present(gate: ModuleType) -> None:
    body = {
        "status": "healthy",
        "details": {"runtime_health": {"status": "HEALTHY", "age_seconds": 1.0}},
    }
    verdict, described = gate.check_health_with_retry(
        "http://x/health", opener=_opener_for(body), sleep_fn=lambda _s: None
    )
    assert verdict.ok is True
    assert "satisfied on attempt 1" in described


@_BOTH_GATES
def test_gate_wait_is_bounded_and_terminates(gate: ModuleType) -> None:
    """AC5: a refresh fails after a known window; it never hangs."""
    calls = {"n": 0}

    def _counting_open(url: str, timeout: int = 10) -> _Resp:
        calls["n"] += 1
        return _Resp(json.dumps(_PRE_VERDICT_BODY).encode())

    bound = derive_verdict_wait_bound(
        check_interval_seconds=300.0, boot_grace_seconds=120.0
    )
    _verdict, _desc = gate.check_health_with_retry(
        "http://x/health", opener=_counting_open, sleep_fn=lambda _s: None
    )
    assert calls["n"] == bound.attempts, (
        f"probed {calls['n']} times, bound declares {bound.attempts} -- "
        "the wait is not the bound it reports"
    )


@_BOTH_GATES
def test_explicit_opt_out_is_recorded_never_inferred(gate: ModuleType) -> None:
    """A monitor-less profile opts out ON THE RECORD.

    Absence is never read as permission: service_kernel starts the monitor
    only ``if use_kafka`` and swallows a start failure, so a crashed monitor
    is indistinguishable from an absent one at this boundary.
    """
    verdict, described = gate.check_health_with_retry(
        "http://x/health",
        opener=_opener_for(_PRE_VERDICT_BODY),
        require_verdict=False,
        sleep_fn=lambda _s: None,
    )
    assert verdict.ok is True
    assert "opted out" in described


# ── Hostile-reviewer findings (omnibase_infra#3208), all three upheld ───────
#
# The reviewer was right on each of these and the first one is the serious
# one: I built the freshness capability, tested it, and wrote AC2 down as
# satisfied -- while no production caller passed the parameter. A test that
# exercises an argument real callers never supply proves nothing about the
# shipped behaviour, which is the exact defect class this ticket exists to
# close.


def _stale_verdict_body(age: float = 9_000.0) -> dict[str, object]:
    return {
        "status": "healthy",
        "details": {"runtime_health": {"status": "HEALTHY", "age_seconds": age}},
    }


@_BOTH_GATES
def test_freshness_is_enforced_without_the_caller_asking(gate: ModuleType) -> None:
    """fp=ab80aaa67e08 / 6e73ced28ece / e85978c66d2b.

    A monitor that emits one verdict then crashes leaves a frozen
    runtime_health in the body forever. If the gate only enforces freshness
    when a caller opts in, the blind window becomes 'before first verdict'
    PLUS FOREVER -- strictly worse than the bug being fixed.
    """
    verdict, _desc = gate.check_health_with_retry(
        "http://x/health",
        opener=_opener_for(_stale_verdict_body()),
        sleep_fn=lambda _s: None,
    )
    assert verdict.ok is False, (
        "a 9000s-old verdict was accepted with no explicit max age -- "
        "freshness is opt-in, so production enforces nothing"
    )


@_BOTH_GATES
def test_a_fatal_probe_does_not_burn_the_whole_window(gate: ModuleType) -> None:
    """fp=bc154ca9c0a3. Only verdict_absent warrants waiting.

    A connection-refused or malformed endpoint is knowable on attempt 1.
    Retrying it for the full derived bound turns a fast, correct failure into
    a ~600s stall on every refresh.
    """
    calls = {"n": 0}

    def _broken_open(url: str, timeout: int = 10) -> _Resp:
        calls["n"] += 1
        return _Resp(b"this is not json at all")

    verdict, _desc = gate.check_health_with_retry(
        "http://x/health", opener=_broken_open, sleep_fn=lambda _s: None
    )
    assert verdict.ok is False
    assert calls["n"] == 1, (
        f"probed {calls['n']} times for an unreadable body -- a permanent "
        "failure should not consume the wait window"
    )


def test_the_recorded_bound_states_worst_case_wall_clock() -> None:
    """fp=b6cd79370117 / 65ad276129d1.

    Each attempt performs an HTTP fetch (timeout 10s) BEFORE sleeping, so real
    elapsed time is attempts * (poll + fetch), not attempts * poll. A bound
    whose stated purpose is to be reviewable must not understate itself.
    """
    bound = derive_verdict_wait_bound(
        check_interval_seconds=300.0, boot_grace_seconds=120.0
    )
    described = bound.describe()
    # STRICTLY greater: `>=` passed even when worst_case was computed as
    # attempts * poll, i.e. with the fetch cost omitted -- the mutation run
    # proved that assertion vacuous, which is the same defect class this
    # ticket exists to close.
    assert bound.worst_case_seconds > bound.total_seconds, (
        "worst case does not exceed sleep time, so the probe fetch cost is not counted"
    )
    expected = bound.attempts * (bound.interval_seconds + bound.probe_timeout_seconds)
    assert bound.worst_case_seconds == expected
    assert str(int(bound.worst_case_seconds)) in described, (
        "the receipt does not state the worst-case wall clock a reviewer "
        "would actually observe"
    )
