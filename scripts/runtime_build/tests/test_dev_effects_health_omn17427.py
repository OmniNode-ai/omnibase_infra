# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev refresh gate probes runtime-effects (:8086), not only runtime (:8085) [OMN-17427].

On 2026-10-09 ``refresh_dev_lane.sh`` returned ``PASS`` / ``SUCCESS`` for the
``.201`` dev lane while ``omninode-runtime-effects`` was in a crash loop (11
restarts by 15:06Z): the gate's health leg fetched ``:8085/health`` and nothing
else, and the effects container's ``State.Status`` happened to read
``running`` between two crashes. A lane whose second runtime cannot stay up is
not serving, so the gate now fetches ``:8086/health`` too, under the same
bounded wait, and a lane whose effects runtime never answers fails closed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import urllib.error
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _SCRIPT_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_DEV = _load("verify_dev_refresh")
_DECISION = _load("lane_rollback_decision")

_MAIN_HEALTH = "http://lane:8085/health"
_EFFECTS_HEALTH = "http://lane:8086/health"
_MANIFEST = "http://lane:8085/v1/introspection/manifest"

_HEALTHY_BODY: dict[str, object] = {
    "status": "healthy",
    "details": {"runtime_health": {"status": "HEALTHY", "age_seconds": 1.0}},
}


class _Resp:
    status = 200

    def __init__(self, body: dict[str, object]) -> None:
        self._body = json.dumps(body).encode()

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *_: object) -> None:
        return None


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class _Lane:
    """A lane whose :8085 is healthy and whose :8086 follows a script."""

    def __init__(self, effects: list[object]) -> None:
        self.effects = list(effects)
        self.effects_calls = 0
        self.main_calls = 0

    def __call__(self, url: str, timeout: float = 10) -> _Resp:
        if url == _MANIFEST:
            return _Resp({"contracts": [1, 2, 3]})
        if url == _MAIN_HEALTH:
            self.main_calls += 1
            return _Resp(_HEALTHY_BODY)
        assert url == _EFFECTS_HEALTH, url
        step = self.effects[min(self.effects_calls, len(self.effects) - 1)]
        self.effects_calls += 1
        if isinstance(step, BaseException):
            raise step
        assert isinstance(step, dict)
        return _Resp(step)


def _refused() -> urllib.error.URLError:
    return urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))


def _gate(lane: _Lane, clock: _Clock, **overrides: object) -> object:
    kwargs: dict[str, object] = {
        "lane": "dev",
        "pre_image_ids": {},
        "container_ids": {},
        "expected_revision": "rev",
        "manifest_url": _MANIFEST,
        "health_url": _MAIN_HEALTH,
        "effects_health_url": _EFFECTS_HEALTH,
        "broker_container": "redpanda",
        "min_contracts": 1,
        "opener": lane,
        "sleep_fn": clock.advance,
        "manifest_clock_fn": clock.now,
        "require_digest_change": False,
    }
    kwargs.update(overrides)
    return _DEV.run_health_gate(**kwargs)


@pytest.mark.unit
def test_a_crash_looping_effects_runtime_fails_the_gate_while_main_is_healthy() -> None:
    """The 2026-10-09 shape: :8085 answers healthy, :8086 never does."""
    lane = _Lane([_refused()])
    clock = _Clock()

    report = _gate(lane, clock)

    assert report.health_ok is True, "the control: :8085 really was healthy"
    assert report.effects_health_ok is False
    assert "runtime-effects" in (report.effects_health_detail or "")
    assert _EFFECTS_HEALTH in (report.effects_health_detail or "")
    assert report.overall != "PASS"
    assert report.to_dict()["effects_health_ok"] is False


@pytest.mark.unit
def test_the_effects_wait_is_bounded_by_the_window_and_then_fails_closed() -> None:
    lane = _Lane([_refused()])
    clock = _Clock()

    _gate(lane, clock, effects_window_seconds=60.0)

    assert lane.effects_calls > 1, "an effects runtime still binding gets retried"
    assert clock.now() <= 60.0 + 1e-6, "the wait outran its own window"
    assert lane.effects_calls <= 6


@pytest.mark.unit
def test_an_effects_runtime_that_binds_late_passes_after_a_bounded_wait() -> None:
    lane = _Lane([_refused(), _refused(), _HEALTHY_BODY])
    clock = _Clock()

    report = _gate(lane, clock)

    assert report.effects_health_ok is True
    assert lane.effects_calls == 3
    assert "satisfied" in (report.effects_verdict_wait or "")


@pytest.mark.unit
def test_a_degraded_effects_verdict_is_not_a_pass() -> None:
    degraded: dict[str, object] = {
        "status": "degraded",
        "details": {"runtime_health": {"status": "DEGRADED", "age_seconds": 1.0}},
    }
    lane = _Lane([degraded])
    clock = _Clock()

    report = _gate(lane, clock)

    assert report.health_ok is True
    assert report.effects_health_ok is False


@pytest.mark.unit
def test_the_effects_probe_is_not_optional() -> None:
    """A gate that can be handed no effects URL is a gate that skips it."""
    with pytest.raises(TypeError, match="effects_health_url"):
        _DEV.run_health_gate(
            lane="dev",
            pre_image_ids={},
            container_ids={},
            expected_revision="rev",
            manifest_url=_MANIFEST,
            health_url=_MAIN_HEALTH,
            broker_container="redpanda",
            min_contracts=1,
        )
    with pytest.raises(SystemExit) as excinfo:
        _DEV.main(["--expected-revision", "rev", "--container-ids", "{}"])
    assert excinfo.value.code == 2


@pytest.mark.unit
def test_a_healthy_effects_runtime_and_a_healthy_main_pass_every_health_leg() -> None:
    lane = _Lane([_HEALTHY_BODY])
    report = _gate(lane, _Clock())
    assert report.health_ok is True
    assert report.effects_health_ok is True
    assert report.to_dict()["effects_health_status"] == "healthy"


@pytest.mark.unit
def test_overall_needs_the_effects_leg() -> None:
    """Positive control, then the single flipped dimension."""
    report = _DEV.HealthGateReport(lane="dev", require_digest_change=False)
    report.services = [
        _DEV.ServiceDigestCheck(
            service=name,
            container=f"c-{name}",
            pre_image_id=None,
            post_image_id="img",
            digest_changed=True,
            revision_label="rev",
            expected_revision="rev",
            revision_match=True,
            container_state="running",
            running=True,
        )
        for name in _DEV.CORE_SERVICE_NAMES
    ]
    report.manifest_ok = True
    report.health_ok = True
    report.cluster_healthy = True
    report.effects_health_ok = True
    assert report.overall == "PASS"

    report.effects_health_ok = False
    assert report.overall == "FAIL"


@pytest.mark.unit
def test_the_rollback_decision_counts_a_dead_effects_runtime_as_not_serving() -> None:
    serving_but_effects_down = {
        "health_ok": True,
        "manifest_ok": True,
        "cluster_healthy": True,
        "core_services_running": True,
        "effects_health_ok": False,
        "errors": [],
    }
    assert "effects_health_ok=false" in _DECISION.health_dimension_failures(
        serving_but_effects_down
    )

    healthy = {**serving_but_effects_down, "effects_health_ok": True}
    assert _DECISION.health_dimension_failures(healthy) == ()

    # The stability gate reports no effects leg and its decision is unchanged.
    stability = {k: v for k, v in healthy.items() if k != "effects_health_ok"}
    assert _DECISION.health_dimension_failures(stability) == ()


@pytest.mark.unit
def test_the_refresh_script_hands_the_gate_the_effects_url_on_both_runs() -> None:
    """Both the first gate and the post-rollback re-verification probe :8086."""
    text = (_SCRIPT_DIR / "refresh_dev_lane.sh").read_text(encoding="utf-8")
    assert "require_contract_var DEV_RUNTIME_EFFECTS_PORT" in text
    assert (
        'EFFECTS_HEALTH_URL="http://${LANE_PROBE_HOST}:${DEV_RUNTIME_EFFECTS_PORT}/health"'
        in text
    )
    assert text.count('--effects-health-url "${EFFECTS_HEALTH_URL}"') == 2
    assert text.count("run_verify \\") == 2
