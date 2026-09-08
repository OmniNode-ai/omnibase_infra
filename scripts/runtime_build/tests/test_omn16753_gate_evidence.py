# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The refresh gate keeps the evidence it decided on (OMN-16753).

Two separable defects, both measured on the 2026-09-08T17:20:47Z stability
refresh, both recorded as FIRST OCCURRENCE in the rolling ledger.

**1. The gate discards the dimension it failed on.** The refresh log and the
receipt recorded the failing criterion as exactly one opaque string::

    health_detail="status='degraded' details.healthy=true (verdict gate: runtime_degraded)"

The ``/health`` body the gate had already decoded carries
``details.runtime_health.dimensions`` -- a list in which every entry names
itself and carries a human-readable detail -- and exactly one of them was
non-HEALTHY. The gate read that block to reach its verdict and then dropped it.
``refresh_stability_lane.sh`` rolls back with ``--force-recreate`` seconds
later, so the container holding the answer is destroyed and its logs go with
it. Recovering which dimension had failed took ~25 minutes off a bus topic,
and that path is retention-bounded: for an older gate run it would not exist
at all. This gate is what decides whether the lane every prod grant's
``stability-proven`` premise resolves from may advance, so a failing verdict
from it has to be actionable on its own.

**2. The manifest fetch has no boot tolerance.** ``check_health_with_retry``
waits a derived, bounded window for the runtime to come up; the manifest fetch
beside it was single-shot. Both the refresh gate and the post-rollback gate hit
``[Errno 104] Connection reset by peer`` on the effects manifest immediately
after the deploy, which forced ``overall=INFRA_ERROR`` and silently shrank the
contract-derived identity base the OMN-15837 declared-groups check scores
against. Same bounded shape as the health probe -- never an unbounded poll, and
still fail-closed when the window expires.

Related Tickets:
    - OMN-16753: this ticket
    - OMN-15837: the declared-groups derivation the effects manifest feeds
    - OMN-17624: the bounded health-verdict wait this mirrors
    - OMN-16994: the dimension whose name was destroyed
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPT_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_health_payload = _load("health_payload")
_verify = _load("verify_stability_refresh")
_verify_dev = _load("verify_dev_refresh")


# The verbatim dimension set the stability lane published at
# 2026-09-08T17:31:44.731789Z, recovered off
# onex.evt.omnibase-infra.runtime-health-check.v1.
_LIVE_DIMENSIONS = [
    {
        "name": "discovery_errors",
        "status": "HEALTHY",
        "detail": "315 contracts loaded cleanly",
    },
    {
        "name": "topic_coverage",
        "status": "HEALTHY",
        "detail": "All expected consumer group(s) covered",
    },
    {"name": "consumer_coverage", "status": "HEALTHY", "detail": "no stranded groups"},
    {
        "name": "projection_attachment",
        "status": "HEALTHY",
        "detail": "All 13 declared projection(s) attached",
    },
    {
        "name": "projection_dlq_saturation",
        "status": "DEGRADED",
        "detail": (
            "1 projection(s) routed 100% of consumed events to a DLQ/quarantine "
            "sink over 8 flow window(s) - offsets commit, so lag reads 0 over a "
            "total loss: projection_work_events"
        ),
    },
    {
        "name": "projection_write_path",
        "status": "HEALTHY",
        "detail": "no non-writing projection is attached",
    },
]

_DEGRADED_BODY = {
    "status": "degraded",
    "details": {
        "healthy": True,
        "runtime_health": {
            "status": "DEGRADED",
            "age_seconds": 12.0,
            "dimensions": _LIVE_DIMENSIONS,
        },
    },
}


class _FakeResponse:
    def __init__(self, payload: bytes, status: int = 200) -> None:
        self._payload = payload
        self.status = status

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


def _opener(body: object, status: int = 200):
    payload = json.dumps(body).encode()

    def _open(url, timeout=10):
        return _FakeResponse(payload, status=status)

    return _open


# =============================================================================
# 1. The non-HEALTHY dimensions survive the verdict
# =============================================================================


@pytest.mark.unit
class TestDimensionCapture:
    def test_extracts_only_the_non_healthy_dimensions(self) -> None:
        found = _health_payload.extract_non_healthy_dimensions(
            json.dumps(_DEGRADED_BODY)
        )
        assert [(d.name, d.status) for d in found] == [
            ("projection_dlq_saturation", "DEGRADED")
        ]
        assert "projection_work_events" in found[0].detail

    def test_a_healthy_body_yields_no_dimensions(self) -> None:
        body = {
            "status": "healthy",
            "details": {
                "runtime_health": {
                    "status": "HEALTHY",
                    "age_seconds": 1.0,
                    "dimensions": _LIVE_DIMENSIONS[:1],
                }
            },
        }
        assert _health_payload.extract_non_healthy_dimensions(json.dumps(body)) == ()

    def test_an_unreadable_body_yields_no_dimensions_and_does_not_raise(self) -> None:
        assert _health_payload.extract_non_healthy_dimensions(b"not json") == ()

    def test_the_verdict_carries_the_dimensions(self) -> None:
        verdict = _health_payload.evaluate_health_body(
            json.dumps(_DEGRADED_BODY),
            require_verdict=True,
            max_verdict_age_seconds=900.0,
        )
        assert not verdict.ok
        assert [d.name for d in verdict.dimensions] == ["projection_dlq_saturation"]

    def test_the_prose_detail_names_the_dimension_too(self) -> None:
        """``reason`` alone is ``runtime_degraded`` -- which dimension is the point."""
        verdict = _health_payload.evaluate_health_body(
            json.dumps(_DEGRADED_BODY),
            require_verdict=True,
            max_verdict_age_seconds=900.0,
        )
        assert "projection_dlq_saturation" in verdict.detail


@pytest.mark.unit
class TestReceiptKeepsTheDimensions:
    @pytest.mark.parametrize("module", [_verify, _verify_dev], ids=["stability", "dev"])
    def test_report_dict_carries_name_and_detail(self, module) -> None:
        report = module.HealthGateReport(lane="stability-test")
        report.health_dimensions = [
            {
                "name": d.name,
                "status": d.status,
                "detail": d.detail,
            }
            for d in _health_payload.extract_non_healthy_dimensions(
                json.dumps(_DEGRADED_BODY)
            )
        ]
        rendered = report.to_dict()
        assert rendered["health_dimensions"][0]["name"] == "projection_dlq_saturation"
        assert "projection_work_events" in rendered["health_dimensions"][0]["detail"]

    def test_a_degraded_gate_run_lands_the_dimension_in_the_receipt(self) -> None:
        """End to end through ``run_health_gate``: the receipt is not lossy."""

        def combo_opener(url, timeout=10):
            if "manifest" in url:
                return _opener({"contracts": []})(url, timeout=timeout)
            return _opener(_DEGRADED_BODY)(url, timeout=timeout)

        report = _verify.run_health_gate(
            lane="stability-test",
            pre_image_ids={},
            expected_revision="rev",
            manifest_url="http://x/manifest",
            health_url="http://x/health",
            broker_container="redpanda-container",
            min_contracts=1,
            declared_groups_file=_SCRIPT_DIR / "consumer_groups_stability.yaml",
            effects_manifest_url=None,
            compose_file=None,
            runner=_always_failing_runner,
            opener=combo_opener,
            sleep_fn=lambda _s: None,
        )
        assert report.health_dimensions
        assert report.health_dimensions[0]["name"] == "projection_dlq_saturation"
        assert (
            "projection_work_events"
            in report.to_dict()["health_dimensions"][0]["detail"]
        )


def _always_failing_runner(
    cmd, capture_output=True, text=True, timeout=30, check=False
):
    """Every docker/rpk probe fails. The health leg is what these tests read."""
    import subprocess

    return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="down")


# =============================================================================
# 2. The manifest fetch gets the same bounded retry the health probe has
# =============================================================================


@pytest.mark.unit
class TestManifestFetchRetry:
    def test_a_reset_during_boot_is_retried_and_then_succeeds(self) -> None:
        attempts: list[str] = []

        def opener(url, timeout=10):
            attempts.append(url)
            if len(attempts) < 3:
                raise OSError(104, "Connection reset by peer")
            return _FakeResponse(json.dumps({"contracts": [1, 2]}).encode())

        payload, err = _verify.fetch_manifest_with_retry(
            "http://x/manifest", opener=opener, sleep_fn=lambda _s: None
        )
        assert err is None
        assert payload == {"contracts": [1, 2]}
        assert len(attempts) == 3

    def test_the_retry_is_bounded_and_still_fails_closed(self) -> None:
        attempts: list[str] = []

        def opener(url, timeout=10):
            attempts.append(url)
            raise OSError(104, "Connection reset by peer")

        payload, err = _verify.fetch_manifest_with_retry(
            "http://x/manifest",
            opener=opener,
            attempts=4,
            sleep_fn=lambda _s: None,
        )
        assert payload is None
        assert err is not None
        assert "Connection reset by peer" in err
        assert len(attempts) == 4

    def test_a_malformed_payload_is_terminal_and_not_retried(self) -> None:
        """Retrying helps a runtime that has not finished booting.

        A 200 carrying a body that is not a manifest is a different fact and
        will not become one by waiting; burning the whole window on it delays
        the gate's verdict for nothing.
        """
        attempts: list[str] = []

        def opener(url, timeout=10):
            attempts.append(url)
            return _FakeResponse(b"<html>nope</html>")

        payload, err = _verify.fetch_manifest_with_retry(
            "http://x/manifest", opener=opener, sleep_fn=lambda _s: None
        )
        assert payload is None
        assert err is not None
        assert len(attempts) == 1

    def test_the_effects_manifest_gets_the_same_tolerance(self) -> None:
        """The friction was on the effects port, so prove it there specifically."""
        seen: list[str] = []

        def combo_opener(url, timeout=10):
            seen.append(url)
            if "effects" in url and seen.count(url) < 3:
                raise OSError(104, "Connection reset by peer")
            if "manifest" in url:
                return _FakeResponse(json.dumps({"contracts": []}).encode())
            return _FakeResponse(json.dumps(_DEGRADED_BODY).encode())

        report = _verify.run_health_gate(
            lane="stability-test",
            pre_image_ids={},
            expected_revision="rev",
            manifest_url="http://x/main/manifest",
            health_url="http://x/health",
            broker_container="redpanda-container",
            min_contracts=0,
            declared_groups_file=_SCRIPT_DIR / "consumer_groups_stability.yaml",
            effects_manifest_url="http://x/effects/manifest",
            compose_file=None,
            runner=_always_failing_runner,
            opener=combo_opener,
            sleep_fn=lambda _s: None,
        )
        assert not any("effects" in e and "reset" in e for e in report.errors), (
            f"effects manifest fetch was not retried: {report.errors}"
        )

    def test_the_dev_gate_manifest_fetch_is_retried_too(self) -> None:
        attempts: list[str] = []

        def opener(url, timeout=10):
            attempts.append(url)
            if len(attempts) < 2:
                raise OSError(104, "Connection reset by peer")
            return _FakeResponse(json.dumps({"contracts": [1]}).encode())

        count, err = _verify_dev.check_manifest_count(
            "http://x/manifest", 1, opener=opener, sleep_fn=lambda _s: None
        )
        assert err is None
        assert count == 1
        assert len(attempts) == 2
