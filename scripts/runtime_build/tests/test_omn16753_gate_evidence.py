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
_manifest_fetch = _load("manifest_fetch")
_verify = _load("verify_stability_refresh")
_verify_dev = _load("verify_dev_refresh")


class _FakeClock:
    """A monotonic clock a test drives, so a bound can be pinned not slept."""

    def __init__(self) -> None:
        self._t = 0.0

    def now(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds


def _budget(total_seconds: float = 3600.0):
    """A budget generous enough that a test's own attempt count is the subject."""
    clock = _FakeClock()
    return _manifest_fetch.RetryBudget(
        total_seconds=total_seconds,
        interval_seconds=15.0,
        sleep_fn=clock.advance,
        monotonic_fn=clock.now,
    )


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
        found = _health_payload.extract_non_healthy_dimensions(_DEGRADED_BODY)
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
        assert _health_payload.extract_non_healthy_dimensions(body) == ()

    def test_an_unreadable_body_yields_no_dimensions_and_does_not_raise(self) -> None:
        """A non-mapping document is total, not a second failure."""
        assert _health_payload.extract_non_healthy_dimensions("not a mapping") == ()
        assert _health_payload.extract_non_healthy_dimensions(None) == ()
        assert _health_payload.extract_non_healthy_dimensions({"details": 3}) == ()

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
    def test_a_degraded_probe_reaches_the_receipt_in_both_gates(self, module) -> None:
        """Through ``run_health_gate``, not a hand-assigned field.

        The first revision built a ``HealthGateReport``, assigned
        ``health_dimensions`` by hand and asserted ``to_dict`` round-tripped
        it -- which tests dataclass serialization and would have passed even if
        ``run_health_gate`` never populated the field. Both gates are driven
        for real here; the parametrization is only worth having if each arm
        exercises its own gate.
        """

        def combo_opener(url, timeout=10):
            if "manifest" in url:
                return _opener({"contracts": []})(url, timeout=timeout)
            return _opener(_DEGRADED_BODY)(url, timeout=timeout)

        kwargs: dict[str, object] = {
            "lane": "stability-test",
            "pre_image_ids": {},
            "expected_revision": "rev",
            "manifest_url": "http://x/manifest",
            "health_url": "http://x/health",
            "broker_container": "redpanda-container",
            "min_contracts": 0,
            "runner": _always_failing_runner,
            "opener": combo_opener,
            "sleep_fn": lambda _s: None,
        }
        if module is _verify:
            kwargs.update(
                {
                    "declared_groups_file": _SCRIPT_DIR
                    / "consumer_groups_stability.yaml",
                    "effects_manifest_url": None,
                    "compose_file": None,
                }
            )
        else:
            kwargs["container_ids"] = {}
        report = module.run_health_gate(**kwargs)
        rendered = report.to_dict()
        assert rendered["health_dimensions"], (
            "run_health_gate did not populate health_dimensions"
        )
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

        fetched = _verify.fetch_manifest_within_budget(
            "http://x/manifest", opener=opener, budget=_budget()
        )
        assert fetched.error is None
        assert fetched.payload == {"contracts": [1, 2]}
        assert len(attempts) == 3

    def test_the_retry_is_bounded_and_still_fails_closed(self) -> None:
        attempts: list[str] = []

        def opener(url, timeout=10):
            attempts.append(url)
            raise OSError(104, "Connection reset by peer")

        clock = _FakeClock()
        fetched = _verify.fetch_manifest_within_budget(
            "http://x/manifest",
            opener=opener,
            budget=_manifest_fetch.RetryBudget(
                total_seconds=60.0,
                interval_seconds=15.0,
                sleep_fn=clock.advance,
                monotonic_fn=clock.now,
            ),
        )
        assert fetched.payload is None
        assert fetched.error is not None
        assert "Connection reset by peer" in fetched.error
        # 60 s window / 15 s interval: three sleeps fit (the fourth would
        # consume the whole remainder and buy no attempt), so four attempts.
        assert len(attempts) == 4
        assert clock.now() <= 60.0

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

        fetched = _verify.fetch_manifest_within_budget(
            "http://x/manifest", opener=opener, budget=_budget()
        )
        assert fetched.payload is None
        assert fetched.error is not None
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

        count, err, history = _verify_dev.fetch_manifest_contract_count(
            "http://x/manifest", opener=opener, budget=_budget()
        )
        assert err is None
        assert count == 1
        assert len(attempts) == 2
        assert len(history) == 1


# =============================================================================
# 3. Retry eligibility is a TYPED signal, not an error-string prefix
# =============================================================================


@pytest.mark.unit
class TestRetryEligibilityIsTyped:
    """The first revision decided this with ``err.startswith(...)``.

    That made the retry contract a string shared between two private functions
    in two modules: rewording either message silently stopped every retry and
    regressed a booting lane back to ``INFRA_ERROR`` -- the exact defect
    OMN-16753 closes -- with no test failing. The classification is now
    :class:`EnumManifestFetchFailure`.
    """

    def test_a_transport_failure_is_classified_transport_whatever_it_says(
        self,
    ) -> None:
        def opener(url, timeout=10):
            raise OSError(104, "totally reworded connection problem")

        outcome = _manifest_fetch.fetch_manifest_once("http://x/m", opener=opener)
        assert outcome.failure is _manifest_fetch.EnumManifestFetchFailure.TRANSPORT
        assert outcome.retriable is True

    def test_a_reworded_transport_message_is_still_retried(self) -> None:
        """The property the string prefix could not hold.

        The message deliberately does NOT start with the old sentinel prefix.
        Under the string contract this fetch would have stopped after one
        attempt; it must keep retrying.
        """
        attempts: list[str] = []

        def opener(url, timeout=10):
            attempts.append(url)
            if len(attempts) < 3:
                raise OSError("the runtime is not answering yet")
            return _FakeResponse(json.dumps({"contracts": [1]}).encode())

        fetched = _verify.fetch_manifest_within_budget(
            "http://x/manifest", opener=opener, budget=_budget()
        )
        assert fetched.error is None
        assert len(attempts) == 3

    def test_a_non_transport_failure_is_terminal_and_says_so(self) -> None:
        def opener(url, timeout=10):
            return _FakeResponse(json.dumps(["not", "a", "dict", 5]).encode())

        outcome = _manifest_fetch.fetch_manifest_once("http://x/m", opener=opener)
        # A list IS a manifest shape (contracts), so use a scalar body instead.
        assert outcome.ok

        def scalar_opener(url, timeout=10):
            return _FakeResponse(b"42")

        outcome = _manifest_fetch.fetch_manifest_once(
            "http://x/m", opener=scalar_opener
        )
        assert outcome.failure is _manifest_fetch.EnumManifestFetchFailure.UNREADABLE
        assert outcome.retriable is False


# =============================================================================
# 4. The retry window is bounded ONCE for the whole run, not once per URL
# =============================================================================


@pytest.mark.unit
class TestManifestRetryWindowIsBoundedOverall:
    """MAJOR: two serial per-URL windows doubled the gate's worst case.

    ``run_health_gate`` fetches the main manifest and then the effects
    manifest. With a window each, a partitioned effects port cost ~720 s of
    manifest retrying BEFORE the health leg started its own 360 s wait. The
    budget is now shared, so the second fetch inherits the remainder.
    """

    def test_both_manifests_share_one_window(self) -> None:
        clock = _FakeClock()
        budget = _manifest_fetch.RetryBudget(
            total_seconds=_manifest_fetch.MANIFEST_FETCH_WINDOW_SECONDS,
            interval_seconds=_manifest_fetch.MANIFEST_FETCH_INTERVAL_SECONDS,
            sleep_fn=clock.advance,
            monotonic_fn=clock.now,
        )

        def dead(url, timeout=10):
            # Every attempt burns its full socket timeout, then resets.
            clock.advance(timeout)
            raise OSError(104, "Connection reset by peer")

        main = _verify.fetch_manifest_within_budget(
            "http://x/main/manifest", opener=dead, budget=budget
        )
        effects = _verify.fetch_manifest_within_budget(
            "http://x/effects/manifest", opener=dead, budget=budget
        )

        assert main.error is not None
        assert effects.error is not None
        # THE BOUND. Both fetches together, sleeps and socket timeouts, never
        # exceed the single window -- which is the health leg's own 360 s.
        assert clock.now() <= _manifest_fetch.MANIFEST_FETCH_WINDOW_SECONDS
        # And the second fetch really did inherit a spent budget rather than
        # getting a fresh one: it was never attempted, and says so.
        assert (
            effects.outcome.failure
            is _manifest_fetch.EnumManifestFetchFailure.BUDGET_EXPIRED
        )
        assert "expired before this URL was attempted" in (effects.error or "")

    def test_the_window_matches_the_health_legs_own_wait(self) -> None:
        """Sized to the health probe, not to a multiple of it."""
        assert _manifest_fetch.MANIFEST_FETCH_WINDOW_SECONDS == 360.0


# =============================================================================
# 5. The failure history survives into the receipt
# =============================================================================


@pytest.mark.unit
class TestFailureHistoryIsKept:
    """MINOR: the loop overwrote ``err`` each attempt and reported only the last.

    A first attempt that reset and a last attempt that timed out is a different
    story from twenty-four identical resets, and a gate whose declared purpose
    is preserving evidence should not be the thing that discards it.
    """

    def test_every_failed_attempt_is_recorded_oldest_first(self) -> None:
        messages = ["reset one", "reset two", "reset three"]
        seen: list[str] = []

        def opener(url, timeout=10):
            message = messages[min(len(seen), len(messages) - 1)]
            seen.append(message)
            raise OSError(104, message)

        clock = _FakeClock()
        # 45 s / 15 s leaves room for exactly two sleeps -> three attempts.
        fetched = _verify.fetch_manifest_within_budget(
            "http://x/manifest",
            opener=opener,
            budget=_manifest_fetch.RetryBudget(
                total_seconds=45.0,
                interval_seconds=15.0,
                sleep_fn=clock.advance,
                monotonic_fn=clock.now,
            ),
        )
        assert len(fetched.history) == 3
        assert [m for m in messages if any(m in h for h in fetched.history)] == messages
        assert "reset one" in fetched.history[0]
        # The old loop overwrote ``err`` every attempt: only this last one
        # survived, and the first two were unrecoverable.
        assert fetched.error is not None and "reset three" in fetched.error

    def test_the_receipt_reports_the_last_n_attempts(self) -> None:
        clock = _FakeClock()

        def dead(url, timeout=10):
            if "health" in url:
                return _FakeResponse(json.dumps(_DEGRADED_BODY).encode())
            raise OSError(104, "Connection reset by peer")

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
            opener=dead,
            sleep_fn=clock.advance,
            manifest_window_seconds=45.0,
            manifest_clock_fn=clock.now,
        )
        rendered = report.to_dict()["manifest_fetch_attempts"]
        assert rendered, "the receipt lost the fetch failure history"
        assert len(rendered) <= _manifest_fetch.MANIFEST_FETCH_HISTORY_LIMIT
        assert any("Connection reset by peer" in line for line in rendered)
        # More than the LAST error survives: the receipt carries repeated
        # attempts, which is what distinguishes a persistent failure from an
        # intermittent one.
        assert len(rendered) > 1
        # Both URLs are represented -- the effects fetch shared the main
        # fetch's window rather than being given a fresh one.
        assert any("main/manifest" in line for line in rendered)
        assert any("effects/manifest" in line for line in rendered)
        assert report.errors, "a lane whose manifests never fetched must not read clean"


# =============================================================================
# 6. sleep_fn is a real callable type, and the remote detail is bounded
# =============================================================================


@pytest.mark.unit
class TestTypedSeamsAndBoundedEvidence:
    def test_sleep_fn_is_annotated_as_a_callable_not_object(self) -> None:
        """MINOR: ``object | None`` forced a ``type: ignore[operator]``.

        Asserted on the annotation rather than on behaviour because the defect
        WAS the annotation -- an ignore that suppresses the one error proving
        the parameter is misdeclared.
        """
        import inspect

        for module, name in (
            (_verify, "run_health_gate"),
            (_verify_dev, "run_health_gate"),
            (_health_payload, "wait_for_verdict"),
        ):
            annotation = (
                inspect.signature(getattr(module, name))
                .parameters["sleep_fn"]
                .annotation
            )
            assert "Callable[[float], None]" in str(annotation), (
                f"{module.__name__}.{name} still declares sleep_fn as {annotation}"
            )

    def test_no_operator_ignore_survives_on_a_sleep_call(self) -> None:
        for name in (
            "verify_stability_refresh.py",
            "verify_dev_refresh.py",
            "health_payload.py",
        ):
            text = (_SCRIPT_DIR / name).read_text()
            for line in text.splitlines():
                if "type: ignore[operator]" in line:
                    assert "sleep" not in line, f"{name}: {line.strip()}"

    def test_a_runaway_remote_detail_is_clipped_before_the_receipt(self) -> None:
        """MINOR: the ``/health`` body is network-adjacent and unbounded.

        It is copied verbatim into the receipt artifact and the refresh log
        exactly when the runtime serving it is the thing misbehaving.
        """
        body = {
            "status": "degraded",
            "details": {
                "healthy": True,
                "runtime_health": {
                    "status": "DEGRADED",
                    "age_seconds": 1.0,
                    "dimensions": [
                        {
                            "name": "projection_dlq_saturation",
                            "status": "DEGRADED",
                            "detail": "x" * 50_000,
                        }
                    ],
                },
            },
        }
        found = _health_payload.extract_non_healthy_dimensions(body)
        assert len(found) == 1
        assert len(found[0].detail) < 1_000
        assert "truncated" in found[0].detail
        assert "50000 chars" in found[0].detail

    def test_the_duplicated_verdict_block_key_cannot_drift(self) -> None:
        """MINOR: the scripts copy ``RUNTIME_HEALTH_DETAIL_KEY`` by necessity.

        They must run under a bare ``python3`` with no repo venv, so the
        constant is duplicated rather than imported. A rename on the source
        side would otherwise make ``extract_non_healthy_dimensions`` return
        ``()`` for every body, with no error anywhere -- silently deleting the
        evidence this ticket exists to preserve. This test is the guard the
        duplication had none of.
        """
        from omnibase_infra.runtime.health import container_healthcheck

        assert (
            _health_payload.RUNTIME_HEALTH_DETAIL_KEY
            == container_healthcheck.RUNTIME_HEALTH_DETAIL_KEY
        )
