# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the semantic container healthcheck (OMN-15217).

The load-bearing test in this module is
:class:`TestShallowCheckMask`, which reproduces the defect the ticket was filed
for: a ``/health`` response recorded from the stability lane on 2026-07-27 that
the old container healthcheck reads as healthy while the runtime's own monitor
reports DEGRADED. It asserts BOTH halves — that the old logic passes it (the
mask is real, not hypothetical) and that the new evaluator fails it (the mask is
closed).

Related Tickets:
    - OMN-15217: stability-lane runtime reports DEGRADED while Docker reads healthy
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from omnibase_infra.runtime.health import container_healthcheck as chc
from omnibase_infra.runtime.health.runtime_health_block import (
    RUNTIME_HEALTH_DETAIL_KEY,
)

# =============================================================================
# Recorded payloads
# =============================================================================


def _degraded_payload() -> dict[str, Any]:
    """A ``/health`` body shaped exactly like the observed stability-lane defect.

    Recorded 2026-07-27T12:58Z from ``omninode-stability-test-runtime`` (HTTP
    200, ``status: healthy``, ``degraded: false``), with the
    ``runtime_health`` block this ticket adds carrying the verdict the container
    was logging at the same moment:
    ``status=DEGRADED contracts=296 errors=4``.
    """
    return {
        "status": "degraded",
        "version": "0.38.4",
        "details": {
            "healthy": True,
            "degraded": True,
            "is_running": True,
            "runtime_attached": True,
            "no_handlers_registered": False,
            RUNTIME_HEALTH_DETAIL_KEY: {
                "status": "DEGRADED",
                "observed_at": "2026-07-27T12:58:21+00:00",
                "age_seconds": 42.0,
                "contract_count": 296,
                "discovery_error_count": 4,
                "consumer_group_count": 643,
                "empty_consumer_group_count": 0,
                "subscribe_topic_count": 310,
                "uncovered_topic_count": 0,
                "dimensions": [
                    {
                        "name": "discovery_errors",
                        "status": "DEGRADED",
                        "detail": "4 contract(s) failed to load: node_alpha, node_beta",
                    },
                    {
                        "name": "consumer_coverage",
                        "status": "HEALTHY",
                        "detail": "All 643 expected consumer group(s) covered",
                    },
                ],
            },
        },
    }


def _healthy_payload() -> dict[str, Any]:
    return {
        "status": "healthy",
        "version": "0.38.4",
        "details": {
            "healthy": True,
            "degraded": False,
            "is_running": True,
            "runtime_attached": True,
            RUNTIME_HEALTH_DETAIL_KEY: {
                "status": "HEALTHY",
                "observed_at": "2026-07-27T12:58:21+00:00",
                "age_seconds": 42.0,
                "contract_count": 296,
                "discovery_error_count": 0,
                "dimensions": [
                    {
                        "name": "discovery_errors",
                        "status": "HEALTHY",
                        "detail": "296 contracts loaded cleanly",
                    }
                ],
            },
        },
    }


def _old_shallow_check(http_status: int) -> int:
    """Exit code of the pre-OMN-15217 healthcheck: ``curl -sf <url>``.

    ``curl --fail`` asserts one property — the response code is < 400. The body
    is never read. This function is the whole of the old check's decision logic.
    """
    return 0 if http_status < 400 else 1


# =============================================================================
# The mask (RED against the old logic, GREEN with the new evaluator)
# =============================================================================


@pytest.mark.unit
class TestShallowCheckMask:
    """The defect OMN-15217 was filed for, pinned as a test."""

    def test_old_shallow_check_reports_a_degraded_runtime_as_healthy(self) -> None:
        """RED: `curl -sf /health` passes the recorded DEGRADED payload.

        This is the mask. The runtime is telling the truth in its own payload
        (``runtime_health.status == "DEGRADED"``, four contracts failing to
        load) and the check still exits 0, so `docker ps` prints `(healthy)` and
        any gate reading Docker health treats the lane as stability-proven.
        """
        payload = _degraded_payload()
        assert payload["details"][RUNTIME_HEALTH_DETAIL_KEY]["status"] == "DEGRADED"

        assert _old_shallow_check(http_status=200) == 0

    def test_semantic_check_fails_the_same_payload(self) -> None:
        """GREEN: the new evaluator reads the verdict and fails closed."""
        verdict = chc.evaluate_health_response(
            http_status=200, payload=_degraded_payload()
        )

        assert verdict.verdict == "FAIL"
        assert verdict.reason == "runtime_degraded"
        assert verdict.exit_code == chc.EXIT_UNHEALTHY
        # The failure detail names the failing dimension so `docker inspect`
        # output is actionable without going back to container logs.
        assert "discovery_errors" in verdict.detail
        assert "4 contract(s) failed to load" in verdict.detail

    def test_semantic_check_still_passes_a_genuinely_healthy_runtime(self) -> None:
        """The strict check must not turn every lane red."""
        verdict = chc.evaluate_health_response(
            http_status=200, payload=_healthy_payload()
        )

        assert verdict.verdict == "PASS"
        assert verdict.reason == "healthy"
        assert verdict.exit_code == chc.EXIT_HEALTHY


# =============================================================================
# Verdict table
# =============================================================================


@pytest.mark.unit
class TestVerdictTable:
    """Full decision table for evaluate_health_response()."""

    def test_unreachable_endpoint_fails(self) -> None:
        verdict = chc.evaluate_health_response(http_status=None, payload=None)
        assert (verdict.verdict, verdict.reason) == ("FAIL", "probe_unreachable")

    def test_http_error_fails(self) -> None:
        verdict = chc.evaluate_health_response(http_status=503, payload=None)
        assert (verdict.verdict, verdict.reason) == ("FAIL", "http_error")

    def test_missing_body_fails(self) -> None:
        verdict = chc.evaluate_health_response(http_status=200, payload=None)
        assert (verdict.verdict, verdict.reason) == ("FAIL", "payload_missing")

    def test_unhealthy_status_fails(self) -> None:
        verdict = chc.evaluate_health_response(
            http_status=200, payload={"status": "unhealthy", "details": {}}
        )
        assert (verdict.verdict, verdict.reason) == ("FAIL", "runtime_unhealthy")

    @pytest.mark.parametrize(
        ("body", "case"),
        [
            ({"details": {"healthy": True}}, "absent"),
            ({"status": "", "details": {"healthy": True}}, "empty"),
            ({"status": None}, "null"),
            ({"status": True}, "non-string"),
            ({"status": "starting"}, "unrecognised"),
        ],
    )
    def test_unreadable_status_fails_closed(self, body: dict, case: str) -> None:
        """OMN-17623: an unreadable status is unknown health, never proven health.

        ``ServiceHealth._handle_health`` types its status
        ``Literal["healthy", "degraded", "unhealthy"]`` and every branch assigns
        one of the three, so no ONEX runtime can serve any of these bodies. A
        body outside that closed set means the probe is not talking to a runtime
        health endpoint at all — which this module already fails closed on when
        the body is unparseable (``payload_missing``) or the endpoint is
        unreachable (``probe_unreachable``). Passing the strictly-less-broken
        case was the inconsistency.
        """
        verdict = chc.evaluate_health_response(http_status=200, payload=body)
        assert (verdict.verdict, verdict.reason) == ("FAIL", "status_unreadable"), case

    @pytest.mark.parametrize("status", ["healthy", "HEALTHY", "Healthy"])
    def test_recognised_status_is_case_insensitive(self, status: str) -> None:
        """The closed-set guard must not regress the existing case folding."""
        verdict = chc.evaluate_health_response(
            http_status=200, payload={"status": status}
        )
        assert (verdict.verdict, verdict.reason) == ("PASS", "healthy")

    def test_critical_verdict_fails_even_under_warn_policy(self) -> None:
        """CRITICAL is never downgraded — `warn` only softens DEGRADED."""
        payload = _degraded_payload()
        payload["details"][RUNTIME_HEALTH_DETAIL_KEY]["status"] = "CRITICAL"
        payload["details"][RUNTIME_HEALTH_DETAIL_KEY]["dimensions"][0]["status"] = (
            "CRITICAL"
        )

        verdict = chc.evaluate_health_response(
            http_status=200,
            payload=payload,
            degraded_policy=chc.DEGRADED_POLICY_WARN,
        )
        assert (verdict.verdict, verdict.reason) == ("FAIL", "runtime_critical")

    def test_degraded_passes_under_warn_policy_with_a_reason(self) -> None:
        """`warn` keeps the old pass/fail behavior but names the degradation."""
        verdict = chc.evaluate_health_response(
            http_status=200,
            payload=_degraded_payload(),
            degraded_policy=chc.DEGRADED_POLICY_WARN,
        )
        assert (verdict.verdict, verdict.reason) == ("PASS", "runtime_degraded_warn")
        assert verdict.exit_code == chc.EXIT_HEALTHY
        assert "discovery_errors" in verdict.detail

    def test_process_degraded_without_verdict_fails(self) -> None:
        """Handler-level degradation (no monitor verdict) still fails strictly."""
        verdict = chc.evaluate_health_response(
            http_status=200,
            payload={
                "status": "degraded",
                "details": {"degraded": True, RUNTIME_HEALTH_DETAIL_KEY: None},
            },
        )
        assert (verdict.verdict, verdict.reason) == ("FAIL", "process_degraded")

    def test_absent_verdict_passes_by_default(self) -> None:
        """A liveness probe must not restart a container over a missing verdict.

        The monitor's first cycle lands one check interval (default 300s) after
        boot, and non-Kafka profiles never start it at all. Absence is unknown,
        not unhealthy — so the container check passes.
        """
        verdict = chc.evaluate_health_response(
            http_status=200,
            payload={"status": "healthy", "details": {"healthy": True}},
        )
        assert (verdict.verdict, verdict.reason) == ("PASS", "healthy")

    def test_absent_verdict_fails_closed_for_proof_readers(self) -> None:
        """A proof reader cannot cite a lane whose health is merely unknown."""
        verdict = chc.evaluate_health_response(
            http_status=200,
            payload={"status": "healthy", "details": {"healthy": True}},
            require_verdict=True,
        )
        assert (verdict.verdict, verdict.reason) == ("FAIL", "verdict_absent")

    def test_stale_verdict_fails_closed_for_proof_readers(self) -> None:
        payload = _healthy_payload()
        payload["details"][RUNTIME_HEALTH_DETAIL_KEY]["age_seconds"] = 4000.0

        verdict = chc.evaluate_health_response(
            http_status=200,
            payload=payload,
            require_verdict=True,
            max_verdict_age_seconds=900.0,
        )
        assert (verdict.verdict, verdict.reason) == ("FAIL", "verdict_stale")
        assert "4000s" in verdict.detail

    def test_stale_verdict_does_not_fail_the_liveness_probe(self) -> None:
        """Without --require-verdict, staleness is not a restart trigger."""
        payload = _healthy_payload()
        payload["details"][RUNTIME_HEALTH_DETAIL_KEY]["age_seconds"] = 4000.0

        verdict = chc.evaluate_health_response(
            http_status=200, payload=payload, max_verdict_age_seconds=900.0
        )
        assert verdict.verdict == "PASS"

    def test_unparseable_age_is_not_read_as_fresh(self) -> None:
        payload = _healthy_payload()
        payload["details"][RUNTIME_HEALTH_DETAIL_KEY]["age_seconds"] = "not-a-number"

        verdict = chc.evaluate_health_response(
            http_status=200,
            payload=payload,
            require_verdict=True,
            max_verdict_age_seconds=900.0,
        )
        assert (verdict.verdict, verdict.reason) == ("FAIL", "verdict_stale")
        assert "unknown" in verdict.detail

    def test_non_mapping_verdict_block_is_treated_as_absent(self) -> None:
        verdict = chc.evaluate_health_response(
            http_status=200,
            payload={
                "status": "healthy",
                "details": {RUNTIME_HEALTH_DETAIL_KEY: "DEGRADED"},
            },
            require_verdict=True,
        )
        assert (verdict.verdict, verdict.reason) == ("FAIL", "verdict_absent")

    def test_non_mapping_details_does_not_raise(self) -> None:
        verdict = chc.evaluate_health_response(
            http_status=200, payload={"status": "healthy", "details": []}
        )
        assert verdict.verdict == "PASS"


# =============================================================================
# CLI
# =============================================================================


@pytest.mark.unit
class TestCli:
    """CLI wiring: exit codes and output shape."""

    def test_main_exits_nonzero_on_degraded(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(
            chc, "fetch_health", lambda url, timeout: (200, _degraded_payload())
        )
        exit_code = chc.main(["--url", "http://localhost:8085/health"])

        assert exit_code == chc.EXIT_UNHEALTHY
        assert "FAIL [runtime_degraded]" in capsys.readouterr().out

    def test_main_exits_zero_on_healthy(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(
            chc, "fetch_health", lambda url, timeout: (200, _healthy_payload())
        )
        exit_code = chc.main([])

        assert exit_code == chc.EXIT_HEALTHY
        assert "PASS [healthy]" in capsys.readouterr().out

    def test_json_output_is_machine_readable(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import json

        monkeypatch.setattr(
            chc, "fetch_health", lambda url, timeout: (200, _degraded_payload())
        )
        chc.main(["--json"])

        parsed = json.loads(capsys.readouterr().out)
        assert parsed["verdict"] == "FAIL"
        assert parsed["exit_code"] == 1

    def test_degraded_policy_defaults_to_fail(self) -> None:
        """A forgotten flag must default STRICT, never back to the mask."""
        args = chc.build_parser().parse_args([])
        assert args.degraded_policy == chc.DEGRADED_POLICY_FAIL
        assert args.require_verdict is False

    def test_defaults_probe_the_containers_own_endpoint(self) -> None:
        args = chc.build_parser().parse_args([])
        assert args.url == chc.DEFAULT_HEALTH_URL
        assert args.max_verdict_age_seconds is None

    def test_configuration_never_comes_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Config lives on the command line, not in ambient env.

        The check-env-reads gate keeps runtime configuration in the overlay, and
        a healthcheck's own configuration belongs where `docker inspect` shows
        it. An env var must not be able to silently relax the check.
        """
        monkeypatch.setenv("ONEX_CONTAINER_HEALTHCHECK_DEGRADED_POLICY", "warn")
        monkeypatch.setenv("ONEX_HTTP_PORT", "18085")

        args = chc.build_parser().parse_args([])

        assert args.degraded_policy == chc.DEGRADED_POLICY_FAIL
        assert args.url == chc.DEFAULT_HEALTH_URL


# =============================================================================
# Deployment seams
# =============================================================================


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


@pytest.mark.unit
class TestDeploymentSeams:
    """Pins that keep the check runnable where it is actually invoked."""

    def test_module_imports_only_the_standard_library(self) -> None:
        """Import cost is a health-probe budget, not a style preference.

        Measured in-container 2026-07-27 on the stability runtime image:
        importing the ``omnibase_infra`` package chain costs ~6.8s against a 10s
        healthcheck timeout; the stdlib-only module starts in ~0.12s. One
        ``from omnibase_infra...`` import in this module turns every probe into
        a flap risk, so the property is pinned rather than documented.
        """
        source = Path(chc.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)

        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.add(node.module.split(".")[0])

        non_stdlib = {
            name
            for name in imported
            if name not in sys.stdlib_module_names and name != "__future__"
        }
        assert non_stdlib == set(), (
            f"container_healthcheck must stay stdlib-only; found {sorted(non_stdlib)}"
        )

    def test_module_reads_no_environment(self) -> None:
        """Pinned so the check-env-reads gate cannot be reintroduced silently."""
        source = Path(chc.__file__).read_text(encoding="utf-8")
        assert "os.environ" not in source
        assert "os.getenv" not in source

    def test_detail_key_matches_the_publisher(self) -> None:
        """The duplicated literal must track runtime_health_block's constant."""
        assert chc.RUNTIME_HEALTH_DETAIL_KEY == RUNTIME_HEALTH_DETAIL_KEY

    def test_dockerfile_installs_the_check_at_the_invoked_path(self) -> None:
        """The Dockerfile COPY target and the compose command must agree."""
        dockerfile = (_repo_root() / "docker" / "Dockerfile.runtime").read_text(
            encoding="utf-8"
        )
        source_rel = "src/omnibase_infra/runtime/health/container_healthcheck.py"

        assert source_rel in dockerfile, (
            "Dockerfile.runtime must COPY the healthcheck module into the image"
        )
        assert "/usr/local/bin/onex-container-healthcheck" in dockerfile

    def test_stability_lane_runs_the_strict_check(self) -> None:
        """The proof lane must not fall back to the shallow check.

        OMN-15217: this lane's Docker health is read as stability-proof for prod
        promotion, so a regression to `curl -sf /health` here would silently
        restore the mask.
        """
        compose = (
            _repo_root() / "docker" / "docker-compose.stability-test.yml"
        ).read_text(encoding="utf-8")

        strict_blocks = re.findall(
            r"/usr/local/bin/onex-container-healthcheck\s*\n\s*-\s*--degraded-policy\s*\n\s*-\s*fail",
            compose,
        )
        assert len(strict_blocks) == 3, (
            "all three runtime containers (omninode-runtime, runtime-effects, "
            "runtime-worker) must run the strict check on the stability lane; "
            f"found {len(strict_blocks)}. Leaving one on `curl -sf` leaves the "
            "lane able to report healthy while that runtime is DEGRADED — "
            "verified live on runtime-worker 2026-07-27T14:18Z."
        )

    def test_stability_lane_runtime_containers_are_not_autohealed(self) -> None:
        """Strict health + autoheal would restart-loop a restart-immune defect."""
        compose = (
            _repo_root() / "docker" / "docker-compose.stability-test.yml"
        ).read_text(encoding="utf-8")

        override_count = len(re.findall(r"^\s*labels: !override$", compose, re.M))
        assert override_count == 3, (
            "omninode-runtime, runtime-effects and runtime-worker must !override "
            "labels on the stability lane — compose appends label sequences, so "
            f"the base service's autoheal=true survives a plain `labels:` block; "
            f"found {override_count} override block(s)"
        )

        # No label list in this overlay may re-arm autoheal (comments explaining
        # why it is off are fine; a `- "autoheal=..."` entry is not).
        armed = [
            line
            for line in compose.splitlines()
            if re.match(r"^\s*-\s*[\"']?autoheal=", line)
        ]
        assert armed == [], f"autoheal re-armed on the stability lane: {armed}"
