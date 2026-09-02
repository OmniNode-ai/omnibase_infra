# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The lane refresh health-gate is healthy ONLY on ``status: healthy`` [OMN-17563].

Regression bar for the defect that let ``refresh_stability_lane.sh`` write
``"overall": "PASS"`` for the stability lane at 2026-09-02T14:58Z while the
lane's own ``onex-container-healthcheck --degraded-policy fail`` probe called
the same digest unhealthy ten minutes later.

Measured pre-fix behaviour of BOTH ``check_health`` copies -- every one of
these returned ``(True, ...)``:

    {"status": "degraded",  "details": {"healthy": true}}   -> healthy
    {"status": "unhealthy", "details": {"healthy": true}}   -> healthy
    {"details": {"healthy": true}}  (no status at all)      -> healthy

The `unhealthy` and the no-status rows are worse than the ticket described:
the old ``str(payload.get("status", "")).lower()`` reduced a missing status to
``""``, and the ``or details_healthy`` arm then carried the verdict on its own.

These tests pin the verdict for both verifiers from the same table, because
the defect was a copy in two files and a fix in one file only would let them
diverge again.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, TracebackType

import pytest

_SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from health_payload import (
    HEALTH_POLICY_STATUS_ONLY_STRICT,
    HealthPayload,
    HealthPayloadError,
    evaluate_health_body,
    parse_health_payload,
    unreachable_verdict,
)


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _SCRIPT_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_STABILITY = _load("verify_stability_refresh")
_DEV = _load("verify_dev_refresh")

#: Both verifiers, so neither can be fixed without the other.
_VERIFIERS = pytest.mark.parametrize(
    "verifier",
    [
        pytest.param(_STABILITY, id="stability"),
        pytest.param(_DEV, id="dev"),
    ],
)


class _FakeResponse:
    """Minimal ``urlopen`` stand-in: context manager with ``.read``/``.status``."""

    def __init__(self, body: bytes, status: int = 200) -> None:
        self._body = body
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        return False


def _opener(body: object, status: int = 200) -> object:
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()

    def _open(url: str, timeout: int = 10) -> _FakeResponse:
        return _FakeResponse(raw, status)

    return _open


# ─── the verdict table ───────────────────────────────────────────────────────
#
# `expected_ok` is the ONLY column that matters for the gate. Every False row
# below returned True before OMN-17563.

_HEALTHY_BODY = {"status": "healthy", "details": {"healthy": True}}

_VERDICT_TABLE = [
    pytest.param(
        {"status": "degraded", "details": {"healthy": True, "degraded": True}},
        False,
        id="degraded-with-nested-true",
    ),
    pytest.param(
        {"status": "unhealthy", "details": {"healthy": True}},
        False,
        id="unhealthy-with-nested-true",
    ),
    pytest.param({"details": {"healthy": True}}, False, id="missing-status"),
    pytest.param(
        {"status": "", "details": {"healthy": True}}, False, id="empty-status"
    ),
    pytest.param({"status": None}, False, id="null-status"),
    pytest.param({"status": True}, False, id="non-string-status"),
    pytest.param({"status": "HEALTHY"}, True, id="uppercase-status-healthy"),
    pytest.param({"status": "healthy"}, True, id="healthy-no-details"),
    pytest.param(_HEALTHY_BODY, True, id="healthy-with-nested-true"),
    pytest.param(
        {"status": "healthy", "details": {"healthy": False}},
        True,
        id="healthy-with-nested-false-status-still-wins",
    ),
    pytest.param({"status": "degraded"}, False, id="degraded-no-details"),
    pytest.param({"status": "starting"}, False, id="unrecognised-status"),
]


@pytest.mark.unit
@pytest.mark.parametrize(("body", "expected_ok"), _VERDICT_TABLE)
def test_evaluate_health_body_verdict_table(body: dict, expected_ok: bool) -> None:
    verdict = evaluate_health_body(json.dumps(body).encode())
    assert verdict.ok is expected_ok, verdict.detail
    assert verdict.policy == HEALTH_POLICY_STATUS_ONLY_STRICT


@pytest.mark.unit
@_VERIFIERS
@pytest.mark.parametrize(("body", "expected_ok"), _VERDICT_TABLE)
def test_check_health_verdict_table(
    verifier: ModuleType, body: dict, expected_ok: bool
) -> None:
    """The same table through each verifier's own ``check_health``."""
    verdict = verifier.check_health("http://x/health", opener=_opener(body))
    assert verdict.ok is expected_ok, verdict.detail
    assert verdict.policy == HEALTH_POLICY_STATUS_ONLY_STRICT


# ─── malformed / unreachable bodies fail closed ─────────────────────────────

_MALFORMED = [
    pytest.param(b"", id="empty-body"),
    pytest.param(b"not json at all", id="not-json"),
    pytest.param(b'{"status": "healthy"', id="truncated-json"),
    pytest.param(b"[]", id="json-array-not-object"),
    pytest.param(b'"healthy"', id="json-string-not-object"),
    pytest.param(b"null", id="json-null"),
    pytest.param(b"200", id="json-number"),
]


@pytest.mark.unit
@pytest.mark.parametrize("raw", _MALFORMED)
def test_malformed_body_fails_closed(raw: bytes) -> None:
    verdict = evaluate_health_body(raw)
    assert verdict.ok is False
    assert verdict.status is None
    assert verdict.detail


@pytest.mark.unit
@_VERIFIERS
@pytest.mark.parametrize("raw", _MALFORMED)
def test_check_health_malformed_body_fails_closed(
    verifier: ModuleType, raw: bytes
) -> None:
    verdict = verifier.check_health("http://x/health", opener=_opener(raw))
    assert verdict.ok is False
    assert verdict.status is None


@pytest.mark.unit
@_VERIFIERS
def test_check_health_non_200_fails_closed(verifier: ModuleType) -> None:
    """A healthy BODY behind a 503 is still a FAIL."""
    verdict = verifier.check_health(
        "http://x/health", opener=_opener(_HEALTHY_BODY, status=503)
    )
    assert verdict.ok is False
    assert "503" in verdict.detail


@pytest.mark.unit
@_VERIFIERS
def test_check_health_transport_failure_fails_closed(verifier: ModuleType) -> None:
    def _boom(url: str, timeout: int = 10) -> object:
        raise OSError("connection refused")

    verdict = verifier.check_health("http://x/health", opener=_boom)
    assert verdict.ok is False
    assert verdict.status is None
    assert "connection refused" in verdict.detail


# ─── parse-level contract ───────────────────────────────────────────────────


@pytest.mark.unit
def test_parse_rejects_missing_status() -> None:
    with pytest.raises(HealthPayloadError, match="no top-level 'status'"):
        parse_health_payload(json.dumps({"details": {"healthy": True}}))


@pytest.mark.unit
def test_parse_normalises_status_and_keeps_nested_flag_for_the_record() -> None:
    payload = parse_health_payload(
        json.dumps({"status": "  DeGrAdEd ", "details": {"healthy": True}})
    )
    assert payload == HealthPayload(status="degraded", details_healthy=True)
    assert payload.healthy is False
    # The disagreement is legible: both sources are rendered.
    assert payload.describe() == "status='degraded' details.healthy=true"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("body", "expected"),
    [
        pytest.param({"status": "healthy"}, None, id="details-absent"),
        pytest.param({"status": "healthy", "details": {}}, None, id="flag-absent"),
        pytest.param(
            {"status": "healthy", "details": "nope"}, None, id="details-not-a-dict"
        ),
        pytest.param(
            {"status": "healthy", "details": {"healthy": "yes"}},
            None,
            id="flag-not-a-bool",
        ),
        pytest.param(
            {"status": "healthy", "details": {"healthy": False}}, False, id="flag-false"
        ),
        pytest.param(
            {"status": "healthy", "details": {"healthy": True}}, True, id="flag-true"
        ),
    ],
)
def test_nested_flag_absent_is_not_flag_false(
    body: dict, expected: bool | None
) -> None:
    """``None`` (no flag) and ``False`` (flag says no) stay distinguishable."""
    assert parse_health_payload(json.dumps(body)).details_healthy is expected


@pytest.mark.unit
def test_unreachable_verdict_is_never_ok() -> None:
    verdict = unreachable_verdict("health endpoint returned HTTP 500")
    assert verdict.ok is False
    assert verdict.status is None
    assert verdict.policy == HEALTH_POLICY_STATUS_ONLY_STRICT


@pytest.mark.unit
def test_health_payload_is_frozen() -> None:
    payload = parse_health_payload(json.dumps(_HEALTHY_BODY))
    with pytest.raises(AttributeError):
        payload.status = "degraded"  # type: ignore[misc]


# ─── gate + receipt: a degraded body must sink `overall`, and the receipt ────
#     must name the policy that decided it (OMN-17563 AC-3)


def _all_green_runner(revision: str = "deadbeef1234") -> object:
    """`docker`/`rpk` stub where every non-health check passes."""

    def _run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        joined = " ".join(cmd)
        if "{{.Image}}" in joined:
            stdout = "sha256:new\n"
        elif _REVISION_LABEL_FMT in joined:
            stdout = f"{revision}\n"
        elif "cluster health" in joined:
            stdout = "Healthy:      true\n"
        elif "cluster config get" in joined:
            stdout = "1000\n"
        elif "topic list" in joined:
            stdout = "NAME    PARTITIONS   REPLICAS\ntopic-a  1            1\n"
        elif "group describe" in joined:
            stdout = "STATE        Stable\n"
        else:
            stdout = ""
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=stdout, stderr=""
        )

    return _run


_REVISION_LABEL_FMT = "org.opencontainers.image.revision"

_DEGRADED_BODY = {"status": "degraded", "details": {"healthy": True, "degraded": True}}


def _stability_gate(health_body: dict) -> object:
    def _open(url: str, timeout: int = 10) -> _FakeResponse:
        if "manifest" in url:
            return _FakeResponse(
                json.dumps({"contracts": [{} for _ in range(10_000)]}).encode()
            )
        return _FakeResponse(json.dumps(health_body).encode())

    return _STABILITY.run_health_gate(
        lane="stability-test",
        pre_image_ids=dict.fromkeys(_STABILITY.CORE_SERVICES, "sha256:old"),
        expected_revision="deadbeef1234",
        manifest_url="http://x/v1/introspection/manifest",
        health_url="http://x/health",
        broker_container="redpanda",
        min_contracts=1,
        consumer_groups=["g1"],
        runner=_all_green_runner(),
        opener=_open,
        sleep_fn=lambda _s: None,
    )


@pytest.mark.unit
def test_gate_overall_is_pass_when_status_healthy() -> None:
    """Control: with every other check green, a healthy body PASSes."""
    gate = _stability_gate({"status": "healthy", "details": {"healthy": True}})
    assert gate.errors == []
    assert gate.health_ok is True
    assert gate.overall == "PASS"


@pytest.mark.unit
def test_gate_overall_is_fail_on_degraded_body_with_nested_true() -> None:
    """The 2026-09-02T14:58Z false PASS. Same inputs, opposite verdict."""
    gate = _stability_gate(_DEGRADED_BODY)
    assert gate.health_ok is False
    assert gate.health_status == "degraded"
    assert gate.overall == "FAIL"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("body", "expected_status"),
    [
        pytest.param(_DEGRADED_BODY, "degraded", id="degraded"),
        pytest.param({"status": "healthy"}, "healthy", id="healthy"),
    ],
)
def test_receipt_records_the_policy_that_produced_health_ok(
    body: dict, expected_status: str
) -> None:
    """AC-3: a later reader can tell a strict PASS from a lenient one."""
    gate = _stability_gate(body)
    receipt = _STABILITY.build_receipt(
        lane="stability-test",
        prior_refs={"omnibase_infra": "old"},
        new_refs={"omnibase_infra": "deadbeef1234"},
        ancestry_ok=True,
        ancestry_commands=["git merge-base --is-ancestor old deadbeef1234"],
        build_scope=["omnibase_infra"],
        gate=gate,
        rollback_triggered=False,
        rollback_gate=None,
    )
    health_gate = receipt["health_gate"]
    assert isinstance(health_gate, dict)
    assert health_gate["health_policy"] == HEALTH_POLICY_STATUS_ONLY_STRICT
    assert health_gate["health_status"] == expected_status
    # The receipt is what a later prod-promotion session reads. Round-trip it.
    assert json.loads(json.dumps(receipt))["health_gate"]["health_policy"] == (
        HEALTH_POLICY_STATUS_ONLY_STRICT
    )


# ─── the sibling import must survive real script execution ──────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "script", ["verify_stability_refresh.py", "verify_dev_refresh.py"]
)
def test_script_runs_standalone_from_an_unrelated_cwd(
    script: str, tmp_path: Path
) -> None:
    """``health_payload`` is a sibling import, so prove the deploy path works.

    ``refresh_*_lane.sh`` runs these as ``<python> <abs path to script>`` from
    whatever cwd it happens to be in, sometimes under a bare ``python3`` with
    no repo venv. If the sibling import did not resolve there, the gate would
    die at import time on the .201 host and nowhere else.
    """
    result = subprocess.run(
        [sys.executable, str(_SCRIPT_DIR / script), "--help"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    assert "usage:" in result.stdout


# ─── parity with the lane's own container probe (OMN-17563 AC-1) ────────────
#
# AC-1 asks for the gate to consume "the same verdict the lane's container
# probe consumes". The stability lane's probe, read live off
# ``Healthcheck.Test``, is:
#
#     ["CMD", "python", "/usr/local/bin/onex-container-healthcheck",
#      "--degraded-policy", "fail"]
#
# -- i.e. ``evaluate_health_response(degraded_policy="fail")`` with
# ``require_verdict`` left at its default False. This test asserts the two
# agree body-for-body, so the gate cannot drift lenient again without a red
# test. It is a TEST-only import: the verifier scripts must stay importable
# under a bare ``python3`` with no repo venv, so they may not depend on
# ``omnibase_infra`` at runtime.

_LANE_PROBE_POLICY = "fail"


# Bodies where the gate is deliberately STRICTER than the probe. Measured, not
# assumed: `evaluate_health_response` reads `payload.get("status", "")`, fails
# only on the literal "unhealthy"/"degraded", and falls through every other
# value -- including an ABSENT status -- to `PASS "healthy"`. So the probe
# accepts all five of these. That is the same fail-open class as OMN-17563 on
# a different surface, filed as OMN-17623 rather than fixed here: that module is
# also the Docker LIVENESS probe watched by autoheal, so tightening it turns
# an unreadable body into a restart loop, which is a real design call and not
# a drive-by edit. AC-4 forbids loosening the probe; nothing here does.
_GATE_STRICTER_THAN_PROBE = frozenset(
    {
        "missing-status",
        "empty-status",
        "null-status",
        "non-string-status",
        "unrecognised-status",
    }
)


@pytest.mark.unit
@pytest.mark.parametrize(("body", "expected_ok"), _VERDICT_TABLE)
def test_gate_is_never_more_permissive_than_the_lane_container_probe(
    body: dict, expected_ok: bool, request: pytest.FixtureRequest
) -> None:
    """The property that matters: gate PASS implies probe PASS, never the reverse.

    OMN-17563 was exactly the reverse holding -- the gate said PASS where the
    probe said FAIL. This asserts that direction can never come back.
    """
    from omnibase_infra.runtime.health.container_healthcheck import (
        evaluate_health_response,
    )

    gate = evaluate_health_body(json.dumps(body).encode())
    probe = evaluate_health_response(
        http_status=200, payload=body, degraded_policy=_LANE_PROBE_POLICY
    )
    probe_ok = probe.verdict == "PASS"

    assert gate.ok is expected_ok
    if gate.ok:
        assert probe_ok, (
            f"gate PASSed a body the lane probe FAILs "
            f"({probe.reason}) -- this is the OMN-17563 defect: {body}"
        )

    case_id = request.node.callspec.id
    if case_id in _GATE_STRICTER_THAN_PROBE:
        assert gate.ok is False and probe_ok is True, (
            f"{case_id} is recorded as gate-stricter-than-probe; if the probe "
            f"was tightened, delete this row rather than loosening the gate"
        )
    else:
        assert gate.ok is probe_ok, (
            f"gate={gate.ok} probe={probe.verdict}/{probe.reason} for {body}"
        )


@pytest.mark.unit
def test_gate_agrees_with_the_probe_on_the_live_degraded_body() -> None:
    """The real 2026-09-02 stability body: status folded down by OMN-15217."""
    from omnibase_infra.runtime.health.container_healthcheck import (
        evaluate_health_response,
    )

    body = {
        "status": "degraded",
        "details": {
            "healthy": True,
            "degraded": True,
            "runtime_health": {
                "status": "DEGRADED",
                "age_seconds": 107.8,
                "nonwriting_projection_count": 15,
                "dimensions": [
                    {
                        "name": "projection_write_path",
                        "status": "DEGRADED",
                        "detail": "15 of 37 projections write nothing",
                    }
                ],
            },
        },
    }
    gate = evaluate_health_body(json.dumps(body).encode())
    probe = evaluate_health_response(
        http_status=200, payload=body, degraded_policy=_LANE_PROBE_POLICY
    )
    assert gate.ok is False
    assert probe.verdict == "FAIL"
    assert probe.reason == "runtime_degraded"


@pytest.mark.unit
def test_known_residual_gate_is_blind_to_an_absent_monitor_verdict() -> None:
    """Pinned, not hidden: the gate and the probe are BOTH blind here.

    ``ServiceRuntimeHealthMonitor`` publishes its first verdict roughly one
    ``RUNTIME_HEALTH_CHECK_INTERVAL`` (300s) after boot, and
    ``fold_runtime_verdict_into_status`` cannot degrade ``status`` before that
    verdict exists. A refresh gate that probes inside that window therefore
    reads a genuine ``status: healthy`` -- which is what the
    ``20260902T145841Z-810c818a10b3`` receipt actually recorded
    (``health_detail: "status=healthy"``), NOT a nested-flag override.

    Closing this needs the gate to require a fresh ``details.runtime_health``
    the way ``onex-container-healthcheck --require-verdict`` can; that is a
    behaviour change to refresh timing on both lanes and is tracked as
    OMN-17624. Asserted here so the gap is a recorded fact with a red test the
    day someone claims it is closed.
    """
    from omnibase_infra.runtime.health.container_healthcheck import (
        evaluate_health_response,
    )

    pre_verdict_body = {"status": "healthy", "details": {"healthy": True}}

    gate = evaluate_health_body(json.dumps(pre_verdict_body).encode())
    lane_probe = evaluate_health_response(
        http_status=200, payload=pre_verdict_body, degraded_policy=_LANE_PROBE_POLICY
    )
    strict_probe = evaluate_health_response(
        http_status=200,
        payload=pre_verdict_body,
        degraded_policy=_LANE_PROBE_POLICY,
        require_verdict=True,
    )

    # Gate and the lane's ACTUAL probe agree -- both accept it.
    assert gate.ok is True
    assert lane_probe.verdict == "PASS"
    # Only --require-verdict, which the lane does not pass, would reject it.
    assert strict_probe.verdict == "FAIL"
    assert strict_probe.reason == "verdict_absent"
