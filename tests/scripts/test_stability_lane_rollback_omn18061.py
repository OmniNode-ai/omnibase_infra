# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The stability lane decides a rollback by the same rule as the dev lane [OMN-18061].

``refresh_dev_lane.sh`` learned on 2026-09-08 (OMN-16729, ``f6c182cc``) not to
recreate a serving lane over a revision-label mismatch. ``refresh_stability_lane.sh``
did not: it kept "roll back on ANY health-gate FAIL", which is the trigger that
destroyed the dev lane at 18:48:41Z. The stability lane is the surface every
live prod grant's ``stability-proven`` premise resolves from, so the same
trigger there rolls a proof lane back onto the OOM-looping image.

Four defects, each with a REAL recorded gate output as its fixture:

1. Provenance-only FAIL recreates a healthy lane
   (``tests/scripts/fixtures/omn18061/dev-20260908T184841Z-gate.json``).
2. Two implementations of one rule -- the dev-lane block is inline bash, so
   fixing one lane cannot fix the other. Both now call ONE object.
3. A DEGRADED FIRST verdict is terminal on attempt 1, so the operator's
   "zero OOM kills over 10 min on the NEW image" exit criterion is unreachable
   through this script: the first verdict a fresh runtime can publish arrives
   at ~``check_interval`` (300s) and the rollback fires ~30s later. Measured
   twice -- 5m41s at attempt 2, 5m32s at attempt 3.
4. The rollback anchors four ``CORE_SERVICES`` while the refresh deploys ten
   ``REFRESH_BUILD_SERVICES``, so a rollback leaves the lane on MIXED revisions
   (attempt 3: four core on ``2ea74bc4``, six projection writers on ``915a1044``).

No test here starts a container, opens a socket, or sleeps. The receipts are
read from disk; the re-probe clock is injected.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_BUILD = REPO_ROOT / "scripts" / "runtime_build"
REFRESH_DEV = RUNTIME_BUILD / "refresh_dev_lane.sh"
REFRESH_STABILITY = RUNTIME_BUILD / "refresh_stability_lane.sh"
DECISION_MODULE = RUNTIME_BUILD / "lane_rollback_decision.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "omn18061"

sys.path.insert(0, str(RUNTIME_BUILD))

from health_payload import (
    HealthVerdict,
    derive_verdict_wait_bound,
    wait_for_verdict,
)
from lane_rollback_decision import (
    RESULT_FAILED_BUILD_PROVENANCE,
    RESULT_ROLLBACK_REQUIRED,
    decide_lane_rollback,
)

from tests.unit.docker.test_stability_sasl_omn23363 import _proof_lane_overlay

#: The dimensions the gates that wrote the 2026-09-08 receipts did not yet
#: emit. Supplied here EXPLICITLY rather than defaulted inside the loader: the
#: decision fails closed on an absent dimension (an unreported dimension is not
#: a proven one), so silently filling one in would be the test proving the
#: fixture rather than the rule. Every value below is what the post-fix gate
#: reports for the lane state the receipt itself describes.
_RECORDING_GATE_DID_NOT_EMIT: dict[str, dict[str, Any]] = {
    # 18:48:41Z: the lane was SERVING (health_ok=true, manifest_ok=true,
    # errors=[]) when this gate ran. The stranded ``State=created`` containers
    # are what the rollback that followed CREATED, not what it found.
    "dev-20260908T184841Z-gate.json": {
        "core_services_running": True,
        "core_services_not_running": [],
    },
    # Attempt 3, 00:03Z: health_ok=false from a DEGRADED verdict; every other
    # dimension green, containers up and serving /health.
    "stability-attempt3-20260908T235241Z-gate.json": {
        "core_services_running": True,
        "core_services_not_running": [],
    },
    # Attempt 2, 17:41Z: same shape, plus one recorded probe error.
    "stability-attempt2-20260908T172047Z-gate.json": {
        "core_services_running": True,
        "core_services_not_running": [],
    },
}


def load_gate(name: str) -> dict[str, Any]:
    """Load a REAL recorded health-gate block, with absent dimensions named."""
    gate: dict[str, Any] = json.loads((FIXTURES / name).read_text())
    for key, value in _RECORDING_GATE_DID_NOT_EMIT[name].items():
        assert gate.get(key) is None, (
            f"{name} already reports {key}; the reconstruction is stale and "
            "would now be masking the recorded value"
        )
        gate[key] = value
    return gate


# ---------------------------------------------------------------------------
# 1. The provenance-only shape -- the fixture is the receipt that caused this
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_18_48_41z_shape_is_provenance_only_not_a_rollback() -> None:
    """RED fixture (a): the recorded 18:48:41Z gate must NOT warrant a recreate.

    health_ok, manifest_ok and cluster_healthy are all true and errors is
    empty; the only failing dimensions are provenance. The stability lane's
    old rule produced ROLLBACK on exactly this input.
    """
    gate = load_gate("dev-20260908T184841Z-gate.json")
    assert gate["health_ok"] is True
    assert gate["revision_readback_ok"] is False
    assert gate["errors"] == []

    decision = decide_lane_rollback(gate, ancestry_ok=True, branch="warm")

    assert decision.result == RESULT_FAILED_BUILD_PROVENANCE
    assert decision.rollback_warranted is False
    assert decision.lane_is_healthy is True
    assert decision.unhealthy_dimensions == ()
    # Named, so the receipt says WHICH provenance dimensions failed.
    assert "revision_readback_ok=false" in decision.provenance_failures
    assert "digest_changed=false" in decision.provenance_failures
    assert decision.suppressed_reason is not None


@pytest.mark.unit
def test_red_control_the_pre_fix_rule_rolls_back_the_same_input() -> None:
    """RED CONTROL: the rule this replaces DOES destroy the 18:48:41Z lane.

    Without this control the test above proves only that some function returns
    a string; it does not show the old behaviour was different. The pre-fix
    stability rule was ``overall != PASS -> rollback``, nothing else.
    """
    gate = load_gate("dev-20260908T184841Z-gate.json")
    pre_fix_rollback = gate["overall"] != "PASS"
    assert pre_fix_rollback is True, (
        "the pre-fix rule must fire on this input, or this fixture is not the "
        "regression it is cited as"
    )
    assert (
        decide_lane_rollback(gate, ancestry_ok=True, branch="warm").rollback_warranted
        is False
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mutation", "expected_dimension"),
    [
        ({"health_ok": False}, "health_ok=false"),
        ({"manifest_ok": False}, "manifest_ok=false"),
        ({"cluster_healthy": False}, "cluster_healthy=false"),
        (
            {
                "core_services_running": False,
                "core_services_not_running": ["runtime-worker=created"],
            },
            "core_services_running=false[runtime-worker=created]",
        ),
        ({"errors": ["health fetch failed: HTTP Error 503"]}, "gate_errors=1"),
    ],
)
def test_negative_controls_a_health_failure_still_rolls_back(
    mutation: dict[str, Any], expected_dimension: str
) -> None:
    """NEGATIVE CONTROLS: every health dimension still warrants a recreate.

    The fix narrows what counts as a reason to destroy a lane. These prove it
    did not narrow to nothing -- an UNHEALTHY verdict, an unreadable probe, or
    a core service that is not running all still roll back.
    """
    gate = load_gate("dev-20260908T184841Z-gate.json")
    gate.update(mutation)

    decision = decide_lane_rollback(gate, ancestry_ok=True, branch="warm")

    assert decision.result == RESULT_ROLLBACK_REQUIRED
    assert decision.rollback_warranted is True
    assert decision.lane_is_healthy is False
    assert expected_dimension in decision.unhealthy_dimensions


@pytest.mark.unit
def test_an_unreported_health_dimension_fails_closed() -> None:
    """A dimension the gate did not report is not a proven one.

    The decision reads the gate's own JSON; a gate that omits
    ``core_services_running`` has not shown the containers are up, and a lane
    the gate cannot see is never treated as healthy.
    """
    gate = json.loads((FIXTURES / "dev-20260908T184841Z-gate.json").read_text())
    assert gate.get("core_services_running") is None

    decision = decide_lane_rollback(gate, ancestry_ok=True, branch="warm")

    assert decision.rollback_warranted is True
    assert any(
        d.startswith("core_services_running=false")
        for d in decision.unhealthy_dimensions
    )


# ---------------------------------------------------------------------------
# 2. ONE decision object -- identity, not similarity
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_both_lanes_call_the_same_decision_object() -> None:
    """Identity test: dev and stability invoke the SAME module, by path.

    Similarity is what produced this ticket -- the dev lane was fixed and the
    stability lane kept the bug for eight hours because each owned its own
    copy of the rule. Two scripts referencing one file is the property that
    makes the next fix reach both.
    """
    dev = REFRESH_DEV.read_text()
    stability = REFRESH_STABILITY.read_text()
    for name, text in (("dev", dev), ("stability", stability)):
        assert "lane_rollback_decision.py" in text, (
            f"{name} lane does not call the shared decision module"
        )
        assert "DECISION_SCRIPT" in text, f"{name} lane must resolve it by variable"


@pytest.mark.unit
def test_neither_lane_keeps_a_private_copy_of_the_rule() -> None:
    """No script re-derives the dimension split inline.

    A second copy of the rule is the defect, so its absence is the assertion.
    The shared module is the only place these dimension names may be composed
    into a rollback decision.
    """
    for script in (REFRESH_DEV, REFRESH_STABILITY):
        text = script.read_text()
        assert "UNHEALTHY_DIMENSIONS+=(" not in text, (
            f"{script.name} still builds the health-dimension list itself"
        )
        assert "PROVENANCE_FAILURES+=(" not in text, (
            f"{script.name} still builds the provenance list itself"
        )


@pytest.mark.unit
def test_decision_cli_is_pure_and_reports_the_recorded_shape(tmp_path: Path) -> None:
    """The CLI both scripts call returns the same verdict as the function.

    Drives the real module as a subprocess, the way the shell scripts do.
    """
    gate_path = tmp_path / "gate1.json"
    gate = load_gate("dev-20260908T184841Z-gate.json")
    gate_path.write_text(json.dumps(gate))

    result = subprocess.run(
        [
            sys.executable,
            str(DECISION_MODULE),
            "--gate-json",
            str(gate_path),
            "--ancestry-ok",
            "true",
            "--branch",
            "warm",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["result"] == RESULT_FAILED_BUILD_PROVENANCE
    assert payload["rollback_warranted"] is False
    assert (
        payload == decide_lane_rollback(gate, ancestry_ok=True, branch="warm").to_dict()
    )


@pytest.mark.unit
def test_decision_cli_fails_closed_on_an_unreadable_gate(tmp_path: Path) -> None:
    """An unreadable gate report is never a healthy lane."""
    result = subprocess.run(
        [
            sys.executable,
            str(DECISION_MODULE),
            "--gate-json",
            str(tmp_path / "does-not-exist.json"),
            "--ancestry-ok",
            "true",
            "--branch",
            "warm",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["rollback_warranted"] is True
    assert payload["lane_is_healthy"] is False


# ---------------------------------------------------------------------------
# 3. A DEGRADED FIRST verdict re-probes; only a persistent one is terminal
# ---------------------------------------------------------------------------


def _degraded(detail: str = "runtime health DEGRADED") -> HealthVerdict:
    """The verdict shape attempt 3 recorded: readable status, not healthy."""
    return HealthVerdict(
        ok=False,
        policy="verdict_required.v1",
        status="degraded",
        details_healthy=True,
        detail=detail,
        reason="runtime_degraded",
    )


def _healthy() -> HealthVerdict:
    return HealthVerdict(
        ok=True,
        policy="verdict_required.v1",
        status="healthy",
        details_healthy=True,
        detail="status='healthy'",
        reason=None,
    )


def _unreadable() -> HealthVerdict:
    return HealthVerdict(
        ok=False,
        policy="verdict_required.v1",
        status=None,
        details_healthy=None,
        detail="health fetch failed: HTTP Error 503: Service Unavailable",
        reason="status_unreadable",
    )


@pytest.mark.unit
def test_degraded_first_verdict_reprobes_instead_of_terminating() -> None:
    """RED fixture (b): attempt 3's DEGRADED-at-~300s verdict must re-probe.

    The monitor sleeps one ``check_interval`` BEFORE its first check, so the
    first verdict a fresh runtime can publish arrives at ~300s. Treating that
    verdict as terminal makes the operator's "zero OOM kills over 10 minutes on
    the NEW image" criterion unreachable through this script for any first
    verdict that is not already HEALTHY -- and attempt 3's own evidence is zero
    OOM kills for the whole 5m32s it was allowed to live.

    The clock is injected: this test sleeps for zero real seconds.
    """
    slept: list[float] = []
    probes = [_degraded(), _degraded(), _healthy()]
    bound = derive_verdict_wait_bound(
        check_interval_seconds=300.0, boot_grace_seconds=120.0
    )

    verdict, described = wait_for_verdict(
        lambda: probes.pop(0), bound=bound, sleep_fn=slept.append
    )

    assert verdict.ok is True, described
    assert "terminal" not in described
    assert slept, "a DEGRADED first verdict must have waited before re-probing"
    assert sum(slept) > 0


@pytest.mark.unit
def test_degraded_after_the_window_is_terminal() -> None:
    """A verdict STILL degraded after boot_grace + check_interval is terminal.

    The re-probe is a window, not a retry-until-green: a lane that stays
    DEGRADED for the whole window is a genuine finding and the last observed
    verdict is what the gate reports.
    """
    bound = derive_verdict_wait_bound(
        check_interval_seconds=300.0, boot_grace_seconds=120.0
    )
    calls = {"n": 0}

    def _probe() -> HealthVerdict:
        calls["n"] += 1
        return _degraded()

    verdict, described = wait_for_verdict(_probe, bound=bound, sleep_fn=lambda _s: None)

    assert verdict.ok is False
    assert verdict.status == "degraded"
    assert calls["n"] == bound.attempts, "the whole window must be used"
    # The description must name the verdict that settled. "exhausted without a
    # verdict" would read as a monitor that never published, which is a
    # different finding from one that published DEGRADED every time.
    assert "exhausted" in described
    assert "degraded" in described


@pytest.mark.unit
def test_the_window_covers_boot_grace_plus_check_interval() -> None:
    """The re-probe window reaches the first verdict a fresh runtime can emit.

    120s grace + 300s interval: the first check lands at 300s, strictly after
    the grace window, so the bound must not expire before it.
    """
    bound = derive_verdict_wait_bound(
        check_interval_seconds=300.0, boot_grace_seconds=120.0
    )
    assert bound.first_visible_verdict_seconds == 300.0
    assert bound.total_seconds >= 300.0


@pytest.mark.unit
def test_unreadable_health_is_still_terminal_on_the_first_probe() -> None:
    """NEGATIVE CONTROL: waiting is for a verdict that may yet appear.

    An unreadable body or a dead endpoint will not become readable by being
    asked again inside this window, and treating it as waitable would convert
    a hard failure into a five-minute stall.
    """
    bound = derive_verdict_wait_bound(
        check_interval_seconds=300.0, boot_grace_seconds=120.0
    )
    calls = {"n": 0}

    def _probe() -> HealthVerdict:
        calls["n"] += 1
        return _unreadable()

    verdict, described = wait_for_verdict(_probe, bound=bound, sleep_fn=lambda _s: None)

    assert verdict.ok is False
    assert calls["n"] == 1, "an unreadable probe must not be retried"
    assert "terminal" in described


@pytest.mark.unit
def test_attempt3_gate_still_rolls_back_once_the_verdict_is_terminal() -> None:
    """RED fixture (b), decision half: a settled DEGRADED IS a health failure.

    The re-probe changes WHEN the verdict is taken, never what it means. Once
    ``health_ok=false`` survives the window it is a lane-health dimension and
    the rollback is correct.
    """
    gate = load_gate("stability-attempt3-20260908T235241Z-gate.json")
    assert gate["health_ok"] is False
    assert gate["manifest_ok"] is True
    assert gate["cluster_healthy"] is True
    assert gate["revision_readback_ok"] is True
    assert gate["errors"] == []

    decision = decide_lane_rollback(gate, ancestry_ok=True, branch="warm")

    assert decision.result == RESULT_ROLLBACK_REQUIRED
    assert decision.unhealthy_dimensions == ("health_ok=false",)
    assert decision.provenance_failures == ()


@pytest.mark.unit
def test_attempt2_gate_rolls_back_because_the_gate_could_not_see_the_lane() -> None:
    """RED fixture (c): the recorded attempt-2 shape is a ROLLBACK, stated.

    Attempt 2 recorded ``overall=INFRA_ERROR`` with one probe error (a manifest
    fetch reset) alongside a DEGRADED verdict. The decision fails closed on a
    non-empty ``errors``: a gate that could not run its probes has not shown
    the lane is serving, and that asymmetry with the provenance side is
    deliberate.

    The non-destructive improvement for this shape is NOT in this rule -- it is
    OMN-16753's manifest retry, already landed in ``915a1044``, which routes a
    transient fetch reset to ``manifest_fetch_attempts`` instead of ``errors``.
    Attempt 3's receipt is the proof: five recorded fetch failures, and
    ``errors == []``.
    """
    gate = load_gate("stability-attempt2-20260908T172047Z-gate.json")
    assert gate["overall"] == "INFRA_ERROR"
    assert len(gate["errors"]) == 1

    decision = decide_lane_rollback(gate, ancestry_ok=True, branch="warm")

    assert decision.result == RESULT_ROLLBACK_REQUIRED
    assert "gate_errors=1" in decision.unhealthy_dimensions

    # POSITIVE CONTROL for the claim above: the same shape with the transient
    # routed where 915a1044 now routes it is no longer a destructive verdict
    # once the verdict has settled healthy.
    repaired = dict(gate)
    repaired["errors"] = []
    repaired["manifest_fetch_attempts"] = gate["errors"]
    repaired["health_ok"] = True
    assert (
        decide_lane_rollback(
            repaired, ancestry_ok=True, branch="warm"
        ).rollback_warranted
        is False
    )


# ---------------------------------------------------------------------------
# 4. The rollback set equals the deployed set
# ---------------------------------------------------------------------------


#: The service-set declarations, in the order the script declares them. Later
#: arrays expand earlier ones, so they are evaluated together.
_SERVICE_ARRAYS = ("CORE_SERVICES", "REFRESH_BUILD_SERVICES", "ROLLBACK_SERVICES")


def _bash_arrays(script: Path) -> dict[str, list[str]]:
    """Evaluate the real script's service-set declarations, in bash.

    Evaluated rather than pattern-matched: ``ROLLBACK_SERVICES`` expands
    ``REFRESH_BUILD_SERVICES``, which expands ``CORE_SERVICES``, and a regex
    that reads the literal text would report the expansion token instead of
    the ten services it stands for -- which is precisely the mistake the array
    exists to prevent.
    """
    # ``readonly`` is stripped so the three declarations can be evaluated in
    # one shell; the range ends at the first line closing the literal, which
    # is the same line for a single-line array and a bare ``)`` for a
    # multi-line one.
    extract = "; ".join(
        f"eval \"$(sed -n '/^readonly {name}=(/,/)$/p' \"$1\" | sed 's/^readonly //')\""
        for name in _SERVICE_ARRAYS
    )
    emit = "; ".join(
        f'printf "{name}\\t%s\\n" "${{{name}[*]}}"' for name in _SERVICE_ARRAYS
    )
    result = subprocess.run(
        ["bash", "-c", f"set -e; {extract}; {emit}", "_", str(script)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    arrays: dict[str, list[str]] = {}
    for line in result.stdout.splitlines():
        name, _, values = line.partition("\t")
        arrays[name] = values.split()
    assert set(arrays) == set(_SERVICE_ARRAYS), arrays
    return arrays


@pytest.mark.unit
def test_the_rollback_set_is_the_deployed_set() -> None:
    """The rollback must anchor every service the refresh deployed.

    Attempt 3 left the lane on MIXED revisions -- four core services rolled
    back to ``2ea74bc4`` while six projection writers stayed on ``915a1044``,
    because the preflight anchor and the recreate both used the four-service
    ``CORE_SERVICES`` while the build used the ten-service
    ``REFRESH_BUILD_SERVICES``. Second occurrence; attempt 2 recorded the same
    shape. A partial rollback is not a rollback: it produces a lane state that
    was never built, tested or proven.
    """
    text = REFRESH_STABILITY.read_text()
    assert "ROLLBACK_SERVICES" in text, (
        "the rollback set must be named, not implied by CORE_SERVICES"
    )
    arrays = _bash_arrays(REFRESH_STABILITY)
    rollback_set = set(arrays["ROLLBACK_SERVICES"])
    build_set = set(arrays["REFRESH_BUILD_SERVICES"])
    core_set = set(arrays["CORE_SERVICES"])

    # The build set is CORE_SERVICES plus this lane's own services; the
    # rollback set must equal the whole thing, not its core prefix.
    #
    # OMN-18114: this was pinned to a literal 10, which made a CORRECT widening
    # of the build scope fail a test about rollback coverage. The count was
    # never the property under test -- the three assertions below are -- and a
    # literal here means every service legitimately added to the lane arrives
    # with an unrelated red test and an invitation to "just bump the number".
    # OMN-21288: effects now run inside main and must not be rebuilt or restored
    # as a separate service. Keep the remaining build coverage.
    assert len(build_set) >= 10, sorted(build_set)
    assert "runtime-effects" not in core_set | build_set | rollback_set
    assert core_set < build_set, "CORE_SERVICES must be a strict subset of the build"
    # OMN-23363: the two SASL one-shots run a pinned upstream image, so there is no
    # <project>-<service>:latest image to anchor or retag, and the flip they assert
    # lives in the broker. They are the ONLY deployed services a rollback does not
    # touch. The build set is not edited to make room for them: the difference
    # between the deployed set and the rollback set must BE exactly those two names,
    # so a third service dropped from the rollback fails here, and so does one of the
    # two coming back without an image to anchor.
    sasl_one_shots = {"lane-redpanda-scram-user", "lane-redpanda-sasl-enable"}
    assert build_set - rollback_set == sasl_one_shots, (
        f"deployed services a rollback skips: {sorted(build_set - rollback_set)}"
    )
    assert rollback_set <= build_set, sorted(rollback_set - build_set)
    assert rollback_set > core_set, (
        "a rollback set equal to CORE_SERVICES is the mixed-revision defect"
    )


@pytest.mark.unit
def test_folded_lane_verifier_checks_only_the_remaining_core_services() -> None:
    from verify_stability_refresh import CORE_SERVICES

    assert set(CORE_SERVICES) == {
        "omninode-runtime",
        "runtime-worker",
        "projection-api",
    }


@pytest.mark.unit
@pytest.mark.parametrize("container_exists", [False, True])
@pytest.mark.parametrize("main_logged_fold", [False, True])
@pytest.mark.parametrize("census_readable", [False, True])
def test_folded_effects_verification_runs_on_pass_without_retiring(
    container_exists: bool,
    main_logged_fold: bool,
    census_readable: bool,
    tmp_path: Path,
) -> None:
    """A failed fold readback changes the receipt result without recreating."""
    text = REFRESH_STABILITY.read_text()
    start = text.index("    # OMN-21288: the deploy handler retires effects")
    verification = text[start : text.index("elif [[", start)]
    success = text.index('    RESULT="SUCCESS"')
    assert success < start < text.index("elif [[", success)
    (marker,) = (
        line
        for line in text.splitlines()
        if line.startswith("readonly FOLDED_ROLES_LOG_MARKER=")
    )
    script = f"""
set -euo pipefail
{marker}
CONTAINER_EXISTS={str(container_exists).lower()}
CENSUS_READABLE={str(census_readable).lower()}
MAIN_LOG={"'Auto-wiring folded roles: profile=main'" if main_logged_fold else "'booted'"}
COMPOSE_PROJECT=lane-a
RESULT=SUCCESS
ROLLBACK_SERVICES=(omninode-runtime)
DOCKER_CALLS='{tmp_path / "docker-calls"}'
declare -A CORE_CONTAINERS=([omninode-runtime]=lane-a-runtime)
err() {{ printf 'ERR: %s\\n' "$*"; }}
log() {{ printf '%s\\n' "$*"; }}
docker() {{
    printf '%s\\n' "$*" >> "$DOCKER_CALLS"
    case "$1" in
        ps) "$CENSUS_READABLE" || return 7
            if "$CONTAINER_EXISTS"; then printf 'c0ffee\\n'; fi ;;
        logs) printf '%s\\n' "$MAIN_LOG" ;;
        images) printf 'preflight-4\\npreflight-3\\npreflight-2\\npreflight-1\\n' ;;
        rmi) printf '%s\\n' "$*" >&2 ;;
        *) return 9 ;;
    esac
}}
"""
    result = subprocess.run(
        ["bash", "-c", script + verification + 'printf "RESULT=%s\\n" "$RESULT"'],
        capture_output=True,
        text=True,
        check=True,
    )
    assert ("did not attach its folded roles" in result.stdout) is not main_logged_fold
    assert ("runtime-effects containers still exist" in result.stdout) is (
        container_exists and census_readable
    )
    expected = (
        "SUCCESS"
        if main_logged_fold and not container_exists and census_readable
        else "FAILED_FOLDED_EFFECTS"
    )
    assert f"RESULT={expected}" in result.stdout
    calls = (tmp_path / "docker-calls").read_text()
    assert ("rmi lane-a-omninode-runtime:preflight-1" in calls) is (
        expected == "SUCCESS"
    )


@pytest.mark.unit
def test_the_preflight_anchor_covers_every_service_the_rollback_restores() -> None:
    """Every service the rollback retags must have had a preflight tag taken.

    A rollback that retags a service with no preflight anchor cannot restore
    it; a rollback that anchors fewer services than it recreates is the
    mixed-revision defect wearing the fix's name.
    """
    text = REFRESH_STABILITY.read_text()
    anchor_block = text[
        text.index("=== Tag preflight rollback anchor") : text.index(
            "=== Build + restart"
        )
    ]
    assert 'for svc in "${ROLLBACK_SERVICES[@]}"' in anchor_block, (
        "the preflight anchor must be taken over the full rollback set"
    )


@pytest.mark.unit
def test_a_rollback_without_a_full_anchor_set_refuses_rather_than_partially_rolls_back() -> (
    None
):
    """If any anchor is missing the script refuses, and names the mixed state.

    The ticket's alternative to rolling back the full set is to REFUSE and say
    so. Both are acceptable; silently restoring a subset is not, because the
    receipt then reads as a completed rollback over a lane nobody has ever run.
    """
    text = REFRESH_STABILITY.read_text()
    assert "FAILED_ROLLBACK_ANCHOR_INCOMPLETE" in text, (
        "there must be a named refusal result for a partial anchor set"
    )
    assert "missing_rollback_anchors" in text, (
        "the receipt must name which services could not be anchored"
    )


# ---------------------------------------------------------------------------
# 5. The DLQ-saturation finding names the consumer GROUP, not only the topic
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_dlq_saturation_detail_names_the_consumer_group() -> None:
    """The gate annotates the saturation dimension with the consumer group(s).

    Attempt 3's recorded detail named two TOPICS
    (``onex.evt.omniclaude.session-ended.v1``,
    ``onex.evt.omniclaude.session-started.v1``) and no group. The group is the
    fact that identifies the cause, and after the rollback destroyed the
    runtime it had to be recovered by hand from broker enumeration. The gate
    already enumerates every consumer group on the lane, so it can name them
    at the moment the dimension is recorded rather than leaving the next
    reader to re-derive them from a lane that no longer exists.
    """
    sys.path.insert(0, str(RUNTIME_BUILD))
    from verify_stability_refresh import annotate_dimension_consumer_groups

    dimension = {
        "name": "projection_dlq_saturation",
        "status": "DEGRADED",
        "detail": (
            "No projection is fully DLQ-routed over 8 flow window(s) (2 topic(s) "
            "carried flow no single declaring projection could be attributed, and "
            "are excluded from every ratio: onex.evt.omniclaude.session-ended.v1, "
            "onex.evt.omniclaude.session-started.v1)"
        ),
    }
    groups = [
        "stability-test.omnibase_infra.node_session_projection.consume.1.0.0."
        "__t.onex.evt.omniclaude.session-ended.v1",
        "stability-test.omnibase_infra.node_session_projection.consume.1.0.0."
        "__t.onex.evt.omniclaude.session-started.v1",
        "stability-test.omnibase_infra.node_unrelated.consume.1.0.0."
        "__t.onex.evt.platform.something-else.v1",
    ]

    annotated = annotate_dimension_consumer_groups(dimension, groups)

    assert annotated["consumer_groups"] == groups[:2], (
        "only the groups whose topic the detail actually names are attributed"
    )
    assert "consumer group" in str(annotated["detail"])
    assert "node_session_projection" in str(annotated["detail"])
    # NEGATIVE CONTROL: an unrelated group must not be swept in.
    assert "node_unrelated" not in str(annotated["detail"])


@pytest.mark.unit
def test_dlq_saturation_annotation_says_so_when_no_group_can_be_attributed() -> None:
    """A zero here is stated, never rendered as an absence.

    "No consumer group named" and "no consumer group exists" are different
    facts, and the second is the finding.
    """
    sys.path.insert(0, str(RUNTIME_BUILD))
    from verify_stability_refresh import annotate_dimension_consumer_groups

    dimension = {
        "name": "projection_dlq_saturation",
        "status": "DEGRADED",
        "detail": "…excluded from every ratio: onex.evt.omniclaude.session-ended.v1",
    }

    annotated = annotate_dimension_consumer_groups(dimension, [])

    assert annotated["consumer_groups"] == []
    assert "no consumer group" in str(annotated["detail"]).lower()


# ---------------------------------------------------------------------------
# 5. The broker's SASL state follows the ref the refresh deploys (OMN-24964)
# ---------------------------------------------------------------------------
#
# The script's own text is run, not mirrored: the read of the target ref's overlay and
# the deploy step are cut out of ``refresh_stability_lane.sh`` and run in bash against a
# REAL git repository holding one ref with the flip and one without, with the script's
# own ``git_clone``, the project name read from ``docker/runtime-policy.env`` the way the
# script reads it, and a stub ``docker`` that keeps the broker's ``enable_sasl`` in a
# file, so the state the broker is left in is what the tests assert.

LANE_OVERLAY = _proof_lane_overlay()
OVERLAY_PATH = LANE_OVERLAY.relative_to(REPO_ROOT).as_posix()
_FLIP_ONE_SHOT = "lane-redpanda-sasl-enable"
_AUTH_FILE = "/lane/broker-auth.env"


def _script_text() -> str:
    return REFRESH_STABILITY.read_text()


def _script_slice(start_prefix: str, end_prefix: str, *, include_end: bool) -> str:
    """The script's lines from the one starting ``start_prefix`` to ``end_prefix``."""
    lines = _script_text().splitlines()
    (first,) = [i for i, ln in enumerate(lines) if ln.startswith(start_prefix)]
    last = next(
        i for i in range(first + 1, len(lines)) if lines[i].startswith(end_prefix)
    )
    return "\n".join(lines[first : last + (1 if include_end else 0)])


def _git_clone_function() -> str:
    return _script_slice("git_clone() {", "}", include_end=True)


def _read_block() -> str:
    """The script's read of the target ref's overlay, ahead of the checkout."""
    return _script_slice(
        "PRE_SASL_REF=false", 'git_clone "${INFRA_CLONE}" checkout', include_end=False
    )


def _deploy_step() -> str:
    """From the build-and-restart log line through the deploy-failure guard's ``fi``."""
    lines = _script_text().splitlines()
    first = next(
        i for i, ln in enumerate(lines) if ln.startswith('log "=== Build + restart')
    )
    guard = next(
        i
        for i in range(first, len(lines))
        if lines[i].startswith('if [[ "${DEPLOY_EXIT}" -ne 0 ]]')
    )
    end = next(i for i in range(guard, len(lines)) if lines[i] == "fi")
    return "\n".join(lines[first : end + 1])


def _make_two_ref_repo(root: Path) -> Path:
    """A real clone with ``old`` (overlay without the flip) and ``new`` (with it)."""
    repo = root / "clone"
    (repo / "docker").mkdir(parents=True)

    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            env=scrub_git_location_env(os.environ),
        )

    git("init", "-q", "-b", "old")
    overlay = repo / OVERLAY_PATH
    overlay.write_text("services:\n  redpanda:\n    restart: no\n")
    git("add", OVERLAY_PATH)
    git("commit", "-q", "-m", "old")
    git("checkout", "-q", "-b", "new")
    overlay.write_text(
        f"services:\n  redpanda:\n    restart: no\n  {_FLIP_ONE_SHOT}:\n    restart: no\n"
    )
    git("commit", "-q", "-am", "new")
    return repo


_DOCKER_STUB = r"""
docker() {
    echo "$*" >> "$CALLS"
    [[ "$1" == run ]] || return 0
    local cmd="${@: -1}"
    if [[ "$cmd" == *"set enable_sasl"* ]]; then
        local want; want="$(sed -E 's/.*set enable_sasl ([a-z]+) .*/\1/' <<<"$cmd")"
        [[ "$want" != "${FAIL_SET:-none}" ]] || return 1
        echo "$want" > "$STATE"
    else
        cat "$STATE"
    fi
}
"""


def _run_refresh_steps(
    tmp_path: Path,
    *,
    ref: str,
    broker_before: str = "true",
    deploy_exit: int = 0,
    deploy_sets: str | None = None,
    fail_set: str | None = None,
    auth_file: str | None = _AUTH_FILE,
    sha: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[str], str]:
    """Run the script's ref read and deploy step; return (proc, docker calls, final enable_sasl)."""
    repo = _make_two_ref_repo(tmp_path)
    resolved = (
        sha
        or subprocess.run(
            ["git", "-C", str(repo), "rev-parse", ref],
            check=True,
            capture_output=True,
            text=True,
            env=scrub_git_location_env(os.environ),
        ).stdout.strip()
    )
    state, calls = tmp_path / "enable_sasl", tmp_path / "docker.calls"
    state.write_text(broker_before + "\n")
    deploy = tmp_path / "deploy-stub"
    deploy.write_text(
        "#!/bin/sh\n"
        f'echo "state=$(cat {state}) scope=$RUNTIME_BUILD_SERVICES_OVERRIDE" > {tmp_path}/deploy.seen\n'
        + (f"echo {deploy_sets} > {state}\n" if deploy_sets else "")
        + f"exit {deploy_exit}\n"
    )
    deploy.chmod(0o755)
    program = "\n".join(
        [
            "set -euo pipefail",
            f'source "{REPO_ROOT}/docker/runtime-policy.env"',
            'readonly COMPOSE_PROJECT="${STABILITY_TEST_COMPOSE_PROJECT}"',
            'log() { echo "$*"; }',
            'err() { echo "$*" >&2; }',
            _git_clone_function(),
            _DOCKER_STUB,
            f'CALLS="{calls}" STATE="{state}" FAIL_SET="{fail_set or "none"}"',
            f'INFRA_CLONE="{repo}" REF={ref} RESOLVED_REF_SHA={resolved}',
            f'LANE_COMPOSE_FILE="${{INFRA_CLONE}}/{OVERLAY_PATH}"',
            f'LANE_BROKER_AUTH_ENV_FILE="{auth_file}"' if auth_file else ":",
            "readonly CORE_SERVICES=(omninode-runtime)",
            'readonly REFRESH_BUILD_SERVICES=("${CORE_SERVICES[@]}" lane-redpanda-scram-user lane-redpanda-sasl-enable writer)',
            'readonly ROLLBACK_SERVICES=("${CORE_SERVICES[@]}" writer)',
            f'DEPLOY_RUNTIME="{deploy}" REPO_ROOT=/repo OMNI_HOME=/omni',
            "DEPLOY_SIBLING_REFS= RUNTIME_COMPOSE_WAIT_TIMEOUT_SECONDS=5 NEW_INFRA_SHA_SHORT=abc",
            "OMNIBASE_INFRA_COMPOSE_PROJECT=x",
            _read_block(),
            _deploy_step(),
            'echo "PRE_SASL_REF=${PRE_SASL_REF}"',
        ]
    )
    env = scrub_git_location_env(
        {k: v for k, v in os.environ.items() if k != "LANE_BROKER_AUTH_ENV_FILE"}
    )
    proc = subprocess.run(
        ["bash", "-c", program],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env=env,
    )
    recorded = calls.read_text().splitlines() if calls.exists() else []
    return proc, recorded, state.read_text().strip()


def _sets(calls: list[str]) -> list[str]:
    return [c for c in calls if "set enable_sasl" in c]


@pytest.mark.unit
def test_a_ref_with_the_flip_leaves_the_broker_alone(tmp_path: Path) -> None:
    proc, calls, final = _run_refresh_steps(tmp_path, ref="new", broker_before="true")
    assert proc.returncode == 0, proc.stderr
    assert "PRE_SASL_REF=false" in proc.stdout
    assert _sets(calls) == []
    assert final == "true"
    assert "lane-redpanda-sasl-enable" in (tmp_path / "deploy.seen").read_text()


@pytest.mark.unit
def test_unflip_a_ref_before_the_flip_un_flips_the_broker_just_before_the_deploy(
    tmp_path: Path,
) -> None:
    """The deploy sees ``enable_sasl`` false, and the one-shots are not in its scope.

    Red at 2604e98f6da8: the un-flip ran, but before the checkout and the build, so a
    failure there left the lane's SASL clients facing a PLAINTEXT broker.
    """
    proc, calls, final = _run_refresh_steps(tmp_path, ref="old", broker_before="true")
    assert proc.returncode == 0, proc.stderr
    assert "PRE_SASL_REF=true" in proc.stdout
    seen = (tmp_path / "deploy.seen").read_text()
    assert seen.startswith("state=false "), seen
    assert _FLIP_ONE_SHOT not in seen
    assert final == "false"
    (call,) = _sets(calls)
    assert f"--env-file {_AUTH_FILE}" in call
    assert "grep -qx false" in call, "the un-flip must read enable_sasl back as false"


@pytest.mark.unit
def test_unflip_the_network_is_the_one_the_overlay_names(tmp_path: Path) -> None:
    """``<project>-network`` is derived in the script; the overlay's network carries that name."""
    proc, calls, _ = _run_refresh_steps(tmp_path, ref="old", broker_before="true")
    assert proc.returncode == 0, proc.stderr
    named = re.search(
        r"^networks:\n  omnibase-infra-network:\n    name: (\S+)$",
        LANE_OVERLAY.read_text(),
        re.MULTILINE,
    )
    assert named is not None, "the overlay names no omnibase-infra-network"
    network = named.group(1)
    assert all(f"--network {network} " in call for call in _sets(calls)), (
        network,
        calls,
    )


@pytest.mark.unit
def test_unflip_a_git_grep_failure_is_not_a_pre_sasl_ref(tmp_path: Path) -> None:
    """A ref git cannot read is an error that stops the refresh, never a licence to un-flip.

    Red at 2604e98f6da8: any ``git grep`` failure read as "no flip in this ref" and turned
    auth off on a broker published on the LAN.
    """
    proc, calls, final = _run_refresh_steps(
        tmp_path, ref="old", broker_before="true", sha="0" * 40
    )
    assert proc.returncode == 65, (proc.stdout, proc.stderr)
    assert "SASL_REF_UNREADABLE" in proc.stderr
    assert calls == []
    assert final == "true"
    assert not (tmp_path / "deploy.seen").exists()


@pytest.mark.unit
def test_unflip_a_ref_before_the_flip_on_a_host_with_no_auth_file_touches_no_broker(
    tmp_path: Path,
) -> None:
    proc, calls, _ = _run_refresh_steps(tmp_path, ref="old", auth_file=None)
    assert proc.returncode == 0, proc.stderr
    assert calls == []


@pytest.mark.unit
def test_unflip_a_failed_un_flip_stops_the_refresh_before_anything_is_built(
    tmp_path: Path,
) -> None:
    proc, _, final = _run_refresh_steps(
        tmp_path, ref="old", broker_before="true", fail_set="false"
    )
    assert proc.returncode == 65, (proc.stdout, proc.stderr)
    assert "SASL_UNFLIP_FAILED" in proc.stderr
    assert not (tmp_path / "deploy.seen").exists()
    assert final == "true"


@pytest.mark.unit
def test_reflip_a_deploy_that_fails_after_the_un_flip_puts_the_flip_back(
    tmp_path: Path,
) -> None:
    """Red at 2604e98f6da8: nothing re-flipped, so the SASL clients still running met a
    PLAINTEXT broker for as long as it took someone to notice."""
    proc, calls, final = _run_refresh_steps(
        tmp_path, ref="old", broker_before="true", deploy_exit=3
    )
    assert proc.returncode == 3, (proc.stdout, proc.stderr)
    assert [c.split("set enable_sasl ")[1].split()[0] for c in _sets(calls)] == [
        "false",
        "true",
    ]
    assert final == "true"


@pytest.mark.unit
def test_reflip_a_broker_that_was_never_flipped_is_not_flipped_by_a_failed_deploy(
    tmp_path: Path,
) -> None:
    proc, calls, final = _run_refresh_steps(
        tmp_path, ref="old", broker_before="false", deploy_exit=3
    )
    assert proc.returncode == 3, (proc.stdout, proc.stderr)
    assert final == "false"
    assert all("set enable_sasl true" not in c for c in calls)


@pytest.mark.unit
def test_reflip_a_flip_that_fails_after_enabling_sasl_leaves_the_broker_as_found(
    tmp_path: Path,
) -> None:
    """The flip one-shot sets ``enable_sasl`` true and then fails a later check (an
    unauthenticated client still reaches a listener): the PLAINTEXT clients that never
    got recreated must find the broker un-flipped again."""
    proc, _, final = _run_refresh_steps(
        tmp_path, ref="new", broker_before="false", deploy_exit=1, deploy_sets="true"
    )
    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    assert final == "false"


def _rollback_branch() -> str:
    """The script's rollback branch, from the trigger to the targeted recreate."""
    return _script_slice(
        "    ROLLBACK_TRIGGERED=true",
        '        "${ROLLBACK_SERVICES[@]}" || ROLLBACK_RECREATE_EXIT=$?',
        include_end=True,
    )


def _run_rollback(
    tmp_path: Path,
    *,
    broker_before: str,
    pre_sasl_ref: bool,
    broker_now: str,
    fail_set: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], str, str]:
    state, calls = tmp_path / "enable_sasl", tmp_path / "docker.calls"
    state.write_text(broker_now + "\n")
    recreate = tmp_path / "recreate.argv"
    program = "\n".join(
        [
            "set -euo pipefail",
            'log() { echo "$*"; }',
            'err() { echo "$*" >&2; }',
            _DOCKER_STUB,
            f'CALLS="{calls}" STATE="{state}" FAIL_SET="{fail_set or "none"}"',
            "COMPOSE_PROJECT=lane UTC_NOW=t RUNTIME_COMPOSE_WAIT_TIMEOUT_SECONDS=5 ROLLBACK_TRIGGERED=false",
            f'LANE_BROKER_AUTH_ENV_FILE="{_AUTH_FILE}"',
            f"BROKER_SASL_BEFORE={broker_before} PRE_SASL_REF={'true' if pre_sasl_ref else 'false'}",
            "ROLLBACK_SERVICES=(omninode-runtime writer) LANE_COMPOSE_FILE_ARGS=(-f overlay.yml)",
            "# the script's rollback recreate goes through compose_up_bounded; record its argv",
            f'compose_up_bounded() {{ shift; echo "auth=${{LANE_BROKER_AUTH_ENV_FILE-unset}} $*" > "{recreate}"; }}',
            _script_slice(
                "broker_sasl() {", "restore_broker_sasl() {", include_end=False
            ),
            _rollback_branch(),
        ]
    )
    proc = subprocess.run(
        ["bash", "-c", program], capture_output=True, text=True, timeout=60, check=False
    )
    return (
        proc,
        state.read_text().strip(),
        recreate.read_text() if recreate.exists() else "",
    )


@pytest.mark.unit
def test_rollback_after_a_failed_first_sasl_refresh_returns_the_lane_to_plaintext(
    tmp_path: Path,
) -> None:
    """The lane was PLAINTEXT, the refresh flipped it, the health gate failed.

    The rolled-back clients are recreated from the SASL overlay the clone now holds, so
    without this they would come back on SASL settings the lane never had before the
    refresh. The broker is un-flipped and the recreate runs with no auth file.
    """
    proc, final, argv = _run_rollback(
        tmp_path, broker_before="false", pre_sasl_ref=False, broker_now="true"
    )
    assert proc.returncode == 0, proc.stderr
    assert final == "false"
    assert argv.startswith("auth=/dev/null docker compose"), argv
    assert "omninode-runtime writer" in argv


@pytest.mark.unit
def test_rollback_of_a_lane_that_was_already_on_sasl_keeps_the_flip_and_the_auth_file(
    tmp_path: Path,
) -> None:
    proc, final, argv = _run_rollback(
        tmp_path, broker_before="true", pre_sasl_ref=False, broker_now="true"
    )
    assert proc.returncode == 0, proc.stderr
    assert final == "true"
    assert argv.startswith(f"auth={_AUTH_FILE} docker compose"), argv


@pytest.mark.unit
def test_rollback_that_cannot_un_flip_keeps_the_auth_file_so_clients_match_the_broker(
    tmp_path: Path,
) -> None:
    proc, final, argv = _run_rollback(
        tmp_path,
        broker_before="false",
        pre_sasl_ref=False,
        broker_now="true",
        fail_set="false",
    )
    assert proc.returncode == 0, proc.stderr
    assert "SASL_UNFLIP_FAILED" in proc.stderr
    assert final == "true"
    assert argv.startswith(f"auth={_AUTH_FILE} docker compose"), argv
