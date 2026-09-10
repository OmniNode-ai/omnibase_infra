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
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

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
                "core_services_not_running": ["runtime-effects=created"],
            },
            "core_services_running=false[runtime-effects=created]",
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
    # Shrink-only instead: 10 is the floor this ticket found, and losing a
    # service from the build scope is the regression worth catching.
    assert len(build_set) >= 10, sorted(build_set)
    assert core_set < build_set, "CORE_SERVICES must be a strict subset of the build"
    assert rollback_set == build_set, (
        f"rollback set {sorted(rollback_set)} != deployed set {sorted(build_set)}"
    )
    assert rollback_set > core_set, (
        "a rollback set equal to CORE_SERVICES is the mixed-revision defect"
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
