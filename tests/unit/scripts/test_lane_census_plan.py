# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pure-planner tests for the lane-census reconciler (OMN-13011).

The planner diffs the versioned desired-state lane manifest against a live docker
inventory and emits typed drift findings. These tests pin every drift kind plus
the headline red fixture: the 2026-06-11 prod outage (runtime containers absent +
broker network detached) MUST produce drift findings that would have ticketed
hours before a human noticed.
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_PLAN_PATH = _REPO / "scripts" / "lane_census_plan.py"
_MANIFEST_PATH = _REPO / "deploy" / "lane-census" / "lane-manifest.yaml"


def _load_planner() -> Any:
    spec = importlib.util.spec_from_file_location("lane_census_plan", _PLAN_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PLAN = _load_planner()
MANIFEST = PLAN.load_manifest(_MANIFEST_PATH)

# OMN-18320 — the `prod` lane fixture is FROZEN here instead of read from the live
# manifest.
#
# The lab compose lane named `prod` was shut down on 2026-09-13 under an operator
# consent row and removed from deploy/lane-census/lane-manifest.yaml, which broke
# every test below: they all used that lane as the planner's representative
# fixture. Repointing them at a lane that still exists would have silently
# rewritten what the red fixture below REPLAYS — the 2026-06-11 outage was a prod
# outage, and a stability-test-shaped replay of it is a different test wearing the
# same docstring.
#
# So the spec is pinned here verbatim, exactly as it stood in the manifest at
# origin/dev before removal. Every assertion in this module is unchanged. The
# planner is lane-agnostic, so this costs nothing and removes a coupling that
# should never have existed: a pure-planner unit test does not need a lane to be
# live, or even to be declared, in order to pin the planner's behaviour.
#
# This is the lab compose lane only. Production is the AWS `onex-prod` namespace.
_RETIRED_PROD_LANE: dict[str, Any] = {
    # OMN-19088 added the one field the manifest schema now requires; the lane ran
    # on the lab host until its removal. Nothing else in this spec moved.
    "hosts": ["lab-201"],
    "compose_file": "docker/docker-compose.prod.yml",
    "compose_project": "omnibase-infra-prod",
    "network": "omnibase-infra-prod-network",
    "image_tag_pattern": ".+",
    "services": [
        {"name": "omnibase-infra-prod-postgres", "kind": "service", "replicas": 1},
        {"name": "omnibase-infra-prod-redpanda", "kind": "service", "replicas": 1},
        {"name": "omnibase-infra-prod-redpanda-partition-cap", "kind": "oneshot"},
        {"name": "omnibase-infra-prod-valkey", "kind": "service", "replicas": 1},
        {"name": "omnibase-infra-prod-forward-migration", "kind": "oneshot"},
        {"name": "omnibase-infra-prod-migration-gate", "kind": "keepalive"},
        {"name": "omnibase-infra-prod-intelligence-migration", "kind": "oneshot"},
        {"name": "omninode-prod-runtime", "kind": "service", "replicas": 1},
        {"name": "omninode-prod-runtime-effects", "kind": "service", "replicas": 1},
        {"name": "omnimarket-prod-projection-api", "kind": "service", "replicas": 1},
        {"name": "omninode-prod-runtime-worker", "kind": "service", "replicas": 1},
        {"name": "omninode-prod-agent-actions-consumer", "kind": "profile_gated"},
        {"name": "omninode-prod-skill-lifecycle-consumer", "kind": "profile_gated"},
        {"name": "omnibase-prod-intelligence-api", "kind": "profile_gated"},
        {"name": "omninode-prod-contract-resolver", "kind": "profile_gated"},
    ],
}

MANIFEST["lanes"]["prod"] = copy.deepcopy(_RETIRED_PROD_LANE)

# OMN-19088: the planner resolves the host it runs on through the manifest's
# hosts registry. The lab host's hostname, as the census reports it there.
_LAB_HOST = "omninode-pc"


def _container(
    name: str,
    *,
    lane: str,
    state: str = "running",
    status: str = "Up 3 hours",
    image: str = "omninode-runtime:0.37.0",
) -> dict[str, str]:
    return {
        "Names": name,
        "State": state,
        "Status": status,
        "Image": image,
        "Labels": f"com.omninode.lane={lane},com.omninode.layer=runtime",
    }


def _healthy_prod_containers() -> list[dict[str, str]]:
    """All declared prod services running, migrations exited 0.

    OMN-16803: `kind: profile_gated` services are deliberately omitted. The lane's
    compose file disables them via a profile override, so ABSENT is their healthy
    state — modelling them as Running would make the "healthy lane" fixture assert
    the exact condition the census now flags as drift.
    """
    services = list(MANIFEST["lanes"]["prod"]["services"])
    rows: list[dict[str, str]] = []
    for svc in services:
        if svc.get("kind") == "profile_gated":
            continue
        if svc.get("kind") == "oneshot":
            rows.append(
                _container(
                    svc["name"],
                    lane="prod",
                    state="exited",
                    status="Exited (0) 2 hours ago",
                )
            )
        else:
            rows.append(_container(svc["name"], lane="prod"))
    return rows


def _kinds(findings: list[dict[str, str]]) -> set[str]:
    return {f["kind"] for f in findings}


def test_healthy_prod_lane_no_drift() -> None:
    """A fully-running prod lane with its network present yields zero drift."""
    envelope = {
        "lane": "prod",
        "host": _LAB_HOST,
        "containers": _healthy_prod_containers(),
        "networks": ["omnibase-infra-prod-network"],
        "runtime_tag": None,
    }
    plan = PLAN.build_plan(envelope, MANIFEST)
    assert plan["has_drift"] is False, plan["findings"]
    assert plan["lanes_checked"] == ["prod"]


def test_tonight_prod_red_fixture_runtime_absent_and_network_detached() -> None:
    """RED FIXTURE — the 2026-06-11 outage.

    prod runtime containers (main/effects/worker/projection-api) absent AND the
    broker network detached. The reconciler MUST emit a network_detached finding
    plus a container_absent finding for every missing runtime service — exactly
    the ticket that should have fired hours before a human noticed.
    """
    # Only the infra/migration layer is up; the four runtime services and the
    # network are gone.
    surviving = [
        _container("omnibase-infra-prod-postgres", lane="prod"),
        _container("omnibase-infra-prod-redpanda", lane="prod"),
        _container("omnibase-infra-prod-valkey", lane="prod"),
        _container(
            "omnibase-infra-prod-migration-gate",
            lane="prod",
            state="exited",
            status="Exited (0) 4 hours ago",
        ),
    ]
    envelope = {
        "lane": "prod",
        "host": _LAB_HOST,
        "containers": surviving,
        "networks": [],  # broker network detached
        "runtime_tag": None,
    }
    plan = PLAN.build_plan(envelope, MANIFEST)

    assert plan["has_drift"] is True
    kinds = _kinds(plan["findings"])
    assert "network_detached" in kinds, plan["findings"]
    assert "container_absent" in kinds, plan["findings"]

    absent = {
        f["container"] for f in plan["findings"] if f["kind"] == "container_absent"
    }
    # Every runtime service must be named as absent.
    for runtime_svc in (
        "omninode-prod-runtime",
        "omninode-prod-runtime-effects",
        "omnimarket-prod-projection-api",
        "omninode-prod-runtime-worker",
    ):
        assert runtime_svc in absent, f"{runtime_svc} not reported absent: {absent}"

    # network_detached + container_absent are all critical severity.
    crit = [f for f in plan["findings"] if f["severity"] == "critical"]
    assert crit, "tonight's outage must be critical severity"


def test_worker_replicas_zero_silent_drop_is_drift() -> None:
    """The WORKER_REPLICAS silent-zero regression (OMN-12988/12990).

    A worker scaled to 0 produces no container; the planner must report it as
    drift on the declared replicas:1 service.
    """
    containers = _healthy_prod_containers()
    containers = [c for c in containers if c["Names"] != "omninode-prod-runtime-worker"]
    envelope = {
        "lane": "prod",
        "host": _LAB_HOST,
        "containers": containers,
        "networks": ["omnibase-infra-prod-network"],
        "runtime_tag": None,
    }
    plan = PLAN.build_plan(envelope, MANIFEST)
    absent = {
        f["container"] for f in plan["findings"] if f["kind"] == "container_absent"
    }
    assert "omninode-prod-runtime-worker" in absent


def test_oneshot_failed_exit_nonzero_is_critical_drift() -> None:
    """A migration container that Exited non-zero is critical drift."""
    containers = _healthy_prod_containers()
    for c in containers:
        if c["Names"] == "omnibase-infra-prod-migration-gate":
            c["state"] = "exited"
            c["State"] = "exited"
            c["Status"] = "Exited (1) 5 minutes ago"
    envelope = {
        "lane": "prod",
        "host": _LAB_HOST,
        "containers": containers,
        "networks": ["omnibase-infra-prod-network"],
        "runtime_tag": None,
    }
    plan = PLAN.build_plan(envelope, MANIFEST)
    failed = [f for f in plan["findings"] if f["kind"] == "oneshot_failed"]
    assert failed and failed[0]["severity"] == "critical", plan["findings"]


def test_oneshot_stuck_running_is_warning_drift() -> None:
    """A migration container still Running (never completed) is warning drift."""
    containers = _healthy_prod_containers()
    for c in containers:
        if c["Names"] == "omnibase-infra-prod-forward-migration":
            c["State"] = "running"
            c["Status"] = "Up 6 hours"
    envelope = {
        "lane": "prod",
        "host": _LAB_HOST,
        "containers": containers,
        "networks": ["omnibase-infra-prod-network"],
        "runtime_tag": None,
    }
    plan = PLAN.build_plan(envelope, MANIFEST)
    stuck = [f for f in plan["findings"] if f["kind"] == "oneshot_stuck"]
    assert stuck and stuck[0]["severity"] == "warning"


def test_keepalive_migration_gate_running_is_not_drift() -> None:
    """OMN-13772 census false-positive regression.

    migration-gate is a long-running healthcheck sentinel BY DESIGN
    (`while true; do sleep 3600; done` + continuous healthcheck), yet the
    manifest classified it `oneshot`, so every healthy lane ticketed
    oneshot_stuck. Reclassified `keepalive`: Running is its healthy steady
    state and must produce ZERO findings.
    """
    containers = _healthy_prod_containers()
    gate = next(
        c for c in containers if c["Names"] == "omnibase-infra-prod-migration-gate"
    )
    assert gate["State"] == "running", "helper must model the keepalive as Running"
    envelope = {
        "lane": "prod",
        "host": _LAB_HOST,
        "containers": containers,
        "networks": ["omnibase-infra-prod-network"],
        "runtime_tag": None,
    }
    plan = PLAN.build_plan(envelope, MANIFEST)
    stuck = [f for f in plan["findings"] if f["kind"] == "oneshot_stuck"]
    assert not stuck, f"keepalive migration-gate flagged oneshot_stuck: {stuck}"
    assert plan["has_drift"] is False, plan["findings"]


def test_keepalive_exited_nonzero_is_still_critical_drift() -> None:
    """A keepalive that DIED non-zero is still oneshot_failed critical drift."""
    containers = _healthy_prod_containers()
    for c in containers:
        if c["Names"] == "omnibase-infra-prod-migration-gate":
            c["State"] = "exited"
            c["Status"] = "Exited (137) 5 minutes ago"
    envelope = {
        "lane": "prod",
        "host": _LAB_HOST,
        "containers": containers,
        "networks": ["omnibase-infra-prod-network"],
        "runtime_tag": None,
    }
    plan = PLAN.build_plan(envelope, MANIFEST)
    failed = [f for f in plan["findings"] if f["kind"] == "oneshot_failed"]
    assert failed and failed[0]["severity"] == "critical", plan["findings"]


def test_migration_gate_is_keepalive_in_every_compose_lane() -> None:
    """Manifest ratchet: migration-gate must stay `keepalive` (OMN-13772).

    Flipping it back to `oneshot` reintroduces the census false-positive on
    every healthy lane; flipping it to `service` would hard-require a
    container whose restart policy is `restart: "no"` on some lanes.
    """
    # OMN-18320: `prod` removed — this ratchet reads the LIVE manifest, and the lab
    # compose prod lane was retired 2026-09-13 and is no longer declared there. The
    # frozen fixture above is deliberately not consulted here: this test exists to
    # catch a regression in a lane someone can still deploy.
    for lane in ("stability-test", "judge"):
        gate = next(
            svc
            for svc in MANIFEST["lanes"][lane]["services"]
            if svc["name"].endswith("-migration-gate")
        )
        assert gate["kind"] == "keepalive", (
            f"lane {lane!r}: migration-gate kind is {gate['kind']!r}, expected "
            f"'keepalive' (long-running healthcheck sentinel, OMN-13772)"
        )


def test_unexpected_lane_labeled_container_is_drift() -> None:
    """A container labeled for the lane but not declared is unexpected_container."""
    containers = _healthy_prod_containers()
    containers.append(_container("omninode-prod-rogue-shadow", lane="prod"))
    envelope = {
        "lane": "prod",
        "host": _LAB_HOST,
        "containers": containers,
        "networks": ["omnibase-infra-prod-network"],
        "runtime_tag": None,
    }
    plan = PLAN.build_plan(envelope, MANIFEST)
    unexpected = {
        f["container"] for f in plan["findings"] if f["kind"] == "unexpected_container"
    }
    assert "omninode-prod-rogue-shadow" in unexpected


def test_image_tag_mismatch_is_drift() -> None:
    """A running container whose tag fails the lane pattern is drift."""
    # Pin a strict pattern lane manifest in-memory.
    # OMN-18320: this test re-reads the manifest from disk rather than using the
    # module-level MANIFEST, so it needs its own copy of the frozen retired-lane
    # spec — and a DEEP copy, because it mutates image_tag_pattern and would
    # otherwise corrupt the shared fixture for every test that runs after it.
    manifest = PLAN.load_manifest(_MANIFEST_PATH)
    manifest["lanes"]["prod"] = copy.deepcopy(_RETIRED_PROD_LANE)
    manifest["lanes"]["prod"]["image_tag_pattern"] = r"0\.37\..+"
    containers = _healthy_prod_containers()
    for c in containers:
        if c["Names"] == "omninode-prod-runtime":
            c["Image"] = "omninode-runtime:0.30.0-stale"
    envelope = {
        "lane": "prod",
        "host": _LAB_HOST,
        "containers": containers,
        "networks": ["omnibase-infra-prod-network"],
        "runtime_tag": None,
    }
    plan = PLAN.build_plan(envelope, manifest)
    mism = [f for f in plan["findings"] if f["kind"] == "image_tag_mismatch"]
    assert mism and mism[0]["container"] == "omninode-prod-runtime"


def test_optional_dev_lane_entirely_down_is_not_drift() -> None:
    """The optional dev lane being fully down must NOT ticket (developer lane)."""
    envelope = {
        "lane": "dev",
        "host": _LAB_HOST,
        "containers": [],
        "networks": [],
        "runtime_tag": None,
    }
    plan = PLAN.build_plan(envelope, MANIFEST)
    assert plan["has_drift"] is False, plan["findings"]


def test_optional_dev_lane_partially_up_is_drift() -> None:
    """A partially-up optional lane IS reconciled (one service up, one missing)."""
    envelope = {
        "lane": "dev",
        "host": _LAB_HOST,
        "containers": [
            _container("omninode-runtime", lane="dev"),
            # omninode-runtime-effects missing
        ],
        "networks": ["omnibase-infra_default"],
        "runtime_tag": None,
    }
    plan = PLAN.build_plan(envelope, MANIFEST)
    absent = {
        f["container"] for f in plan["findings"] if f["kind"] == "container_absent"
    }
    assert "omninode-runtime-effects" in absent


def test_unknown_lane_raises() -> None:
    with pytest.raises(ValueError):
        PLAN.build_plan(
            {
                "lane": "does-not-exist",
                "host": _LAB_HOST,
                "containers": [],
                "networks": [],
            },
            MANIFEST,
        )


def test_all_lanes_default_excludes_nothing_required() -> None:
    """With lane=None all manifest lanes are checked."""
    envelope = {
        "lane": None,
        "host": _LAB_HOST,
        "containers": [],
        "networks": [],
        "runtime_tag": None,
    }
    plan = PLAN.build_plan(envelope, MANIFEST)
    # OMN-19088: every lane is either checked on this host or reported
    # not-applicable because the manifest declares it for another host. None is
    # silently dropped.
    checked = set(plan["lanes_checked"])
    not_applicable = set(plan["lanes_not_applicable"])
    assert checked.isdisjoint(not_applicable)
    assert checked | not_applicable == set(MANIFEST["lanes"].keys())
    assert "prod" in checked


def test_tag_parsing_handles_registry_host_port() -> None:
    """`_tag_of` must not mistake a registry host:port for the tag."""
    assert PLAN._tag_of("registry.local:5000/omninode-runtime:0.37.0") == "0.37.0"
    assert PLAN._tag_of("omninode-runtime") == "latest"
    assert PLAN._tag_of("omninode-runtime@sha256:abc") == "latest"


def test_profile_gated_absent_is_not_drift() -> None:
    """OMN-16803 root cause — the four false criticals that hid a real outage.

    agent-actions-consumer / skill-lifecycle-consumer / intelligence-api /
    omninode-contract-resolver carry `profiles: !override ["<lane>-disabled"]`
    in the lane overlay, so they are not members of the lane's `runtime` profile
    and no sanctioned `up` can start them. Declared `kind: service` they produced
    four permanent container_absent criticals per lane, which is why a genuinely
    degraded stability lane read as standing noise for a month. Absent MUST be
    clean.
    """
    gated = [
        s["name"]
        for s in MANIFEST["lanes"]["prod"]["services"]
        if s.get("kind") == "profile_gated"
    ]
    assert gated, "prod manifest must declare the profile-disabled services"

    envelope = {
        "lane": "prod",
        "host": _LAB_HOST,
        "containers": _healthy_prod_containers(),
        "networks": ["omnibase-infra-prod-network"],
        "runtime_tag": None,
    }
    plan = PLAN.build_plan(envelope, MANIFEST)
    assert plan["has_drift"] is False, plan["findings"]
    absent = [f for f in plan["findings"] if f["container"] in gated]
    assert not absent, f"profile_gated services flagged while absent: {absent}"


def test_profile_gated_running_is_warning_drift() -> None:
    """The assertion runs the other way for `profile_gated`: PRESENT is the drift.

    A running container here means something started it outside the lane's active
    profile (e.g. a warm restart naming it explicitly on the CLI, which bypasses
    profile filtering) — the surprise worth surfacing.
    """
    gated = next(
        s["name"]
        for s in MANIFEST["lanes"]["prod"]["services"]
        if s.get("kind") == "profile_gated"
    )
    containers = _healthy_prod_containers()
    containers.append(_container(gated, lane="prod"))
    envelope = {
        "lane": "prod",
        "host": _LAB_HOST,
        "containers": containers,
        "networks": ["omnibase-infra-prod-network"],
        "runtime_tag": None,
    }
    plan = PLAN.build_plan(envelope, MANIFEST)
    present = [f for f in plan["findings"] if f["kind"] == "profile_gated_present"]
    assert present, f"running profile_gated container not flagged: {plan['findings']}"
    assert present[0]["container"] == gated
    assert present[0]["severity"] == "warning"


def test_profile_gated_running_is_not_reported_unexpected() -> None:
    """A declared profile_gated container is still DECLARED.

    It must surface as profile_gated_present (which says what is actually wrong),
    never as unexpected_container (which would say the manifest does not know
    about it at all).
    """
    gated = next(
        s["name"]
        for s in MANIFEST["lanes"]["prod"]["services"]
        if s.get("kind") == "profile_gated"
    )
    containers = _healthy_prod_containers()
    containers.append(_container(gated, lane="prod"))
    envelope = {
        "lane": "prod",
        "host": _LAB_HOST,
        "containers": containers,
        "networks": ["omnibase-infra-prod-network"],
        "runtime_tag": None,
    }
    plan = PLAN.build_plan(envelope, MANIFEST)
    unexpected = [f for f in plan["findings"] if f["kind"] == "unexpected_container"]
    assert not unexpected, f"declared profile_gated flagged unexpected: {unexpected}"
