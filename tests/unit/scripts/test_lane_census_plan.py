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
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.helpers.lane_census_fixtures import NEUTRAL_SNAPSHOT

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_PLAN_PATH = _REPO / "scripts" / "lane_census_plan.py"


def _load_planner() -> Any:
    spec = importlib.util.spec_from_file_location("lane_census_plan", _PLAN_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PLAN = _load_planner()

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

# OMN-19088: the planner resolves the host it runs on through the manifest's
# hosts registry. The lab host's hostname, as the census reports it there.
_LAB_HOST = "omninode-pc"

_LAB_SYNC_FIXTURES = _REPO / "tests" / "fixtures" / "lab_sync"


def _lane_from_capture(capture: Path) -> tuple[str, dict[str, Any]]:
    """A lane declaration read off a recorded inventory, kind by observed state.

    A container the capture saw Exited is a run-to-completion ``oneshot``, a
    migration gate is a ``keepalive`` sentinel, and every other one a required
    ``service``. Network and compose project are the capture's own.
    """
    envelope = json.loads(capture.read_text(encoding="utf-8"))["envelope"]
    lane = str(envelope["lane"])
    rows = envelope["containers"]
    services: list[dict[str, Any]] = []
    for row in rows:
        if row["Names"].endswith("-migration-gate"):
            services.append({"name": row["Names"], "kind": "keepalive"})
        elif row["State"] == "exited":
            services.append({"name": row["Names"], "kind": "oneshot"})
        else:
            services.append({"name": row["Names"], "kind": "service", "replicas": 1})
    return lane, {
        "hosts": list(_RETIRED_PROD_LANE["hosts"]),
        "compose_project": rows[0]["Labels"]["com.docker.compose.project"],
        "network": next(n for n in envelope["networks"] if lane in n),
        "image_tag_pattern": ".+",
        "services": services,
    }


# OMN-20934 -- the manifest is FROZEN here instead of read from disk.
#
# The lab's lane manifest is private deployment config and this repository no
# longer ships it. These are pure-planner tests: they pin the planner's
# behaviour over a declaration, and the 2026-06-11 replay and the lab-sync
# captures below need only the lanes they were recorded on. So the manifest is
# built here, the way the retired `prod` lane already was: the `dev` lane's
# declaration as it stood in the manifest at the move, verbatim, and the
# second captured lane read off its own capture. Which lanes the lab declares
# today is a fact about the lab, checked where its manifest lives.
_DEV_LANE: dict[str, Any] = {
    "optional": True,
    "hosts": list(_RETIRED_PROD_LANE["hosts"]),
    "compose_file": "docker/docker-compose.generated.yml",
    "compose_project": "omnibase-infra",
    "network": "omnibase-infra-network",
    "image_tag_pattern": ".+",
    "services": [
        {
            "name": "omnibase-infra-postgres",
            "kind": "service",
            "replicas": 1,
            "restart_bound": 5,
        },
        {
            "name": "omnibase-infra-redpanda",
            "kind": "service",
            "replicas": 1,
            "restart_bound": 5,
        },
        {
            "name": "omnibase-infra-valkey",
            "kind": "service",
            "replicas": 1,
            "restart_bound": 5,
        },
        {
            "name": "omnimarket-projection-api",
            "kind": "service",
            "replicas": 1,
            "restart_bound": 5,
        },
        {
            "name": "omninode-runtime",
            "kind": "service",
            "replicas": 1,
            "restart_bound": 5,
        },
        {
            "name": "omninode-runtime-effects",
            "kind": "service",
            "replicas": 1,
            "restart_bound": 5,
        },
        {
            "name": "omnimarket-projection-tenant-registry-writer",
            "kind": "service",
            "replicas": 1,
            "restart_bound": 5,
        },
        {
            "name": "omnimarket-projection-delegation-writer",
            "kind": "service",
            "replicas": 1,
            "restart_bound": 5,
        },
        {
            "name": "omnimarket-projection-registration-writer",
            "kind": "service",
            "replicas": 1,
            "restart_bound": 5,
        },
        {
            "name": "omnimarket-projection-savings-writer",
            "kind": "service",
            "replicas": 1,
            "restart_bound": 2,
        },
        {
            "name": "omnimarket-projection-tenant-credentials-writer",
            "kind": "service",
            "replicas": 1,
            "restart_bound": 5,
        },
        {
            "name": "omnimarket-projection-live-events-writer",
            "kind": "service",
            "replicas": 1,
            "restart_bound": 5,
        },
        {
            "name": "omnimarket-tenant-projection-writer",
            "kind": "service",
            "replicas": 1,
        },
        {"name": "omnibase-infra-forward-migration", "kind": "oneshot", "replicas": 1},
        {"name": "omnibase-infra-cloud-migration", "kind": "oneshot", "replicas": 1},
        {
            "name": "omnibase-infra-cloud-migration-files",
            "kind": "oneshot",
            "replicas": 1,
        },
        {"name": "onex-api", "kind": "service", "replicas": 1},
        {
            "name": "omninode-infra-routing-decisions-consumer",
            "kind": "service",
            "replicas": 1,
        },
    ],
}

_CAPTURED_LANE, _CAPTURED_LANE_SPEC = _lane_from_capture(
    _LAB_SYNC_FIXTURES / "inventory_container_unhealthy.json"
)

MANIFEST: dict[str, Any] = {
    "schema_version": "1.0.0",
    "default_image_tag_pattern": ".+",
    "hosts": {_RETIRED_PROD_LANE["hosts"][0]: {"aliases": [_LAB_HOST]}},
    "lanes": {
        "prod": copy.deepcopy(_RETIRED_PROD_LANE),
        "dev": copy.deepcopy(_DEV_LANE),
        _CAPTURED_LANE: _CAPTURED_LANE_SPEC,
    },
}
PLAN.validate_hosts(MANIFEST)
PLAN.validate_service_bounds(MANIFEST)


def _container(
    name: str,
    *,
    lane: str,
    state: str = "running",
    status: str = "Up 3 hours",
    image: str = "omninode-runtime:0.37.0",
) -> dict[str, Any]:
    return {
        "Names": name,
        "State": state,
        "Status": status,
        "Image": image,
        "RestartCount": 0,
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
    manifest = copy.deepcopy(MANIFEST)
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
    envelope: dict[str, Any] = {
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
    envelope: dict[str, Any] = {
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


# ---------------------------------------------------------------------------
# OMN-19411 — lab release sync, wave 0 task T0.2 (seam L0.2: collector <->
# planner <-> receipt check and lab alarm).
#
# The eleven lab-sync kinds are DECLARED in the planner's kind table (T0.2) and
# evaluated against the generated desired state (T1.2, OMN-19414). Each has one
# known-bad inventory fixture under
# tests/fixtures/lab_sync/, captured read-only from .201 on 2026-09-24 (or
# reconstructed from a recorded incident row, or synthetic, as each fixture's
# `capture` and `provenance` say), plus one clean fixture.
#
# Fixture contract (lab-sync-inventory-fixture.v1):
#   source_row        E1-E9 of the lab release sync plan's section 1, or synthetic
#   capture           live | reconstructed | synthetic
#   expected_kind     the one kind the fixture must produce (null for clean)
#   expected_findings every (kind, lane, container) it must produce
#   desired           an excerpt of the lab-desired-state.v1 document (OMN-19410)
#                     naming ONLY the fields this fixture exercises; a field it
#                     omits is not declared and is not compared. null means the
#                     document could not be read.
#   envelope          the planner's stdin envelope, with the collector fields
#                     each kind needs (LAB_SYNC_FINDING_KINDS envelope_fields)
#
# `desired` is cut down to the fields the fixture exercises; the tests adapt it to
# the lab-desired-state.v1 shapes the generator emits (`_desired_v1`).
# ---------------------------------------------------------------------------

_EVENT_PATH = _REPO / "scripts" / "lane_census_event.py"
_FIXTURES = _REPO / "tests" / "fixtures" / "lab_sync"

#: Section 4 B of the lab release sync plan, in its order. Restated here rather
#: than read from the planner so the test is a second, independent copy.
_PLAN_LAB_SYNC_KINDS = (
    "revision_mismatch",
    "config_hash_mismatch",
    "package_version_mismatch",
    "container_unhealthy",
    "container_restart_loop",
    "undeclared_container",
    "broker_config_mismatch",
    "runner_count_mismatch",
    "runner_workdir_mismatch",
    "runner_offline",
    "desired_state_unreadable",
)

#: The drift event's key set and value types as origin/dev built it at
#: schema_version 1.0.0 (lane_census_event.build_event before OMN-19411),
#: frozen here as the 1.0.0 consumer's view.
_V1_0_0_EVENT_FIELDS: dict[str, tuple[type, ...]] = {
    "schema_version": (str,),
    "event_type": (str,),
    "topic": (str,),
    "host": (str,),
    "host_id": (str, type(None)),
    "emitted_at": (str,),
    "severity": (str,),
    "lanes_checked": (list,),
    "lanes_not_applicable": (list,),
    "lanes_skipped_optional_down": (list,),
    "drift_count": (int,),
    "findings": (list,),
    "alert_key": (str,),
    "ticket_title": (str,),
    "ticket_body": (str,),
}


def _load_event_module() -> Any:
    spec = importlib.util.spec_from_file_location("lane_census_event", _EVENT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lab_sync_fixtures() -> list[Path]:
    return sorted(_FIXTURES.glob("inventory_*.json"))


def _fixture(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _bad_fixture_params() -> list[Any]:
    return [
        pytest.param(path, id=path.stem.removeprefix("inventory_"))
        for path in _lab_sync_fixtures()
        if path.stem != "inventory_clean"
    ]


def _frozen_manifest() -> dict[str, Any]:
    """The test manifest with the services the historical captures lack removed.

    These captures contain runtime containers only; freeze that topology while
    still reading their existing services' current bounds.
    """
    manifest = copy.deepcopy(MANIFEST)
    manifest["lanes"]["dev"]["services"] = [
        svc
        for svc in manifest["lanes"]["dev"]["services"]
        if svc["name"]
        not in {
            "omnibase-infra-postgres",
            "omnibase-infra-redpanda",
            "omnibase-infra-valkey",
            "omnimarket-projection-api",
        }
    ]
    return manifest


_HOST = "host-a"
_EXCERPT_PROJECTS = {"dev": "omnibase-infra"}


def _desired_v1(fixture: dict[str, Any]) -> Any:
    """A fixture's `desired` excerpt cut into the lab-desired-state.v1 shapes.

    The excerpts T0.2 captured name containers, brokers and runners as maps. The
    generator (OMN-19410, OMN-19413) emits arrays of named entries, a broker
    profile by compose service and runner pools, and that is what the planner
    reads, so the fixtures are adapted here and never compared in their own
    shape. A container entry that declares only a restart bound is dropped: the
    bound lives in the lane manifest (OMN-19416), not in the desired state.
    Already-v1 input is returned as it is.
    """
    desired = fixture["desired"]
    if desired is None or isinstance(desired, list):
        return desired
    if isinstance(desired.get("surface"), dict):
        return desired
    envelope = fixture["envelope"]
    out: dict[str, Any] = {}
    if "runners" in desired:
        runners = desired["runners"]
        host = runners["host"]
        out["surface"] = {
            "id": f"{host}/runners",
            "kind": "runner_fleet",
            "host": host,
            "compose_project": "docker",
            "mode": "reconcile",
        }
        pool: dict[str, Any] = {"name": "fleet", "workdir": runners["workdir"]}
        if "expected_count" in runners:
            pool["expected_count"] = runners["expected_count"]
        entries = {
            name: dict(fields) for name, fields in runners.get("containers", {}).items()
        }
        entries.update({name: {} for name in runners.get("names", [])})
        out["runners"] = {
            "pools": [pool],
            "containers": [
                {"name": name, "pool": "fleet", **fields}
                for name, fields in sorted(entries.items())
            ],
        }
        return out
    if "allowed_undeclared" in desired:
        out["surface"] = {
            "id": f"{_HOST}/host",
            "kind": "host",
            "host": _HOST,
            "compose_project": None,
            "mode": "reconcile",
        }
        out["allowed_undeclared"] = [
            {
                "name": f"admit-{entry['compose_project']}",
                "match": {"compose_project": entry["compose_project"]},
                "owner": entry["owner"],
                "reason": entry["reason"],
            }
            for entry in desired["allowed_undeclared"]
        ]
        return out
    out["surface"] = {
        "id": f"{_HOST}/{desired['surface']}",
        "kind": "compose_lane",
        "host": _HOST,
        "mode": "reconcile",
    }
    project = _EXCERPT_PROJECTS.get(desired["surface"])
    if project:
        out["surface"]["compose_project"] = project
    containers = []
    for name, fields in sorted(desired.get("containers", {}).items()):
        fields = {k: v for k, v in fields.items() if k != "restart_bound"}
        if fields:
            containers.append({"name": name, **fields})
    out["containers"] = containers
    if "brokers" in desired:
        ((container, config),) = desired["brokers"].items()
        service = next(
            r["Labels"]["com.docker.compose.service"]
            for r in envelope["containers"]
            if r["Names"] == container
        )
        out["broker"] = {
            "service": service,
            "cluster_config": {k: str(v) for k, v in sorted(config.items())},
        }
    return out


def _fixture_v1(path: Path) -> dict[str, Any]:
    fixture = _fixture(path)
    fixture["desired"] = _desired_v1(fixture)
    return fixture


def _plan_with_desired(fixture: dict[str, Any]) -> dict[str, Any]:
    envelope = dict(fixture["envelope"])
    envelope["desired_state"] = _desired_v1(fixture)
    plan: dict[str, Any] = PLAN.build_plan(envelope, _frozen_manifest())
    return plan


def test_lab_sync_kind_table_declares_the_eleven_plan_kinds() -> None:
    """The planner's table is the plan's section 4 B list, in order, once each."""
    declared = tuple(spec.kind for spec in PLAN.LAB_SYNC_FINDING_KINDS)
    assert declared == _PLAN_LAB_SYNC_KINDS
    for spec in PLAN.LAB_SYNC_FINDING_KINDS:
        assert spec.severity in {"critical", "warning"}, spec
        assert spec.subject in {"container", "broker", "runner", "desired_state"}, spec
        assert spec.envelope_fields, f"{spec.kind} names no collector field"


def test_lab_sync_kinds_are_new_and_every_kind_has_one_severity() -> None:
    legacy = [spec.kind for spec in PLAN.CENSUS_FINDING_KINDS]
    lab_sync = [spec.kind for spec in PLAN.LAB_SYNC_FINDING_KINDS]
    assert not set(legacy) & set(lab_sync)
    assert len(set(legacy + lab_sync)) == len(legacy + lab_sync)
    assert set(PLAN.FINDING_KIND_SEVERITY) == set(legacy + lab_sync)


def test_every_lab_sync_kind_is_produced_by_a_fixture() -> None:
    """Each declared kind is evaluated: its own fixture produces it."""
    produced = set()
    for path in _lab_sync_fixtures():
        produced |= {f["kind"] for f in _plan_with_desired(_fixture(path))["findings"]}
    assert set(_PLAN_LAB_SYNC_KINDS) <= produced, set(_PLAN_LAB_SYNC_KINDS) - produced


def test_finding_severity_comes_from_the_kind_table() -> None:
    finding = PLAN._finding("dev", "container_absent", "omninode-runtime", "x")
    assert finding["severity"] == "critical"
    with pytest.raises(KeyError):
        PLAN._finding("dev", "not_a_declared_kind", "omninode-runtime", "x")


def test_lab_sync_fixture_count_is_kind_table_plus_one() -> None:
    """AC1: one fixture per lab-sync kind, plus one clean fixture."""
    fixtures = _lab_sync_fixtures()
    assert len(fixtures) == len(PLAN.LAB_SYNC_FINDING_KINDS) + 1, [
        p.name for p in fixtures
    ]
    kinds = [_fixture(p)["expected_kind"] for p in fixtures]
    assert kinds.count(None) == 1
    assert sorted(k for k in kinds if k) == sorted(_PLAN_LAB_SYNC_KINDS)
    for path in fixtures:
        kind = _fixture(path)["expected_kind"]
        assert path.name == f"inventory_{kind or 'clean'}.json"


@pytest.mark.parametrize(
    "path", [pytest.param(p, id=p.stem) for p in _lab_sync_fixtures()]
)
def test_lab_sync_fixture_names_its_source_row(path: Path) -> None:
    """AC1: every fixture says where it came from."""
    fixture = _fixture(path)
    assert fixture["fixture_schema"] == "lab-sync-inventory-fixture.v1"
    assert re.fullmatch(r"E[1-9]|synthetic", fixture["source_row"]), fixture
    assert fixture["capture"] in {"live", "reconstructed", "synthetic"}
    assert (fixture["capture"] == "synthetic") == (fixture["source_row"] == "synthetic")
    assert fixture["provenance"].strip()
    for finding in fixture["expected_findings"]:
        assert finding["kind"] == fixture["expected_kind"]


@pytest.mark.parametrize(
    "path", [p for p in _lab_sync_fixtures() if p.stem != "inventory_clean"]
)
def test_lab_sync_fixture_carries_its_kinds_collector_fields(path: Path) -> None:
    """The collector side of the seam: each fixture has what its kind grades."""
    fixture = _fixture(path)
    spec = next(
        s for s in PLAN.LAB_SYNC_FINDING_KINDS if s.kind == fixture["expected_kind"]
    )
    envelope = fixture["envelope"]
    for field in spec.envelope_fields:
        if field == "desired_state":
            assert fixture["desired"] is None
            continue
        on_rows = any(field in row for row in envelope["containers"])
        assert field in envelope or on_rows, f"{spec.kind} needs {field!r}"
    if spec.kind != "desired_state_unreadable":
        assert isinstance(fixture["desired"], dict)


@pytest.mark.parametrize(
    "path", [pytest.param(p, id=p.stem) for p in _lab_sync_fixtures()]
)
def test_lab_sync_fixture_raises_no_census_kind_today(path: Path) -> None:
    """Each fixture is clean on every kind that exists today.

    So when T1.2 evaluates it, the finding it produces is its own kind and
    nothing else.
    """
    plan = _plan_with_desired(_fixture(path))
    assert not [
        f
        for f in plan["findings"]
        if f["kind"] in {spec.kind for spec in PLAN.CENSUS_FINDING_KINDS}
    ], plan["findings"]


@pytest.mark.parametrize("path", _bad_fixture_params())
def test_lab_sync_fixture_produces_its_named_kind(path: Path) -> None:
    """AC1: every known-bad fixture produces its named finding and no other lab-sync kind."""
    fixture = _fixture(path)
    plan = _plan_with_desired(fixture)
    got = {(f["kind"], f["container"]) for f in plan["findings"]}
    expected = {(f["kind"], f["container"]) for f in fixture["expected_findings"]}
    assert expected <= got, (fixture["expected_kind"], plan["findings"])
    other = {k for k, _ in got if k in _PLAN_LAB_SYNC_KINDS} - {
        fixture["expected_kind"]
    }
    assert not other, other


def test_lab_sync_clean_fixture_produces_no_finding() -> None:
    plan = _plan_with_desired(_fixture(_FIXTURES / "inventory_clean.json"))
    assert plan["has_drift"] is False, plan["findings"]


@pytest.mark.parametrize(
    ("seconds", "expected"), [(1799, False), (1800, True), (1801, True)]
)
def test_unhealthy_continuous_threshold(seconds: int, expected: bool) -> None:
    fixture = _fixture(_FIXTURES / "inventory_container_unhealthy.json")
    row = next(
        r
        for r in fixture["envelope"]["containers"]
        if r["Names"] == "omninode-stability-test-runtime"
    )
    row["Health"] = {"Status": "unhealthy", "FailingStreak": seconds}
    row["HealthcheckIntervalSeconds"] = 1
    findings = _plan_with_desired(fixture)["findings"]
    assert any(f["kind"] == "container_unhealthy" for f in findings) is expected


@pytest.mark.parametrize("status", ["healthy", "starting"])
def test_unhealthy_streak_does_not_grade_other_health_states(status: str) -> None:
    fixture = _fixture(_FIXTURES / "inventory_container_unhealthy.json")
    for row in fixture["envelope"]["containers"]:
        if row["Names"] == "omninode-stability-test-runtime":
            row["Health"]["Status"] = status
    assert not _plan_with_desired(fixture)["findings"]


@pytest.mark.parametrize("bound", [1801, 0, -1, True, "1800"])
def test_unhealthy_invalid_manifest_bound_refused_at_load(
    tmp_path: Path, bound: Any
) -> None:
    import yaml

    manifest = copy.deepcopy(MANIFEST)
    manifest["lanes"]["stability-test"]["services"][0]["unhealthy_after_seconds"] = (
        bound
    )
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest))
    with pytest.raises(ValueError, match="unhealthy_after_seconds"):
        PLAN.load_manifest(path)


def test_unhealthy_lower_service_bound() -> None:
    fixture = _fixture(_FIXTURES / "inventory_container_unhealthy.json")
    manifest = copy.deepcopy(MANIFEST)
    for svc in manifest["lanes"]["stability-test"]["services"]:
        if svc["name"] == "omninode-stability-test-runtime":
            svc["unhealthy_after_seconds"] = 60
    for row in fixture["envelope"]["containers"]:
        if row["Names"] == "omninode-stability-test-runtime":
            row["Health"]["FailingStreak"] = 2
    plan = PLAN.build_plan(fixture["envelope"], manifest)
    assert any(f["kind"] == "container_unhealthy" for f in plan["findings"])


@pytest.mark.parametrize(("restarts", "expected"), [(2, False), (3, True)])
def test_restart_loop_uses_manifest_bound(restarts: int, expected: bool) -> None:
    fixture = _fixture(_FIXTURES / "inventory_clean.json")
    for row in fixture["envelope"]["containers"]:
        if row["Names"] == "omnimarket-projection-savings-writer":
            row["RestartCount"] = restarts
    plan = _plan_with_desired(fixture)
    assert (
        any(f["kind"] == "container_restart_loop" for f in plan["findings"]) is expected
    )


# OMN-17427: RestartCount is cumulative over the container's life, so a past
# loop reads as a present one until the container is recreated. The dev broker
# census of 2026-10-20T18:49:18Z: 201 restarts, up and healthy since 17:45:04Z.
_BROKER_OBSERVED_AT = "2026-10-20T18:49:18.941827+00:00"
_BROKER_STARTED_AT = "2026-10-20T17:45:04.123456789Z"


def _restart_loop_plan(
    *,
    restarts: int,
    started_at: str | None,
    observed_at: str | None = _BROKER_OBSERVED_AT,
    state: str = "running",
) -> list[dict[str, str]]:
    fixture = _fixture(_FIXTURES / "inventory_clean.json")
    if observed_at is not None:
        fixture["envelope"]["observed_at"] = observed_at
    for row in fixture["envelope"]["containers"]:
        if row["Names"] == "omnimarket-projection-savings-writer":
            row["RestartCount"] = restarts
            row["State"] = state
            if state != "running":
                row["Status"] = f"{state.capitalize()} (1) 5 seconds ago"
            if started_at is not None:
                row["StartedAt"] = started_at
    return [
        f
        for f in _plan_with_desired(fixture)["findings"]
        if f["kind"] == "container_restart_loop"
    ]


def test_restart_loop_past_loop_up_since_its_last_start_is_not_a_finding() -> None:
    assert _restart_loop_plan(restarts=201, started_at=_BROKER_STARTED_AT) == []


def test_restart_loop_restarted_within_the_window_is_a_finding() -> None:
    # The control: a real loop. Docker's restart backoff caps at one minute, so
    # a looping container was always started moments before the census looked.
    findings = _restart_loop_plan(restarts=202, started_at="2026-10-20T18:48:48Z")
    assert len(findings) == 1
    assert "within the last" in findings[0]["detail"]


@pytest.mark.parametrize(
    ("up_seconds", "expected"), [(1799, True), (1800, False), (1801, False)]
)
def test_restart_loop_window_boundary(up_seconds: int, expected: bool) -> None:
    from datetime import UTC, datetime, timedelta

    observed = datetime(2026, 10, 20, 18, 49, 18, tzinfo=UTC)
    started = (observed - timedelta(seconds=up_seconds)).isoformat()
    findings = _restart_loop_plan(
        restarts=201, started_at=started, observed_at=observed.isoformat()
    )
    assert bool(findings) is expected


@pytest.mark.parametrize(
    ("started_at", "observed_at", "state"),
    [
        (None, _BROKER_OBSERVED_AT, "running"),
        (_BROKER_STARTED_AT, None, "running"),
        ("", _BROKER_OBSERVED_AT, "running"),
        ("0001-01-01T00:00:00Z", _BROKER_OBSERVED_AT, "running"),
        ("not-a-time", _BROKER_OBSERVED_AT, "running"),
        (_BROKER_STARTED_AT, _BROKER_OBSERVED_AT, "restarting"),
        (_BROKER_STARTED_AT, _BROKER_OBSERVED_AT, "exited"),
    ],
)
def test_restart_loop_unprovable_quiet_stays_a_finding(
    started_at: str | None, observed_at: str | None, state: str
) -> None:
    """A count past the bound is a finding unless the census can SHOW it is old."""
    assert _restart_loop_plan(
        restarts=201, started_at=started_at, observed_at=observed_at, state=state
    )


def test_restart_loop_under_the_bound_ignores_start_time() -> None:
    assert _restart_loop_plan(restarts=2, started_at="2026-10-20T18:49:00Z") == []


def _event_over(fixture: dict[str, Any]) -> dict[str, Any]:
    """The drift event a T1.2 planner would publish for this fixture."""
    events = _load_event_module()
    plan = {
        "schema_version": PLAN.SCHEMA_VERSION,
        "host": "lab-201",
        "lanes_checked": [fixture["envelope"]["lane"]],
        "lanes_not_applicable": [],
        "lanes_skipped_optional_down": [],
        "findings": [
            PLAN._finding(f["lane"] or "", f["kind"], f["container"], "fixture")
            for f in fixture["expected_findings"]
        ],
    }
    event: dict[str, Any] = events.build_event(host="omninode-pc", plan=plan)
    return event


@pytest.mark.parametrize(
    "path", [pytest.param(p, id=p.stem) for p in _lab_sync_fixtures()]
)
def test_schema_version_1_1_0_event_validates(path: Path) -> None:
    """AC2: the event over every fixture's findings validates at 1.1.0."""
    events = _load_event_module()
    event = _event_over(_fixture(path))
    assert event["schema_version"] == "1.1.0"
    assert events.validate_event(event, kind_severity=PLAN.FINDING_KIND_SEVERITY) == []


@pytest.mark.parametrize(
    "path", [pytest.param(p, id=p.stem) for p in _lab_sync_fixtures()]
)
def test_schema_version_1_0_0_consumer_parses_a_1_1_0_event(path: Path) -> None:
    """AC2: every key a 1.0.0 consumer reads is present, with its 1.0.0 type."""
    event = _event_over(_fixture(path))
    assert set(event) == set(_V1_0_0_EVENT_FIELDS)
    for key, types in _V1_0_0_EVENT_FIELDS.items():
        assert isinstance(event[key], types), (key, event[key])
    for finding in event["findings"]:
        assert set(finding) == {"lane", "kind", "container", "detail", "severity"}
        assert all(isinstance(v, str) for v in finding.values())


def test_schema_version_1_1_0_passes_the_repo_1_0_0_consumers(tmp_path: Path) -> None:
    """AC2: the staleness gate and the refresh decision read a 1.1.0 snapshot."""
    sys.path.insert(0, str(_REPO / "scripts"))
    try:
        from check_lane_census_age import check_census_age
        from lane_census_refresh_decision import validate_candidate
    finally:
        sys.path.remove(str(_REPO / "scripts"))
    event = _event_over(_fixture(_FIXTURES / "inventory_container_unhealthy.json"))
    snapshot = tmp_path / "census-snapshot.json"
    snapshot.write_text(json.dumps(event), encoding="utf-8")
    assert check_census_age(snapshot, 7) == 0
    validate_candidate(event)


def test_schema_version_observed_event_is_1_1_0() -> None:
    events = _load_event_module()
    observed = events.build_observed_event(host="omninode-pc", plan={"findings": []})
    assert observed["schema_version"] == "1.1.0"


def test_schema_version_census_snapshot_still_validates() -> None:
    """A census snapshot in the committed shape validates (OMN-20934: neutral copy)."""
    events = _load_event_module()
    snapshot = json.loads(NEUTRAL_SNAPSHOT.read_text(encoding="utf-8"))
    assert (
        events.validate_event(snapshot, kind_severity=PLAN.FINDING_KIND_SEVERITY) == []
    )


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        pytest.param(
            lambda e: e.update(schema_version="2.0.0"), "schema_version", id="version"
        ),
        pytest.param(lambda e: e.pop("alert_key"), "alert_key", id="missing-key"),
        pytest.param(
            lambda e: e["findings"][0].update(kind="container_on_fire"),
            "not a declared kind",
            id="undeclared-kind",
        ),
        pytest.param(
            lambda e: e["findings"][0].update(severity="warning"),
            "declared 'critical'",
            id="wrong-severity",
        ),
        pytest.param(
            lambda e: e.update(drift_count=0), "drift_count", id="drift-count"
        ),
        pytest.param(lambda e: e.update(extra=1), "outside the contract", id="extra"),
    ],
)
def test_schema_version_validator_refuses_a_broken_event(
    mutate: Any, needle: str
) -> None:
    events = _load_event_module()
    event = _event_over(_fixture(_FIXTURES / "inventory_container_unhealthy.json"))
    mutate(event)
    errors = events.validate_event(event, kind_severity=PLAN.FINDING_KIND_SEVERITY)
    assert any(needle in e for e in errors), errors


# ---------------------------------------------------------------------------
# OMN-19414 — lab release sync, wave 1 task T1.2: the planner evaluates the
# desired-state kinds. AC1 is the fixture tests above (-k lab_sync) and the
# exact-set and mutation tests below; AC2 is the `unreadable` tests.
# ---------------------------------------------------------------------------

_LAB_SYNC_KIND_SET = frozenset(_PLAN_LAB_SYNC_KINDS)
_CLEAN = _FIXTURES / "inventory_clean.json"


def _lab_sync_pairs(plan: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (f["kind"], f["container"])
        for f in plan["findings"]
        if f["kind"] in _LAB_SYNC_KIND_SET
    }


def _row(fixture: dict[str, Any], name: str) -> dict[str, Any]:
    row: dict[str, Any] = next(
        r for r in fixture["envelope"]["containers"] if r["Names"] == name
    )
    return row


@pytest.mark.parametrize("path", _bad_fixture_params())
def test_lab_sync_fixture_produces_exactly_its_findings(path: Path) -> None:
    """AC1: the findings are the fixture's, no more and no fewer."""
    fixture = _fixture(path)
    expected = {(f["kind"], f["container"]) for f in fixture["expected_findings"]}
    assert _lab_sync_pairs(_plan_with_desired(fixture)) == expected


def test_lab_sync_clean_fixture_with_one_revision_changed_names_that_container() -> (
    None
):
    """Positive control for the clean fixture: its silence is the rule, not a blind spot."""
    fixture = _fixture_v1(_CLEAN)
    fixture["desired"]["containers"][0]["revision"] = "f" * 40
    name = fixture["desired"]["containers"][0]["name"]
    plan = _plan_with_desired(fixture)
    assert _lab_sync_pairs(plan) == {("revision_mismatch", name)}
    assert plan["has_drift"] is True


@pytest.mark.parametrize(("label_length", "drifted"), [(12, False), (11, True)])
def test_lab_sync_revision_label_may_be_a_twelve_character_prefix(
    label_length: int, drifted: bool
) -> None:
    fixture = _fixture_v1(_CLEAN)
    want = fixture["desired"]["containers"][0]
    _row(fixture, want["name"])["Labels"]["org.opencontainers.image.revision"] = want[
        "revision"
    ][:label_length]
    flagged = bool(_lab_sync_pairs(_plan_with_desired(fixture)))
    assert flagged is drifted


def test_lab_sync_config_hash_compares_a_lane_container() -> None:
    fixture = _fixture_v1(_CLEAN)
    want = fixture["desired"]["containers"][0]
    want["config_hash"] = "0" * 64
    plan = _plan_with_desired(fixture)
    assert _lab_sync_pairs(plan) == {("config_hash_mismatch", want["name"])}
    want["config_hash"] = _row(fixture, want["name"])["Labels"][
        "com.docker.compose.config-hash"
    ]
    assert not _lab_sync_pairs(_plan_with_desired(fixture))


def test_lab_sync_package_names_are_compared_in_normalized_form() -> None:
    fixture = _fixture(_FIXTURES / "inventory_package_version_mismatch.json")
    name = fixture["expected_findings"][0]["container"]
    row = _row(fixture, name)
    row["Packages"] = {"omnibase_core": "0.47.20", "OmniBase.Infra": "0.38.56"}
    assert not _lab_sync_pairs(_plan_with_desired(fixture))
    del row["Packages"]["omnibase_core"]
    plan = _plan_with_desired(fixture)
    assert _lab_sync_pairs(plan) == {("package_version_mismatch", name)}
    assert "(not installed)" in plan["findings"][0]["detail"]


def test_lab_sync_broker_config_reads_numbers_as_rpk_prints_them() -> None:
    fixture = _fixture(_FIXTURES / "inventory_broker_config_mismatch.json")
    fixture["envelope"]["broker_config"]["omnibase-infra-redpanda"] = {
        "retention_bytes": 1073741824,
        "segment_fallocation_step": "1048576",
    }
    assert not _lab_sync_pairs(_plan_with_desired(fixture))


@pytest.mark.parametrize(
    ("drop", "needle"),
    [
        pytest.param("Packages", "Packages", id="packages"),
        pytest.param("broker_config", "broker_config", id="broker"),
        pytest.param("github_runners", "github_runners", id="runners"),
    ],
)
def test_lab_sync_declared_facet_the_collector_did_not_supply_is_never_clean(
    drop: str, needle: str
) -> None:
    fixture_name = {
        "Packages": "inventory_package_version_mismatch.json",
        "broker_config": "inventory_broker_config_mismatch.json",
        "github_runners": "inventory_runner_offline.json",
    }[drop]
    fixture = _fixture(_FIXTURES / fixture_name)
    if drop == "Packages":
        del _row(fixture, fixture["expected_findings"][0]["container"])["Packages"]
    else:
        del fixture["envelope"][drop]
    with pytest.raises(ValueError, match=needle):
        _plan_with_desired(fixture)


def test_lab_sync_runner_workdir_matching_the_fleet_path_is_clean() -> None:
    fixture = _fixture(_FIXTURES / "inventory_runner_workdir_mismatch.json")
    workdir = fixture["desired"]["runners"]["workdir"]
    _row(fixture, "omninode-deploy-runner")["Labels"][
        "com.docker.compose.project.working_dir"
    ] = workdir
    assert not _lab_sync_pairs(_plan_with_desired(fixture))


def test_lab_sync_runner_count_names_the_host() -> None:
    fixture = _fixture(_FIXTURES / "inventory_runner_count_mismatch.json")
    plan = _plan_with_desired(fixture)
    (finding,) = [f for f in plan["findings"] if f["kind"] == "runner_count_mismatch"]
    assert finding["container"] == fixture["desired"]["runners"]["host"]
    assert "runs 3 runner container(s)" in finding["detail"]
    assert "declares 4" in finding["detail"]


def test_lab_sync_runner_offline_needs_a_running_container() -> None:
    """A runner whose container is gone is the count's finding, not a second one."""
    fixture = _fixture(_FIXTURES / "inventory_runner_offline.json")
    fixture["envelope"]["containers"] = [
        r
        for r in fixture["envelope"]["containers"]
        if r["Names"] != "omninode-runner-1"
    ]
    kinds = {kind for kind, _ in _lab_sync_pairs(_plan_with_desired(fixture))}
    assert kinds == {"runner_count_mismatch"}


def test_lab_sync_allowed_undeclared_admits_its_class() -> None:
    fixture = _fixture_v1(_FIXTURES / "inventory_undeclared_container.json")
    fixture["desired"]["allowed_undeclared"] = [
        {
            "name": "dockerhub-mirror",
            "match": {"compose_project": "dockerhub-mirror"},
            "owner": "lab-operator",
            "reason": "pull-through registry mirror",
        }
    ]
    assert not _lab_sync_pairs(_plan_with_desired(fixture))


def test_lab_sync_allowed_undeclared_age_bound_needs_an_age_the_collector_read() -> (
    None
):
    fixture = _fixture_v1(_FIXTURES / "inventory_undeclared_container.json")
    fixture["desired"]["allowed_undeclared"] = [
        {
            "name": "dockerhub-mirror",
            "match": {"compose_project": "dockerhub-mirror", "max_age_seconds": 60},
            "owner": "lab-operator",
            "reason": "pull-through registry mirror",
        }
    ]
    expected = {("undeclared_container", "dockerhub-mirror")}
    assert _lab_sync_pairs(_plan_with_desired(fixture)) == expected
    _row(fixture, "dockerhub-mirror")["AgeSeconds"] = 30
    assert not _lab_sync_pairs(_plan_with_desired(fixture))
    _row(fixture, "dockerhub-mirror")["AgeSeconds"] = 61
    assert _lab_sync_pairs(_plan_with_desired(fixture)) == expected


def test_lab_sync_undeclared_is_graded_only_when_a_host_surface_owns_the_list() -> None:
    fixture = _fixture_v1(_FIXTURES / "inventory_undeclared_container.json")
    fixture["desired"]["surface"] = {
        "id": f"{_HOST}/omnibase-infra",
        "kind": "compose_lane",
        "host": _HOST,
        "compose_project": "omnibase-infra",
        "mode": "reconcile",
    }
    fixture["desired"]["allowed_undeclared"] = []
    assert not _lab_sync_pairs(_plan_with_desired(fixture))


def test_lab_sync_admitted_container_restarting_past_the_bound_is_a_loop() -> None:
    fixture = _fixture(_FIXTURES / "inventory_container_restart_loop.json")
    row = _row(fixture, "omnimarket-projection-registration")
    row["RestartCount"] = PLAN.DEFAULT_ALLOWED_UNDECLARED_RESTART_BOUND
    assert not _lab_sync_pairs(_plan_with_desired(fixture))
    row["RestartCount"] += 1
    assert _lab_sync_pairs(_plan_with_desired(fixture)) == {
        ("container_restart_loop", "omnimarket-projection-registration")
    }


def test_lab_sync_list_of_documents_is_graded_as_one_state() -> None:
    fixture = _fixture_v1(_FIXTURES / "inventory_undeclared_container.json")
    clean = _fixture_v1(_CLEAN)
    fixture["desired"] = [clean["desired"], fixture["desired"]]
    fixture["envelope"]["containers"] = [
        *clean["envelope"]["containers"],
        _row(
            _fixture(_FIXTURES / "inventory_undeclared_container.json"),
            "dockerhub-mirror",
        ),
    ]
    assert _lab_sync_pairs(_plan_with_desired(fixture)) == {
        ("undeclared_container", "dockerhub-mirror")
    }


# --- AC2: an unreadable desired state is a finding, never zero findings -------


def test_unreadable_null_desired_state_is_critical_drift_not_clean() -> None:
    plan = _plan_with_desired(
        _fixture(_FIXTURES / "inventory_desired_state_unreadable.json")
    )
    (finding,) = [
        f for f in plan["findings"] if f["kind"] == "desired_state_unreadable"
    ]
    assert finding["severity"] == "critical"
    assert finding["container"] == "desired_state"
    assert plan["has_drift"] is True
    assert plan["lab_sync_evaluated"] is True


@pytest.mark.parametrize("desired", [None, [], {}, "x", 5, [1], [{}, 3]], ids=repr)
def test_unreadable_malformed_desired_states(desired: Any) -> None:
    envelope = dict(_fixture(_CLEAN)["envelope"])
    envelope["desired_state"] = desired
    plan = PLAN.build_plan(envelope, _frozen_manifest())
    kinds = [f["kind"] for f in plan["findings"]]
    if desired == {}:
        # An empty document is a document: nothing declared, nothing to compare.
        assert kinds == []
    else:
        assert kinds == ["desired_state_unreadable"], plan["findings"]


def test_unreadable_reason_rides_in_the_detail() -> None:
    envelope = dict(_fixture(_CLEAN)["envelope"])
    envelope["desired_state"] = None
    envelope["desired_state_error"] = "containers[2].config_hash: not a sha256"
    plan = PLAN.build_plan(envelope, _frozen_manifest())
    assert "containers[2].config_hash: not a sha256" in plan["findings"][0]["detail"]


def test_unreadable_envelope_without_a_desired_state_key_is_not_evaluated() -> None:
    """Before the driver supplies a state, the plan says the kinds were not graded."""
    envelope = dict(_fixture(_CLEAN)["envelope"])
    plan = PLAN.build_plan(envelope, _frozen_manifest())
    assert plan["lab_sync_evaluated"] is False
    assert not _lab_sync_pairs(plan)


def _run_planner(
    tmp_path: Path, envelope: dict[str, Any], desired_state: str | None
) -> subprocess.CompletedProcess[str]:
    import yaml

    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump(_frozen_manifest()), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "LANE_CENSUS_DESIRED_STATE"}
    env["LANE_MANIFEST"] = str(manifest)
    if desired_state is not None:
        env["LANE_CENSUS_DESIRED_STATE"] = desired_state
    return subprocess.run(
        [sys.executable, str(_PLAN_PATH)],
        input=json.dumps(envelope),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_unreadable_run_with_the_desired_state_file_removed_exits_nonzero(
    tmp_path: Path,
) -> None:
    """AC2: the file is gone; the run fails and names desired_state_unreadable."""
    envelope = dict(_fixture(_CLEAN)["envelope"])
    result = _run_planner(tmp_path, envelope, str(tmp_path / "removed.json"))
    assert result.returncode == PLAN.EXIT_DESIRED_STATE_UNREADABLE != 0
    plan = json.loads(result.stdout)
    assert [f["kind"] for f in plan["findings"]] == ["desired_state_unreadable"]
    assert "removed.json" in plan["findings"][0]["detail"]
    assert plan["has_drift"] is True


@pytest.mark.parametrize(
    "document",
    [
        pytest.param("{", id="truncated"),
        pytest.param("[]", id="not-an-object"),
        pytest.param(
            (_FIXTURES / "desired_state_missing_field.json").read_text(),
            id="missing-field",
        ),
        pytest.param(
            (_FIXTURES / "desired_state_extra_field.json").read_text(),
            id="extra-field",
        ),
    ],
)
def test_unreadable_run_with_a_refused_document_exits_nonzero(
    tmp_path: Path, document: str
) -> None:
    path = tmp_path / "desired.json"
    path.write_text(document, encoding="utf-8")
    result = _run_planner(tmp_path, dict(_fixture(_CLEAN)["envelope"]), str(path))
    assert result.returncode == PLAN.EXIT_DESIRED_STATE_UNREADABLE
    plan = json.loads(result.stdout)
    assert [f["kind"] for f in plan["findings"]] == ["desired_state_unreadable"]


def test_unreadable_run_with_an_empty_directory_exits_nonzero(tmp_path: Path) -> None:
    empty = tmp_path / "desired"
    empty.mkdir()
    result = _run_planner(tmp_path, dict(_fixture(_CLEAN)["envelope"]), str(empty))
    assert result.returncode == PLAN.EXIT_DESIRED_STATE_UNREADABLE


def test_unreadable_positive_control_a_valid_document_exits_zero(
    tmp_path: Path,
) -> None:
    """The same run with a readable state exits 0 and grades the kinds."""
    path = tmp_path / "desired.json"
    path.write_text(
        (_FIXTURES / "desired_state_valid_host.json").read_text(), encoding="utf-8"
    )
    result = _run_planner(tmp_path, dict(_fixture(_CLEAN)["envelope"]), str(path))
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["lab_sync_evaluated"] is True
    assert plan["findings"] == []


def test_unreadable_run_with_no_desired_state_named_keeps_the_old_exit(
    tmp_path: Path,
) -> None:
    result = _run_planner(tmp_path, dict(_fixture(_CLEAN)["envelope"]), None)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["lab_sync_evaluated"] is False


def test_unreadable_run_reads_a_directory_of_documents(tmp_path: Path) -> None:
    directory = tmp_path / "desired"
    directory.mkdir()
    for name in (
        "desired_state_valid_host.json",
        "desired_state_valid_runner_fleet.json",
    ):
        (directory / name).write_text((_FIXTURES / name).read_text(), encoding="utf-8")
    envelope = dict(_fixture(_CLEAN)["envelope"])
    result = _run_planner(tmp_path, envelope, str(directory))
    assert result.returncode == 0, result.stderr
    kinds = {f["kind"] for f in json.loads(result.stdout)["findings"]}
    # The fleet document declares runners the clean inventory does not run.
    assert "runner_count_mismatch" in kinds
    assert "desired_state_unreadable" not in kinds
