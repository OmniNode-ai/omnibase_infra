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
    """All declared prod services running, migrations exited 0."""
    services = list(MANIFEST["lanes"]["prod"]["services"])
    rows: list[dict[str, str]] = []
    for svc in services:
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
        "containers": containers,
        "networks": ["omnibase-infra-prod-network"],
        "runtime_tag": None,
    }
    plan = PLAN.build_plan(envelope, MANIFEST)
    stuck = [f for f in plan["findings"] if f["kind"] == "oneshot_stuck"]
    assert stuck and stuck[0]["severity"] == "warning"


def test_unexpected_lane_labeled_container_is_drift() -> None:
    """A container labeled for the lane but not declared is unexpected_container."""
    containers = _healthy_prod_containers()
    containers.append(_container("omninode-prod-rogue-shadow", lane="prod"))
    envelope = {
        "lane": "prod",
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
    manifest = PLAN.load_manifest(_MANIFEST_PATH)
    manifest["lanes"]["prod"]["image_tag_pattern"] = r"0\.37\..+"
    containers = _healthy_prod_containers()
    for c in containers:
        if c["Names"] == "omninode-prod-runtime":
            c["Image"] = "omninode-runtime:0.30.0-stale"
    envelope = {
        "lane": "prod",
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
            {"lane": "does-not-exist", "containers": [], "networks": []}, MANIFEST
        )


def test_all_lanes_default_excludes_nothing_required() -> None:
    """With lane=None all manifest lanes are checked."""
    envelope = {"lane": None, "containers": [], "networks": [], "runtime_tag": None}
    plan = PLAN.build_plan(envelope, MANIFEST)
    assert set(plan["lanes_checked"]) == set(MANIFEST["lanes"].keys())


def test_tag_parsing_handles_registry_host_port() -> None:
    """`_tag_of` must not mistake a registry host:port for the tag."""
    assert PLAN._tag_of("registry.local:5000/omninode-runtime:0.37.0") == "0.37.0"
    assert PLAN._tag_of("omninode-runtime") == "latest"
    assert PLAN._tag_of("omninode-runtime@sha256:abc") == "latest"
