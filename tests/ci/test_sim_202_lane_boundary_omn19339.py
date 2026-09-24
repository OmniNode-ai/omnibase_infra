# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19339 — the sim-202 lane is ungoverned, isolated, and on .202 only.

sim-202 is row S2 of the unified verification plan: an isolated OmniNode stack
on the .202 host with its own Postgres, Redpanda and Valkey, used for fault
injection. It is a premise for nothing. Three properties keep it that way, and
each is asserted in both directions where a direction exists:

* **No governance acquired, none eroded.** The lane is absent from
  ``GOVERNED_LANES`` and ``GRANT_INTERLOCK_LANES`` (ticket AC4), and those two
  sets still hold exactly the lanes they held before this lane existed.
* **Declared for .202 and nowhere else.** The manifest names ``lab-202`` as its
  only host, and ``lab-202`` resolves from the name ``hostname`` prints there,
  so the host-aware census (OMN-19088) evaluates it on .202 and reports every
  .201 lane not-applicable.
* **Nothing reachable from another host.** Every port the overlay publishes
  binds the loopback address and is one of the reserved block, so no .201
  client can open a connection into its plaintext broker or databases, and none
  of its identifiers can satisfy omni_home's no-raw-prod-bypass matcher.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.runtime.models.enum_bifrost_lane_locale import (
    EnumBifrostLaneLocale,
)
from omnibase_infra.runtime.models.model_bifrost_lane_overlay import (
    ModelBifrostLaneOverlay,
)
from scripts.lane_census_plan import build_plan, resolve_host, validate_hosts
from scripts.preflight_lane_deploy_attribution import (
    GOVERNED_LANES,
    GRANT_INTERLOCK_LANES,
)

pytestmark = pytest.mark.ci

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "deploy" / "lane-census" / "lane-manifest.yaml"
OVERLAY_PATH = ROOT / "docker" / "docker-compose.sim-202.yml"
BIFROST_OVERLAY_PATH = ROOT / "docker" / "lane-overlays" / "sim-202.bifrost.yaml"

LANE = "sim-202"
HOST_ID = "lab-202"
HOSTNAME_ON_HOST = "omnipc2"
COMPOSE_PROJECT = "omnibase-infra-sim-202"
NETWORK = "omnibase-infra-sim-202-network"

#: Plan row S2's block, minus the gateway API port (62090), which is reserved
#: and deliberately unbound: this lane runs no gateway.
EXPECTED_PUBLISHED_PORTS = {
    "62085",  # runtime main
    "62086",  # runtime effects
    "62002",  # projection API
    "62436",  # Postgres
    "62379",  # Valkey
    "62092",  # Redpanda Kafka (external listener)
    "62644",  # Redpanda admin
}

#: Mirrors omni_home's no-raw-prod-bypass scanner (`_GOVERNED_PROJECTS`,
#: `_PROD_PORTS` + `_STABILITY_PORTS`). Restated, not imported: it lives in
#: another repository, and the point is to prove non-collision with it.
GOVERNED_COMPOSE_PROJECTS = (
    "omnibase-infra-prod",
    "omnibase-infra-stability-test",
    "omnibase-infra-judge",
)
GOVERNED_LANE_PORT_LITERALS = ("28085", "28086", "18085", "18086")

#: Services the lane must never declare. Keycloak and Infisical are the roles
#: plan's non-roles on .202; runtime-worker belongs to lanes with a worker.
FORBIDDEN_SERVICE_FRAGMENTS = ("keycloak", "infisical", "runtime-worker")


def _manifest() -> dict:
    return yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))


def _published_port_lines() -> list[str]:
    raw = OVERLAY_PATH.read_text(encoding="utf-8")
    return re.findall(r'^\s+-\s+"([^"]*:\d+:\d+)"\s*$', raw, re.MULTILINE)


def test_sim_202_is_not_deploy_attribution_governed() -> None:
    """AC4, gate 1. OUT for sim-202; the governed set is unchanged."""
    assert LANE not in GOVERNED_LANES
    assert frozenset({"stability-test", "prod", "judge"}) == GOVERNED_LANES, (
        f"adding sim-202 must not change the governed lane set; got {sorted(GOVERNED_LANES)}"
    )


def test_sim_202_is_not_in_the_prod_grant_interlock() -> None:
    """AC4, gate 2. No grant resolves anything from this lane."""
    assert LANE not in GRANT_INTERLOCK_LANES
    assert frozenset({"stability-test"}) == GRANT_INTERLOCK_LANES


def test_manifest_declares_sim_202_on_lab_202_only() -> None:
    lane = _manifest()["lanes"][LANE]
    assert lane["hosts"] == [HOST_ID], (
        f"sim-202 runs on .202 only; a second host would make the census expect "
        f"it there too. Got {lane['hosts']}"
    )
    assert lane["compose_project"] == COMPOSE_PROJECT
    assert lane["network"] == NETWORK
    assert lane["compose_file"] == "docker/docker-compose.sim-202.yml"
    assert lane["optional"] is True, (
        "the stack is torn down between SIM runs; a non-optional entry would file "
        "a census finding for a lane that is correctly absent"
    )


def test_lab_202_resolves_from_its_hostname_and_carries_only_sim_202() -> None:
    manifest = _manifest()
    validate_hosts(manifest)
    assert resolve_host(HOSTNAME_ON_HOST, manifest) == HOST_ID
    lanes_on_host = sorted(
        name for name, spec in manifest["lanes"].items() if HOST_ID in spec["hosts"]
    )
    assert lanes_on_host == [LANE], (
        f"no .201 lane may be declared for .202; got {lanes_on_host}"
    )


def test_sim_202_declares_no_forbidden_service() -> None:
    names = [s["name"] for s in _manifest()["lanes"][LANE]["services"]]
    offending = [n for n in names for f in FORBIDDEN_SERVICE_FRAGMENTS if f in n]
    assert not offending, f"sim-202 must not declare {offending}"


def test_every_published_port_is_loopback_and_in_the_block() -> None:
    lines = _published_port_lines()
    assert lines, (
        "positive control: the overlay publishes ports, the scrape must see them"
    )
    published: set[str] = set()
    for line in lines:
        host_ip, host_port, _container_port = line.split(":")
        assert host_ip == "127.0.0.1", (
            f"{line!r} binds {host_ip!r}; every sim-202 port binds the loopback "
            "address so no other host can reach the plaintext broker or databases"
        )
        published.add(host_port)
    assert published == EXPECTED_PUBLISHED_PORTS


def test_sim_202_identifiers_cannot_trip_the_no_raw_prod_bypass_matcher() -> None:
    for governed in GOVERNED_COMPOSE_PROJECTS:
        assert governed not in COMPOSE_PROJECT
    for port in EXPECTED_PUBLISHED_PORTS:
        for literal in GOVERNED_LANE_PORT_LITERALS:
            assert literal not in port, (
                f"port {port} contains governed literal {literal}"
            )


def test_overlay_never_builds_on_its_host() -> None:
    """Decision 19: no source tree with `.git` on .202, so no build there.

    Every runtime service runs the pinned image the operator names, and the
    overlay refuses to render without it rather than falling back to `build:`.
    """
    raw = OVERLAY_PATH.read_text(encoding="utf-8")
    assert "${SIM_202_RUNTIME_IMAGE:?" in raw
    image_refs = re.findall(
        r"^\s+image:\s*\*sim-202-runtime-image\s*$", raw, re.MULTILINE
    )
    assert len(image_refs) == 3, (
        "omninode-runtime, runtime-effects and projection-api each run the pinned "
        f"image; found {len(image_refs)} references"
    )


# --- AC3: the census on .202 --------------------------------------------------


def _row(name: str, project: str, *, running: bool = True) -> dict[str, Any]:
    return {
        "Names": name,
        "State": "running" if running else "exited",
        "Status": "Up 2 hours" if running else "Exited (0) 1 hour ago",
        "Image": "onex-lab/omninode-runtime:pinned",
        "Labels": {"com.docker.compose.project": project},
    }


def _healthy_sim_202_rows() -> list[dict[str, Any]]:
    lane = _manifest()["lanes"][LANE]
    return [
        _row(s["name"], lane["compose_project"], running=s.get("kind") != "oneshot")
        for s in lane["services"]
    ]


def _plan_on_202(rows: list[dict[str, Any]]) -> dict[str, Any]:
    envelope = {
        "host": HOSTNAME_ON_HOST,
        "lane": None,
        "containers": rows,
        "networks": [NETWORK],
        "runtime_tag": None,
    }
    return build_plan(envelope, _manifest())


def test_census_on_202_reads_sim_202_clean_and_every_other_lane_not_applicable() -> (
    None
):
    plan = _plan_on_202(_healthy_sim_202_rows())
    manifest = _manifest()
    assert plan["host"] == HOST_ID
    assert plan["lanes_checked"] == [LANE]
    assert set(plan["lanes_not_applicable"]) == set(manifest["lanes"]) - {LANE}
    assert plan["findings"] == []
    assert plan["has_drift"] is False


def test_census_on_202_still_reports_a_stopped_sim_202_runtime() -> None:
    """Negative control: scoping to .202 must not hide a real outage there."""
    rows = [
        r for r in _healthy_sim_202_rows() if r["Names"] != "omninode-sim-202-runtime"
    ]
    plan = _plan_on_202(rows)
    assert plan["has_drift"] is True
    assert any(
        f["lane"] == LANE and f.get("container") == "omninode-sim-202-runtime"
        for f in plan["findings"]
    ), plan["findings"]


def test_a_201_lane_container_on_202_is_a_finding() -> None:
    """A .201 lane running on .202 is the isolation failure, and it is named."""
    rows = [*_healthy_sim_202_rows(), _row("omninode-runtime", "omnibase-infra")]
    plan = _plan_on_202(rows)
    assert any(
        f["lane"] == "dev" and f["kind"] == "lane_on_undeclared_host"
        for f in plan["findings"]
    ), plan["findings"]


def test_sim_202_binds_no_model_server_on_any_lab_host() -> None:
    """Seam SIM.1: no route to a .201 listener, and none to .202's model server.

    The dogfood overlay binds the .201 model server; this lane mounts its own,
    cloud-locale overlay with zero local backends, and never the dogfood file.
    """
    overlay = ModelBifrostLaneOverlay.model_validate(
        yaml.safe_load(BIFROST_OVERLAY_PATH.read_text(encoding="utf-8"))
    )
    assert overlay.lane == LANE
    assert overlay.locale is EnumBifrostLaneLocale.CLOUD
    assert overlay.backends == ()
    compose = OVERLAY_PATH.read_text(encoding="utf-8")
    assert "dogfood.bifrost.yaml" not in compose
    assert "192.168." not in compose
