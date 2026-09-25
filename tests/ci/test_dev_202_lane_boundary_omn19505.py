# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19505 — the dev-202 lane is ungoverned, isolated, and on .202 only.

dev-202 is task B2 of the second-deploy-slot plan (epic OMN-19500): a second
dev-shaped deployed lane on the .202 host, with its own Postgres, Redpanda and
Valkey, so a second deploy-agent instance can run a second runtime deploy slot.
Four properties keep it inside that role, and each is asserted in both
directions where a direction exists:

* **No governance acquired, none eroded.** The lane is absent from
  ``GOVERNED_LANES`` and ``GRANT_INTERLOCK_LANES`` (ticket AC2), and those two
  sets still hold exactly the lanes they held before this lane existed.
* **Declared for .202 and nowhere else.** The manifest names ``lab-202`` as its
  only host, so the host-aware census evaluates it on .202 and reports every
  .201 lane not-applicable (ticket AC3).
* **Nothing reachable from another host, nothing reaching .201.** Every port the
  overlay publishes binds the loopback address and is one of the declared
  block; the overlay names no .201 address and mounts no .201-bound Bifrost
  overlay.
* **Every service the .201 dev lane declares, under a dev-202 name** (ticket
  AC1), and no Keycloak or Infisical (the roles plan's non-roles on .202).
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
OVERLAY_PATH = ROOT / "docker" / "docker-compose.dev-202.yml"
BIFROST_OVERLAY_PATH = ROOT / "docker" / "lane-overlays" / "dev-202.bifrost.yaml"

LANE = "dev-202"
HOST_ID = "lab-202"
HOSTNAME_ON_HOST = "omnipc2"
COMPOSE_PROJECT = "omnibase-infra-dev-202"
NETWORK = "omnibase-infra-dev-202-network"

#: The declared block. 61090 (onex-api) is published by a disabled service, so
#: it is reserved and never bound; it is still in the block.
EXPECTED_PUBLISHED_PORTS = {
    "61085",  # runtime main
    "61086",  # runtime effects
    "61002",  # projection API
    "61087",  # agent actions consumer
    "61091",  # contract resolver
    "61092",  # skill lifecycle consumer
    "61093",  # context audit consumer
    "61053",  # intelligence API
    "61006",  # phoenix
    "61090",  # onex-api (disabled here, reserved)
    "61436",  # Postgres
    "61379",  # Valkey
    "61192",  # Redpanda Kafka (external listener)
    "61182",  # Redpanda proxy
    "61181",  # schema registry
    "61644",  # Redpanda admin
}

#: Mirrors omni_home's no-raw-prod-bypass scanner, as the sim-202 boundary test
#: does. Restated, not imported: it lives in another repository.
GOVERNED_COMPOSE_PROJECTS = (
    "omnibase-infra-prod",
    "omnibase-infra-stability-test",
    "omnibase-infra-judge",
)
GOVERNED_LANE_PORT_LITERALS = ("28085", "28086", "18085", "18086")

#: The roles plan's non-roles on .202.
FORBIDDEN_SERVICE_FRAGMENTS = ("keycloak", "infisical")

#: The .201 dev lane's declared service -> its dev-202 name. AC1.
DEV_TO_DEV_202 = {
    "omninode-runtime": "omninode-dev-202-runtime",
    "omninode-runtime-effects": "omninode-dev-202-runtime-effects",
    "omnimarket-projection-tenant-registry-writer": "omnimarket-dev-202-projection-tenant-registry-writer",
    "omnimarket-projection-delegation-writer": "omnimarket-dev-202-projection-delegation-writer",
    "omnimarket-projection-registration-writer": "omnimarket-dev-202-projection-registration-writer",
    "omnimarket-projection-savings-writer": "omnimarket-dev-202-projection-savings-writer",
    "omnimarket-projection-tenant-credentials-writer": "omnimarket-dev-202-projection-tenant-credentials-writer",
    "omnimarket-projection-live-events-writer": "omnimarket-dev-202-projection-live-events-writer",
    "omnimarket-tenant-projection-writer": "omnimarket-dev-202-tenant-projection-writer",
    "omnibase-infra-forward-migration": "omnibase-infra-dev-202-forward-migration",
    "omnibase-infra-cloud-migration": "omnibase-infra-dev-202-cloud-migration",
    "omnibase-infra-cloud-migration-files": "omnibase-infra-dev-202-cloud-migration-files",
    "onex-api": "onex-api-dev-202",
    "omninode-infra-routing-decisions-consumer": "omninode-dev-202-infra-routing-decisions-consumer",
}

#: Declared but disabled on .202: the cloud image only the .201 lab-overlay pin
#: delivers.
PROFILE_GATED = {
    "omnibase-infra-dev-202-cloud-migration",
    "omnibase-infra-dev-202-cloud-migration-files",
    "onex-api-dev-202",
}


def _manifest() -> dict[str, Any]:
    return yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))


def _published_port_lines() -> list[str]:
    raw = OVERLAY_PATH.read_text(encoding="utf-8")
    return re.findall(r'^\s+-\s+"([^"]*:\d+:\d+)"\s*$', raw, re.MULTILINE)


def test_dev_202_lane_boundary_is_not_deploy_attribution_governed() -> None:
    """AC2, gate 1. OUT for dev-202; the governed set is unchanged."""
    assert LANE not in GOVERNED_LANES
    assert frozenset({"stability-test", "prod", "judge"}) == GOVERNED_LANES


def test_dev_202_lane_boundary_is_not_in_the_prod_grant_interlock() -> None:
    """AC2, gate 2. No grant resolves anything from this lane."""
    assert LANE not in GRANT_INTERLOCK_LANES
    assert frozenset({"stability-test"}) == GRANT_INTERLOCK_LANES


def test_dev_202_lane_boundary_manifest_declares_lab_202_only() -> None:
    lane = _manifest()["lanes"][LANE]
    assert lane["hosts"] == [HOST_ID]
    assert lane["compose_project"] == COMPOSE_PROJECT
    assert lane["network"] == NETWORK
    assert lane["compose_file"] == "docker/docker-compose.dev-202.yml"
    assert lane["optional"] is True, (
        "declared before it is built; a non-optional entry would file a census "
        "finding for a lane that is correctly absent"
    )


def test_dev_202_lane_boundary_declares_every_dev_service_under_its_name() -> None:
    """AC1. The .201 dev lane's declared set maps one to one onto dev-202."""
    manifest = _manifest()
    dev_declared = {s["name"] for s in manifest["lanes"]["dev"]["services"]}
    assert dev_declared == set(DEV_TO_DEV_202), (
        "the .201 dev lane's declared services changed; map the new one to its "
        f"dev-202 name here and in the overlay. dev declares {sorted(dev_declared)}"
    )
    by_name = {s["name"]: s for s in manifest["lanes"][LANE]["services"]}
    missing = sorted(v for v in DEV_TO_DEV_202.values() if v not in by_name)
    assert not missing, f"dev-202 does not declare {missing}"
    gated = {n for n, s in by_name.items() if s.get("kind") == "profile_gated"}
    assert gated == PROFILE_GATED


def test_dev_202_lane_boundary_declares_no_forbidden_service() -> None:
    names = [s["name"] for s in _manifest()["lanes"][LANE]["services"]]
    offending = [n for n in names for f in FORBIDDEN_SERVICE_FRAGMENTS if f in n]
    assert not offending, f"dev-202 must not declare {offending}"
    raw = OVERLAY_PATH.read_text(encoding="utf-8")
    for service in ("keycloak", "infisical"):
        block = re.search(rf"^  {service}:\n((?:    .*\n)+)", raw, re.MULTILINE)
        assert block is not None, f"the overlay must disable {service}"
        assert 'profiles: !override ["dev-202-disabled"]' in block.group(1)
        assert "container_name" not in block.group(1)


def test_dev_202_lane_boundary_every_published_port_is_loopback_and_in_the_block() -> (
    None
):
    lines = _published_port_lines()
    assert lines, "positive control: the overlay publishes ports"
    published: set[str] = set()
    for line in lines:
        host_ip, host_port, _container_port = line.split(":")
        assert host_ip == "127.0.0.1", (
            f"{line!r} binds {host_ip!r}; every dev-202 port binds the loopback "
            "address so no other host can reach its broker or databases"
        )
        published.add(host_port)
    assert published == EXPECTED_PUBLISHED_PORTS


def test_dev_202_lane_boundary_block_is_clear_of_every_other_lane() -> None:
    """No port of the block is published by any other committed compose file."""
    others: set[str] = set()
    for path in sorted((ROOT / "docker").glob("docker-compose*.yml")):
        if path == OVERLAY_PATH:
            continue
        others.update(re.findall(r"\b(61\d{3})\b", path.read_text(encoding="utf-8")))
    assert not (others & EXPECTED_PUBLISHED_PORTS), sorted(
        others & EXPECTED_PUBLISHED_PORTS
    )


def test_dev_202_lane_boundary_identifiers_cannot_trip_the_no_raw_prod_bypass_matcher() -> (
    None
):
    for governed in GOVERNED_COMPOSE_PROJECTS:
        assert governed not in COMPOSE_PROJECT
    for port in EXPECTED_PUBLISHED_PORTS:
        for literal in GOVERNED_LANE_PORT_LITERALS:
            assert literal not in port


def test_dev_202_lane_boundary_reaches_nothing_on_201() -> None:
    """The only tie to .201 is the deploy agent's control-topic read, not the
    stack's. The overlay names no .201 address, mounts no dev-lane Bifrost
    overlay, and leaves the external omnimemory network .202 does not have."""
    raw = OVERLAY_PATH.read_text(encoding="utf-8")
    body = "\n".join(
        line for line in raw.splitlines() if not line.lstrip().startswith("#")
    )
    assert "192.168." not in body
    assert "./lane-overlays/dev.bifrost.yaml" not in body
    assert "omnimemory-network" not in body
    mount = "./lane-overlays/dev-202.bifrost.yaml:/app/config/delegation/dev-202.bifrost.yaml:ro"
    assert body.count(mount) == 4, (
        "omninode-runtime, runtime-effects, runtime-worker and "
        "tenant-projection-writer each mount the dev-202 overlay"
    )
    assert (
        body.count(
            "BIFROST_LANE_OVERLAY_PATH: /app/config/delegation/dev-202.bifrost.yaml"
        )
        == 4
    )
    overlay = ModelBifrostLaneOverlay.model_validate(
        yaml.safe_load(BIFROST_OVERLAY_PATH.read_text(encoding="utf-8"))
    )
    assert overlay.lane == LANE
    assert overlay.locale is EnumBifrostLaneLocale.CLOUD
    assert overlay.backends == ()


def test_dev_202_lane_boundary_relabels_every_dev_lane_label() -> None:
    """A container on .202 still labelled com.omninode.lane=dev is the .201 dev
    lane on an undeclared host. Every service the dev overlay labels `dev` is
    relabelled here."""
    dev_lane = (ROOT / "docker" / "docker-compose.dev-lane.yml").read_text(
        encoding="utf-8"
    )
    labelled_in_dev: set[str] = set()
    current: str | None = None
    for line in dev_lane.splitlines():
        m = re.match(r"^  ([A-Za-z0-9_.-]+):\s*$", line)
        if m:
            current = m.group(1)
        elif current and re.search(r'"com\.omninode\.lane=dev"', line):
            labelled_in_dev.add(current)
    assert labelled_in_dev, "positive control: the dev overlay labels services dev"
    raw = OVERLAY_PATH.read_text(encoding="utf-8")
    relabelled: set[str] = set()
    current = None
    for line in raw.splitlines():
        m = re.match(r"^  ([A-Za-z0-9_.-]+):\s*$", line)
        if m:
            current = m.group(1)
        elif current and re.match(r"^\s+com\.omninode\.lane: dev-202\s*$", line):
            relabelled.add(current)
    assert labelled_in_dev <= relabelled, sorted(labelled_in_dev - relabelled)


# --- AC3: the census on .202 --------------------------------------------------


def _row(name: str, project: str, *, running: bool = True) -> dict[str, Any]:
    return {
        "Names": name,
        "State": "running" if running else "exited",
        "Status": "Up 2 hours" if running else "Exited (0) 1 hour ago",
        "Image": "onex-lab/omninode-runtime:pinned",
        "Labels": {"com.docker.compose.project": project, "com.omninode.lane": LANE},
    }


def _healthy_dev_202_rows() -> list[dict[str, Any]]:
    lane = _manifest()["lanes"][LANE]
    return [
        _row(s["name"], lane["compose_project"], running=s.get("kind") != "oneshot")
        for s in lane["services"]
        if s.get("kind") != "profile_gated"
    ]


def _plan_on_202(rows: list[dict[str, Any]], networks: list[str]) -> dict[str, Any]:
    envelope = {
        "host": HOSTNAME_ON_HOST,
        "lane": None,
        "containers": rows,
        "networks": networks,
        "runtime_tag": None,
    }
    return build_plan(envelope, _manifest())


def test_dev_202_lane_boundary_host_resolves_and_carries_dev_202() -> None:
    manifest = _manifest()
    validate_hosts(manifest)
    assert resolve_host(HOSTNAME_ON_HOST, manifest) == HOST_ID
    lanes_on_host = sorted(
        name for name, spec in manifest["lanes"].items() if HOST_ID in spec["hosts"]
    )
    assert lanes_on_host == ["dev-202", "sim-202"]


def test_dev_202_lane_boundary_census_on_202_reads_clean() -> None:
    plan = _plan_on_202(_healthy_dev_202_rows(), [NETWORK])
    manifest = _manifest()
    assert plan["host"] == HOST_ID
    assert sorted(plan["lanes_checked"]) == ["dev-202", "sim-202"]
    assert set(plan["lanes_not_applicable"]) == set(manifest["lanes"]) - {
        "dev-202",
        "sim-202",
    }
    assert plan["findings"] == []
    assert plan["has_drift"] is False


def test_dev_202_lane_boundary_census_still_reports_a_stopped_runtime() -> None:
    """Negative control: a real outage on dev-202 is still named."""
    rows = [
        r for r in _healthy_dev_202_rows() if r["Names"] != "omninode-dev-202-runtime"
    ]
    plan = _plan_on_202(rows, [NETWORK])
    assert plan["has_drift"] is True
    assert any(
        f["lane"] == LANE and f.get("container") == "omninode-dev-202-runtime"
        for f in plan["findings"]
    ), plan["findings"]


def test_dev_202_lane_boundary_a_201_dev_container_on_202_is_a_finding() -> None:
    rows = [
        *_healthy_dev_202_rows(),
        {
            **_row("omninode-runtime", "omnibase-infra"),
            "Labels": {"com.docker.compose.project": "omnibase-infra"},
        },
    ]
    plan = _plan_on_202(rows, [NETWORK])
    assert any(
        f["lane"] == "dev" and f["kind"] == "lane_on_undeclared_host"
        for f in plan["findings"]
    ), plan["findings"]
