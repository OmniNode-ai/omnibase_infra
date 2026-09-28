# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19543 — the dev-200 lane is ungoverned, isolated, and on .200 only.

dev-200 is the .200 slot of the operator's one-deploy-slot-per-lab-host ruling
(epic OMN-19500), built on the dev-202 template (OMN-19505): a dev-shaped
deployed lane on the .200 host with its own Postgres, Redpanda and Valkey, so a
deploy-agent instance there can run one more runtime deploy slot. The dev-202
boundary test's properties, restated for this host:

* **No governance acquired, none eroded.** Absent from ``GOVERNED_LANES`` and
  ``GRANT_INTERLOCK_LANES``, which still hold exactly what they held.
* **Declared for .200 and nowhere else.** The manifest names ``lab-200`` as its
  only host, so the host-aware census evaluates it on .200 beside the dogfood
  lane and reports every .201 and .202 lane not-applicable.
* **Nothing reachable from another host, nothing reaching .201.** Every port the
  overlay publishes binds the loopback address and nothing else, and is one of
  the declared 42xxx block, below the macOS ephemeral range 49152-65535.
* **Reachable by a verify runner container anyway.** .200 runs Docker Desktop,
  which forwards a container's ``host.docker.internal`` to the macOS host's
  loopback, so no bridge-gateway publish exists or is needed (measured on .200
  2026-09-25 with throwaway containers; the overlay's VERIFY-RUNNER REACH).
* **Every service the .201 dev lane declares, under a dev-200 name**, and no
  Keycloak or Infisical.
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
OVERLAY_PATH = ROOT / "docker" / "docker-compose.dev-200.yml"
BIFROST_OVERLAY_PATH = ROOT / "docker" / "lane-overlays" / "dev-200.bifrost.yaml"

LANE = "dev-200"
HOST_ID = "lab-200"
HOSTNAME_ON_HOST = "Stickybeatz-Studio.local"
COMPOSE_PROJECT = "omnibase-infra-dev-200"
NETWORK = "omnibase-infra-dev-200-network"

#: The declared block. 42090 (onex-api) is published by a disabled service, so
#: it is reserved and never bound; it is still in the block.
EXPECTED_PUBLISHED_PORTS = {
    "42085",  # runtime main
    "42086",  # runtime effects
    "42002",  # projection API
    "42087",  # agent actions consumer
    "42091",  # contract resolver
    "42092",  # skill lifecycle consumer
    "42093",  # context audit consumer
    "42053",  # intelligence API
    "42006",  # phoenix
    "42090",  # onex-api (disabled here, reserved)
    "42436",  # Postgres
    "42379",  # Valkey
    "42192",  # Redpanda Kafka (external listener)
    "42182",  # Redpanda proxy
    "42181",  # schema registry
    "42644",  # Redpanda admin
}

#: Mirrors omni_home's no-raw-prod-bypass scanner, as the sim-202 boundary test
#: does. Restated, not imported: it lives in another repository.
GOVERNED_COMPOSE_PROJECTS = (
    "omnibase-infra-prod",
    "omnibase-infra-stability-test",
    "omnibase-infra-judge",
)
GOVERNED_LANE_PORT_LITERALS = ("28085", "28086", "18085", "18086")

LOOPBACK = "127.0.0.1"

#: The macOS ephemeral port range on .200 (``sysctl net.inet.ip.portrange``).
MACOS_EPHEMERAL_FIRST = 49152

#: Not run on this lane, as on dev-202.
FORBIDDEN_SERVICE_FRAGMENTS = ("keycloak", "infisical")

#: The .201 dev lane's declared service -> its dev-200 name. AC1.
DEV_TO_DEV_200 = {
    "omninode-runtime": "omninode-dev-200-runtime",
    "omninode-runtime-effects": "omninode-dev-200-runtime-effects",
    "omnimarket-projection-tenant-registry-writer": "omnimarket-dev-200-projection-tenant-registry-writer",
    "omnimarket-projection-delegation-writer": "omnimarket-dev-200-projection-delegation-writer",
    "omnimarket-projection-registration-writer": "omnimarket-dev-200-projection-registration-writer",
    "omnimarket-projection-savings-writer": "omnimarket-dev-200-projection-savings-writer",
    "omnimarket-projection-tenant-credentials-writer": "omnimarket-dev-200-projection-tenant-credentials-writer",
    "omnimarket-projection-live-events-writer": "omnimarket-dev-200-projection-live-events-writer",
    "omnimarket-tenant-projection-writer": "omnimarket-dev-200-tenant-projection-writer",
    "omnibase-infra-forward-migration": "omnibase-infra-dev-200-forward-migration",
    "omnibase-infra-cloud-migration": "omnibase-infra-dev-200-cloud-migration",
    "omnibase-infra-cloud-migration-files": "omnibase-infra-dev-200-cloud-migration-files",
    "onex-api": "onex-api-dev-200",
    "omninode-infra-routing-decisions-consumer": "omninode-dev-200-infra-routing-decisions-consumer",
}

#: Declared but disabled on .200: the cloud image only the .201 lab-overlay pin
#: delivers.
PROFILE_GATED = {
    "omnibase-infra-dev-200-cloud-migration",
    "omnibase-infra-dev-200-cloud-migration-files",
    "onex-api-dev-200",
}


def _manifest() -> dict[str, Any]:
    return yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))


def _published_port_lines() -> list[str]:
    raw = OVERLAY_PATH.read_text(encoding="utf-8")
    return re.findall(r'^\s+-\s+"([^"]*:\d+:\d+)"\s*$', raw, re.MULTILINE)


def test_dev_200_lane_boundary_is_not_deploy_attribution_governed() -> None:
    """AC2, gate 1. OUT for dev-200; the governed set is unchanged."""
    assert LANE not in GOVERNED_LANES
    assert frozenset({"stability-test", "prod", "judge"}) == GOVERNED_LANES


def test_dev_200_lane_boundary_is_not_in_the_prod_grant_interlock() -> None:
    """AC2, gate 2. No grant resolves anything from this lane."""
    assert LANE not in GRANT_INTERLOCK_LANES
    assert frozenset({"stability-test"}) == GRANT_INTERLOCK_LANES


def test_dev_200_lane_boundary_manifest_declares_lab_200_only() -> None:
    lane = _manifest()["lanes"][LANE]
    assert lane["hosts"] == [HOST_ID]
    assert lane["compose_project"] == COMPOSE_PROJECT
    assert lane["network"] == NETWORK
    assert lane["compose_file"] == "docker/docker-compose.dev-200.yml"
    assert lane["optional"] is True, (
        "declared before it is built; a non-optional entry would file a census "
        "finding for a lane that is correctly absent"
    )


def test_dev_200_lane_boundary_declares_every_dev_service_under_its_name() -> None:
    """AC1. The .201 dev lane's declared set maps one to one onto dev-200."""
    manifest = _manifest()
    dev_declared = {s["name"] for s in manifest["lanes"]["dev"]["services"]}
    assert dev_declared == set(DEV_TO_DEV_200), (
        "the .201 dev lane's declared services changed; map the new one to its "
        f"dev-200 name here and in the overlay. dev declares {sorted(dev_declared)}"
    )
    by_name = {s["name"]: s for s in manifest["lanes"][LANE]["services"]}
    missing = sorted(v for v in DEV_TO_DEV_200.values() if v not in by_name)
    assert not missing, f"dev-200 does not declare {missing}"
    gated = {n for n, s in by_name.items() if s.get("kind") == "profile_gated"}
    assert gated == PROFILE_GATED


def test_dev_200_lane_boundary_declares_no_forbidden_service() -> None:
    names = [s["name"] for s in _manifest()["lanes"][LANE]["services"]]
    offending = [n for n in names for f in FORBIDDEN_SERVICE_FRAGMENTS if f in n]
    assert not offending, f"dev-200 must not declare {offending}"
    raw = OVERLAY_PATH.read_text(encoding="utf-8")
    for service in ("keycloak", "infisical"):
        block = re.search(rf"^  {service}:\n((?:    .*\n)+)", raw, re.MULTILINE)
        assert block is not None, f"the overlay must disable {service}"
        assert 'profiles: !override ["dev-200-disabled"]' in block.group(1)
        assert "container_name" not in block.group(1)


def _bindings() -> dict[str, set[str]]:
    """Host port -> the set of host addresses it is published on."""
    bindings: dict[str, set[str]] = {}
    for line in _published_port_lines():
        host_ip, host_port, _container_port = line.split(":")
        bindings.setdefault(host_port, set()).add(host_ip)
    return bindings


def test_dev_200_lane_boundary_every_port_entry_names_a_host_address() -> None:
    """A port entry without a host address publishes on every interface."""
    raw = OVERLAY_PATH.read_text(encoding="utf-8")
    entries = re.findall(r'^\s+-\s+"([^"]*:\d+)"\s*$', raw, re.MULTILINE)
    assert entries, "positive control: the overlay publishes ports"
    assert sorted(entries) == sorted(_published_port_lines()), (
        "every published entry has the host-address:host-port:container-port form"
    )


def test_dev_200_lane_boundary_every_published_port_binds_loopback_only() -> None:
    bindings = _bindings()
    assert bindings, "positive control: the overlay publishes ports"
    for port, addresses in bindings.items():
        assert addresses == {LOOPBACK}, (
            f"port {port} binds {sorted(addresses)}; a dev-200 port binds only the "
            "loopback address, never all interfaces, a LAN address or a bridge "
            "gateway (Docker Desktop has none on the macOS host)"
        )
    assert set(bindings) == EXPECTED_PUBLISHED_PORTS


def test_dev_200_lane_boundary_block_sits_below_the_macos_ephemeral_range() -> None:
    """dev-202's 61xxx block would fall inside macOS's 49152-65535 ephemeral
    range, where an outbound connection can take the port first."""
    for port in EXPECTED_PUBLISHED_PORTS:
        assert 42000 <= int(port) < 43000 < MACOS_EPHEMERAL_FIRST, port


def test_dev_200_lane_boundary_block_is_clear_of_every_other_lane() -> None:
    """No port of the block is published by any other committed compose file."""
    others: set[str] = set()
    for path in sorted((ROOT / "docker").glob("docker-compose*.yml")):
        if path == OVERLAY_PATH:
            continue
        others.update(re.findall(r"\b(42\d{3})\b", path.read_text(encoding="utf-8")))
    assert not (others & EXPECTED_PUBLISHED_PORTS), sorted(
        others & EXPECTED_PUBLISHED_PORTS
    )


def test_dev_200_lane_boundary_identifiers_cannot_trip_the_no_raw_prod_bypass_matcher() -> (
    None
):
    for governed in GOVERNED_COMPOSE_PROJECTS:
        assert governed not in COMPOSE_PROJECT
    for port in EXPECTED_PUBLISHED_PORTS:
        for literal in GOVERNED_LANE_PORT_LITERALS:
            assert literal not in port


def test_dev_200_lane_boundary_reaches_nothing_on_201() -> None:
    """The only tie to .201 is the deploy agent's control-topic read, not the
    stack's. The overlay names no .201 address, mounts no dev-lane Bifrost
    overlay, and leaves the external omnimemory network."""
    raw = OVERLAY_PATH.read_text(encoding="utf-8")
    body = "\n".join(
        line for line in raw.splitlines() if not line.lstrip().startswith("#")
    )
    assert "192.168." not in body
    assert "./lane-overlays/dev.bifrost.yaml" not in body
    assert "omnimemory-network" not in body
    mount = "./lane-overlays/dev-200.bifrost.yaml:/app/config/delegation/dev-200.bifrost.yaml:ro"
    assert body.count(mount) == 4, (
        "omninode-runtime, runtime-effects, runtime-worker and "
        "tenant-projection-writer each mount the dev-200 overlay"
    )
    assert (
        body.count(
            "BIFROST_LANE_OVERLAY_PATH: /app/config/delegation/dev-200.bifrost.yaml"
        )
        == 4
    )
    overlay = ModelBifrostLaneOverlay.model_validate(
        yaml.safe_load(BIFROST_OVERLAY_PATH.read_text(encoding="utf-8"))
    )
    assert overlay.lane == LANE
    assert overlay.locale is EnumBifrostLaneLocale.CLOUD
    assert overlay.backends == ()


def test_dev_200_lane_boundary_relabels_every_dev_lane_label() -> None:
    """A container on .200 still labelled com.omninode.lane=dev is the .201 dev
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
        elif current and re.match(r"^\s+com\.omninode\.lane: dev-200\s*$", line):
            relabelled.add(current)
    assert labelled_in_dev <= relabelled, sorted(labelled_in_dev - relabelled)


# --- AC3: the census on .200 --------------------------------------------------


def _row(name: str, project: str, *, running: bool = True) -> dict[str, Any]:
    return {
        "Names": name,
        "State": "running" if running else "exited",
        "Status": "Up 2 hours" if running else "Exited (0) 1 hour ago",
        "Image": "onex-lab/omninode-runtime:pinned",
        "Labels": {"com.docker.compose.project": project, "com.omninode.lane": LANE},
    }


def _healthy_dev_200_rows() -> list[dict[str, Any]]:
    lane = _manifest()["lanes"][LANE]
    return [
        _row(s["name"], lane["compose_project"], running=s.get("kind") != "oneshot")
        for s in lane["services"]
        if s.get("kind") != "profile_gated"
    ]


def _plan_on_200(rows: list[dict[str, Any]], networks: list[str]) -> dict[str, Any]:
    envelope = {
        "host": HOSTNAME_ON_HOST,
        "lane": None,
        "containers": rows,
        "networks": networks,
        "runtime_tag": None,
    }
    return build_plan(envelope, _manifest())


def test_dev_200_lane_boundary_host_resolves_and_carries_dev_200() -> None:
    manifest = _manifest()
    validate_hosts(manifest)
    assert resolve_host(HOSTNAME_ON_HOST, manifest) == HOST_ID
    lanes_on_host = sorted(
        name for name, spec in manifest["lanes"].items() if HOST_ID in spec["hosts"]
    )
    assert lanes_on_host == ["dev-200", "dogfood"]


def test_dev_200_lane_boundary_census_on_200_reads_clean() -> None:
    plan = _plan_on_200(_healthy_dev_200_rows(), [NETWORK])
    manifest = _manifest()
    assert plan["host"] == HOST_ID
    assert sorted(plan["lanes_checked"]) == ["dev-200", "dogfood"]
    assert set(plan["lanes_not_applicable"]) == set(manifest["lanes"]) - {
        "dev-200",
        "dogfood",
    }
    assert plan["findings"] == []
    assert plan["has_drift"] is False


def test_dev_200_lane_boundary_census_still_reports_a_stopped_runtime() -> None:
    """Negative control: a real outage on dev-200 is still named."""
    rows = [
        r for r in _healthy_dev_200_rows() if r["Names"] != "omninode-dev-200-runtime"
    ]
    plan = _plan_on_200(rows, [NETWORK])
    assert plan["has_drift"] is True
    assert any(
        f["lane"] == LANE and f.get("container") == "omninode-dev-200-runtime"
        for f in plan["findings"]
    ), plan["findings"]


def test_dev_200_lane_boundary_a_201_dev_container_on_200_is_a_finding() -> None:
    rows = [
        *_healthy_dev_200_rows(),
        {
            **_row("omninode-runtime", "omnibase-infra"),
            "Labels": {"com.docker.compose.project": "omnibase-infra"},
        },
    ]
    plan = _plan_on_200(rows, [NETWORK])
    assert any(
        f["lane"] == "dev" and f["kind"] == "lane_on_undeclared_host"
        for f in plan["findings"]
    ), plan["findings"]
