# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19544 — the dev-105 lane is ungoverned, isolated, and on .105 only.

dev-105 is the .105 row of the lab deploy lanes plan (epic OMN-19500): the
dev-105 lane's overlay (OMN-19505) moved to the .105 laptop, with its own
Postgres, Redpanda and Valkey, so a deploy-agent instance there can run one more
runtime deploy slot, time-shared with the prove-105 proof stack. The same
properties as dev-105, with one difference that comes from the host:

* **No governance acquired, none eroded.** Absent from ``GOVERNED_LANES`` and
  ``GRANT_INTERLOCK_LANES``, which still hold exactly what they held.
* **Declared for .105 and nowhere else.** The manifest names ``lab-105`` as its
  only host, so the census on .105 evaluates it beside dogfood.
* **Loopback only, including the verify ports.** .105 runs Docker Desktop,
  where a runner container's ``host.docker.internal`` reaches a publish on the
  host's loopback (measured on .105 2026-09-25, ledger RELEASE 10:34:52Z, lane
  dev-105-lane), so unlike dev-105 no port is published on the Linux docker
  bridge gateway, an address that does not exist on the macOS host.
* **Every service the .201 dev lane declares, under a dev-105 name**, and no
  Keycloak or Infisical.
* **A broker sized for the host.** The overlay's Redpanda default memory is 2G,
  not dev-105's 4G, and the dev lane's disk dials (OMN-19082) are inherited, not
  overridden.
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
OVERLAY_PATH = ROOT / "docker" / "docker-compose.dev-105.yml"
BIFROST_OVERLAY_PATH = ROOT / "docker" / "lane-overlays" / "dev-105.bifrost.yaml"

LANE = "dev-105"
HOST_ID = "lab-105"
HOSTNAME_ON_HOST = "omnibook"
COMPOSE_PROJECT = "omnibase-infra-dev-105"
NETWORK = "omnibase-infra-dev-105-network"

#: The declared block. 61090 (onex-api) is published by a disabled service, so
#: it is reserved and never bound; it is still in the block.
EXPECTED_PUBLISHED_PORTS = {
    "43085",  # runtime main
    "43086",  # runtime effects
    "43302",  # projection API (43002 is the judge lane's default)
    "43087",  # agent actions consumer
    "43091",  # contract resolver
    "43092",  # skill lifecycle consumer
    "43093",  # context audit consumer
    "43053",  # intelligence API
    "43006",  # phoenix
    "43090",  # onex-api (disabled here, reserved)
    "43436",  # Postgres
    "43379",  # Valkey
    "43192",  # Redpanda Kafka (external listener)
    "43182",  # Redpanda proxy
    "43181",  # schema registry
    "43644",  # Redpanda admin
}

#: Mirrors omni_home's no-raw-prod-bypass scanner, as the dev-202 boundary test
#: does. Restated, not imported: it lives in another repository.
GOVERNED_COMPOSE_PROJECTS = (
    "omnibase-infra-prod",
    "omnibase-infra-stability-test",
    "omnibase-infra-judge",
)
GOVERNED_LANE_PORT_LITERALS = ("28085", "28086", "18085", "18086")

LOOPBACK = "127.0.0.1"

#: The Linux docker bridge gateway dev-202 publishes its verify ports on
#: (infra#4109). It does not exist on a Docker Desktop host, so a publish on it
#: would fail at ``up`` on .105; it must not appear.
LINUX_DOCKER_BRIDGE_GATEWAY = "172.17.0.1"

#: The ports a fleet verify job reads through ``host.docker.internal``: runtime
#: main, effects and the projection API. On .105 they are loopback publishes.
VERIFY_READ_PORTS = {"43085", "43086", "43302"}

#: The dev-202 overlay's broker memory default, and this lane's.
DEV_202_REDPANDA_MEMORY_DEFAULT = "${DEV_202_REDPANDA_MEMORY:-4G}"
REDPANDA_MEMORY_DEFAULT = "${DEV_105_REDPANDA_MEMORY:-2G}"

#: Not run on .105 (no identity or secret store on a lab laptop).
FORBIDDEN_SERVICE_FRAGMENTS = ("keycloak", "infisical")

#: The .201 dev lane's declared service -> its dev-105 name. AC1.
DEV_TO_DEV_105 = {
    "omninode-runtime": "omninode-dev-105-runtime",
    "omninode-runtime-effects": "omninode-dev-105-runtime-effects",
    "omnimarket-projection-tenant-registry-writer": "omnimarket-dev-105-projection-tenant-registry-writer",
    "omnimarket-projection-delegation-writer": "omnimarket-dev-105-projection-delegation-writer",
    "omnimarket-projection-registration-writer": "omnimarket-dev-105-projection-registration-writer",
    "omnimarket-projection-savings-writer": "omnimarket-dev-105-projection-savings-writer",
    "omnimarket-projection-tenant-credentials-writer": "omnimarket-dev-105-projection-tenant-credentials-writer",
    "omnimarket-projection-live-events-writer": "omnimarket-dev-105-projection-live-events-writer",
    "omnimarket-tenant-projection-writer": "omnimarket-dev-105-tenant-projection-writer",
    "omnibase-infra-forward-migration": "omnibase-infra-dev-105-forward-migration",
    "omnibase-infra-cloud-migration": "omnibase-infra-dev-105-cloud-migration",
    "omnibase-infra-cloud-migration-files": "omnibase-infra-dev-105-cloud-migration-files",
    "onex-api": "onex-api-dev-105",
    "omninode-infra-routing-decisions-consumer": "omninode-dev-105-infra-routing-decisions-consumer",
}

#: Declared but disabled on .105: the cloud image only the .201 lab-overlay pin
#: delivers.
PROFILE_GATED = {
    "omnibase-infra-dev-105-cloud-migration",
    "omnibase-infra-dev-105-cloud-migration-files",
    "onex-api-dev-105",
}


def _manifest() -> dict[str, Any]:
    return yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))


def _published_port_lines() -> list[str]:
    raw = OVERLAY_PATH.read_text(encoding="utf-8")
    return re.findall(r'^\s+-\s+"([^"]*:\d+:\d+)"\s*$', raw, re.MULTILINE)


def test_dev_105_lane_boundary_is_not_deploy_attribution_governed() -> None:
    """AC2, gate 1. OUT for dev-105; the governed set is unchanged."""
    assert LANE not in GOVERNED_LANES
    assert frozenset({"stability-test", "prod", "judge"}) == GOVERNED_LANES


def test_dev_105_lane_boundary_is_not_in_the_prod_grant_interlock() -> None:
    """AC2, gate 2. No grant resolves anything from this lane."""
    assert LANE not in GRANT_INTERLOCK_LANES
    assert frozenset({"stability-test"}) == GRANT_INTERLOCK_LANES


def test_dev_105_lane_boundary_manifest_declares_lab_105_only() -> None:
    lane = _manifest()["lanes"][LANE]
    assert lane["hosts"] == [HOST_ID]
    assert lane["compose_project"] == COMPOSE_PROJECT
    assert lane["network"] == NETWORK
    assert lane["compose_file"] == "docker/docker-compose.dev-105.yml"
    assert lane["optional"] is True, (
        "declared before it is built; a non-optional entry would file a census "
        "finding for a lane that is correctly absent"
    )


def test_dev_105_lane_boundary_declares_every_dev_service_under_its_name() -> None:
    """AC1. The .201 dev lane's declared set maps one to one onto dev-105."""
    manifest = _manifest()
    dev_declared = {s["name"] for s in manifest["lanes"]["dev"]["services"]}
    assert dev_declared == set(DEV_TO_DEV_105), (
        "the .201 dev lane's declared services changed; map the new one to its "
        f"dev-105 name here and in the overlay. dev declares {sorted(dev_declared)}"
    )
    by_name = {s["name"]: s for s in manifest["lanes"][LANE]["services"]}
    missing = sorted(v for v in DEV_TO_DEV_105.values() if v not in by_name)
    assert not missing, f"dev-105 does not declare {missing}"
    gated = {n for n, s in by_name.items() if s.get("kind") == "profile_gated"}
    assert gated == PROFILE_GATED


def test_dev_105_lane_boundary_declares_no_forbidden_service() -> None:
    names = [s["name"] for s in _manifest()["lanes"][LANE]["services"]]
    offending = [n for n in names for f in FORBIDDEN_SERVICE_FRAGMENTS if f in n]
    assert not offending, f"dev-105 must not declare {offending}"
    raw = OVERLAY_PATH.read_text(encoding="utf-8")
    for service in ("keycloak", "infisical"):
        block = re.search(rf"^  {service}:\n((?:    .*\n)+)", raw, re.MULTILINE)
        assert block is not None, f"the overlay must disable {service}"
        assert 'profiles: !override ["dev-105-disabled"]' in block.group(1)
        assert "container_name" not in block.group(1)


def _bindings() -> dict[str, set[str]]:
    """Host port -> the set of host addresses it is published on."""
    bindings: dict[str, set[str]] = {}
    for line in _published_port_lines():
        host_ip, host_port, _container_port = line.split(":")
        bindings.setdefault(host_port, set()).add(host_ip)
    return bindings


def test_dev_105_lane_boundary_every_port_entry_names_a_host_address() -> None:
    """A port entry without a host address publishes on every interface."""
    raw = OVERLAY_PATH.read_text(encoding="utf-8")
    entries = re.findall(r'^\s+-\s+"([^"]*:\d+)"\s*$', raw, re.MULTILINE)
    assert entries, "positive control: the overlay publishes ports"
    assert sorted(entries) == sorted(_published_port_lines()), (
        "every published entry has the host-address:host-port:container-port form"
    )


def test_dev_105_lane_boundary_every_published_port_binds_loopback_only() -> None:
    bindings = _bindings()
    assert bindings, "positive control: the overlay publishes ports"
    for port, addresses in bindings.items():
        assert addresses == {LOOPBACK}, (
            f"port {port} binds {sorted(addresses)}; a dev-105 port binds only the "
            "loopback address, never all interfaces, a LAN address or the Linux "
            "docker bridge gateway"
        )
    assert set(bindings) == EXPECTED_PUBLISHED_PORTS


def test_dev_105_lane_boundary_verify_ports_are_loopback_publishes() -> None:
    """A runner container on Docker Desktop reaches a loopback publish through
    host.docker.internal (measured on .105 2026-09-25), so the verify job's
    three ports are published on loopback, and nothing names the Linux bridge
    gateway, which the macOS host does not have."""
    bindings = _bindings()
    assert set(bindings) >= VERIFY_READ_PORTS
    for port in VERIFY_READ_PORTS:
        assert bindings[port] == {LOOPBACK}
    raw = OVERLAY_PATH.read_text(encoding="utf-8")
    body = "\n".join(
        line for line in raw.splitlines() if not line.lstrip().startswith("#")
    )
    assert LINUX_DOCKER_BRIDGE_GATEWAY not in body
    dev_202 = (ROOT / "docker" / "docker-compose.dev-202.yml").read_text(
        encoding="utf-8"
    )
    assert f'"{LINUX_DOCKER_BRIDGE_GATEWAY}:61085:8085"' in dev_202, (
        "positive control: the dev-202 overlay does publish on the gateway"
    )


def test_dev_105_lane_boundary_broker_is_sized_for_the_host() -> None:
    """The .105 VM is 15.6 GiB with dogfood holding about 4.1 GiB, so the broker
    takes 2G; the disk dials come from the dev lane's partition-cap override
    (OMN-19082), which this overlay renames and never replaces."""
    raw = OVERLAY_PATH.read_text(encoding="utf-8")
    assert f"- {REDPANDA_MEMORY_DEFAULT}" in raw
    dev_202 = (ROOT / "docker" / "docker-compose.dev-202.yml").read_text(
        encoding="utf-8"
    )
    assert f"- {DEV_202_REDPANDA_MEMORY_DEFAULT}" in dev_202, "positive control"
    block = re.search(r"^  redpanda-partition-cap:\n((?:    .*\n)+)", raw, re.MULTILINE)
    assert block is not None
    assert block.group(1).strip() == (
        "container_name: omnibase-infra-dev-105-redpanda-partition-cap"
    ), "the overlay renames the one-shot and keeps the dev lane's disk dials"
    dev_lane = (ROOT / "docker" / "docker-compose.dev-lane.yml").read_text(
        encoding="utf-8"
    )
    assert "segment_fallocation_step 1048576" in dev_lane


def test_dev_105_lane_boundary_block_is_clear_of_every_other_lane() -> None:
    """No port of the block is published by any other committed compose file."""
    others: set[str] = set()
    for path in sorted((ROOT / "docker").glob("docker-compose*.yml")):
        if path == OVERLAY_PATH:
            continue
        others.update(re.findall(r"\b(43\d{3})\b", path.read_text(encoding="utf-8")))
    assert not (others & EXPECTED_PUBLISHED_PORTS), sorted(
        others & EXPECTED_PUBLISHED_PORTS
    )


def test_dev_105_lane_boundary_identifiers_cannot_trip_the_no_raw_prod_bypass_matcher() -> (
    None
):
    for governed in GOVERNED_COMPOSE_PROJECTS:
        assert governed not in COMPOSE_PROJECT
    for port in EXPECTED_PUBLISHED_PORTS:
        for literal in GOVERNED_LANE_PORT_LITERALS:
            assert literal not in port


def test_dev_105_lane_boundary_reaches_nothing_on_201() -> None:
    """The only tie to .201 is the deploy agent's control-topic read, not the
    stack's. The overlay names no .201 address, mounts no dev-lane Bifrost
    overlay, and leaves the external omnimemory network .105 does not have."""
    raw = OVERLAY_PATH.read_text(encoding="utf-8")
    body = "\n".join(
        line for line in raw.splitlines() if not line.lstrip().startswith("#")
    )
    assert "192.168." not in body
    assert "./lane-overlays/dev.bifrost.yaml" not in body
    assert "omnimemory-network" not in body
    mount = "./lane-overlays/dev-105.bifrost.yaml:/app/config/delegation/dev-105.bifrost.yaml:ro"
    assert body.count(mount) == 4, (
        "omninode-runtime, runtime-effects, runtime-worker and "
        "tenant-projection-writer each mount the dev-105 overlay"
    )
    assert (
        body.count(
            "BIFROST_LANE_OVERLAY_PATH: /app/config/delegation/dev-105.bifrost.yaml"
        )
        == 4
    )
    overlay = ModelBifrostLaneOverlay.model_validate(
        yaml.safe_load(BIFROST_OVERLAY_PATH.read_text(encoding="utf-8"))
    )
    assert overlay.lane == LANE
    assert overlay.locale is EnumBifrostLaneLocale.CLOUD
    assert overlay.backends == ()


def test_dev_105_lane_boundary_relabels_every_dev_lane_label() -> None:
    """A container on .105 still labelled com.omninode.lane=dev is the .201 dev
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
        elif current and re.match(r"^\s+com\.omninode\.lane: dev-105\s*$", line):
            relabelled.add(current)
    assert labelled_in_dev <= relabelled, sorted(labelled_in_dev - relabelled)


# --- the census on .105 -------------------------------------------------------


def _row(name: str, project: str, *, running: bool = True) -> dict[str, Any]:
    return {
        "Names": name,
        "State": "running" if running else "exited",
        "Status": "Up 2 hours" if running else "Exited (0) 1 hour ago",
        "Image": "onex-lab/omninode-runtime:pinned",
        "Labels": {"com.docker.compose.project": project, "com.omninode.lane": LANE},
    }


def _healthy_dev_105_rows() -> list[dict[str, Any]]:
    lane = _manifest()["lanes"][LANE]
    return [
        _row(s["name"], lane["compose_project"], running=s.get("kind") != "oneshot")
        for s in lane["services"]
        if s.get("kind") != "profile_gated"
    ]


def _plan_on_105(rows: list[dict[str, Any]], networks: list[str]) -> dict[str, Any]:
    envelope = {
        "host": HOSTNAME_ON_HOST,
        "lane": None,
        "containers": rows,
        "networks": networks,
        "runtime_tag": None,
    }
    return build_plan(envelope, _manifest())


def test_dev_105_lane_boundary_host_resolves_and_carries_dev_105() -> None:
    manifest = _manifest()
    validate_hosts(manifest)
    assert resolve_host(HOSTNAME_ON_HOST, manifest) == HOST_ID
    lanes_on_host = sorted(
        name for name, spec in manifest["lanes"].items() if HOST_ID in spec["hosts"]
    )
    assert lanes_on_host == ["dev-105", "dogfood"]


def test_dev_105_lane_boundary_census_on_105_reads_clean() -> None:
    plan = _plan_on_105(_healthy_dev_105_rows(), [NETWORK])
    manifest = _manifest()
    assert plan["host"] == HOST_ID
    assert sorted(plan["lanes_checked"]) == ["dev-105", "dogfood"]
    assert set(plan["lanes_not_applicable"]) == set(manifest["lanes"]) - {
        "dev-105",
        "dogfood",
    }
    assert plan["findings"] == []
    assert plan["has_drift"] is False


def test_dev_105_lane_boundary_census_still_reports_a_stopped_runtime() -> None:
    """Negative control: a real outage on dev-105 is still named."""
    rows = [
        r for r in _healthy_dev_105_rows() if r["Names"] != "omninode-dev-105-runtime"
    ]
    plan = _plan_on_105(rows, [NETWORK])
    assert plan["has_drift"] is True
    assert any(
        f["lane"] == LANE and f.get("container") == "omninode-dev-105-runtime"
        for f in plan["findings"]
    ), plan["findings"]


def test_dev_105_lane_boundary_a_201_dev_container_on_105_is_a_finding() -> None:
    rows = [
        *_healthy_dev_105_rows(),
        {
            **_row("omninode-runtime", "omnibase-infra"),
            "Labels": {"com.docker.compose.project": "omnibase-infra"},
        },
    ]
    plan = _plan_on_105(rows, [NETWORK])
    assert any(
        f["lane"] == "dev" and f["kind"] == "lane_on_undeclared_host"
        for f in plan["findings"]
    ), plan["findings"]
