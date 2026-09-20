# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18890 — the ephemeral pre-PR verify pool, declared before it is built.

Task 1 of epic OMN-18888. Two pool slots are declared in the desired-state lane
census BEFORE anything is stood up, following the `lakshman` reservation
precedent: the compose project names, the network the slots must attach to and
the reserved port block are claimed in the authority first, so a slot that later
appears on the host is reconciled rather than invisible. A running-but-undeclared
container is the dangerous direction (retro B-6 / OMN-13034).

What these tests pin, and why each one is a real invariant rather than a
restatement of the manifest:

* **Absent is silent, present is reconciled.** A pool slot only exists for the
  minutes a branch is under verification, so the steady state is ABSENT. That is
  `optional: true` at ``scripts/lane_census_plan.py:157``. The pair of tests
  below asserts BOTH halves, because either alone is a trap: an entry that is
  never silent files a census ticket every tick on a pool nobody has built, and
  an entry that is always silent is a lane the census cannot police at all.
* **The zero is proven with a positive control.** A planner run that errored
  would also report zero findings for these lanes. Every zero asserted here is
  taken from a run that simultaneously returns rows for another lane.
* **The pool acquires no governance and erodes none.** A slot is a premise for
  nothing: it promotes nothing, no other lane reads it, and it must never appear
  in a promotion's evidence chain. It therefore stays out of
  ``GOVERNED_LANES`` and ``GRANT_INTERLOCK_LANES`` — and stability-test must
  still be in both. Both directions are asserted, for the reason the collaborator
  lane's boundary test gives: widening a lane set is exactly the edit that
  silently drops an existing member.

Slot 1's runtime ports, 28085/28086, are the RETIRED lab `prod` lane's old ports
(OMN-18320 shut that lane down on 2026-09-13, which is why they are free). They
are still the literals in omni_home's no-raw-prod-bypass matcher `_PROD_PORTS`.
That matcher pairs a port literal with a ``docker tag``/``docker commit`` verb
only — its recreate arm keys on the compose PROJECT name — so a pool bring-up
recipe cannot trip it, and ``test_pool_project_names_trip_no_governed_matcher``
pins the project-name half of that here. The port half is a fact about a file in
another repository and is recorded in this module's docstring rather than
asserted across a repo boundary.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.preflight_lane_deploy_attribution import (
    GOVERNED_LANES,
    GRANT_INTERLOCK_LANES,
)

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_MANIFEST_PATH = _REPO / "deploy" / "lane-census" / "lane-manifest.yaml"
_PLAN_PATH = _REPO / "scripts" / "lane_census_plan.py"
_GENERATOR_PATH = _REPO / "scripts" / "generate_claude_lane_block.py"

POOL_LANES = ("prepr-1", "prepr-2")

#: Every port the pool reserves, slot by slot. Sourced from section 3.3 of the
#: settled plan, which enumerated the host's listening sockets live on
#: 2026-09-20T12:42Z and picked from what was free.
POOL_PORTS: dict[str, dict[str, str]] = {
    "prepr-1": {
        "main": "28085",
        "effects": "28086",
        "gateway": "28090",
        "projection_api": "23002",
    },
    "prepr-2": {
        "main": "38085",
        "effects": "38086",
        "gateway": "38090",
        "projection_api": "33002",
    },
}


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PLAN = _load_module(_PLAN_PATH, "lane_census_plan")
GENERATOR = _load_module(_GENERATOR_PATH, "generate_claude_lane_block")
MANIFEST = PLAN.load_manifest(_MANIFEST_PATH)


def _container(name: str, *, running: bool = True) -> dict[str, str]:
    return {
        "Names": name,
        "State": "running" if running else "exited",
        "Status": "Up 3 minutes" if running else "Exited (0) 1 minute ago",
        "Image": "onex-lab/omnicloud-core:branch-head",
        "Labels": "com.docker.compose.project=omnibase-infra-prepr-1",
    }


def _envelope(containers: list[dict[str, str]]) -> dict[str, Any]:
    """An inventory that always carries a positive control.

    ``omnibase-infra-decoy`` is labelled for the ``dev`` lane and declared by no
    manifest entry, so every run built from this envelope returns at least one
    ``unexpected_container`` finding. A zero for the pool lanes read off a run
    that also returns that row is a measured zero; a zero read off a run that
    returns nothing at all proves only that the run produced nothing.
    """
    decoy = {
        "Names": "omnibase-infra-decoy",
        "State": "running",
        "Status": "Up 3 hours",
        "Image": "onex-lab/omnicloud-core:dev",
        "Labels": {"com.omninode.lane": "dev"},
    }
    return {
        "lane": None,
        "containers": [decoy, *containers],
        "networks": ["omnibase-infra-network", "omnibase-infra_default"],
        "runtime_tag": None,
    }


def _findings_for(plan: dict[str, Any], lane: str) -> list[dict[str, str]]:
    return [f for f in plan["findings"] if f["lane"] == lane]


@pytest.mark.parametrize("lane", POOL_LANES)
def test_pool_slot_is_declared_with_its_project_and_network(lane: str) -> None:
    """Each slot claims its compose project and the network it must attach to."""
    spec = MANIFEST["lanes"][lane]
    assert spec["compose_project"] == f"omnibase-infra-{lane}"
    # A slot runs no Postgres, Redpanda or Valkey of its own: it reaches the dev
    # lane's dependency SERVERS by service hostname, which is only possible from
    # that lane's network. Isolation is by namespace on every other axis.
    assert spec["network"] == "omnibase-infra-network"
    assert spec["optional"] is True, (
        f"lane {lane!r} must be optional: a pool slot's steady state is ABSENT, "
        f"and a non-optional entry would file a census ticket on every tick"
    )


@pytest.mark.parametrize("lane", POOL_LANES)
def test_absent_pool_slot_is_named_and_reports_zero_drift(lane: str) -> None:
    """The slot is visible to the census while absent, and is not drift."""
    plan = PLAN.build_plan(_envelope([]), MANIFEST)
    assert lane in plan["lanes_checked"], (
        f"lane {lane!r} must appear in lanes_checked even while absent — a slot "
        f"the census cannot name is a slot it cannot police"
    )
    assert _findings_for(plan, lane) == []
    # Positive control: the same run returns rows for a lane that is NOT in the
    # pool, so the zero above is a measured zero rather than an errored-and-empty
    # read. `dev` cannot serve as that control — it is optional and entirely down
    # in this envelope, so the planner skips it for the same reason it skips the
    # pool, which would make the control and the subject the same observation.
    assert _findings_for(plan, "stability-test"), (
        "positive control failed: the planner returned no findings for the "
        "non-optional control lane, so the pool's zero proves nothing"
    )


@pytest.mark.parametrize("lane", POOL_LANES)
def test_running_pool_slot_reports_its_inventory(lane: str) -> None:
    """A partially-up slot is reconciled normally: absences become findings."""
    spec = MANIFEST["lanes"][lane]
    services = [s for s in spec["services"] if s.get("kind", "service") == "service"]
    assert len(services) >= 2, "a slot must declare more than one service"
    first, *rest = services

    plan = PLAN.build_plan(_envelope([_container(first["name"])]), MANIFEST)
    findings = _findings_for(plan, lane)
    absent = {f["container"] for f in findings if f["kind"] == "container_absent"}
    assert absent == {s["name"] for s in rest}, (
        f"lane {lane!r}: a slot with one service up must report every OTHER "
        f"declared service absent; got {sorted(absent)}"
    )
    assert first["name"] not in absent


@pytest.mark.parametrize("lane", POOL_LANES)
def test_pool_slot_declares_the_runtime_family_and_a_worker(lane: str) -> None:
    """The worker is declared, so the silent-zero class cannot recur here.

    OMN-12988 scaled a worker to 0 and nothing noticed. A slot runs one (the
    plan's per-slot memory budget counts it at 432 MiB), so it is declared
    ``kind: service, replicas: 1`` rather than left to the compose file.
    """
    names = {s["name"] for s in MANIFEST["lanes"][lane]["services"]}
    slot = lane  # `prepr-1` / `prepr-2` is the container-name infix
    assert f"omninode-{slot}-runtime" in names
    assert f"omninode-{slot}-runtime-effects" in names
    assert f"omninode-{slot}-runtime-worker" in names
    assert f"omnimarket-{slot}-projection-api" in names


@pytest.mark.parametrize("lane", POOL_LANES)
def test_pool_slot_declares_no_dependency_server(lane: str) -> None:
    """A slot reuses the shared servers; declaring one would be a second copy.

    The whole design reuses the dev lane's Postgres, Redpanda, Valkey and
    Keycloak and isolates by namespace. A slot that declared its own would both
    double the resident cost the pool was sized against and make the census
    expect a container no sanctioned bring-up creates.
    """
    names = " ".join(s["name"] for s in MANIFEST["lanes"][lane]["services"])
    for dependency in ("postgres", "redpanda", "valkey", "keycloak"):
        assert dependency not in names, (
            f"lane {lane!r} declares a {dependency} container: a pool slot "
            f"reuses the dev lane's dependency servers and runs none of its own"
        )


def test_pool_ports_are_declared_and_collide_with_nothing() -> None:
    """The reserved block is in the manifest and overlaps no other lane's ports.

    The failure this prevents is the OMN-13581 cross-lane displacement class: a
    slot picking a port ad hoc and taking it from a governed lane.
    """
    raw = _MANIFEST_PATH.read_text(encoding="utf-8")
    for lane, ports in POOL_PORTS.items():
        for role, port in ports.items():
            assert re.search(rf"^#.*\b{port}\b", raw, re.MULTILINE), (
                f"port {port} ({lane} {role}) is not declared in the lane "
                f"manifest's reserved block"
            )

    all_ports = [p for ports in POOL_PORTS.values() for p in ports.values()]
    assert len(set(all_ports)) == len(all_ports), "the two slots overlap"

    # No pool port may be a port another DECLARED lane publishes. The generator's
    # port map is the machine-readable side of that claim, read here for lanes
    # the manifest actually declares — a row for a lane that no longer exists
    # cannot displace anything, and OMN-18890 removed the one such row (`prod`,
    # retired 2026-09-13) rather than leave two lanes claiming 28085.
    other_lane_ports = {
        port
        for lane, ports in GENERATOR._LANE_PORT_MAP.items()
        if lane not in POOL_LANES and lane in MANIFEST["lanes"]
        for port in ports.values()
        if port != "—"
    }
    assert other_lane_ports, "no other lane declares a port: the check is vacuous"
    assert not other_lane_ports.intersection(all_ports), (
        f"pool ports collide with another lane: "
        f"{sorted(other_lane_ports.intersection(all_ports))}"
    )


def test_pool_slots_are_not_governed_and_stability_test_still_is() -> None:
    """Written in both directions, because widening a set is how a member drops."""
    for lane in POOL_LANES:
        assert lane not in GOVERNED_LANES, (
            f"{lane!r} must not be a governed lane: a pool slot promotes nothing "
            f"and must never appear in a promotion's evidence chain"
        )
        assert lane not in GRANT_INTERLOCK_LANES
    assert "stability-test" in GOVERNED_LANES
    assert "stability-test" in GRANT_INTERLOCK_LANES
    assert "judge" in GOVERNED_LANES


def test_pool_project_names_trip_no_governed_matcher() -> None:
    """No governed compose project name is a substring of a pool project name.

    omni_home's no-raw-prod-bypass scanner classifies a recreate line as a
    governed-lane mutation when the line CONTAINS a governed project name. Every
    lane on this host shares the ``omnibase-infra`` prefix, so this is a real
    property of the names chosen here and not a tautology.
    """
    governed_projects = (
        "omnibase-infra-prod",
        "omnibase-infra-stability-test",
        "omnibase-infra-judge",
    )
    for lane in POOL_LANES:
        project = MANIFEST["lanes"][lane]["compose_project"]
        for governed in governed_projects:
            assert governed not in project, (
                f"pool project {project!r} contains governed project name "
                f"{governed!r} — a bring-up recipe naming it would be classified "
                f"a governed-lane mutation"
            )


def test_generated_lane_block_carries_both_slots_with_their_ports() -> None:
    """The generated table states the pool truthfully, ports and boundary."""
    snapshot = GENERATOR._load_snapshot(
        _REPO / "deploy" / "lane-census" / "census-snapshot.json"
    )
    block = GENERATOR.generate_block(MANIFEST, snapshot)
    for lane, ports in POOL_PORTS.items():
        assert f"| {lane} (optional) |" in block, (
            f"lane {lane!r} is missing from the generated lane table"
        )
        row = next(line for line in block.splitlines() if line.startswith(f"| {lane} "))
        assert f"`{ports['main']}`" in row
        assert f"`{ports['effects']}`" in row
        assert "never" in row.lower(), (
            f"lane {lane!r}'s boundary must state what it may never be used for"
        )


@pytest.mark.parametrize("lane", POOL_LANES)
def test_absent_pool_slot_is_reported_skipped_not_clean(lane: str) -> None:
    """Zero findings must be distinguishable from reconciled-and-clean.

    OMN-18890. An empty pool is the pool working, and it produces exactly the
    same empty findings list as a lane the census checked and found healthy.
    Without a signal the two are one observation, and the generated lane table
    read the wrong one: it rendered an optional lane that was entirely down as
    ``N running (census clean)``. The pool makes that the normal case, so the
    planner reports the skip and the generator reads it.
    """
    plan = PLAN.build_plan(_envelope([]), MANIFEST)
    assert lane in plan["lanes_skipped_optional_down"]
    assert set(plan["lanes_skipped_optional_down"]).issubset(
        set(plan["lanes_checked"])
    ), "a skipped lane must still be a checked lane"
    # Non-optional lanes are never in this list, whatever their state.
    assert "stability-test" not in plan["lanes_skipped_optional_down"]

    # And the generated table says absent rather than running.
    snapshot = {
        "lanes_checked": plan["lanes_checked"],
        "lanes_skipped_optional_down": plan["lanes_skipped_optional_down"],
        "findings": plan["findings"],
        "emitted_at": "2026-09-20T00:00:00+00:00",
    }
    block = GENERATOR.generate_block(MANIFEST, snapshot)
    row = next(line for line in block.splitlines() if line.startswith(f"| {lane} "))
    assert "0 running — optional lane down" in row, row
    assert "census clean" not in row


@pytest.mark.parametrize("lane", POOL_LANES)
def test_partially_up_pool_slot_is_not_reported_skipped(lane: str) -> None:
    """The skip signal must not swallow a half-torn-down slot."""
    first = next(
        s
        for s in MANIFEST["lanes"][lane]["services"]
        if s.get("kind", "service") == "service"
    )
    plan = PLAN.build_plan(_envelope([_container(first["name"])]), MANIFEST)
    assert lane not in plan["lanes_skipped_optional_down"]
    assert _findings_for(plan, lane), "a partially-up slot must report drift"


def test_pool_lanes_are_absent_from_the_compose_parity_ratchet() -> None:
    """Until Task 4 lands a compose file, the parity ratchet must not cover them.

    The ratchet diffs a lane's declared services against ``container_name``
    values scraped from ``docker/docker-compose.<lane>.yml``. There is no such
    file for a pool slot yet, so including one would fail on a missing file
    rather than on drift. The pairing is the same one the ``lakshman``
    reservation made: the compose file and the ratchet entry land together.
    """
    parity = (
        _REPO / "tests" / "unit" / "scripts" / "test_lane_census_manifest_parity.py"
    ).read_text(encoding="utf-8")
    compose_lanes = re.search(r"_COMPOSE_LANES = \(([^)]*)\)", parity)
    assert compose_lanes
    for lane in POOL_LANES:
        compose_file = _REPO / "docker" / f"docker-compose.{lane}.yml"
        if compose_file.exists():
            assert f'"{lane}"' in compose_lanes.group(1), (
                f"docker-compose.{lane}.yml now exists, so {lane!r} must join "
                f"_COMPOSE_LANES in the same change"
            )
        else:
            assert f'"{lane}"' not in compose_lanes.group(1)
