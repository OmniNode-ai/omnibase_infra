# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-17150 — the collaborator lane is ungoverned, and stays that way.

Adding a fifth runtime lane means widening ``RuntimeProfileName``, a Literal the
four production lanes all validate against. The risk that widening creates is
not that the new lane fails to work; it is that the new lane silently *acquires*
or silently *erodes* governance:

* **Acquired governance** — someone later adds ``lakshman`` to
  :data:`~scripts.preflight_lane_deploy_attribution.GOVERNED_LANES`, or to the
  grant interlock, or to omni_home's ``no-raw-prod-bypass`` matcher, on the
  reasoning that "it is a .201 lane like the others". It is not. It is a
  fully-mutable sandbox owned by one external collaborator, and gating it would
  push its owner off the sanctioned path for zero governance gain — the same
  reasoning that keeps the ``dev`` lane out of all three.

* **Eroded governance** — the far worse direction. Widening a shared lane type
  is exactly the kind of edit that quietly drops an existing lane out of a
  frozenset. Every assertion below is therefore written in BOTH directions: the
  collaborator lane must be OUT, and stability-test / prod / judge must still be
  IN.

The third gate — ``no-raw-prod-bypass``
---------------------------------------
That gate's scanner lives in the **omni_home** repo
(``tests/test_no_raw_prod_bypass_policy.py``, OMN-13434 / OMN-15243) and cannot
be imported from here. What CAN be pinned here is the property the scanner's
matcher actually depends on, which is a fact about identifiers owned by THIS
repo: the scanner classifies a line as a governed-lane mutation when the line
contains one of the governed compose project names as a substring, or one of the
prod/stability lane port literals. So the checks below assert that this repo's
collaborator-lane identifiers can never satisfy that matcher — no governed
project name is a substring of ``omnibase-infra-lakshman``, and no governed port
literal is a substring of any port this lane publishes.

That is a real invariant, not a restatement: ``omnibase-infra-lakshman`` shares
the ``omnibase-infra`` prefix with every lane on the host, and a differently
chosen port (say ``18085``-adjacent) genuinely would have tripped the substring
matcher. Both are checked rather than assumed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from omnibase_infra.runtime.models.model_runtime_policy_contract import (
    ModelRuntimePolicyContract,
)
from scripts.preflight_lane_deploy_attribution import (
    GOVERNED_LANES,
    GRANT_INTERLOCK_LANES,
)

pytestmark = pytest.mark.ci

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "contracts" / "services" / "runtime_policy.contract.yaml"
MANIFEST_PATH = ROOT / "deploy" / "lane-census" / "lane-manifest.yaml"
DEPLOY_RUNTIME_PATH = ROOT / "scripts" / "deploy-runtime.sh"
DEPLOY_AGENT_EVENTS_PATH = (
    ROOT / "scripts" / "deploy-agent" / "deploy_agent" / "events.py"
)

LANE = "lakshman"
COMPOSE_PROJECT = "omnibase-infra-lakshman"

#: Mirrors the omni_home scanner's ``_GOVERNED_PROJECTS`` tuple. Kept as a
#: literal here on purpose: this test's job is to prove our identifiers cannot
#: collide with those strings, so importing them (impossible across repos) is
#: not what is wanted — restating them and asserting non-collision is.
GOVERNED_COMPOSE_PROJECTS = (
    "omnibase-infra-prod",
    "omnibase-infra-stability-test",
    "omnibase-infra-judge",
)
#: Mirrors the scanner's ``_PROD_PORTS`` + ``_STABILITY_PORTS``.
GOVERNED_LANE_PORT_LITERALS = ("28085", "28086", "18085", "18086")


def _contract() -> ModelRuntimePolicyContract:
    return ModelRuntimePolicyContract.model_validate(
        yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    )


def test_collaborator_lane_is_not_deploy_attribution_governed() -> None:
    """Gate 1: lane-deploy attribution. OUT for lakshman, IN for the other three."""
    assert LANE not in GOVERNED_LANES, (
        f"{LANE!r} must not be in GOVERNED_LANES. It is a fully-mutable "
        "collaborator sandbox, in the same class as `dev` — which is also "
        "deliberately absent. Governing it would force its owner off the "
        "sanctioned path with no governance gain."
    )
    assert "dev" not in GOVERNED_LANES, (
        "the dev lane's absence is the precedent this lane's absence rests on; "
        "if dev became governed, re-derive the collaborator lane's stance"
    )
    assert frozenset({"stability-test", "prod", "judge"}) == GOVERNED_LANES, (
        "widening RuntimeProfileName must not add to OR remove from the "
        f"governed lane set; got {sorted(GOVERNED_LANES)}"
    )


def test_collaborator_lane_is_not_in_the_prod_grant_interlock() -> None:
    """Gate 2: live-prod-grant interlock. Only stability-test may be in it.

    This is the load-bearing one. The interlock exists because the OMN-13418
    prod-promotion gate resolves its ``stability-proven`` premise from the
    stability lane, so an unattributed stability rebuild erodes the basis of
    every live grant. No grant resolves anything from the collaborator lane, so
    a rebuild here cannot erode any premise — and must never be treated as
    though it could, in either direction.
    """
    assert LANE not in GRANT_INTERLOCK_LANES
    assert frozenset({"stability-test"}) == GRANT_INTERLOCK_LANES, (
        "the grant interlock covers exactly the lane the `stability-proven` "
        f"premise is resolved from; got {sorted(GRANT_INTERLOCK_LANES)}"
    )


def test_collaborator_project_name_cannot_match_the_raw_bypass_scanner() -> None:
    """Gate 3a: no governed compose project is a substring of ours."""
    for governed in GOVERNED_COMPOSE_PROJECTS:
        assert governed not in COMPOSE_PROJECT, (
            f"compose project {COMPOSE_PROJECT!r} contains the governed project "
            f"name {governed!r} as a substring, so omni_home's no-raw-prod-bypass "
            "scanner would flag every ordinary `up -d` of this lane as a raw "
            "governed-lane mutation"
        )
    # Guard the guard: the assertion above is only meaningful while the scanner
    # really does key on these strings. If a governed project is ever renamed
    # such that it becomes a prefix of ours, this catches it.
    assert COMPOSE_PROJECT.startswith("omnibase-infra-")
    assert COMPOSE_PROJECT not in GOVERNED_COMPOSE_PROJECTS


def test_collaborator_lane_ports_cannot_match_the_raw_bypass_scanner() -> None:
    """Gate 3b: no governed lane port literal appears in any port we publish."""
    contract = _contract()
    profile = contract.profiles[LANE]
    published = {
        str(profile.main_port),
        str(profile.effects_port),
        "45436",  # postgres
        "46379",  # valkey
        "55092",  # redpanda kafka (external)
        "55644",  # redpanda admin
        "53002",  # projection-api
        "58080",  # keycloak, reserved and unused
    }
    for port in published:
        for governed_port in GOVERNED_LANE_PORT_LITERALS:
            assert governed_port not in port, (
                f"reserved port {port!r} contains the governed lane port "
                f"literal {governed_port!r}; omni_home's no-raw-prod-bypass "
                "scanner matches ports by substring and would flag this lane"
            )


def test_compose_file_publishes_only_the_reserved_port_block() -> None:
    """The reservation is only worth having if the compose file honours it.

    OMN-17143 reserved eight ports after proving them free, specifically to stop
    an ad-hoc pick from displacing a governed lane (OMN-13581). This asserts the
    compose file publishes host ports from that block and nothing else.
    """
    reserved = {"58085", "58086", "45436", "46379", "55092", "55644", "53002"}
    raw = (ROOT / "docker" / "docker-compose.lakshman.yml").read_text(encoding="utf-8")
    # Host-side of every published mapping: literal "HOST:CONTAINER" entries and
    # `${VAR:-HOST}:CONTAINER` defaults alike.
    published = set(re.findall(r'"(?:\$\{[A-Z_]+:[-?][^}]*\}|\d+):\d+"', raw))
    host_ports = {
        m.group(1)
        for entry in published
        if (m := re.match(r'"?(?:\$\{[A-Z_]+:-)?(\d+)', entry))
    }
    # The two runtime ports come from the policy env, not a literal, so add the
    # contract's own values rather than pretending the scrape found them.
    profile = _contract().profiles[LANE]
    host_ports |= {str(profile.main_port), str(profile.effects_port)}

    unreserved = host_ports - reserved
    assert not unreserved, (
        "docker-compose.lakshman.yml publishes host ports outside the OMN-17143 "
        f"reserved block: {sorted(unreserved)}"
    )


def test_sanctioned_deploy_script_refuses_the_collaborator_lane() -> None:
    """``deploy-runtime.sh`` must not learn this lane.

    ``resolve_lane_overlay_filename`` fails closed on an unknown lane. That is
    the correct behaviour here and not an oversight to fix: the collaborator lane
    is brought up by its owner with a plain ``docker compose --profile lakshman
    up -d``, never through the governed deploy path, and never by the deploy
    agent. Adding it to the case arm below would quietly hand it the deploy
    agent's attribution + grant machinery.
    """
    text = DEPLOY_RUNTIME_PATH.read_text(encoding="utf-8")
    assert "stability-test|prod|judge)" in text, (
        "the sanctioned deploy script's lane allowlist changed shape; re-derive "
        "whether the collaborator lane is still excluded"
    )
    assert f"{LANE})" not in text
    assert f"|{LANE}" not in text

    events = DEPLOY_AGENT_EVENTS_PATH.read_text(encoding="utf-8")
    assert LANE.upper() not in events, (
        "the deploy agent's EnumRuntimeLane must not gain a collaborator lane "
        "member — the deploy agent is the governed promotion path"
    )


def test_collaborator_lane_declares_itself_optional_in_the_census() -> None:
    """A declared-but-unbuilt optional lane must not emit census drift.

    ``optional: true`` is what keeps a lane that is entirely down from ticketing
    on every census tick. The lane ships declared and not built, so dropping this
    flag in the same window would turn the census noisy for a lane nobody has
    stood up yet.
    """
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    lane_spec = manifest["lanes"][LANE]

    assert lane_spec["optional"] is True
    assert lane_spec["compose_project"] == COMPOSE_PROJECT
    assert lane_spec["network"] == "omnibase-infra-lakshman-network"
    assert lane_spec["compose_file"] == "docker/docker-compose.lakshman.yml"
    # The governed lanes must NOT have become optional along the way.
    for governed in ("stability-test", "prod", "judge"):
        assert not manifest["lanes"][governed].get("optional", False), (
            f"lane {governed!r} became optional — an entirely-absent governed "
            "lane would stop being reported as drift"
        )


def test_collaborator_lane_runtime_addresses_are_distinct_from_every_lane() -> None:
    """No process on this lane may collide with another lane's runtime address.

    The contract model already rejects duplicates globally; this states the
    property at the lane level so the failure message names the lane rather than
    an anonymous address collision.
    """
    contract = _contract()
    ours = {
        process.runtime_address
        for process in contract.profiles[LANE].processes.values()
    }
    theirs = {
        process.runtime_address
        for name, profile in contract.profiles.items()
        if name != LANE
        for process in profile.processes.values()
    }

    assert ours.isdisjoint(theirs)
    for address in ours:
        assert f"/{LANE}/" in address, (
            f"runtime address {address!r} does not name its own lane, so a "
            "message routed by address could land on the wrong lane"
        )
