# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17919: the lane-mirror source leg must dial an UNAMBIGUOUSLY-ADVERTISED listener.

The second half of the same outage. OMN-17034 wired the leg and this ticket's
first fix (#3205, 47aa19a31) taught it to read the wire shape -- and the mirror
still moved **zero** records, because a Kafka client does not keep talking to
the address it was configured with. It bootstraps there once, reads the
cluster's **advertised** listener out of the metadata response, and every
subsequent fetch / join / commit goes to *that* address.

Measured on .201, 2026-09-05, read-only:

    docker exec omninode-gateway-forwarder getent hosts redpanda
      -> 172.19.0.7                                      # the DEV broker

    docker inspect omnibase-infra-redpanda                -> aliases
      ['omnibase-infra-redpanda', 'redpanda']             # 172.19.0.7
    docker inspect omnibase-infra-stability-test-redpanda -> aliases
      ['omnibase-infra-stability-test-redpanda', 'redpanda']  # 172.22.0.4

Both lane brokers advertise their INTERNAL listener as the bare name
``redpanda``, and both carry ``redpanda`` as a Docker network alias, because
each lane's compose file names the service ``redpanda``. The forwarder is
joined to both lane networks, so it resolves the bare name to whichever
network Docker's embedded DNS answers from -- dev. Configuring the source leg
as ``omnibase-infra-stability-test-redpanda:9092`` therefore bootstrapped
correctly against 172.22.0.4 and then walked straight back to dev: the
consumer group came up ``Stable`` on **dev** with LAG 0 and ``Dead`` on
stability, holding one partition where stability's topic has six.

``test_lane_brokers_are_addressed_by_unique_container_name_not_bare_redpanda``
in ``test_lane_mirror_omn17034.py`` stayed green throughout, because it
constrains the address this deployment *configures* and the defect is in the
address the broker *advertises back*. That is the gap these tests close.

The fix is listener selection, not name disambiguation. Redpanda advertises
per-listener: a client that bootstraps on stability's EXTERNAL listener is
handed the external advertised address, which this lane pins to a literal
routable endpoint (``docker-compose.stability-test.yml``, OMN-12832) and not to
a name that collides with anything. Proven live from inside the forwarder
container on 2026-09-05, read-only:

    docker exec omnibase-infra-redpanda \
      rpk cluster metadata -X brokers=100.109.203.94:39092
      -> BROKERS  ID 0*  HOST 100.109.203.94  PORT 39092

    docker exec omninode-gateway-forwarder python3 -c "socket.connect(...)"
      -> CONNECT OK 100.109.203.94:39092

No stability mutation is involved: that external advertised address is already
this lane's committed contract data and is not changed by this fix.

These assertions are cross-file on purpose. Pinning only the string in
``beta-gateway-canary.yaml`` would re-freeze the same class of defect the
OMN-17034 assertion froze -- a value that looks right in isolation. What has to
hold is a RELATIONSHIP: the endpoint this deployment dials must be one the
source lane's own compose file says it advertises back, and that advertised
host must not collide with another lane the forwarder is joined to.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_STABILITY_COMPOSE = _REPO_ROOT / "docker" / "docker-compose.stability-test.yml"
_DEV_COMPOSE = _REPO_ROOT / "docker" / "docker-compose.infra.yml"
_GATEWAY_CANARY = _REPO_ROOT / "docker" / "gateway" / "beta-gateway-canary.yaml"

# The advertised-listener spec is read out of the compose files as TEXT, not as
# parsed YAML: both files carry custom compose-merge tags (`!!merge <<:`,
# `!override`) that `yaml.safe_load` refuses, and the dev file's value is a
# `${VAR:?...}` interpolation that only means anything as a string anyway.
#
# Anchored on the `advertise` keyword: `--kafka-addr` (the BIND address, always
# `0.0.0.0`) has identical syntax and comes first in both files. Matching it by
# accident is how a test like this reads as green while asserting nothing.
_ADVERTISE_KEY = r"advertised?[-_]kafka[-_]addr"
_INTERNAL_RE = re.compile(
    _ADVERTISE_KEY + r"[^\n]*?internal://(?P<internal>[^,\s\"']+)"
)
_EXTERNAL_RE = re.compile(
    _ADVERTISE_KEY + r"[^\n]*?internal://[^,]+,external://(?P<external>[^\s\"',]+)"
)


def _advertised_listeners(compose_path: Path) -> dict[str, str]:
    """Return {listener_name: advertised host:port} for a lane's Kafka listeners.

    ``external`` is omitted when the lane renders it from a deploy-time
    ``${VAR}`` interpolation rather than pinning a literal. That is not a
    parser limitation, it is the fact that decides which lanes can be dialled
    on their external listener at all: the dev lane interpolates
    ``DEV_REDPANDA_ADVERTISE_HOST``, so what it advertises externally is not
    knowable from this repo, while the stability lane pins a literal in the
    compose contract itself (OMN-12832) precisely so that it is.
    """
    text = compose_path.read_text(encoding="utf-8")
    internal = _INTERNAL_RE.search(text)
    assert internal, f"no advertised kafka listener spec found in {compose_path}"
    listeners = {"internal": internal.group("internal")}
    external = _EXTERNAL_RE.search(text)
    if external is not None and "${" not in external.group("external"):
        listeners["external"] = external.group("external")
    return listeners


def _host_of(endpoint: str) -> str:
    """`host:port` -> `host` (the compose specs here are never IPv6 literals)."""
    return endpoint.rsplit(":", 1)[0]


def _gateway_config() -> dict[str, Any]:
    return cast(
        "dict[str, Any]", yaml.safe_load(_GATEWAY_CANARY.read_text(encoding="utf-8"))
    )


# ---------------------------------------------------------------------------
# The condition that made the leg inert
# ---------------------------------------------------------------------------


def test_both_lane_brokers_advertise_the_same_bare_internal_name() -> None:
    """The premise. If this ever stops holding, the fix below can be simplified.

    This is not a wish -- it is the measured state of the two lanes, asserted so
    that the reason the source leg dials the external listener stays legible.
    Both compose files name the service `redpanda`, so Docker puts a `redpanda`
    alias on both lane networks AND both brokers advertise `redpanda:9092` on
    their internal listener. A container joined to both cannot tell them apart.
    """
    dev_internal = _advertised_listeners(_DEV_COMPOSE)["internal"]
    stability_internal = _advertised_listeners(_STABILITY_COMPOSE)["internal"]

    assert _host_of(dev_internal) == _host_of(stability_internal) == "redpanda"

    # And the reason the fix is asymmetric -- the source leg moves to an external
    # listener while the dev mirror target stays on a container name: only the
    # stability lane pins a literal external advertised address.
    assert "external" in _advertised_listeners(_STABILITY_COMPOSE)
    assert "external" not in _advertised_listeners(_DEV_COMPOSE)


# ---------------------------------------------------------------------------
# The invariant the fix installs
# ---------------------------------------------------------------------------


def test_source_leg_dials_a_listener_whose_advertised_host_is_unambiguous() -> None:
    """The source bootstrap must resolve to a listener no other lane can answer for.

    This is the assertion that would have failed before the fix: the configured
    endpoint `omnibase-infra-stability-test-redpanda:9092` is the source lane's
    INTERNAL listener, whose advertised host is the bare `redpanda` the dev lane
    also answers to.
    """
    stability = _advertised_listeners(_STABILITY_COMPOSE)
    dev = _advertised_listeners(_DEV_COMPOSE)
    source_bootstrap = _gateway_config()["lane_mirror_source_bus"]["bootstrap_servers"]

    listener = {
        # The internal listener answers on the in-network port (9092); the
        # external listener answers on the published host port. Which one a
        # client lands on is decided by the endpoint it bootstraps against.
        f"{_STABILITY_COMPOSE.name}:internal": stability["internal"],
        f"{_STABILITY_COMPOSE.name}:external": stability["external"],
    }
    assert source_bootstrap == stability["external"], (
        "the lane_mirror source leg must bootstrap on the source lane's EXTERNAL "
        "listener, whose advertised address is a literal routable endpoint. "
        f"configured={source_bootstrap!r} advertised_listeners={listener!r}"
    )

    advertised_back = _host_of(stability["external"])
    assert advertised_back != _host_of(stability["internal"]), (
        "the advertised host of the listener the source leg dials must differ "
        "from the ambiguous internal one"
    )
    assert advertised_back != _host_of(dev["internal"]), (
        f"the source leg's advertised host {advertised_back!r} collides with the "
        "dev lane's advertised host; every fetch/join/commit would be re-routed "
        "to dev, which is the OMN-17919 defect"
    )


def test_mirror_target_leg_is_not_the_source_lane() -> None:
    """The dev mirror target must not resolve back to the source broker.

    ``ModelGatewayForwarderRuntimeConfig`` already refuses a source and a mirror
    lane configured with the same string. That check could not see the defect
    this ticket fixes -- the two strings differed while both *resolved* to dev,
    which is a mirror republishing a lane onto itself. Asserting the two dial
    endpoints are distinct AND that neither is the other lane's advertised
    address is the version of that check that would have caught it.
    """
    config = _gateway_config()
    source = config["lane_mirror_source_bus"]["bootstrap_servers"]
    dev_target = config["lane_mirror_buses"]["dev"]["bootstrap_servers"]
    stability = _advertised_listeners(_STABILITY_COMPOSE)

    assert source != dev_target
    assert dev_target not in stability.values(), (
        "the dev mirror target is addressed at one of the SOURCE lane's own "
        "advertised endpoints; the mirror would republish stability's records "
        "onto stability"
    )
    # The dev target keeps its unique container name: the forwarder is joined to
    # the dev network, so `omnibase-infra-redpanda` bootstraps unambiguously.
    # Its re-resolution of dev's advertised bare `redpanda` lands on dev because
    # dev is the network Docker answers that alias from -- correct today, and
    # recorded as a residual on OMN-17919 rather than left implicit.
    assert dev_target.startswith("omnibase-infra-redpanda:")


def test_source_leg_is_plaintext_like_the_listener_it_dials() -> None:
    """The external listener carries no SASL/TLS; the config must not claim it does.

    Proven live 2026-09-05: `rpk cluster metadata -X brokers=100.109.203.94:39092`
    returned cluster metadata with no credentials supplied.
    """
    source = _gateway_config()["lane_mirror_source_bus"]
    assert source["security_protocol"] == "PLAINTEXT"
    # Unchanged by this fix, and load-bearing: the whole point of the redeploy is
    # to consume the backlog the group never read (OMN-15781).
    assert source["auto_offset_reset"] == "earliest"
    assert source["enable_auto_commit"] is False
