# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Declared bus transport for the chain canary's broker legs (OMN-17926).

WHAT WENT WRONG
---------------
``chain-canary.yml`` hardcoded ``host.docker.internal:19092`` as the default
bootstrap for both broker legs. On 2026-09-07 at ~16:40Z, OMN-18012 Phase B
enabled SASL/SCRAM-SHA-256 on the .201 dev-lane Redpanda EXTERNAL listener --
SASL over PLAINTEXT, no TLS. A client that opens plaintext against that
listener is disconnected during the handshake, and aiokafka surfaces that as
``KafkaConnectionError: Unable to bootstrap from [('host.docker.internal',
19092, AF_UNSPEC)]``.

The bracket is exact, and it is the reason this is a probe defect rather than a
plane defect: run 34139858442 (2026-09-07T15:44Z) still reached
``projection_readback_not_configured`` -- the bus WAS readable -- and run
34148855051 (2026-09-07T17:44Z) is the first of an unbroken sequence of
``quarantine_probe_failed``, byte-identical through 2026-09-08. Every one of
those runs fired its delegation successfully first. The chain was alive; the
canary could not read it.

WHY A HARDCODED ADDRESS IS THE DEFECT, NOT THE OUTDATED VALUE
-------------------------------------------------------------
Two things were wrong with the literal and only one of them was the address.

``host.docker.internal`` is a Docker-Desktop alias. It exists on this runner
ONLY because ``docker/docker-compose.runners.yml`` gives the deploy runner an
``extra_hosts: host.docker.internal:host-gateway`` mapping; on any other Linux
runner it resolves to nothing at all. An address that depends on one
container's compose block is not a statement about where the lane's broker is.

More importantly, an address literal carries no TRANSPORT. The lane's
``security_protocol`` and ``sasl_mechanism`` are declared in omnimarket's
``config/ci_bus_lanes.yaml`` -- the same overlay
``runtime-rebuild-trigger.yml`` already reads -- precisely so a client reads
the transport instead of inferring it. OMN-18012's own write-up is the
authority: "Credential PRESENCE is not a statement about transport. The lane
is." A literal in a workflow default is the inverse mistake -- an address with
no transport at all.

So this module resolves BOTH halves from the one declaration, and the request
model refuses a Docker-Desktop alias outright rather than accepting an address
whose meaning depends on which container is reading it.

CREDENTIALS ARE NEVER RESOLVED HERE. The overlay is config, not secret: it
carries the broker, the protocol and the mechanism, and nothing else. SASL
username and password reach the client through the standard ``KAFKA_SASL_*``
environment variables that ``build_aiokafka_auth_kwargs_from_env`` reads --
never through argv, where they would land in a process list and in the run log.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "DOCKER_DESKTOP_HOST_ALIASES",
    "INMEMORY_BROKER",
    "ModelLaneTransport",
    "host_aliases_in",
    "lane_transport_env",
    "load_lane_transport",
]

#: Hostnames that exist only inside a Docker-Desktop VM, or on a Linux
#: container that was explicitly handed a ``host-gateway`` mapping. None of
#: them names a broker: two runners resolving the same literal reach two
#: different machines, or nothing.
DOCKER_DESKTOP_HOST_ALIASES: frozenset[str] = frozenset(
    {
        "host.docker.internal",
        "gateway.docker.internal",
        "vm.docker.internal",
        "docker.for.mac.host.internal",
        "docker.for.mac.localhost",
        "docker.for.win.host.internal",
        "docker.for.win.localhost",
    }
)

#: The overlay's sentinel for "no cross-process broker on this lane".
INMEMORY_BROKER = "inmemory"

_SASL_PROTOCOLS: frozenset[str] = frozenset({"SASL_PLAINTEXT", "SASL_SSL"})
# Spelled out rather than composed with `|`: the union-usage validator counts a
# BitOr in an annotated assignment, and a set union here reads to it as one more
# non-optional type union against the repo budget.
_VALID_SECURITY_PROTOCOLS: frozenset[str] = frozenset(
    {"PLAINTEXT", "SSL", "SASL_PLAINTEXT", "SASL_SSL"}
)


class ModelLaneTransport(BaseModel):
    """One lane's declared bus transport: where, and how to speak to it.

    The two halves travel together on purpose. A bootstrap address with no
    declared protocol is what let a plaintext client open against a SASL
    listener and report the lane unreadable for 25 consecutive runs.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    lane: str = Field(description="Lane id this transport was declared under")
    bootstrap_servers: str = Field(description="Declared broker, host:port")
    security_protocol: str = Field(
        description="librdkafka security protocol, as the overlay declares it"
    )
    sasl_mechanism: str | None = Field(
        default=None, description="Required for a SASL protocol, absent otherwise"
    )


def host_aliases_in(bootstrap: str) -> tuple[str, ...]:
    """Docker-Desktop aliases used as a host in ``bootstrap``, in order.

    Accepts the comma-separated multi-broker form aiokafka takes, because a
    single bad entry in a list is exactly as unresolvable as a bad singleton.
    """
    found: list[str] = []
    for entry in bootstrap.split(","):
        host = entry.strip().rsplit(":", 1)[0].strip().lower()
        # An IPv6 literal is bracketed; strip the brackets before comparing.
        host = host.removeprefix("[").removesuffix("]")
        if host in DOCKER_DESKTOP_HOST_ALIASES:
            found.append(host)
    return tuple(found)


def load_lane_transport(overlay_path: Path, lane: str) -> ModelLaneTransport:
    """Read ``lane``'s declared transport out of a ``ci_bus_lanes.yaml`` overlay.

    Every failure raises rather than returning a default. A probe that guessed
    a broker would report on a lane nobody asked about, which is the OMN-14800
    silent-repoint failure one level down.
    """
    if not lane.strip():
        message = "a lane id is required to resolve a declared transport"
        raise ValueError(message)

    try:
        raw: object = yaml.safe_load(overlay_path.read_text(encoding="utf-8"))
    except OSError as exc:
        message = f"cannot read the lane overlay at {overlay_path}: {exc}"
        raise ValueError(message) from exc
    except yaml.YAMLError as exc:
        message = f"the lane overlay at {overlay_path} is not valid YAML: {exc}"
        raise ValueError(message) from exc

    if not isinstance(raw, dict):
        message = f"the lane overlay at {overlay_path} is not a YAML mapping"
        raise ValueError(message)

    lanes = raw.get("lanes")
    if not isinstance(lanes, dict) or lane not in lanes:
        declared = sorted(lanes) if isinstance(lanes, dict) else []
        message = (
            f"lane {lane!r} is not declared in {overlay_path}; declared lanes: "
            f"{declared}"
        )
        raise ValueError(message)

    declaration = lanes[lane]
    if not isinstance(declaration, dict):
        message = f"lane {lane!r} in {overlay_path} is not a mapping"
        raise ValueError(message)

    broker = str(declaration.get("broker") or "").strip()
    if not broker:
        message = f"lane {lane!r} in {overlay_path} declares no broker"
        raise ValueError(message)
    if broker == INMEMORY_BROKER:
        message = (
            f"lane {lane!r} declares the in-memory bus, which no cross-process "
            "probe can read. Point the canary at a lane with a real broker "
            "rather than reporting a check it never ran."
        )
        raise ValueError(message)

    aliases = host_aliases_in(broker)
    if aliases:
        message = (
            f"lane {lane!r} in {overlay_path} declares broker {broker!r}, whose "
            f"host {aliases[0]!r} is a Docker-Desktop alias. A declaration must "
            "name an address every reader resolves the same way."
        )
        raise ValueError(message)

    protocol = str(declaration.get("security_protocol") or "").strip().upper()
    if protocol not in _VALID_SECURITY_PROTOCOLS:
        message = (
            f"lane {lane!r} in {overlay_path} declares security_protocol "
            f"{declaration.get('security_protocol')!r}; expected one of "
            f"{sorted(_VALID_SECURITY_PROTOCOLS)}. Refusing to infer the "
            "transport from the environment (OMN-18012)."
        )
        raise ValueError(message)

    mechanism_raw = declaration.get("sasl_mechanism")
    mechanism = str(mechanism_raw).strip() if mechanism_raw else None
    if protocol in _SASL_PROTOCOLS and not mechanism:
        message = (
            f"lane {lane!r} declares {protocol} but no sasl_mechanism; a SASL "
            "protocol without a mechanism cannot be spoken."
        )
        raise ValueError(message)
    if protocol not in _SASL_PROTOCOLS and mechanism:
        message = (
            f"lane {lane!r} declares sasl_mechanism {mechanism!r} beside "
            f"non-SASL protocol {protocol}; that is a contradiction, not a "
            "default to resolve."
        )
        raise ValueError(message)

    return ModelLaneTransport(
        lane=lane,
        bootstrap_servers=broker,
        security_protocol=protocol,
        sasl_mechanism=mechanism,
    )


def lane_transport_env(transport: ModelLaneTransport) -> dict[str, str]:
    """The declared transport as the ``KAFKA_*`` env the client already reads.

    Exactly the non-secret half. ``KAFKA_SASL_USERNAME`` / ``KAFKA_SASL_PASSWORD``
    are deliberately absent: they come from the job's secret injection, so no
    credential ever passes through this module, a workflow output, or argv.
    """
    env = {
        "KAFKA_BOOTSTRAP_SERVERS": transport.bootstrap_servers,
        "KAFKA_SECURITY_PROTOCOL": transport.security_protocol,
    }
    if transport.sasl_mechanism:
        env["KAFKA_SASL_MECHANISM"] = transport.sasl_mechanism
    return env
