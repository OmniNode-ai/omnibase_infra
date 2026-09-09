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

THE PROJECTION DSN IS DECLARED THE SAME WAY, BY NAME (OMN-18060)
---------------------------------------------------------------
Chain-canary run 34281968883 proved three of the five OMN-16025 links and
failed closed on link 2 with ``projection_readback_not_configured -- no DSN was
configured for the projection readback``. The refusal was right; the gate was
unclosable, because there was nowhere to say where the DSN comes from.

The same overlay is that place, and the rule that made the broker safe to
commit applies unchanged: what is declared is a NAME, never a value. A lane's
optional ``projection_readback.dsn_env`` block names the environment variable
the DSN arrives under; the value is injected into the job environment from the
lab store, read by the node out of ``os.environ``, and never passes through
argv, a workflow output, or a log line. ``load_lane_projection_readback``
REFUSES a declaration whose ``dsn_env`` parses as a DSN, so a value pasted
where a name belongs is a red gate rather than a committed credential, and
``dsn_shaped_argv_flags`` refuses a DSN that reached the process through the
command line whatever flag carried it.

The declaration is dev-lane-only, enforced here rather than trusted: the
chain canary publishes a live delegation and then reads a database, and
``chain-canary.yml``'s own header scopes both halves to the dev lane. A
projection DSN declared against ``stability``, ``judge`` or ``prod`` would
point the probe at a lane it has no authorization to read, so it raises
instead of resolving.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_chain_canary_effect.model_lane_projection_readback import (
    ModelLaneProjectionReadback,
)

__all__ = [
    "DOCKER_DESKTOP_HOST_ALIASES",
    "INMEMORY_BROKER",
    "PROJECTION_READBACK_DSN_ENV_NAME_VAR",
    "PROJECTION_READBACK_LANES",
    "ModelLaneProjectionReadback",
    "ModelLaneTransport",
    "dsn_shaped_argv_flags",
    "host_aliases_in",
    "lane_transport_env",
    "load_lane_projection_readback",
    "load_lane_transport",
    "looks_like_a_dsn",
    "projection_readback_env",
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


# --- Declared projection DSN, by NAME (OMN-18060) -----------------------------

#: The only lane a projection readback may be declared on. The chain canary
#: publishes a live delegation and then reads a database; ``chain-canary.yml``
#: scopes both halves to the dev lane (compose project ``omnibase-infra``, the
#: pre-authorized fully-mutable test platform). stability-test, judge and prod
#: are read-only surfaces this probe has no ticket for, so a declaration
#: against one of them is refused rather than honoured.
PROJECTION_READBACK_LANES: frozenset[str] = frozenset({"dev"})

#: Env var the workflow exports the resolved NAME under. Deliberately distinct
#: from the name it points at: this one carries a NAME and may be echoed, the
#: one it names carries the DSN and may not.
PROJECTION_READBACK_DSN_ENV_NAME_VAR = "CHAIN_CANARY_PROJECTION_DSN_ENV"

_PROJECTION_READBACK_KEY = "projection_readback"
_DSN_ENV_KEY = "dsn_env"

# A POSIX-shell environment variable NAME. Narrow on purpose: nothing matching
# this can also be a DSN, because a DSN needs at least a ':' and a '/'.
_ENV_NAME_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")

# Markers of a connection STRING. `://` is the scheme separator of every libpq
# URI form; the keyword/value form is spelled `host=... dbname=...`. Matching on
# structure rather than on a keyword list is what makes this refuse a DSN shape
# nobody anticipated.
_DSN_URI_SCHEMES: tuple[str, ...] = (
    "postgres://",
    "postgresql://",
    "postgres+asyncpg://",
    "postgresql+asyncpg://",
)
_DSN_KEYWORD_MARKERS: tuple[str, ...] = (
    "dbname=",
    "host=",
    "password=",
    "user=",
)


def looks_like_a_dsn(value: str) -> bool:
    """Whether ``value`` parses as a Postgres connection string.

    Used to refuse a VALUE anywhere a NAME is expected. Never include the
    value in a message built from this — the whole reason it is being refused
    is that it is a credential.
    """
    lowered = value.strip().lower()
    if not lowered:
        return False
    if any(lowered.startswith(scheme) for scheme in _DSN_URI_SCHEMES):
        return True
    if "://" in lowered:
        return True
    return any(marker in lowered for marker in _DSN_KEYWORD_MARKERS)


def dsn_shaped_argv_flags(argv: Sequence[str]) -> tuple[str, ...]:
    """Flag names in ``argv`` whose value parses as a DSN, in order.

    Returns the FLAG, never the value. A DSN on a command line is readable by
    every process on the host through ``/proc/<pid>/cmdline`` (and by anyone
    reading the run log, since the dispatch step echoes what it ran), so the
    refusal message this feeds must not reprint the thing it is refusing.

    Positional occurrences are reported as ``argv[<index>]`` so a DSN that
    arrived without a flag is still named precisely enough to find.
    """
    offenders: list[str] = []
    for index, token in enumerate(argv):
        if not looks_like_a_dsn(token):
            continue
        # `--flag=value` carries both halves in one token; report the flag.
        if token.startswith("-") and "=" in token:
            offenders.append(token.split("=", 1)[0])
            continue
        previous = argv[index - 1] if index > 0 else ""
        if previous.startswith("-"):
            offenders.append(previous)
            continue
        offenders.append(f"argv[{index}]")
    return tuple(offenders)


def load_lane_projection_readback(
    overlay_path: Path, lane: str
) -> ModelLaneProjectionReadback:
    """Read ``lane``'s declared projection-readback DSN NAME out of the overlay.

    Every failure raises. There is no default and no fallback to the bus
    terminal: OMN-14843 measured 26 of 38 correlations stranded mid-FSM while
    the topic layer was healthy at the same moment, so a green terminal is not
    evidence about the projection layer and must never stand in for one.
    """
    if not lane.strip():
        message = "a lane id is required to resolve a declared projection readback"
        raise ValueError(message)
    lane_key = lane.strip()

    if lane_key not in PROJECTION_READBACK_LANES:
        message = (
            f"lane {lane_key!r} may not declare a projection readback; only "
            f"{sorted(PROJECTION_READBACK_LANES)} may. The chain canary "
            "publishes a live delegation and then reads a database, and both "
            "halves are dev-lane-only by the workflow's own declared scope. "
            "stability-test, judge and prod are read-only surfaces this probe "
            "has no authorization to read."
        )
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
    if not isinstance(lanes, dict) or lane_key not in lanes:
        declared = sorted(lanes) if isinstance(lanes, dict) else []
        message = (
            f"lane {lane_key!r} is not declared in {overlay_path}; declared "
            f"lanes: {declared}"
        )
        raise ValueError(message)

    declaration = lanes[lane_key]
    if not isinstance(declaration, dict):
        message = f"lane {lane_key!r} in {overlay_path} is not a mapping"
        raise ValueError(message)

    block = declaration.get(_PROJECTION_READBACK_KEY)
    if block is None:
        message = (
            f"lane {lane_key!r} in {overlay_path} declares no "
            f"{_PROJECTION_READBACK_KEY!r} block, so there is no DSN reference "
            "for OMN-16025 link 2. Declare the NAME the DSN is injected under "
            f"(a {_DSN_ENV_KEY!r} entry) — never the DSN itself."
        )
        raise ValueError(message)
    if not isinstance(block, dict):
        message = (
            f"lane {lane_key!r} in {overlay_path} declares "
            f"{_PROJECTION_READBACK_KEY!r} as {type(block).__name__}, not a "
            "mapping"
        )
        raise ValueError(message)

    dsn_env = str(block.get(_DSN_ENV_KEY) or "").strip()
    if not dsn_env:
        message = (
            f"lane {lane_key!r} in {overlay_path} declares "
            f"{_PROJECTION_READBACK_KEY!r} with no {_DSN_ENV_KEY!r} entry"
        )
        raise ValueError(message)

    if looks_like_a_dsn(dsn_env):
        # The value is deliberately absent from this message.
        message = (
            f"lane {lane_key!r} in {overlay_path} declares a {_DSN_ENV_KEY} "
            "entry that parses as a connection string. That is a VALUE where a "
            "NAME belongs, and this overlay is committed, CODEOWNERS-reviewed "
            "config that must never carry a credential. Declare the NAME the "
            "DSN is injected under and put the value in the lab store under "
            "that name."
        )
        raise ValueError(message)

    if not _ENV_NAME_PATTERN.match(dsn_env):
        message = (
            f"lane {lane_key!r} in {overlay_path} declares {_DSN_ENV_KEY}="
            f"{dsn_env!r}, which is not a POSIX environment variable name "
            f"({_ENV_NAME_PATTERN.pattern}). The workflow injects the secret "
            "under this literal name, so a name the shell cannot export is a "
            "readback that silently never runs."
        )
        raise ValueError(message)

    return ModelLaneProjectionReadback(lane=lane_key, dsn_env=dsn_env)


def projection_readback_env(
    declaration: ModelLaneProjectionReadback,
) -> dict[str, str]:
    """The declared NAME as the one env var the workflow may export.

    Exactly the non-secret half, and the asymmetry is the point: this returns
    the NAME of the variable the DSN arrives in. The DSN itself is injected by
    the job's secret block under that same name and is never read, written or
    echoed by this module.
    """
    return {PROJECTION_READBACK_DSN_ENV_NAME_VAR: declaration.dsn_env}
