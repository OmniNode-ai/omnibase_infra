# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Resolve a delegation's broker ADDRESS from the declared lane, never the env.

OMN-16871. ``onex delegate --bus kafka`` with no explicit broker used to let
``EventBusKafka`` read ``KAFKA_BOOTSTRAP_SERVERS``. On the launching Mac that
variable is ``192.168.86.201:39092``  # onex-allow-internal-ip OMN-16871 reason="forensic quote of the ambient value this module removes from the resolution path; not a configurable endpoint"
-- the STABILITY-TEST lane's external Redpanda listener, not the dev lane's
``:19092``. So every ad hoc delegation from a developer shell published onto a
governed proof lane, and the lane's own consumer-group list recorded it: four
``local.omnibase_core.runtime_local_terminal_run_*`` groups belonging to CLI
processes on that Mac were sitting on the stability broker when OMN-16871 was
re-verified on 2026-09-16.

That is not a stale value to flip. An ambient environment variable must never
be able to select a governed lane, which is why this module deletes the env
path rather than repointing it: with the resolved bus on ``kafka``, a caller
that named no lane and no explicit broker is REFUSED, and the refusal names
the missing selection.

WHERE THE MAPPING COMES FROM, AND WHY NOT A NEW FILE
----------------------------------------------------
``omnimarket``'s checked-in ``config/ci_bus_lanes.yaml`` already is the
lane -> broker authority. It was created for exactly this failure one level up
(OMN-14800: a repo secret silently repointed dev -> stability and the
publisher stayed green while emitting to the wrong lane), it already declares
``dev``, ``stability-test`` and the in-memory lanes, and it already carries
the transport beside the address (OMN-18012). ``omnibase_infra`` already
SHIPS the one reader for it --
``nodes/node_chain_canary_effect/lane_transport.py::load_lane_transport`` --
with the refusals this path needs: an undeclared lane, an in-memory lane, a
Docker-Desktop alias host, a missing protocol and a SASL protocol with no
mechanism all raise instead of resolving.

So this module adds no parser and no second declaration. It locates the one
declaration, delegates every semantic judgement to the one reader, and owns
only the CLI-side policy: which selections are complete, and what a refusal
says. The reader is imported lazily inside the function that needs it,
because its package ``__init__`` constructs the chain-canary node and the CLI
must not pay that at import time.

The declaration is located under the workspace root -- resolved from the
``--omnibase-path`` flag (``$OMNIBASE_PATH``) that ``onex delegate`` binds for
the omnimarket drift guard, so no new environment read enters the CLI. When it
cannot be located the command is refused with the path it looked for. A
released wheel carries no copy of the overlay (``omnimarket``'s packaging
ships ``src/omnimarket/**/*.yaml`` only, and this file lives at the repo
root), so on such an install every kafka delegation refuses by name instead of
silently addressing whatever the shell exported. Refusing loudly is the point;
packaging the declaration onto the release path is tracked separately.
"""

from __future__ import annotations

from pathlib import Path

from omnibase_infra.cli.model_delegate_lane_selection import ModelDelegateLaneSelection

__all__ = [
    "LANE_DECLARATION_RELATIVE_PATH",
    "DelegateLaneSelectionError",
    "declared_lane_ids",
    "resolve_lane_target",
    "resolve_lane_declaration_path",
    "resolve_lane_selection",
]

#: Where the lane -> broker declaration sits inside the canonical workspace.
#: A relative path, joined onto ``$OMNI_HOME``: rule 6 forbids an absolute one,
#: and the canonical-clone convention this rides on is the same one
#: ``omnimarket_drift_guard`` already resolves against.
LANE_DECLARATION_RELATIVE_PATH = Path("omnimarket") / "config" / "ci_bus_lanes.yaml"

#: The transport this module governs. Every other bus is in-process, so it has
#: no address to resolve and a lane selection against it is a contradiction.
_KAFKA_BUS = "kafka"


class DelegateLaneSelectionError(Exception):
    """A delegation's broker target could not be resolved from a declaration.

    Raised for an incomplete selection, an undeclared lane, and an
    unreachable or malformed declaration alike. Every one of those is a
    refusal: there is no state of this module that answers with an address it
    did not read out of a checked-in lane declaration.
    """


def resolve_lane_declaration_path(omni_home: Path | None) -> Path:
    """Locate the lane declaration under ``omni_home``.

    Raises:
        DelegateLaneSelectionError: ``omni_home`` is unset, or the declaration
            is not there. Both name what was looked for, because a lane target
            that cannot be read is refused rather than guessed.
    """
    if omni_home is None:
        message = (
            "cannot locate the lane declaration: no workspace root is set. "
            "Pass --omnibase-path <workspace root> (or export "
            "$OMNIBASE_PATH, which that flag binds to; the sanctioned wrapper "
            "omnibase_infra/scripts/onex binds it for you) so "
            f"{LANE_DECLARATION_RELATIVE_PATH} can be read. The broker "
            "address is never taken from the environment (OMN-16871)."
        )
        raise DelegateLaneSelectionError(message)

    declaration = omni_home / LANE_DECLARATION_RELATIVE_PATH
    if not declaration.is_file():
        message = (
            f"the lane declaration {declaration} does not exist. It is the "
            "only source a broker address is resolved from, so a delegation "
            "over the shared bus is refused without it (OMN-16871)."
        )
        raise DelegateLaneSelectionError(message)
    return declaration


def declared_lane_ids(declaration: Path) -> tuple[str, ...]:
    """Lane ids the declaration carries, sorted.

    Raises:
        DelegateLaneSelectionError: the declaration cannot be read or parsed,
            or declares no lanes at all.
    """
    import yaml

    try:
        raw: object = yaml.safe_load(declaration.read_text(encoding="utf-8"))
    except OSError as exc:
        message = f"cannot read the lane declaration at {declaration}: {exc}"
        raise DelegateLaneSelectionError(message) from exc
    except yaml.YAMLError as exc:
        message = f"the lane declaration at {declaration} is not valid YAML: {exc}"
        raise DelegateLaneSelectionError(message) from exc

    lanes = raw.get("lanes") if isinstance(raw, dict) else None
    if not isinstance(lanes, dict) or not lanes:
        message = (
            f"the lane declaration at {declaration} declares no lanes; there "
            "is nothing to address a delegation to."
        )
        raise DelegateLaneSelectionError(message)
    return tuple(sorted(str(lane_id) for lane_id in lanes))


def resolve_lane_selection(
    *, lane: str, omni_home: Path | None
) -> ModelDelegateLaneSelection:
    """Resolve ``lane`` to its declared broker and transport.

    Every semantic judgement -- undeclared lane, in-memory lane, unresolvable
    host alias, missing or contradictory transport -- belongs to
    ``load_lane_transport``, the shipped reader for this declaration. This
    function locates the file, delegates, and restates a failure in the
    vocabulary of the flag the caller typed.

    Raises:
        DelegateLaneSelectionError: the declaration is unreachable, or it does
            not bind ``lane`` to a usable broker.
    """
    # Lazy: ``node_chain_canary_effect``'s package __init__ constructs the
    # canary node, which the CLI has no reason to import on every invocation.
    from omnibase_infra.nodes.node_chain_canary_effect.lane_transport import (
        load_lane_transport,
    )

    declaration = resolve_lane_declaration_path(omni_home)
    try:
        transport = load_lane_transport(declaration, lane)
    except ValueError as exc:
        message = f"--lane {lane!r} is not usable: {exc}"
        raise DelegateLaneSelectionError(message) from exc

    return ModelDelegateLaneSelection(
        lane=transport.lane,
        bootstrap_servers=transport.bootstrap_servers,
        security_protocol=transport.security_protocol,
        sasl_mechanism=transport.sasl_mechanism,
        declared_in=declaration,
    )


def _incomplete_selection_message(omni_home: Path | None) -> str:
    """The refusal text for a shared-bus delegation that named no target."""
    head = (
        "this delegation resolves to the shared kafka bus but names no "
        "broker: pass --lane <lane id> to address the lane whose declared "
        "broker it should publish to. The address is NOT read from "
        "KAFKA_BOOTSTRAP_SERVERS any more (OMN-16871) -- on the launching "
        "host that variable names the governed stability-test lane, so an "
        "ambient value silently selected a proof lane."
    )
    try:
        declaration = resolve_lane_declaration_path(omni_home)
        lanes = declared_lane_ids(declaration)
    except DelegateLaneSelectionError as exc:
        return f"{head} Declared lanes cannot be listed: {exc}"
    return f"{head} Lanes declared in {declaration}: {', '.join(lanes)}."


def resolve_lane_target(
    *,
    bus: str,
    lane: str | None,
    kafka_bootstrap: str | None,
    omni_home: Path | None,
) -> ModelDelegateLaneSelection | None:
    """The declared lane this run addresses, or ``None`` when none was selected.

    ``None`` is returned in exactly two complete cases: the resolved bus is
    in-process, so there is no broker to address; or the caller stated the
    address directly with ``--kafka-bootstrap``, which is its own provenance
    and resolves through no declaration.

    On the shared bus exactly one explicit target is required. Neither flag is
    a refusal -- that is the OMN-16871 fix, because the alternative was the
    ambient environment. Both flags together is also a refusal, because two
    addresses is not a selection.

    Raises:
        DelegateLaneSelectionError: the selection is incomplete, contradictory,
            or names a lane the declaration does not bind to a broker.
    """
    if bus != _KAFKA_BUS:
        if lane is not None:
            message = (
                f"--lane {lane!r} is only valid with --bus kafka (got --bus "
                f"{bus}). An in-process bus has no broker to address, so a "
                "lane selection here would be silently ignored."
            )
            raise DelegateLaneSelectionError(message)
        return None

    if lane is not None and kafka_bootstrap is not None:
        message = (
            f"--lane {lane!r} and --kafka-bootstrap {kafka_bootstrap!r} both "
            "name a broker for this delegation. Pass one: --lane resolves the "
            "address from the lane declaration, --kafka-bootstrap states it "
            "directly (for a lane-internal address no declaration can carry, "
            "such as a container on the lane's own compose network)."
        )
        raise DelegateLaneSelectionError(message)

    if kafka_bootstrap is not None:
        return None

    if lane is None:
        raise DelegateLaneSelectionError(_incomplete_selection_message(omni_home))

    return resolve_lane_selection(lane=lane, omni_home=omni_home)
