# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Which lane is this runtime (OMN-18769).

A runtime's health verdict is only usable per-lane if the runtime can name its
own lane. Before this module nothing in the process could: the compose project
name, the k3s namespace and the overlay are all facts about the DEPLOYMENT, and
none of them is visible from inside the container.

The deployment states its lane in ``ONEX_RUNTIME_LANE``. OMN-19812 separates
placement from health keying: a runtime may name its lane for placement without
speaking for the lane's health, using ``ONEX_RUNTIME_LANE_HEALTH_SPEAKER=false``.

**Why this returns ``None`` instead of raising.** Operating rule 8 says fail
fast on missing env rather than pick a silent default, and that rule is about
*defaults*: a wrong default is worse than a crash. There is no default here.
``None`` is not a default, it is the honest answer "this deployment did not say"
-- and it is propagated as ``None`` all the way to the consumer, which DROPS the
event rather than attributing it to a lane. Raising instead would take every
already-deployed runtime that predates the variable and stop it emitting health
events at all, which converts an observability improvement into an outage.

**Why the value set is closed.** An unrecognised lane name is refused
(``None``) rather than passed through. The consumer keys a row on this value;
a typo would silently mint a phantom lane that looks exactly like a real one.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from functools import lru_cache

from omnibase_core.constants.constants_runtime_lanes import (
    LAB_RUNTIME_LANES,
    REGISTERED_RUNTIME_LANES,
)

logger = logging.getLogger(__name__)

#: The environment variable a deployment sets to name its own lane.
ENV_RUNTIME_LANE = "ONEX_RUNTIME_LANE"

#: Whether this runtime speaks for its lane's health (OMN-19812).
ENV_RUNTIME_LANE_HEALTH_SPEAKER = "ONEX_RUNTIME_LANE_HEALTH_SPEAKER"

#: The lanes a runtime's HEALTH may be keyed on. Deliberately the LAB set and
#: nothing else (OMN-18769 AC6): the stability-test, judge and collaborator
#: lanes are read-only surfaces, and a runtime there has no business publishing
#: a lane-keyed health fact that a dashboard would then present as actionable.
#:
#: This is NOT the set of lanes a runtime may declare. Since OMN-19408 every
#: deployment may name itself from ``REGISTERED_RUNTIME_LANES`` (core) so that
#: a lane-scoped node contract can refuse to attach there; see
#: :func:`resolve_declared_runtime_lane`. A stability-test runtime names its
#: lane for placement and still publishes its health with no lane.
KNOWN_LANES: frozenset[str] = LAB_RUNTIME_LANES


@lru_cache(maxsize=1)
def _warn_absent_lane() -> None:
    """Announce, ONCE for this process, that no lane was declared.

    OMN-19144. The absent case is the one that actually happened and it was the
    only one this module said nothing about: no deployment on any lane declared
    the variable, so every health event the lab dev lane emitted carried
    ``lane: null``, the lane-keyed projection dropped each one because a fact
    whose lane cannot be named cannot be keyed, and the offsets committed
    anyway. Consumer lag read 0 over a total loss. The consumer cannot tell a
    lane-less emitter from a lane it does not hold; this process can, and is
    the only thing in a position to say so.

    Once per process, not once per call. The resolver runs on every health
    emit, which is every check interval for the life of the container, and a
    warning at that volume is the one that gets filtered out -- which is worse
    than none, because a filtered line still reads as coverage. The fact being
    reported is a property of the deployment, so stating it at first resolution
    says everything a repeat would.
    """
    logger.warning(
        "%s is not set — this runtime cannot name its lane, so its health "
        "events carry lane=null and every lane-keyed consumer DROPS them "
        "silently, and every lane-scoped node contract fails closed. Set it "
        "to this deployment's lane, one of %s, in the deployment that runs "
        "this process; only the lab lanes %s key a health row "
        "(OMN-19144, OMN-19408).",
        ENV_RUNTIME_LANE,
        sorted(REGISTERED_RUNTIME_LANES),
        sorted(KNOWN_LANES),
    )


@lru_cache(maxsize=8)
def _note_non_lab_lane(lane: str) -> None:
    """Say ONCE per process that a registered non-lab lane keys no health row.

    OMN-19408. A stability-test runtime declares its lane so that lane-scoped
    node contracts can refuse to attach there. That declaration is the
    deployment telling the truth, not a typo, so it must not trip the
    per-emit "not a known lane" warning below -- a warning every check
    interval for the life of the container is the volume that gets filtered.
    """
    logger.info(
        "%s=%r is a registered runtime lane but not a lab lane %s; this "
        "runtime's health events carry no lane, by OMN-18769 AC6",
        ENV_RUNTIME_LANE,
        lane,
        sorted(KNOWN_LANES),
    )


@lru_cache(maxsize=1)
def _note_health_non_speaker() -> None:
    """Say ONCE per process that this runtime leaves lane health to its speaker."""
    logger.info(
        "%s=false — this runtime names its lane for placement only; its "
        "health events carry no lane (OMN-19812)",
        ENV_RUNTIME_LANE_HEALTH_SPEAKER,
    )


@lru_cache(maxsize=8)
def _warn_invalid_health_speaker(value: str) -> None:
    """Say ONCE per process and value that the health-speaker value is refused."""
    logger.warning(
        "%s=%r is invalid; accepted values are absent/blank, 'true', or "
        "'false' — emitting no health lane (OMN-19812)",
        ENV_RUNTIME_LANE_HEALTH_SPEAKER,
        value,
    )


def resolve_declared_runtime_lane(
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Return the lane this runtime's deployment declares, for PLACEMENT.

    OMN-19408. The value a lane-scoped node contract (``runtime_lanes``) is
    checked against. Any registered lane is accepted -- stability-test
    included -- because the question here is "which deployment is this", not
    "which lab lane-health row is this".
    The health-speaker setting does not affect placement (OMN-19812).

    Returns ``None`` when the variable is absent, blank or not a registered
    lane. ``None`` is not a default: the auto-wiring ownership filter treats
    it as "cannot name its lane" and fails closed for every lane-scoped
    contract, recording a discovery error rather than attaching or silently
    skipping.

    Args:
        environ: Override for the process environment. Injected by tests; the
            default reads ``os.environ``.
    """
    source: Mapping[str, str] = os.environ if environ is None else environ
    raw = (source.get(ENV_RUNTIME_LANE) or "").strip().lower()
    if raw in REGISTERED_RUNTIME_LANES:
        return raw
    return None


def describe_undeclared_runtime_lane(environ: Mapping[str, str] | None = None) -> str:
    """Say why :func:`resolve_declared_runtime_lane` returned ``None``.

    For the fail-closed discovery error a lane-scoped contract records on a
    runtime that cannot name its lane: absent and misspelt are different fixes.
    """
    source: Mapping[str, str] = os.environ if environ is None else environ
    raw = (source.get(ENV_RUNTIME_LANE) or "").strip()
    if not raw:
        return f"{ENV_RUNTIME_LANE} is not set"
    return f"{ENV_RUNTIME_LANE}={raw!r} is not a registered lane"


def resolve_runtime_lane(environ: Mapping[str, str] | None = None) -> str | None:
    """Return the lab lane this runtime speaks for, or ``None`` for unkeyed health.

    OMN-19812: ``ONEX_RUNTIME_LANE_HEALTH_SPEAKER=false`` disables health
    keying while preserving placement. Absent, blank or ``true`` keeps the
    current behavior; any other value refuses to speak for health.

    Args:
        environ: Override for the process environment. Injected by tests; the
            default reads ``os.environ``.
    """
    source: Mapping[str, str] = os.environ if environ is None else environ
    speaker = (source.get(ENV_RUNTIME_LANE_HEALTH_SPEAKER) or "").strip().lower()
    if speaker == "false":
        _note_health_non_speaker()
        return None
    if speaker not in ("", "true"):
        _warn_invalid_health_speaker(speaker)
        return None
    raw = (source.get(ENV_RUNTIME_LANE) or "").strip().lower()
    if not raw:
        _warn_absent_lane()
        return None
    if raw in REGISTERED_RUNTIME_LANES and raw not in KNOWN_LANES:
        _note_non_lab_lane(raw)
        return None
    if raw not in KNOWN_LANES:
        logger.warning(
            "%s=%r is not a known lane %s — emitting no lane rather than a phantom one",
            ENV_RUNTIME_LANE,
            raw,
            sorted(KNOWN_LANES),
        )
        return None
    return raw


__all__: list[str] = [
    "ENV_RUNTIME_LANE",
    "ENV_RUNTIME_LANE_HEALTH_SPEAKER",
    "KNOWN_LANES",
    "describe_undeclared_runtime_lane",
    "resolve_declared_runtime_lane",
    "resolve_runtime_lane",
]
