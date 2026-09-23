# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Which lane is this runtime (OMN-18769).

A runtime's health verdict is only usable per-lane if the runtime can name its
own lane. Before this module nothing in the process could: the compose project
name, the k3s namespace and the overlay are all facts about the DEPLOYMENT, and
none of them is visible from inside the container.

So the deployment states it, in one environment variable, and the runtime reads
it back.

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
from functools import lru_cache

logger = logging.getLogger(__name__)

#: The environment variable a deployment sets to name its own lane.
ENV_RUNTIME_LANE = "ONEX_RUNTIME_LANE"

#: The lanes a runtime may claim to be. Deliberately the LAB set and nothing
#: else: the stability-test, judge and collaborator lanes are read-only
#: surfaces, and a runtime there has no business publishing a lane-keyed health
#: fact that a dashboard would then present as actionable.
KNOWN_LANES: frozenset[str] = frozenset({"compose-dev", "onex-lab", "onex-lab-k3s"})


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
        "silently. Set it to one of %s in the deployment that runs this "
        "process (OMN-19144).",
        ENV_RUNTIME_LANE,
        sorted(KNOWN_LANES),
    )


def resolve_runtime_lane(environ: dict[str, str] | None = None) -> str | None:
    """Return this runtime's declared lane, or ``None`` when it has none.

    Args:
        environ: Override for the process environment. Injected by tests; the
            default reads ``os.environ``.
    """
    source = os.environ if environ is None else environ
    raw = (source.get(ENV_RUNTIME_LANE) or "").strip().lower()
    if not raw:
        _warn_absent_lane()
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


__all__: list[str] = ["ENV_RUNTIME_LANE", "KNOWN_LANES", "resolve_runtime_lane"]
