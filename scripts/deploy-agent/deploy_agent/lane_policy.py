# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lane fence for the deploy agent (OMN-16939).

One deploy-agent process consumes ONE control bus, and the dev control bus
deliberately carries commands for more than one runtime lane: the CI publisher
(``omnibase_infra/.github/workflows/runtime-rebuild-trigger.yml``) routes both
``dev`` and ``stability-test`` rebuild requests through the dev broker and lets
``runtime_lane`` in the signed payload decide the deployment target.

That means an agent attached to the dev broker will happily execute a
``stability-test`` rebuild, and an agent attached to the prod broker will
happily execute anything that lands there. The process itself carries no
statement of which lanes it is allowed to touch, so its blast radius is
whatever a publisher chooses to put on its topic.

``DEPLOY_AGENT_ALLOWED_LANES`` makes that statement explicit and mechanical.
It is REQUIRED and fail-closed: an unset, empty, or unparseable value aborts
startup rather than defaulting to "every lane", because a silent default is
exactly the failure this fence exists to remove.
"""

from __future__ import annotations

import os

from deploy_agent.events import EnumRuntimeLane
from deploy_agent.tracking_ref import ENV_TRACKING_REF

ENV_ALLOWED_LANES = "DEPLOY_AGENT_ALLOWED_LANES"


class LaneNotAllowedError(RuntimeError):
    """Raised when a command targets a lane this agent may not deploy."""


def parse_allowed_lanes(raw: str) -> frozenset[EnumRuntimeLane]:
    """Parse a comma-separated allow-list into runtime lanes.

    Raises ``ValueError`` on an empty list or an unknown lane name — never
    silently drops an entry, because a typo would otherwise narrow the fence
    without anyone noticing.
    """
    tokens = [token.strip() for token in raw.split(",")]
    named = [token for token in tokens if token]
    if not named:
        raise ValueError(
            f"{ENV_ALLOWED_LANES} is empty; name at least one runtime lane "
            f"(one of: {', '.join(lane.value for lane in EnumRuntimeLane)})"
        )
    lanes: set[EnumRuntimeLane] = set()
    for token in named:
        try:
            lanes.add(EnumRuntimeLane(token))
        except ValueError as exc:
            raise ValueError(
                f"{ENV_ALLOWED_LANES} names unknown runtime lane {token!r}; "
                f"known lanes: {', '.join(lane.value for lane in EnumRuntimeLane)}"
            ) from exc
    return frozenset(lanes)


def load_allowed_lanes_from_env() -> frozenset[EnumRuntimeLane]:
    """Load the required lane allow-list from the environment.

    There is intentionally no default. A deploy agent that does not declare
    which lanes it may deploy is a deploy agent whose blast radius is decided
    by whoever publishes to its topic.
    """
    raw = os.environ.get(ENV_ALLOWED_LANES, "").strip()
    if not raw:
        raise RuntimeError(
            f"{ENV_ALLOWED_LANES} is required for deploy-agent; it declares which "
            "runtime lanes this process may deploy. There is no default: an "
            "undeclared fence means the control bus decides the blast radius. "
            "Set it on the systemd unit (e.g. DEPLOY_AGENT_ALLOWED_LANES=dev)."
        )
    return parse_allowed_lanes(raw)


def assert_lane_allowed(
    lane: EnumRuntimeLane, allowed: frozenset[EnumRuntimeLane]
) -> None:
    """Raise ``LaneNotAllowedError`` when ``lane`` is outside the fence."""
    if lane not in allowed:
        raise LaneNotAllowedError(
            f"rebuild command targets runtime_lane={lane.value!r}, which is not in "
            f"this agent's {ENV_ALLOWED_LANES}="
            f"{','.join(sorted(item.value for item in allowed))}"
        )


def resolve_default_runtime_lane_from_env() -> EnumRuntimeLane:
    """Resolve the lane an operator command targets when ``--runtime-lane`` is omitted.

    OMN-16442. ``ModelRebuildRequested.runtime_lane`` is REQUIRED and has no
    default, which is correct: the field decides which lane a command mutates,
    and a literal default in the model would hand every caller the same blast
    radius. But the operator trigger runs *inside* a lane's own environment,
    which already declares that lane twice over, so requiring the flag on every
    invocation would be ceremony rather than safety.

    Resolution order, both sources DECLARED and neither a literal:

    1. ``DEPLOY_AGENT_ALLOWED_LANES`` when it names exactly ONE lane. This is
       the strongest statement available -- it is the process's own fence
       (OMN-16939), and a single-lane fence leaves no ambiguity about which
       lane a command from this environment is for. A multi-lane fence is
       ambiguous by construction and does NOT resolve.
    2. ``DEPLOY_AGENT_TRACKING_REF`` when the branch it names is also a lane
       name (the dev unit declares ``dev`` for both). A tracking ref is a
       branch, not a lane, so this only resolves when the two coincide.

    Anything else raises, naming both variables and the flag. There is no
    fallback lane: guessing ``dev`` here is the same class of defect as the
    ``origin/main`` literal OMN-16442 removed from the tracking ref.
    """
    raw_lanes = os.environ.get(ENV_ALLOWED_LANES, "").strip()
    if raw_lanes:
        lanes = parse_allowed_lanes(raw_lanes)
        if len(lanes) == 1:
            return next(iter(lanes))

    raw_ref = os.environ.get(ENV_TRACKING_REF, "").strip()
    if raw_ref:
        try:
            return EnumRuntimeLane(raw_ref)
        except ValueError:
            pass

    raise RuntimeError(
        "cannot resolve a default runtime_lane for this command: "
        f"{ENV_ALLOWED_LANES}={raw_lanes!r} does not name exactly one lane and "
        f"{ENV_TRACKING_REF}={raw_ref!r} does not name a lane "
        f"(known lanes: {', '.join(lane.value for lane in EnumRuntimeLane)}). "
        "Pass --runtime-lane explicitly rather than let the trigger guess which "
        "lane it is about to rebuild."
    )
