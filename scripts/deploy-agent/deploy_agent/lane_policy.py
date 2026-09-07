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
