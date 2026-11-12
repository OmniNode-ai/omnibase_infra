# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The policy of the automatic version-predecessor handover (OMN-19998).

Auto-wiring joins each contract topic under a consumer group id built from
the contract version, so a version bump changes the group id, and the new
group has no committed offset and starts at the bus default, which drops
every record published between the old group stopping and the new one
joining. The version handover closes that window by seeding the successor
group from every predecessor group of a different contract version, with
the same rules and fence a declared consumer-group handover applies.

The policy is declared as a JSON object in an environment variable, with
these defaults::

    {"enabled": true, "from_auto_offset_reset": "latest",
     "fence": {"wait_seconds": 60, "poll_seconds": 2}}
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.event_bus.model_consumer_group_fence import (
    ModelConsumerGroupFence,
)

__all__ = [
    "VERSION_HANDOVER_ENV_VAR",
    "ModelVersionHandoverPolicy",
    "load_version_handover_policy",
    "select_version_predecessors",
]

VERSION_HANDOVER_ENV_VAR = "ONEX_RUNTIME_VERSION_HANDOVER"

_PURPOSE_MARKER = ".consume."
_SCOPE_INFIXES = (".__i.", ".__t.")


class ModelVersionHandoverPolicy(BaseModel):
    """Where a version bump seeds the new group from, and how long to fence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = Field(
        default=True,
        description="Take over the predecessor groups' committed offsets "
        "before the successor joins.",
    )
    from_auto_offset_reset: Literal["earliest", "latest"] = Field(
        default="latest",
        description="Where a predecessor started on a partition it never "
        "committed, as the handover's source side.",
    )
    fence: ModelConsumerGroupFence = Field(
        default_factory=ModelConsumerGroupFence,
        description="How long each predecessor group must be confirmed "
        "memberless before its position is taken over.",
    )


def load_version_handover_policy(
    environ: Mapping[str, str],
) -> ModelVersionHandoverPolicy:
    """The version handover policy of this process.

    Unset or blank is the model's defaults. Otherwise the variable is a JSON
    object validated as ``ModelVersionHandoverPolicy``.

    Raises:
        ValueError: The variable is not valid JSON, not an object, or not a
            valid policy; an unreadable policy must not read as absent, which
            would silently run the default the operator tried to change.
    """
    raw = environ.get(VERSION_HANDOVER_ENV_VAR, "").strip()
    if not raw:
        return ModelVersionHandoverPolicy()
    try:
        declared = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{VERSION_HANDOVER_ENV_VAR} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(declared, dict):
        raise ValueError(f"{VERSION_HANDOVER_ENV_VAR} must be a JSON object")
    return ModelVersionHandoverPolicy.model_validate(declared)


def _split_group(group_id: str) -> tuple[str, str, str] | None:
    """``(prefix, version, scope suffix)`` of a versioned group id, or ``None``."""
    prefix, marker, rest = group_id.partition(_PURPOSE_MARKER)
    if not marker or not prefix or not rest:
        return None
    cut = min(
        (index for infix in _SCOPE_INFIXES if (index := rest.find(infix)) >= 0),
        default=len(rest),
    )
    version, suffix = rest[:cut], rest[cut:]
    if not version:
        return None
    return prefix, version, suffix


def select_version_predecessors(
    successor_group: str, listed_groups: Iterable[str]
) -> tuple[str, ...]:
    """Listed groups that differ from ``successor_group`` only in the version.

    The version segment contains dots (``1.2.0``), so the id is split on the
    first ``.consume.`` and the scope suffix starts at the first ``.__i.`` or
    ``.__t.`` infix. A group qualifies when its prefix and suffix equal the
    successor's and its version differs. The successor itself never qualifies.
    """
    successor = _split_group(successor_group)
    if successor is None:
        return ()
    prefix, version, suffix = successor
    found: set[str] = set()
    for listed in listed_groups:
        parts = _split_group(listed)
        if parts is None:
            continue
        if parts[0] == prefix and parts[2] == suffix and parts[1] != version:
            found.add(listed)
    return tuple(sorted(found))
