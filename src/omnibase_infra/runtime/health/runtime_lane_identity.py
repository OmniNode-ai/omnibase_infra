# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Which lane is this runtime, and what is it for (OMN-18769, OMN-19747).

Operator ruling 2026-09-26 (firm): the set of runtime lanes and each lane's
role are declared in a deployment overlay that whoever runs the runtime
supplies. They are not compiled into a package and not read from our lab's
files, because this architecture ships to customers who have no access to our
lab.

So a runtime learns its lane in two steps, both at startup:

1. **Identity.** The deployment names the lane in ``ONEX_RUNTIME_LANE`` and its
   environment in ``ONEX_ENVIRONMENT``, the bootstrap identity it already sets.
2. **Declaration.** The runtime reads the ``runtime.lane`` overlay document at
   the scope ``(environment, lane)`` from its one overlay source, and core's
   :meth:`ModelRuntimeLaneDeclaration.resolve` validates it.

Every way that can fail raises :class:`ProtocolConfigurationError` and the
kernel refuses to start, naming what is missing. A runtime that cannot name
its lane never runs DEGRADED with its lane-scoped contracts silently dropped,
which is what runtimes absent from the compiled lane list did (OMN-19408).

OMN-19812 separates placement from health keying: a runtime may name its lane
for placement without speaking for the lane's health, using
``ONEX_RUNTIME_LANE_HEALTH_SPEAKER=false``.

The kernel resolves once and :func:`establish_runtime_lane` holds the result
for the life of the process; the auto-wiring ownership filter and the health
monitor read it back through :func:`established_runtime_lane`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

from omnibase_core.enums.enum_runtime_lane_role import EnumRuntimeLaneRole
from omnibase_core.models.config_overlay import (
    RUNTIME_LANE_ENV_VAR,
    ModelRuntimeLaneDeclaration,
)
from omnibase_infra.handlers.handler_runtime_lane_resolution import (
    HandlerRuntimeLaneResolution,
)
from omnibase_infra.models.model_runtime_lane_resolution import (
    ModelRuntimeLaneResolution,
)

logger = logging.getLogger(__name__)

#: The environment variable a deployment sets to name its own lane.
ENV_RUNTIME_LANE = RUNTIME_LANE_ENV_VAR

#: The environment variable a deployment sets to name its environment, the
#: first segment of every overlay scope.
ENV_RUNTIME_ENVIRONMENT = "ONEX_ENVIRONMENT"

#: Whether this runtime speaks for its lane's health (OMN-19812).
ENV_RUNTIME_LANE_HEALTH_SPEAKER = "ONEX_RUNTIME_LANE_HEALTH_SPEAKER"

#: The one runtime profile whose health is keyed on its lane (OMN-19144).
_HEALTH_SPEAKING_PROFILE = "main"

_established: list[ModelRuntimeLaneResolution] = []


def resolve_runtime_lane_declaration(
    *,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> ModelRuntimeLaneResolution:
    """Read the startup lane through the contract's filesystem handler."""
    return HandlerRuntimeLaneResolution().resolve(
        environ=os.environ if environ is None else environ, home=home
    )


def establish_runtime_lane(resolution: ModelRuntimeLaneResolution) -> None:
    """Hold the startup resolution for the life of this process."""
    _established[:] = [resolution]
    logger.info("Runtime lane resolved: %s", resolution.log_line())


def established_runtime_lane() -> ModelRuntimeLaneResolution | None:
    """The lane the kernel resolved at startup, or ``None`` before it has."""
    return _established[0] if _established else None


def clear_established_runtime_lane() -> None:
    """Forget the established lane. For tests only."""
    _established.clear()


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


def resolve_runtime_lane(
    declaration: ModelRuntimeLaneDeclaration | None = None,
    *,
    runtime_profile: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Return the lane a HEALTH fact is keyed on, or ``None``.

    OMN-18769 AC6, unchanged in effect: only a lab lane keys a health row. A
    lane is a lab lane when its overlay grants it the ``lab`` role, not when a
    package lists its name. A runtime on any other lane publishes its health
    with no lane.

    OMN-19144: one process speaks for a lane. The lane-health row is an upsert
    keyed by lane alone, so only the ``main`` profile keys its health; every
    other profile on the same lane publishes with no lane.

    OMN-19812: ``ONEX_RUNTIME_LANE_HEALTH_SPEAKER=false`` disables health
    keying while preserving placement. Absent, blank or ``true`` keeps the
    current behavior; any other value refuses to speak for health.

    Args:
        declaration: The lane to judge. Defaults to the established lane.
        runtime_profile: This process's runtime profile, when known. A profile
            other than ``main`` keys no health row.
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
    if runtime_profile is not None and runtime_profile != _HEALTH_SPEAKING_PROFILE:
        return None
    if declaration is None:
        resolution = established_runtime_lane()
        if resolution is None:
            return None
        declaration = resolution.declaration
    if declaration.has_roles((EnumRuntimeLaneRole.LAB,)):
        return declaration.lane_id
    return None


__all__: list[str] = [
    "ENV_RUNTIME_ENVIRONMENT",
    "ENV_RUNTIME_LANE",
    "ENV_RUNTIME_LANE_HEALTH_SPEAKER",
    "clear_established_runtime_lane",
    "establish_runtime_lane",
    "established_runtime_lane",
    "resolve_runtime_lane",
    "resolve_runtime_lane_declaration",
]
