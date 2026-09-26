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

The kernel resolves once and :func:`establish_runtime_lane` holds the result
for the life of the process; the auto-wiring ownership filter and the health
monitor read it back through :func:`established_runtime_lane`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path

from pydantic import ValidationError

from omnibase_core.enums.enum_config_overlay_key import EnumConfigOverlayKey
from omnibase_core.enums.enum_runtime_lane_role import EnumRuntimeLaneRole
from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_core.models.config_overlay import (
    RUNTIME_LANE_ENV_VAR,
    ModelConfigOverlayScope,
    ModelRuntimeLaneDeclaration,
)
from omnibase_infra.config_overlay.models import ModelRuntimeLaneResolution
from omnibase_infra.config_overlay.select_source import select_config_overlay_source
from omnibase_infra.errors import ProtocolConfigurationError

logger = logging.getLogger(__name__)

#: The environment variable a deployment sets to name its own lane.
ENV_RUNTIME_LANE = RUNTIME_LANE_ENV_VAR

#: The environment variable a deployment sets to name its environment, the
#: first segment of every overlay scope.
ENV_RUNTIME_ENVIRONMENT = "ONEX_ENVIRONMENT"

#: The one runtime profile whose health is keyed on its lane (OMN-19144).
_HEALTH_SPEAKING_PROFILE = "main"

_established: list[ModelRuntimeLaneResolution] = []


def resolve_runtime_lane_declaration(
    *,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> ModelRuntimeLaneResolution:
    """Resolve this runtime's lane from its overlay, or refuse naming why.

    Args:
        environ: Process environment override, for tests.
        home: Home directory override, for tests.

    Raises:
        ProtocolConfigurationError: the environment or lane is unset or not a
            scope segment, no overlay source (or both) is configured, the
            ``runtime.lane`` document is absent at the scope, invalid, or
            declares a different lane.
    """
    env: Mapping[str, str] = os.environ if environ is None else environ
    key = EnumConfigOverlayKey.RUNTIME_LANE
    environment = (env.get(ENV_RUNTIME_ENVIRONMENT) or "").strip()
    declared_lane = env.get(ENV_RUNTIME_LANE)
    lane = (declared_lane or "").strip()
    if not environment:
        raise ProtocolConfigurationError(
            f"{ENV_RUNTIME_ENVIRONMENT} is not set, so this runtime cannot name "
            f"the overlay scope its {key.value} document lives at and refuses to "
            "start. Set it in the deployment that runs this process."
        )
    if not _is_slug(environment):
        raise ProtocolConfigurationError(
            f"{ENV_RUNTIME_ENVIRONMENT}={environment!r} is not an overlay scope "
            "segment (lowercase letters, digits and hyphens, at most 64); this "
            "runtime refuses to start."
        )
    if not _is_slug(lane):
        # resolve() names an unset lane and a malformed one, each precisely,
        # before it looks at any document.
        try:
            ModelRuntimeLaneDeclaration.resolve(
                declared_lane_id=declared_lane,
                document=None,
                where=f"the {key.value} overlay at environment {environment!r}",
            )
        except ModelOnexError as exc:
            raise ProtocolConfigurationError(exc.message) from exc
    scope = ModelConfigOverlayScope(environment=environment, lane=lane)
    source = select_config_overlay_source(environ=env, home=home)
    document = source.get_document(key, scope)
    where = source.describe(key, scope)
    try:
        declaration = ModelRuntimeLaneDeclaration.resolve(
            declared_lane_id=declared_lane, document=document, where=where
        )
    except ModelOnexError as exc:
        raise ProtocolConfigurationError(exc.message) from exc
    if document is None:  # resolve() refuses a missing document; typing only
        raise ProtocolConfigurationError(f"no {key.value} document at {where}")
    return ModelRuntimeLaneResolution(
        declaration=declaration,
        scope=scope,
        source=source.source,
        location=str(source.path_for(key, scope)),
        sha256=document.sha256,
    )


def _is_slug(value: str) -> bool:
    """Whether ``value`` is one overlay scope segment."""
    try:
        ModelConfigOverlayScope(environment=value, lane=value)
    except ValidationError:
        return False
    return True


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


def resolve_runtime_lane(
    declaration: ModelRuntimeLaneDeclaration | None = None,
    *,
    runtime_profile: str | None = None,
) -> str | None:
    """Return the lane a HEALTH fact is keyed on, or ``None``.

    OMN-18769 AC6, unchanged in effect: only a lab lane keys a health row. A
    lane is a lab lane when its overlay grants it the ``lab`` role, not when a
    package lists its name. A runtime on any other lane publishes its health
    with no lane.

    OMN-19144: one process speaks for a lane. The lane-health row is an upsert
    keyed by lane alone, so only the ``main`` profile keys its health; every
    other profile on the same lane publishes with no lane.

    Args:
        declaration: The lane to judge. Defaults to the established lane.
        runtime_profile: This process's runtime profile, when known. A profile
            other than ``main`` keys no health row.
    """
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
    "clear_established_runtime_lane",
    "establish_runtime_lane",
    "established_runtime_lane",
    "resolve_runtime_lane",
    "resolve_runtime_lane_declaration",
]
