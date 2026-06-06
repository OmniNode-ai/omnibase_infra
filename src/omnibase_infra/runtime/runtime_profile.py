# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime profile schema for ONEX kernel bootstrap.

Defines ``ModelRuntimeProfile`` which gates optional subsystems (e.g.
``ConfigPrefetcher``) based on the deployment environment.  The profile is
loaded once during bootstrap, keyed by the ``RUNTIME_PROFILE`` environment
variable, and consulted by any subsystem that needs to vary its behaviour
across local-dev / staging / production.

Profiles:
    local-dev  -- No external dependencies assumed; all optional subsystems
                  disabled by default so the runtime boots offline.
    main       -- Primary event-orchestration runtime.
    effects    -- Effect-lane runtime for effect-owned contracts and consumers.
    workers    -- Worker runtime profile used by the runtime-worker service.
    projection-api -- Projection API runtime profile.
    canary     -- Canary runtime profile for isolated contract experiments.
    staging    -- Best-effort mode; prefetcher runs but missing secrets are
                  logged as warnings and boot continues.
    production -- Strict mode; missing required secrets cause a hard failure.

The ``prefetch_policy`` field governs ``ConfigPrefetcher`` wiring:

    * ``"disabled"``     -- Prefetcher is not invoked.
    * ``"best_effort"``  -- Prefetcher runs; errors / missing keys are logged
                           as structured warnings and boot continues.
    * ``"required"``     -- Prefetcher runs; any missing or errored key raises
                           a ``ProtocolConfigurationError`` with the full list
                           of missing key names.

Profile data is built at module import time (``_PROFILES``).  New profiles
can be registered by constructing a ``ModelRuntimeProfile`` and inserting it
into ``_PROFILES``; the ``load_runtime_profile()`` helper handles fallback.
"""

from __future__ import annotations

import logging
import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class ModelRuntimeProfile(BaseModel):
    """Schema for a named runtime deployment profile.

    Attributes:
        name: Canonical profile name (e.g. ``"local-dev"``, ``"production"``).
        prefetch_policy: How ``ConfigPrefetcher`` behaves during kernel boot.
            ``"disabled"`` skips prefetch entirely.
            ``"best_effort"`` runs prefetch but tolerates missing keys.
            ``"required"`` runs prefetch and raises on any missing key.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., min_length=1)
    prefetch_policy: Literal["disabled", "best_effort", "required"] = Field(
        default="disabled"
    )


# Built-in profile definitions.
_PROFILES: dict[str, ModelRuntimeProfile] = {
    "local-dev": ModelRuntimeProfile(
        name="local-dev",
        prefetch_policy="disabled",
    ),
    # "default" is an alias for local-dev so zero-config local runs work.
    "default": ModelRuntimeProfile(
        name="default",
        prefetch_policy="disabled",
    ),
    # "main" matches the auto-wiring RUNTIME_PROFILE default.
    "main": ModelRuntimeProfile(
        name="main",
        prefetch_policy="disabled",
    ),
    # Runtime lane profiles must preserve identity. Consumers use the resolved
    # profile name to decide ownership; falling back to "default" can subscribe
    # secondary runtimes to main-owned workflow topics.
    "effects": ModelRuntimeProfile(
        name="effects",
        prefetch_policy="disabled",
    ),
    "workers": ModelRuntimeProfile(
        name="workers",
        prefetch_policy="disabled",
    ),
    "projection-api": ModelRuntimeProfile(
        name="projection-api",
        prefetch_policy="disabled",
    ),
    "canary": ModelRuntimeProfile(
        name="canary",
        prefetch_policy="disabled",
    ),
    "staging": ModelRuntimeProfile(
        name="staging",
        prefetch_policy="best_effort",
    ),
    "production": ModelRuntimeProfile(
        name="production",
        prefetch_policy="required",
    ),
}


def load_runtime_profile(profile_name: str | None = None) -> ModelRuntimeProfile:
    """Return the ``ModelRuntimeProfile`` for *profile_name*.

    If *profile_name* is ``None`` the ``RUNTIME_PROFILE`` environment variable
    is consulted.  Unknown names fall back to the ``"default"`` profile and a
    warning is emitted so operators can detect misconfiguration without
    crashing the runtime.

    Args:
        profile_name: Explicit override; defaults to ``RUNTIME_PROFILE`` env var.

    Returns:
        ``ModelRuntimeProfile`` for the resolved name.
    """
    raw = profile_name or os.getenv("RUNTIME_PROFILE") or "default"
    name = raw.strip().lower()
    profile = _PROFILES.get(name)
    if profile is None:
        logger.warning(
            "Unknown RUNTIME_PROFILE %r — falling back to 'default' profile "
            "(prefetch_policy=disabled).  Known profiles: %s",
            name,
            list(_PROFILES),
        )
        profile = _PROFILES["default"]
    return profile


__all__ = [
    "ModelRuntimeProfile",
    "load_runtime_profile",
]
