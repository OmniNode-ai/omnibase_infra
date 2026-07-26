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

Lane-scoped secret-policy override (OMN-14951):
    ``RUNTIME_PROFILE`` encodes topic-ownership ROLE identity (main / effects /
    workers / projection-api / canary) and consumers depend on that identity
    for routing decisions -- it must never be repurposed to also carry
    secret-gating semantics (see the "Runtime lane profiles must preserve
    identity" comment on ``_PROFILES`` below). Because every role-based
    profile above hardcodes ``prefetch_policy="disabled"``, ``"required"`` is
    structurally unreachable in any deployed lane today: dev/stability/judge/
    prod are all generated from the same role manifests, and the
    ``"production"``/``"staging"`` profile names are never referenced by any
    docker-catalog service manifest.

    ``ONEX_SECRET_POLICY`` is a second, independent env var -- lane-scoped,
    not role-scoped -- that overrides the resolved profile's
    ``prefetch_policy`` without touching ``RUNTIME_PROFILE`` or its role
    identity. Set it once per lane (e.g. in the lane's
    ``docker/runtime-policy.env`` block), not per role, so every role running
    in a ``required`` lane inherits the fail-loud policy regardless of which
    role-named ``RUNTIME_PROFILE`` it boots with.
"""

from __future__ import annotations

import logging
import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# OMN-14951: lane-scoped override, independent of RUNTIME_PROFILE. See module
# docstring "Lane-scoped secret-policy override" section for why this must
# stay a separate env var rather than reusing/extending RUNTIME_PROFILE.
_SECRET_POLICY_ENV_VAR = "ONEX_SECRET_POLICY"
_VALID_PREFETCH_POLICIES = frozenset({"disabled", "best_effort", "required"})


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


# The infra runtime owns each profile's *behaviour* (prefetch policy); the
# canonical *name set* lives in omnibase_core (OMN-12957) so contract validation
# can enforce ``runtime_profiles`` membership without a core->infra dependency.
# The two must stay in lockstep: a name core blesses that infra cannot boot — or
# an infra profile core does not know about — is a silent-orphan hazard. The
# parity guard is the test ``test_profiles_match_core_registry`` (a hard import-
# time raise here would crash the runtime kernel on any core/infra version skew,
# so the invariant is enforced at test/CI time instead of at import).


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

    # OMN-14951: ONEX_SECRET_POLICY is a lane-scoped override, independent of
    # RUNTIME_PROFILE's role identity. See module docstring.
    override_raw = os.getenv(_SECRET_POLICY_ENV_VAR)
    if override_raw:
        override = override_raw.strip().lower()
        if override in _VALID_PREFETCH_POLICIES:
            if override != profile.prefetch_policy:
                logger.info(
                    "%s=%s overrides profile %r prefetch_policy=%r "
                    "(role identity unchanged)",
                    _SECRET_POLICY_ENV_VAR,
                    override,
                    profile.name,
                    profile.prefetch_policy,
                )
                profile = profile.model_copy(update={"prefetch_policy": override})
        else:
            logger.warning(
                "Invalid %s=%r (expected one of %s) — ignoring override, using "
                "profile %r prefetch_policy=%r",
                _SECRET_POLICY_ENV_VAR,
                override_raw,
                sorted(_VALID_PREFETCH_POLICIES),
                profile.name,
                profile.prefetch_policy,
            )
    return profile


def resolve_secret_resolver_config_path() -> str:
    """Resolve ``ONEX_SECRET_RESOLVER_CONFIG_PATH`` (OMN-14951).

    This module is the config-resolution boundary
    (``scripts/check-env-reads.sh``'s allowlist) for lane-scoped runtime
    policy env vars -- ``RUNTIME_PROFILE`` and ``ONEX_SECRET_POLICY`` are
    already read here. The rendered secret-resolver config path is the same
    class of read (a deploy-time-rendered artifact path,
    ``render_secret_resolver_config.py``'s output), so it belongs at this
    same boundary rather than as a new raw ``os.environ`` read scattered into
    ``runtime_host_process.py``.

    Returns:
        The configured path (stripped), or ``""`` if unset/blank.
    """
    return os.environ.get("ONEX_SECRET_RESOLVER_CONFIG_PATH", "").strip()


__all__ = [
    "ModelRuntimeProfile",
    "load_runtime_profile",
    "resolve_secret_resolver_config_path",
]
