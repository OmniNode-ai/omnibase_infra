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
into ``_PROFILES``.  ``load_runtime_profile()`` resolves a name to a profile
and REFUSES an unregistered one (OMN-17985); ``resolve_runtime_profile_name()``
is the single validated read of the variable for ownership decisions.

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

from omnibase_infra.errors import ProtocolConfigurationError

logger = logging.getLogger(__name__)

# OMN-17985: the ownership default when RUNTIME_PROFILE is unset. This is the
# value the auto-wiring ownership filter has always used, kept exactly so that
# consolidating the reads changes validation and nothing else.
_OWNERSHIP_DEFAULT_PROFILE = "main"

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
    # OMN-17556: the consolidated TENANT-domain projection writer. It is the
    # ONE process holding the tenant_projection binding's store-resolved
    # credential, so every contract that resolves that binding names this
    # profile and nothing else. Role identity matters here for the same reason
    # it matters for effects/workers: the resolved profile name is what decides
    # ownership, and falling through to "default" would put a second claimant
    # on main-owned topics. prefetch_policy stays "disabled" like every other
    # role-based profile -- the credential is resolved at the binding boundary
    # through SecretResolver, not prefetched into the kernel.
    "tenant-projection": ModelRuntimeProfile(
        name="tenant-projection",
        prefetch_policy="disabled",
    ),
    # OMN-17985: the seven STANDALONE projection writers already deployed on
    # onex-dev. Unlike every profile above, the name is NOT what wires these
    # processes: each is a `python -m <handler>` BaseProjectionRunner with its
    # own explicit KAFKA_CONSUMER_GROUP, and the runner never reads
    # RUNTIME_PROFILE. What the name settles is OWNERSHIP -- until it existed,
    # no contract could declare it, so each writer's contract was ALSO claimed
    # by a shared runtime (two by `effects`, five by `main` through the
    # undeclared-defaults-to-main rule).
    #
    # They must exist here, not only in core's registry, for two mechanical
    # reasons this repo enforces: `test_profiles_match_core_registry` asserts
    # exact set equality with REGISTERED_RUNTIME_PROFILES, and
    # `test_consumer_attached_profiles_actually_load` requires every
    # consumer-attached name to resolve through `load_runtime_profile`. Since
    # OMN-17985 made that function refuse an unknown name, a name core blesses
    # that this dict lacks is now a hard boot failure rather than a silent
    # fallback -- which is exactly the drift the parity guard exists to catch.
    #
    # prefetch_policy stays "disabled" like every other role-based profile.
    "projection-writer-delegation": ModelRuntimeProfile(
        name="projection-writer-delegation",
        prefetch_policy="disabled",
    ),
    "projection-writer-hook-ledger": ModelRuntimeProfile(
        name="projection-writer-hook-ledger",
        prefetch_policy="disabled",
    ),
    "projection-writer-live-events": ModelRuntimeProfile(
        name="projection-writer-live-events",
        prefetch_policy="disabled",
    ),
    "projection-writer-registration": ModelRuntimeProfile(
        name="projection-writer-registration",
        prefetch_policy="disabled",
    ),
    "projection-writer-savings": ModelRuntimeProfile(
        name="projection-writer-savings",
        prefetch_policy="disabled",
    ),
    "projection-writer-tenant-credentials": ModelRuntimeProfile(
        name="projection-writer-tenant-credentials",
        prefetch_policy="disabled",
    ),
    "projection-writer-tenant-registry": ModelRuntimeProfile(
        name="projection-writer-tenant-registry",
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


def _resolve_profile(raw: str) -> ModelRuntimeProfile:
    """Normalize *raw* and return its profile, or refuse (OMN-17985)."""
    name = raw.strip().lower()
    profile = _PROFILES.get(name)
    if profile is None:
        raise ProtocolConfigurationError(
            f"Unknown RUNTIME_PROFILE {name!r}. A runtime profile is ROLE "
            f"IDENTITY: the auto-wiring ownership filter keeps a contract only "
            f"when the profile is named in the contract's `runtime_profiles` "
            f"list, or the profile is 'main'. An unregistered name satisfies "
            f"neither, so every contract would be skipped and the process would "
            f"wire nothing while still passing readiness. Known profiles: "
            f"{sorted(_PROFILES)}"
        )
    return profile


def resolve_runtime_profile_name() -> str:
    """Return the validated ROLE IDENTITY carried by ``RUNTIME_PROFILE``.

    This is the single resolution point for every reader of that variable that
    needs an ownership identity (OMN-17985). Before it existed the variable was
    read raw in a dozen places with two different fallbacks -- ``"main"`` at the
    auto-wiring ownership filter, ``"default"`` at the boot banner -- so the
    process could describe itself as one role while wiring as another, and
    neither read validated the value at all.

    Unset or blank resolves to ``"main"``, which is the ownership default the
    auto-wiring filter has always applied; consolidating the reads is meant to
    add validation, not to move that default. An unknown value is refused on the
    same terms as :func:`load_runtime_profile`.

    Returns:
        The normalized, registered profile name.

    Raises:
        ProtocolConfigurationError: If the value is not a registered profile.
    """
    raw = os.getenv("RUNTIME_PROFILE") or _OWNERSHIP_DEFAULT_PROFILE
    if not raw.strip():
        raw = _OWNERSHIP_DEFAULT_PROFILE
    return _resolve_profile(raw).name


def load_runtime_profile(profile_name: str | None = None) -> ModelRuntimeProfile:
    """Return the ``ModelRuntimeProfile`` for *profile_name*.

    If *profile_name* is ``None`` the ``RUNTIME_PROFILE`` environment variable
    is consulted; unset or blank resolves to the ``"default"`` profile.

    An UNKNOWN name is refused (OMN-17985). It used to fall back to
    ``"default"`` with a warning, which discarded the role identity the value
    carries while the process went on to pass readiness. Because the auto-wiring
    ownership filter separately keeps a contract only when the profile is in the
    contract's declared list or the profile is ``"main"``, an unregistered name
    matches neither: every contract is skipped, the manifest empties, zero
    subscriptions are wired, and nothing downstream reports an error. A role the
    runtime cannot resolve is not a milder role, it is an unknown deployment, so
    the boot fails instead.

    Args:
        profile_name: Explicit override; defaults to ``RUNTIME_PROFILE`` env var.

    Returns:
        ``ModelRuntimeProfile`` for the resolved name.

    Raises:
        ProtocolConfigurationError: If the resolved name is not in ``_PROFILES``.
    """
    raw = profile_name or os.getenv("RUNTIME_PROFILE") or "default"
    profile = _resolve_profile(raw)

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
    "resolve_runtime_profile_name",
    "resolve_secret_resolver_config_path",
]
