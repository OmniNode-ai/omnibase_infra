# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime-profile and runtime-lane contract ownership filtering for auto-wiring.

Contracts may declare ``runtime_profiles`` to assign their Kafka subscriptions
to a specific runtime process. Unscoped legacy contracts default to ``main`` so
the effects/worker runtimes do not join general compute groups and steal work
from the primary runtime.

Contracts may also declare ``runtime_lanes`` (OMN-19408): the deployments the
node may attach on at all. A profile is a role INSIDE a lane; a lane is WHICH
deployment. The .201 stability-test main runtime is ``main`` exactly as the dev
lane's is, so only a lane scope can keep a lab-only node off it. Profile
ownership is decided first; a runtime that owns a lane-scoped contract by
profile then attaches it only when its declared lane (``ONEX_RUNTIME_LANE``,
any registered lane) is in the scope. A runtime that cannot name a registered
lane fails CLOSED: the contract is not attached and a discovery error naming it
is added to the manifest, which the runtime health monitor reports as a
DEGRADED ``discovery_errors`` dimension. It is never a silent skip.

Both the kernel's wiring and the health monitor's expectations read this one
filter, so what is attached and what health expects cannot disagree.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from omnibase_core.constants.constants_runtime_lanes import REGISTERED_RUNTIME_LANES
from omnibase_core.models.contracts.subcontracts.model_runtime_lane_scope import (
    ModelRuntimeLaneScope,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelDiscoveredContract,
    ModelDiscoveryError,
)
from omnibase_infra.runtime.health.runtime_lane_identity import (
    ENV_RUNTIME_LANE,
    describe_undeclared_runtime_lane,
    resolve_declared_runtime_lane,
)

logger = logging.getLogger(__name__)


def _normalize_runtime_profile(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("runtime_profile must be a string")
    profile = value.strip().lower()
    if not profile:
        raise ValueError("runtime_profile cannot be blank")
    return profile


def extract_runtime_profiles_from_contract(
    raw_contract: Mapping[str, object],
) -> tuple[str, ...]:
    """Return normalized runtime-profile ownership declared by raw contract YAML."""
    profiles_raw = raw_contract.get("runtime_profiles")
    descriptor_raw = raw_contract.get("descriptor")
    if profiles_raw is None and isinstance(descriptor_raw, Mapping):
        profiles_raw = descriptor_raw.get("runtime_profiles")

    if profiles_raw is None:
        return ()
    if isinstance(profiles_raw, str):
        raw_values = (profiles_raw,)
    elif isinstance(profiles_raw, (list, tuple)):
        raw_values = tuple(profiles_raw)
    else:
        raise TypeError("runtime_profiles must be a string or sequence of strings")

    profiles: list[str] = []
    for raw in raw_values:
        profiles.append(_normalize_runtime_profile(raw))
    return tuple(dict.fromkeys(profiles))


def _runtime_lane_admits_raw_contract(
    raw_contract: Mapping[str, object],
    environ: Mapping[str, str] | None,
) -> bool:
    """Return whether this runtime's lane admits a raw contract's lane scope.

    Fail-closed, like the manifest filter: an unparseable scope or an
    undeclared runtime lane never admits a lane-scoped contract. The manifest
    filter is where the error is RECORDED; this raw path serves the legacy
    runtime-host subscription loops and only has to refuse.
    """
    lanes_raw = raw_contract.get("runtime_lanes")
    if lanes_raw is None:
        return True
    try:
        scope = ModelRuntimeLaneScope.model_validate({"lanes": lanes_raw})
    except ValidationError:
        return False
    return scope.admits(resolve_declared_runtime_lane(environ))


def runtime_profile_owns_contract(
    raw_contract: Mapping[str, object],
    runtime_profile: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return whether runtime_profile owns a raw contract's subscriptions.

    Contracts without ``runtime_profiles`` default to ``main`` ownership. This
    mirrors auto-wiring ownership filtering for legacy runtime-host subscription
    paths that operate on raw contract dictionaries, including the
    ``runtime_lanes`` scope (OMN-19408): a lane-scoped contract is owned only
    on a runtime whose declared lane is in the scope.

    Args:
        environ: Override for the process environment the runtime lane is read
            from. Injected by tests; the default reads ``os.environ``.
    """
    normalized_profile = _normalize_runtime_profile(runtime_profile)
    runtime_profiles = extract_runtime_profiles_from_contract(raw_contract)
    if runtime_profiles:
        profile_owns = normalized_profile in runtime_profiles
    else:
        profile_owns = normalized_profile == "main"
    if not profile_owns:
        return False
    return _runtime_lane_admits_raw_contract(raw_contract, environ)


class ModelRuntimeProfileOwnershipResult(BaseModel):
    """Result of filtering an auto-wiring manifest by runtime profile."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    runtime_profile: str = Field(..., min_length=1)
    manifest: ModelAutoWiringManifest
    skipped_contracts: tuple[str, ...] = Field(default_factory=tuple)
    runtime_lane: str | None = Field(
        default=None,
        description=(
            "The registered lane this runtime declared (ONEX_RUNTIME_LANE), or "
            "None when it declared none or an unregistered one (OMN-19408)."
        ),
    )
    lane_excluded_contracts: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Contracts this runtime's profile owns but whose runtime_lanes "
            "scope excludes the runtime's DECLARED lane -- the correct, "
            "error-free state for a node on a lane it is not meant for. "
            "A contract refused because the runtime declared NO lane is not "
            "listed here; it is a discovery error in manifest.errors."
        ),
    )

    @field_validator("runtime_profile", mode="before")
    @classmethod
    def normalize_runtime_profile(cls, value: object) -> str:
        return _normalize_runtime_profile(value)


def _undeclared_lane_error(
    contract: ModelDiscoveredContract,
    scope: ModelRuntimeLaneScope,
    environ: Mapping[str, str] | None,
) -> ModelDiscoveryError:
    """The fail-closed record for a lane-scoped contract on a lane-less runtime."""
    declared = describe_undeclared_runtime_lane(environ)
    return ModelDiscoveryError(
        entry_point_name=contract.entry_point_name,
        package_name=contract.package_name,
        error=(
            f"contract '{contract.name}' is scoped to runtime lanes "
            f"{list(scope.lanes)} and this runtime cannot name its lane "
            f"({declared}); NOT attached, fail-closed (OMN-19408). Declare "
            f"{ENV_RUNTIME_LANE} in this runtime's deployment as one of "
            f"{sorted(REGISTERED_RUNTIME_LANES)}."
        ),
    )


def filter_manifest_for_runtime_profile(
    manifest: ModelAutoWiringManifest,
    runtime_profile: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> ModelRuntimeProfileOwnershipResult:
    """Return a manifest containing only contracts owned by runtime_profile.

    Contracts without ``runtime_profiles`` default to ``main`` ownership.
    Contracts with an explicit list are wired only by profiles named in that
    list.

    Of the contracts the profile owns, one that declares ``runtime_lanes``
    (OMN-19408) is kept only when this runtime's declared lane is in its scope.
    A declared lane outside the scope drops it without error (listed in
    ``lane_excluded_contracts``). No registered lane at all drops it AND adds a
    discovery error naming it: fail-closed, never a silent skip.

    Args:
        environ: Override for the process environment the runtime lane is read
            from. Injected by tests; the default reads ``os.environ``.
    """
    normalized_profile = _normalize_runtime_profile(runtime_profile)
    runtime_lane = resolve_declared_runtime_lane(environ)
    owned_contracts: list[ModelDiscoveredContract] = []
    skipped_contracts: list[str] = []
    lane_excluded_contracts: list[str] = []
    lane_errors: list[ModelDiscoveryError] = []

    for contract in manifest.contracts:
        if contract.runtime_profiles:
            profile_owns = normalized_profile in contract.runtime_profiles
        else:
            profile_owns = normalized_profile == "main"
        if not profile_owns:
            skipped_contracts.append(contract.name)
            continue

        scope = contract.runtime_lanes
        if scope is not None and not scope.admits(runtime_lane):
            if runtime_lane is None:
                error = _undeclared_lane_error(contract, scope, environ)
                logger.error("Auto-wiring lane scope: %s", error.error)
                lane_errors.append(error)
            else:
                lane_excluded_contracts.append(contract.name)
            continue
        owned_contracts.append(contract)

    if lane_excluded_contracts:
        logger.info(
            "Auto-wiring runtime lane scope: lane=%s excluded=%s (OMN-19408)",
            runtime_lane,
            lane_excluded_contracts,
        )

    return ModelRuntimeProfileOwnershipResult(
        runtime_profile=normalized_profile,
        manifest=ModelAutoWiringManifest(
            contracts=tuple(owned_contracts),
            errors=(*manifest.errors, *lane_errors),
        ),
        skipped_contracts=tuple(skipped_contracts),
        runtime_lane=runtime_lane,
        lane_excluded_contracts=tuple(lane_excluded_contracts),
    )


__all__ = [
    "ModelRuntimeProfileOwnershipResult",
    "extract_runtime_profiles_from_contract",
    "filter_manifest_for_runtime_profile",
    "runtime_profile_owns_contract",
]
