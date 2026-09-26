# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime-profile and runtime-lane contract ownership filtering for auto-wiring.

Contracts may declare ``runtime_profiles`` to assign their Kafka subscriptions
to a specific runtime process. Unscoped legacy contracts default to ``main`` so
the effects/worker runtimes do not join general compute groups and steal work
from the primary runtime.

Contracts may also scope themselves to runtime lanes (OMN-19408, OMN-19747).
A profile is a role INSIDE a lane; a lane is WHICH deployment. The lane and
what it is for come from the deployment's ``runtime.lane`` overlay document,
which the kernel resolves before discovery and refuses to start without, so a
runtime that reaches this filter always has a lane.

- ``runtime_lane_roles`` (the replacement): the contract attaches only on a
  lane whose overlay grants every role listed.
- ``runtime_lanes`` (transitional, removed by LO9): the contract attaches only
  on a lane whose id is listed.

A contract excluded by either is listed in ``lane_excluded_contracts``: that is
the correct, error-free state for a node on a lane it is not meant for. There
is no discovery-error branch for a lane-less runtime, because no such runtime
gets this far. A caller outside the kernel that has no established lane and
meets a lane-scoped contract is refused, never answered with a guess.

Both the kernel's wiring and the health monitor's expectations read this one
filter, so what is attached and what health expects cannot disagree.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from omnibase_core.models.config_overlay import ModelRuntimeLaneDeclaration
from omnibase_core.models.contracts.subcontracts.model_runtime_lane_role_requirement import (
    ModelRuntimeLaneRoleRequirement,
)
from omnibase_core.models.contracts.subcontracts.model_runtime_lane_scope import (
    ModelRuntimeLaneScope,
)
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.health.runtime_lane_identity import (
    ENV_RUNTIME_LANE,
    established_runtime_lane,
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


def _lane_for(
    lane: ModelRuntimeLaneDeclaration | None, contract_name: str
) -> ModelRuntimeLaneDeclaration:
    """The lane to judge a lane-scoped contract against, or a refusal."""
    if lane is not None:
        return lane
    resolution = established_runtime_lane()
    if resolution is not None:
        return resolution.declaration
    raise ProtocolConfigurationError(
        f"contract {contract_name!r} is scoped to runtime lanes and no runtime "
        "lane has been resolved in this process. The kernel resolves it from "
        f"the deployment's runtime.lane overlay ({ENV_RUNTIME_LANE}) before "
        "discovery; a caller outside the kernel passes lane= explicitly."
    )


def lane_admits(
    lane: ModelRuntimeLaneDeclaration,
    *,
    roles: ModelRuntimeLaneRoleRequirement | None,
    lanes: ModelRuntimeLaneScope | None,
) -> bool:
    """Whether a lane admits a contract's role requirement and lane list.

    Both must admit when both are declared. Neither declared is unscoped.
    """
    if roles is not None and not roles.admits(lane):
        return False
    return lanes is None or lanes.admits(lane.lane_id)


def _runtime_lane_admits_raw_contract(
    raw_contract: Mapping[str, object],
    lane: ModelRuntimeLaneDeclaration | None,
) -> bool:
    """Return whether this runtime's lane admits a raw contract's lane scope.

    Fail-closed: an unparseable scope never admits. This raw path serves the
    legacy runtime-host subscription loops and only has to refuse.
    """
    roles_raw = raw_contract.get("runtime_lane_roles")
    lanes_raw = raw_contract.get("runtime_lanes")
    if roles_raw is None and lanes_raw is None:
        return True
    try:
        roles = (
            None
            if roles_raw is None
            else ModelRuntimeLaneRoleRequirement.model_validate({"roles": roles_raw})
        )
        lanes = (
            None
            if lanes_raw is None
            else ModelRuntimeLaneScope.model_validate({"lanes": lanes_raw})
        )
    except ValidationError:
        return False
    name = str(raw_contract.get("name", "<unnamed>"))
    return lane_admits(_lane_for(lane, name), roles=roles, lanes=lanes)


def runtime_profile_owns_contract(
    raw_contract: Mapping[str, object],
    runtime_profile: str,
    *,
    lane: ModelRuntimeLaneDeclaration | None = None,
) -> bool:
    """Return whether runtime_profile owns a raw contract's subscriptions.

    Contracts without ``runtime_profiles`` default to ``main`` ownership. This
    mirrors auto-wiring ownership filtering for legacy runtime-host subscription
    paths that operate on raw contract dictionaries, including the
    lane scope (OMN-19408, OMN-19747): a lane-scoped contract is owned only
    on a runtime whose lane admits it.

    Args:
        lane: The lane to judge against. Defaults to the lane the kernel
            established at startup.
    """
    normalized_profile = _normalize_runtime_profile(runtime_profile)
    runtime_profiles = extract_runtime_profiles_from_contract(raw_contract)
    if runtime_profiles:
        profile_owns = normalized_profile in runtime_profiles
    else:
        profile_owns = normalized_profile == "main"
    if not profile_owns:
        return False
    return _runtime_lane_admits_raw_contract(raw_contract, lane)


class ModelRuntimeProfileOwnershipResult(BaseModel):
    """Result of filtering an auto-wiring manifest by runtime profile."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    runtime_profile: str = Field(..., min_length=1)
    manifest: ModelAutoWiringManifest
    skipped_contracts: tuple[str, ...] = Field(default_factory=tuple)
    runtime_lane: str | None = Field(
        default=None,
        description=(
            "The lane id the contracts were judged against (OMN-19747), or None "
            "when the manifest held no lane-scoped contract to judge."
        ),
    )
    lane_excluded_contracts: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Contracts this runtime's profile owns but whose runtime_lane_roles "
            "or runtime_lanes scope excludes the runtime's lane -- the correct, "
            "error-free state for a node on a lane it is not meant for."
        ),
    )

    @field_validator("runtime_profile", mode="before")
    @classmethod
    def normalize_runtime_profile(cls, value: object) -> str:
        return _normalize_runtime_profile(value)


def filter_manifest_for_runtime_profile(
    manifest: ModelAutoWiringManifest,
    runtime_profile: str,
    *,
    lane: ModelRuntimeLaneDeclaration | None = None,
) -> ModelRuntimeProfileOwnershipResult:
    """Return a manifest containing only contracts owned by runtime_profile.

    Contracts without ``runtime_profiles`` default to ``main`` ownership.
    Contracts with an explicit list are wired only by profiles named in that
    list.

    Of the contracts the profile owns, one that declares
    ``runtime_lane_roles`` or ``runtime_lanes`` is kept only when this
    runtime's lane admits it, and is otherwise listed in
    ``lane_excluded_contracts`` without error.

    Args:
        lane: The lane to judge against. Defaults to the lane the kernel
            established at startup; with neither, a lane-scoped contract is
            refused with :class:`ProtocolConfigurationError`.
    """
    normalized_profile = _normalize_runtime_profile(runtime_profile)
    judged_lane: ModelRuntimeLaneDeclaration | None = None
    owned_contracts: list[ModelDiscoveredContract] = []
    skipped_contracts: list[str] = []
    lane_excluded_contracts: list[str] = []

    for contract in manifest.contracts:
        if contract.runtime_profiles:
            profile_owns = normalized_profile in contract.runtime_profiles
        else:
            profile_owns = normalized_profile == "main"
        if not profile_owns:
            skipped_contracts.append(contract.name)
            continue

        roles = contract.runtime_lane_roles
        lanes = contract.runtime_lanes
        if roles is not None or lanes is not None:
            judged_lane = _lane_for(lane, contract.name)
            if not lane_admits(judged_lane, roles=roles, lanes=lanes):
                lane_excluded_contracts.append(contract.name)
                continue
        owned_contracts.append(contract)

    if lane_excluded_contracts:
        logger.info(
            "Auto-wiring runtime lane scope: lane=%s excluded=%s (OMN-19747)",
            judged_lane.lane_id if judged_lane is not None else None,
            lane_excluded_contracts,
        )

    return ModelRuntimeProfileOwnershipResult(
        runtime_profile=normalized_profile,
        manifest=ModelAutoWiringManifest(
            contracts=tuple(owned_contracts),
            errors=manifest.errors,
        ),
        skipped_contracts=tuple(skipped_contracts),
        runtime_lane=judged_lane.lane_id if judged_lane is not None else None,
        lane_excluded_contracts=tuple(lane_excluded_contracts),
    )


__all__ = [
    "ModelRuntimeProfileOwnershipResult",
    "lane_admits",
    "extract_runtime_profiles_from_contract",
    "filter_manifest_for_runtime_profile",
    "runtime_profile_owns_contract",
]
