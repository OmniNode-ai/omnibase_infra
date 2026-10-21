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

A lane may be an INSTANCE of a registered lane class (OMN-20867): a second
copy of a lab lane on another host declares its own registered lane, named
``<class>-<instance>``. Shipped contracts may not name a host's lane, so the
deployment's private overlay says it, in two variables read together:
``ONEX_RUNTIME_LANE_CLASS`` (the class) and ``ONEX_RUNTIME_LANE_CLASS_CONTRACTS``
(the class contracts this instance takes). Only the named contracts are
borrowed, never the whole class: a lane-keyed contract such as the lab
lane-health projection folds rows for the class's own lane and would consume
and drop everything on an instance. A selection that cannot be honoured is a
discovery error, never a silent skip.

A deployment may also select the nodes it loads at all (OMN-17427):
``ONEX_RUNTIME_NODE_SELECTION`` names contracts, and every other discovered
contract is left unattached before profile or lane ownership is decided. Unset,
nothing is excluded. A named contract that no discovered contract is becomes a
discovery error, never a silent zero.

A lane may fold other roles into one runtime (OMN-17427):
``ONEX_RUNTIME_FOLDED_PROFILES`` names the profiles whose contracts this
process attaches in addition to its own role's, so one runtime container
carries the effects, worker and projection-writer nodes instead of a container
per role. The role identity (``RUNTIME_PROFILE``) is unchanged; lane scope and
node selection apply to a folded contract exactly as to an owned one.

Both the kernel's wiring and the health monitor's expectations read this one
filter, so what is attached and what health expects cannot disagree.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from omnibase_core.constants.constants_runtime_lanes import REGISTERED_RUNTIME_LANES
from omnibase_core.models.contracts.subcontracts.model_runtime_lane_scope import (
    ModelRuntimeLaneScope,
)
from omnibase_infra.models.runtime.model_runtime_lane_class_selection import (
    ModelRuntimeLaneClassSelection,
)
from omnibase_infra.models.runtime.model_runtime_node_selection import (
    ENV_RUNTIME_NODE_SELECTION,
    ModelRuntimeNodeSelection,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelDiscoveredContract,
    ModelDiscoveryError,
)
from omnibase_infra.runtime.health.runtime_lane_identity import (
    ENV_RUNTIME_LANE,
    ENV_RUNTIME_LANE_CLASS,
    ENV_RUNTIME_LANE_CLASS_CONTRACTS,
    describe_undeclared_runtime_lane,
    resolve_declared_runtime_lane,
)
from omnibase_infra.runtime.runtime_profile import resolve_runtime_folded_profiles

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


def resolve_runtime_lane_class_selection(
    runtime_lane: str | None,
    environ: Mapping[str, str] | None = None,
) -> ModelRuntimeLaneClassSelection:
    """Read the deployment's lane-class selection for a runtime's declared lane.

    Both variables unset is no selection. Otherwise the class must be a
    registered lane, the declared lane must be ``<class>-<instance>``, and at
    least one contract must be named; anything else is returned as a refusal
    with no class, so nothing is borrowed.
    """
    source: Mapping[str, str] = os.environ if environ is None else environ
    raw_class = (source.get(ENV_RUNTIME_LANE_CLASS) or "").strip().lower()
    raw_contracts = (source.get(ENV_RUNTIME_LANE_CLASS_CONTRACTS) or "").strip()
    if not raw_class and not raw_contracts:
        return ModelRuntimeLaneClassSelection()
    contracts = tuple(
        dict.fromkeys(part.strip() for part in raw_contracts.split(",") if part.strip())
    )
    refusal = ""
    if not raw_class:
        refusal = f"{ENV_RUNTIME_LANE_CLASS_CONTRACTS} is set but {ENV_RUNTIME_LANE_CLASS} is not"
    elif not contracts:
        refusal = (
            f"{ENV_RUNTIME_LANE_CLASS}={raw_class!r} names no contracts; set "
            f"{ENV_RUNTIME_LANE_CLASS_CONTRACTS}"
        )
    elif raw_class not in REGISTERED_RUNTIME_LANES:
        refusal = (
            f"{ENV_RUNTIME_LANE_CLASS}={raw_class!r} is not a registered lane "
            f"{sorted(REGISTERED_RUNTIME_LANES)}"
        )
    elif runtime_lane is None:
        refusal = (
            f"{ENV_RUNTIME_LANE_CLASS}={raw_class!r} is set but this runtime "
            f"declares no registered {ENV_RUNTIME_LANE}"
        )
    elif not runtime_lane.startswith(f"{raw_class}-"):
        refusal = (
            f"{ENV_RUNTIME_LANE}={runtime_lane!r} is not an instance of lane "
            f"class {raw_class!r}; an instance is named '{raw_class}-<instance>'"
        )
    return ModelRuntimeLaneClassSelection(
        lane_class=None if refusal else raw_class,
        contracts=contracts,
        refusal=refusal,
    )


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
    runtime_lane = resolve_declared_runtime_lane(environ)
    if scope.admits(runtime_lane):
        return True
    name = raw_contract.get("name")
    return isinstance(name, str) and resolve_runtime_lane_class_selection(
        runtime_lane, environ
    ).admits(name, scope)


def _profile_owns(
    contract_profiles: Sequence[str],
    owned_profiles: tuple[str, ...],
) -> bool:
    """Whether any profile this process carries owns a contract's subscriptions.

    A contract with no ``runtime_profiles`` belongs to ``main``.
    """
    if contract_profiles:
        return any(profile in contract_profiles for profile in owned_profiles)
    return "main" in owned_profiles


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
    on a runtime whose declared lane is in the scope. A contract the
    deployment's node selection (OMN-17427) does not name is never owned. A
    contract a folded profile owns (``ONEX_RUNTIME_FOLDED_PROFILES``,
    OMN-17427) is owned as well.

    Args:
        environ: Override for the process environment the runtime lane is read
            from. Injected by tests; the default reads ``os.environ``.
    """
    normalized_profile = _normalize_runtime_profile(runtime_profile)
    name = raw_contract.get("name")
    if not ModelRuntimeNodeSelection.from_env(environ).admits(
        name if isinstance(name, str) else ""
    ):
        return False
    owned_profiles = (normalized_profile, *resolve_runtime_folded_profiles(environ))
    runtime_profiles = extract_runtime_profiles_from_contract(raw_contract)
    if not _profile_owns(runtime_profiles, owned_profiles):
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
    runtime_lane_class: str | None = Field(
        default=None,
        description=(
            "The lane class this runtime is an instance of "
            "(ONEX_RUNTIME_LANE_CLASS), or None when none is selected or the "
            "selection was refused (OMN-20867)."
        ),
    )
    lane_class_admitted_contracts: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Contracts attached because the deployment selected them from the "
            "runtime's lane class, though its own lane is outside their scope."
        ),
    )
    node_selection_excluded_contracts: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Contracts left unattached because the deployment's node selection "
            "(ONEX_RUNTIME_NODE_SELECTION) does not name them (OMN-17427). "
            "Empty when no selection is set."
        ),
    )

    folded_profiles: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Roles this runtime carries besides its own, from "
            "ONEX_RUNTIME_FOLDED_PROFILES (OMN-17427). Empty when none is folded."
        ),
    )
    folded_contracts: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Attached contracts that this runtime's own role does not own and a "
            "folded profile does (OMN-17427)."
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


def _lane_class_error(entry_point_name: str, error: str) -> ModelDiscoveryError:
    """The fail-closed record for a lane-class selection that cannot be honoured."""
    record = ModelDiscoveryError(
        entry_point_name=entry_point_name,
        error=f"{error}; not borrowed from a lane class, fail-closed (OMN-20867).",
    )
    logger.error("Auto-wiring lane class: %s", record.error)
    return record


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

    A contract the declared lane excludes is still kept when the deployment
    selected it from the runtime's lane class (OMN-20867), listed in
    ``lane_class_admitted_contracts``. A refused selection, a selected name no
    discovered contract has, and a selected contract whose scope excludes the
    class each add a discovery error.

    A contract a folded profile (``ONEX_RUNTIME_FOLDED_PROFILES``, OMN-17427)
    owns is kept on the same lane terms and listed in ``folded_contracts``.

    Before any of that, a deployment node selection (OMN-17427) drops every
    contract it does not name, listed in ``node_selection_excluded_contracts``;
    a name it carries that no discovered contract has adds a discovery error.

    Args:
        environ: Override for the process environment the runtime lane is read
            from. Injected by tests; the default reads ``os.environ``.
    """
    normalized_profile = _normalize_runtime_profile(runtime_profile)
    folded_profiles = tuple(
        profile
        for profile in resolve_runtime_folded_profiles(environ)
        if profile != normalized_profile
    )
    runtime_lane = resolve_declared_runtime_lane(environ)
    owned_contracts: list[ModelDiscoveredContract] = []
    folded_contracts: list[str] = []
    skipped_contracts: list[str] = []
    lane_excluded_contracts: list[str] = []
    lane_errors: list[ModelDiscoveryError] = []
    lane_class_admitted: list[str] = []
    node_selection = ModelRuntimeNodeSelection.from_env(environ)
    node_selection_excluded: list[str] = []
    discovered_names = {contract.name for contract in manifest.contracts}
    for name in node_selection.contracts:
        if name not in discovered_names:
            record = ModelDiscoveryError(
                entry_point_name=name,
                error=(
                    f"{ENV_RUNTIME_NODE_SELECTION} names {name!r}, which no "
                    "discovered contract is; nothing attached for it, "
                    "fail-closed (OMN-17427)."
                ),
            )
            logger.error("Auto-wiring node selection: %s", record.error)
            lane_errors.append(record)
    selection = resolve_runtime_lane_class_selection(runtime_lane, environ)
    if selection.refusal:
        lane_errors.append(_lane_class_error(ENV_RUNTIME_LANE_CLASS, selection.refusal))
    elif selection.lane_class is not None:
        discovered = {contract.name for contract in manifest.contracts}
        for name in selection.contracts:
            if name not in discovered:
                lane_errors.append(
                    _lane_class_error(
                        name,
                        f"{ENV_RUNTIME_LANE_CLASS_CONTRACTS} names {name!r}, "
                        "which no discovered contract is",
                    )
                )

    for contract in manifest.contracts:
        if not node_selection.admits(contract.name):
            node_selection_excluded.append(contract.name)
            continue
        if not _profile_owns(contract.runtime_profiles, (normalized_profile,)):
            if not _profile_owns(contract.runtime_profiles, folded_profiles):
                skipped_contracts.append(contract.name)
                continue
            folded_contracts.append(contract.name)

        scope = contract.runtime_lanes
        if scope is not None and not scope.admits(runtime_lane):
            if selection.admits(contract.name, scope):
                lane_class_admitted.append(contract.name)
                owned_contracts.append(contract)
                continue
            if (
                selection.lane_class is not None
                and contract.name in selection.contracts
            ):
                lane_errors.append(
                    _lane_class_error(
                        contract.entry_point_name,
                        f"contract '{contract.name}' is selected from lane class "
                        f"{selection.lane_class!r} but its runtime_lanes "
                        f"{list(scope.lanes)} do not admit that class",
                    )
                )
                continue
            if runtime_lane is None:
                error = _undeclared_lane_error(contract, scope, environ)
                logger.error("Auto-wiring lane scope: %s", error.error)
                lane_errors.append(error)
            else:
                lane_excluded_contracts.append(contract.name)
            continue
        owned_contracts.append(contract)

    if folded_contracts:
        logger.info(
            "Auto-wiring folded profiles: profile=%s folded=%s carries %d "
            "contract(s) of the folded roles (OMN-17427)",
            normalized_profile,
            list(folded_profiles),
            len(folded_contracts),
        )
    if node_selection_excluded:
        logger.info(
            "Auto-wiring node selection: attached only %s; excluded %d other "
            "contract(s) (OMN-17427)",
            list(node_selection.contracts),
            len(node_selection_excluded),
        )
    if lane_class_admitted:
        logger.info(
            "Auto-wiring runtime lane scope: lane=%s class=%s attached_by_class=%s "
            "(OMN-20867)",
            runtime_lane,
            selection.lane_class,
            lane_class_admitted,
        )
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
            skips=manifest.skips,
        ),
        skipped_contracts=tuple(skipped_contracts),
        runtime_lane=runtime_lane,
        lane_excluded_contracts=tuple(lane_excluded_contracts),
        runtime_lane_class=selection.lane_class,
        lane_class_admitted_contracts=tuple(lane_class_admitted),
        node_selection_excluded_contracts=tuple(node_selection_excluded),
        folded_profiles=folded_profiles,
        folded_contracts=tuple(
            name
            for name in folded_contracts
            if name in {contract.name for contract in owned_contracts}
        ),
    )


__all__ = [
    "ModelRuntimeProfileOwnershipResult",
    "extract_runtime_profiles_from_contract",
    "filter_manifest_for_runtime_profile",
    "resolve_runtime_lane_class_selection",
    "runtime_profile_owns_contract",
]
