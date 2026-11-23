# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Boot only the lane's declared boot set; leave the rest dormant (OMN-21592).

Plan task A4 (section 3.11.5): on a lane whose hot-load overlay row is
``enabled`` with ``mode: on_demand`` and carries a ``lazy_start`` block, the
kernel wires only the front-door contract and the declared boot-priority set.
Every other contract the runtime profile owns is dormant: it is left out of the
manifest the kernel wires and subscribes, so its handler class is never
imported, no dispatcher is registered and no topic is provisioned for it, and
the boot's universe topic warm (which creates every declared topic) is off.
The dormant contracts are kept here for the front door, which starts each one
on its first request.

A lane with no ``lazy_start`` block boots exactly as before.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING

from omnibase_infra.models.lazy_start import (
    ModelLazyStartPolicy,
    ModelLazyStartServingStatus,
)
from omnibase_infra.models.model_contract_hotload_lane_overlay import (
    ModelContractHotloadLaneOverlay,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelDiscoveredContract,
)

if TYPE_CHECKING:
    from omnibase_infra.handlers.lazy_start_front_door import LazyStartFrontDoor

__all__: list[str] = [
    "StoreLazyStartDormant",
    "boot_manifest",
    "boot_universe_warm_enabled",
    "lazy_start_policy_of_this_lane",
    "split_boot_manifest",
]

logger = logging.getLogger(__name__)


class StoreLazyStartDormant:
    """The dormant contracts and policy of this process's boot, for the front door."""

    _policy: ModelLazyStartPolicy | None = None
    _contracts: dict[str, ModelDiscoveredContract] = {}
    _front_door: LazyStartFrontDoor | None = None
    _wired: dict[str, ModelDiscoveredContract] = {}
    _provisioner: object | None = None

    @classmethod
    def record(
        cls,
        policy: ModelLazyStartPolicy,
        dormant: tuple[ModelDiscoveredContract, ...],
        provisioner: object | None = None,
        wired: tuple[ModelDiscoveredContract, ...] = (),
    ) -> None:
        cls._policy = policy
        cls._contracts = {contract.name: contract for contract in dormant}
        cls._wired = {contract.name: contract for contract in wired}
        cls._provisioner = provisioner

    @classmethod
    def provisioner(cls) -> object | None:
        """The boot's topic provisioner, reused by every start."""
        return cls._provisioner

    @classmethod
    def policy(cls) -> ModelLazyStartPolicy | None:
        return cls._policy

    @classmethod
    def contracts(cls) -> Mapping[str, ModelDiscoveredContract]:
        return dict(cls._contracts)

    @classmethod
    def wired_contracts(cls) -> Mapping[str, ModelDiscoveredContract]:
        """The contracts wired at boot: closure edges run through them (OMN-26396)."""
        return dict(cls._wired)

    @classmethod
    def record_front_door(cls, front_door: LazyStartFrontDoor) -> None:
        """Keep the boot's front door so ``/ready`` can read its serving set (OMN-26316)."""
        cls._front_door = front_door

    @classmethod
    def serving_status(cls) -> ModelLazyStartServingStatus | None:
        """The front door's serving set and wake failures, or None before it exists."""
        if cls._policy is None or cls._front_door is None:
            return None
        return cls._front_door.serving_status()

    @classmethod
    def serving_detail(cls) -> dict[str, object]:
        """The ``lazy_start`` block of the ``/ready`` detail, or nothing (OMN-26316)."""
        status = cls.serving_status()
        if status is None:
            return {}
        return {"lazy_start": status.model_dump(mode="json")}

    @classmethod
    def clear(cls) -> None:
        cls._policy = None
        cls._front_door = None
        cls._contracts = {}
        cls._wired = {}
        cls._provisioner = None


def lazy_start_policy_of_this_lane(
    environ: Mapping[str, str] | None = None,
) -> ModelLazyStartPolicy | None:
    """The lane row's lazy-start block when the row is on_demand, else None."""
    row = ModelContractHotloadLaneOverlay.row_of_this_lane(environ)
    return None if row is None else row.lazy_start_policy


def split_boot_manifest(
    manifest: ModelAutoWiringManifest,
    policy: ModelLazyStartPolicy,
) -> tuple[ModelAutoWiringManifest, tuple[ModelDiscoveredContract, ...]]:
    """The manifest of the boot set, and the dormant contracts, in discovery order."""
    boot_set = policy.boot_set
    boot = tuple(c for c in manifest.contracts if c.name in boot_set)
    dormant = tuple(c for c in manifest.contracts if c.name not in boot_set)
    return manifest.model_copy(update={"contracts": boot}), dormant


def boot_manifest(
    manifest: ModelAutoWiringManifest,
    provisioner: object | None = None,
    environ: Mapping[str, str] | None = None,
) -> ModelAutoWiringManifest:
    """The kernel's boot manifest: the boot set on a lazy-start lane, else unchanged."""
    policy = lazy_start_policy_of_this_lane(environ)
    if policy is None:
        return manifest
    boot, dormant = split_boot_manifest(manifest, policy)
    StoreLazyStartDormant.record(policy, dormant, provisioner, boot.contracts)
    missing = sorted(policy.boot_set - {c.name for c in boot.contracts})
    logger.info(
        "Lazy start: %d contract(s) wired at boot, %d dormant until their first "
        "request; boot set names not owned here=%s (OMN-21592)",
        boot.total_discovered,
        len(dormant),
        missing,
    )
    return boot


def boot_universe_warm_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Whether boot creates every declared topic: off on a lazy-start lane."""
    source: Mapping[str, str] = os.environ if environ is None else environ
    if source.get("ONEX_BOOT_UNIVERSE_PROVISION", "1") == "0":
        return False
    return lazy_start_policy_of_this_lane(source) is None
