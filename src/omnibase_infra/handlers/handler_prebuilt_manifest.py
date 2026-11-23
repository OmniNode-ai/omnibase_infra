# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Serve the node catalog and auto-wiring manifest the image build wrote (OMN-26361).

For a given image the installed packages are fixed, so boot step 0 (announce every
``onex.nodes`` contract) and ``discover_contracts`` (parse each contract again) give
the same answer on every boot. The image build runs both once and writes the catalog
and the manifest as one JSON file stamped with the installed distributions, their
versions and every contract's content hash (the node runtime beside this node's
contract does the build). This handler is what the kernel runs at boot: it checks the
stamp against the installed metadata without importing a node or parsing a contract,
and serves the file only when nothing differs. Any difference is returned by name and
the caller falls back to the announce-and-discover path.

A package that is not baked into the image is not in the file. It announces at
runtime and :func:`merge_runtime_announcements` folds those announcements into the
loaded catalog with the catalog's own ``apply``, the fold the boot step uses.
"""

from __future__ import annotations

import asyncio
import importlib.resources
import logging
from collections.abc import Awaitable, Callable, Iterable
from importlib.metadata import entry_points
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from omnibase_core.models.events.contract_registration.model_node_announcement import (
    ModelNodeAnnouncement,
)
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.node_announce.model_boot_clock import ModelBootClock
from omnibase_infra.models.node_announce.model_boot_node_catalog import (
    ModelBootNodeCatalog,
)
from omnibase_infra.models.node_announce.model_package_announce_result import (
    ModelPackageAnnounceResult,
)
from omnibase_infra.models.node_announce.model_prebuilt_manifest_request import (
    ModelPrebuiltManifestRequest,
)
from omnibase_infra.models.node_announce.model_prebuilt_manifest_result import (
    ModelPrebuiltManifestResult,
)
from omnibase_infra.models.node_announce.model_prebuilt_manifest_stamp import (
    ModelPrebuiltManifestStamp,
)
from omnibase_infra.models.node_announce.model_prebuilt_node_manifest import (
    ModelPrebuiltNodeManifest,
)
from omnibase_infra.models.runtime.model_node_exclusion_lane_env import (
    NODE_ENTRY_POINT_GROUP,
)
from omnibase_infra.utils.util_runtime_packages import (
    get_active_runtime_packages,
    is_gateway_cloud_mirroring_enabled,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION: Final = 1

InstalledFileReader = Callable[
    [ModelNodeAnnouncement], tuple[bytes | None, Path | None]
]


def prebuilt_manifest_path() -> Path:
    """Where the image build writes the file: beside the boot catalog node's contract."""
    return Path(
        str(
            importlib.resources.files("omnibase_infra").joinpath(
                "nodes", "node_boot_catalog_effect", "prebuilt_node_manifest.json"
            )
        )
    )


def installed_fingerprint() -> tuple[tuple[str, str, str], ...]:
    """The ``onex.nodes`` listing from package metadata alone: no entry point is loaded."""
    out: list[tuple[str, str, str]] = []
    for ep in entry_points(group=NODE_ENTRY_POINT_GROUP):
        dist = ep.dist
        out.append(
            (
                ep.name,
                dist.name if dist is not None else "unknown",
                str(dist.version) if dist is not None else "0.0.0",
            )
        )
    return tuple(sorted(out))


def _file_hash(path: str) -> str:
    """The canonical content hash of the file now on disk; empty when it cannot be read."""
    try:
        return ModelNodeAnnouncement.canonical_content_hash(Path(path).read_bytes())
    except OSError:
        return ""


def stamp_of(
    fingerprint: tuple[tuple[str, str, str], ...],
    contract_hashes: tuple[tuple[str, str], ...],
    workload: str,
) -> ModelPrebuiltManifestStamp:
    """A stamp over the given listing and hashes under the environment switches now set."""
    active = get_active_runtime_packages()
    return ModelPrebuiltManifestStamp(
        fingerprint=fingerprint,
        contract_hashes=contract_hashes,
        active_packages=",".join(sorted(active)) if active else "",
        gateway_cloud_mirroring=is_gateway_cloud_mirroring_enabled(),
        workload=workload,
    )


class HandlerPrebuiltManifest:
    """Read the prebuilt node manifest and say whether its stamp still holds."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def handle(
        self, request: ModelPrebuiltManifestRequest
    ) -> ModelPrebuiltManifestResult:
        if not request.path.is_file():
            return ModelPrebuiltManifestResult(reason="absent")
        try:
            prebuilt = ModelPrebuiltNodeManifest.model_validate_json(
                request.path.read_bytes()
            )
        except (OSError, ValidationError, ValueError) as exc:
            return ModelPrebuiltManifestResult(
                reason=f"unreadable:{type(exc).__name__}"
            )
        if prebuilt.schema_version != SCHEMA_VERSION:
            return ModelPrebuiltManifestResult(
                reason="stale",
                differing_fields=(
                    f"schema_version:{prebuilt.schema_version}->{SCHEMA_VERSION}",
                ),
            )
        current = stamp_of(
            installed_fingerprint(),
            tuple((p, _file_hash(p)) for p, _ in prebuilt.stamp.contract_hashes),
            request.workload,
        )
        differing = prebuilt.stamp.differing_fields(current)
        if differing:
            return ModelPrebuiltManifestResult(
                reason="stale", differing_fields=differing
            )
        return ModelPrebuiltManifestResult(prebuilt=prebuilt)


def merge_runtime_announcements(
    loaded: ModelPackageAnnounceResult,
    announcements: Iterable[ModelNodeAnnouncement],
    read_installed: InstalledFileReader,
) -> ModelPackageAnnounceResult:
    """Fold announcements made after boot into a catalog loaded from the prebuilt file.

    A hot-loaded package is not in the image's file, so it announces at runtime. Each
    announcement goes through the catalog's own ``apply``, the fold the boot step
    runs; the entries the file carried stay, and an announcement for a name the file
    holds is admitted or refused by the catalog's own rules.
    """
    catalog = loaded.catalog
    added: list[ModelNodeAnnouncement] = []
    outcomes = list(loaded.outcomes)
    for announcement in announcements:
        installed_bytes, installed_path = read_installed(announcement)
        catalog, outcome, _changes = catalog.apply(
            announcement, installed_bytes=installed_bytes, installed_path=installed_path
        )
        added.append(announcement)
        outcomes.append(outcome)
    return ModelPackageAnnounceResult(
        inventory=loaded.inventory,
        announcements=(*loaded.announcements, *added),
        catalog=catalog,
        outcomes=tuple(outcomes),
    )


async def run_boot_catalog(
    *,
    workload: str,
    announce: Callable[..., Awaitable[ModelPackageAnnounceResult]],
    path: Path | None = None,
) -> ModelBootNodeCatalog:
    """Boot step 0: serve the image's prebuilt file when its stamp holds, else announce.

    ``announce`` is boot step 0's announcer, called as ``announce(workload=...)``. A
    stale or unreadable file is logged with every stamp field that differed and the
    boot takes the announce path it took before the file existed. The boot summary
    line (OMN-26561) reads the longest event-loop gap of this phase, measured from here.
    """
    with ModelBootClock.phase("boot_step0"):
        return await _run_boot_catalog(workload=workload, announce=announce, path=path)


async def _run_boot_catalog(
    *,
    workload: str,
    announce: Callable[..., Awaitable[ModelPackageAnnounceResult]],
    path: Path | None,
) -> ModelBootNodeCatalog:
    handler = HandlerPrebuiltManifest()
    result = await asyncio.to_thread(
        handler.handle,
        ModelPrebuiltManifestRequest(
            path=path or prebuilt_manifest_path(), workload=workload
        ),
    )
    if result.prebuilt is not None:
        logger.info(
            "Boot step 0: loaded prebuilt node manifest (%d contracts, 0 parsed)",
            result.prebuilt.manifest.total_discovered,
        )
        return ModelBootNodeCatalog(
            announced=result.prebuilt.announced, manifest=result.prebuilt.manifest
        )
    if result.reason != "absent":
        logger.warning(
            "Prebuilt node manifest not used (%s); differing stamp fields: %s",
            result.reason,
            ", ".join(result.differing_fields[:20]) or "none",
        )
    return ModelBootNodeCatalog(announced=await announce(workload=workload))


__all__ = [
    "SCHEMA_VERSION",
    "HandlerPrebuiltManifest",
    "InstalledFileReader",
    "installed_fingerprint",
    "merge_runtime_announcements",
    "prebuilt_manifest_path",
    "run_boot_catalog",
    "stamp_of",
]
