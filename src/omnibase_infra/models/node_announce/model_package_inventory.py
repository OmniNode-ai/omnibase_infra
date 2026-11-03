# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Everything the metadata reader found in the installed package set (OMN-21590)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.node_announce.model_package_contract_record import (
    ModelPackageContractRecord,
)
from omnibase_infra.models.node_announce.model_package_plugin_record import (
    ModelPackagePluginRecord,
)


class ModelPackageInventory(BaseModel):
    """Records for the active packages, in the order the metadata listed them.

    ``plugins`` holds the kernel plugin entries of the same packages (OMN-21591); they
    are announced as ``kernel_plugin`` entries and are never contracts to wire.

    ``fingerprint`` covers every entry point of the group, active or not, as
    ``(name, distribution, version)``: it is what changes when a package is
    installed or removed, and the discovery memo compares it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    records: tuple[ModelPackageContractRecord, ...] = ()
    plugins: tuple[ModelPackagePluginRecord, ...] = Field(
        default=(),
        description="Kernel plugins of the active packages, in the order the metadata listed them",
    )
    fingerprint: tuple[tuple[str, str, str], ...] = Field(default=())

    @property
    def resolution_complete(self) -> bool:
        """False when any entry failed to load or resolve, so a memo must not be trusted."""
        return all(r.error_stage not in ("load", "resolve") for r in self.records)
