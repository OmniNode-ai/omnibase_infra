# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Plugin discovery report model.

The ModelPluginDiscoveryReport for structured reporting of plugin discovery
outcomes across an entire entry-point group.

Design Pattern:
    The report aggregates all ``ModelPluginDiscoveryEntry`` results
    produced while scanning one entry-point group, providing quick-access
    properties for filtering rejected entries and detecting errors.

Related:
    - OMN-2012: Create ModelPluginDiscoveryReport + ModelPluginDiscoveryEntry
    - OMN-1346: Registration Code Extraction
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.models.model_plugin_discovery_entry import (
    ModelPluginDiscoveryEntry,
)

__all__ = ["ModelPluginDiscoveryReport"]


class ModelPluginDiscoveryReport(BaseModel):
    """Structured report for a single entry-point group discovery pass.

    Aggregates all ``ModelPluginDiscoveryEntry`` results produced while scanning
    one entry-point group (e.g. ``omnibase_infra.projectors``). Provides
    quick-access properties for filtering rejected entries and detecting errors.

    Attributes:
        group: Entry-point group name that was scanned.
        discovered_count: Total number of entry-points found in the group.
        accepted: Plugin IDs that were successfully registered.
        entries: Complete tuple of ``ModelPluginDiscoveryEntry`` results.
    """

    model_config = ConfigDict(frozen=True)

    group: str
    discovered_count: int
    accepted: tuple[str, ...] = Field(default_factory=tuple)
    entries: tuple[ModelPluginDiscoveryEntry, ...] = Field(default_factory=tuple)

    @property
    def rejected(self) -> list[ModelPluginDiscoveryEntry]:
        """Return entries whose status is not ``"accepted"``."""
        return [e for e in self.entries if e.status != "accepted"]

    @property
    def has_errors(self) -> bool:
        """Detect whether any entry suffered an import or instantiation failure."""
        error_statuses = {"import_error", "instantiation_error"}
        return any(e.status in error_statuses for e in self.entries)
