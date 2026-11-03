# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One kernel plugin of one installed package, as the metadata reader found it (OMN-21591)."""

from __future__ import annotations

import yaml
from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.models.events.contract_registration.model_node_announcement import (
    ModelNodeAnnouncement,
)


class ModelPackagePluginRecord(BaseModel):
    """What the metadata reader learned about one kernel plugin entry.

    A plugin has no contract file. Its announcement is hashed over a descriptor
    document built from the package's own metadata, so the catalog's admission
    check (installed bytes hash to the announced hash) holds for it as it does
    for a node contract. The record is read from metadata only; nothing is
    imported to produce it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, description="Plugin name the overlay enables")
    distribution: str = Field(description="Installed distribution name")
    version: str = Field(description="Exact installed distribution version")
    target: str = Field(
        min_length=1, description="Importable ``module:attribute`` of the plugin class"
    )

    @property
    def descriptor(self) -> bytes:
        """The deterministic descriptor document the announcement is hashed over."""
        return yaml.safe_dump(
            {
                "kernel_plugin": self.name,
                "distribution": self.distribution,
                "version": self.version,
                "target": self.target,
            },
            sort_keys=True,
        ).encode("utf-8")

    @property
    def content_hash(self) -> str:
        """Canonical content hash of the descriptor."""
        return ModelNodeAnnouncement.canonical_content_hash(self.descriptor)
