# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result of boot step 0: the inventory, the announcements and the catalog built from them (OMN-21590)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from omnibase_core.models.events.contract_registration.model_node_announcement import (
    ModelNodeAnnouncement,
)
from omnibase_core.models.nodes.package_announce.model_catalog_outcome import (
    ModelCatalogOutcome,
)
from omnibase_core.models.nodes.package_announce.model_node_catalog import (
    ModelNodeCatalog,
)
from omnibase_infra.models.node_announce.model_package_inventory import (
    ModelPackageInventory,
)


class ModelPackageAnnounceResult(BaseModel):
    """The catalog the container receives, with what produced it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    inventory: ModelPackageInventory
    announcements: tuple[ModelNodeAnnouncement, ...]
    catalog: ModelNodeCatalog
    outcomes: tuple[ModelCatalogOutcome, ...]
