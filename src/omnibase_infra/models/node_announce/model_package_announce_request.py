# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request of boot step 0: announce the installed package set and build the catalog (OMN-21590)."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.models.nodes.package_announce.model_node_announce_policy import (
    ModelNodeAnnouncePolicy,
)


class ModelPackageAnnounceRequest(BaseModel):
    """What the lane overlay binds for one run of step 0.

    ``policy`` of ``None`` means the lane binds no admission restriction: the
    catalog admits every installed contract, as discovery did before the catalog
    existed, and the runtime profile filter downstream still decides ownership.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    topic: str = Field(description="Announcement topic the announcer publishes to")
    announcer_id: UUID = Field(description="Identity of this announcer")
    generation: int = Field(
        ge=1, description="Generation of this announcer's inventory"
    )
    announced_at: datetime = Field(
        description="Timezone-aware time of the announcement"
    )
    active_packages: frozenset[str] | None = Field(
        default=None, description="Runtime package surface; None means every package"
    )
    policy: ModelNodeAnnouncePolicy | None = Field(
        default=None,
        description="Admission policy; None means the permissive Tier 0 policy",
    )
    catalog_group: str = Field(
        default="node-catalog", description="Consumer group the catalog subscribes with"
    )
