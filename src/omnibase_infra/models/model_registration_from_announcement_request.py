# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request: one node announcement plus the runtime identity to complete it."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from omnibase_core.models.events.contract_registration.model_node_announcement import (
    ModelNodeAnnouncement,
)
from omnibase_infra.models.model_runtime_identity import ModelRuntimeIdentity


class ModelRegistrationFromAnnouncementRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    announcement: ModelNodeAnnouncement
    identity: ModelRuntimeIdentity


__all__ = ["ModelRegistrationFromAnnouncementRequest"]
