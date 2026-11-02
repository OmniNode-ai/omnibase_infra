# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result: the introspection event for an announcement, or why there is none."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from omnibase_infra.models.registration.model_node_introspection_event import (
    ModelNodeIntrospectionEvent,
)


class ModelRegistrationFromAnnouncementResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    event: ModelNodeIntrospectionEvent | None = None
    skipped_reason: str = ""


__all__ = ["ModelRegistrationFromAnnouncementResult"]
