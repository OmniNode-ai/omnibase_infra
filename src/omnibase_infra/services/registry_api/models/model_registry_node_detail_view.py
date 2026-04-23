# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Registry node detail view model."""

from __future__ import annotations

from pydantic import Field

from omnibase_infra.services.registry_api.models.model_registry_node_view import (
    ModelRegistryNodeView,
)


class ModelRegistryNodeDetailView(ModelRegistryNodeView):
    """Detailed registry node view with projection-backed detail fields."""

    protocols: list[str] = Field(
        default_factory=list,
        description="Protocols declared in the registration projection",
    )
    intent_types: list[str] = Field(
        default_factory=list,
        description="Intent types declared in the registration projection",
    )


__all__ = ["ModelRegistryNodeDetailView"]
