# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Deterministic grants for a principal across its logical brokers."""

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_broker_grant_derive_compute.models.model_broker_grant import (
    ModelBrokerGrant,
)


class ModelBrokerGrantDerivation(BaseModel):
    """Sorted, deduplicated desired permissions for one principal."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    principal: str = Field(min_length=1)
    grants: tuple[ModelBrokerGrant, ...]
