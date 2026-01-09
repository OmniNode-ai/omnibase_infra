# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Consul KV Get Recurse Payload Model.

This module provides the payload model for consul.kv_get with recurse=True.
"""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field

from omnibase_infra.handlers.models.consul.model_consul_kv_item import (
    ModelConsulKVItem,
)
from omnibase_infra.handlers.models.consul.model_payload_consul import (
    ModelPayloadConsul,
)
from omnibase_infra.handlers.models.consul.registry_payload_consul import (
    RegistryPayloadConsul,
)


@RegistryPayloadConsul.register("kv_get_recurse")
class ModelConsulKVGetRecursePayload(ModelPayloadConsul):
    """Payload for consul.kv_get with recurse=True.

    Attributes:
        operation_type: Discriminator literal "kv_get_recurse".
        found: True if any keys were found under the prefix.
        items: List of KV items found under the prefix.
        count: Number of items found.
        index: The Consul response index for blocking queries.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    operation_type: Literal["kv_get_recurse"] = Field(
        default="kv_get_recurse", description="Discriminator for payload type"
    )
    found: bool = Field(description="True if any keys were found under the prefix")
    items: list[ModelConsulKVItem] = Field(
        description="List of KV items found under the prefix"
    )
    count: int = Field(description="Number of items found")
    index: int = Field(description="Consul response index for blocking queries")


__all__: list[str] = ["ModelConsulKVGetRecursePayload"]
