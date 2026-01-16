# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Consul KV Get Not Found Payload Model.

This module provides the payload model for consul.kv_get when key is not found.
"""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field

from omnibase_infra.handlers.models.consul.model_payload_consul import (
    ModelPayloadConsul,
)
from omnibase_infra.handlers.models.consul.registry_payload_consul import (
    RegistryPayloadConsul,
)


@RegistryPayloadConsul.register("kv_get_not_found")
class ModelConsulKVGetNotFoundPayload(ModelPayloadConsul):
    """Payload for consul.kv_get when key is not found.

    Attributes:
        operation_type: Discriminator literal "kv_get_not_found".
        found: Always False for this payload type.
        key: The KV key path that was queried.
        value: Always None for not-found case.
        index: The Consul response index for blocking queries.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", coerce_numbers_to_str=False, from_attributes=True
    )

    operation_type: Literal["kv_get_not_found"] = Field(
        default="kv_get_not_found", description="Discriminator for payload type"
    )
    found: Literal[False] = Field(
        default=False, description="Indicates the key was not found"
    )
    key: str = Field(description="The KV key path that was queried")
    value: None = Field(default=None, description="Always None for not-found case")
    index: int = Field(description="Consul response index for blocking queries")


__all__: list[str] = ["ModelConsulKVGetNotFoundPayload"]
