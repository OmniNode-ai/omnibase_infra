# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Consul KV Put Payload Model.

This module provides the payload model for consul.kv_put result.
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


@RegistryPayloadConsul.register("kv_put")
class ModelConsulKVPutPayload(ModelPayloadConsul):
    """Payload for consul.kv_put result.

    Attributes:
        operation_type: Discriminator literal "kv_put".
        success: True if the put operation succeeded.
        key: The KV key path that was written.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    operation_type: Literal["kv_put"] = Field(
        default="kv_put", description="Discriminator for payload type"
    )
    success: bool = Field(description="True if the put operation succeeded")
    key: str = Field(description="The KV key path that was written")


__all__: list[str] = ["ModelConsulKVPutPayload"]
