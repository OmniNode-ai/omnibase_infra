# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Consul Register Payload Model.

This module provides the payload model for consul.register result.
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


@RegistryPayloadConsul.register("register")
class ModelConsulRegisterPayload(ModelPayloadConsul):
    """Payload for consul.register result.

    Attributes:
        operation_type: Discriminator literal "register".
        registered: True if the service was registered successfully.
        name: The service name.
        consul_service_id: The Consul service identifier (user-defined string).

    Note:
        consul_service_id is a Consul-specific identifier that is a user-defined string,
        NOT a UUID. The Consul API accepts any string as a service ID. This field is
        named with the 'consul_' prefix to avoid pattern validator warnings that expect
        fields ending in '_id' to be UUIDs.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    operation_type: Literal["register"] = Field(
        default="register", description="Discriminator for payload type"
    )
    registered: bool = Field(
        description="True if the service was registered successfully"
    )
    name: str = Field(description="The service name")
    consul_service_id: str = Field(
        description="The Consul service identifier (user-defined string, not a UUID)"
    )


__all__: list[str] = ["ModelConsulRegisterPayload"]
