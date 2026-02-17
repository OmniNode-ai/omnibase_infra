# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Single configuration requirement model extracted from an ONEX contract.

.. versionadded:: 0.10.0
    Created as part of OMN-2287.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumInfraTransportType


class ModelConfigRequirement(BaseModel):
    """A single configuration requirement extracted from a contract.

    Attributes:
        key: The configuration key name (e.g., ``POSTGRES_DSN``).
        transport_type: The transport type this key belongs to.
        source_contract: Path to the contract file that declared this requirement.
        source_field: The contract field path that triggered this requirement
            (e.g., ``metadata.transport_type``, ``handler_routing.handlers[0].handler_type``).
        required: Whether this key is required for the handler to function.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    key: str = Field(..., description="Configuration key name.")
    transport_type: EnumInfraTransportType = Field(
        ..., description="Transport type this key belongs to."
    )
    source_contract: Path = Field(
        ..., description="Path to the contract that declared this requirement."
    )
    source_field: str = Field(
        default="", description="Contract field path that triggered this."
    )
    required: bool = Field(default=True, description="Whether this key is required.")
