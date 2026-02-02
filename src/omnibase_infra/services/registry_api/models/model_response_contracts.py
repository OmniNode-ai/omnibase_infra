# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Response model for list_contracts endpoint.

Related Tickets:
    - OMN-1845: Contract Registry Persistence
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.services.registry_api.models.model_contract_view import (
    ModelContractView,
)
from omnibase_infra.services.registry_api.models.model_pagination_info import (
    ModelPaginationInfo,
)
from omnibase_infra.services.registry_api.models.model_warning import ModelWarning


class ModelResponseListContracts(BaseModel):
    """Response model for the GET /registry/contracts endpoint.

    Provides a paginated list of registered contracts with optional warnings
    for partial success scenarios.

    Attributes:
        contracts: List of registered contracts matching the query
        pagination: Pagination information for the result set
        warnings: List of warnings for partial success scenarios
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    contracts: list[ModelContractView] = Field(
        default_factory=list,
        description="List of registered contracts matching the query",
    )
    pagination: ModelPaginationInfo = Field(
        ...,
        description="Pagination information for the result set",
    )
    warnings: list[ModelWarning] = Field(
        default_factory=list,
        description="Warnings for partial success scenarios",
    )


__all__ = ["ModelResponseListContracts"]
