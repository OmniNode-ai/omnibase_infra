# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Contract reference model for dashboard display.

Related Tickets:
    - OMN-1845: Contract Registry Persistence
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelContractRef(BaseModel):
    """Lightweight contract reference.

    Used to reference a contract without including full details,
    suitable for embedding in topic views.

    Attributes:
        contract_id: Unique identifier in format node_name:major.minor.patch
        node_name: Name of the node this contract belongs to
        version: Semantic version string in format major.minor.patch
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # ONEX_EXCLUDE: pattern_validator - contract_id is a derived natural key (name:version), not UUID
    contract_id: str = Field(
        ...,
        description="Unique identifier in format node_name:major.minor.patch",
    )
    # ONEX_EXCLUDE: pattern_validator - node_name is the contract name, not an entity reference
    node_name: str = Field(
        ...,
        description="Name of the node this contract belongs to",
    )
    version: str = Field(
        ...,
        description="Semantic version string in format major.minor.patch",
    )


__all__ = ["ModelContractRef"]
