# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Typed ownership subset of a node contract."""

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.models.contracts.subcontracts.model_db_ownership_subcontract import (
    ModelDbOwnershipSubcontract,
)


class ModelNodeOwnershipDocument(BaseModel):
    """Ownership-relevant typed subset of a node contract."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    name: str = Field(..., min_length=1)
    db_io: ModelDbOwnershipSubcontract


__all__ = ["ModelNodeOwnershipDocument"]
