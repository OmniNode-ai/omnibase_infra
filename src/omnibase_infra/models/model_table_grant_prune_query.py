# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Query identifying a relation for prune-grant derivation."""

from pydantic import BaseModel, ConfigDict, Field


class ModelTableGrantPruneQuery(BaseModel):
    """Query identifying a relation for prune-grant derivation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_name: str = Field(min_length=1)
    table_name: str = Field(min_length=1)
