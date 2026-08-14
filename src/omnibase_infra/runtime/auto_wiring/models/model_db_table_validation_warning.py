# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Typed warning for a declared table absent from a catalog connection."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelDbTableValidationWarning(BaseModel):
    """One declared table absent from the current physical connection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: Literal["missing_db_table"] = "missing_db_table"
    severity: Literal["warning"] = "warning"
    table: str
    database_ref: str
    schema: str  # type: ignore[assignment]
    node: str


__all__ = ["ModelDbTableValidationWarning"]
