# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed request for a derived write in a caller-owned transaction."""

from collections.abc import Callable, Mapping

from pydantic import BaseModel, ConfigDict, SkipValidation

from omnibase_infra.protocols.protocol_projection_write_cursor import (
    ProtocolProjectionWriteCursor,
)


class ModelProjectionDerivedWriteRequest(BaseModel):
    """SQL write inputs and callbacks supplied by the transaction owner."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
        arbitrary_types_allowed=True,
    )

    # This structural protocol is not runtime-checkable; retain the cursor as-is.
    cursor: SkipValidation[ProtocolProjectionWriteCursor]
    physical_schema: str
    table: str
    stored_rows: tuple[dict[str, object], ...]
    conflict_columns: tuple[str, ...]
    input_columns: tuple[str, ...]
    derived_columns: tuple[str, ...]
    returning: tuple[str, ...]
    derive: Callable[[Mapping[str, object]], Mapping[str, object]]
    adapt: Callable[[dict[str, object]], dict[str, object]]


__all__ = ["ModelProjectionDerivedWriteRequest"]
