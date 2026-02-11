# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Output model for checkpoint validation.

Ticket: OMN-2143
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelCheckpointValidateOutput(BaseModel):
    """Result of checkpoint structural validation.

    Warning:
        **Non-standard __bool__ behavior**: Returns ``True`` only when
        ``is_valid`` is True. Differs from typical Pydantic behavior.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    is_valid: bool = Field(
        ...,
        description="Whether the checkpoint passed all validation checks.",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID echoed from the input.",
    )
    errors: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Validation errors that make the checkpoint unusable.",
    )
    warnings: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Non-fatal warnings about checkpoint quality.",
    )

    def __bool__(self) -> bool:
        """Allow using result in boolean context.

        Warning:
            **Non-standard __bool__ behavior**: Returns ``True`` only when
            ``is_valid`` is True. Differs from typical Pydantic behavior.
        """
        return self.is_valid


__all__: list[str] = ["ModelCheckpointValidateOutput"]
