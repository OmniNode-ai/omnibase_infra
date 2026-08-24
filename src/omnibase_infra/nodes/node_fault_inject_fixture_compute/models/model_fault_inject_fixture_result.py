# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Result model for the fault-injection fixture node.

Ticket: OMN-16265
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelFaultInjectFixtureResult(BaseModel):
    """Deterministic, size-controlled result of one fixture run.

    ``padding_byte_length`` is computed independently of ``inflate_result_bytes``
    (from the actual serialized ``padding`` field) so a caller/test can assert
    the two agree rather than trusting the echoed request value.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(
        ...,
        description="Correlation ID echoed from the command.",
    )
    inflate_result_bytes: int = Field(
        ...,
        description="The inflate_result_bytes value requested by the command.",
    )
    padding: str = Field(
        default="",
        description=(
            "Deterministic ASCII filler sized to inflate_result_bytes. This "
            "is the field that amplifies the serialized result past the "
            "broker's publish size ceiling when a non-zero size is requested."
        ),
    )
    padding_byte_length: int = Field(
        ...,
        ge=0,
        description="Actual UTF-8 byte length of `padding`, computed from the field itself.",
    )


__all__: list[str] = ["ModelFaultInjectFixtureResult"]
