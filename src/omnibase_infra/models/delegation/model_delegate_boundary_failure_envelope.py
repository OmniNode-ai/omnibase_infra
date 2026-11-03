# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A boundary failure terminal as it arrives off the bus, inside its envelope (OMN-22216).

.. versionadded:: OMN-22216
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from omnibase_infra.models.delegation.model_delegate_boundary_failure_terminal import (
    ModelDelegateBoundaryFailureTerminal,
)

__all__ = ["ModelDelegateBoundaryFailureEnvelope"]


class ModelDelegateBoundaryFailureEnvelope(BaseModel):
    """The envelope carrier: a boundary failure terminal under ``payload``."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    envelope_id: UUID = Field(...)
    payload: ModelDelegateBoundaryFailureTerminal = Field(...)

    @classmethod
    def find(cls, carrier: object) -> ModelDelegateBoundaryFailureTerminal | None:
        """The boundary failure terminal ``carrier`` holds, enveloped or bare, or ``None``."""
        try:
            return cls.model_validate(carrier).payload
        except ValidationError:
            pass
        try:
            return ModelDelegateBoundaryFailureTerminal.model_validate(carrier)
        except ValidationError:
            return None
