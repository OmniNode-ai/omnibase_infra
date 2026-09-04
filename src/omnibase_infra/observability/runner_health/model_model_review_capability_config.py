# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Source-controlled model-review runner-overlay contract."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelModelReviewCapabilityConfig(BaseModel):
    """Opaque references required for a model-review runner overlay.

    The UUIDs are logical references only. They are deliberately not endpoint
    addresses, credential values, or environment-variable names.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    active: bool = Field(
        default=False,
        description=(
            "Whether a separately provisioned runner overlay may be considered "
            "for model review. Defaults to False so this config record alone "
            "cannot activate any runner."
        ),
    )
    runner_label: Literal["model-review"] = Field(
        default="model-review",
        description="Required runner capability label.",
    )
    credential_reference_id: UUID = Field(
        ...,
        description="Opaque logical identifier for the required credential reference.",
    )
    endpoint_reference_id: UUID = Field(
        ...,
        description="Opaque logical identifier for the required endpoint reference.",
    )
    healthcheck_reference_id: UUID = Field(
        ...,
        description="Opaque logical identifier for the required health assertion.",
    )

    @model_validator(mode="after")
    def validate_reference_ids_are_distinct(
        self,
    ) -> ModelModelReviewCapabilityConfig:
        """Keep credential, endpoint, and health assertions independently provable."""
        reference_ids = {
            self.credential_reference_id,
            self.endpoint_reference_id,
            self.healthcheck_reference_id,
        }
        if len(reference_ids) != 3:
            msg = "credential, endpoint, and healthcheck reference IDs must differ"
            raise ValueError(msg)
        return self


__all__ = ["ModelModelReviewCapabilityConfig"]
