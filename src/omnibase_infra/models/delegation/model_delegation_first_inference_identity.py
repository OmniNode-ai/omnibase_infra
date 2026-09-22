# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Identity observed on a delegation run's first inference attempt."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelDelegationFirstInferenceIdentity(BaseModel):
    """Identity of the actual first inference rung, from its attempt record.

    Backend and model are route/provider labels, not UUID entity identifiers.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    backend_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    tier: str = Field(min_length=1)
    provider: str | None = Field()

    @model_validator(mode="after")
    def _validate_identity_values(self) -> ModelDelegationFirstInferenceIdentity:
        for name in ("backend_id", "model_id", "tier"):
            value = getattr(self, name)
            if not value.strip() or value != value.strip():
                raise ValueError(f"{name} must be nonblank and trimmed")
        if self.provider is not None and (
            not self.provider.strip() or self.provider != self.provider.strip()
        ):
            raise ValueError("provider must be nonblank and trimmed when present")
        return self
