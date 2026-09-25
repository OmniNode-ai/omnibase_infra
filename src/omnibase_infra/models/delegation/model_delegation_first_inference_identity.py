# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Identity observed on a delegation run's first inference attempt (OMN-18930)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelDelegationFirstInferenceIdentity(BaseModel):
    """Identity of the actual first inference rung, read from its attempt record.

    ``backend`` and ``model`` are the routing labels the attempt record carries
    (its ``backend_id`` and ``model_id`` wire keys), not ONEX entity references.
    ``provider`` is required and may be an explicit ``None`` when no evidence
    names the first attempt's provider; it is never copied from the final one.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    backend: str = Field(min_length=1)
    model: str = Field(min_length=1)
    tier: str = Field(min_length=1)
    provider: str | None = Field()

    @model_validator(mode="after")
    def _validate_identity_values(self) -> ModelDelegationFirstInferenceIdentity:
        for name in ("backend", "model", "tier"):
            value = getattr(self, name)
            if not value.strip() or value != value.strip():
                raise ValueError(f"{name} must be nonblank and trimmed")
        if self.provider is not None and (
            not self.provider.strip() or self.provider != self.provider.strip()
        ):
            raise ValueError("provider must be nonblank and trimmed when present")
        return self
