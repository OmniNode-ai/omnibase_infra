# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The explicit credential declaration of a lane-added Bifrost backend (OMN-17099)."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.runtime.models.enum_bifrost_lane_credential_kind import (
    EnumBifrostLaneCredentialKind,
)


class ModelBifrostLaneBackendCredential(BaseModel):
    """A credential REFERENCE or an explicit ``none`` — never a secret value.

    ``secret_ref`` is a logical name the effect boundary resolves through the
    secret store, the same field a base-contract backend declares. Whatever a
    lane writes here is a house credential by construction: omnimarket derives
    the house set from every resolved backend's reference
    (``customer_key_terminus.house_credential_refs``), so a customer route can
    never bind it (OMN-17082, C10).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    kind: EnumBifrostLaneCredentialKind
    secret_ref: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _validate_kind_matches_reference(self) -> Self:
        if self.kind is EnumBifrostLaneCredentialKind.SECRET_REF:
            if self.secret_ref is None or not self.secret_ref.strip():
                raise ValueError(
                    "credential kind 'secret_ref' must name the secret_ref it "
                    "authenticates with"
                )
        elif self.secret_ref is not None:
            raise ValueError(
                "credential kind 'none' must not carry a secret_ref; declare "
                "kind 'secret_ref' to authenticate"
            )
        return self


__all__ = ["ModelBifrostLaneBackendCredential"]
