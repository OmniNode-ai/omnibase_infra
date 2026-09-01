# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""ModelGatewayIdentity -- who the gateway says the stored credential is (OMN-17028).

The point of this model is that the identity is the SERVER'S answer, not the
local config's. ``~/.onex/config.yaml`` carries a ``tenant_slug`` the operator
typed; the gateway resolves the tenant from the key itself. Keeping the two
apart in the type system is what lets ``onex auth status`` report a credential
that authenticates as a tenant other than the one its own label claims --
rather than printing the label back and calling that verification.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelGatewayIdentity"]


class ModelGatewayIdentity(BaseModel):
    """The tenant a gateway resolved from a presented credential."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: UUID = Field(
        description=(
            "Immutable tenant UUID. Typed as a UUID rather than the string the "
            "wire carries so a gateway answering 200 with a tenant_id that is "
            "not one is refused here, at the parse, instead of being reported "
            "as a verified identity."
        )
    )
    tenant_slug: str = Field(
        min_length=1, description="Tenant slug as the gateway resolved it."
    )
