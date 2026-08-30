# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""ModelGatewayCredentialBase -- what every resolved gateway credential has (OMN-17205).

onex-api resolves a caller's tenant from either of two credential kinds on
equal footing: an OIDC bearer minted from a client secret, or a tenant API key
presented in ``x-api-key``. Both are the SAME thing to a caller -- one tenant,
one gateway origin, one way to authenticate -- and differ only in the
authenticator they carry.

This base states that shared part once so callers can name a single type. It is
deliberately a base class and not a ``ModelA | ModelB`` union: the union spelling
adds a non-optional union to every signature that touches a credential, and the
repo's union budget ratchet (``onex-validate-unions``) exists precisely to stop
that growth. A base class also keeps the shared invariants -- both fields
non-empty, frozen, ``extra="forbid"`` -- in one place instead of two.

It is not instantiable-and-meaningful on its own: a bare base carries no
authenticator, and every consumer narrows to a concrete subclass before
authenticating. That narrowing is a single ``isinstance`` at the one seam that
builds headers, not a condition threaded through the call graph.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelGatewayCredentialBase"]


class ModelGatewayCredentialBase(BaseModel):
    """Tenant identity plus the gateway origin, common to every credential kind."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    tenant_slug: str = Field(min_length=1)
    # Gateway origin the projection/attach paths are appended to. Supplied by
    # stored configuration, never a literal in code.
    base_url: str = Field(min_length=1)
