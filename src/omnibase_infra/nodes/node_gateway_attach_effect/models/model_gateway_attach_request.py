# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Attach request -- the edge dials in with a client-credentials token."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelGatewayAttachRequest(BaseModel):
    """Input to ``gateway.attach``: one edge presenting a bearer token."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Raw access token obtained by the edge via the Keycloak client_credentials
    # grant against its per-tenant confidential client. Never logged verbatim
    # by any handler in this node.
    access_token: str = Field(min_length=1)
    # Caller-declared edge instance identity (host label), used only for
    # session bookkeeping/observability -- never trusted for authorization,
    # which is derived entirely from the validated token claims.
    edge_instance_id: str = Field(min_length=1, max_length=255)


__all__ = ["ModelGatewayAttachRequest"]
