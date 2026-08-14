# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Detach request -- explicit, edge-initiated session teardown."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelGatewayDetachRequest(BaseModel):
    """Input to ``gateway.detach``.

    OMN-15918 R2: ``access_token`` is required so
    ``HandlerGatewayDetach.handle`` can bind the caller to the STORED
    session's tenant/principal/client identity before deleting -- the
    previous shape (``session_id`` + free-text ``reason``, no credential)
    let any caller holding a session identifier detach any tenant's session.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: UUID
    access_token: str = Field(min_length=1)
    reason: str = Field(min_length=1, max_length=500)


__all__ = ["ModelGatewayDetachRequest"]
