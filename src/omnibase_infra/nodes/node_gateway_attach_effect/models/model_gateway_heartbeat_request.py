# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Heartbeat request -- also the per-tick revocation re-check."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelGatewayHeartbeatRequest(BaseModel):
    """Input to ``gateway.heartbeat``.

    Carries a fresh access token (client-credentials tokens are short-lived;
    the edge re-mints one per heartbeat cadence) so each heartbeat performs a
    real Keycloak introspection call -- this is the mechanism that makes
    revocation observable within one heartbeat interval rather than only at
    the stale token's original exp.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: UUID
    access_token: str = Field(min_length=1)


__all__ = ["ModelGatewayHeartbeatRequest"]
