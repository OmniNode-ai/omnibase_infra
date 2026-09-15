# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Heartbeat request -- also the per-tick revocation re-check."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, SecretStr


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
    # OMN-18385: ``SecretStr``, not ``str``. This model is reconstructed
    # from a bus payload and is re-serialised into logs, error contexts and
    # dead-letter envelopes; ``SecretStr`` renders as a mask in every one of
    # those. Two customer bearer tokens reached
    # onex.dlq.omnibase-infra.commands.v1 in cleartext (offsets 137/138,
    # onex-dev) because this field was a plain string. The handler reads the
    # real value with ``.get_secret_value()`` at its single unwrap point.
    access_token: SecretStr = Field(min_length=1)


__all__ = ["ModelGatewayHeartbeatRequest"]
