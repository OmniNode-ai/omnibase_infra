# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Detach request -- explicit, edge-initiated session teardown."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, SecretStr


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
    # OMN-18385: ``SecretStr``, not ``str``. This model is reconstructed
    # from a bus payload and is re-serialised into logs, error contexts and
    # dead-letter envelopes; ``SecretStr`` renders as a mask in every one of
    # those. Two customer bearer tokens reached
    # onex.dlq.omnibase-infra.commands.v1 in cleartext (offsets 137/138,
    # onex-dev) because this field was a plain string. The handler reads the
    # real value with ``.get_secret_value()`` at its single unwrap point.
    access_token: SecretStr = Field(min_length=1)
    reason: str = Field(min_length=1, max_length=500)


__all__ = ["ModelGatewayDetachRequest"]
