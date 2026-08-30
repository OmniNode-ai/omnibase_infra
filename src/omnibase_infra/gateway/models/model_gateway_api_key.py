# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""ModelGatewayApiKeyCredential -- a tenant API key for control-plane reads (OMN-17205).

WHY THIS EXISTS ALONGSIDE ``ModelGatewayCredential``
    onex-api resolves a caller's tenant from either of two credential kinds on
    equal footing: an OIDC bearer, or a tenant API key in the ``x-api-key``
    header (``routers/ledger_projection._resolve_tenant``). The bearer arm
    cannot serve the operator's own read today -- the only credential a tenant
    can hold is the P0B machine client, whose grant carries
    ``aud=redpanda-events`` (OMN-16687), not a control-plane audience. The API
    key arm has no such coupling and is mintable by the tenant, so it is the
    kind the operator probe presents.

    Two models rather than one with optional fields: a single model with
    ``client_secret | api_key`` both nullable would represent the
    half-configured state that ``StoreGatewayCredential`` exists to refuse, and
    every consumer would have to re-check which half is populated.

``api_key`` is a ``SecretStr`` for the same reason ``client_secret`` is: this
value travels through tracebacks, ``repr()`` and structured log records the
moment anything downstream raises. Read it only where it goes on the wire.
"""

from __future__ import annotations

from pydantic import Field, SecretStr

from omnibase_infra.gateway.models.model_gateway_credential_base import (
    ModelGatewayCredentialBase,
)

__all__ = ["ModelGatewayApiKeyCredential"]


class ModelGatewayApiKeyCredential(ModelGatewayCredentialBase):
    """A tenant API key plus the gateway origin it authenticates against."""

    # The plaintext key. The server resolves the tenant from it; nothing about
    # the tenant is taken from the request otherwise.
    api_key: SecretStr
    # The name this key is stored under in ``~/.onex/credentials.json``. Kept
    # on the resolved model so an error message can say WHICH reference failed
    # without the caller re-reading config.
    api_key_ref: str = Field(min_length=1)
