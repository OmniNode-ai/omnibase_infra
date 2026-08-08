# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Frozen configuration for the gateway attach effect node.

All values resolve from ``contract.yaml`` and contract overlays; secret
material (Keycloak admin credentials) is referenced by name here and
resolved from Infisical at the effect boundary inside
``service_keycloak_token_validator`` -- never read from a bare env var and
never embedded in this model (operator ruling 2026-08-08: config lives in
contract overlays + Infisical, env for bootstrap only).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelGatewayAttachConfig(BaseModel):
    """Frozen gateway attach node configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Keycloak realm issuer used to validate the access token's ``iss`` claim
    # and to derive the JWKS endpoint. Resolved indirectly: this is a
    # contract ref name, not the literal URL (feedback_all_urls_from_contracts).
    keycloak_issuer_ref: str = Field(default="gateway.attach.keycloak.issuer")
    # Token introspection endpoint ref (RFC 7662). Introspection, not local
    # exp-only validation, is what makes revocation-within-TTL observable:
    # a disabled Keycloak client makes the introspection response
    # ``active: false`` immediately, independent of the token's own exp.
    keycloak_introspection_ref: str = Field(
        default="gateway.attach.keycloak.introspection"
    )
    keycloak_admin_client_ref: str = Field(
        default="gateway.attach.keycloak.admin_client_credentials"
    )
    # Audience every attach token must carry.
    required_audience: str = Field(default="gateway-attach")
    # Session lifecycle. Heartbeat interval mirrors the forwarder's
    # liveness.heartbeat_interval_seconds (node_bus_forwarder_effect
    # contract.yaml) so link-health projections can share one cadence.
    heartbeat_interval_seconds: int = Field(default=15, gt=0)
    # A session is considered DEGRADED once this many seconds pass with no
    # successful heartbeat re-validation.
    session_degraded_after_seconds: int = Field(default=60, gt=0)
    # Hard ceiling on session lifetime regardless of token exp -- bounds the
    # blast radius of a token whose exp claim is misconfigured too far out.
    max_session_ttl_seconds: int = Field(default=3600, gt=0)


__all__ = ["ModelGatewayAttachConfig"]
