# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Frozen configuration for the gateway attach effect node.

All values resolve from ``contract.yaml`` and contract overlays; secret
material (Keycloak admin credentials) is referenced by name here and
resolved from Infisical at the effect boundary inside
``HandlerGatewayHeartbeat._introspect`` -- never read from a bare env var and
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
    # JWKS endpoint ref (RFC 7517). Fetched at attach time (and re-checked at
    # heartbeat time to re-bind identity) so the token's signature is
    # verified against Keycloak's real signing keys before any claim is
    # trusted -- OMN-15918 R1: attach-time decode previously trusted claims
    # from a structurally-valid-but-unsigned/forged token.
    keycloak_jwks_ref: str = Field(default="gateway.attach.keycloak.jwks")
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
    # Enforced (not merely stored) since OMN-16022: heartbeat and detach
    # reject a session past the ``expires_at`` derived from this.
    max_session_ttl_seconds: int = Field(default=3600, gt=0)
    # OMN-16022: bounded degraded mode. The maximum time a session may
    # survive without a SUCCESSFUL Keycloak revalidation, after which it is
    # torn down regardless of whether Keycloak is reachable.
    #
    # Why 900s: it is one attach-token lifetime. The per-tenant
    # client-credentials token this node validates is minted with a 900s
    # maximum TTL (invariant 2 of the OMN-15952 renewal design), so a
    # session that has gone a full token lifetime without revalidation has
    # already outlived the credential that justified it -- continuing to
    # trust it past that point extends trust strictly beyond anything the
    # IdP ever asserted. Choosing the token lifetime rather than a smaller
    # number also keeps a routine Keycloak blip (seconds to minutes,
    # absorbed by circuit_breaker_reset_timeout_seconds and by the
    # OMN-15918 outage/revocation split) from ever reaching the ceiling:
    # only a sustained partition does.
    #
    # This is deliberately NOT max_session_ttl_seconds. That is the 3600s
    # *session* ceiling applied at attach; this is the *revalidation*
    # bound, a different bound at a different layer. See the OMN-15952
    # design doc rev-3 correction in section 2 -- conflating the two is
    # exactly the error this pair of constants exists to prevent.
    max_unverified_session_seconds: int = Field(default=900, gt=0)
    # Circuit breaker (MixinAsyncCircuitBreaker) thresholds shared by the
    # JWKS fetch (attach + heartbeat) and RFC 7662 introspection (heartbeat)
    # HTTP calls to Keycloak -- OMN-15918 R4: distinguishes "Keycloak
    # unreachable" (raise InfraUnavailableError, session left untouched)
    # from "Keycloak said inactive" (real revocation).
    circuit_breaker_threshold: int = Field(default=5, gt=0)
    circuit_breaker_reset_timeout_seconds: float = Field(default=30.0, gt=0)


__all__ = ["ModelGatewayAttachConfig"]
