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
    max_session_ttl_seconds: int = Field(default=3600, gt=0)
    # OMN-15952 renewal cycle -- the contract-declared terms an unattended
    # runtime must obey to survive its own session ceiling. These are
    # CLIENT-facing policy (handed back at attach in
    # ModelGatewayRenewalDirective), not server-side bounds: nothing on this
    # node tears a session down because of them.
    #
    # How early re-grant + re-attach must be COMPLETE, ahead of the
    # session's expires_at. 120s against a 900s attach token leaves ~87% of
    # the token's life before renewal starts, while still covering the three
    # things that have to fit inside the margin: worst-case clock skew
    # between the runtime, Keycloak and this node; the round trip of the
    # token grant plus the attach call plus this node's own JWKS
    # verification; and at least one backoff-retry of a transient failure.
    # A margin sized only to the happy-path round trip is the classic
    # expiry-boundary defect -- the token is valid when the request is sent
    # and expired when it is validated.
    renewal_margin_seconds: int = Field(default=120, gt=0)
    # Width of the decorrelation window that opens before renewal_margin.
    # A fleet provisioned in one bootstrap batch shares an attach instant,
    # so without jitter it also shares a renewal instant and stampedes
    # Keycloak's token endpoint every cycle, forever -- the synchronization
    # is self-sustaining because a batch that renews together stays
    # together. Each runtime picks its own moment uniformly in
    # [renew_not_before, renew_at]. Zero is permitted (ge=0) so a
    # single-runtime deployment can opt out of spreading it does not need,
    # which is why this is not gt=0 like the margin.
    renewal_jitter_seconds: int = Field(default=30, ge=0)
    # Circuit breaker (MixinAsyncCircuitBreaker) thresholds shared by the
    # JWKS fetch (attach + heartbeat) and RFC 7662 introspection (heartbeat)
    # HTTP calls to Keycloak -- OMN-15918 R4: distinguishes "Keycloak
    # unreachable" (raise InfraUnavailableError, session left untouched)
    # from "Keycloak said inactive" (real revocation).
    circuit_breaker_threshold: int = Field(default=5, gt=0)
    circuit_breaker_reset_timeout_seconds: float = Field(default=30.0, gt=0)


__all__ = ["ModelGatewayAttachConfig"]
