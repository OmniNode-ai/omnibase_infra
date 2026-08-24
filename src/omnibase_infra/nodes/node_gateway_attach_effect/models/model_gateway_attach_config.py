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
    # The audience every attach token must carry -- and, since OMN-16023,
    # the ONLY audience it may carry. The validator asserts exact set
    # equality rather than membership: a mapper that later adds a second
    # audience to the ga-* client produces a dual-audience token, which a
    # membership check (or a denylist of one known-bad broker audience)
    # would pass and which invariant 4 of the OMN-15952 renewal design
    # forbids outright.
    required_audience: str = Field(default="gateway-attach")
    # OMN-16023: the maximum `exp - iat` this node will accept on a
    # presented attach token, asserted by the validator regardless of what
    # Keycloak was configured to issue. Without it, "900s max, never
    # lengthened" is a convention an admin voids in one click: a realm
    # accessTokenLifespan bump or a client-lifespan override silently
    # widens every token minted thereafter and nothing on the validating
    # side notices. Under the OMN-15952 re-grant loop that widening is
    # re-minted roughly every 15 minutes per runtime, turning a one-time
    # misconfiguration into continuous exposure.
    #
    # Three distinct bounds now live on this config and none of them is a
    # synonym for another:
    #   max_attach_token_lifetime_seconds (900s, here) -- the shape of a
    #     presented TOKEN, asserted in the validator.
    #   max_unverified_session_seconds (900s) -- how long a SESSION may go
    #     un-revalidated, enforced in the heartbeat handler (OMN-16022).
    #   max_session_ttl_seconds (3600s) -- how long a SESSION may live at
    #     all, applied at attach.
    # The first two share a value because both derive from the same fact
    # (the ga-* client's 900s token lifetime); they are separate fields
    # because they bound different things and are independently tunable.
    max_attach_token_lifetime_seconds: int = Field(default=900, gt=0)
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
