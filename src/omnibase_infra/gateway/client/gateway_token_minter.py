# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""GatewayTokenMinter — credential -> gateway JWT (OMN-15922, OMN-16687).

Two hops, not one: a ``client_credentials`` grant against the tenant realm's
token endpoint, then a server-side exchange of the resulting machine token for
an attach-audience token. One audience assertion per hop, one in-memory cache.
Deliberately no ``refresh_token`` path: RFC 6749 §4.4.3 says the
client-credentials grant SHOULD NOT issue one, and Keycloak does not -- so
"refresh" here means re-mint, which is also exactly what the gateway's
RE_ATTACH renewal mode assumes its clients do.

WHY TWO HOPS (OMN-16687 -- the single-hop shape could never go green)
    The P0B provisioner mints TWO clients per tenant, and the split is
    deliberate, not a missing mapper (``keycloak_client_manager.py``):

    * the machine/broker client, ``clientId == principal_id``, carrying
      ``aud=redpanda-events`` only. This is the one a tenant can hold: its
      secret is returned on create and re-obtainable via
      ``POST /v1/tenants/{id}/credentials/rotate``. It is what
      ``onex auth login`` stores.
    * the attach client, ``ga-*``, carrying ``aud=gateway-attach`` only. Its
      secret is never returned, logged, cached, or persisted by onex-api --
      ``_reconcile_protocol_mappers`` fail-closed asserts the broker client
      does NOT also carry ``gateway-attach``.

    So no credential a client can hold grants an attach-audience token
    directly. A minter that granted once against Keycloak and asserted
    ``{"gateway-attach"}`` was asserting something no real tenant credential
    could ever satisfy -- a fail-fast that can only ever fail. The attach
    audience is obtained the only way it is obtainable: by presenting the
    machine token to ``POST /v1/auth/gateway-token``, which mints the attach
    token server-side, bound to the tenant the presented token verifiably
    belongs to. The ``ga-*`` secret stays on the server, which is the boundary
    the two-client split exists to draw.

WHAT THE AUDIENCE CHECKS ARE, AND ARE NOT
    They are NOT security controls. This client verifies no signature and
    holds no JWKS; the gateway does that (``service_keycloak_token_validator``
    on the node, ``gateway_auth.py`` at the edge). A forged token would sail
    past these checks and die at the gateway, which is the correct place for
    it to die.

    They ARE fail-fasts that name the wrong-credential case at the hop that
    can still explain it. Presenting a ``ga-*`` credential (or a user-session
    token) to the exchange earns a bare 401 from a remote service; asserting
    the input audience here says which audience was held and which was needed.

    Both comparisons MIRROR their server-side counterpart rather than being
    merely compatible with it:

    * the exchange INPUT rule is ``gateway_auth.validate_exchange_input_
      claims`` -- ``aud`` minus the role-resolved audiences must EQUAL
      ``{"redpanda-events"}``, and an ``aud`` already carrying
      ``gateway-attach`` is refused outright (the exchange does not consume
      its own output).
    * the attach rule is exact SET equality with ``{"gateway-attach"}``. A
      superset is rejected at the gateway, so it is rejected here -- a client
      that accepted more than the gateway does would report success and then
      fail on the wire.

CACHING
    In memory, per instance, re-minted once ``now`` reaches ``exp - skew``.
    The cached value is the ATTACH token; the machine token is never cached,
    since it exists only to be spent on the exchange within the same call.
    Not written to disk: a cached bearer on disk is a credential at rest with
    none of the protections the actual credential file gets, in exchange for
    saving one sub-second mint per process.
"""

from __future__ import annotations

import base64
import binascii
import json
from datetime import UTC, datetime, timedelta
from typing import Final

from pydantic import SecretStr

from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_infra.gateway.models.model_gateway_access_token import (
    ModelGatewayAccessToken,
)
from omnibase_infra.gateway.models.model_gateway_credential import (
    ModelGatewayCredential,
)
from omnibase_infra.protocols.protocol_gateway_transport import (
    ProtocolGatewayTransport,
)

__all__ = [
    "GATEWAY_ATTACH_AUDIENCES",
    "GATEWAY_TOKEN_EXCHANGE_PATH",
    "MACHINE_CREDENTIAL_AUDIENCES",
    "ROLE_RESOLVED_AUDIENCES",
    "GatewayTokenMinter",
]

# Exact audience set the gateway requires, mirrored from
# node_gateway_attach_effect/contract.yaml (required_audience: gateway-attach)
# and onex-api gateway_auth.GATEWAY_EXPECTED_AUDIENCES. A contract term, not a
# deployment knob -- which is why it is a constant here rather than config.
GATEWAY_ATTACH_AUDIENCES: Final[frozenset[str]] = frozenset({"gateway-attach"})

# Exact audience set the EXCHANGE requires of its input, mirrored from
# gateway_auth.EXCHANGE_INPUT_EXPECTED_AUDIENCES (== {BROKER_TOKEN_AUDIENCE})
# and stamped by keycloak_client_manager on the per-tenant machine client.
# Same contract-term reasoning as above: it is what the server compares
# against, so it is a constant, not configuration a caller could widen.
MACHINE_CREDENTIAL_AUDIENCES: Final[frozenset[str]] = frozenset({"redpanda-events"})

# Audiences Keycloak adds on its own from realm role resolution rather than
# from any audience mapper we declare. Discounted before the input comparison
# because the server discounts them (gateway_auth.KEYCLOAK_ROLE_RESOLVED_
# AUDIENCES); not discounting them would reject every real token.
ROLE_RESOLVED_AUDIENCES: Final[frozenset[str]] = frozenset({"account"})

# onex-api's attach-token exchange, mirrored from main.GATEWAY_TOKEN_EXCHANGE_
# PATH. A path on the gateway origin the credential already names -- the same
# way GatewaySessionKeeper holds _ATTACH_PATH / _DETACH_PATH -- so there is no
# second endpoint for an operator to configure, or to get wrong.
GATEWAY_TOKEN_EXCHANGE_PATH: Final[str] = "/v1/auth/gateway-token"

# Re-grant this far ahead of exp. Sized to cover clock skew between this
# machine and Keycloak plus one in-flight request: a token that is valid when
# the request leaves and expired when the gateway validates it is the classic
# expiry-boundary defect, and it presents as an intermittent 401.
_DEFAULT_SKEW_SECONDS: Final[int] = 30


class GatewayTokenMinter:
    """Mints and caches gateway access tokens for one credential."""

    def __init__(
        self,
        *,
        transport: ProtocolGatewayTransport,
        credential: ModelGatewayCredential,
        skew_seconds: int = _DEFAULT_SKEW_SECONDS,
    ) -> None:
        self._transport = transport
        self._credential = credential
        self._skew = timedelta(seconds=skew_seconds)
        self._cached: ModelGatewayAccessToken | None = None

    async def token_for(
        self, *, now: datetime, force_refresh: bool = False
    ) -> ModelGatewayAccessToken:
        """Return a token valid at ``now``, re-granting if inside the skew window.

        Args:
            now: Caller-supplied instant. Injected rather than read from the
                clock so the boundary cases are driven directly by tests.
            force_refresh: Skip the cache and grant unconditionally. Used by the
                attach path, where a still-valid-but-short token is not good
                enough: the gateway stamps a session's ``expires_at`` from
                ``min(token exp, max_session_ttl_seconds)``, so attaching with a
                token that has two minutes left buys a two-minute session. That
                is exactly the state the renewal cycle is trying to leave, so
                renewing into it would loop.

        Returns:
            A token whose ``expires_at`` is at least ``skew`` beyond ``now``.

        Raises:
            ModelOnexError: If the grant fails, the response is malformed, or
                the audience is not exactly the gateway-attach set.
        """
        cached = self._cached
        if (
            not force_refresh
            and cached is not None
            and now < cached.expires_at - self._skew
        ):
            return cached
        minted = await self._grant(now=now)
        self._cached = minted
        return minted

    async def _grant(self, *, now: datetime) -> ModelGatewayAccessToken:
        """Grant a machine token, then exchange it for an attach token."""
        machine_token = await self._grant_machine_token()
        return await self._exchange_for_attach_token(machine_token, now=now)

    async def _grant_machine_token(self) -> str:
        """``client_credentials`` against the realm; returns the raw JWT.

        The token is returned as a bare string rather than a
        ``ModelGatewayAccessToken`` on purpose: it is not a gateway access
        token, it cannot attach, and it is spent on the exchange inside the
        same call. Giving it the same type as the thing it is exchanged for
        is how one ends up presented to the gateway by mistake.
        """
        response = await self._transport.post_form(
            self._credential.token_endpoint,
            form={
                "grant_type": "client_credentials",
                "client_id": self._credential.client_id,
                "client_secret": self._credential.client_secret.get_secret_value(),
            },
            headers={"Accept": "application/json"},
        )
        if response.status != 200:
            # The body is NOT echoed: an OAuth2 error response is attacker- and
            # proxy-influenced, and an error path that pastes a remote body into
            # a local message is how secrets and tokens reach logs.
            raise ModelOnexError(
                "gateway token grant rejected by "
                f"{self._credential.token_endpoint} (HTTP {response.status}) for "
                f"client_id '{self._credential.client_id}'. Check the credential "
                "with 'onex auth status', or re-run 'onex auth login' with a "
                "freshly rotated secret.",
                error_code=EnumCoreErrorCode.AUTHENTICATION_ERROR,
            )

        payload = self._decode_json_object(
            await response.text(), source=self._credential.token_endpoint
        )
        machine_token = self._require_string(payload, "access_token")
        self._assert_exchange_input_audience(self._audiences_of(machine_token))
        return machine_token

    async def _exchange_for_attach_token(
        self, machine_token: str, *, now: datetime
    ) -> ModelGatewayAccessToken:
        """Trade the machine token for the tenant's attach token.

        The exchange binds the minted token to the tenant the PRESENTED token
        verifiably belongs to (``azp == derive_principal_id(tenant_id)``), so
        nothing in this request names a tenant -- there is no body at all. A
        tenant selector here would be a selector the server would have to
        refuse anyway.
        """
        url = self._credential.base_url.rstrip("/") + GATEWAY_TOKEN_EXCHANGE_PATH
        response = await self._transport.post_json(
            url,
            body="{}",
            headers={
                "Authorization": f"Bearer {machine_token}",
                "Accept": "application/json",
            },
        )
        if response.status != 200:
            raise ModelOnexError(
                f"gateway attach-token exchange rejected by {url} (HTTP "
                f"{response.status}) for principal "
                f"'{self._credential.client_id}'. A 401 means the machine "
                "credential was refused or is not bound to its own tenant; a "
                "503 means the exchange is not enabled on this deployment. "
                "Check the credential with 'onex auth status'.",
                error_code=EnumCoreErrorCode.AUTHENTICATION_ERROR,
            )

        payload = self._decode_json_object(await response.text(), source=url)
        access_token = self._require_string(payload, "access_token")
        expires_in = self._require_int(payload, "expires_in")
        audiences = self._audiences_of(access_token)

        if audiences != GATEWAY_ATTACH_AUDIENCES:
            raise ModelOnexError(
                "the attach-token exchange returned a token carrying audience "
                f"{sorted(audiences)} but the gateway requires exactly "
                f"{sorted(GATEWAY_ATTACH_AUDIENCES)} (set equality, not "
                f"membership). The exchange at {url} is minting against the "
                "wrong Keycloak client, or that client's audience mapper has "
                "drifted -- this is a server-side defect, not a bad credential.",
                error_code=EnumCoreErrorCode.AUTHENTICATION_ERROR,
            )

        return ModelGatewayAccessToken(
            access_token=SecretStr(access_token),
            expires_at=now + timedelta(seconds=expires_in),
            audiences=audiences,
        )

    def _assert_exchange_input_audience(self, audiences: frozenset[str]) -> None:
        """Mirror ``gateway_auth.validate_exchange_input_claims``'s aud rule.

        Both branches name the fix, because the two wrong credentials fail for
        opposite reasons: an attach-audience token is the ``ga-*`` secret that
        was never meant to leave the server, and anything else is simply not
        the tenant's machine credential.
        """
        if GATEWAY_ATTACH_AUDIENCES & audiences:
            raise ModelOnexError(
                f"the credential for '{self._credential.client_id}' grants a "
                f"token carrying {sorted(GATEWAY_ATTACH_AUDIENCES)}, which the "
                "attach-token exchange refuses as input -- it does not consume "
                "its own output. Store the tenant's machine credential (the "
                "principal_id client, from 'POST /v1/tenants/{id}/credentials/"
                "rotate') instead.",
                error_code=EnumCoreErrorCode.AUTHENTICATION_ERROR,
            )
        effective = audiences - ROLE_RESOLVED_AUDIENCES
        if effective != MACHINE_CREDENTIAL_AUDIENCES:
            raise ModelOnexError(
                "the granted machine token carries audience "
                f"{sorted(effective)} but the attach-token exchange requires "
                f"exactly {sorted(MACHINE_CREDENTIAL_AUDIENCES)} (set equality "
                f"after discounting {sorted(ROLE_RESOLVED_AUDIENCES)}). The "
                f"Keycloak client '{self._credential.client_id}' is not the "
                "tenant's provisioned machine client.",
                error_code=EnumCoreErrorCode.AUTHENTICATION_ERROR,
            )

    # -- parsing -----------------------------------------------------------

    def _audiences_of(self, access_token: str) -> frozenset[str]:
        """Normalise the ``aud`` claim to a set, mirroring the gateway.

        RFC 7519 §4.1.3 permits ``aud`` to be a single case-sensitive string OR
        an array of them, and Keycloak switches to the array form the moment a
        client carries more than one audience mapper. Multiplicity and order
        must therefore not be observable -- ``aud`` is a set.
        """
        segments = access_token.split(".")
        if len(segments) != 3:
            raise ModelOnexError(
                "the token endpoint returned an access_token that is not a JWT "
                f"(expected 3 dot-separated segments, found {len(segments)}).",
                error_code=EnumCoreErrorCode.PARSING_ERROR,
            )
        padding = "=" * (-len(segments[1]) % 4)
        try:
            claims_bytes = base64.urlsafe_b64decode(segments[1] + padding)
        except (binascii.Error, ValueError) as exc:
            raise ModelOnexError(
                "the access_token's claims segment is not valid base64url.",
                error_code=EnumCoreErrorCode.PARSING_ERROR,
            ) from exc

        claims = self._decode_json_object(
            claims_bytes.decode("utf-8", errors="replace"),
            source="access_token claims",
        )
        raw = claims.get("aud")
        if isinstance(raw, str):
            return frozenset({raw})
        if isinstance(raw, list):
            if not all(isinstance(entry, str) for entry in raw):
                raise ModelOnexError(
                    "the access_token's 'aud' claim is a list containing "
                    "non-string entries.",
                    error_code=EnumCoreErrorCode.PARSING_ERROR,
                )
            return frozenset(str(entry) for entry in raw)
        raise ModelOnexError(
            "the access_token carries no usable 'aud' claim; every hop of the "
            "mint asserts one, so an audience-less token cannot be classified "
            "as either a machine credential or an attach token.",
            error_code=EnumCoreErrorCode.AUTHENTICATION_ERROR,
        )

    def _decode_json_object(self, raw: str, *, source: str) -> dict[str, object]:
        try:
            document = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ModelOnexError(
                f"{source} returned a body that is not valid JSON.",
                error_code=EnumCoreErrorCode.PARSING_ERROR,
            ) from exc
        if not isinstance(document, dict):
            raise ModelOnexError(
                f"{source} returned {type(document).__name__}, expected a JSON object.",
                error_code=EnumCoreErrorCode.PARSING_ERROR,
            )
        return {str(key): value for key, value in document.items()}

    def _require_string(self, payload: dict[str, object], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise ModelOnexError(
                f"the token endpoint response has no usable '{key}' field.",
                error_code=EnumCoreErrorCode.PARSING_ERROR,
            )
        return value

    def _require_int(self, payload: dict[str, object], key: str) -> int:
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ModelOnexError(
                f"the token endpoint response has no usable '{key}' field.",
                error_code=EnumCoreErrorCode.PARSING_ERROR,
            )
        return value

    @staticmethod
    def utc_now() -> datetime:
        """The clock, in one named place, so callers do not each pick one."""
        return datetime.now(UTC)
