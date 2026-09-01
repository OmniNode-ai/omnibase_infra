# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""GatewayIdentityVerifier -- prove the stored key actually authenticates (OMN-17028).

WHY A LOCAL PRINT IS NOT A STATUS
    ``onex auth status`` used to answer "what does my config file say", which is
    a question the operator can answer with ``cat``. The question they are
    actually asking is "am I authenticated", and the only authority on that is
    the gateway. This class asks it: one ``GET /v1/whoami`` presenting the
    stored key, and the tenant that comes back is the server's answer, never
    the local label echoed.

WHY THE API-KEY KIND ONLY
    An ``onxk_`` key is presented directly -- there is nothing to mint, so the
    check is one round trip and cannot fail for a token-endpoint reason that
    has nothing to do with being authenticated. The client-credentials
    (attach-plane) credential is a different question with a different answer
    surface: it is proven by minting, which is what ``onex auth token`` does.
    Verifying it here would either duplicate that mint or silently report
    "verified" from a grant that carries the wrong audience for attach.

SECRET DISCIPLINE
    The key is unwrapped on exactly one line, the one that puts it in the
    header. No response body is interpolated into an operator-facing message:
    an error body is proxy- and attacker-influenced, and pasting one into a
    terminal is how credentials reach issue threads.
"""

from __future__ import annotations

import json
from typing import Final, Protocol, runtime_checkable

from pydantic import ValidationError

from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_core.protocols.http.protocol_http_client import ProtocolHttpResponse
from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.gateway.models.model_gateway_api_key import (
    ModelGatewayApiKeyCredential,
)
from omnibase_infra.gateway.models.model_gateway_identity import ModelGatewayIdentity

__all__ = ["GATEWAY_WHOAMI_PATH", "GatewayIdentityVerifier", "ProtocolWhoamiTransport"]

#: The onex-api route that answers "which tenant is this credential".
GATEWAY_WHOAMI_PATH: Final[str] = "/v1/whoami"


@runtime_checkable
class ProtocolWhoamiTransport(Protocol):
    """The single HTTP shape this check needs.

    Declared here rather than widening ``ProtocolGatewayTransport``: that
    protocol is ``runtime_checkable`` and is satisfied structurally by every
    existing in-memory fake, so adding a method to it would silently
    un-satisfy all of them.
    """

    async def get(
        self,
        url: str,
        timeout: float | None = None,
        headers: dict[str, str] | None = None,
    ) -> ProtocolHttpResponse: ...


class GatewayIdentityVerifier:
    """One authenticated ``whoami`` against the gateway the credential names."""

    def __init__(
        self,
        *,
        transport: ProtocolWhoamiTransport,
        credential: ModelGatewayApiKeyCredential,
    ) -> None:
        self._transport = transport
        self._credential = credential

    @property
    def url(self) -> str:
        return f"{self._credential.base_url.rstrip('/')}{GATEWAY_WHOAMI_PATH}"

    async def verify(self) -> ModelGatewayIdentity:
        """Ask the gateway who this credential is.

        Returns:
            The tenant the gateway resolved from the presented key.

        Raises:
            ModelOnexError: If the gateway refused the key, could not be
                reached, or answered something this check cannot read as an
                identity. Never returns a partially resolved identity: a
                caller that treated "could not tell" as "authenticated" is the
                failure this whole surface exists to prevent.
        """
        try:
            response = await self._transport.get(
                self.url,
                headers={
                    # The one line that reads the key, and it goes straight out.
                    "x-api-key": self._credential.api_key.get_secret_value(),
                    "Accept": "application/json",
                },
            )
        except InfraUnavailableError as exc:
            raise ModelOnexError(
                f"could not reach {self.url} to verify the stored credential. "
                "The credential itself was not judged -- re-run with "
                "'--no-verify' to print the stored coordinates without a "
                "network call.",
                error_code=EnumCoreErrorCode.SERVICE_UNAVAILABLE,
            ) from exc

        if response.status in (401, 403):
            raise ModelOnexError(
                f"{self._credential.base_url} refused the stored API key "
                f"(HTTP {response.status}). The key filed under "
                f"'{self._credential.api_key_ref}' is revoked, mistyped, or "
                "belongs to a different gateway. Mint a new one in the "
                "dashboard and re-run onboarding.",
                error_code=EnumCoreErrorCode.AUTHENTICATION_ERROR,
            )

        if response.status != 200:
            raise ModelOnexError(
                f"{self.url} answered HTTP {response.status}; the stored "
                "credential could not be verified either way.",
                error_code=EnumCoreErrorCode.SERVICE_UNAVAILABLE,
            )

        payload = self._decode(await response.text())
        if payload is None:
            raise ModelOnexError(
                f"{self.url} answered 200 with a body that is not a JSON "
                "object; refusing to report an identity it did not state.",
                error_code=EnumCoreErrorCode.CONFIGURATION_PARSE_ERROR,
            )

        tenant_id = payload.get("tenant_id")
        tenant_slug = payload.get("tenant_slug")
        if not isinstance(tenant_id, str) or not isinstance(tenant_slug, str):
            raise ModelOnexError(
                f"{self.url} answered 200 carrying no tenant identity; "
                "refusing to read that as authenticated.",
                error_code=EnumCoreErrorCode.CONFIGURATION_PARSE_ERROR,
            )

        try:
            return ModelGatewayIdentity(tenant_id=tenant_id, tenant_slug=tenant_slug)
        except ValidationError as exc:
            # A 200 whose tenant_id is not a UUID is a gateway this client does
            # not recognise -- possibly a captive portal or a proxy's own JSON.
            # Refusing is the only safe reading: reporting it would tell the
            # operator they are authenticated somewhere nobody identified.
            raise ModelOnexError(
                f"{self.url} answered 200 with a tenant identity this client "
                "cannot read; refusing to report it as verified.",
                error_code=EnumCoreErrorCode.CONFIGURATION_PARSE_ERROR,
            ) from exc

    @staticmethod
    def _decode(raw: str) -> dict[str, object] | None:
        try:
            document = json.loads(raw)
        except (ValueError, TypeError):
            return None
        return document if isinstance(document, dict) else None
