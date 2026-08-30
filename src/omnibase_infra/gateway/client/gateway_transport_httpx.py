# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The one component in the ``onex auth`` slice that opens a socket (OMN-15922).

EFFECT layer: this is external I/O and nothing else. Every decision the gateway
client makes -- whether a status is acceptable, whether an audience matches,
whether a renewal window has opened -- lives above this line in the services,
which is what lets the whole cycle be driven in tests by an in-memory fake with
no network and no sleeping. The rule that keeps that true is the one below.

WHAT THIS DOES NOT DO: CLASSIFY
    Neither method raises on a non-2xx. That is the ``ProtocolGatewayTransport``
    contract, and it is load-bearing rather than lax: a 401 from Keycloak and a
    401 from the gateway need *different* operator-facing messages naming
    different remediations, and only the caller knows which call it made. An
    adapter that raised here would flatten both into one transport error and
    throw away the status the caller needs to say anything useful. So a reached
    server always comes back as a response.

    A server that was NOT reached is the opposite case and does raise: there is
    no status to hand back, and returning a synthetic one (a fake 503) would be
    indistinguishable from a real server that answered 503 -- which is the
    difference between "your gateway is refusing you" and "your gateway is not
    there". ``InfraUnavailableError`` keeps them apart.

SECRET DISCIPLINE
    ``post_form`` carries ``client_secret`` and ``post_json`` carries a Bearer.
    Nothing here logs a request, a body, a header, or an exception payload, and
    the raised errors name only the URL's operation -- never the form, never the
    headers. httpx's own exception reprs do not carry request bodies, but the
    error path deliberately does not interpolate ``exc`` for that reason.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import httpx

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import InfraUnavailableError, ModelInfraErrorContext

__all__ = ["GatewayTransportHttpx", "GatewayHttpResponse"]

# One grant or one attach against a control-plane endpoint. Long enough to
# absorb a cold Keycloak realm, short enough that an unattended runtime's
# renewal cycle still fits inside the margin the directive declares.
_DEFAULT_TIMEOUT_SECONDS: float = 10.0


class GatewayHttpResponse:
    """A reached server's answer, satisfying ``ProtocolHttpResponse``.

    Holds the body as already-read text rather than a live stream: the caller
    reads it at most once, and a streamed body would outlive the
    ``httpx.AsyncClient`` context that produced it.
    """

    def __init__(self, status: int, body: str) -> None:
        self._status = status
        self._body = body

    @property
    def status(self) -> int:
        return self._status

    async def text(self) -> str:
        return self._body

    async def json(self) -> object:
        """Present for protocol conformance; the gateway client never calls it.

        The client parses through typed Pydantic models instead, so that the
        wire shape has exactly one authority. Kept because
        ``ProtocolHttpResponse`` declares it and a partial implementation would
        fail the structural check for the next caller, not this one.
        """
        return json.loads(self._body)


class GatewayTransportHttpx:
    """``ProtocolGatewayTransport`` over ``httpx.AsyncClient``."""

    def __init__(self, *, timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS) -> None:
        self._timeout = timeout_seconds

    async def post_form(
        self,
        url: str,
        *,
        form: Mapping[str, str],
        headers: Mapping[str, str],
    ) -> GatewayHttpResponse:
        """POST ``form`` url-encoded. Carries ``client_secret`` -- never logged."""
        return await self._post(
            url,
            operation="gateway_token_grant",
            data=dict(form),
            content=None,
            headers=dict(headers),
        )

    async def post_json(
        self,
        url: str,
        *,
        body: str,
        headers: Mapping[str, str],
    ) -> GatewayHttpResponse:
        """POST an already-serialized JSON body. Carries a Bearer -- never logged."""
        return await self._post(
            url,
            operation="gateway_request",
            data=None,
            content=body.encode("utf-8"),
            headers=dict(headers),
        )

    async def get(
        self,
        url: str,
        timeout: float | None = None,
        headers: dict[str, str] | None = None,
    ) -> GatewayHttpResponse:
        """GET ``url``. Satisfies ``ProtocolHttpClient`` (OMN-17205).

        Added as a method on the ADAPTER, never to ``ProtocolGatewayTransport``:
        that protocol is ``runtime_checkable`` and is satisfied structurally by
        every existing in-memory fake, so widening it would silently
        un-satisfy all of them. Widening a concrete implementation cannot.

        Carries a Bearer in ``headers`` -- like every other method here, the
        headers are never logged and never interpolated into an error.
        """
        try:
            client = httpx.AsyncClient(  # no-contract-check: the seam
                timeout=timeout if timeout is not None else self._timeout
            )
            async with client:
                response = await client.get(url, headers=dict(headers or {}))
        except httpx.HTTPError as exc:
            raise InfraUnavailableError(
                f"gateway transport could not reach {url}",
                context=ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.HTTP,
                    operation="cloud_ledger_read",
                ),
            ) from exc

        return GatewayHttpResponse(response.status_code, response.text)

    async def _post(
        self,
        url: str,
        *,
        operation: str,
        data: dict[str, str] | None,
        content: bytes | None,
        headers: dict[str, str],
    ) -> GatewayHttpResponse:
        try:
            # The freestanding-IO guard exists to catch imperative IO that
            # BYPASSES a transport contract. Here the raw call is what BACKS
            # one: this module is the sole ProtocolGatewayTransport
            # implementation, and confining the socket to this single line is
            # precisely what keeps the credential store, token minter, renewal
            # planner and session keeper transport-free and driveable by an
            # in-memory fake. An outbound OAuth2 client_credentials grant plus
            # gateway attach from a CLI has no bus-mediated transport to route
            # through -- it is a client calling out, not a node emitting.
            timeout = self._timeout
            client = httpx.AsyncClient(timeout=timeout)  # no-contract-check: the seam
            async with client:
                response = await client.post(
                    url,
                    data=data,
                    content=content,
                    headers=headers,
                )
        except httpx.HTTPError as exc:
            # The URL, not the payload: the form carries a client secret and the
            # headers carry a bearer, so neither is interpolated here.
            raise InfraUnavailableError(
                f"gateway transport could not reach {url}",
                context=ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.HTTP,
                    operation=operation,
                ),
            ) from exc

        return GatewayHttpResponse(response.status_code, response.text)
