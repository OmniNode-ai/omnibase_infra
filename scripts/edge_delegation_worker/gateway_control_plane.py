# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Outbound-only HTTP client for the gateway attach/heartbeat/detach surface.

Calls exactly the three endpoints ``omninode_infra/docker/onex-api/routers/
gateway.py`` already exposes at ``/v1/gateway/{attach,heartbeat,detach}``.
Every call in this module is a plain outbound POST this process initiates --
nothing here listens on a socket or accepts an inbound connection, matching
the "cloud never connects inward" constraint the connectivity design is
built on.

Every function takes the API base URL as an explicit parameter. Nothing in
this module reads ``ONEX_API_BASE_URL`` or any other environment variable --
that variable name is a known localhost trap (memory
``project_beta_outside_submit_credential_gap``) and silently defaulting to
it here would make the worker attach to the wrong tenant's control plane
without any operator signal.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

import httpx

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraProtocolError,
    InfraUnavailableError,
    ModelInfraErrorContext,
)
from scripts.edge_delegation_worker.models import (
    ModelGatewayHeartbeatResult,
    ModelGatewayRenewalDirective,
    ModelGatewaySession,
    ModelWorkerCredential,
)

_ATTACH_PATH = "/v1/gateway/attach"
_HEARTBEAT_PATH = "/v1/gateway/heartbeat"
_DETACH_PATH = "/v1/gateway/detach"


def _context(*, operation: str, target_name: str) -> ModelInfraErrorContext:
    return ModelInfraErrorContext(
        transport_type=EnumInfraTransportType.HTTP,
        operation=operation,
        target_name=target_name,
    )


async def resolve_access_token(
    credential: ModelWorkerCredential,
    *,
    http_client: httpx.AsyncClient,
) -> str:
    """Resolve a bearer access token from the loaded credential.

    ``bearer_token`` mode returns the pre-issued token verbatim (no network
    call). ``client_credentials`` mode performs an OAuth2 client_credentials
    grant against the credential's ``token_endpoint``.
    """
    if credential.auth_mode == "bearer_token":
        assert credential.bearer_token is not None  # enforced by the model validator
        return credential.bearer_token

    assert credential.client_id is not None
    assert credential.client_secret is not None
    assert credential.token_endpoint is not None

    form: dict[str, str] = {
        "grant_type": "client_credentials",
        "client_id": credential.client_id,
        "client_secret": credential.client_secret,
    }
    if credential.scope:
        form["scope"] = credential.scope

    try:
        response = await http_client.post(credential.token_endpoint, data=form)
    except httpx.TimeoutException as exc:
        raise InfraUnavailableError(
            "token endpoint request timed out",
            context=_context(
                operation="resolve_access_token", target_name=credential.token_endpoint
            ),
        ) from exc
    except httpx.HTTPError as exc:
        raise InfraUnavailableError(
            f"token endpoint request failed: {exc}",
            context=_context(
                operation="resolve_access_token", target_name=credential.token_endpoint
            ),
        ) from exc

    if response.status_code != httpx.codes.OK:
        raise InfraAuthenticationError(
            f"token endpoint returned HTTP {response.status_code}",
            context=_context(
                operation="resolve_access_token", target_name=credential.token_endpoint
            ),
        )

    try:
        body = response.json()
    except ValueError as exc:
        raise InfraProtocolError(
            "token endpoint response was not valid JSON",
            context=_context(
                operation="resolve_access_token", target_name=credential.token_endpoint
            ),
        ) from exc

    access_token = body.get("access_token") if isinstance(body, dict) else None
    if not isinstance(access_token, str) or not access_token:
        raise InfraProtocolError(
            "token endpoint response did not include a string access_token",
            context=_context(
                operation="resolve_access_token", target_name=credential.token_endpoint
            ),
        )
    return access_token


def _auth_headers(access_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {access_token}"}


def _parse_renewal(
    payload: dict[str, object] | None,
) -> ModelGatewayRenewalDirective | None:
    if not payload:
        return None
    return ModelGatewayRenewalDirective(
        mode=str(payload["mode"]),
        renew_not_before=_parse_datetime(payload["renew_not_before"]),
        renew_at=_parse_datetime(payload["renew_at"]),
        session_expires_at=_parse_datetime(payload["session_expires_at"]),
    )


def _parse_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise ValueError(f"expected an ISO-8601 datetime string, got {type(value)!r}")


async def attach(
    api_base: str,
    *,
    access_token: str,
    edge_instance_id: str,
    http_client: httpx.AsyncClient,
) -> ModelGatewaySession:
    """Attach this edge to a tenant-bound gateway session.

    Fails closed: a timeout, a non-200 response, or a response body that
    does not carry the fields this worker needs (``session_id``,
    ``expires_at``, ``heartbeat_interval_seconds``) all raise -- there is no
    path that fabricates a usable session from an ambiguous response.
    """
    url = api_base.rstrip("/") + _ATTACH_PATH
    try:
        response = await http_client.post(
            url,
            json={"edge_instance_id": edge_instance_id},
            headers=_auth_headers(access_token),
        )
    except httpx.TimeoutException as exc:
        raise InfraUnavailableError(
            "gateway attach request timed out",
            context=_context(operation="gateway_attach", target_name=url),
        ) from exc
    except httpx.HTTPError as exc:
        raise InfraUnavailableError(
            f"gateway attach request failed: {exc}",
            context=_context(operation="gateway_attach", target_name=url),
        ) from exc

    if response.status_code != httpx.codes.OK:
        raise InfraAuthenticationError(
            f"gateway attach returned HTTP {response.status_code}: {response.text[:500]}",
            context=_context(operation="gateway_attach", target_name=url),
        )

    try:
        body = response.json()
        session = body.get("session", body)
        return ModelGatewaySession(
            session_id=UUID(str(session["session_id"])),
            expires_at=_parse_datetime(session["expires_at"]),
            heartbeat_interval_seconds=int(session["heartbeat_interval_seconds"]),
            renewal=_parse_renewal(body.get("renewal")),
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise InfraProtocolError(
            "gateway attach response did not match the expected session shape",
            context=_context(operation="gateway_attach", target_name=url),
        ) from exc


async def heartbeat(
    api_base: str,
    *,
    access_token: str,
    session_id: UUID,
    http_client: httpx.AsyncClient,
) -> ModelGatewayHeartbeatResult:
    """Re-validate liveness/non-revocation for an attached session."""
    url = api_base.rstrip("/") + _HEARTBEAT_PATH
    try:
        response = await http_client.post(
            url,
            json={"session_id": str(session_id)},
            headers=_auth_headers(access_token),
        )
    except httpx.TimeoutException as exc:
        raise InfraUnavailableError(
            "gateway heartbeat request timed out",
            context=_context(operation="gateway_heartbeat", target_name=url),
        ) from exc
    except httpx.HTTPError as exc:
        raise InfraUnavailableError(
            f"gateway heartbeat request failed: {exc}",
            context=_context(operation="gateway_heartbeat", target_name=url),
        ) from exc

    if response.status_code != httpx.codes.OK:
        raise InfraAuthenticationError(
            f"gateway heartbeat returned HTTP {response.status_code}: {response.text[:500]}",
            context=_context(operation="gateway_heartbeat", target_name=url),
        )

    try:
        body = response.json()
        return ModelGatewayHeartbeatResult(
            session_id=UUID(str(body["session_id"])),
            termination_reason=body.get("termination_reason"),
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise InfraProtocolError(
            "gateway heartbeat response did not match the expected shape",
            context=_context(operation="gateway_heartbeat", target_name=url),
        ) from exc


async def detach(
    api_base: str,
    *,
    access_token: str,
    session_id: UUID,
    reason: str,
    http_client: httpx.AsyncClient,
) -> None:
    """Explicit edge-initiated teardown. Best-effort but not silent on failure."""
    url = api_base.rstrip("/") + _DETACH_PATH
    try:
        response = await http_client.post(
            url,
            json={"session_id": str(session_id), "reason": reason},
            headers=_auth_headers(access_token),
        )
    except httpx.TimeoutException as exc:
        raise InfraUnavailableError(
            "gateway detach request timed out",
            context=_context(operation="gateway_detach", target_name=url),
        ) from exc
    except httpx.HTTPError as exc:
        raise InfraUnavailableError(
            f"gateway detach request failed: {exc}",
            context=_context(operation="gateway_detach", target_name=url),
        ) from exc

    if response.status_code != httpx.codes.OK:
        raise InfraAuthenticationError(
            f"gateway detach returned HTTP {response.status_code}: {response.text[:500]}",
            context=_context(operation="gateway_detach", target_name=url),
        )
