# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read the cloud ledger projection by correlation id, from anywhere (OMN-17205).

WHY THIS EXISTS
    ``beta/GOAL.md`` row 0b's probe terminates in a read of the CLOUD hook
    ledger by correlation id. On 2026-08-30 the operator could not run it from
    their own machine at all: the staging Kubernetes API is unreachable from
    there (``dial tcp ...:6443: i/o timeout``) and the deployed onex-api served
    no projection route. A ledger the person whose work it records cannot read
    is not a ledger, and a goal row whose probe cannot be executed cannot fail,
    so it cannot catch a drop.

    This is the client half. The server half is
    ``GET /v1/projections/hook-events/by-correlation`` on onex-api.

WHAT IT IS NOT
    Not a Kubernetes client, and not a Kafka consumer. Customers -- and
    operators -- reach the platform over authenticated HTTPS through the
    gateway, never by ``kubectl exec`` into a pod to look at a table. That is
    the same boundary the product sells; a probe that needs cluster credentials
    proves something no customer could ever reproduce.

ADDRESSING AND CREDENTIAL
    Both come from ``~/.onex`` via ``StoreGatewayCredential``: the base URL from
    the non-secret ``gateway:`` config block, the client secret by REFERENCE
    from the 0600 credentials file. There is no host literal in this module (a
    test asserts that), no environment variable, and no way to pass a secret on
    the command line.

WHY THIS MINTS ITS OWN TOKEN INSTEAD OF CALLING GatewayTokenMinter
    ``GatewayTokenMinter`` mints a token for the gateway ATTACH audience: it
    grants a machine token and then exchanges it, asserting on the way that the
    grant carried exactly the broker audience. That assertion is correct for
    attach and wrong here -- this is an ordinary authenticated control-plane
    read, not a session attach -- so reusing it would make a read fail for a
    reason that has nothing to do with reading. The grant below is the same
    RFC 6749 s4.4 ``client_credentials`` request with no exchange and no
    attach-specific audience rule.

SECRET DISCIPLINE
    The secret is read out of ``SecretStr`` on exactly one line, the one that
    puts it on the wire to the token endpoint. The minted token is never
    returned, never logged, and never placed in ``ModelCloudLedgerRead``. No
    remote response body is interpolated into an operator-facing message: an
    error body is proxy- and attacker-influenced, and pasting one into a local
    message is how tokens reach terminals and issue threads.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from typing import Protocol, runtime_checkable
from urllib.parse import urlencode

from omnibase_core.protocols.http.protocol_http_client import ProtocolHttpResponse
from omnibase_core.types import JsonType
from omnibase_infra.enums.enum_cloud_ledger_verdict import EnumCloudLedgerVerdict
from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.gateway.models.model_cloud_ledger_read import (
    ModelCloudLedgerRead,
)
from omnibase_infra.gateway.models.model_gateway_credential import (
    ModelGatewayCredential,
)

__all__ = [
    "CLOUD_LEDGER_CORRELATION_PATH",
    "CloudLedgerReader",
    "ProtocolCloudLedgerTransport",
]

#: The onex-api route this reader calls. It is the exact path the OMN-17202
#: chain probe already declares as ``cloud_projection_path`` in its node
#: contract, so the two lanes address one surface rather than two.
CLOUD_LEDGER_CORRELATION_PATH = "/v1/projections/hook-events/by-correlation"

#: Bounded by the server too; sent so the intent is explicit on the wire.
_DEFAULT_LIMIT = 10


@runtime_checkable
class ProtocolCloudLedgerTransport(Protocol):
    """The two HTTP shapes this read needs: an OAuth2 grant and a GET.

    Declared here rather than widening ``ProtocolGatewayTransport``: that
    protocol is ``runtime_checkable`` and is satisfied structurally by every
    existing fake, so adding a method to it would silently un-satisfy all of
    them -- a breaking change to a live seam in service of a new caller.
    """

    async def post_form(
        self,
        url: str,
        *,
        form: Mapping[str, str],
        headers: Mapping[str, str],
    ) -> ProtocolHttpResponse: ...

    async def get(
        self,
        url: str,
        timeout: float | None = None,
        headers: dict[str, str] | None = None,
    ) -> ProtocolHttpResponse: ...


class TokenRefusedError(Exception):
    """The control plane would not issue a token for this credential."""


class CloudLedgerReader:
    """One authenticated correlation-id read against the cloud ledger."""

    def __init__(
        self,
        *,
        transport: ProtocolCloudLedgerTransport,
        credential: ModelGatewayCredential,
    ) -> None:
        self._transport = transport
        self._credential = credential

    # -- public --------------------------------------------------------------

    async def read(
        self,
        *,
        correlation_id: str,
        now: datetime,
        limit: int = _DEFAULT_LIMIT,
        include_payload: bool = False,
    ) -> ModelCloudLedgerRead:
        """Ask the cloud ledger about ``correlation_id`` and classify the answer.

        Never raises for a reachable-but-unhappy control plane: every outcome
        an operator can act on is a verdict, because a probe that raises on
        "no row" is a probe whose failure mode is indistinguishable from a bug
        in the probe.

        Args:
            correlation_id: The id stamped on the emitted hook event.
            now: Caller-supplied instant, injected rather than read from the
                clock so the token-freshness boundary is drivable by tests.
            limit: Maximum rows to request.
            include_payload: Ask the server for the verbatim event bodies. Off
                by default -- the payload is a raw hook body and the default
                read must be safe to paste.

        Returns:
            A ``ModelCloudLedgerRead`` whose ``verdict`` names the outcome and
            whose ``exit_code`` is 0 only when a row came back.
        """
        url = self._url(
            correlation_id=correlation_id,
            limit=limit,
            include_payload=include_payload,
        )

        try:
            token = await self._grant_control_plane_token()
        except TokenRefusedError as exc:
            return ModelCloudLedgerRead(
                verdict=EnumCloudLedgerVerdict.UNAUTHENTICATED,
                correlation_id=correlation_id,
                url=url,
                detail=str(exc),
            )
        except InfraUnavailableError:
            return ModelCloudLedgerRead(
                verdict=EnumCloudLedgerVerdict.UNAVAILABLE,
                correlation_id=correlation_id,
                url=url,
                detail=(
                    "could not reach the token endpoint "
                    f"{self._credential.token_endpoint}"
                ),
            )

        try:
            response = await self._transport.get(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                },
            )
        except InfraUnavailableError:
            return ModelCloudLedgerRead(
                verdict=EnumCloudLedgerVerdict.UNAVAILABLE,
                correlation_id=correlation_id,
                url=url,
                detail=f"could not reach {self._credential.base_url}",
            )

        return await self._classify(
            response, correlation_id=correlation_id, url=url, now=now
        )

    # -- internals -----------------------------------------------------------

    def _url(self, *, correlation_id: str, limit: int, include_payload: bool) -> str:
        query: dict[str, str] = {
            "correlation_id": correlation_id,
            "limit": str(limit),
        }
        if include_payload:
            query["include_payload"] = "true"
        base = self._credential.base_url.rstrip("/")
        return f"{base}{CLOUD_LEDGER_CORRELATION_PATH}?{urlencode(query)}"

    async def _grant_control_plane_token(self) -> str:
        """RFC 6749 s4.4 ``client_credentials`` against the stored realm."""
        response = await self._transport.post_form(
            self._credential.token_endpoint,
            form={
                "grant_type": "client_credentials",
                "client_id": self._credential.client_id,
                # The one line that reads the secret, and it goes straight out.
                "client_secret": self._credential.client_secret.get_secret_value(),
            },
            headers={"Accept": "application/json"},
        )
        if response.status != 200:
            raise TokenRefusedError(
                f"token endpoint {self._credential.token_endpoint} refused the "
                f"stored credential for client_id "
                f"'{self._credential.client_id}' (HTTP {response.status}). "
                "Check it with 'onex auth status', or re-run 'onex auth login' "
                "with a freshly rotated secret."
            )
        payload = self._decode(await response.text())
        if payload is None:
            raise TokenRefusedError(
                f"token endpoint {self._credential.token_endpoint} returned a "
                "200 that was not a JSON object"
            )
        token = payload.get("access_token")
        if not isinstance(token, str) or not token:
            raise TokenRefusedError(
                f"token endpoint {self._credential.token_endpoint} returned a "
                "200 carrying no access_token"
            )
        return token

    async def _classify(
        self,
        response: ProtocolHttpResponse,
        *,
        correlation_id: str,
        url: str,
        now: datetime,
    ) -> ModelCloudLedgerRead:
        status = response.status

        if status in (401, 403):
            return ModelCloudLedgerRead(
                verdict=EnumCloudLedgerVerdict.UNAUTHENTICATED,
                correlation_id=correlation_id,
                url=url,
                http_status=status,
                detail=(
                    "the control plane refused the presented credential. Note "
                    "that this API also answers 401 for a route it does not "
                    "serve, so a 401 here is 'refused OR absent', never "
                    "'no such row'."
                ),
            )

        if status != 200:
            return ModelCloudLedgerRead(
                verdict=EnumCloudLedgerVerdict.UNAVAILABLE,
                correlation_id=correlation_id,
                url=url,
                http_status=status,
                detail=f"the control plane answered HTTP {status}",
            )

        payload = self._decode(await response.text())
        if payload is None:
            return ModelCloudLedgerRead(
                verdict=EnumCloudLedgerVerdict.UNAVAILABLE,
                correlation_id=correlation_id,
                url=url,
                http_status=status,
                detail="the control plane answered 200 with a non-JSON body",
            )

        data_state = payload.get("data_state")
        verdict = _DATA_STATE_TO_VERDICT.get(str(data_state))
        if verdict is None:
            return ModelCloudLedgerRead(
                verdict=EnumCloudLedgerVerdict.UNAVAILABLE,
                correlation_id=correlation_id,
                url=url,
                http_status=status,
                detail=(
                    "the control plane answered 200 with an unrecognised "
                    f"data_state {data_state!r} -- refusing to guess"
                ),
            )

        raw_rows = payload.get("rows")
        rows: list[dict[str, JsonType]] = (
            [r for r in raw_rows if isinstance(r, dict)]
            if isinstance(raw_rows, list)
            else []
        )
        count = payload.get("count")
        return ModelCloudLedgerRead(
            verdict=verdict,
            correlation_id=str(payload.get("correlation_id") or correlation_id),
            projection=str(payload.get("projection") or ""),
            url=url,
            http_status=status,
            count=count if isinstance(count, int) and count >= 0 else len(rows),
            rows=rows,
            detail=_DETAIL_FOR[verdict].format(as_of=now.isoformat()),
        )

    @staticmethod
    def _decode(raw: str) -> dict[str, JsonType] | None:
        try:
            document = json.loads(raw)
        except (ValueError, TypeError):
            return None
        return document if isinstance(document, dict) else None


_DATA_STATE_TO_VERDICT: dict[str, EnumCloudLedgerVerdict] = {
    "found": EnumCloudLedgerVerdict.FOUND,
    "not_found": EnumCloudLedgerVerdict.NOT_FOUND,
    "projection_absent": EnumCloudLedgerVerdict.PROJECTION_ABSENT,
}

_DETAIL_FOR: dict[EnumCloudLedgerVerdict, str] = {
    EnumCloudLedgerVerdict.FOUND: (
        "the cloud ledger holds this correlation id (read at {as_of})"
    ),
    EnumCloudLedgerVerdict.NOT_FOUND: (
        "the cloud projection exists and holds no row for this correlation id "
        "(read at {as_of}). The chain did not deliver, or has not yet."
    ),
    EnumCloudLedgerVerdict.PROJECTION_ABSENT: (
        "the cloud projection table does not exist on this plane (read at "
        "{as_of}). This is the sink gap, not an empty result."
    ),
}
