# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the attach/heartbeat/detach handler trio.

This is the end-to-end slice proof at the unit level (OMN-15753's local
component): attach -> authenticated ACTIVE session -> heartbeat (active) ->
heartbeat (introspection reports revoked) -> session torn down. The
introspection HTTP call and the JWKS fetch live inline in the handlers
(moved out of a freestanding services/ module to satisfy the
imperative-contract-guard's handlers/-only I/O boundary) and are faked via
monkeypatch here.

OMN-15918 hardening coverage added in this revision (RED-before/GREEN-after
for each CodeRabbit-flagged gap):
  - R1 (JWT signature verification): ``_fake_jwt`` from the pre-hardening
    revision (``alg: none``, unsigned) is gone -- every token here is a real
    RS256-signed JWT verified against a mocked JWKS response.
    ``test_attach_rejects_forged_signature`` / ``test_attach_jwks_outage_*``
    are the direct proof.
  - R2 (identity binding): ``test_heartbeat_rejects_identity_mismatch`` /
    ``test_detach_rejects_identity_mismatch`` prove a validly-signed token
    for the WRONG tenant/principal/client is rejected before the store is
    touched.
  - R3 (atomic transitions): ``test_heartbeat_does_not_resurrect_*`` proves
    the read-introspect-write race CodeRabbit flagged is closed via
    ``put_if_present``.
  - R4 (outage vs revocation): ``test_heartbeat_transport_outage_*`` /
    ``test_heartbeat_introspection_non200_*`` prove a Keycloak outage raises
    ``InfraUnavailableError`` and leaves the session untouched, rather than
    tearing it down as if revoked.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from pydantic import SecretStr

from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.nodes.node_gateway_attach_effect.handlers.handler_gateway_attach import (
    HandlerGatewayAttach,
)
from omnibase_infra.nodes.node_gateway_attach_effect.handlers.handler_gateway_detach import (
    HandlerGatewayDetach,
)
from omnibase_infra.nodes.node_gateway_attach_effect.handlers.handler_gateway_detach import (
    SessionNotFoundError as DetachSessionNotFoundError,
)
from omnibase_infra.nodes.node_gateway_attach_effect.handlers.handler_gateway_heartbeat import (
    HandlerGatewayHeartbeat,
)
from omnibase_infra.nodes.node_gateway_attach_effect.handlers.handler_gateway_heartbeat import (
    SessionNotFoundError as HeartbeatSessionNotFoundError,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.enum_gateway_session_event_type import (
    EnumGatewaySessionEventType,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.enum_gateway_session_status import (
    EnumGatewaySessionStatus,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.enum_gateway_session_termination_reason import (
    EnumGatewaySessionTerminationReason,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_attach_config import (
    ModelGatewayAttachConfig,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_attach_request import (
    ModelGatewayAttachRequest,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_detach_request import (
    ModelGatewayDetachRequest,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_heartbeat_request import (
    ModelGatewayHeartbeatRequest,
)
from omnibase_infra.nodes.node_gateway_attach_effect.services.service_gateway_session_policy import (
    SessionExpiredError,
)
from omnibase_infra.nodes.node_gateway_attach_effect.services.service_keycloak_token_validator import (
    TokenValidationError,
)
from omnibase_infra.nodes.node_gateway_attach_effect.services.store_gateway_session_memory import (
    StoreGatewaySessionMemory,
)
from tests.unit.nodes.node_gateway_attach_effect._jwt_test_support import (
    OTHER_KID,
    TENANT_KID,
    generate_key_material,
    jwks_response_body,
    sign_claims,
)

TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")
OTHER_TENANT_ID = UUID("22222222-2222-2222-2222-222222222222")
ISSUER = "https://keycloak.example/realms/omninode"


def _claims(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "iss": ISSUER,
        "sub": "svc-acct-abc",
        "aud": "gateway-attach",
        "tenant_id": str(TENANT_ID),
        "tenant_slug": "acme",
        "principal_id": "t-11111111111111111111111111111111",
        "azp": "gw-tenant-acme",
        "exp": 9999999999,
    }
    base.update(overrides)
    return base


class _FakeSecretResolver:
    def __init__(self, values: dict[str, str]) -> None:
        self._values = values

    async def get_secret_async(
        self, logical_name: str, required: bool = True, correlation_id: object = None
    ) -> SecretStr:
        return SecretStr(self._values[logical_name])


class _FakeResponse:
    def __init__(self, status_code: int, body: Any) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> Any:
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class _ScriptedAsyncClient:
    """Routes ``get``/``post`` to pre-scripted results (response or exception).

    Both ``HandlerGatewayAttach``/``HandlerGatewayHeartbeat``/
    ``HandlerGatewayDetach._fetch_jwks`` (GET) and
    ``HandlerGatewayHeartbeat._introspect`` (POST) open their own
    ``httpx.AsyncClient()`` context -- this fake stands in for all of them
    within one monkeypatched test, so each test scripts exactly the
    GET/POST outcomes it needs and nothing else.
    """

    def __init__(
        self,
        *,
        get_result: _FakeResponse | Exception | None = None,
        post_result: _FakeResponse | Exception | None = None,
    ) -> None:
        self._get_result = get_result
        self._post_result = post_result

    async def __aenter__(self) -> _ScriptedAsyncClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def get(self, *args: object, **kwargs: object) -> _FakeResponse:
        if self._get_result is None:
            raise AssertionError("unexpected GET call in this test")
        if isinstance(self._get_result, Exception):
            raise self._get_result
        return self._get_result

    async def post(self, *args: object, **kwargs: object) -> _FakeResponse:
        if self._post_result is None:
            raise AssertionError("unexpected POST call in this test")
        if isinstance(self._post_result, Exception):
            raise self._post_result
        return self._post_result


def _patch_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    get_result: _FakeResponse | Exception | None = None,
    post_result: _FakeResponse | Exception | None = None,
) -> None:
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **_: _ScriptedAsyncClient(
            get_result=get_result, post_result=post_result
        ),
    )


@pytest.fixture
def config() -> ModelGatewayAttachConfig:
    return ModelGatewayAttachConfig()


@pytest.fixture
def tenant_key():
    return generate_key_material(TENANT_KID)


@pytest.fixture
def attacker_key():
    return generate_key_material(OTHER_KID)


@pytest.fixture
def jwks_ok(tenant_key) -> _FakeResponse:
    return _FakeResponse(200, jwks_response_body(tenant_key))


@pytest.fixture
def secret_resolver(config: ModelGatewayAttachConfig) -> _FakeSecretResolver:
    return _FakeSecretResolver(
        {
            config.keycloak_issuer_ref: ISSUER,
            config.keycloak_introspection_ref: "https://keycloak.example/introspect",
            config.keycloak_jwks_ref: "https://keycloak.example/jwks",
            f"{config.keycloak_admin_client_ref}.client_id": "admin-cli",
            f"{config.keycloak_admin_client_ref}.client_secret": "admin-secret",
        }
    )


async def _attach(
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
    store: StoreGatewaySessionMemory,
    monkeypatch: pytest.MonkeyPatch,
    tenant_key,
    jwks_ok: _FakeResponse,
    **claim_overrides: object,
):
    _patch_client(monkeypatch, get_result=jwks_ok)
    handler = HandlerGatewayAttach(
        config=config,
        session_store=store,
        secret_resolver=secret_resolver,  # type: ignore[arg-type]
    )
    token = sign_claims(tenant_key, _claims(**claim_overrides))
    return await handler.handle(
        ModelGatewayAttachRequest(access_token=token, edge_instance_id="edge-201")
    )


# --------------------------------------------------------------------------- #
# Attach
# --------------------------------------------------------------------------- #


async def test_attach_registers_active_session(
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
    monkeypatch: pytest.MonkeyPatch,
    tenant_key,
    jwks_ok: _FakeResponse,
) -> None:
    store = StoreGatewaySessionMemory()
    response = await _attach(
        config, secret_resolver, store, monkeypatch, tenant_key, jwks_ok
    )

    assert response.session.status is EnumGatewaySessionStatus.ACTIVE
    assert response.session.tenant_id == TENANT_ID
    assert response.session_event.event_type is EnumGatewaySessionEventType.ATTACHED
    stored = await store.get(response.session.session_id)
    assert stored is not None


async def test_attach_rejects_expired_token(
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
    monkeypatch: pytest.MonkeyPatch,
    tenant_key,
    jwks_ok: _FakeResponse,
) -> None:
    store = StoreGatewaySessionMemory()
    with pytest.raises(TokenValidationError):
        await _attach(
            config,
            secret_resolver,
            store,
            monkeypatch,
            tenant_key,
            jwks_ok,
            exp=1,
        )
    assert store._sessions == {}


async def test_attach_rejects_mismatched_issuer(
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
    monkeypatch: pytest.MonkeyPatch,
    tenant_key,
    jwks_ok: _FakeResponse,
) -> None:
    """R3 (pre-existing): iss must match the resolved configured issuer."""
    store = StoreGatewaySessionMemory()
    with pytest.raises(TokenValidationError, match=r"issuer|verification"):
        await _attach(
            config,
            secret_resolver,
            store,
            monkeypatch,
            tenant_key,
            jwks_ok,
            iss="https://attacker.example/realms/evil",
        )
    assert store._sessions == {}


async def test_attach_accepts_matching_issuer(
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
    monkeypatch: pytest.MonkeyPatch,
    tenant_key,
    jwks_ok: _FakeResponse,
) -> None:
    store = StoreGatewaySessionMemory()
    response = await _attach(
        config, secret_resolver, store, monkeypatch, tenant_key, jwks_ok, iss=ISSUER
    )
    assert response.session.status is EnumGatewaySessionStatus.ACTIVE


async def test_attach_rejects_forged_signature(
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
    monkeypatch: pytest.MonkeyPatch,
    attacker_key,
    jwks_ok: _FakeResponse,
) -> None:
    """OMN-15918 R1: a structurally-perfect token signed by a key that is NOT
    in the tenant's real JWKS must never attach -- this is the forged-token
    gap CodeRabbit flagged and the pre-hardening ``decode_claims`` missed
    entirely (it never referenced the signature segment at all)."""
    store = StoreGatewaySessionMemory()
    _patch_client(monkeypatch, get_result=jwks_ok)
    handler = HandlerGatewayAttach(
        config=config,
        session_store=store,
        secret_resolver=secret_resolver,  # type: ignore[arg-type]
    )
    forged = sign_claims(attacker_key, _claims())

    with pytest.raises(TokenValidationError):
        await handler.handle(
            ModelGatewayAttachRequest(access_token=forged, edge_instance_id="edge-201")
        )
    assert store._sessions == {}


async def test_attach_jwks_outage_raises_unavailable(
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
    monkeypatch: pytest.MonkeyPatch,
    tenant_key,
) -> None:
    """OMN-15918 R4: a JWKS-endpoint outage must surface as
    InfraUnavailableError (retry-able), never silently accept the token."""
    store = StoreGatewaySessionMemory()
    _patch_client(monkeypatch, get_result=httpx.ConnectError("unreachable"))
    handler = HandlerGatewayAttach(
        config=config,
        session_store=store,
        secret_resolver=secret_resolver,  # type: ignore[arg-type]
    )
    token = sign_claims(tenant_key, _claims())

    with pytest.raises(InfraUnavailableError):
        await handler.handle(
            ModelGatewayAttachRequest(access_token=token, edge_instance_id="edge-201")
        )
    assert store._sessions == {}


# --------------------------------------------------------------------------- #
# Heartbeat
# --------------------------------------------------------------------------- #


async def test_heartbeat_active_token_keeps_session_active(
    monkeypatch: pytest.MonkeyPatch,
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
    tenant_key,
    jwks_ok: _FakeResponse,
) -> None:
    store = StoreGatewaySessionMemory()
    attach_response = await _attach(
        config, secret_resolver, store, monkeypatch, tenant_key, jwks_ok
    )

    _patch_client(
        monkeypatch,
        get_result=jwks_ok,
        post_result=_FakeResponse(200, {"active": True, "client_id": "gw-tenant-acme"}),
    )
    heartbeat = HandlerGatewayHeartbeat(
        config=config,
        session_store=store,
        secret_resolver=secret_resolver,  # type: ignore[arg-type]
    )
    hb_token = sign_claims(tenant_key, _claims())
    hb_response = await heartbeat.handle(
        ModelGatewayHeartbeatRequest(
            session_id=attach_response.session.session_id, access_token=hb_token
        )
    )

    assert hb_response.termination_reason is None
    assert hb_response.session.status is EnumGatewaySessionStatus.ACTIVE
    assert (
        hb_response.session_event.event_type is EnumGatewaySessionEventType.HEARTBEAT_OK
    )


async def test_heartbeat_after_keycloak_revocation_tears_down_session(
    monkeypatch: pytest.MonkeyPatch,
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
    tenant_key,
    jwks_ok: _FakeResponse,
) -> None:
    """The revocation proof: introspection active:false kills the session."""
    store = StoreGatewaySessionMemory()
    attach_response = await _attach(
        config, secret_resolver, store, monkeypatch, tenant_key, jwks_ok
    )
    session_id = attach_response.session.session_id
    assert await store.get(session_id) is not None

    # Simulate: operator disabled the tenant's Keycloak client between attach
    # and this heartbeat tick.
    _patch_client(
        monkeypatch,
        get_result=jwks_ok,
        post_result=_FakeResponse(200, {"active": False}),
    )
    heartbeat = HandlerGatewayHeartbeat(
        config=config,
        session_store=store,
        secret_resolver=secret_resolver,  # type: ignore[arg-type]
    )
    hb_token = sign_claims(tenant_key, _claims())
    hb_response = await heartbeat.handle(
        ModelGatewayHeartbeatRequest(session_id=session_id, access_token=hb_token)
    )

    assert hb_response.termination_reason is EnumGatewaySessionTerminationReason.REVOKED
    assert hb_response.session.status is EnumGatewaySessionStatus.REVOKED
    assert hb_response.session_event.event_type is EnumGatewaySessionEventType.REVOKED
    assert await store.get(session_id) is None


async def test_heartbeat_unknown_session_raises() -> None:
    store = StoreGatewaySessionMemory()
    config = ModelGatewayAttachConfig()
    heartbeat = HandlerGatewayHeartbeat(
        config=config,
        session_store=store,
        secret_resolver=_FakeSecretResolver({}),  # type: ignore[arg-type]
    )
    with pytest.raises(HeartbeatSessionNotFoundError):
        await heartbeat.handle(
            ModelGatewayHeartbeatRequest(session_id=uuid4(), access_token="whatever")
        )


async def test_heartbeat_rejects_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
    tenant_key,
    jwks_ok: _FakeResponse,
) -> None:
    """OMN-15918 R2: a validly-signed token for a DIFFERENT tenant/principal
    must never re-validate someone else's session, even though it decodes
    and verifies cleanly on its own. Introspection must never even run."""
    store = StoreGatewaySessionMemory()
    attach_response = await _attach(
        config, secret_resolver, store, monkeypatch, tenant_key, jwks_ok
    )
    session_id = attach_response.session.session_id

    # POST is intentionally left unscripted (None): if the handler reached
    # introspection despite the identity mismatch, this test fails loudly.
    _patch_client(monkeypatch, get_result=jwks_ok, post_result=None)
    heartbeat = HandlerGatewayHeartbeat(
        config=config,
        session_store=store,
        secret_resolver=secret_resolver,  # type: ignore[arg-type]
    )
    mismatched_token = sign_claims(
        tenant_key,
        _claims(
            tenant_id=str(OTHER_TENANT_ID),
            principal_id="t-attacker",
            azp="gw-tenant-other",
        ),
    )

    with pytest.raises(TokenValidationError, match="identity"):
        await heartbeat.handle(
            ModelGatewayHeartbeatRequest(
                session_id=session_id, access_token=mismatched_token
            )
        )
    # Session is untouched -- rejection, not resurrection or revocation.
    stored = await store.get(session_id)
    assert stored is not None
    assert stored.status is EnumGatewaySessionStatus.ACTIVE


async def test_heartbeat_transport_outage_does_not_revoke_session(
    monkeypatch: pytest.MonkeyPatch,
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
    tenant_key,
    jwks_ok: _FakeResponse,
) -> None:
    """OMN-15918 R4: a transport error talking to Keycloak introspection must
    raise InfraUnavailableError and must never revoke -- the session
    survives the outage.

    OMN-16022 narrowed what "leave the session as it was" means. The
    load-bearing half of R4 is unchanged and asserted below: the session is
    still present and is NOT REVOKED. What changed is that the session is
    now marked DEGRADED on entry to the revocation-blind window, so the
    state is observable while the outage is happening. DEGRADED is a
    survival state, not a teardown -- the session keeps working and
    recovers to ACTIVE on the next successful introspection."""
    store = StoreGatewaySessionMemory()
    attach_response = await _attach(
        config, secret_resolver, store, monkeypatch, tenant_key, jwks_ok
    )
    session_id = attach_response.session.session_id

    _patch_client(
        monkeypatch, get_result=jwks_ok, post_result=httpx.ConnectError("unreachable")
    )
    heartbeat = HandlerGatewayHeartbeat(
        config=config,
        session_store=store,
        secret_resolver=secret_resolver,  # type: ignore[arg-type]
    )
    hb_token = sign_claims(tenant_key, _claims())

    with pytest.raises(InfraUnavailableError):
        await heartbeat.handle(
            ModelGatewayHeartbeatRequest(session_id=session_id, access_token=hb_token)
        )
    stored = await store.get(session_id)
    assert stored is not None, "an outage must never tear a session down"
    assert stored.status is not EnumGatewaySessionStatus.REVOKED
    assert stored.status is EnumGatewaySessionStatus.DEGRADED


async def test_heartbeat_introspection_non200_does_not_revoke_session(
    monkeypatch: pytest.MonkeyPatch,
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
    tenant_key,
    jwks_ok: _FakeResponse,
) -> None:
    store = StoreGatewaySessionMemory()
    attach_response = await _attach(
        config, secret_resolver, store, monkeypatch, tenant_key, jwks_ok
    )
    session_id = attach_response.session.session_id

    _patch_client(monkeypatch, get_result=jwks_ok, post_result=_FakeResponse(500, {}))
    heartbeat = HandlerGatewayHeartbeat(
        config=config,
        session_store=store,
        secret_resolver=secret_resolver,  # type: ignore[arg-type]
    )
    hb_token = sign_claims(tenant_key, _claims())

    with pytest.raises(InfraUnavailableError):
        await heartbeat.handle(
            ModelGatewayHeartbeatRequest(session_id=session_id, access_token=hb_token)
        )
    stored = await store.get(session_id)
    assert stored is not None, "an outage must never tear a session down"
    assert stored.status is not EnumGatewaySessionStatus.REVOKED
    assert stored.status is EnumGatewaySessionStatus.DEGRADED


async def test_heartbeat_does_not_resurrect_concurrently_detached_session(
    monkeypatch: pytest.MonkeyPatch,
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
    tenant_key,
    jwks_ok: _FakeResponse,
) -> None:
    """OMN-15918 R3 (the ticket's required concurrency proof): a detach that
    lands between heartbeat's read and its final write must not be
    resurrected. Deterministically simulated (no timing/flakiness) by
    deleting the session out from under the store exactly at the moment
    ``put_if_present`` would otherwise resurrect it."""

    class _ResurrectionRaceStore(StoreGatewaySessionMemory):
        def __init__(self) -> None:
            super().__init__()
            self.detach_before_next_write = False

        async def put_if_present(self, session):  # type: ignore[override]
            if self.detach_before_next_write:
                self.detach_before_next_write = False
                await self.delete(session.session_id)
            return await super().put_if_present(session)

    store = _ResurrectionRaceStore()
    attach_response = await _attach(
        config, secret_resolver, store, monkeypatch, tenant_key, jwks_ok
    )
    session_id = attach_response.session.session_id

    _patch_client(
        monkeypatch,
        get_result=jwks_ok,
        post_result=_FakeResponse(200, {"active": True, "client_id": "gw-tenant-acme"}),
    )
    heartbeat = HandlerGatewayHeartbeat(
        config=config,
        session_store=store,
        secret_resolver=secret_resolver,  # type: ignore[arg-type]
    )
    store.detach_before_next_write = True
    hb_token = sign_claims(tenant_key, _claims())

    with pytest.raises(HeartbeatSessionNotFoundError):
        await heartbeat.handle(
            ModelGatewayHeartbeatRequest(session_id=session_id, access_token=hb_token)
        )
    # The critical assertion: NOT resurrected. Pre-R3, an unconditional
    # `put()` here would have silently recreated the just-detached session.
    assert await store.get(session_id) is None


# --------------------------------------------------------------------------- #
# Detach
# --------------------------------------------------------------------------- #


async def test_detach_removes_session(
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
    monkeypatch: pytest.MonkeyPatch,
    tenant_key,
    jwks_ok: _FakeResponse,
) -> None:
    store = StoreGatewaySessionMemory()
    attach_response = await _attach(
        config, secret_resolver, store, monkeypatch, tenant_key, jwks_ok
    )
    session_id = attach_response.session.session_id

    _patch_client(monkeypatch, get_result=jwks_ok)
    detach = HandlerGatewayDetach(
        config=config,
        session_store=store,
        secret_resolver=secret_resolver,  # type: ignore[arg-type]
    )
    detach_token = sign_claims(tenant_key, _claims())
    detach_response = await detach.handle(
        ModelGatewayDetachRequest(
            session_id=session_id, access_token=detach_token, reason="edge shutdown"
        )
    )

    assert detach_response.status is EnumGatewaySessionStatus.DETACHED
    assert await store.get(session_id) is None


async def test_detach_unknown_session_raises(
    config: ModelGatewayAttachConfig,
) -> None:
    store = StoreGatewaySessionMemory()
    detach = HandlerGatewayDetach(
        config=config,
        session_store=store,
        secret_resolver=_FakeSecretResolver({}),  # type: ignore[arg-type]
    )
    with pytest.raises(DetachSessionNotFoundError):
        await detach.handle(
            ModelGatewayDetachRequest(
                session_id=uuid4(), access_token="whatever", reason="x"
            )
        )


async def test_detach_rejects_identity_mismatch(
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
    monkeypatch: pytest.MonkeyPatch,
    tenant_key,
    jwks_ok: _FakeResponse,
) -> None:
    """OMN-15918 R2: any caller holding a session_id could previously detach
    any tenant's session (zero credential check at all). A validly-signed
    token for a different identity must be rejected, and the session must
    survive the attempt."""
    store = StoreGatewaySessionMemory()
    attach_response = await _attach(
        config, secret_resolver, store, monkeypatch, tenant_key, jwks_ok
    )
    session_id = attach_response.session.session_id

    _patch_client(monkeypatch, get_result=jwks_ok)
    detach = HandlerGatewayDetach(
        config=config,
        session_store=store,
        secret_resolver=secret_resolver,  # type: ignore[arg-type]
    )
    mismatched_token = sign_claims(
        tenant_key,
        _claims(
            tenant_id=str(OTHER_TENANT_ID),
            principal_id="t-attacker",
            azp="gw-tenant-other",
        ),
    )

    with pytest.raises(TokenValidationError, match="identity"):
        await detach.handle(
            ModelGatewayDetachRequest(
                session_id=session_id,
                access_token=mismatched_token,
                reason="malicious detach attempt",
            )
        )
    assert await store.get(session_id) is not None


# --------------------------------------------------------------------------- #
# Introspection transport-level coverage
# --------------------------------------------------------------------------- #


class TestHeartbeatIntrospection:
    """Transport-level coverage for ``HandlerGatewayHeartbeat._introspect``.

    OMN-15918 R4: a transport error or non-200 now raises
    ``InfraUnavailableError`` instead of the pre-hardening fail-closed
    ``False`` return -- that old behavior is exactly the bug (an outage
    silently reading as "revoked").
    """

    async def test_client_id_mismatch_returns_false(
        self,
        monkeypatch: pytest.MonkeyPatch,
        config: ModelGatewayAttachConfig,
        secret_resolver: _FakeSecretResolver,
    ) -> None:
        """A clean 200 response with a MISMATCHED client_id is genuine
        revocation-class, not an outage -- still returns False."""
        _patch_client(
            monkeypatch,
            post_result=_FakeResponse(
                200, {"active": True, "client_id": "gw-tenant-someone-else"}
            ),
        )
        heartbeat = HandlerGatewayHeartbeat(
            config=config,
            session_store=StoreGatewaySessionMemory(),
            secret_resolver=secret_resolver,  # type: ignore[arg-type]
        )
        result = await heartbeat._introspect(
            access_token="tok", client_id="gw-tenant-acme"
        )
        assert result is False

    async def test_transport_error_raises_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        config: ModelGatewayAttachConfig,
        secret_resolver: _FakeSecretResolver,
    ) -> None:
        _patch_client(monkeypatch, post_result=httpx.ConnectError("unreachable"))
        heartbeat = HandlerGatewayHeartbeat(
            config=config,
            session_store=StoreGatewaySessionMemory(),
            secret_resolver=secret_resolver,  # type: ignore[arg-type]
        )
        with pytest.raises(InfraUnavailableError):
            await heartbeat._introspect(access_token="tok", client_id="gw-tenant-acme")

    async def test_non_200_raises_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        config: ModelGatewayAttachConfig,
        secret_resolver: _FakeSecretResolver,
    ) -> None:
        _patch_client(monkeypatch, post_result=_FakeResponse(500, {}))
        heartbeat = HandlerGatewayHeartbeat(
            config=config,
            session_store=StoreGatewaySessionMemory(),
            secret_resolver=secret_resolver,  # type: ignore[arg-type]
        )
        with pytest.raises(InfraUnavailableError):
            await heartbeat._introspect(access_token="tok", client_id="gw-tenant-acme")

    async def test_repeated_transport_failures_open_circuit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        config: ModelGatewayAttachConfig,
        secret_resolver: _FakeSecretResolver,
    ) -> None:
        """After ``circuit_breaker_threshold`` failures, the breaker itself
        opens and short-circuits further calls -- still InfraUnavailableError,
        proving the circuit breaker is actually wired, not just individual
        try/except handling around each call."""
        _patch_client(monkeypatch, post_result=httpx.ConnectError("unreachable"))
        heartbeat = HandlerGatewayHeartbeat(
            config=config,
            session_store=StoreGatewaySessionMemory(),
            secret_resolver=secret_resolver,  # type: ignore[arg-type]
        )
        for _ in range(config.circuit_breaker_threshold):
            with pytest.raises(InfraUnavailableError):
                await heartbeat._introspect(
                    access_token="tok", client_id="gw-tenant-acme"
                )
        assert heartbeat._introspection_circuit._circuit_breaker_open is True
        # One more call: circuit is open, must fail fast without even
        # attempting a POST (post_result stays scripted the same way, so
        # this alone doesn't distinguish -- the fast-fail is asserted via
        # the breaker's own open state above, which _check_circuit_breaker
        # reads before any transport call is attempted).
        with pytest.raises(InfraUnavailableError):
            await heartbeat._introspect(access_token="tok", client_id="gw-tenant-acme")


# --------------------------------------------------------------------------- #
# OMN-16022: session expiry enforcement + bounded degraded mode
#
# RED-first. Every test in this section fails against the pre-OMN-16022
# handlers, and each failure is behavioral (a wrong outcome), not an
# ImportError against a symbol that does not exist yet:
#   - expiry: the handler happily revalidates a session whose ``expires_at``
#     is in the past and leaves it in the store.
#   - ceiling: an introspection outage raises InfraUnavailableError forever,
#     so a session survives un-revalidated for the entire outage.
#   - degraded entry: nothing marks the session DEGRADED and nothing alarms.
#   - detach: an expired session detaches normally instead of being rejected.
# --------------------------------------------------------------------------- #


async def test_heartbeat_rejects_session_past_expires_at(
    monkeypatch: pytest.MonkeyPatch,
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
    tenant_key,
    jwks_ok: _FakeResponse,
) -> None:
    """OMN-16022 AC (a): ``expires_at`` is stored but never enforced.

    Keycloak is scripted fully healthy here (JWKS resolves, introspection
    returns ``active: true``) so the ONLY thing under test is the stored
    expiry. Pre-fix the handler returns a normal HEARTBEAT_OK response and
    the session stays in the store forever -- the stored ceiling is
    declarative only.
    """
    store = StoreGatewaySessionMemory()
    attach_response = await _attach(
        config, secret_resolver, store, monkeypatch, tenant_key, jwks_ok
    )
    session_id = attach_response.session.session_id
    await store.put(
        attach_response.session.model_copy(
            update={"expires_at": datetime.now(UTC) - timedelta(seconds=1)}
        )
    )

    _patch_client(
        monkeypatch,
        get_result=jwks_ok,
        post_result=_FakeResponse(200, {"active": True, "client_id": "gw-tenant-acme"}),
    )
    heartbeat = HandlerGatewayHeartbeat(
        config=config,
        session_store=store,
        secret_resolver=secret_resolver,  # type: ignore[arg-type]
    )
    hb_token = sign_claims(tenant_key, _claims())
    response = await heartbeat.handle(
        ModelGatewayHeartbeatRequest(session_id=session_id, access_token=hb_token)
    )

    assert response.termination_reason is EnumGatewaySessionTerminationReason.EXPIRED
    assert response.session.status is EnumGatewaySessionStatus.EXPIRED
    assert response.session_event.event_type is EnumGatewaySessionEventType.EXPIRED
    assert await store.get(session_id) is None


async def test_heartbeat_quarantines_session_past_unverified_ceiling(
    monkeypatch: pytest.MonkeyPatch,
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
    tenant_key,
    jwks_ok: _FakeResponse,
) -> None:
    """OMN-16022 AC (b): unbounded fail-open is a revocation-suppression vector.

    The session was last successfully revalidated 1000s ago (longer than the
    degraded-mode ceiling) and Keycloak is unreachable. Pre-fix the handler
    raises InfraUnavailableError and leaves the session ACTIVE -- it survives
    un-revalidated for as long as the outage (or an attacker-induced
    partition) lasts.
    """
    store = StoreGatewaySessionMemory()
    attach_response = await _attach(
        config, secret_resolver, store, monkeypatch, tenant_key, jwks_ok
    )
    session_id = attach_response.session.session_id
    await store.put(
        attach_response.session.model_copy(
            update={
                "last_heartbeat_at": datetime.now(UTC)
                - timedelta(seconds=config.max_unverified_session_seconds + 60)
            }
        )
    )

    _patch_client(
        monkeypatch, get_result=jwks_ok, post_result=httpx.ConnectError("unreachable")
    )
    heartbeat = HandlerGatewayHeartbeat(
        config=config,
        session_store=store,
        secret_resolver=secret_resolver,  # type: ignore[arg-type]
    )
    hb_token = sign_claims(tenant_key, _claims())
    response = await heartbeat.handle(
        ModelGatewayHeartbeatRequest(session_id=session_id, access_token=hb_token)
    )

    assert (
        response.termination_reason
        is EnumGatewaySessionTerminationReason.UNVERIFIED_CEILING
    )
    assert response.session.status is EnumGatewaySessionStatus.QUARANTINED
    assert response.session_event.event_type is EnumGatewaySessionEventType.QUARANTINED
    assert await store.get(session_id) is None


async def test_heartbeat_alarms_on_entry_to_degraded_mode(
    monkeypatch: pytest.MonkeyPatch,
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
    tenant_key,
    jwks_ok: _FakeResponse,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """OMN-16022 AC (b): alarm on ENTRY to degraded mode, not only on breach.

    Below the ceiling the OMN-15918 invariant is preserved exactly -- an
    outage still raises InfraUnavailableError and still never revokes. What
    is missing pre-fix is that entering degraded mode is invisible: the
    session stays ACTIVE and nothing is logged, so the operator learns about
    a revocation-blind window only when the ceiling finally fires.
    """
    store = StoreGatewaySessionMemory()
    attach_response = await _attach(
        config, secret_resolver, store, monkeypatch, tenant_key, jwks_ok
    )
    session_id = attach_response.session.session_id

    _patch_client(
        monkeypatch, get_result=jwks_ok, post_result=httpx.ConnectError("unreachable")
    )
    heartbeat = HandlerGatewayHeartbeat(
        config=config,
        session_store=store,
        secret_resolver=secret_resolver,  # type: ignore[arg-type]
    )
    hb_token = sign_claims(tenant_key, _claims())

    with caplog.at_level(logging.WARNING):
        with pytest.raises(InfraUnavailableError):
            await heartbeat.handle(
                ModelGatewayHeartbeatRequest(
                    session_id=session_id, access_token=hb_token
                )
            )

    stored = await store.get(session_id)
    assert stored is not None, "an outage must never revoke (OMN-15918 R4)"
    assert stored.status is EnumGatewaySessionStatus.DEGRADED
    assert any("degraded" in record.getMessage().lower() for record in caplog.records)


async def test_detach_rejects_session_past_expires_at(
    monkeypatch: pytest.MonkeyPatch,
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
    tenant_key,
    jwks_ok: _FakeResponse,
) -> None:
    """OMN-16022 AC (a): detach is a session-consuming path too.

    Pre-fix detach reads the expired session, spends a JWKS round-trip on
    it, and reports a normal DETACHED outcome -- an expired session is
    still a usable session on every path that reads one.
    """
    store = StoreGatewaySessionMemory()
    attach_response = await _attach(
        config, secret_resolver, store, monkeypatch, tenant_key, jwks_ok
    )
    session_id = attach_response.session.session_id
    await store.put(
        attach_response.session.model_copy(
            update={"expires_at": datetime.now(UTC) - timedelta(seconds=1)}
        )
    )

    _patch_client(monkeypatch, get_result=jwks_ok)
    detach = HandlerGatewayDetach(
        config=config,
        session_store=store,
        secret_resolver=secret_resolver,  # type: ignore[arg-type]
    )
    detach_token = sign_claims(tenant_key, _claims())

    with pytest.raises(SessionExpiredError, match="expired"):
        await detach.handle(
            ModelGatewayDetachRequest(
                session_id=session_id, access_token=detach_token, reason="edge shutdown"
            )
        )
    assert await store.get(session_id) is None


async def test_healthy_runtime_reattaches_after_enforced_expiry(
    monkeypatch: pytest.MonkeyPatch,
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
    tenant_key,
    jwks_ok: _FakeResponse,
) -> None:
    """OMN-16022 AC (c): enforced expiry stays compatible with OMN-15952.

    The remedy for an enforced expiry is re-attach, and re-attach must
    remain cheap and automatic for a runtime whose credential is still
    good -- that asymmetry (healthy re-attaches, revoked cannot) is what
    makes the ceiling safe to enforce at all.
    """
    store = StoreGatewaySessionMemory()
    attach_response = await _attach(
        config, secret_resolver, store, monkeypatch, tenant_key, jwks_ok
    )
    first_session_id = attach_response.session.session_id
    await store.put(
        attach_response.session.model_copy(
            update={"expires_at": datetime.now(UTC) - timedelta(seconds=1)}
        )
    )

    _patch_client(
        monkeypatch,
        get_result=jwks_ok,
        post_result=_FakeResponse(200, {"active": True, "client_id": "gw-tenant-acme"}),
    )
    heartbeat = HandlerGatewayHeartbeat(
        config=config,
        session_store=store,
        secret_resolver=secret_resolver,  # type: ignore[arg-type]
    )
    await heartbeat.handle(
        ModelGatewayHeartbeatRequest(
            session_id=first_session_id,
            access_token=sign_claims(tenant_key, _claims()),
        )
    )
    assert await store.get(first_session_id) is None

    reattach_response = await _attach(
        config, secret_resolver, store, monkeypatch, tenant_key, jwks_ok
    )
    assert reattach_response.session.session_id != first_session_id
    assert reattach_response.session.status is EnumGatewaySessionStatus.ACTIVE
    assert await store.get(reattach_response.session.session_id) is not None


async def test_ceiling_fires_even_when_keycloak_is_healthy(
    monkeypatch: pytest.MonkeyPatch,
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
    tenant_key,
    jwks_ok: _FakeResponse,
) -> None:
    """The ceiling measures revalidation staleness, not Keycloak reachability.

    Introspection is scripted fully healthy here. A session that has gone
    longer than the ceiling without a successful revalidation is still torn
    down -- at a 15s heartbeat cadence that is 60+ consecutive missed
    heartbeats, i.e. a dead edge, and the remedy is the same cheap
    re-attach. Tying the ceiling to "is Keycloak up right now" would put
    the bound back under the control of whoever can take Keycloak down.
    """
    store = StoreGatewaySessionMemory()
    attach_response = await _attach(
        config, secret_resolver, store, monkeypatch, tenant_key, jwks_ok
    )
    session_id = attach_response.session.session_id
    await store.put(
        attach_response.session.model_copy(
            update={
                "last_heartbeat_at": datetime.now(UTC)
                - timedelta(seconds=config.max_unverified_session_seconds + 1)
            }
        )
    )

    _patch_client(
        monkeypatch,
        get_result=jwks_ok,
        post_result=_FakeResponse(200, {"active": True, "client_id": "gw-tenant-acme"}),
    )
    heartbeat = HandlerGatewayHeartbeat(
        config=config,
        session_store=store,
        secret_resolver=secret_resolver,  # type: ignore[arg-type]
    )
    response = await heartbeat.handle(
        ModelGatewayHeartbeatRequest(
            session_id=session_id, access_token=sign_claims(tenant_key, _claims())
        )
    )

    assert (
        response.termination_reason
        is EnumGatewaySessionTerminationReason.UNVERIFIED_CEILING
    )
    assert await store.get(session_id) is None


async def test_degraded_alarm_fires_once_not_every_tick(
    monkeypatch: pytest.MonkeyPatch,
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
    tenant_key,
    jwks_ok: _FakeResponse,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Alarm on ENTRY means entry, not every heartbeat for the whole outage.

    A fleet riding out a multi-minute Keycloak outage would otherwise emit
    one record per session per heartbeat interval, which is how an alarm
    stops being read at exactly the moment it matters.
    """
    store = StoreGatewaySessionMemory()
    attach_response = await _attach(
        config, secret_resolver, store, monkeypatch, tenant_key, jwks_ok
    )
    session_id = attach_response.session.session_id

    _patch_client(
        monkeypatch, get_result=jwks_ok, post_result=httpx.ConnectError("unreachable")
    )
    heartbeat = HandlerGatewayHeartbeat(
        config=config,
        session_store=store,
        secret_resolver=secret_resolver,  # type: ignore[arg-type]
    )
    hb_token = sign_claims(tenant_key, _claims())

    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            with pytest.raises(InfraUnavailableError):
                await heartbeat.handle(
                    ModelGatewayHeartbeatRequest(
                        session_id=session_id, access_token=hb_token
                    )
                )

    entries = [
        record
        for record in caplog.records
        if getattr(record, "alarm", None) == "gateway.session.degraded_entered"
    ]
    assert len(entries) == 1
    assert entries[0].degraded_reason == "introspection_unavailable"  # type: ignore[attr-defined]
    stored = await store.get(session_id)
    assert stored is not None
    assert stored.status is EnumGatewaySessionStatus.DEGRADED


async def test_revoked_runtime_cannot_hold_a_session_after_reattach(
    monkeypatch: pytest.MonkeyPatch,
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
    tenant_key,
    jwks_ok: _FakeResponse,
) -> None:
    """OMN-16022 AC (c), negative half: re-attach is not a revocation bypass.

    Honest scope: attach itself does not introspect (it checks signature,
    issuer, audience and exp only), so a client disabled while one of its
    tokens is still unexpired CAN complete a re-attach. What it cannot do
    is keep the session -- the first heartbeat introspects and tears it
    down. Revocation latency at the attach boundary is therefore bounded by
    heartbeat_interval_seconds, not zero; that residual is recorded in the
    node contract's known_gaps.
    """
    store = StoreGatewaySessionMemory()
    reattach = await _attach(
        config, secret_resolver, store, monkeypatch, tenant_key, jwks_ok
    )
    session_id = reattach.session.session_id

    _patch_client(
        monkeypatch,
        get_result=jwks_ok,
        post_result=_FakeResponse(200, {"active": False}),
    )
    heartbeat = HandlerGatewayHeartbeat(
        config=config,
        session_store=store,
        secret_resolver=secret_resolver,  # type: ignore[arg-type]
    )
    response = await heartbeat.handle(
        ModelGatewayHeartbeatRequest(
            session_id=session_id, access_token=sign_claims(tenant_key, _claims())
        )
    )

    assert response.termination_reason is EnumGatewaySessionTerminationReason.REVOKED
    assert await store.get(session_id) is None


async def test_degraded_session_recovers_to_active_when_keycloak_returns(
    monkeypatch: pytest.MonkeyPatch,
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
    tenant_key,
    jwks_ok: _FakeResponse,
) -> None:
    """DEGRADED is a survival state, not a one-way trip to teardown.

    Asserts the claim the two OMN-15918 outage tests now lean on: marking a
    session DEGRADED during an outage does not doom it. The next heartbeat
    that actually reaches Keycloak restores ACTIVE and, critically,
    advances last_heartbeat_at -- which resets the ceiling clock, so a
    recovered session is not quarantined for time it spent degraded.
    """
    store = StoreGatewaySessionMemory()
    attach_response = await _attach(
        config, secret_resolver, store, monkeypatch, tenant_key, jwks_ok
    )
    session_id = attach_response.session.session_id
    hb_token = sign_claims(tenant_key, _claims())
    heartbeat = HandlerGatewayHeartbeat(
        config=config,
        session_store=store,
        secret_resolver=secret_resolver,  # type: ignore[arg-type]
    )

    _patch_client(
        monkeypatch, get_result=jwks_ok, post_result=httpx.ConnectError("unreachable")
    )
    with pytest.raises(InfraUnavailableError):
        await heartbeat.handle(
            ModelGatewayHeartbeatRequest(session_id=session_id, access_token=hb_token)
        )
    degraded = await store.get(session_id)
    assert degraded is not None
    assert degraded.status is EnumGatewaySessionStatus.DEGRADED

    _patch_client(
        monkeypatch,
        get_result=jwks_ok,
        post_result=_FakeResponse(200, {"active": True, "client_id": "gw-tenant-acme"}),
    )
    recovered = await heartbeat.handle(
        ModelGatewayHeartbeatRequest(session_id=session_id, access_token=hb_token)
    )

    assert recovered.termination_reason is None
    assert recovered.session.status is EnumGatewaySessionStatus.ACTIVE
    assert recovered.session.last_heartbeat_at > degraded.last_heartbeat_at
