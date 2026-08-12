# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the attach/heartbeat/detach handler trio.

This is the end-to-end slice proof at the unit level (OMN-15753's local
component): attach -> authenticated ACTIVE session -> heartbeat (active) ->
heartbeat (introspection reports revoked) -> session torn down. The
introspection HTTP call lives inline in ``HandlerGatewayHeartbeat._introspect``
(moved out of a freestanding services/ module to satisfy the
imperative-contract-guard's handlers/-only I/O boundary) and is faked via
monkeypatch here, including ``TestHeartbeatIntrospection`` for the
transport-level fail-closed cases that used to live in
test_service_keycloak_token_validator.py.
"""

from __future__ import annotations

import base64
import json
from uuid import UUID, uuid4

import httpx
import pytest
from pydantic import SecretStr

from omnibase_infra.nodes.node_gateway_attach_effect.handlers.handler_gateway_attach import (
    HandlerGatewayAttach,
)
from omnibase_infra.nodes.node_gateway_attach_effect.handlers.handler_gateway_detach import (
    HandlerGatewayDetach,
    SessionNotFoundError,
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
from omnibase_infra.nodes.node_gateway_attach_effect.services.service_keycloak_token_validator import (
    TokenValidationError,
)
from omnibase_infra.nodes.node_gateway_attach_effect.services.store_gateway_session_memory import (
    StoreGatewaySessionMemory,
)

TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")


def _fake_jwt(**overrides: object) -> str:
    claims = {
        "iss": "https://keycloak.example/realms/omninode",
        "sub": "svc-acct-abc",
        "aud": "gateway-attach",
        "tenant_id": str(TENANT_ID),
        "tenant_slug": "acme",
        "principal_id": "t-11111111111111111111111111111111",
        "azp": "gw-tenant-acme",
        "exp": 9999999999,
    }
    claims.update(overrides)
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
    payload = (
        base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    )
    return f"{header}.{payload}.sig"


class _FakeSecretResolver:
    def __init__(self, values: dict[str, str]) -> None:
        self._values = values

    async def get_secret_async(
        self, logical_name: str, required: bool = True, correlation_id: object = None
    ) -> SecretStr:
        return SecretStr(self._values[logical_name])


class _FakeResponse:
    def __init__(self, status_code: int, body: dict[str, object]) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> dict[str, object]:
        return self._body


class _FakeAsyncClient:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def post(self, *args: object, **kwargs: object) -> _FakeResponse:
        return self._response


@pytest.fixture
def config() -> ModelGatewayAttachConfig:
    return ModelGatewayAttachConfig()


@pytest.fixture
def secret_resolver(config: ModelGatewayAttachConfig) -> _FakeSecretResolver:
    return _FakeSecretResolver(
        {
            config.keycloak_issuer_ref: "https://keycloak.example/realms/omninode",
            config.keycloak_introspection_ref: "https://keycloak.example/introspect",
            f"{config.keycloak_admin_client_ref}.client_id": "admin-cli",
            f"{config.keycloak_admin_client_ref}.client_secret": "admin-secret",
        }
    )


async def test_attach_registers_active_session(
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
) -> None:
    store = StoreGatewaySessionMemory()
    handler = HandlerGatewayAttach(
        config=config,
        session_store=store,
        secret_resolver=secret_resolver,  # type: ignore[arg-type]
    )

    response = await handler.handle(
        ModelGatewayAttachRequest(access_token=_fake_jwt(), edge_instance_id="edge-201")
    )

    assert response.session.status is EnumGatewaySessionStatus.ACTIVE
    assert response.session.tenant_id == TENANT_ID
    assert response.session_event.event_type is EnumGatewaySessionEventType.ATTACHED
    stored = await store.get(response.session.session_id)
    assert stored is not None


async def test_attach_rejects_expired_token(
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
) -> None:
    store = StoreGatewaySessionMemory()
    handler = HandlerGatewayAttach(
        config=config,
        session_store=store,
        secret_resolver=secret_resolver,  # type: ignore[arg-type]
    )

    with pytest.raises(TokenValidationError):
        await handler.handle(
            ModelGatewayAttachRequest(
                access_token=_fake_jwt(exp=0), edge_instance_id="edge-201"
            )
        )
    # Nothing was ever registered for the rejected token.
    assert store._sessions == {}


async def test_attach_rejects_mismatched_issuer(
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
) -> None:
    """R3: HandlerGatewayAttach resolves keycloak_issuer_ref and rejects a
    token whose iss claim does not match, end-to-end through the handler."""
    store = StoreGatewaySessionMemory()
    handler = HandlerGatewayAttach(
        config=config,
        session_store=store,
        secret_resolver=secret_resolver,  # type: ignore[arg-type]
    )

    with pytest.raises(TokenValidationError, match="issuer"):
        await handler.handle(
            ModelGatewayAttachRequest(
                access_token=_fake_jwt(iss="https://attacker.example/realms/evil"),
                edge_instance_id="edge-201",
            )
        )
    assert store._sessions == {}


async def test_attach_accepts_matching_issuer(
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
) -> None:
    """R3 happy path: a token whose iss matches the resolved configured
    issuer attaches successfully end-to-end through the handler."""
    store = StoreGatewaySessionMemory()
    handler = HandlerGatewayAttach(
        config=config,
        session_store=store,
        secret_resolver=secret_resolver,  # type: ignore[arg-type]
    )

    response = await handler.handle(
        ModelGatewayAttachRequest(
            access_token=_fake_jwt(iss="https://keycloak.example/realms/omninode"),
            edge_instance_id="edge-201",
        )
    )

    assert response.session.status is EnumGatewaySessionStatus.ACTIVE


async def test_heartbeat_active_token_keeps_session_active(
    monkeypatch: pytest.MonkeyPatch,
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
) -> None:
    store = StoreGatewaySessionMemory()
    attach = HandlerGatewayAttach(
        config=config,
        session_store=store,
        secret_resolver=secret_resolver,  # type: ignore[arg-type]
    )
    attach_response = await attach.handle(
        ModelGatewayAttachRequest(access_token=_fake_jwt(), edge_instance_id="edge-201")
    )

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **_: _FakeAsyncClient(
            _FakeResponse(200, {"active": True, "client_id": "gw-tenant-acme"})
        ),
    )
    heartbeat = HandlerGatewayHeartbeat(
        config=config,
        session_store=store,
        secret_resolver=secret_resolver,  # type: ignore[arg-type]
    )
    hb_response = await heartbeat.handle(
        ModelGatewayHeartbeatRequest(
            session_id=attach_response.session.session_id, access_token=_fake_jwt()
        )
    )

    assert hb_response.revoked is False
    assert hb_response.session.status is EnumGatewaySessionStatus.ACTIVE
    assert (
        hb_response.session_event.event_type is EnumGatewaySessionEventType.HEARTBEAT_OK
    )


async def test_heartbeat_after_keycloak_revocation_tears_down_session(
    monkeypatch: pytest.MonkeyPatch,
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
) -> None:
    """The revocation proof: introspection active:false kills the session."""
    store = StoreGatewaySessionMemory()
    attach = HandlerGatewayAttach(
        config=config,
        session_store=store,
        secret_resolver=secret_resolver,  # type: ignore[arg-type]
    )
    attach_response = await attach.handle(
        ModelGatewayAttachRequest(access_token=_fake_jwt(), edge_instance_id="edge-201")
    )
    session_id = attach_response.session.session_id
    assert await store.get(session_id) is not None

    # Simulate: operator disabled the tenant's Keycloak client between attach
    # and this heartbeat tick.
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **_: _FakeAsyncClient(_FakeResponse(200, {"active": False})),
    )
    heartbeat = HandlerGatewayHeartbeat(
        config=config,
        session_store=store,
        secret_resolver=secret_resolver,  # type: ignore[arg-type]
    )
    hb_response = await heartbeat.handle(
        ModelGatewayHeartbeatRequest(session_id=session_id, access_token=_fake_jwt())
    )

    assert hb_response.revoked is True
    assert hb_response.session.status is EnumGatewaySessionStatus.REVOKED
    assert hb_response.session_event.event_type is EnumGatewaySessionEventType.REVOKED
    # Session is gone from the store -- a subsequent heartbeat/detach on it 404s.
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
            ModelGatewayHeartbeatRequest(session_id=uuid4(), access_token=_fake_jwt())
        )


async def test_detach_removes_session(
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
) -> None:
    store = StoreGatewaySessionMemory()
    attach = HandlerGatewayAttach(
        config=config,
        session_store=store,
        secret_resolver=secret_resolver,  # type: ignore[arg-type]
    )
    attach_response = await attach.handle(
        ModelGatewayAttachRequest(access_token=_fake_jwt(), edge_instance_id="edge-201")
    )
    session_id = attach_response.session.session_id

    detach = HandlerGatewayDetach(session_store=store)
    detach_response = await detach.handle(
        ModelGatewayDetachRequest(session_id=session_id, reason="edge shutdown")
    )

    assert detach_response.status is EnumGatewaySessionStatus.DETACHED
    assert await store.get(session_id) is None


async def test_detach_unknown_session_raises() -> None:
    store = StoreGatewaySessionMemory()
    detach = HandlerGatewayDetach(session_store=store)
    with pytest.raises(SessionNotFoundError):
        await detach.handle(ModelGatewayDetachRequest(session_id=uuid4(), reason="x"))


class TestHeartbeatIntrospection:
    """Transport-level fail-closed coverage for ``HandlerGatewayHeartbeat._introspect``.

    Moved here from test_service_keycloak_token_validator.py when the RFC 7662
    HTTP call relocated from a freestanding services/ module into this
    handler (imperative-contract-guard's handlers/-only I/O boundary).
    """

    async def test_client_id_mismatch_returns_false(
        self,
        monkeypatch: pytest.MonkeyPatch,
        config: ModelGatewayAttachConfig,
        secret_resolver: _FakeSecretResolver,
    ) -> None:
        response = _FakeResponse(
            200, {"active": True, "client_id": "gw-tenant-someone-else"}
        )
        monkeypatch.setattr(
            httpx, "AsyncClient", lambda **_: _FakeAsyncClient(response)
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

    async def test_transport_error_fails_closed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        config: ModelGatewayAttachConfig,
        secret_resolver: _FakeSecretResolver,
    ) -> None:
        class _RaisingAsyncClient(_FakeAsyncClient):
            async def post(self, *args: object, **kwargs: object) -> _FakeResponse:
                raise httpx.ConnectError("unreachable")

        monkeypatch.setattr(
            httpx,
            "AsyncClient",
            lambda **_: _RaisingAsyncClient(_FakeResponse(200, {})),
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

    async def test_non_200_fails_closed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        config: ModelGatewayAttachConfig,
        secret_resolver: _FakeSecretResolver,
    ) -> None:
        response = _FakeResponse(500, {})
        monkeypatch.setattr(
            httpx, "AsyncClient", lambda **_: _FakeAsyncClient(response)
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
