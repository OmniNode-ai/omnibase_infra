# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler-level proof of the OMN-15952 unattended renewal contract.

Three claims, driven through the real handlers against real RS256-signed
tokens and a mocked JWKS endpoint -- no browser session anywhere in the
loop, which is the gap this ticket exists to close:

  1. **Attach serves the renewal cycle.** RED before this ticket:
     ``ModelGatewayAttachResponse`` had no ``renewal`` field, so a runtime
     was told how often to heartbeat and nothing about how to survive its
     own ceiling.
  2. **A heartbeat never moves ``expires_at``** -- at, before, or after the
     ceiling. This is the boundary the design's rev-3 correction turned the
     whole contract on: if a heartbeat could extend the session, renewal
     would be in-place and this ticket would be writing a different
     document. The assertion is on the session-store WRITES rather than on
     the handler's return value, so it holds regardless of whether a
     post-ceiling heartbeat returns, revokes, or raises -- the invariant is
     "no write ever moved the ceiling", not "this particular call
     succeeded".
  3. **Re-attach after expiry mints a NEW session from a fresh grant.** The
     successor carries a new ``session_id`` and its own later ceiling; the
     predecessor is left exactly as it was. This is what makes the contract
     compatible with the merged ``gateway_sessions`` projection
     (omninode_infra#899), whose UPSERT is keyed on ``session_id`` alone:
     a re-attach writes a new row rather than mutating the old one, so two
     runtimes on one tenant -- or one runtime across a renewal boundary --
     never collide.

No browser dependency is imported or stubbed here, structurally: the only
network surfaces faked are Keycloak's JWKS and introspection endpoints.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from omnibase_infra.nodes.node_gateway_attach_effect.handlers.handler_gateway_attach import (
    HandlerGatewayAttach,
)
from omnibase_infra.nodes.node_gateway_attach_effect.handlers.handler_gateway_heartbeat import (
    HandlerGatewayHeartbeat,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.enum_gateway_renewal_mode import (
    EnumGatewayRenewalMode,
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
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_attach_response import (
    ModelGatewayAttachResponse,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_heartbeat_request import (
    ModelGatewayHeartbeatRequest,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_session import (
    ModelGatewaySession,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_session_event import (
    ModelGatewaySessionEvent,
)
from omnibase_infra.nodes.node_gateway_attach_effect.services.service_gateway_renewal_policy import (
    assert_expiry_not_extended,
)
from omnibase_infra.nodes.node_gateway_attach_effect.services.store_gateway_session_memory import (
    StoreGatewaySessionMemory,
)
from tests.unit.nodes.node_gateway_attach_effect._jwt_test_support import (
    TENANT_KID,
    generate_key_material,
    jwks_response_body,
    sign_claims,
)

pytestmark = pytest.mark.unit

TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")
ISSUER = "https://keycloak.example/realms/omninode"
EDGE_INSTANCE_ID = "edge-201"


def _claims(*, expires_in_seconds: int) -> dict[str, object]:
    """Claim set for a client_credentials token with a chosen remaining life."""
    issued_at = int(datetime.now(UTC).timestamp())
    return {
        "iss": ISSUER,
        "sub": "svc-acct-abc",
        "aud": "gateway-attach",
        "tenant_id": str(TENANT_ID),
        "tenant_slug": "acme",
        "principal_id": "t-11111111111111111111111111111111",
        "azp": "ga-tenant-acme",
        # OMN-16023: iat is a required claim — the validator bounds exp - iat.
        "iat": issued_at,
        "exp": issued_at + expires_in_seconds,
    }


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
        return self._body


class _ScriptedAsyncClient:
    def __init__(
        self,
        *,
        get_result: _FakeResponse | None = None,
        post_result: _FakeResponse | None = None,
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
        return self._get_result

    async def post(self, *args: object, **kwargs: object) -> _FakeResponse:
        if self._post_result is None:
            raise AssertionError("unexpected POST call in this test")
        return self._post_result


def _patch_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    get_result: _FakeResponse | None = None,
    post_result: _FakeResponse | None = None,
) -> None:
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **_: _ScriptedAsyncClient(
            get_result=get_result, post_result=post_result
        ),
    )


class _WriteRecordingStore(StoreGatewaySessionMemory):
    """In-memory store that keeps every session revision ever written.

    The renewal contract's central negative is about writes, not about
    return values: a handler could return an untouched session while
    persisting an extended one. Recording the writes is the only way to
    assert on what was actually stored.
    """

    def __init__(self) -> None:
        super().__init__()
        self.writes: list[ModelGatewaySession] = []

    async def put(self, session: ModelGatewaySession) -> None:
        self.writes.append(session)
        await super().put(session)

    async def put_if_present(self, session: ModelGatewaySession) -> bool:
        applied = await super().put_if_present(session)
        if applied:
            self.writes.append(session)
        return applied


@pytest.fixture
def config() -> ModelGatewayAttachConfig:
    return ModelGatewayAttachConfig()


@pytest.fixture
def tenant_key():
    return generate_key_material(TENANT_KID)


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
    *,
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
    store: StoreGatewaySessionMemory,
    monkeypatch: pytest.MonkeyPatch,
    tenant_key: Any,
    jwks_ok: _FakeResponse,
    expires_in_seconds: int,
):
    """One full attach against a freshly granted client_credentials token."""
    _patch_client(monkeypatch, get_result=jwks_ok)
    handler = HandlerGatewayAttach(
        config=config,
        session_store=store,
        secret_resolver=secret_resolver,  # type: ignore[arg-type]
    )
    token = sign_claims(tenant_key, _claims(expires_in_seconds=expires_in_seconds))
    response = await handler.handle(
        ModelGatewayAttachRequest(access_token=token, edge_instance_id=EDGE_INSTANCE_ID)
    )
    return response, token


# --------------------------------------------------------------------------- #
# 1. Attach serves the renewal cycle
# --------------------------------------------------------------------------- #


async def test_attach_hands_the_runtime_its_renewal_cycle(
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
    monkeypatch: pytest.MonkeyPatch,
    tenant_key: Any,
    jwks_ok: _FakeResponse,
) -> None:
    store = StoreGatewaySessionMemory()
    response, _ = await _attach(
        config=config,
        secret_resolver=secret_resolver,
        store=store,
        monkeypatch=monkeypatch,
        tenant_key=tenant_key,
        jwks_ok=jwks_ok,
        expires_in_seconds=900,
    )

    renewal = response.renewal
    assert renewal.mode is EnumGatewayRenewalMode.RE_ATTACH
    assert renewal.session_expires_at == response.session.expires_at
    assert renewal.renew_not_before <= renewal.renew_at < response.session.expires_at
    assert renewal.margin_seconds == config.renewal_margin_seconds
    assert renewal.jitter_seconds == config.renewal_jitter_seconds
    # The window is inside this session's life, not before it began.
    assert renewal.renew_not_before >= response.session.attached_at


# --------------------------------------------------------------------------- #
# 2. A heartbeat never moves expires_at
# --------------------------------------------------------------------------- #


async def test_heartbeat_within_validity_does_not_extend_expires_at(
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
    monkeypatch: pytest.MonkeyPatch,
    tenant_key: Any,
    jwks_ok: _FakeResponse,
) -> None:
    store = _WriteRecordingStore()
    attach_response, token = await _attach(
        config=config,
        secret_resolver=secret_resolver,
        store=store,
        monkeypatch=monkeypatch,
        tenant_key=tenant_key,
        jwks_ok=jwks_ok,
        expires_in_seconds=900,
    )
    attached = attach_response.session

    _patch_client(
        monkeypatch,
        get_result=jwks_ok,
        post_result=_FakeResponse(
            200, {"active": True, "client_id": attached.keycloak_client_id}
        ),
    )
    heartbeat = HandlerGatewayHeartbeat(
        config=config,
        session_store=store,
        secret_resolver=secret_resolver,  # type: ignore[arg-type]
    )
    result = await heartbeat.handle(
        ModelGatewayHeartbeatRequest(session_id=attached.session_id, access_token=token)
    )

    # Liveness advanced; the ceiling did not.
    assert result.session.last_heartbeat_at >= attached.last_heartbeat_at
    assert_expiry_not_extended(attached, result.session)
    for written in store.writes:
        assert_expiry_not_extended(attached, written)


async def test_no_heartbeat_at_or_past_the_ceiling_ever_extends_it(
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
    monkeypatch: pytest.MonkeyPatch,
    tenant_key: Any,
    jwks_ok: _FakeResponse,
) -> None:
    """The boundary case the whole contract turns on.

    A session is planted at the ceiling and then past it, and heartbeats are
    driven against it with Keycloak reporting the token still active -- the
    most favourable possible conditions for an implementation that wanted to
    extend. The assertion is on the store's write log, so it is indifferent
    to whether the handler returns normally, reports the session terminated,
    or raises: whatever it does, it must not have moved ``expires_at``.
    """
    store = _WriteRecordingStore()
    attach_response, token = await _attach(
        config=config,
        secret_resolver=secret_resolver,
        store=store,
        monkeypatch=monkeypatch,
        tenant_key=tenant_key,
        jwks_ok=jwks_ok,
        expires_in_seconds=900,
    )
    attached = attach_response.session

    _patch_client(
        monkeypatch,
        get_result=jwks_ok,
        post_result=_FakeResponse(
            200, {"active": True, "client_id": attached.keycloak_client_id}
        ),
    )
    heartbeat = HandlerGatewayHeartbeat(
        config=config,
        session_store=store,
        secret_resolver=secret_resolver,  # type: ignore[arg-type]
    )

    for offset in (timedelta(0), timedelta(seconds=1), timedelta(seconds=600)):
        at_or_past_ceiling = attached.model_copy(
            update={"last_heartbeat_at": attached.expires_at + offset}
        )
        # Plant the aged session directly; the point is the ceiling, not the
        # wall clock.
        await StoreGatewaySessionMemory.put(store, at_or_past_ceiling)
        store.writes.clear()

        # A post-ceiling heartbeat may legitimately succeed, tear the
        # session down, or raise depending on the enforcement landed at the
        # time. None of those outcomes is allowed to move the ceiling.
        with contextlib.suppress(Exception):
            await heartbeat.handle(
                ModelGatewayHeartbeatRequest(
                    session_id=attached.session_id, access_token=token
                )
            )

        for written in store.writes:
            assert_expiry_not_extended(attached, written)
        surviving = await store.get(attached.session_id)
        if surviving is not None:
            assert_expiry_not_extended(attached, surviving)


# --------------------------------------------------------------------------- #
# 3. Re-attach after expiry mints a NEW session from a fresh grant
# --------------------------------------------------------------------------- #


async def test_re_attach_after_expiry_mints_a_new_session_with_a_fresh_grant(
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
    monkeypatch: pytest.MonkeyPatch,
    tenant_key: Any,
    jwks_ok: _FakeResponse,
) -> None:
    store = StoreGatewaySessionMemory()

    # Cycle 1: a short-lived grant, so the session's ceiling is close.
    first, _ = await _attach(
        config=config,
        secret_resolver=secret_resolver,
        store=store,
        monkeypatch=monkeypatch,
        tenant_key=tenant_key,
        jwks_ok=jwks_ok,
        expires_in_seconds=60,
    )

    # Cycle 2: the runtime re-grants against Keycloak (a fresh
    # client_credentials token, modelled here as a newly signed one with its
    # own exp) and attaches again. Same tenant, same edge instance, no
    # browser.
    second, _ = await _attach(
        config=config,
        secret_resolver=secret_resolver,
        store=store,
        monkeypatch=monkeypatch,
        tenant_key=tenant_key,
        jwks_ok=jwks_ok,
        expires_in_seconds=900,
    )

    # A successor session, not a mutated one.
    assert second.session.session_id != first.session.session_id
    assert second.session.expires_at > first.session.expires_at
    assert second.session.tenant_id == first.session.tenant_id
    assert second.session.edge_instance_id == first.session.edge_instance_id
    assert second.renewal.session_expires_at == second.session.expires_at

    # The predecessor is untouched -- keyed by session_id, both rows exist
    # independently, which is exactly what the merged gateway_sessions
    # projection's UPSERT on session_id alone requires.
    stored_first = await store.get(first.session.session_id)
    assert stored_first is not None
    assert_expiry_not_extended(first.session, stored_first)


async def test_re_attach_needs_no_browser_session_surface(
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
    monkeypatch: pytest.MonkeyPatch,
    tenant_key: Any,
    jwks_ok: _FakeResponse,
) -> None:
    """Structural, not behavioural: the only secret refs the cycle resolves
    are Keycloak's, and the only network surface faked is Keycloak's.

    A test that merely observed no browser call in one run would pass
    against an implementation that had a browser fallback it happened not to
    take. Asserting on the resolved dependency set proves the absence.
    """
    store = StoreGatewaySessionMemory()
    resolved: list[str] = []

    class _RecordingResolver(_FakeSecretResolver):
        async def get_secret_async(
            self,
            logical_name: str,
            required: bool = True,
            correlation_id: object = None,
        ) -> SecretStr:
            resolved.append(logical_name)
            return await super().get_secret_async(
                logical_name, required, correlation_id
            )

    recording = _RecordingResolver(dict(secret_resolver._values))

    for _ in range(2):
        await _attach(
            config=config,
            secret_resolver=recording,
            store=store,
            monkeypatch=monkeypatch,
            tenant_key=tenant_key,
            jwks_ok=jwks_ok,
            expires_in_seconds=900,
        )

    assert resolved, "attach resolved no secrets at all -- fixture is inert"
    assert all(name.startswith("gateway.attach.keycloak.") for name in resolved), (
        f"renewal cycle reached a non-Keycloak credential surface: {sorted(set(resolved))}"
    )


async def test_two_runtimes_on_one_tenant_get_two_sessions(
    config: ModelGatewayAttachConfig,
    secret_resolver: _FakeSecretResolver,
    monkeypatch: pytest.MonkeyPatch,
    tenant_key: Any,
    jwks_ok: _FakeResponse,
) -> None:
    """Session identity is session_id, never (tenant, principal).

    Two runtimes on one tenant is the expected steady state, and the merged
    projection keys on session_id alone precisely so one runtime's attach
    cannot evict the other's. Proven here at the node, where the row is
    born, because a projection cannot fix an identity the node collapsed.
    """
    store = StoreGatewaySessionMemory()
    _patch_client(monkeypatch, get_result=jwks_ok)
    handler = HandlerGatewayAttach(
        config=config,
        session_store=store,
        secret_resolver=secret_resolver,  # type: ignore[arg-type]
    )
    token = sign_claims(tenant_key, _claims(expires_in_seconds=900))

    first = await handler.handle(
        ModelGatewayAttachRequest(access_token=token, edge_instance_id="edge-a")
    )
    second = await handler.handle(
        ModelGatewayAttachRequest(access_token=token, edge_instance_id="edge-b")
    )

    assert first.session.session_id != second.session.session_id
    assert first.session.principal_id == second.session.principal_id
    assert await store.get(first.session.session_id) is not None
    assert await store.get(second.session.session_id) is not None


def test_renewal_directive_is_not_optional_on_the_attach_response() -> None:
    """A response that omits the cycle must not validate.

    Optionality here would let a build ship where unattended runtimes are
    told nothing about renewal and nothing fails -- the exact silent gap
    OMN-15952 was filed against.
    """
    now = datetime.now(UTC)
    session = ModelGatewaySession(
        session_id=uuid4(),
        tenant_id=TENANT_ID,
        tenant_slug="acme",
        principal_id="t-acme",
        keycloak_client_id="ga-tenant-acme",
        edge_instance_id=EDGE_INSTANCE_ID,
        status=EnumGatewaySessionStatus.ACTIVE,
        attached_at=now,
        last_heartbeat_at=now,
        expires_at=now + timedelta(seconds=900),
    )
    event = ModelGatewaySessionEvent(
        event_type=EnumGatewaySessionEventType.ATTACHED,
        session_id=session.session_id,
        tenant_id=session.tenant_id,
        tenant_slug=session.tenant_slug,
        principal_id=session.principal_id,
        edge_instance_id=session.edge_instance_id,
        emitted_at=now,
    )

    with pytest.raises(ValidationError):
        ModelGatewayAttachResponse(
            session=session,
            heartbeat_interval_seconds=15,
            session_event=event,
        )
