# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17423 AC3: the gateway log stream is greppable, and carries no credential.

AC3 asks the AC1 probe to find zero credential sentinels across the collected
service logs. The 2026-09-25 re-run returned zero on all six surfaces, and only
``onex-api`` could admit that as evidence -- it carried a positive control. The
gateway lifecycle, which is the credential-carrying path, logged nothing at all
on its success paths, so a zero there distinguished a redacted log from a silent
one not at all.

The unit tests in ``tests/unit/nodes/node_gateway_attach_effect/test_handlers.py``
assert on ``LogRecord`` objects. This one asserts on the **rendered stream** --
the bytes a ``kubectl logs ... | grep`` actually reads -- with the two mechanisms
that have to hold together for AC3 to mean anything both in place:

* AC2's shared ``CredentialRedactionFilter``, installed on the stream handler by
  ``install_credential_redaction_filter()``, the same call
  ``service_kernel.configure_logging()`` makes in the runtime bootstrap; and
* AC3's lifecycle markers, emitted by the three handlers.

The assertion is therefore the AC3 grep itself, run against a real formatted
stream rather than against a probe's recollection of one. Its positive control
is the session id: without a located traversal, a zero on the token search is a
statement about silence.
"""

from __future__ import annotations

import io
import logging
import time
from typing import Any
from uuid import UUID

import httpx
import pytest
from pydantic import SecretStr

from omnibase_infra.nodes.node_gateway_attach_effect.handlers.handler_gateway_attach import (
    HandlerGatewayAttach,
)
from omnibase_infra.nodes.node_gateway_attach_effect.handlers.handler_gateway_detach import (
    HandlerGatewayDetach,
)
from omnibase_infra.nodes.node_gateway_attach_effect.handlers.handler_gateway_heartbeat import (
    HandlerGatewayHeartbeat,
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
from omnibase_infra.nodes.node_gateway_attach_effect.services.store_gateway_session_memory import (
    StoreGatewaySessionMemory,
)
from omnibase_infra.utils.util_log_credential_redaction import (
    install_credential_redaction_filter,
)
from tests.unit.nodes.node_gateway_attach_effect._jwt_test_support import (
    TENANT_KID,
    generate_key_material,
    jwks_response_body,
    sign_claims,
)

pytestmark = pytest.mark.integration

TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")
ISSUER = "https://keycloak.example/realms/omninode"
CLIENT_ID = "gw-tenant-acme"
EDGE_INSTANCE_ID = "edge-201"


class _Resolver:
    def __init__(self, values: dict[str, str]) -> None:
        self._values = values

    async def get_secret_async(
        self, logical_name: str, required: bool = True, correlation_id: object = None
    ) -> SecretStr:
        return SecretStr(self._values[logical_name])


class _Response:
    def __init__(self, status_code: int, body: Any) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> Any:
        return self._body


class _Client:
    def __init__(self, get_result: Any, post_result: Any) -> None:
        self._get_result = get_result
        self._post_result = post_result

    async def __aenter__(self) -> _Client:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def get(self, *args: object, **kwargs: object) -> Any:
        return self._get_result

    async def post(self, *args: object, **kwargs: object) -> Any:
        return self._post_result


def _claims() -> dict[str, object]:
    issued_at = int(time.time())
    return {
        "iss": ISSUER,
        "sub": "svc-acct-abc",
        "aud": "gateway-attach",
        "tenant_id": str(TENANT_ID),
        "tenant_slug": "acme",
        "principal_id": "t-11111111111111111111111111111111",
        "azp": CLIENT_ID,
        "iat": issued_at,
        "exp": issued_at + 900,
    }


def _all_handlers() -> list[logging.Handler]:
    """Every handler `install_credential_redaction_filter` can reach."""
    loggers: list[logging.Logger] = [logging.getLogger()]
    loggers += [
        obj
        for obj in logging.getLogger().manager.loggerDict.values()
        if isinstance(obj, logging.Logger)
    ]
    return [handler for logger in loggers for handler in logger.handlers]


@pytest.fixture
def gateway_log_stream() -> Any:
    """A real stream handler on the root logger, filtered exactly as production.

    ``install_credential_redaction_filter()`` is AC2's mechanism and is what
    ``service_kernel.configure_logging()`` calls in the runtime bootstrap, so
    the fixture calls it rather than attaching a filter by hand -- the point of
    this test is that the shipped mechanism and the new lifecycle lines compose.

    Teardown has to undo ALL of it, not just this handler. The installer walks
    every reachable logger and filters every handler it finds, pytest's own
    capture handlers included; removing only the handler added here would leave
    a redaction filter attached to the session's shared handlers and silently
    rewrite later tests' log assertions. So the exact filter objects this call
    adds are recorded per handler and removed again.
    """
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root = logging.getLogger()
    previous_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    before = {id(h): list(h.filters) for h in _all_handlers()}
    installed = install_credential_redaction_filter()
    assert installed >= 1, "AC2's redaction filter did not attach to any handler"
    added = [
        (h, f)
        for h in _all_handlers()
        for f in h.filters
        if f not in before.get(id(h), [])
    ]
    try:
        yield buffer
    finally:
        for target, added_filter in added:
            target.removeFilter(added_filter)
        root.removeHandler(handler)
        root.setLevel(previous_level)


async def test_gateway_lifecycle_stream_is_greppable_and_credential_free(
    monkeypatch: pytest.MonkeyPatch,
    gateway_log_stream: io.StringIO,
) -> None:
    config = ModelGatewayAttachConfig()
    key = generate_key_material(TENANT_KID)
    resolver = _Resolver(
        {
            config.keycloak_issuer_ref: ISSUER,
            config.keycloak_introspection_ref: "https://keycloak.example/introspect",
            config.keycloak_jwks_ref: "https://keycloak.example/jwks",
            f"{config.keycloak_admin_client_ref}.client_id": "admin-cli",
            f"{config.keycloak_admin_client_ref}.client_secret": "admin-secret",
        }
    )
    store = StoreGatewaySessionMemory()
    jwks = _Response(200, jwks_response_body(key))
    active = _Response(200, {"active": True, "client_id": CLIENT_ID})
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_: _Client(jwks, active))

    attach_token = sign_claims(key, _claims())
    attach_response = await HandlerGatewayAttach(
        config=config,
        session_store=store,
        secret_resolver=resolver,  # type: ignore[arg-type]
    ).handle(
        ModelGatewayAttachRequest(
            access_token=attach_token, edge_instance_id=EDGE_INSTANCE_ID
        )
    )
    session_id = attach_response.session.session_id

    heartbeat_token = sign_claims(key, _claims())
    await HandlerGatewayHeartbeat(
        config=config,
        session_store=store,
        secret_resolver=resolver,  # type: ignore[arg-type]
    ).handle(
        ModelGatewayHeartbeatRequest(
            session_id=session_id, access_token=heartbeat_token
        )
    )

    detach_token = sign_claims(key, _claims())
    await HandlerGatewayDetach(
        config=config,
        session_store=store,
        secret_resolver=resolver,  # type: ignore[arg-type]
    ).handle(
        ModelGatewayDetachRequest(
            session_id=session_id,
            access_token=detach_token,
            reason="edge shutdown",
        )
    )

    stream = gateway_log_stream.getvalue()

    # Positive control -- the AC3 probe can LOCATE this traversal on this
    # surface. Without it, the credential search below proves only silence.
    assert stream.count(str(session_id)) == 3, (
        "expected one located lifecycle line per transition; "
        f"got {stream.count(str(session_id))} in:\n{stream}"
    )
    for marker in ("attached", "heartbeat", "detached"):
        assert f"gateway session {marker}" in stream

    # The AC3 assertion, run as the grep it stands for.
    for name, token in (
        ("attach", attach_token),
        ("heartbeat", heartbeat_token),
        ("detach", detach_token),
    ):
        assert token not in stream, f"{name} bearer token reached the log stream"
        for segment in token.split("."):
            if len(segment) > 16:
                assert segment not in stream, f"{name} token segment reached the stream"
