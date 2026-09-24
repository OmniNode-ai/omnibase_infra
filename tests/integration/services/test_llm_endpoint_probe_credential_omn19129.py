# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The endpoint prober authenticates over a real socket (OMN-19129).

The unit suite pins the prober by patching ``httpx.AsyncClient.get``.  This
module drives the real client against a real HTTP server on the loopback
interface, shaped like the GLM coding-plan surface measured from
``omninode-runtime-effects``:

* every path answers 401 when the request carries no bearer token, before it
  decides whether the path exists;
* ``/health`` and ``/v1/models`` answer 404 when authenticated;
* ``/models`` answers 200 for the right token and 401 for any other.

So the only thing that can turn the endpoint HEALTHY is the credential reaching
the wire on the declared path, and a genuinely rejected credential must still
classify ``AUTH_FAILED``.

Related Tickets:
    - OMN-19129: the prober authenticates and uses the declared path
    - OMN-16900: terminal auth classification
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import AsyncMock

import pytest

from omnibase_infra.models.health.enum_llm_endpoint_probe_state import (
    EnumLlmEndpointProbeState,
)
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.services.service_llm_endpoint_health import (
    ModelLlmEndpointHealthConfig,
    ServiceLlmEndpointHealth,
)

pytestmark = pytest.mark.integration

# Assembled at runtime so no credential-shaped literal exists in the source.
_GOOD_SECRET = "-".join(("integration", "probe", "credential"))
_BASE_PATH = "/api/coding/paas/v4"


class _SurfaceHandler(BaseHTTPRequestHandler):
    """A vendor surface that answers 401 before 404, like the measured one."""

    seen_paths: list[str] = []
    seen_auth: list[str | None] = []

    def do_GET(self) -> None:
        type(self).seen_paths.append(self.path)
        auth = self.headers.get("Authorization")
        type(self).seen_auth.append(auth)
        if auth is None:
            self._answer(401)
        elif self.path != f"{_BASE_PATH}/models":
            self._answer(404)
        elif auth == f"Bearer {_GOOD_SECRET}":
            self._answer(200, b'{"object":"list","data":[]}')
        else:
            self._answer(401)

    def _answer(self, code: int, body: bytes = b"{}") -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


@pytest.fixture
def surface() -> Iterator[str]:
    """Serve the surface on an ephemeral loopback port; yield its base URL."""
    _SurfaceHandler.seen_paths = []
    _SurfaceHandler.seen_auth = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SurfaceHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}{_BASE_PATH}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def event_bus() -> AsyncMock:
    bus = AsyncMock(spec=ProtocolEventBusLike)
    bus.publish_envelope = AsyncMock()
    return bus


def _config(base_url: str) -> ModelLlmEndpointHealthConfig:
    return ModelLlmEndpointHealthConfig(
        endpoints={"glm": base_url},
        endpoint_auth_env={"glm": "LLM_GLM_API_KEY"},
        endpoint_probe_paths={"glm": ("/models",)},
    )


def _resolver(value: str | None) -> Callable[[str], str | None]:
    def _resolve(_name: str) -> str | None:
        return value

    return _resolve


@pytest.mark.asyncio
async def test_a_good_credential_on_the_declared_path_is_healthy(
    surface: str, event_bus: AsyncMock
) -> None:
    service = ServiceLlmEndpointHealth(
        config=_config(surface),
        event_bus=event_bus,
        secret_resolver=_resolver(_GOOD_SECRET),
    )

    await service.probe_all()
    await service.probe_all()

    assert service.get_status()["glm"].probe_state is EnumLlmEndpointProbeState.HEALTHY
    assert set(_SurfaceHandler.seen_paths) == {f"{_BASE_PATH}/models"}, (
        "the prober reached a path the backend does not declare"
    )
    assert _SurfaceHandler.seen_auth
    assert all(a == f"Bearer {_GOOD_SECRET}" for a in _SurfaceHandler.seen_auth), (
        "a probe went out without the declared credential"
    )


@pytest.mark.asyncio
async def test_a_rejected_credential_still_classifies_auth_failed(
    surface: str, event_bus: AsyncMock
) -> None:
    service = ServiceLlmEndpointHealth(
        config=_config(surface),
        event_bus=event_bus,
        secret_resolver=_resolver("-".join(("wrong", "credential"))),
    )

    await service.probe_all()
    await service.probe_all()

    assert (
        service.get_status()["glm"].probe_state is EnumLlmEndpointProbeState.AUTH_FAILED
    )
    assert _SurfaceHandler.seen_auth
    assert all(a is not None for a in _SurfaceHandler.seen_auth), (
        "a rejection was measured on an anonymous probe, which proves nothing"
    )


def test_a_declared_credential_without_a_resolver_refuses_to_construct(
    event_bus: AsyncMock,
) -> None:
    with pytest.raises(ValueError, match="secret_resolver"):
        ServiceLlmEndpointHealth(
            config=_config("http://127.0.0.1:9/api"),
            event_bus=event_bus,
        )
