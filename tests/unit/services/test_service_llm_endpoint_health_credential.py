# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The endpoint prober must authenticate, and probe a path the surface serves.

OMN-16900 taught the health service to treat a sustained 401 as a terminal
credential failure rather than an outage.  It resolved each endpoint's
credential variable in order to decide whether the endpoint was worth probing
at all, and then dropped the value: the probe itself went out with no
``Authorization`` header.

On an auth-gated vendor surface that is a closed loop.

Measured against the GLM coding-plan endpoint from ``omninode-runtime-effects``:

* ``GET /health`` with no header -> 401, code 1001, "Authentication parameter
  not received in Header"
* ``GET /v1/models`` with no header -> 401, code 1001
* ``GET /health`` authenticated -> 404, the path does not exist
* ``GET /v1/models`` authenticated -> 404, the path does not exist
* ``GET /models`` authenticated -> 200, the model list
* ``POST /chat/completions`` authenticated -> 200, a completion with usage

The surface answers 401 *before* it answers 404, so an anonymous prober cannot
tell "this path is absent" from "this key is bad" and reports the second for
both.  Every backend whose credential resolved was therefore classified
``AUTH_FAILED`` on its second probe and removed itself from the routing ladder,
while the credential was fine the whole time.

This suite pins both halves of the fix and, just as importantly, pins that the
genuine case still works: a real rejection, with a credential attached, on a
path the surface serves, must still classify ``AUTH_FAILED``.

Related Tickets:
    - OMN-19129: this fix
    - OMN-19127: the GLM rung suppressed on the dev lane
    - OMN-16900: terminal auth classification and the SKIPPED_NO_AUTH partition
    - OMN-2255: original LLM endpoint health checker
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from omnibase_infra.models.health.enum_llm_endpoint_probe_state import (
    EnumLlmEndpointProbeState,
)
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.services.service_llm_endpoint_health import (
    ModelLlmEndpointHealthConfig,
    ServiceLlmEndpointHealth,
)

_GLM_BASE = "https://api.z.ai/api/coding/paas/v4"
# Assembled at runtime so no credential-shaped literal exists in the source.
_SENTINEL_SECRET = "-".join(("sentinel", "probe", "credential", "value"))


@pytest.fixture
def mock_event_bus() -> AsyncMock:
    """Return a mock ProtocolEventBusLike."""
    bus = AsyncMock(spec=ProtocolEventBusLike)
    bus.publish_envelope = AsyncMock()
    return bus


def _glm_config(**overrides: object) -> ModelLlmEndpointHealthConfig:
    """Build a config mirroring the live GLM wiring on the dev lane."""
    kwargs: dict[str, object] = {
        "endpoints": {"glm": _GLM_BASE},
        "endpoint_auth_env": {"glm": "LLM_GLM_API_KEY"},
        "endpoint_probe_paths": {"glm": ("/models",)},
    }
    kwargs.update(overrides)
    # Why: the factory is typed per-field; the spread is validated by pydantic.
    return ModelLlmEndpointHealthConfig(**kwargs)  # type: ignore[arg-type]


def _resolver(value: str | None) -> object:
    """Return a secret resolver that always answers with *value*."""

    def _resolve(_name: str) -> str | None:
        return value

    return _resolve


class TestProbeCarriesTheCredential:
    """AC1: an endpoint that declares a credential is probed with it."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_probe_sends_an_authorization_header(
        self,
        mock_event_bus: AsyncMock,
    ) -> None:
        """The declared credential reaches the wire as a bearer token."""
        seen: list[dict[str, str] | None] = []

        async def mock_get(url: str, **kwargs: object) -> httpx.Response:
            headers = kwargs.get("headers")
            # Why: the service passes a plain dict or omits the kwarg.
            seen.append(headers)  # type: ignore[arg-type]
            return httpx.Response(200, request=httpx.Request("GET", url))

        service = ServiceLlmEndpointHealth(
            config=_glm_config(),
            event_bus=mock_event_bus,
            secret_resolver=_resolver(_SENTINEL_SECRET),  # type: ignore[arg-type]
        )

        with patch.object(httpx.AsyncClient, "get", side_effect=mock_get):
            await service.probe_all()

        assert seen, "no probe was issued at all"
        assert seen[0] == {"Authorization": f"Bearer {_SENTINEL_SECRET}"}, (
            "the probe went out without the credential the service had "
            "already resolved to decide this endpoint was probeable"
        )
        assert (
            service.get_status()["glm"].probe_state is EnumLlmEndpointProbeState.HEALTHY
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_endpoint_without_a_declared_credential_sends_no_header(
        self,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Negative control: a local endpoint is not given a bearer token."""
        seen: list[dict[str, str] | None] = []

        async def mock_get(url: str, **kwargs: object) -> httpx.Response:
            seen.append(kwargs.get("headers"))  # type: ignore[arg-type]
            return httpx.Response(200, request=httpx.Request("GET", url))

        service = ServiceLlmEndpointHealth(
            config=ModelLlmEndpointHealthConfig(
                endpoints={"coder": "http://192.168.86.201:8000"},
            ),
            event_bus=mock_event_bus,
        )

        with patch.object(httpx.AsyncClient, "get", side_effect=mock_get):
            await service.probe_all()

        assert seen == [{}]

    @pytest.mark.unit
    def test_declared_credential_without_a_resolver_is_a_construction_error(
        self,
    ) -> None:
        """Wiring that would probe anonymously must fail loudly, not quietly.

        This is the exact shape of the OMN-16900 gap. Without this guard the
        service starts, probes without a header, and reports a rejected
        credential — a diagnosis that sends the operator to rotate a key that
        was never the problem.
        """
        with pytest.raises(ValueError, match="secret_resolver"):
            ServiceLlmEndpointHealth(config=_glm_config())


class TestDeclaredProbePath:
    """AC3: the probed path is the one the backend declares."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_declared_path_is_requested_and_synthesis_is_not(
        self,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Only ``/models`` is requested — never the synthesized 404 paths."""
        requested: list[str] = []

        async def mock_get(url: str, **kwargs: object) -> httpx.Response:
            requested.append(url)
            # Mirror the live surface: the synthesized paths do not exist.
            if url.endswith(("/health", "/v1/models")):
                return httpx.Response(404, request=httpx.Request("GET", url))
            return httpx.Response(200, request=httpx.Request("GET", url))

        service = ServiceLlmEndpointHealth(
            config=_glm_config(),
            event_bus=mock_event_bus,
            secret_resolver=_resolver(_SENTINEL_SECRET),  # type: ignore[arg-type]
        )

        with patch.object(httpx.AsyncClient, "get", side_effect=mock_get):
            await service.probe_all()

        assert requested == [f"{_GLM_BASE}/models"]
        assert (
            service.get_status()["glm"].probe_state is EnumLlmEndpointProbeState.HEALTHY
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_undeclared_endpoint_keeps_the_local_synthesis(
        self,
        mock_event_bus: AsyncMock,
    ) -> None:
        """A local vLLM server with no declaration still gets /health first."""
        requested: list[str] = []

        async def mock_get(url: str, **kwargs: object) -> httpx.Response:
            requested.append(url)
            return httpx.Response(200, request=httpx.Request("GET", url))

        service = ServiceLlmEndpointHealth(
            config=ModelLlmEndpointHealthConfig(
                endpoints={"coder": "http://192.168.86.201:8000"},
            ),
            event_bus=mock_event_bus,
        )

        with patch.object(httpx.AsyncClient, "get", side_effect=mock_get):
            await service.probe_all()

        assert requested == ["http://192.168.86.201:8000/health"]


class TestGenuineRejectionStillClassifies:
    """AC2: the fix must not make a real credential failure invisible."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_authenticated_401_on_the_served_path_is_auth_failed(
        self,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Positive control: a bad key, correctly sent, is still AUTH_FAILED.

        This is the case the whole classification exists for. The wrong key
        reaches the surface on the path the surface serves, and comes back
        rejected — that is a credential verdict and must stay one.
        """

        async def mock_get(url: str, **kwargs: object) -> httpx.Response:
            headers = kwargs.get("headers") or {}
            # Why: the mock inspects the header dict the service passed.
            token = headers.get("Authorization", "")  # type: ignore[union-attr]
            if token == "Bearer wrong-key":
                return httpx.Response(401, request=httpx.Request("GET", url))
            return httpx.Response(200, request=httpx.Request("GET", url))

        service = ServiceLlmEndpointHealth(
            config=_glm_config(auth_failure_threshold=2),
            event_bus=mock_event_bus,
            secret_resolver=_resolver("wrong-key"),  # type: ignore[arg-type]
        )

        with patch.object(httpx.AsyncClient, "get", side_effect=mock_get):
            await service.probe_all()
            await service.probe_all()

        status = service.get_status()["glm"]
        assert status.probe_state is EnumLlmEndpointProbeState.AUTH_FAILED
        assert status.available is False
        assert "401" in status.error

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_a_credential_that_stops_resolving_is_reported_by_name(
        self,
        mock_event_bus: AsyncMock,
    ) -> None:
        """A vanished secret must not degrade into an anonymous probe."""
        requested: list[str] = []

        async def mock_get(url: str, **kwargs: object) -> httpx.Response:
            requested.append(url)
            return httpx.Response(200, request=httpx.Request("GET", url))

        service = ServiceLlmEndpointHealth(
            config=_glm_config(),
            event_bus=mock_event_bus,
            secret_resolver=_resolver(None),  # type: ignore[arg-type]
        )

        with patch.object(httpx.AsyncClient, "get", side_effect=mock_get):
            await service.probe_all()

        status = service.get_status()["glm"]
        assert status.probe_state is EnumLlmEndpointProbeState.AUTH_FAILED
        assert "LLM_GLM_API_KEY" in status.error
        assert requested == [], "an unresolved credential was probed anonymously"


class TestCredentialNeverLeaks:
    """AC5: the value reaches httpx and nothing else."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_secret_is_absent_from_logs_status_and_events(
        self,
        mock_event_bus: AsyncMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Probe with a sentinel secret, then hunt for it everywhere."""

        async def mock_get(url: str, **kwargs: object) -> httpx.Response:
            return httpx.Response(401, request=httpx.Request("GET", url))

        service = ServiceLlmEndpointHealth(
            config=_glm_config(auth_failure_threshold=1),
            event_bus=mock_event_bus,
            secret_resolver=_resolver(_SENTINEL_SECRET),  # type: ignore[arg-type]
        )

        with (
            caplog.at_level(logging.DEBUG),
            patch.object(httpx.AsyncClient, "get", side_effect=mock_get),
        ):
            await service.probe_all()
            await service.probe_all()

        assert _SENTINEL_SECRET not in caplog.text
        status = service.get_status()["glm"]
        assert _SENTINEL_SECRET not in status.error
        assert _SENTINEL_SECRET not in str(status.model_dump())
        published = str(mock_event_bus.publish_envelope.call_args_list)
        assert _SENTINEL_SECRET not in published

    @pytest.mark.unit
    def test_config_repr_carries_variable_names_not_values(self) -> None:
        """The config holds the credential's NAME; there is no value to leak."""
        cfg = _glm_config()
        assert cfg.endpoint_auth_env == {"glm": "LLM_GLM_API_KEY"}
        assert _SENTINEL_SECRET not in repr(cfg)


class TestDeclarationValidation:
    """A declaration that names nothing probeable is a wiring typo."""

    @pytest.mark.unit
    def test_auth_declaration_for_an_unknown_endpoint_is_rejected(self) -> None:
        """Silently not applying a declaration is the bug, not the guard."""
        with pytest.raises(ValueError, match="endpoint_auth_env"):
            ModelLlmEndpointHealthConfig(
                endpoints={"coder": "http://192.168.86.201:8000"},
                endpoint_auth_env={"glm": "LLM_GLM_API_KEY"},
            )

    @pytest.mark.unit
    def test_probe_path_declaration_for_an_unknown_endpoint_is_rejected(
        self,
    ) -> None:
        """Same guard on the path map."""
        with pytest.raises(ValueError, match="endpoint_probe_paths"):
            ModelLlmEndpointHealthConfig(
                endpoints={"coder": "http://192.168.86.201:8000"},
                endpoint_probe_paths={"glm": ("/models",)},
            )

    @pytest.mark.unit
    def test_a_skipped_endpoint_may_not_carry_a_probe_declaration(self) -> None:
        """SKIPPED_NO_AUTH endpoints are never probed, so never declared."""
        with pytest.raises(ValueError, match="endpoint_probe_paths"):
            ModelLlmEndpointHealthConfig(
                endpoints={},
                unauthenticated_endpoints={"glm": _GLM_BASE},
                endpoint_probe_paths={"glm": ("/models",)},
            )
