# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The endpoint prober never sends a request to the GLM Coding Plan (OMN-19127).

The GLM rung read 401 on every probe because the prober called z.ai's Coding
Plan endpoint directly. OMN-19129 settled the credential half with a completion
call (200: the key was fine) and fixed the prober's credential and path. The
Coding Plan terms bar calling the model API from our own services, so OMN-20173
made ``PluginLlm`` ignore a Coding Plan ``LLM_GLM_URL``. That guard sat in one
caller; ``ServiceLlmEndpointHealth`` still probed, with the bearer credential
attached, any Coding Plan URL a config handed it. The guard now lives in
``ModelLlmEndpointHealthConfig``, which every prober is built from, so no
caller can reopen the path. GLM is used only through Claude Code.

Related Tickets:
    - OMN-19127: the GLM rung health-suppressed on the dev lane
    - OMN-19129: the prober's credential and probe-path fix
    - OMN-20173: no direct caller of the Coding Plan
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import yaml

from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.services.service_llm_endpoint_health import (
    ModelLlmEndpointHealthConfig,
    ServiceLlmEndpointHealth,
)

_CODING_PLAN_BASE = "https://api.z.ai/api/coding/paas/v4"
_GENERAL_API_BASE = "https://api.z.ai/api/paas/v4"
_SENTINEL_SECRET = "-".join(("sentinel", "probe", "credential", "value"))


def _resolver(_name: str) -> str | None:
    return _SENTINEL_SECRET


def _service(endpoints: dict[str, str]) -> ServiceLlmEndpointHealth:
    config = ModelLlmEndpointHealthConfig(
        endpoints=endpoints,
        endpoint_auth_env=dict.fromkeys(endpoints, "LLM_GLM_API_KEY"),
        endpoint_probe_paths=dict.fromkeys(endpoints, ("/models",)),
    )
    return ServiceLlmEndpointHealth(
        config=config,
        event_bus=AsyncMock(spec=ProtocolEventBusLike),
        secret_resolver=_resolver,
    )


async def _probe(service: ServiceLlmEndpointHealth) -> list[str]:
    requested: list[str] = []

    async def mock_get(url: str, **kwargs: object) -> httpx.Response:
        requested.append(url)
        return httpx.Response(200, request=httpx.Request("GET", url))

    with patch.object(httpx.AsyncClient, "get", side_effect=mock_get):
        await service.probe_all()
    return requested


@pytest.mark.unit
@pytest.mark.asyncio
async def test_coding_plan_endpoint_is_never_requested() -> None:
    """No request, authenticated or not, reaches the Coding Plan surface."""
    requested = await _probe(_service({"glm": _CODING_PLAN_BASE}))

    assert requested == [], f"the prober called the Coding Plan: {requested}"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_coding_plan_endpoint_is_absent_from_the_status_map() -> None:
    """A skipped Coding Plan endpoint reports no state, as PluginLlm's ignore does."""
    service = _service({"glm": _CODING_PLAN_BASE})
    await _probe(service)

    assert "glm" not in service.get_status()
    assert service.circuit_breaker_names == ()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_other_endpoints_beside_a_coding_plan_endpoint_are_still_probed() -> None:
    """Positive control: the guard removes only the Coding Plan endpoint."""
    service = _service({"glm": _CODING_PLAN_BASE, "vendor": _GENERAL_API_BASE})
    requested = await _probe(service)

    assert requested == [f"{_GENERAL_API_BASE}/models"]
    assert set(service.get_status()) == {"vendor"}


_REGISTRY = Path(__file__).resolve().parents[3] / "docker/catalog/model_registry.yaml"


def _http_entries_with_a_credential(
    models: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        entry
        for entry in models
        if entry.get("transport") == "http"
        and entry.get("base_url_env")
        and entry.get("api_key_env")
    ]


@pytest.mark.unit
def test_every_credentialed_http_backend_declares_the_path_it_serves() -> None:
    """AC3: no auth-gated backend falls back to the synthesized /health probe.

    An auth-gated vendor answers 401 on a path it does not serve, which the
    prober reads as a rejected key. The catalog must name each such backend's
    served path (``probe_path``) so the synthesis never applies to it.
    """
    models = yaml.safe_load(_REGISTRY.read_text(encoding="utf-8"))["models"]
    credentialed = _http_entries_with_a_credential(models)

    assert credentialed, "positive control: the catalog has a credentialed http entry"
    undeclared = [e["model_key"] for e in credentialed if not e.get("probe_path")]
    assert undeclared == []


@pytest.mark.unit
def test_the_detector_flags_a_credentialed_backend_with_no_probe_path() -> None:
    """Failing control: the AC3 check sees a planted entry with no ``probe_path``."""
    planted = [
        {
            "model_key": "planted",
            "transport": "http",
            "base_url_env": "LLM_PLANTED_URL",
            "api_key_env": "PLANTED_API_KEY",
        }
    ]

    flagged = [
        e["model_key"]
        for e in _http_entries_with_a_credential(planted)
        if not e.get("probe_path")
    ]

    assert flagged == ["planted"]
