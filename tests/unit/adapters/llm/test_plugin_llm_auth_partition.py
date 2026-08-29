# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""PluginLlm must not hand auth-dead endpoints to the health probe loop.

`PluginLlm.start_consumers` is the construction path that actually runs in the
`omninode-runtime` / `omninode-runtime-effects` containers, so the OMN-16900
classification only bites in production if this wiring passes the split through.
It derives *which* endpoints are auth-gated from the model registry contract
rather than a second hardcoded list.

Related Tickets:
    - OMN-16900: auth-state classification for LLM endpoint health probes
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from omnibase_infra.adapters.llm.plugin_llm import (
    _auth_env_by_url_env,
    _partition_endpoints_by_auth,
)


@pytest.fixture
def registry(tmp_path: Path) -> Path:
    """Write a registry with one auth-gated and one open HTTP endpoint."""
    path = tmp_path / "model_registry.yaml"
    path.write_text(
        textwrap.dedent(
            """
            models:
              - model_key: qwen3-coder-30b
                provider: local
                transport: http
                base_url_env: LLM_CODER_URL
              - model_key: glm-4.5
                provider: zhipu
                transport: http
                base_url_env: LLM_GLM_URL
                api_key_env: LLM_GLM_API_KEY
              - model_key: glm-5v-turbo
                provider: zhipu
                transport: sdk
                api_key_env: ZHIPU_API_KEY
            """
        ).strip(),
        encoding="utf-8",
    )
    return path


@pytest.mark.unit
def test_auth_env_map_is_derived_from_the_registry(registry: Path) -> None:
    """Only HTTP entries that declare an api_key_env are auth-gated."""
    assert _auth_env_by_url_env(registry) == {"LLM_GLM_URL": "LLM_GLM_API_KEY"}


@pytest.mark.unit
def test_missing_registry_degrades_to_no_auth_requirements(tmp_path: Path) -> None:
    """A pip-installed layout without docker/ must not crash plugin startup."""
    assert _auth_env_by_url_env(tmp_path / "absent.yaml") == {}


@pytest.mark.unit
def test_malformed_registry_degrades_to_no_auth_requirements(tmp_path: Path) -> None:
    """A registry with no 'models' list is reported, not raised through."""
    path = tmp_path / "model_registry.yaml"
    path.write_text("not-a-mapping", encoding="utf-8")
    assert _auth_env_by_url_env(path) == {}


_ENDPOINTS = {
    "LLM_CODER_URL": "http://192.168.86.201:8000",
    "LLM_GLM_URL": "https://api.z.ai/api/coding/paas/v4",
}
_AUTH_MAP = {"LLM_GLM_URL": "LLM_GLM_API_KEY"}


@pytest.mark.unit
def test_partition_routes_unresolved_secret_out_of_the_probe_set() -> None:
    """An overlay that does not resolve the key marks the endpoint auth-dead."""
    probeable, unauthenticated = _partition_endpoints_by_auth(
        endpoints=_ENDPOINTS,
        auth_env_by_url_env=_AUTH_MAP,
        resolved_config={"SOMETHING_ELSE": "x"},
    )

    assert probeable == {"coder": "http://192.168.86.201:8000"}
    assert unauthenticated == {"glm": "https://api.z.ai/api/coding/paas/v4"}


@pytest.mark.unit
def test_partition_keeps_endpoint_probeable_when_secret_resolves() -> None:
    """Negative control: a resolved key leaves the endpoint probeable."""
    probeable, unauthenticated = _partition_endpoints_by_auth(
        endpoints=_ENDPOINTS,
        auth_env_by_url_env=_AUTH_MAP,
        resolved_config={"LLM_GLM_API_KEY": "a-real-key"},
    )

    assert set(probeable) == {"coder", "glm"}
    assert unauthenticated == {}


@pytest.mark.unit
def test_partition_classifies_nothing_without_a_resolved_overlay() -> None:
    """Legacy env-var boot has no authoritative secret view — classify nothing.

    Guessing "absent" from an unloaded overlay would silently stop probing a
    perfectly healthy endpoint. The service's terminal-AUTH_FAILED backoff is
    the layer that handles this case instead.
    """
    probeable, unauthenticated = _partition_endpoints_by_auth(
        endpoints=_ENDPOINTS,
        auth_env_by_url_env=_AUTH_MAP,
        resolved_config=None,
    )

    assert set(probeable) == {"coder", "glm"}
    assert unauthenticated == {}


@pytest.mark.unit
def test_live_registry_declares_glm_as_auth_gated() -> None:
    """Guard the live contract: the GLM entries must stay auth-gated.

    This is the entry that produced 5+ days of 401s on .201. If a future
    catalog edit drops `api_key_env` from the zhipu entries, the health service
    silently returns to hammering them, so pin it here. Read-only — this test
    never writes to the catalog (OMN-16442 owns that surface).
    """
    live_registry = (
        Path(__file__).parents[4] / "docker" / "catalog" / "model_registry.yaml"
    )
    if not live_registry.exists():  # pragma: no cover - clone-only path
        pytest.skip("operational docker/ tree not present in this layout")

    assert _auth_env_by_url_env(live_registry).get("LLM_GLM_URL") == "LLM_GLM_API_KEY"
