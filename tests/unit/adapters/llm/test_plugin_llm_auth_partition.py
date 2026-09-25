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
    - OMN-19129: the probe must carry the credential it classified against
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from omnibase_infra.adapters.llm.plugin_llm import (
    _partition_endpoints_by_auth,
    _registry_declarations,
)


def _resolver(values: dict[str, str]):
    """Return a kernel-style credential resolver over a fixed mapping."""

    def _resolve(name: str) -> str | None:
        return values.get(name)

    return _resolve


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
                probe_path: /health
              - model_key: glm-4.5
                provider: zhipu
                transport: http
                base_url_env: LLM_GLM_URL
                api_key_env: LLM_GLM_API_KEY
                probe_path: /models
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
    auth_env, _ = _registry_declarations(registry)
    assert auth_env == {"LLM_GLM_URL": "LLM_GLM_API_KEY"}


@pytest.mark.unit
def test_probe_paths_are_derived_from_the_registry(registry: Path) -> None:
    """OMN-19129: each HTTP entry's declared probe path reaches the plugin."""
    _, probe_paths = _registry_declarations(registry)
    assert probe_paths == {
        "LLM_CODER_URL": "/health",
        "LLM_GLM_URL": "/models",
    }


@pytest.mark.unit
def test_missing_registry_degrades_to_no_auth_requirements(tmp_path: Path) -> None:
    """A pip-installed layout without docker/ must not crash plugin startup."""
    assert _registry_declarations(tmp_path / "absent.yaml") == ({}, {})


@pytest.mark.unit
def test_malformed_registry_degrades_to_no_auth_requirements(tmp_path: Path) -> None:
    """A registry with no 'models' list is reported, not raised through."""
    path = tmp_path / "model_registry.yaml"
    path.write_text("not-a-mapping", encoding="utf-8")
    assert _registry_declarations(path) == ({}, {})


_ENDPOINTS = {
    "LLM_CODER_URL": "http://192.168.86.201:8000",
    "LLM_GLM_URL": "https://api.z.ai/api/coding/paas/v4",
}
_AUTH_MAP = {"LLM_GLM_URL": "LLM_GLM_API_KEY"}


@pytest.mark.unit
def test_partition_routes_unresolved_secret_out_of_the_probe_set() -> None:
    """An overlay that does not resolve the key marks the endpoint auth-dead."""
    probeable, unauthenticated, auth_env = _partition_endpoints_by_auth(
        endpoints=_ENDPOINTS,
        auth_env_by_url_env=_AUTH_MAP,
        secret_resolver=_resolver({"SOMETHING_ELSE": "x"}),
    )

    assert probeable == {"coder": "http://192.168.86.201:8000"}
    assert unauthenticated == {"glm": "https://api.z.ai/api/coding/paas/v4"}
    # An endpoint that is never probed needs no credential wiring.
    assert auth_env == {}


@pytest.mark.unit
def test_partition_keeps_endpoint_probeable_when_secret_resolves() -> None:
    """Negative control: a resolved key leaves the endpoint probeable."""
    probeable, unauthenticated, auth_env = _partition_endpoints_by_auth(
        endpoints=_ENDPOINTS,
        auth_env_by_url_env=_AUTH_MAP,
        secret_resolver=_resolver({"LLM_GLM_API_KEY": "resolved"}),
    )

    assert set(probeable) == {"coder", "glm"}
    assert unauthenticated == {}
    # OMN-19129: the credential's NAME must reach the probe, or the endpoint
    # is probed anonymously and 401s forever on a key that is perfectly good.
    assert auth_env == {"glm": "LLM_GLM_API_KEY"}


@pytest.mark.unit
def test_partition_classifies_auth_gated_endpoints_dead_without_a_resolver() -> None:
    """A missing resolver is a wiring gap, never a licence to probe anonymously.

    This inverts the pre-OMN-19129 expectation. The old behaviour left the
    endpoint probeable and relied on the service's terminal-AUTH_FAILED
    backoff to notice, but that backoff was reacting to a 401 the probe caused
    itself by sending no credential. Declining to probe is the honest answer:
    with no way to authenticate, no probe can tell us anything.
    """
    probeable, unauthenticated, auth_env = _partition_endpoints_by_auth(
        endpoints=_ENDPOINTS,
        auth_env_by_url_env=_AUTH_MAP,
        secret_resolver=None,
    )

    assert probeable == {"coder": "http://192.168.86.201:8000"}
    assert unauthenticated == {"glm": "https://api.z.ai/api/coding/paas/v4"}
    assert auth_env == {}


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

    auth_env, probe_paths = _registry_declarations(live_registry)
    assert auth_env.get("LLM_GLM_URL") == "LLM_GLM_API_KEY"
    # OMN-19129: and it must declare a path the coding-plan surface serves.
    # Measured live 2026-09-21: authenticated GET .../v4/models -> 200, while
    # .../v4/health and .../v4/v1/models -> 404.
    assert probe_paths.get("LLM_GLM_URL") == "/models"


@pytest.mark.unit
def test_every_http_registry_endpoint_declares_a_probe_path() -> None:
    """OMN-19129 AC3: no backend is probed on a synthesized guess.

    The prober used to append /health and /v1/models to every base URL. That
    is right for a local vLLM server and wrong for a vendor surface, and the
    404s it produced were indistinguishable from a rejected credential once
    the request went out unauthenticated.
    """
    live_registry = (
        Path(__file__).parents[4] / "docker" / "catalog" / "model_registry.yaml"
    )
    if not live_registry.exists():  # pragma: no cover - clone-only path
        pytest.skip("operational docker/ tree not present in this layout")

    import yaml

    raw = yaml.safe_load(live_registry.read_text(encoding="utf-8"))
    http_url_envs = {
        str(entry["base_url_env"])
        for entry in raw["models"]
        if entry.get("transport") == "http" and entry.get("base_url_env")
    }
    _, probe_paths = _registry_declarations(live_registry)

    missing = sorted(http_url_envs - set(probe_paths))
    assert not missing, (
        f"http-transport entries without a declared probe_path: {missing}. "
        "An endpoint with no declaration falls back to synthesized paths."
    )
