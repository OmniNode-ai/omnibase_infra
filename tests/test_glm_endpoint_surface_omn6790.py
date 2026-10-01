# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20173 retires OMN-6790's Coding Plan endpoint pins: Claude Code only."""

from pathlib import Path

import pytest
import yaml

from scripts import (
    check_llm_endpoint_env_contract,
    check_required_env_vars,
    validate_env,
)

pytestmark = pytest.mark.unit
_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("lane", ["dogfood", "judge", "lakshman", "infra"])
def test_glm_compose_url_defaults_to_empty(lane: str) -> None:
    source = (_ROOT / "docker" / f"docker-compose.{lane}.yml").read_text()
    assert "LLM_GLM_URL: ${LLM_GLM_URL:-}" in source


@pytest.mark.parametrize(
    "variable", ["LLM_GLM_URL", "LLM_GLM_MODEL_NAME", "LLM_GLM_API_KEY"]
)
def test_glm_variables_are_optional(variable: str) -> None:
    source = (_ROOT / "docker" / "docker-compose.infra.yml").read_text()
    assert f"{variable}: ${{{variable}:-}}" in source
    manifest = (_ROOT / "docker" / "required-env-vars.manifest.txt").read_text()
    assert variable not in manifest.splitlines()


def test_http_glm_catalog_entries_are_retired() -> None:
    data = yaml.safe_load(
        (_ROOT / "docker" / "catalog" / "model_registry.yaml").read_text()
    )
    models = {entry["model_key"]: entry for entry in data["models"]}
    assert not {"glm-4.5", "glm-5", "glm-5.1"} & models.keys()
    assert models["glm-5v-turbo"]["transport"] == "sdk"
    assert models["glm-5v-turbo"]["api_key_env"] == "ZHIPU_API_KEY"
    assert models["openrouter"]["transport"] == "http"


def test_glm_is_optional_in_env_validators(tmp_path: Path) -> None:
    """OMN-20173: empty GLM values remain valid across the existing gates."""
    assert not validate_env.validate({}).errors
    env = tmp_path / "empty.env"
    env.write_text("LLM_GLM_URL=\nLLM_GLM_MODEL_NAME=\nLLM_GLM_API_KEY=\n")
    assert check_llm_endpoint_env_contract.main(["--env-file", str(env)]) == 0
    assert (
        check_required_env_vars.main(
            [
                "--compose-file",
                str(_ROOT / "docker/docker-compose.infra.yml"),
                "--manifest-file",
                str(_ROOT / "docker/required-env-vars.manifest.txt"),
            ]
        )
        == 0
    )


def test_llm_catalog_readers_retire_glm(monkeypatch: pytest.MonkeyPatch) -> None:
    from omnibase_infra.services.routing_api import routes
    from omnibase_infra.services.service_llm_endpoint_health import (
        ModelLlmEndpointHealthConfig,
    )

    registry = _ROOT / "docker/catalog/model_registry.yaml"
    monkeypatch.setattr(routes, "_REGISTRY_PATH", registry)
    monkeypatch.setattr(routes, "_registry_cache", ())
    models = {entry.model_key for entry in routes._load_registry()}
    assert not {"glm-4.5", "glm-5", "glm-5.1"} & models
    assert {"openrouter", "glm-5v-turbo", "qwen3-coder-30b"} <= models
    config = ModelLlmEndpointHealthConfig.from_model_registry(
        registry_path=registry,
        env_resolver=lambda _: "https://example.org/v1",
    )
    assert not {"glm-4.5", "glm-5", "glm-5.1"} & config.endpoints.keys()
    assert {"openrouter", "qwen3-coder-30b"} <= config.endpoints.keys()
