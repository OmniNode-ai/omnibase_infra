# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for runtime secret resolver config rendering."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
import yaml

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.models.model_secret_resolver_config import (
    ModelSecretResolverConfig,
)
from omnibase_infra.runtime.render_secret_resolver_config import (
    render_secret_resolver_config,
)

pytestmark = [pytest.mark.unit]


def _resolver_config_payload() -> dict[str, object]:
    return {
        "mappings": [
            {
                "logical_name": "llm.openrouter.api_key",
                "source": {
                    "source_type": "infisical",
                    "source_path": "OPENROUTER_API_KEY",
                },
            },
            {
                "logical_name": "llm.glm.api_key",
                "source": {
                    "source_type": "infisical",
                    "source_path": "LLM_GLM_API_KEY",
                },
            },
        ],
        "enable_convention_fallback": False,
    }


def _load_rendered(path: Path) -> ModelSecretResolverConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ModelSecretResolverConfig.model_validate(raw)


def test_render_secret_resolver_config_from_lane_overlay_json(tmp_path: Path) -> None:
    target = tmp_path / "delegation" / "secret_resolver.yaml"

    rendered = render_secret_resolver_config(
        target_path=target,
        environ={
            "ONEX_SECRET_RESOLVER_CONFIG_JSON": json.dumps(_resolver_config_payload()),
        },
    )

    assert rendered == target
    config = _load_rendered(target)
    assert config.enable_convention_fallback is False
    assert [mapping.logical_name for mapping in config.mappings] == [
        "llm.openrouter.api_key",
        "llm.glm.api_key",
    ]
    assert config.mappings[0].source.source_type == "infisical"
    assert config.mappings[0].source.source_path == "OPENROUTER_API_KEY"
    assert stat.S_IMODE(target.stat().st_mode) == stat.S_IRUSR | stat.S_IWUSR


def test_render_secret_resolver_config_from_source_file(tmp_path: Path) -> None:
    source = tmp_path / "source.yaml"
    target = tmp_path / "rendered.yaml"
    source.write_text(
        yaml.safe_dump(_resolver_config_payload(), sort_keys=False),
        encoding="utf-8",
    )

    rendered = render_secret_resolver_config(
        target_path=target,
        environ={"ONEX_SECRET_RESOLVER_SOURCE_CONFIG_PATH": str(source)},
    )

    assert rendered == target
    assert _load_rendered(target).mappings[0].logical_name == "llm.openrouter.api_key"


def test_render_secret_resolver_config_wraps_source_yaml_errors(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.yaml"
    source.write_text("mappings: [", encoding="utf-8")

    with pytest.raises(
        ProtocolConfigurationError,
        match="Secret resolver source config could not be loaded",
    ):
        render_secret_resolver_config(
            target_path=tmp_path / "rendered.yaml",
            environ={"ONEX_SECRET_RESOLVER_SOURCE_CONFIG_PATH": str(source)},
        )


def test_empty_secret_resolver_config_path_disables_render(tmp_path: Path) -> None:
    rendered = render_secret_resolver_config(
        target_path=None,
        environ={"ONEX_SECRET_RESOLVER_CONFIG_PATH": ""},
    )

    assert rendered is None
    assert not list(tmp_path.iterdir())


def test_secret_resolver_renderer_rejects_duplicate_logical_refs(
    tmp_path: Path,
) -> None:
    payload = _resolver_config_payload()
    mappings = payload["mappings"]
    assert isinstance(mappings, list)
    mappings.append(mappings[0])

    with pytest.raises(ProtocolConfigurationError, match="duplicate logical mappings"):
        render_secret_resolver_config(
            target_path=tmp_path / "secret_resolver.yaml",
            environ={"ONEX_SECRET_RESOLVER_CONFIG_JSON": json.dumps(payload)},
        )
