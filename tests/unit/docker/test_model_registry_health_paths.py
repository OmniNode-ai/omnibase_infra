# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests: model_registry.yaml health_path entries for Phase 2 LLM models [OMN-8995]."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_REGISTRY_PATH = _PROJECT_ROOT / "docker" / "catalog" / "model_registry.yaml"

# Phase 2 primary and fallback model keys per OMN-8995 DoD.
# glm-4.5 is a cloud endpoint: health_path="" means always-healthy per HandlerModelRouter.
_PHASE2_HEALTH_PATHS: dict[str, str] = {
    "deepseek-r1-14b": "/health",
    "deepseek-r1-32b": "/health",
    "qwen3-coder-30b": "/health",
    "glm-4.5": "",
}


@pytest.fixture(scope="module")
def registry() -> list[dict[str, object]]:
    data: dict[str, list[dict[str, object]]] = yaml.safe_load(
        _REGISTRY_PATH.read_text()
    )
    return data["models"]


@pytest.mark.unit
def test_registry_file_exists() -> None:
    assert _REGISTRY_PATH.exists(), f"model_registry.yaml not found at {_REGISTRY_PATH}"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model_key", "expected_path"), list(_PHASE2_HEALTH_PATHS.items())
)
def test_phase2_model_has_health_path(
    registry: list[dict[str, object]],
    model_key: str,
    expected_path: str,
) -> None:
    by_key = {str(m["model_key"]): m for m in registry}
    assert model_key in by_key, (
        f"model_key '{model_key}' not found in model_registry.yaml"
    )
    entry = by_key[model_key]
    assert "health_path" in entry, f"'{model_key}' is missing health_path field"
    assert entry["health_path"] == expected_path, (
        f"'{model_key}' health_path={entry['health_path']!r}, expected {expected_path!r}"
    )
