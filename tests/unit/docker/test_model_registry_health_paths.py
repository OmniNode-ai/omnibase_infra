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
#
# OMN-16442 (2026-08-28): "deepseek-r1-14b" REMOVED from this expectation set.
# Its entry was deleted from docker/catalog/model_registry.yaml because its
# base_url_env (LLM_CODER_FAST_URL) pointed at .201:8001 — the RTX 4090 slot
# physically removed from the host for RMA (OMN-16407). Live re-probe: `curl
# http://192.168.86.201:8001/v1/models` -> exit 7 "Couldn't connect to server".
# A health_path assertion on a model row that must not exist would force the
# dead row to be kept purely to satisfy a test.
_PHASE2_HEALTH_PATHS: dict[str, str] = {
    "deepseek-r1-32b": "/health",
    "qwen3-coder-30b": "/health",
    "glm-4.5": "",
}

# OMN-16442: model_keys whose endpoint no longer exists on the fleet. Asserted
# ABSENT so a future edit cannot reintroduce a row pointing at dead hardware.
_RETIRED_MODEL_KEYS: frozenset[str] = frozenset({"deepseek-r1-14b", "qwen3-next-80b"})


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


@pytest.mark.unit
def test_retired_model_keys_are_absent(registry: list[dict[str, object]]) -> None:
    """OMN-16442: no catalog row may name a decommissioned endpoint.

    * ``deepseek-r1-14b`` -> LLM_CODER_FAST_URL -> .201:8001, RTX 4090 pulled
      for RMA (OMN-16407).
    * ``qwen3-next-80b``  -> LLM_QWEN3_NEXT_URL -> .200:8102, no listener.

    Both endpoints were re-probed 2026-08-28 and return curl exit 7 "Couldn't
    connect to server"; contracts/llm_endpoints.yaml marks both slots
    ``disabled``, and a disabled slot owns no url_env_var — so a catalog row
    referencing one would also break
    tests/unit/contracts/test_topology_registry_consistency.py.
    """
    present = {model["model_key"] for model in registry}
    assert not (present & _RETIRED_MODEL_KEYS), (
        "docker/catalog/model_registry.yaml names a retired model_key: "
        f"{sorted(present & _RETIRED_MODEL_KEYS)}"
    )
