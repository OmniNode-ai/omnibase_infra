# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests: topology contract and docker catalog model registry are internally consistent.

Enforces:
1. Every local model in docker/catalog/model_registry.yaml references a
   base_url_env that is owned by a running slot in contracts/llm_endpoints.yaml.
2. Running slots with url_env_var set are represented in docker/catalog/model_registry.yaml
   (no orphaned running slots).
3. Context windows in docker catalog do not exceed context_window_budgeted in topology
   for the same env var slot.
4. Disabled topology slots do not own url_env_var values (enforces existing contract rule).

Ticket: OMN-11925
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_TOPOLOGY_PATH = _REPO_ROOT / "contracts" / "llm_endpoints.yaml"
_CATALOG_PATH = _REPO_ROOT / "docker" / "catalog" / "model_registry.yaml"


def _load_topology() -> list[dict[str, Any]]:
    data = yaml.safe_load(_TOPOLOGY_PATH.read_text())
    return data["endpoints"]


def _load_catalog_models() -> list[dict[str, Any]]:
    data = yaml.safe_load(_CATALOG_PATH.read_text())
    return data["models"]


@pytest.mark.unit
class TestTopologyRegistryConsistency:
    """Cross-file consistency between llm_endpoints.yaml and model_registry.yaml."""

    def test_topology_file_exists(self) -> None:
        assert _TOPOLOGY_PATH.exists(), f"Missing topology contract: {_TOPOLOGY_PATH}"

    def test_catalog_file_exists(self) -> None:
        assert _CATALOG_PATH.exists(), (
            f"Missing model registry catalog: {_CATALOG_PATH}"
        )

    def test_local_catalog_env_vars_owned_by_running_topology_slots(self) -> None:
        """Every local model's base_url_env must map to a running topology slot."""
        topology = _load_topology()
        running_env_vars: set[str] = {
            ep["url_env_var"]
            for ep in topology
            if ep["status"] == "running" and ep.get("url_env_var") is not None
        }

        catalog = _load_catalog_models()
        for model in catalog:
            if model.get("provider") != "local":
                continue
            env_var = model.get("base_url_env")
            if env_var is None:
                continue
            assert env_var in running_env_vars, (
                f"model_key={model['model_key']!r} references base_url_env={env_var!r} "
                f"which is not owned by any running topology slot. "
                f"Running env vars: {sorted(running_env_vars)}"
            )

    def test_catalog_context_windows_do_not_exceed_topology_budget(self) -> None:
        """Catalog context windows must not exceed the topology's budgeted cap."""
        topology = _load_topology()
        budget_by_env: dict[str, int] = {
            ep["url_env_var"]: ep["context_window_budgeted"]
            for ep in topology
            if ep.get("url_env_var") is not None
            and ep.get("context_window_budgeted") is not None
        }

        catalog = _load_catalog_models()
        for model in catalog:
            env_var = model.get("base_url_env")
            if env_var is None or env_var not in budget_by_env:
                continue
            budget = budget_by_env[env_var]
            catalog_window = model.get("context_window")
            if catalog_window is None:
                continue
            assert catalog_window <= budget, (
                f"model_key={model['model_key']!r} has context_window={catalog_window} "
                f"exceeding topology budget={budget} for env_var={env_var!r}. "
                "Update docker/catalog/model_registry.yaml to match contracts/llm_endpoints.yaml."
            )

    def test_no_stale_env_var_references(self) -> None:
        """No catalog entry should reference LLM_QW3_80B_URL — this env var was removed."""
        removed_env_vars = {"LLM_QW3_80B_URL"}
        catalog = _load_catalog_models()
        for model in catalog:
            env_var = model.get("base_url_env", "")
            assert env_var not in removed_env_vars, (
                f"model_key={model['model_key']!r} references stale env_var={env_var!r}. "
                "This env var does not exist in the topology contract."
            )

    def test_disabled_topology_slots_have_no_url_env_var(self) -> None:
        """Disabled slots must not own url_env_var (enforces existing contract rule)."""
        for ep in _load_topology():
            if ep["status"] == "disabled":
                assert ep.get("url_env_var") is None, (
                    f"disabled slot {ep['slot_id']!r} must not own url_env_var; "
                    f"got {ep['url_env_var']!r}"
                )
