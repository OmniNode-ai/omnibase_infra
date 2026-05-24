# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Inspection tests for ``contracts/llm_endpoints.yaml`` (plan Task 4).

Enforces stable-canonical + operator-annotation schema, required slot_ids,
reasoning-moe-35b fields, and a closed role taxonomy. Planned slots may
have null host/port/endpoint_url/model_hf_id. See OMN-9292 / OMN-9294.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_CONTRACT_PATH = _REPO_ROOT / "contracts" / "llm_endpoints.yaml"

_STABLE_CANONICAL_KEYS: frozenset[str] = frozenset(
    [
        "slot_id",
        "host",
        "port",
        "endpoint_url",
        "url_env_var",
        "role_env_alias",
        "model_hf_id",
        "role",
        "status",
    ]
)
_OPERATOR_ANNOTATION_KEYS: frozenset[str] = frozenset(
    ["hardware", "context_window_budgeted", "launchd_unit_or_none", "notes"]
)
_REQUIRED_SLOT_IDS: frozenset[str] = frozenset(
    [
        "coder-5090",
        "coder-fast-4090",
        "embeddings-201",
        "reasoning-deepseek-32b",
        "reasoning-moe-35b",
        "embeddings-200",
        "second-opinion-gemma",
        "vision-planned",
        "stt-planned",
        "tts-planned",
        "reranker-planned",
    ]
)
_ROLE_TAXONOMY: frozenset[str] = frozenset(
    [
        "coder_slow",
        "coder_fast",
        "embedding",
        "reasoning",
        "reasoning_fast",
        "reasoning_lightweight",
        "reasoning_transient",
        "vision",
        "stt",
        "tts",
        "reranker",
    ]
)
_PLANNED_NULLABLE_FIELDS: frozenset[str] = frozenset(
    ["host", "port", "endpoint_url", "model_hf_id"]
)
_RUNTIME_REQUIRED_URL_ENV_VARS: frozenset[str] = frozenset(
    [
        "LLM_CODER_URL",
        "LLM_CODER_FAST_URL",
        "LLM_EMBEDDING_URL",
        "LLM_DEEPSEEK_R1_URL",
    ]
)
_SUPPORTED_TOPOLOGY_FIELDS: frozenset[str] = frozenset(
    [
        "role",
        "status",
        "endpoint_url",
        "model_hf_id",
        "url_env_var",
        "role_env_alias",
        "launchd_unit_or_none",
        "context_window_budgeted",
    ]
)


def _load_endpoints() -> list[dict[str, Any]]:
    assert _CONTRACT_PATH.exists(), f"Missing contract file: {_CONTRACT_PATH}"
    data = yaml.safe_load(_CONTRACT_PATH.read_text())
    assert isinstance(data, dict) and "endpoints" in data, "Need top-level 'endpoints'"
    endpoints = data["endpoints"]
    assert isinstance(endpoints, list) and endpoints, "'endpoints' must be non-empty"
    return endpoints


@pytest.mark.unit
class TestLlmEndpointsContract:
    """Schema inspection for the canonical LLM topology contract."""

    def test_schema_and_required_fields(self) -> None:
        expected = _STABLE_CANONICAL_KEYS | _OPERATOR_ANNOTATION_KEYS
        for ep in _load_endpoints():
            assert expected.issubset(ep.keys()), (
                f"Entry {ep.get('slot_id')!r} missing keys: {expected - ep.keys()}"
            )
            for key in ("slot_id", "role", "status"):
                assert ep.get(key), f"Entry {ep.get('slot_id')!r} has empty {key}"

    def test_running_slots_have_core_fields_non_null(self) -> None:
        """Running slots must have host/port/endpoint_url/model_hf_id non-null.

        url_env_var and role_env_alias are allowed to be null on running slots
        (Docker-internal slots have no external env-var; aliases may be unassigned).
        disabled, on_demand, and planned slots are explicitly exempt.
        """
        for slot in _load_endpoints():
            if slot["status"] == "running":
                for key in _PLANNED_NULLABLE_FIELDS:
                    assert slot[key] is not None, (
                        f"running slot {slot['slot_id']!r} must have non-null {key}"
                    )

    def test_supported_topology_fields_are_declared_on_every_slot(self) -> None:
        """Real YAML carries the convergence fields supported by today's schema."""
        for slot in _load_endpoints():
            missing = _SUPPORTED_TOPOLOGY_FIELDS - slot.keys()
            assert not missing, (
                f"slot {slot.get('slot_id')!r} missing supported topology fields: "
                f"{sorted(missing)}"
            )

            assert slot["role"], f"slot {slot['slot_id']!r} must declare endpoint role"
            assert slot["status"], (
                f"slot {slot['slot_id']!r} must declare deployment status"
            )

            if slot["endpoint_url"] is not None:
                parsed = urlparse(slot["endpoint_url"])
                assert parsed.scheme in {"http", "https"}, (
                    f"slot {slot['slot_id']!r} endpoint_url must be http(s)"
                )
                assert parsed.hostname == slot["host"], (
                    f"slot {slot['slot_id']!r} endpoint_url host must match host"
                )
                assert parsed.port == slot["port"], (
                    f"slot {slot['slot_id']!r} endpoint_url port must match port"
                )
                assert parsed.path in {"", "/"}, (
                    f"slot {slot['slot_id']!r} endpoint_url must be a base URL"
                )
                assert not parsed.query and not parsed.fragment, (
                    f"slot {slot['slot_id']!r} endpoint_url must not carry query/fragment"
                )

            if slot["model_hf_id"] is not None:
                assert "/" in slot["model_hf_id"], (
                    f"slot {slot['slot_id']!r} model_hf_id must include namespace/model"
                )

            if slot["launchd_unit_or_none"] is not None:
                assert slot["launchd_unit_or_none"].startswith("com."), (
                    f"slot {slot['slot_id']!r} launchd unit must be a launchd label"
                )

            if slot["context_window_budgeted"] is not None:
                assert slot["context_window_budgeted"] > 0, (
                    f"slot {slot['slot_id']!r} context window budget must be positive"
                )

    def test_endpoint_alias_fields_are_canonical_when_present(self) -> None:
        for slot in _load_endpoints():
            for key in ("url_env_var", "role_env_alias"):
                value = slot.get(key)
                if value is None:
                    continue
                assert value.startswith("LLM_") and value.endswith("_URL"), (
                    f"slot {slot['slot_id']!r} {key} must use LLM_*_URL naming"
                )

    def test_required_slot_ids_present(self) -> None:
        present = {ep["slot_id"] for ep in _load_endpoints()}
        missing = _REQUIRED_SLOT_IDS - present
        assert not missing, f"Missing required slot_ids: {sorted(missing)}"

    def test_role_values_are_closed_taxonomy(self) -> None:
        for ep in _load_endpoints():
            assert ep["role"] in _ROLE_TAXONOMY, (
                f"Entry {ep['slot_id']!r} role {ep['role']!r} outside taxonomy"
            )

    def test_reasoning_moe_35b_fields(self) -> None:
        by_slot = {ep["slot_id"]: ep for ep in _load_endpoints()}
        ep = by_slot.get("reasoning-moe-35b")
        assert ep is not None, "Missing required slot 'reasoning-moe-35b'"
        assert ep["model_hf_id"] == "mlx-community/Qwen3.6-35B-A3B-8bit"
        assert ep["endpoint_url"] == "http://192.168.86.200:8102"
        assert ep["url_env_var"] == "LLM_QWEN3_NEXT_URL"
        assert ep["role_env_alias"] == "LLM_REASONING_FAST_URL"
        assert ep["role"] == "reasoning_fast"

    def test_runtime_required_env_vars_are_owned_by_running_slots(self) -> None:
        """Runtime-required URL env vars must point at running canonical slots."""
        by_env = {
            ep["url_env_var"]: ep
            for ep in _load_endpoints()
            if ep.get("url_env_var") is not None
        }

        for env_var in _RUNTIME_REQUIRED_URL_ENV_VARS:
            ep = by_env.get(env_var)
            assert ep is not None, f"{env_var} is not assigned to any endpoint slot"
            assert ep["status"] == "running", (
                f"{env_var} is assigned to non-running slot {ep['slot_id']!r}"
            )

    def test_disabled_slots_do_not_own_runtime_url_env_vars(self) -> None:
        for ep in _load_endpoints():
            if ep["status"] == "disabled":
                assert ep["url_env_var"] is None, (
                    f"disabled slot {ep['slot_id']!r} must not own url_env_var"
                )
