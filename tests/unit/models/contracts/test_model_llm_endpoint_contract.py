# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ModelLlmEndpointContract + ModelLlmEndpointEntry.

Related tickets:
    - OMN-9750: typed ModelLlmEndpointContract + loader (no runtime cutover)
    - OMN-9738: parent epic — contract-model validation
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from omnibase_infra.models.contracts.model_llm_endpoint_contract import (
    ModelLlmEndpointContract,
    ModelLlmEndpointEntry,
    load_llm_endpoint_contract,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CONTRACTS_DIR = Path(__file__).parent.parent.parent.parent.parent / "contracts"
_LLM_ENDPOINTS_YAML = _CONTRACTS_DIR / "llm_endpoints.yaml"


# ---------------------------------------------------------------------------
# ModelLlmEndpointEntry
# ---------------------------------------------------------------------------


class TestModelLlmEndpointEntry:
    def test_minimal_running_entry(self) -> None:
        entry = ModelLlmEndpointEntry(
            slot_id="coder-5090",
            host="192.168.86.201",
            port=8000,
            endpoint_url="http://192.168.86.201:8000",
            url_env_var="LLM_CODER_URL",
            role_env_alias="LLM_CODER_SLOW_URL",
            model_hf_id="cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit",
            role="coder_slow",
            status="running",
        )
        assert entry.slot_id == "coder-5090"
        assert entry.port == 8000
        assert entry.status == "running"
        assert entry.notes == ""

    def test_planned_entry_nullable_fields(self) -> None:
        entry = ModelLlmEndpointEntry(
            slot_id="vision-planned",
            host=None,
            port=None,
            endpoint_url=None,
            url_env_var=None,
            role_env_alias="LLM_VISION_URL",
            model_hf_id=None,
            role="vision",
            status="planned",
        )
        assert entry.host is None
        assert entry.port is None
        assert entry.model_hf_id is None

    def test_missing_slot_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelLlmEndpointEntry.model_validate(  # type: ignore[call-arg]
                {"role": "coder_slow", "status": "running"}
            )

    def test_missing_role_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelLlmEndpointEntry.model_validate({"slot_id": "x", "status": "running"})

    def test_missing_status_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelLlmEndpointEntry.model_validate({"slot_id": "x", "role": "coder_slow"})

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelLlmEndpointEntry.model_validate(
                {
                    "slot_id": "x",
                    "role": "coder_slow",
                    "status": "running",
                    "unknown_field": "boom",
                }
            )

    def test_running_missing_required_fields_rejected(self) -> None:
        with pytest.raises(ValidationError, match="running endpoints require non-null"):
            ModelLlmEndpointEntry(slot_id="x", role="coder_slow", status="running")

    def test_running_missing_subset_rejected(self) -> None:
        with pytest.raises(ValidationError, match="port, endpoint_url, model_hf_id"):
            ModelLlmEndpointEntry(
                slot_id="x",
                role="coder_slow",
                status="running",
                host="192.168.86.201",
            )

    def test_frozen_immutability(self) -> None:
        entry = ModelLlmEndpointEntry(
            slot_id="x",
            role="coder_slow",
            status="running",
            host="192.168.86.201",
            port=8000,
            endpoint_url="http://192.168.86.201:8000",
            model_hf_id="org/model",
        )
        with pytest.raises((ValidationError, TypeError)):
            entry.slot_id = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ModelLlmEndpointContract
# ---------------------------------------------------------------------------


class TestModelLlmEndpointContract:
    def test_empty_contract(self) -> None:
        contract = ModelLlmEndpointContract()
        assert contract.endpoints == []

    def test_round_trip_minimal(self) -> None:
        raw = {
            "endpoints": [
                {
                    "slot_id": "coder-5090",
                    "host": "192.168.86.201",
                    "port": 8000,
                    "endpoint_url": "http://192.168.86.201:8000",
                    "url_env_var": "LLM_CODER_URL",
                    "role_env_alias": None,
                    "model_hf_id": "cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit",
                    "role": "coder_slow",
                    "status": "running",
                }
            ]
        }
        contract = ModelLlmEndpointContract.model_validate(raw)
        assert len(contract.endpoints) == 1
        assert contract.endpoints[0].url_env_var == "LLM_CODER_URL"

    def test_running_filter(self) -> None:
        raw = {
            "endpoints": [
                {
                    "slot_id": "a",
                    "role": "coder_slow",
                    "status": "running",
                    "host": "192.168.86.201",
                    "port": 8000,
                    "endpoint_url": "http://192.168.86.201:8000",
                    "model_hf_id": "org/model",
                    "url_env_var": "LLM_CODER_URL",
                },
                {
                    "slot_id": "b",
                    "role": "vision",
                    "status": "planned",
                },
            ]
        }
        contract = ModelLlmEndpointContract.model_validate(raw)
        assert len(contract.running()) == 1
        assert contract.running()[0].slot_id == "a"

    def test_by_role_filter(self) -> None:
        _running = {
            "host": "192.168.86.201",
            "port": 8000,
            "endpoint_url": "http://192.168.86.201:8000",
            "model_hf_id": "org/model",
        }
        raw = {
            "endpoints": [
                {"slot_id": "a", "role": "coder_slow", "status": "running", **_running},
                {"slot_id": "b", "role": "embedding", "status": "running", **_running},
                {"slot_id": "c", "role": "coder_slow", "status": "disabled"},
            ]
        }
        contract = ModelLlmEndpointContract.model_validate(raw)
        coders = contract.by_role("coder_slow")
        assert len(coders) == 2
        assert all(e.role == "coder_slow" for e in coders)

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelLlmEndpointContract.model_validate({"endpoints": [], "extra": "boom"})

    def test_frozen_immutability(self) -> None:
        contract = ModelLlmEndpointContract()
        with pytest.raises((ValidationError, TypeError)):
            contract.endpoints = []  # type: ignore[misc]


# ---------------------------------------------------------------------------
# load_llm_endpoint_contract — real YAML round-trip
# ---------------------------------------------------------------------------


class TestLoadLlmEndpointContract:
    def test_loads_real_yaml(self) -> None:
        if not _LLM_ENDPOINTS_YAML.exists():
            pytest.skip("contracts/llm_endpoints.yaml not present in worktree")
        contract = load_llm_endpoint_contract(_LLM_ENDPOINTS_YAML)
        assert len(contract.endpoints) > 0

    def test_real_yaml_has_running_slots(self) -> None:
        if not _LLM_ENDPOINTS_YAML.exists():
            pytest.skip("contracts/llm_endpoints.yaml not present in worktree")
        contract = load_llm_endpoint_contract(_LLM_ENDPOINTS_YAML)
        assert len(contract.running()) > 0

    def test_real_yaml_coder_slot_present(self) -> None:
        if not _LLM_ENDPOINTS_YAML.exists():
            pytest.skip("contracts/llm_endpoints.yaml not present in worktree")
        contract = load_llm_endpoint_contract(_LLM_ENDPOINTS_YAML)
        coders = contract.by_role("coder_slow")
        assert any(e.url_env_var == "LLM_CODER_URL" for e in coders)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_llm_endpoint_contract(tmp_path / "nonexistent.yaml")
