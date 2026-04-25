# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration test: load contracts/llm_endpoints.yaml via ModelLlmEndpointContract.

Validates that the typed loader can parse the real contract file from the repo.
No external infra required — this is a filesystem-only integration test.
"""

from __future__ import annotations

import pytest

from omnibase_infra.models.contracts.enum_llm_endpoint_status import (
    EnumLlmEndpointStatus,
)
from omnibase_infra.models.contracts.model_llm_endpoint_contract import (
    load_llm_endpoint_contract,
)

_CONTRACTS_DIR = __import__("pathlib").Path(__file__).parent.parent.parent / "contracts"
_LLM_ENDPOINTS_YAML = _CONTRACTS_DIR / "llm_endpoints.yaml"

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    not _LLM_ENDPOINTS_YAML.exists(),
    reason="contracts/llm_endpoints.yaml not present in this checkout",
)
def test_load_real_llm_endpoints_yaml() -> None:
    contract = load_llm_endpoint_contract(_LLM_ENDPOINTS_YAML)
    assert isinstance(contract.endpoints, list)
    assert len(contract.endpoints) > 0


@pytest.mark.skipif(
    not _LLM_ENDPOINTS_YAML.exists(),
    reason="contracts/llm_endpoints.yaml not present in this checkout",
)
def test_running_filter_returns_subset() -> None:
    contract = load_llm_endpoint_contract(_LLM_ENDPOINTS_YAML)
    running = contract.running()
    for entry in running:
        assert entry.status == EnumLlmEndpointStatus.RUNNING


@pytest.mark.skipif(
    not _LLM_ENDPOINTS_YAML.exists(),
    reason="contracts/llm_endpoints.yaml not present in this checkout",
)
def test_all_entries_have_required_fields() -> None:
    contract = load_llm_endpoint_contract(_LLM_ENDPOINTS_YAML)
    for entry in contract.endpoints:
        assert entry.slot_id
        assert entry.role
        assert entry.status in EnumLlmEndpointStatus.__members__.values()
