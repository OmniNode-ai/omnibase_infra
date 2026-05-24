# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for LLM endpoint topology contract validation."""

from __future__ import annotations

from pathlib import Path

import yaml

from omnibase_infra.models.contracts.model_llm_endpoint_entry import (
    ModelLlmEndpointEntry,
)

_CONTRACT_PATH = Path(__file__).parents[2] / "contracts" / "llm_endpoints.yaml"


def test_llm_endpoint_contract_entries_validate_against_typed_schema() -> None:
    payload = yaml.safe_load(_CONTRACT_PATH.read_text())
    endpoints = payload["endpoints"]

    parsed = [ModelLlmEndpointEntry.model_validate(item) for item in endpoints]

    assert parsed
    assert {entry.slot_id for entry in parsed}
    assert all(entry.role for entry in parsed)
