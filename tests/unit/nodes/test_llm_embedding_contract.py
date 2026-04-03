# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for LLM embedding node contract field migration [OMN-7410]."""

from __future__ import annotations

from pathlib import Path

import yaml


def test_embedding_contract_uses_standard_subscribe_field() -> None:
    """Embedding contract must use event_bus.subscribe_topics, not consumed_events."""
    contract_path = Path(
        "src/omnibase_infra/nodes/node_llm_embedding_effect/contract.yaml"
    )
    contract = yaml.safe_load(contract_path.read_text())
    subscribe_topics = contract.get("event_bus", {}).get("subscribe_topics", [])
    assert len(subscribe_topics) > 0, "Must use event_bus.subscribe_topics"
    assert "consumed_events" not in contract, (
        "Legacy consumed_events field must be removed"
    )
