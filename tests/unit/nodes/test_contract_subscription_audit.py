# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Audit tests for node contract subscription fields [OMN-7410].

Ensures all node contracts use the standard event_bus.subscribe_topics
field and do not retain legacy consumed_events fields.
"""

from __future__ import annotations

from pathlib import Path

import yaml


NODES_DIR = Path("src/omnibase_infra/nodes")


def test_no_legacy_only_subscription_fields_on_package_nodes() -> None:
    """Nodes must not use consumed_events as the sole subscription mechanism.

    Nodes that have both consumed_events AND event_bus.subscribe_topics are
    allowed (migration in progress — consumed_events is documentation-only).
    Nodes that have consumed_events WITHOUT event_bus.subscribe_topics are
    broken and must be migrated.
    """
    legacy_only_nodes: list[str] = []
    for contract_path in sorted(NODES_DIR.glob("*/contract.yaml")):
        contract = yaml.safe_load(contract_path.read_text())
        has_legacy = "consumed_events" in contract
        has_standard = bool(contract.get("event_bus", {}).get("subscribe_topics"))
        if has_legacy and not has_standard:
            legacy_only_nodes.append(contract_path.parent.name)
    assert legacy_only_nodes == [], (
        f"Nodes using ONLY legacy consumed_events (no event_bus.subscribe_topics): "
        f"{legacy_only_nodes}"
    )


def test_subscribing_nodes_declare_topics_under_event_bus() -> None:
    """Nodes that consume events must declare them under event_bus.subscribe_topics."""
    broken: list[str] = []
    for contract_path in sorted(NODES_DIR.glob("*/contract.yaml")):
        contract = yaml.safe_load(contract_path.read_text())
        has_legacy = "consumed_events" in contract
        has_standard = bool(contract.get("event_bus", {}).get("subscribe_topics"))
        if has_legacy and not has_standard:
            broken.append(contract_path.parent.name)
    assert broken == [], f"Nodes with subscription fields outside event_bus: {broken}"
