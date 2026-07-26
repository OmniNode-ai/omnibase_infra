# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Audit tests for node contract subscription fields [OMN-7410].

Ensures all node contracts use the standard event_bus.subscribe_topics
field and do not retain legacy consumed_events fields.
"""

from __future__ import annotations

from collections import Counter
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


def test_topic_match_contracts_pair_every_subscribed_topic_to_a_handler_route() -> None:
    """Build-time pairing gate (OMN-15168, OMN-14594 precedent, OMN-15006 origin).

    For every node contract whose ``handler_routing.routing_strategy`` is
    ``"topic_match"``, each ``event_bus.subscribe_topics`` entry must have
    EXACTLY ONE corresponding ``handler_routing.handlers[].topic`` entry.

    This is the general, repo-wide form of the check first written ad hoc for
    ``node_ledger_projection_compute`` in
    ``tests/unit/runtime/test_ledger_projection_business_topics_omn15006.py``
    (OMN-15006) — that module's ``test_every_subscribed_topic_has_exactly_one_
    handler_routing_entry`` only ever covered ONE contract. This test iterates
    ALL ``topic_match`` contracts under ``NODES_DIR`` so it guards every future
    diff to any topic_match contract, not just the one ticket that motivated
    it, closing the gap a future contract could silently reopen (OMN-14594's
    "subscribed but never dispatched" NO_DISPATCHER class):

    1. subscribe_topics has strictly MORE entries than paired handler_routing
       topics ("list-only" diff — a topic was added to subscribe_topics with
       no matching handler_routing entry, or the reverse count drifted).
    2. Any individual subscribed topic has zero or more-than-one
       handler_routing entries (missing dispatcher, or an ambiguous/duplicate
       route).

    A contract with `routing_strategy` other than `"topic_match"` (e.g.
    `payload_type_match`, `operation_match`) is out of scope for this
    specific 1:1 topic-keyed pairing shape and is skipped.
    """
    unpaired: dict[str, dict[str, object]] = {}

    for contract_path in sorted(NODES_DIR.glob("*/contract.yaml")):
        contract = yaml.safe_load(contract_path.read_text()) or {}
        routing = contract.get("handler_routing") or {}
        if routing.get("routing_strategy") != "topic_match":
            continue

        node_name = contract_path.parent.name
        subscribed = list(
            dict.fromkeys(contract.get("event_bus", {}).get("subscribe_topics") or [])
        )
        handlers = routing.get("handlers") or []
        routed_topics = [entry.get("topic") for entry in handlers if entry.get("topic")]
        routed_counts = Counter(routed_topics)

        # Overall count parity: subscribe_topics entries vs. paired
        # handler_routing entries with a topic. A pure list-only diff (a
        # subscribe_topics addition with no matching handler_routing entry)
        # trips this even before the per-topic detail below.
        count_mismatch = len(subscribed) != len(routed_topics)

        missing = [topic for topic in subscribed if routed_counts.get(topic, 0) == 0]
        duplicated = {
            topic: n
            for topic, n in routed_counts.items()
            if topic in subscribed and n > 1
        }
        orphaned = sorted(set(routed_topics) - set(subscribed))

        if count_mismatch or missing or duplicated or orphaned:
            unpaired[node_name] = {
                "subscribe_topics_count": len(subscribed),
                "handler_routing_topic_count": len(routed_topics),
                "missing_handler_routing_for": missing,
                "duplicated_handler_routing_for": duplicated,
                "handler_routing_topics_not_subscribed": orphaned,
            }

    assert unpaired == {}, (
        "topic_match contract(s) have an unpaired subscribe_topics/"
        f"handler_routing diff (OMN-14594 NO_DISPATCHER class): {unpaired}"
    )
