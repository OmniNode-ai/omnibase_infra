# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract schema regression tests for OMN-10867.

Verifies that:
1. node_build_loop_write_effect declares event_bus.subscribe_topics so
   wire_from_manifest can wire it (was silently SKIPPED before this fix).
2. node_build_loop_projection_compute declares consumer_purpose='audit',
   which is the expected audit-consumer path (not a bug).
3. The generated topic enum includes the new CMD/EVT topics.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.integration]

_NODES_ROOT = Path(__file__).parent.parent.parent / "src" / "omnibase_infra" / "nodes"


def _load_contract(node_name: str) -> dict:
    path = _NODES_ROOT / node_name / "contract.yaml"
    assert path.exists(), f"contract.yaml missing at {path}"
    with path.open() as f:
        return yaml.safe_load(f)


def test_write_effect_has_subscribe_topics() -> None:
    contract = _load_contract("node_build_loop_write_effect")
    event_bus = contract.get("event_bus", {})
    subscribe_topics = event_bus.get("subscribe_topics", [])
    assert subscribe_topics, (
        "node_build_loop_write_effect must declare event_bus.subscribe_topics "
        "so wire_from_manifest wires it (OMN-10867)"
    )
    assert "onex.cmd.omnibase-infra.build-loop-append.v1" in subscribe_topics


def test_write_effect_handler_routing_declares_handler() -> None:
    contract = _load_contract("node_build_loop_write_effect")
    routing = contract.get("handler_routing", {})
    handlers = routing.get("handlers", [])
    assert handlers, (
        "node_build_loop_write_effect must declare handler_routing.handlers"
    )
    declared = {
        h.get("handler_class") or h.get("handler", {}).get("name") for h in handlers
    }
    assert "HandlerBuildLoopAppend" in declared


def test_projection_compute_has_audit_consumer_purpose() -> None:
    contract = _load_contract("node_build_loop_projection_compute")
    event_bus = contract.get("event_bus", {})
    assert event_bus.get("consumer_purpose") == "audit", (
        "node_build_loop_projection_compute must use consumer_purpose='audit' "
        "to route via the dedicated raw-event projection wiring path"
    )


def test_projection_compute_has_subscribe_topics() -> None:
    contract = _load_contract("node_build_loop_projection_compute")
    event_bus = contract.get("event_bus", {})
    subscribe_topics = event_bus.get("subscribe_topics", [])
    assert subscribe_topics, (
        "node_build_loop_projection_compute must declare subscribe_topics"
    )


def test_topic_enum_includes_new_build_loop_topics() -> None:
    from omnibase_infra.enums.generated.enum_omnibase_infra_topic import (
        EnumOmnibaseInfraTopic,
    )

    topic_values = {t.value for t in EnumOmnibaseInfraTopic}
    assert "onex.cmd.omnibase-infra.build-loop-append.v1" in topic_values, (
        "CMD_BUILD_LOOP_APPEND_V1 missing from generated topic enum (OMN-10867)"
    )
    assert "onex.evt.omnibase-infra.build-loop-appended.v1" in topic_values, (
        "EVT_BUILD_LOOP_APPENDED_V1 missing from generated topic enum (OMN-10867)"
    )
