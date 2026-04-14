# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for MixinNodeIntrospection heartbeat lifecycle (OMN-8691).

Validates start_heartbeat_task() and stop_heartbeat_task() satisfy
ProtocolNodeIntrospection so RuntimeHostProcess can wire them without
leaving the introspection topic stale.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from omnibase_core.enums import EnumNodeKind
from omnibase_infra.mixins import MixinNodeIntrospection
from omnibase_infra.models.discovery import ModelIntrospectionConfig


class _HeartbeatTestNode(MixinNodeIntrospection):
    def __init__(self) -> None:
        config = ModelIntrospectionConfig(
            node_id=uuid4(),
            node_type=EnumNodeKind.COMPUTE,
            node_name="heartbeat-test-node",
            event_bus=None,
            version="0.1.0",
        )
        self.initialize_introspection(config)


@pytest.fixture
def node() -> _HeartbeatTestNode:
    return _HeartbeatTestNode()


@pytest.mark.asyncio
async def test_start_heartbeat_task_creates_task(node: _HeartbeatTestNode) -> None:
    await node.start_heartbeat_task()
    assert node._heartbeat_task is not None
    assert not node._heartbeat_task.done()
    await node.stop_heartbeat_task()


@pytest.mark.asyncio
async def test_stop_heartbeat_task_before_start_is_safe(
    node: _HeartbeatTestNode,
) -> None:
    await node.stop_heartbeat_task()
    assert node._heartbeat_task is None


@pytest.mark.asyncio
async def test_start_heartbeat_task_is_idempotent(node: _HeartbeatTestNode) -> None:
    await node.start_heartbeat_task()
    task_ref = node._heartbeat_task
    await node.start_heartbeat_task()
    assert node._heartbeat_task is task_ref
    await node.stop_heartbeat_task()


@pytest.mark.asyncio
async def test_heartbeat_task_stops_cleanly(node: _HeartbeatTestNode) -> None:
    await node.start_heartbeat_task()
    assert node._heartbeat_task is not None
    await node.stop_heartbeat_task()
    assert node._heartbeat_task is None


@pytest.mark.asyncio
async def test_protocol_methods_present(node: _HeartbeatTestNode) -> None:
    assert callable(getattr(node, "start_heartbeat_task", None))
    assert callable(getattr(node, "stop_heartbeat_task", None))
