# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for FakePostgresAdapter and FakeValkeyClient (OMN-9265).

Validates that both test doubles exercise the full protocol lifecycle
correctly, satisfying the integration-test gate for src/ changes.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.primitives import ModelSemVer
from omnibase_infra.nodes.node_registry_effect.protocols.protocol_postgres_adapter import (
    ProtocolPostgresAdapter,
)
from omnibase_infra.test_utils.fake_postgres_adapter import FakePostgresAdapter
from omnibase_infra.test_utils.fake_valkey_client import FakeValkeyClient


@pytest.mark.integration
async def test_fake_postgres_adapter_protocol_lifecycle() -> None:
    """Full upsert → deactivate lifecycle through the ProtocolPostgresAdapter interface."""
    adapter: ProtocolPostgresAdapter = FakePostgresAdapter()
    node_id = uuid4()
    version = ModelSemVer(major=1, minor=0, patch=0)

    # Upsert succeeds
    result = await adapter.upsert(
        node_id=node_id,
        node_type=EnumNodeKind.EFFECT,
        node_version=version,
        endpoints={"grpc": "localhost:50051"},
        metadata={"env": "test"},
    )
    assert result.success is True
    assert result.error is None

    # Deactivate succeeds and clears the record
    deactivate_result = await adapter.deactivate(node_id)
    assert deactivate_result.success is True

    # The fake tracks deactivated nodes for assertion
    assert node_id in adapter.deactivated  # type: ignore[union-attr]
    assert node_id not in adapter.records  # type: ignore[union-attr]


@pytest.mark.integration
async def test_fake_valkey_client_scheduler_lifecycle() -> None:
    """Simulate the RuntimeScheduler Valkey sequence-number persist/load cycle."""
    client = FakeValkeyClient()
    seq_key = "test:scheduler:seq"

    # Health check equivalent — ping succeeds
    assert await client.ping() is True

    # Nothing stored yet
    assert await client.get(seq_key) is None

    # Persist sequence number
    assert await client.set(seq_key, "100") is True

    # Load sequence number (simulating restart)
    raw = await client.get(seq_key)
    assert raw == "100"
    assert int(raw) == 100

    # Cleanup
    await client.aclose()
    assert client._closed is True
