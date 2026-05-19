# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ``FakePostgresAdapter`` (OMN-9265).

Verifies protocol compliance, happy-path upsert/deactivate, error-path
injection, concurrency safety, and counter tracking.
"""

from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

import pytest

from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.primitives import ModelSemVer
from omnibase_infra.nodes.node_registry_effect.protocols.protocol_postgres_adapter import (
    ProtocolPostgresAdapter,
)
from omnibase_infra.test_utils.fake_postgres_adapter import FakePostgresAdapter


def _semver(major: int = 1, minor: int = 0, patch: int = 0) -> ModelSemVer:
    return ModelSemVer(major=major, minor=minor, patch=patch)


def _make_node_id() -> UUID:
    return uuid4()


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_satisfies_protocol_postgres_adapter() -> None:
    """FakePostgresAdapter must be a structural subtype of ProtocolPostgresAdapter."""
    assert isinstance(FakePostgresAdapter(), ProtocolPostgresAdapter)


# ---------------------------------------------------------------------------
# upsert — happy path
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upsert_returns_success() -> None:
    adapter = FakePostgresAdapter()
    node_id = _make_node_id()
    result = await adapter.upsert(
        node_id=node_id,
        node_type=EnumNodeKind.EFFECT,
        node_version=_semver(),
        endpoints={"grpc": "localhost:50051"},
        metadata={"env": "test"},
    )
    assert result.success is True
    assert result.error is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upsert_stores_record() -> None:
    adapter = FakePostgresAdapter()
    node_id = _make_node_id()
    await adapter.upsert(
        node_id=node_id,
        node_type=EnumNodeKind.EFFECT,
        node_version=_semver(),
        endpoints={"grpc": "localhost:50051"},
        metadata={},
    )
    assert node_id in adapter.records


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upsert_stores_copies_of_mutable_args() -> None:
    """Mutations to caller-owned dicts must not affect stored records."""
    adapter = FakePostgresAdapter()
    node_id = _make_node_id()
    endpoints = {"grpc": "localhost:50051"}
    metadata = {"env": "test"}
    await adapter.upsert(
        node_id=node_id,
        node_type=EnumNodeKind.EFFECT,
        node_version=_semver(),
        endpoints=endpoints,
        metadata=metadata,
    )
    endpoints["extra"] = "mutated"
    metadata["extra"] = "mutated"
    stored = adapter.records[node_id]
    assert "extra" not in stored["endpoints"]  # type: ignore[operator]
    assert "extra" not in stored["metadata"]  # type: ignore[operator]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upsert_overwrites_existing_record() -> None:
    adapter = FakePostgresAdapter()
    node_id = _make_node_id()
    await adapter.upsert(
        node_id=node_id,
        node_type=EnumNodeKind.EFFECT,
        node_version=_semver(1, 0, 0),
        endpoints={"grpc": "old"},
        metadata={},
    )
    await adapter.upsert(
        node_id=node_id,
        node_type=EnumNodeKind.EFFECT,
        node_version=_semver(1, 1, 0),
        endpoints={"grpc": "new"},
        metadata={},
    )
    assert adapter.records[node_id]["endpoints"] == {"grpc": "new"}  # type: ignore[index]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upsert_increments_call_count() -> None:
    adapter = FakePostgresAdapter()
    node_id = _make_node_id()
    for _ in range(3):
        await adapter.upsert(
            node_id=node_id,
            node_type=EnumNodeKind.EFFECT,
            node_version=_semver(),
            endpoints={},
            metadata={},
        )
    assert adapter.upsert_call_count == 3


# ---------------------------------------------------------------------------
# upsert — error injection
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upsert_error_injection_returns_failure() -> None:
    adapter = FakePostgresAdapter(fail_on_upsert=True)
    node_id = _make_node_id()
    result = await adapter.upsert(
        node_id=node_id,
        node_type=EnumNodeKind.EFFECT,
        node_version=_semver(),
        endpoints={},
        metadata={},
    )
    assert result.success is False
    assert result.error is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upsert_error_injection_does_not_store_record() -> None:
    adapter = FakePostgresAdapter(fail_on_upsert=True)
    node_id = _make_node_id()
    await adapter.upsert(
        node_id=node_id,
        node_type=EnumNodeKind.EFFECT,
        node_version=_semver(),
        endpoints={},
        metadata={},
    )
    assert node_id not in adapter.records


# ---------------------------------------------------------------------------
# deactivate — happy path
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_deactivate_returns_success() -> None:
    adapter = FakePostgresAdapter()
    node_id = _make_node_id()
    await adapter.upsert(
        node_id=node_id,
        node_type=EnumNodeKind.EFFECT,
        node_version=_semver(),
        endpoints={},
        metadata={},
    )
    result = await adapter.deactivate(node_id)
    assert result.success is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_deactivate_removes_from_records() -> None:
    adapter = FakePostgresAdapter()
    node_id = _make_node_id()
    await adapter.upsert(
        node_id=node_id,
        node_type=EnumNodeKind.EFFECT,
        node_version=_semver(),
        endpoints={},
        metadata={},
    )
    await adapter.deactivate(node_id)
    assert node_id not in adapter.records


@pytest.mark.unit
@pytest.mark.asyncio
async def test_deactivate_adds_to_deactivated_set() -> None:
    adapter = FakePostgresAdapter()
    node_id = _make_node_id()
    await adapter.deactivate(node_id)
    assert node_id in adapter.deactivated


@pytest.mark.unit
@pytest.mark.asyncio
async def test_deactivate_then_upsert_removes_from_deactivated() -> None:
    """Re-registering a previously deactivated node should clear the deactivated flag."""
    adapter = FakePostgresAdapter()
    node_id = _make_node_id()
    await adapter.deactivate(node_id)
    assert node_id in adapter.deactivated
    await adapter.upsert(
        node_id=node_id,
        node_type=EnumNodeKind.EFFECT,
        node_version=_semver(),
        endpoints={},
        metadata={},
    )
    assert node_id not in adapter.deactivated
    assert node_id in adapter.records


@pytest.mark.unit
@pytest.mark.asyncio
async def test_deactivate_increments_call_count() -> None:
    adapter = FakePostgresAdapter()
    node_id = _make_node_id()
    await adapter.deactivate(node_id)
    await adapter.deactivate(node_id)
    assert adapter.deactivate_call_count == 2


# ---------------------------------------------------------------------------
# deactivate — error injection
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_deactivate_error_injection_returns_failure() -> None:
    adapter = FakePostgresAdapter(fail_on_deactivate=True)
    node_id = _make_node_id()
    result = await adapter.deactivate(node_id)
    assert result.success is False
    assert result.error is not None


# ---------------------------------------------------------------------------
# Concurrency safety
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_concurrent_upserts_are_safe() -> None:
    """Concurrent upserts for distinct nodes should all succeed."""
    adapter = FakePostgresAdapter()
    node_ids = [_make_node_id() for _ in range(20)]
    results = await asyncio.gather(
        *[
            adapter.upsert(
                node_id=nid,
                node_type=EnumNodeKind.EFFECT,
                node_version=_semver(),
                endpoints={},
                metadata={},
            )
            for nid in node_ids
        ]
    )
    assert all(r.success for r in results)
    assert len(adapter.records) == 20
    assert adapter.upsert_call_count == 20
