# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The in-memory mock handlers keep their state in an injected store (OMN-21536)."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.handlers.registration_storage.handler_registration_storage_mock import (
    HandlerRegistrationStorageMock,
)
from omnibase_infra.handlers.registration_storage.models import (
    ModelDeleteRegistrationRequest,
)
from omnibase_infra.handlers.service_discovery.handler_service_discovery_mock import (
    HandlerServiceDiscoveryMock,
)
from omnibase_infra.handlers.service_discovery.models import ModelServiceInfo
from omnibase_infra.handlers.store_keyed_memory import StoreKeyedMemory
from omnibase_infra.nodes.node_registration_storage_effect.models import (
    ModelRegistrationRecord,
)


def _record() -> ModelRegistrationRecord:
    return ModelRegistrationRecord(
        node_id=uuid4(),
        node_type=EnumNodeKind.EFFECT,
        node_version=ModelSemVer.parse("1.0.0"),
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_registration_mock_writes_through_injected_store() -> None:
    store: StoreKeyedMemory[UUID, ModelRegistrationRecord] = StoreKeyedMemory()
    handler = HandlerRegistrationStorageMock(store)
    record = _record()

    await handler.store_registration(record)

    stored = store.get(record.node_id)
    assert stored is not None
    assert stored.node_id == record.node_id
    assert store.count() == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_registration_mock_delete_and_clear_use_store() -> None:
    store: StoreKeyedMemory[UUID, ModelRegistrationRecord] = StoreKeyedMemory()
    handler = HandlerRegistrationStorageMock(store)
    first, second = _record(), _record()
    await handler.store_registration(first)
    await handler.store_registration(second)

    result = await handler.delete_registration(
        ModelDeleteRegistrationRequest(node_id=first.node_id)
    )
    assert result.deleted is True
    assert store.get(first.node_id) is None
    assert store.count() == 1

    await handler.clear()
    assert store.count() == 0
    assert await handler.get_record_count() == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_registration_mock_default_stores_are_not_shared() -> None:
    first = HandlerRegistrationStorageMock()
    second = HandlerRegistrationStorageMock()

    await first.store_registration(_record())

    assert await first.get_record_count() == 1
    assert await second.get_record_count() == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_service_discovery_mock_writes_through_injected_store() -> None:
    store: StoreKeyedMemory[UUID, ModelServiceInfo] = StoreKeyedMemory()
    handler = HandlerServiceDiscoveryMock(store)
    service = ModelServiceInfo(
        service_id=uuid4(), service_name="svc", address="svc.example", port=1
    )

    await handler.register_service(service)
    assert store.count() == 1
    stored = store.get(service.service_id)
    assert stored is not None
    assert stored.service_name == service.service_name

    await handler.deregister_service(service.service_id)
    assert store.count() == 0

    await handler.register_service(service)
    await handler.clear()
    assert store.count() == 0
    assert await handler.get_service_count() == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_service_discovery_mock_default_stores_are_not_shared() -> None:
    first = HandlerServiceDiscoveryMock()
    second = HandlerServiceDiscoveryMock()
    await first.register_service(
        ModelServiceInfo(
            service_id=uuid4(), service_name="svc", address="svc.example", port=1
        )
    )

    assert await first.get_service_count() == 1
    assert await second.get_service_count() == 0


@pytest.mark.unit
def test_keyed_store_round_trip() -> None:
    store: StoreKeyedMemory[str, int] = StoreKeyedMemory()
    assert store.get("a") is None
    assert store.evict("a") is None

    store.put("a", 1)
    store.put("b", 2)
    store.put("a", 3)

    assert store.get("a") == 3
    assert store.keys() == ("a", "b")
    assert store.snapshot() == (3, 2)
    assert store.count() == 2
    assert store.evict("a") == 3
    assert store.count() == 1

    store.reset()
    assert store.count() == 0
    assert store.keys() == ()
