# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Mock handlers over an injected store that already holds records (OMN-21536)."""

from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

import pytest

from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.handlers.registration_storage.handler_registration_storage_mock import (
    HandlerRegistrationStorageMock,
)
from omnibase_infra.handlers.registration_storage.models import (
    ModelDeleteRegistrationRequest,
    ModelUpdateRegistrationRequest,
)
from omnibase_infra.handlers.service_discovery.handler_service_discovery_mock import (
    HandlerServiceDiscoveryMock,
)
from omnibase_infra.handlers.service_discovery.models import ModelServiceInfo
from omnibase_infra.handlers.store_keyed_memory import StoreKeyedMemory
from omnibase_infra.nodes.node_registration_storage_effect.models import (
    ModelRegistrationRecord,
    ModelRegistrationUpdate,
    ModelStorageQuery,
)
from omnibase_infra.nodes.node_service_discovery_effect.models import (
    ModelDiscoveryQuery,
)


def _record() -> ModelRegistrationRecord:
    return ModelRegistrationRecord(
        node_id=uuid4(),
        node_type=EnumNodeKind.EFFECT,
        node_version=ModelSemVer.parse("1.0.0"),
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_registration_handler_sees_prepopulated_store() -> None:
    store: StoreKeyedMemory[UUID, ModelRegistrationRecord] = StoreKeyedMemory()
    existing = _record()
    store.put(existing.node_id, existing)
    handler = HandlerRegistrationStorageMock(store)

    assert await handler.get_record_count() == 1
    assert await handler.get_record(existing.node_id) == existing

    result = await handler.store_registration(existing)
    assert result.operation == "update"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_registration_handlers_share_one_store() -> None:
    store: StoreKeyedMemory[UUID, ModelRegistrationRecord] = StoreKeyedMemory()
    writer = HandlerRegistrationStorageMock(store)
    reader = HandlerRegistrationStorageMock(store)
    record = _record()

    await writer.store_registration(record)
    assert await reader.get_record_count() == 1

    deleted = await reader.delete_registration(
        ModelDeleteRegistrationRequest(node_id=record.node_id)
    )
    assert deleted.deleted is True
    assert await writer.get_record_count() == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_service_discovery_handlers_share_one_store() -> None:
    store: StoreKeyedMemory[UUID, ModelServiceInfo] = StoreKeyedMemory()
    first = HandlerServiceDiscoveryMock(store)
    second = HandlerServiceDiscoveryMock(store)
    service = ModelServiceInfo(
        service_id=uuid4(), service_name="svc", address="svc.example", port=1
    )

    await first.register_service(service)
    assert await second.get_service_count() == 1

    await second.clear()
    assert await first.get_service_count() == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_registration_update_and_query_see_records_from_a_shared_store() -> None:
    store: StoreKeyedMemory[UUID, ModelRegistrationRecord] = StoreKeyedMemory()
    existing = _record()
    store.put(existing.node_id, existing)
    updater = HandlerRegistrationStorageMock(store)
    reader = HandlerRegistrationStorageMock(store)

    result = await updater.update_registration(
        ModelUpdateRegistrationRequest(
            node_id=existing.node_id,
            updates=ModelRegistrationUpdate(capabilities=["shared"]),
        )
    )
    assert result.success is True

    queried = await reader.query_registrations(ModelStorageQuery())
    assert [r.node_id for r in queried.records] == [existing.node_id]
    assert queried.records[0].capabilities == ("shared",)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_service_discovery_query_sees_services_put_in_a_shared_store() -> None:
    store: StoreKeyedMemory[UUID, ModelServiceInfo] = StoreKeyedMemory()
    service = ModelServiceInfo(
        service_id=uuid4(), service_name="svc", address="svc.example", port=1
    )
    store.put(service.service_id, service)
    handler = HandlerServiceDiscoveryMock(store)

    found = await handler.discover_services(ModelDiscoveryQuery(service_name="svc"))
    assert [s.service_id for s in found.services] == [service.service_id]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_stores_through_shared_store_lose_no_record() -> None:
    store: StoreKeyedMemory[UUID, ModelRegistrationRecord] = StoreKeyedMemory()
    handlers = [HandlerRegistrationStorageMock(store) for _ in range(4)]
    records = [_record() for _ in range(20)]

    await asyncio.gather(
        *(handlers[i % 4].store_registration(r) for i, r in enumerate(records))
    )

    assert store.count() == len(records)


def test_store_has_distinguishes_stored_none_from_absent_key() -> None:
    store: StoreKeyedMemory[str, None] = StoreKeyedMemory()
    store.put("present", None)

    assert store.has("present") is True
    assert store.has("absent") is False
