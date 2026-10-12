# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Chain test: runtime contract load -> contract-registered -> registry upsert [OMN-21031].

Nothing in the runtime published ``onex.evt.platform.contract-registered.v1``, so
the contract registry reducer and persistence effect were attached and idle and
``public.contracts`` stayed empty. This test drives the whole chain with the
real parts and no fake handlers except the Postgres write itself:

    discovered contracts (real contract.yaml files on disk)
      -> publish_contract_registrations (the kernel's producer)
      -> published bytes
      -> ContractRegistrationEventRouter.handle_message (the kernel's consumer)
      -> ContractRegistryReducer
      -> ``postgres.upsert_contract`` intent handler
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import yaml

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.nodes.node_contract_registry_reducer.contract_registration_event_router import (
    ContractRegistrationEventRouter,
)
from omnibase_infra.nodes.node_contract_registry_reducer.reducer import (
    ContractRegistryReducer,
)
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.auto_wiring import ModelAutoWiringManifest
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.service_kernel import publish_contract_registrations
from omnibase_infra.topics import SUFFIX_CONTRACT_REGISTERED

NODES_ROOT = Path(__file__).parents[4] / "src" / "omnibase_infra" / "nodes"
REGISTRY_NODE = "node_contract_registry_reducer"
LOADER_CONTRACT = NODES_ROOT / "node_contract_loader_effect" / "contract.yaml"


def _recording_bus() -> tuple[MagicMock, list[tuple[str, bytes]]]:
    """Bus double that serializes ``publish_envelope`` the way Kafka does."""
    published: list[tuple[str, bytes]] = []

    async def publish_envelope(
        envelope: object, topic: str, *, key: bytes | None = None
    ) -> None:
        assert isinstance(envelope, ModelEventEnvelope)
        published.append((topic, json.dumps(envelope.model_dump(mode="json")).encode()))

    bus = MagicMock(spec=ProtocolEventBusLike)
    bus.publish_envelope = AsyncMock(side_effect=publish_envelope)
    return bus, published


def _manifest_for(*node_names: str) -> ModelAutoWiringManifest:
    paths = [NODES_ROOT / name / "contract.yaml" for name in node_names]
    manifest = discover_contracts_from_paths(paths)
    assert manifest.total_errors == 0, manifest.errors
    assert manifest.total_discovered == len(node_names)
    return manifest


@pytest.mark.unit
@pytest.mark.asyncio
async def test_runtime_contract_load_fills_the_contract_registry_chain() -> None:
    """AC1: the published bytes reach the upsert effect through the real router."""
    manifest = _manifest_for(REGISTRY_NODE, "node_contract_persistence_effect")
    bus, published = _recording_bus()

    count = await publish_contract_registrations(
        event_bus=bus,
        topic=SUFFIX_CONTRACT_REGISTERED,
        contracts=manifest.contracts,
        correlation_id=uuid4(),
    )

    assert count == 2
    assert [topic for topic, _ in published] == [SUFFIX_CONTRACT_REGISTERED] * 2

    upsert = MagicMock()
    upsert.handle = AsyncMock(return_value=MagicMock(success=True))
    router = ContractRegistrationEventRouter(
        container=MagicMock(),
        reducer=ContractRegistryReducer(),
        effect_handlers={"postgres.upsert_contract": upsert},
        event_bus=None,
        tick_interval_seconds=60,
    )
    for offset, (topic, value) in enumerate(published):
        await router.handle_message(
            ModelEventMessage(
                topic=topic,
                key=None,
                value=value,
                headers=ModelEventHeaders(
                    source="test",
                    event_type=topic,
                    timestamp=datetime.now(UTC),
                ),
                offset=str(offset),
                partition=0,
            )
        )

    upserts = {
        call.args[0].node_name: call.args[0] for call in upsert.handle.await_args_list
    }
    assert set(upserts) == {REGISTRY_NODE, "node_contract_persistence_effect"}
    source = NODES_ROOT / REGISTRY_NODE / "contract.yaml"
    assert yaml.safe_load(upserts[REGISTRY_NODE].contract_yaml)["name"] == REGISTRY_NODE
    assert (
        upserts[REGISTRY_NODE].contract_hash
        == manifest.contracts[0].contract_content_hash
    )
    assert source.read_text(encoding="utf-8") == upserts[REGISTRY_NODE].contract_yaml


@pytest.mark.unit
@pytest.mark.asyncio
async def test_an_unreadable_contract_is_skipped_and_the_rest_publish() -> None:
    manifest = _manifest_for(REGISTRY_NODE, "node_contract_persistence_effect")
    gone = manifest.contracts[0].model_copy(
        update={"contract_path": Path("/nonexistent/contract.yaml")}
    )
    bus, published = _recording_bus()

    count = await publish_contract_registrations(
        event_bus=bus,
        topic=SUFFIX_CONTRACT_REGISTERED,
        contracts=(gone, manifest.contracts[1]),
        correlation_id=uuid4(),
    )

    assert count == 1
    assert len(published) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_publish_failure_does_not_stop_the_remaining_contracts() -> None:
    manifest = _manifest_for(REGISTRY_NODE, "node_contract_persistence_effect")
    bus = MagicMock(spec=ProtocolEventBusLike)
    bus.publish_envelope = AsyncMock(
        side_effect=[InfraUnavailableError("broker down"), None]
    )

    count = await publish_contract_registrations(
        event_bus=bus,
        topic=SUFFIX_CONTRACT_REGISTERED,
        contracts=manifest.contracts,
        correlation_id=uuid4(),
    )

    assert count == 1
    assert bus.publish_envelope.await_count == 2


@pytest.mark.unit
def test_loader_effect_contract_declares_the_registered_topic() -> None:
    declared = yaml.safe_load(LOADER_CONTRACT.read_text(encoding="utf-8"))
    assert SUFFIX_CONTRACT_REGISTERED in declared["event_bus"]["publish_topics"]
