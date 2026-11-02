# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Chain test: catalog contracts -> registry router -> registry upsert [OMN-21031, OMN-21590].

OMN-21031 filled ``public.contracts`` by publishing one ``contract-registered`` record
per contract and consuming it back. OMN-21590 makes the node catalog the source: the
kernel feeds the registry router the same events in process and a boot publishes no
record on that topic. This test drives the whole chain with the real parts and no
fake handlers except the Postgres write itself:

    discovered contracts (real contract.yaml files on disk)
      -> build_registered_events (reads the catalog's contracts)
      -> ContractRegistrationEventRouter.register_announced (the kernel's feed)
      -> ContractRegistryReducer
      -> ``postgres.upsert_contract`` intent handler
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import yaml

from omnibase_infra.handlers.handler_package_announcer import (
    build_registered_events,
)
from omnibase_infra.models.node_announce.model_node_announcement_topic import (
    ModelNodeAnnouncementTopic,
)
from omnibase_infra.nodes.node_contract_registry_reducer.contract_registration_event_router import (
    ContractRegistrationEventRouter,
)
from omnibase_infra.nodes.node_contract_registry_reducer.reducer import (
    ContractRegistryReducer,
)
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.auto_wiring import ModelAutoWiringManifest
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.topics import SUFFIX_CONTRACT_REGISTERED

NODES_ROOT = Path(__file__).parents[4] / "src" / "omnibase_infra" / "nodes"
REGISTRY_NODE = "node_contract_registry_reducer"
LOADER_CONTRACT = NODES_ROOT / "node_contract_loader_effect" / "contract.yaml"


def _manifest_for(*node_names: str) -> ModelAutoWiringManifest:
    paths = [NODES_ROOT / name / "contract.yaml" for name in node_names]
    manifest = discover_contracts_from_paths(paths)
    assert manifest.total_errors == 0, manifest.errors
    assert manifest.total_discovered == len(node_names)
    return manifest


@pytest.mark.unit
@pytest.mark.asyncio
async def test_catalog_contracts_fill_the_contract_registry_chain() -> None:
    """AC1: the catalog's contracts reach the upsert effect through the real router."""
    manifest = _manifest_for(REGISTRY_NODE, "node_contract_persistence_effect")
    bus = MagicMock(spec=ProtocolEventBusLike)
    bus.publish_envelope = AsyncMock()
    bus.publish = AsyncMock()

    events = build_registered_events(manifest.contracts, uuid4())
    assert len(events) == 2

    upsert = MagicMock()
    upsert.handle = AsyncMock(return_value=MagicMock(success=True))
    router = ContractRegistrationEventRouter(
        container=MagicMock(),
        reducer=ContractRegistryReducer(),
        effect_handlers={"postgres.upsert_contract": upsert},
        event_bus=None,
        tick_interval_seconds=60,
    )
    for event in events:
        await router.register_announced(
            event, uuid4(), topic=ModelNodeAnnouncementTopic.suffix_for("main")
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
    # AC4 (OMN-21590): feeding the registry publishes no record, on any topic.
    bus.publish_envelope.assert_not_awaited()
    bus.publish.assert_not_awaited()


@pytest.mark.unit
def test_an_unreadable_contract_is_skipped_and_the_rest_are_built() -> None:
    manifest = _manifest_for(REGISTRY_NODE, "node_contract_persistence_effect")
    gone = manifest.contracts[0].model_copy(
        update={"contract_path": Path("/nonexistent/contract.yaml")}
    )

    events = build_registered_events((gone, manifest.contracts[1]), uuid4())

    assert [e.node_name for e in events] == [manifest.contracts[1].name]


@pytest.mark.unit
def test_loader_effect_contract_declares_the_registered_topic() -> None:
    declared = yaml.safe_load(LOADER_CONTRACT.read_text(encoding="utf-8"))
    assert SUFFIX_CONTRACT_REGISTERED in declared["event_bus"]["publish_topics"]
