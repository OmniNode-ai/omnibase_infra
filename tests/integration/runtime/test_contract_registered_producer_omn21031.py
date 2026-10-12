# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for the contract-registered producer [OMN-21031].

Discovers every real node contract shipped in ``src/omnibase_infra/nodes`` and
publishes a contract-registered envelope for each through the kernel producer.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.service_kernel import publish_contract_registrations
from omnibase_infra.topics import SUFFIX_CONTRACT_REGISTERED

pytestmark = pytest.mark.integration

NODES_ROOT = Path(__file__).parents[3] / "src" / "omnibase_infra" / "nodes"


@pytest.mark.asyncio
async def test_every_discovered_shipped_contract_is_published_once() -> None:
    manifest = discover_contracts_from_paths(sorted(NODES_ROOT.glob("*/contract.yaml")))
    assert manifest.total_discovered > 0
    bus = MagicMock(spec=ProtocolEventBusLike)
    bus.publish_envelope = AsyncMock()

    count = await publish_contract_registrations(
        event_bus=bus,
        topic=SUFFIX_CONTRACT_REGISTERED,
        contracts=manifest.contracts,
        correlation_id=uuid4(),
    )

    assert count == manifest.total_discovered
    published = {
        call.args[0].payload["node_name"]
        for call in bus.publish_envelope.await_args_list
        if isinstance(call.args[0], ModelEventEnvelope)
    }
    assert published == {contract.name for contract in manifest.contracts}
    assert all(
        call.args[1] == SUFFIX_CONTRACT_REGISTERED
        if len(call.args) > 1
        else call.kwargs["topic"] == SUFFIX_CONTRACT_REGISTERED
        for call in bus.publish_envelope.await_args_list
    )
