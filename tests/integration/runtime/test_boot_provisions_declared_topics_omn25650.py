# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage: boot provisions the declared topics of real on-disk contracts (OMN-25650).

Drives ``provision_unattached_publish_topics`` over every ``contract.yaml``
shipped under ``omnibase_infra.nodes``, read from disk through the real
auto-wiring parser, and asserts each topic those contracts declare in
``runtime_dispatch`` is requested from the provisioner exactly once.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

import omnibase_infra.nodes as nodes_pkg
from omnibase_infra.contracts.contract_yaml import read_contract_yaml
from omnibase_infra.handlers.boot_topic_provisioning import (
    declared_topics_outside_event_bus,
    provision_unattached_publish_topics,
)
from omnibase_infra.runtime.auto_wiring.discovery import _parse_contract
from omnibase_infra.runtime.auto_wiring.models import ModelAutoWiringManifest

pytestmark = pytest.mark.integration


class _RecordingProvisioner:
    def __init__(self) -> None:
        self.created: list[str] = []

    async def ensure_topic_exists(
        self,
        topic_name: str,
        spec: object | None = None,
        correlation_id: UUID | None = None,
    ) -> bool:
        self.created.append(topic_name)
        return True


@pytest.mark.asyncio
async def test_real_node_contracts_declared_topics_are_provisioned() -> None:
    root = Path(nodes_pkg.__file__).parent
    contracts = []
    expected: set[str] = set()
    for path in sorted(root.glob("*/contract.yaml")):
        raw = read_contract_yaml(path)
        declared = declared_topics_outside_event_bus(raw)
        if not declared:
            continue
        expected.update(declared)
        contracts.append(
            _parse_contract(
                contract_path=path,
                entry_point_name=path.parent.name,
                package_name="omnibase_infra",
                package_version="0.0.0",
            )
        )
    manifest = ModelAutoWiringManifest(contracts=tuple(contracts))
    provisioner = _RecordingProvisioner()

    await provision_unattached_publish_topics(
        manifest, (), provisioner, max_concurrent=4
    )

    assert expected, "no real node contract declares a topic outside event_bus"
    assert expected <= set(provisioner.created)
