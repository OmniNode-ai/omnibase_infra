# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for the contract registry feed [OMN-21031, OMN-21590].

Discovers every real node contract shipped in ``src/omnibase_infra/nodes`` and builds
the registry event for each from the catalog's contracts. A boot publishes none of
them: the kernel no longer owns a ``contract-registered`` producer.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_infra.handlers.handler_package_announcer import (
    build_registered_events,
)
from omnibase_infra.runtime import service_kernel
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths

pytestmark = pytest.mark.integration

NODES_ROOT = Path(__file__).parents[3] / "src" / "omnibase_infra" / "nodes"


def test_every_discovered_shipped_contract_yields_one_registry_event() -> None:
    manifest = discover_contracts_from_paths(sorted(NODES_ROOT.glob("*/contract.yaml")))
    assert manifest.total_discovered > 0

    events = build_registered_events(manifest.contracts, uuid4())

    assert {e.node_name for e in events} == {c.name for c in manifest.contracts}
    assert len(events) == manifest.total_discovered


def test_the_kernel_no_longer_owns_a_contract_registered_producer() -> None:
    assert not hasattr(service_kernel, "publish_contract_registrations")
