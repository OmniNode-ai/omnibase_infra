# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The node catalog equals what the entry-point scan found, on every bus (OMN-21590, A2 AC1 and AC7).

The reference is an independent restatement of the scan that ``discover_contracts``
ran before the catalog existed: iterate the ``onex.nodes`` entry points of the active
packages, load each, take the contract file beside the node, read its ``name`` and
hash its bytes in the canonical form. The catalog built by boot step 0 must hold the
same names at the same hashes and the same contract paths -- one difference fails.

The same assertions run with the announcer's bus bound by one overlay value: core's
in-memory bus, or a real broker started by testcontainers. Nothing but the overlay
differs between the two parametrisations.
"""

from __future__ import annotations

import hashlib
import inspect
import subprocess
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import entry_points
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio
import yaml

from omnibase_core.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_core.protocols.services.protocol_announce_bus import ProtocolAnnounceBus
from omnibase_infra.handlers.handler_package_announcer import (
    ENTRY_POINT_GROUP,
    HandlerPackageAnnouncer,
    announcer_identity,
    load_boot_runtime_config,
)
from omnibase_infra.models.node_announce.model_node_announcement_topic import (
    ModelNodeAnnouncementTopic,
)
from omnibase_infra.models.node_announce.model_package_announce_request import (
    ModelPackageAnnounceRequest,
)
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts
from omnibase_infra.utils.util_runtime_packages import (
    get_active_runtime_packages,
    is_runtime_package_active,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@dataclass(frozen=True)
class Overlay:
    """The lane overlay value that binds the announcer's bus."""

    bus: str  # "inmemory" | "kafka"
    bootstrap: str | None = None


def _canonical_hash(raw: bytes) -> str:
    return hashlib.sha256(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def _reference_scan() -> dict[str, tuple[str, Path]]:
    """Name -> (canonical hash, contract path) of the first contract per name."""
    active = get_active_runtime_packages()
    found: dict[str, tuple[str, Path]] = {}
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        dist = ep.dist.name if ep.dist is not None else "unknown"
        if not is_runtime_package_active(dist, active):
            continue
        try:
            node = ep.load()
            explicit = getattr(node, "contract_path", None)
            if explicit is not None and Path(str(explicit)).is_file():
                path = Path(str(explicit))
            elif getattr(node, "__path__", None) is not None:
                path = next(
                    Path(p) / "contract.yaml"
                    for p in node.__path__
                    if (Path(p) / "contract.yaml").is_file()
                )
            else:
                here = Path(inspect.getfile(node)).parent
                path = (
                    here / "contract.yaml"
                    if (here / "contract.yaml").is_file()
                    else here.parent / "contract.yaml"
                )
            raw = path.read_bytes()
            document = yaml.safe_load(raw)
        except (
            ImportError,
            AttributeError,
            OSError,
            RuntimeError,
            StopIteration,
            SyntaxError,
            TypeError,
            ValueError,
            yaml.YAMLError,
        ):
            continue
        if not isinstance(document, dict):
            continue
        name = document.get("name", ep.name)
        found.setdefault(str(name), (_canonical_hash(raw), path))
    return found


def _docker_available() -> bool:
    try:
        return (
            subprocess.run(
                ["docker", "info"], capture_output=True, timeout=15, check=False
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


@pytest_asyncio.fixture(params=["inmemory", "kafka"])
async def overlay(request: pytest.FixtureRequest) -> AsyncIterator[Overlay]:
    """The one thing that differs between the two parametrisations."""
    if request.param == "inmemory":
        yield Overlay(bus="inmemory")
        return
    if not _docker_available():
        pytest.skip("Docker daemon not available for testcontainers")
    from testcontainers.kafka import KafkaContainer

    with KafkaContainer() as container:
        yield Overlay(bus="kafka", bootstrap=container.get_bootstrap_server())


def _bind_bus(overlay: Overlay) -> ProtocolAnnounceBus[object]:
    """Build the bus the overlay names; the test body never names a bus class."""
    if overlay.bus == "inmemory":
        return EventBusInmemory(environment="test", group="catalog-parity")
    from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
    from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
        ModelKafkaEventBusConfig,
    )

    assert overlay.bootstrap is not None
    return EventBusKafka(
        config=ModelKafkaEventBusConfig(
            bootstrap_servers=overlay.bootstrap,
            environment="test",
            auto_offset_reset="earliest",
        )
    )


async def test_catalog_equals_the_entry_point_scan(overlay: Overlay) -> None:
    bus = _bind_bus(overlay)
    suffix = f"{uuid4().hex[:8]}" if overlay.bus == "kafka" else "main"
    request = ModelPackageAnnounceRequest(
        topic=ModelNodeAnnouncementTopic.suffix_for(f"parity-{suffix}"),
        announcer_id=announcer_identity("omnibase_infra.parity-test"),
        generation=1,
        announced_at=datetime.now(UTC),
        active_packages=get_active_runtime_packages(),
        catalog_group=f"catalog-parity-{uuid4().hex[:8]}",
    )
    try:
        result = await HandlerPackageAnnouncer().handle(
            request,
            bus=bus,
            settle_timeout_seconds=load_boot_runtime_config().catalog_settle_timeout_seconds,
        )
    finally:
        stop = getattr(bus, "close", None)
        if stop is not None:
            await stop()

    reference = _reference_scan()
    assert reference, "the installed package set announced nothing"
    catalog = {
        name: (entry.content_hash, entry.contract_path)
        for name, entry in result.catalog.entries.items()
    }

    assert set(catalog) == set(reference), (
        f"only in catalog: {sorted(set(catalog) - set(reference))[:5]}; "
        f"only in scan: {sorted(set(reference) - set(catalog))[:5]}"
    )
    mismatched = {
        name: (catalog[name], reference[name])
        for name in reference
        if catalog[name] != reference[name]
    }
    assert not mismatched, f"hash or path differs for {sorted(mismatched)[:5]}"
    assert result.catalog.stale_generation_rejections == len(
        [o for o in result.outcomes if not o.accepted]
    )


async def test_discovery_reads_the_same_catalog() -> None:
    """The manifest is a read of the catalog: its contracts are catalog entries."""
    manifest = discover_contracts()
    reference = _reference_scan()
    for contract in manifest.contracts:
        assert contract.name in reference
        digest, path = reference[contract.name]
        assert contract.contract_content_hash == digest
        assert contract.contract_path == path
