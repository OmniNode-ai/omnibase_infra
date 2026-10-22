# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21271: the route cache of the delegation dispatch port on the real contracts.

The unit tests of this ticket drive a fake package under ``tmp_path``. These
tests build the port over the REAL installed ``omnibase_infra`` package, so
the cached resolution is checked against every node contract the runtime
image ships: it must equal an uncached discovery, a second resolution must
read no contract, and the stat-only fingerprint must cover exactly the
``nodes/*/contract.yaml`` files discovery walks.
"""

from __future__ import annotations

import importlib
import pathlib
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from omnibase_infra.enums.enum_consumer_group_purpose import EnumConsumerGroupPurpose
from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.event_bus.models.model_publish_receipt import (
    ModelPublishReceipt,
)
from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.runtime.runtime_local_ingress import (
    discover_runtime_local_ingress_routes,
    local_ingress_contract_fingerprint,
)
from omnibase_infra.runtime.service_delegation_dispatch_port import (
    RuntimeDelegationDispatchPort,
)

pytestmark = pytest.mark.integration

_PACKAGE = "omnibase_infra"


class _FakeRuntimeBus:
    """Satisfies the broker transport; route resolution never touches it."""

    environment = "test"
    bootstrap_servers = "localhost:9092"

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: ModelEventHeaders | None = None,
    ) -> ModelPublishReceipt | None:
        raise AssertionError("route resolution tests never publish")

    async def subscribe(
        self,
        topic: str,
        node_identity: ModelNodeIdentity | None = None,
        on_message: Callable[[ModelEventMessage], Awaitable[None]] | None = None,
        *,
        group_id: str | None = None,
        purpose: EnumConsumerGroupPurpose = EnumConsumerGroupPurpose.CONSUME,
        required_for_readiness: bool = False,
    ) -> Callable[[], Awaitable[None]]:
        raise AssertionError("route resolution tests never subscribe")


class _ContractReadCounter:
    """Counts ``Path.read_text`` calls on files named ``contract.yaml``."""

    def __init__(self) -> None:
        self.paths: list[str] = []
        self._lock = threading.Lock()

    @property
    def count(self) -> int:
        with self._lock:
            return len(self.paths)

    def record(self, path: Path) -> None:
        with self._lock:
            self.paths.append(str(path))


@pytest.fixture
def read_counter(monkeypatch: pytest.MonkeyPatch) -> _ContractReadCounter:
    counter = _ContractReadCounter()
    original = pathlib.Path.read_text

    def counting_read_text(
        self: Path, encoding: str | None = None, errors: str | None = None
    ) -> str:
        if self.name == "contract.yaml":
            counter.record(self)
        return original(self, encoding=encoding, errors=errors)

    monkeypatch.setattr(pathlib.Path, "read_text", counting_read_text)
    return counter


@pytest.fixture
def port(monkeypatch: pytest.MonkeyPatch) -> RuntimeDelegationDispatchPort:
    monkeypatch.delenv("ONEX_ACTIVE_RUNTIME_PACKAGES", raising=False)
    return RuntimeDelegationDispatchPort(_FakeRuntimeBus(), package_names=(_PACKAGE,))


def _installed_contracts() -> list[Path]:
    package_file = importlib.import_module(_PACKAGE).__file__
    assert package_file is not None
    return sorted(Path(package_file).parent.glob("nodes/*/contract.yaml"))


@pytest.mark.asyncio
async def test_cached_resolution_equals_uncached_discovery(
    port: RuntimeDelegationDispatchPort,
) -> None:
    resolved = await port._resolved_routes()
    discovered = discover_runtime_local_ingress_routes((_PACKAGE,))

    assert resolved
    assert resolved.keys() == discovered.keys()
    for key, route in discovered.items():
        assert resolved[key] == route


@pytest.mark.asyncio
async def test_second_resolution_reads_no_real_contract(
    port: RuntimeDelegationDispatchPort,
    read_counter: _ContractReadCounter,
) -> None:
    first = await port._resolved_routes()
    reads_after_first = read_counter.count
    assert reads_after_first > 0

    second = await port._resolved_routes()

    assert read_counter.count == reads_after_first
    assert second == first


def test_fingerprint_covers_every_installed_contract() -> None:
    installed = _installed_contracts()
    fingerprint = local_ingress_contract_fingerprint((_PACKAGE,))

    assert installed
    assert len(fingerprint) == len(installed)
    for path, _mtime_ns, _size in fingerprint:
        assert Path(path).exists()
