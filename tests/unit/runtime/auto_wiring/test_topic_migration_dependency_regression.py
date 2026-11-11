# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Optional ports must not scan contracts through unrelated migration services.

A reproduction of the recorded 300s timeout made eleven migration dependency
builds, costing 147.12s on the reproducing host. Optional publisher/consumer
constructors have no required parameters; the empty-set subset check selected
the migration factory for them. Assert the work count rather than a CPU-dependent
deadline, and retain the required migration-services positive control.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from omnibase_infra.runtime.auto_wiring import handler_wiring as wiring

pytestmark = pytest.mark.unit


class OptionalPublisher:
    def __init__(self, event_publisher: object | None = None) -> None:
        pass


class OptionalConsumer:
    def __init__(self, event_consumer: object | None = None) -> None:
        pass


class OptionalPorts:
    def __init__(
        self,
        event_publisher: object | None = None,
        event_consumer: object | None = None,
    ) -> None:
        pass


class MigrationExecutor:
    def __init__(self, provisioner: object, drain_proof_gate: object) -> None:
        pass


@pytest.mark.parametrize(
    "handler_cls", [OptionalPublisher, OptionalConsumer, OptionalPorts]
)
@pytest.mark.parametrize("with_bus", [False, True])
@pytest.mark.asyncio
async def test_optional_runtime_port_does_not_build_migration_dependencies(
    monkeypatch: pytest.MonkeyPatch, handler_cls: type, with_bus: bool
) -> None:
    build = Mock(return_value={"provisioner": object(), "drain_proof_gate": object()})
    monkeypatch.setattr(wiring, "_build_topic_migration_executor_dependencies", build)
    wiring._materialize_known_handler_dependencies(
        handler_name=handler_cls.__name__,
        handler_cls=handler_cls,
        materialized_explicit_dependencies=None,
        event_bus=Mock() if with_bus else None,
        container=None,
        ownership_query=None,
    )
    build.assert_not_called()


def test_required_migration_services_are_still_materialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    services = {"provisioner": object(), "drain_proof_gate": object()}
    build = Mock(return_value=services)
    monkeypatch.setattr(wiring, "_build_topic_migration_executor_dependencies", build)
    result = wiring._materialize_known_handler_dependencies(
        handler_name="MigrationExecutor",
        handler_cls=MigrationExecutor,
        materialized_explicit_dependencies=None,
        event_bus=None,
        container=None,
        ownership_query=None,
    )
    build.assert_called_once_with()
    assert result == {"MigrationExecutor": services}
