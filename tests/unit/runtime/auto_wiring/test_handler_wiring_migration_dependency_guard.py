# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Optional bus callbacks must not trigger topic-migration contract scans."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest

from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _materialize_known_handler_dependencies,
    wire_from_manifest,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

_WIRING = "omnibase_infra.runtime.auto_wiring.handler_wiring"


class HandlerOptionalPublisher:
    """Constructor shape from the recorded downstream manifest timeout."""

    def __init__(
        self, event_publisher: Callable[[str, bytes], None] | None = None
    ) -> None:
        self.event_publisher = event_publisher


class HandlerOptionalConsumer:
    def __init__(
        self, event_consumer: Callable[[str, object], None] | None = None
    ) -> None:
        self.event_consumer = event_consumer


class HandlerOptionalCallbacks:
    def __init__(
        self,
        event_publisher: Callable[[str, bytes], None] | None = None,
        event_consumer: Callable[[str, object], None] | None = None,
    ) -> None:
        self.event_publisher = event_publisher
        self.event_consumer = event_consumer


@pytest.mark.parametrize(
    "handler_cls",
    [HandlerOptionalPublisher, HandlerOptionalConsumer, HandlerOptionalCallbacks],
)
@pytest.mark.parametrize("has_event_bus", [False, True])
def test_optional_callbacks_do_not_build_migration_dependencies(
    handler_cls: type, has_event_bus: bool
) -> None:
    """Count expensive construction, independent of host speed or YAML backend."""
    explicit = {"OtherHandler": {"dependency": object()}}
    publisher = object()
    consumer = object()
    with (
        patch(f"{_WIRING}._build_topic_migration_executor_dependencies") as build,
        patch(f"{_WIRING}._make_sync_event_publisher", return_value=publisher),
        patch(f"{_WIRING}._make_sync_event_consumer", return_value=consumer),
    ):
        materialized = _materialize_known_handler_dependencies(
            handler_name=handler_cls.__name__,
            handler_cls=handler_cls,
            materialized_explicit_dependencies=explicit,
            event_bus=object() if has_event_bus else None,
            container=None,
            ownership_query=None,
        )

    build.assert_not_called()
    assert list(explicit) == ["OtherHandler"]
    if not has_event_bus:
        assert materialized is not None
        assert materialized["OtherHandler"] is explicit["OtherHandler"]
        assert not materialized.get(handler_cls.__name__, {})
    else:
        assert materialized is not None
        assert materialized["OtherHandler"] is explicit["OtherHandler"]
        callbacks = materialized[handler_cls.__name__]
        if handler_cls in (HandlerOptionalPublisher, HandlerOptionalCallbacks):
            assert callbacks["event_publisher"] is publisher
        if handler_cls in (HandlerOptionalConsumer, HandlerOptionalCallbacks):
            assert callbacks["event_consumer"] is consumer
        assert "provisioner" not in callbacks
        assert "drain_proof_gate" not in callbacks


@pytest.mark.asyncio
async def test_real_migration_manifest_still_materializes_required_dependencies() -> (
    None
):
    """Wire the on-disk contract and real handler through the kernel entrypoint."""
    contract_path = (
        Path(__file__).resolve().parents[4]
        / "src/omnibase_infra/nodes/node_topic_migration_executor_effect/contract.yaml"
    )
    manifest = discover_contracts_from_paths([contract_path])
    assert not manifest.errors
    assert len(manifest.contracts) == 1
    provisioner = object()
    drain_proof_gate = object()
    with patch(
        f"{_WIRING}._build_topic_migration_executor_dependencies",
        return_value={"provisioner": provisioner, "drain_proof_gate": drain_proof_gate},
    ) as build:
        report = await wire_from_manifest(
            manifest=manifest,
            dispatch_engine=MessageDispatchEngine(),
            event_bus=None,
            container=ModelONEXContainer(),
            subscribe_immediately=False,
        )

    build.assert_called_once_with()
    assert report.total_failed == 0
    assert report.total_wired == 1
