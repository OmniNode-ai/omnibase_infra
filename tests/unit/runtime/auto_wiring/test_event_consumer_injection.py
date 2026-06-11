# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression coverage for auto-wired handler event_consumer injection (OMN-13005).

The dispatch INPUT crash was OMN-13003. This is the SECOND, distinct defect on
``node_context_roi_runner``: a request/response EFFECT handler declares an
injectable blocking ``event_consumer`` (publish command -> block on correlated
terminal event -> read result fields back), but the runtime auto-wiring only
materialized ``event_publisher`` and had no equivalent for ``event_consumer``.
The handler therefore fell back to its own no-op default that returns ``None``
immediately, so every result row was a degenerate generation-failure even though
the terminal event arrived ~1s later.

These tests drive the REAL wiring path (``_prepare_handler_wiring`` ->
``_materialize_known_handler_dependencies``) rather than constructing the handler
directly, so they exercise the dispatch surface that the handler-isolation golden
chain never touches (memory ``feedback_real_dispatch_path_tests``).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.services.service_handler_resolver import ServiceHandlerResolver
from omnibase_core.services.service_local_handler_ownership_query import (
    ServiceLocalHandlerOwnershipQuery,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import _prepare_handler_wiring
from omnibase_infra.runtime.auto_wiring.models import (
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)

_TERMINAL_TOPIC = "onex.evt.omnimarket.node-generation-completed.v1"


class RecordingEventBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, bytes | None, bytes]] = []

    async def publish(self, topic: str, key: bytes | None, value: bytes) -> None:
        self.published.append((topic, key, value))


class HandlerRoiRunnerShape:
    """Fake with the request/response constructor shape of HandlerContextRoiRunner.

    Mirrors the real handler's degenerate-vs-healthy branch: when the injected
    ``event_consumer`` returns ``None`` (the no-op default), the row is recorded
    as ``failure_stage=generation`` with ``attempt_count=0``. When the consumer
    blocks-correlates and returns the terminal payload, the row is populated.
    """

    # Result topic the fake emits its recorded row to, so the test can assert
    # on the event bus (a side channel) without capturing the handler instance —
    # the resolver requires _import_handler_class to return an actual type.
    ROW_TOPIC = "onex.evt.omnimarket.context-roi-run-completed.v1"

    def __init__(
        self,
        event_publisher: Callable[[str, bytes], None] | None = None,
        event_consumer: Callable[[str, str, float], dict[str, Any] | None]
        | None = None,
    ) -> None:
        self._event_publisher = event_publisher
        # Default no-op consumer returns None immediately (the live defect).
        self._event_consumer = event_consumer or (lambda _t, _c, _to: None)

    async def handle(self, envelope: object) -> None:
        correlation_id = "cid-roi-runner-1"
        if self._event_publisher is not None:
            self._event_publisher(
                "onex.cmd.omnimarket.node-generation-requested.v1",
                b'{"task":"invoice"}',
            )
        terminal = self._event_consumer(_TERMINAL_TOPIC, correlation_id, 120.0)
        if terminal is None:
            row = {"failure_stage": "generation", "attempt_count": 0, "model_id": ""}
        else:
            row = {
                "failure_stage": "none",
                "attempt_count": int(terminal["attempt_count"]),
                "model_id": str(terminal["model_id"]),
            }
        if self._event_publisher is not None:
            self._event_publisher(self.ROW_TOPIC, json.dumps(row).encode("utf-8"))


def _make_roi_runner_contract() -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name="node_context_roi_runner",
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/contract.yaml"),
        entry_point_name="node_context_roi_runner",
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(
            subscribe_topics=("onex.cmd.omnimarket.context-roi-run-requested.v1",),
            publish_topics=(
                "onex.evt.omnimarket.context-roi-run-completed.v1",
                "onex.cmd.omnimarket.node-generation-requested.v1",
            ),
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="HandlerContextRoiRunner",
                        module="omnimarket.nodes.node_context_roi_runner.handlers.handler_context_roi_runner",
                    ),
                ),
            ),
        ),
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_auto_wired_event_consumer_blocks_and_returns_non_degenerate_row() -> (
    None
):
    """RED before OMN-13005: wiring injected no event_consumer -> no-op -> degenerate.

    Drives the trial through the real wiring with a terminal event that arrives
    AFTER a short delay; asserts a NON-DEGENERATE row (attempt_count>=1, populated
    model_id, no failure_stage=generation). With the pre-fix no-op consumer, the
    handler records failure_stage=generation/attempt_count=0 and this assertion
    fails.
    """
    contract = _make_roi_runner_contract()
    event_bus = RecordingEventBus()
    resolver = ServiceHandlerResolver()
    ownership_query = ServiceLocalHandlerOwnershipQuery(
        local_node_names=frozenset({contract.name})
    )

    async def _delayed_terminal(
        *,
        event_bus: object,
        terminal_topic: str,
        correlation_id: str,
        timeout_seconds: float,
    ) -> dict[str, Any] | None:
        # Terminal arrives ~after the handler began waiting — proves blocking.
        await asyncio.sleep(0.05)
        return {
            "correlation_id": correlation_id,
            "attempt_count": 1,
            "model_id": "Qwen3.6-35B-A3B",
            "provider": "local",
            "contract_passed": True,
            "prompt_tokens": 10,
            "completion_tokens": 20,
        }

    with (
        patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=HandlerRoiRunnerShape,
        ),
        patch(
            "omnibase_infra.runtime.service_terminal_event_consumer._await_correlated_terminal",
            side_effect=_delayed_terminal,
        ),
    ):
        prepared = _prepare_handler_wiring(
            contract=contract,
            entry=contract.handler_routing.handlers[0],  # type: ignore[union-attr]
            dispatch_engine=None,
            resolver=resolver,
            ownership_query=ownership_query,
            event_bus=event_bus,
            container=None,
        )

        await prepared.dispatcher(
            ModelEventEnvelope[dict[str, str]](
                payload={"run_id": "t1"},
                event_type="omnimarket.context-roi-run-requested",
            )
        )
        await asyncio.sleep(0)

    row_publishes = [
        json.loads(value.decode("utf-8"))
        for topic, _key, value in event_bus.published
        if topic == HandlerRoiRunnerShape.ROW_TOPIC
    ]
    assert row_publishes, "handler never emitted a result row"
    row = row_publishes[-1]
    assert row["failure_stage"] != "generation", (
        "event_consumer was not injected (handler fell back to no-op returning "
        "None) — the OMN-13005 degenerate-row defect"
    )
    assert row["attempt_count"] >= 1
    assert row["model_id"] == "Qwen3.6-35B-A3B"
