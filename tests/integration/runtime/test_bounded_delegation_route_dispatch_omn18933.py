# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18933 (K6): the real dispatch port, a real Kafka bus identity, one row.

Composes the pieces the runtime composes: an ``EventBusKafka`` built from a
config (never started, so nothing can reach a broker), the real
``RuntimeDelegationDispatchPort`` and the real pre-dispatch validator reading a
declared overlay. On the bounded lane a declaration mismatch must refuse inside
``dispatch()`` before the Pattern-B broker is constructed; the declared row must
pass the gate and reach the broker; a runtime outside the gate must pass through
untouched.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
    ModelKafkaEventBusConfig,
)
from omnibase_infra.runtime import service_delegation_dispatch_port as port_module
from omnibase_infra.runtime.bounded_delegation_routes import (
    resolve_bounded_delegation_route,
)
from omnibase_infra.runtime.runtime_local_ingress import ModelRuntimeLocalIngressRoute

pytestmark = pytest.mark.integration

_EXTERNAL = "192.0.2.10:47092"
_INTERNAL = "redpanda:9092"


class _BrokerReachedError(Exception):
    """Raised by the stand-in broker: the gate let the dispatch through."""


def _route() -> ModelRuntimeLocalIngressRoute:
    return ModelRuntimeLocalIngressRoute(
        node_name="node_delegation_orchestrator",
        contract_name="node_delegation_orchestrator",
        command_topic="onex.cmd.omnibase-infra.delegation-request.v1",
        event_type="omnimarket.delegation-request",
        terminal_event="onex.evt.omnibase-infra.delegation-completed.v1",
        terminal_events=(
            "onex.evt.omnibase-infra.delegation-completed.v1",
            "onex.evt.omnibase-infra.delegation-failed.v1",
        ),
        contract_path="/contracts/node_delegation_orchestrator/contract.yaml",
        package_name="omnimarket",
    )


def _overlay(tmp_path: Path, **row_changes: str) -> Path:
    row = {
        "consumer": "omnimarket.nodes.node_delegation_orchestrator",
        "terminal_route": "terminal_events",
        "repository_owner": "omnimarket",
        **row_changes,
    }
    lanes = {
        "dogfood": {
            "broker": _EXTERNAL,
            "security_protocol": "PLAINTEXT",
            "broker_topology": {
                "external_bootstrap_servers": _EXTERNAL,
                "internal_bootstrap_servers": _INTERNAL,
                "runtime_environment": "dogfood",
            },
            "delegation_routes": [row],
        }
    }
    path = tmp_path / "ci_bus_lanes.yaml"
    path.write_text(yaml.safe_dump({"lanes": lanes}), encoding="utf-8")
    return path


async def _dispatch(
    monkeypatch: pytest.MonkeyPatch, overlay: Path, *, environment: str
) -> None:
    def _stand_in_broker(*args: object, **kwargs: object) -> None:
        raise _BrokerReachedError

    def _resolve(**kwargs: object) -> object:
        return resolve_bounded_delegation_route(
            **kwargs,  # type: ignore[arg-type]
            overlay_path_for_test=overlay,
        )

    monkeypatch.setattr(port_module, "RuntimePatternBBroker", _stand_in_broker)
    monkeypatch.setattr(port_module, "resolve_bounded_delegation_route", _resolve)
    bus = EventBusKafka(
        config=ModelKafkaEventBusConfig(
            bootstrap_servers=_INTERNAL, environment=environment
        )
    )
    port = port_module.RuntimeDelegationDispatchPort(
        bus,  # type: ignore[arg-type]
        routes={
            "omnimarket.node_delegation_orchestrator.delegation.orchestrate": _route()
        },
    )
    await port.dispatch(
        prompt="k6 integration probe",
        task_type="document",
        correlation_id=uuid4(),
        max_tokens=16,
        source_file_path=None,
        source_session_id=None,
        wait=True,
    )


@pytest.mark.asyncio
async def test_declared_row_passes_the_gate_and_reaches_the_broker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(_BrokerReachedError):
        await _dispatch(monkeypatch, _overlay(tmp_path), environment="dogfood")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("row_changes", "message"),
    [
        ({"consumer": "omnimarket.nodes.node_other"}, "consumer mismatch"),
        ({"repository_owner": "omnibase_infra"}, "repository owner mismatch"),
    ],
)
async def test_a_mismatched_row_refuses_before_the_broker_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    row_changes: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(InfraUnavailableError, match=message):
        await _dispatch(
            monkeypatch, _overlay(tmp_path, **row_changes), environment="dogfood"
        )


@pytest.mark.asyncio
async def test_a_runtime_outside_the_gate_dispatches_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """stability-test shares the internal listener but is not a bounded lane."""
    with pytest.raises(_BrokerReachedError):
        await _dispatch(
            monkeypatch,
            _overlay(tmp_path, consumer="omnimarket.nodes.node_other"),
            environment="stability-test",
        )
