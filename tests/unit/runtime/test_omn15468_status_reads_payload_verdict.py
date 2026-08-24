# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cross-boundary seam test for OMN-15468 residual — status must read the payload.

The applier-side guard (``apply_failure_terminal_guard``, landed in #2606) only
re-routes a failure-verdict return value off the success terminal when the
contract declares EXACTLY ONE distinct failure terminal. The ticket's own
measurement (``extract_terminal_event_topics`` docstring, PROVENANCE section)
puts this at 51 of 384 ``omnimarket`` contracts; the other **274 of 275**
terminal-declaring contracts expose exactly ONE top-level terminal topic and
declare no separate failure destination at all. For those, the guard's
``len(failure_terminal_topics) != 1`` branch cannot fire (there is nothing to
re-route to), the failure-verdict payload is published on the sole declared
topic, and ``_status_for_terminal_topic`` — which derives status purely from
topic identity — reports ``completed``. This is the SAME false-success this
ticket opened against, reproduced through the majority contract shape rather
than the minority shape #2606's own test already covers
(``test_omn15468_quota_terminal_seam.py``, which deliberately declares two
distinct terminals).

Boundary driven, real artifacts throughout:

    handler return value (failure verdict, map-miss)
      -> DispatchResultApplier.apply (the real publish-from-return path)
      -> JSON round-trip through ModelEventEnvelope[object] (the real wire
         shape a broker decodes, not the in-memory BaseModel)
      -> discover_runtime_local_ingress_routes (the real contract reader)
      -> _status_for_terminal_topic (the real status derivation)

RED before this change: the published topic is the contract's sole terminal
(there is no other topic to land on), and ``_status_for_terminal_topic``
reports ``"completed"`` for it regardless of the decoded payload's own
``contract_passed=False`` / ``status="failed"`` verdict.

GREEN after: ``_status_for_terminal_topic`` additionally reads the verdict off
the decoded payload (via the same ``resolve_terminal_verdict`` the applier-side
guard uses) and reports ``"failed"`` whenever a topic that would otherwise
read as ``"completed"`` carries an explicit failure verdict.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _declared_failure_terminal_topics,
    _select_dispatch_result_output_topic,
)
from omnibase_infra.runtime.auto_wiring.models.model_discovered_contract import (
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.runtime_local_ingress import (
    discover_runtime_local_ingress_routes,
)
from omnibase_infra.runtime.service_dispatch_result_applier import (
    DispatchResultApplier,
)
from omnibase_infra.runtime.service_pattern_b_broker import _status_for_terminal_topic

SOLE_TOPIC = "onex.evt.omnimarket.node-generation-completed.v1"
COMMAND_TOPIC = "onex.cmd.omnimarket.node-generation.v1"

# Majority shape: ONE top-level terminal_event, no terminal_events map, no
# runtime_dispatch.terminal_events — this is 274 of the 275 terminal-declaring
# contracts measured on the ticket, and is the exact original node_generation_consumer
# reproduction shape minus the runtime_dispatch failure declaration that #2560/#2606
# gave it after the fact.
_SINGLE_TERMINAL_CONTRACT = f"""
name: node_generation_seam_demo
version: 1.0.0
node_type: EFFECT_GENERIC
description: Single-terminal seam fixture contract (OMN-15468 residual).

event_bus:
  version: {{major: 1, minor: 0, patch: 0}}
  subscribe_topics:
    - {COMMAND_TOPIC}
  publish_topics:
    - {SOLE_TOPIC}

terminal_event: {SOLE_TOPIC}

handler_routing:
  routing_strategy: operation_match
  handlers:
    - operation: node-generation.dispatch
""".strip()


class ModelGenerationBenchmarkReturn(BaseModel):
    """A def-B return value shaped like the ticket's original live reproduction.

    Deliberately NOT in the contract's ``published_events`` map — the map-miss
    fallback is the exact condition under which the applier resolves the
    contract's SUCCESS terminal_event as the output topic.
    """

    contract_passed: bool
    correlation_id: UUID
    contract_yaml: str = ""
    handler_source: str = ""


def _write_contract(tmp_path: Path) -> tuple[Path, str]:
    package_root = tmp_path / "genpkg"
    node_dir = package_root / "nodes" / "node_generation_seam_demo"
    node_dir.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    contract_path = node_dir / "contract.yaml"
    contract_path.write_text(_SINGLE_TERMINAL_CONTRACT, encoding="utf-8")
    return contract_path, str(package_root / "__init__.py")


def _discovered_contract(contract_path: Path) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name="node_generation_seam_demo",
        node_type="EFFECT_GENERIC",
        contract_version={"major": 1, "minor": 0, "patch": 0},
        contract_path=contract_path,
        entry_point_name="node_generation_seam_demo",
        package_name="genpkg",
        terminal_event=SOLE_TOPIC,
        event_bus={
            "subscribe_topics": [COMMAND_TOPIC],
            "publish_topics": [SOLE_TOPIC],
        },
    )


def _make_success_result(
    correlation_id: UUID, output_event: BaseModel
) -> ModelDispatchResult:
    return ModelDispatchResult(
        status=EnumDispatchStatus.SUCCESS,
        topic=COMMAND_TOPIC,
        started_at=datetime.now(UTC),
        correlation_id=correlation_id,
        dispatcher_id="seam-test-dispatcher",
        output_events=[output_event],
    )


@pytest.mark.asyncio
async def test_single_terminal_contract_failure_verdict_reports_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure-verdict return value on a single-terminal contract must not
    read as ``completed`` downstream, even though there is no distinct
    failure topic for the applier-side guard to re-route it to.
    """
    correlation_id = uuid4()
    contract_path, package_init = _write_contract(tmp_path)
    contract = _discovered_contract(contract_path)

    # Hop 1 — declared terminals, read the one canonical way. This contract has
    # NO distinct failure terminal: the guard has nowhere to re-route to.
    success_topic = _select_dispatch_result_output_topic(contract)
    assert success_topic == SOLE_TOPIC
    failure_topics = _declared_failure_terminal_topics(
        contract, success_topic=SOLE_TOPIC
    )
    assert failure_topics == (), (
        "fixture precondition: this contract declares no distinct failure "
        f"terminal for the applier-side guard to use; got {failure_topics!r}"
    )

    # Hop 2 — the real applier, wired exactly as _subscribe_contract_topics wires it.
    event_bus = AsyncMock(spec=ProtocolEventBusLike)
    applier = DispatchResultApplier(
        event_bus=event_bus,
        output_topic=SOLE_TOPIC,
        output_topic_map={},
        allowed_output_topics=[SOLE_TOPIC],
        failure_terminal_topics=failure_topics,
    )
    returned = ModelGenerationBenchmarkReturn(
        contract_passed=False,
        correlation_id=correlation_id,
        contract_yaml="",
        handler_source="",
    )
    await applier.apply(
        _make_success_result(correlation_id, returned),
        correlation_id=correlation_id,
    )

    event_bus.publish_envelope.assert_awaited_once()
    published_topic = event_bus.publish_envelope.await_args.kwargs["topic"]
    published_envelope = event_bus.publish_envelope.await_args.kwargs["envelope"]
    assert published_topic == SOLE_TOPIC, (
        "fixture precondition: with zero disambiguated failure terminals the "
        "guard cannot re-route — the failure-verdict payload lands on the "
        f"sole declared topic. Got {published_topic!r}."
    )

    # Hop 3 — the REAL wire round-trip: what a broker actually decodes off
    # Kafka is JSON, not the in-memory BaseModel instance.
    wire_bytes = published_envelope.model_dump_json().encode("utf-8")
    decoded_envelope = ModelEventEnvelope[object].model_validate_json(wire_bytes)
    decoded_payload = decoded_envelope.payload
    assert isinstance(decoded_payload, dict)
    assert decoded_payload["contract_passed"] is False

    # Hop 4 — route discovery sees the (single) terminal.
    monkeypatch.setattr(
        "omnibase_infra.runtime.runtime_local_ingress.importlib.import_module",
        lambda _name: SimpleNamespace(__file__=package_init),
    )
    route = discover_runtime_local_ingress_routes(("genpkg",))[
        "node_generation_seam_demo"
    ]
    assert route.terminal_event == SOLE_TOPIC

    # Hop 5 — broker status must read the decoded payload's own verdict, not
    # just the arrival topic (there is only one topic to arrive on here).
    status = _status_for_terminal_topic(route, published_topic, decoded_payload)
    assert status == "failed", (
        f"terminal arrived on the contract's sole topic {published_topic!r} "
        f"carrying contract_passed=False; broker reported {status!r} — the "
        "outer /skill response for this run is not distinguishable from a "
        "real success"
    )


@pytest.mark.asyncio
async def test_single_terminal_contract_success_verdict_still_reports_completed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An honest success on the same single-terminal contract is unaffected."""
    correlation_id = uuid4()
    contract_path, package_init = _write_contract(tmp_path)
    contract = _discovered_contract(contract_path)
    failure_topics = _declared_failure_terminal_topics(
        contract, success_topic=SOLE_TOPIC
    )

    event_bus = AsyncMock(spec=ProtocolEventBusLike)
    applier = DispatchResultApplier(
        event_bus=event_bus,
        output_topic=SOLE_TOPIC,
        output_topic_map={},
        allowed_output_topics=[SOLE_TOPIC],
        failure_terminal_topics=failure_topics,
    )
    returned = ModelGenerationBenchmarkReturn(
        contract_passed=True,
        correlation_id=correlation_id,
        contract_yaml="handler: {}",
        handler_source="def handle(): ...",
    )
    await applier.apply(
        _make_success_result(correlation_id, returned),
        correlation_id=correlation_id,
    )

    published_topic = event_bus.publish_envelope.await_args.kwargs["topic"]
    published_envelope = event_bus.publish_envelope.await_args.kwargs["envelope"]
    decoded_payload = (
        ModelEventEnvelope[object]
        .model_validate_json(published_envelope.model_dump_json().encode("utf-8"))
        .payload
    )

    monkeypatch.setattr(
        "omnibase_infra.runtime.runtime_local_ingress.importlib.import_module",
        lambda _name: SimpleNamespace(__file__=package_init),
    )
    route = discover_runtime_local_ingress_routes(("genpkg",))[
        "node_generation_seam_demo"
    ]

    status = _status_for_terminal_topic(route, published_topic, decoded_payload)
    assert status == "completed"


def test_status_for_terminal_topic_default_payload_arg_is_backward_compatible() -> None:
    """Existing 2-arg callers (no payload) must keep working, unchanged."""
    from omnibase_infra.runtime.runtime_local_ingress import (
        ModelRuntimeLocalIngressRoute,
    )

    route = ModelRuntimeLocalIngressRoute(
        node_name="node_generation_seam_demo",
        contract_name="node_generation_seam_demo",
        command_topic=COMMAND_TOPIC,
        event_type="whatever.dispatch",
        terminal_event=SOLE_TOPIC,
        terminal_events=(SOLE_TOPIC,),
        contract_path="/nonexistent/contract.yaml",
        package_name="genpkg",
    )
    assert _status_for_terminal_topic(route, SOLE_TOPIC) == "completed"
    assert _status_for_terminal_topic(route, "some-other-topic") == "failed"
