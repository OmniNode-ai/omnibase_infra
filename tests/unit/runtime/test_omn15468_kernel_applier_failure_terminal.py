# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-15468 AC2 — an applier built outside handler_wiring lost BOTH re-routes.

Live defect this pins (``.201`` dev lane, compose project ``omnibase-infra``,
read 2026-09-05): 18 of the trailing 25 records on
``onex.evt.omnimarket.delegate-skill-completed.v1`` carried ``status="failed"``
with a typed ``terminal_failure_cause`` (``provider_quota_exhausted`` /
``provider_error``). The runtime-effects log for the last of them shows
``DispatchResultApplier: Published output event to
onex.evt.omnimarket.delegate-skill-completed.v1`` and NO re-route warning at
all — the guard did not decline to fire, it could not fire.

Cause: ``service_kernel`` hand-registered an applier for
``node_delegate_skill_orchestrator`` by NAME, before the discovery manifest
exists, with neither ``output_topic_map`` nor ``failure_terminal_topics``.
``_subscribe_contract_topics`` treats a passed-in ``result_applier`` as
authoritative (``effective_result_applier = result_applier``), so the
contract-derived applier — the one carrying both inputs — was never built for
that contract. Both re-route mechanisms were therefore dead at once:

* class-based: with an empty ``output_topic_map``,
  ``_resolve_mapped_output_topic`` returns ``self._output_topic`` for EVERY
  returned class, so a ``ModelDelegateSkillFailed`` return routes to the
  SUCCESS terminal;
* verdict-based: with an empty ``failure_terminal_topics``,
  ``apply_failure_terminal_guard`` takes its ``len(...) != 1`` branch and
  returns the success topic while logging nothing (the branch only logs when
  the list is non-empty), which is exactly why the live logs are silent.

The tests below drive the real factory, the real applier, the real wire
round-trip and the real broker status derivation over a fixture contract with
the same shape as ``node_delegate_skill_orchestrator`` (two declared terminals,
``published_events`` for both).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.contract_terminal_events import (
    declared_failure_terminal_topics,
    resolve_terminal_verdict,
)
from omnibase_infra.runtime.runtime_local_ingress import ModelRuntimeLocalIngressRoute
from omnibase_infra.runtime.service_dispatch_result_applier import (
    build_contract_result_applier,
    build_static_result_applier,
)
from omnibase_infra.runtime.service_pattern_b_broker import _status_for_terminal_topic
from omnibase_infra.validators.contract_result_applier_construction import (
    scan,
    scan_source,
)

COMMAND_TOPIC = "onex.cmd.omnimarket.demo-skill.v1"
COMPLETED_TOPIC = "onex.evt.omnimarket.demo-skill-completed.v1"
FAILED_TOPIC = "onex.evt.omnimarket.demo-skill-failed.v1"

# Same shape as node_delegate_skill_orchestrator: a top-level terminal_event
# naming the SUCCESS terminal, a runtime_dispatch terminal_events map naming
# both, and published_events for the two typed terminal classes.
_TWO_TERMINAL_CONTRACT = f"""
name: node_demo_skill_orchestrator
version: 1.0.0
node_type: orchestrator
description: Two-terminal fixture contract (OMN-15468 kernel applier seam).

runtime_dispatch:
  terminal_events:
    success: {COMPLETED_TOPIC}
    failure: {FAILED_TOPIC}

event_bus:
  version: {{major: 1, minor: 0, patch: 0}}
  subscribe_topics:
    - {COMMAND_TOPIC}
  publish_topics:
    - {COMPLETED_TOPIC}
    - {FAILED_TOPIC}

terminal_event: {COMPLETED_TOPIC}

published_events:
  - event_type: DemoSkillCompleted
    topic: {COMPLETED_TOPIC}
    description: Demo skill completed.
  - event_type: DemoSkillFailed
    topic: {FAILED_TOPIC}
    description: Demo skill failed.
""".strip()


class ModelDemoSkillResponse(BaseModel):
    """A def-B return value whose CLASS misses ``published_events``.

    This is the live shape: ``ModelDelegateSkillResponse`` states its verdict in
    a ``status`` field, and the base class is not one of the two declared
    ``published_events`` entries.
    """

    status: str
    correlation_id: UUID
    terminal_failure_cause: str | None = None


class ModelDemoSkillFailed(BaseModel):
    """A def-B return value whose CLASS IS declared in ``published_events``."""

    correlation_id: UUID


def _write_contract(tmp_path: Path) -> Path:
    node_dir = tmp_path / "demopkg" / "nodes" / "node_demo_skill_orchestrator"
    node_dir.mkdir(parents=True)
    contract_path = node_dir / "contract.yaml"
    contract_path.write_text(_TWO_TERMINAL_CONTRACT, encoding="utf-8")
    return contract_path


def _dispatch_result(
    correlation_id: UUID, output_event: BaseModel
) -> ModelDispatchResult:
    return ModelDispatchResult(
        status=EnumDispatchStatus.SUCCESS,
        topic=COMMAND_TOPIC,
        started_at=datetime.now(UTC),
        correlation_id=correlation_id,
        dispatcher_id="omn15468-kernel-seam",
        output_events=[output_event],
    )


async def _published_topic(
    applier: object, correlation_id: UUID, event: BaseModel
) -> str:
    await applier.apply(  # type: ignore[attr-defined]
        _dispatch_result(correlation_id, event),
        correlation_id=correlation_id,
    )
    bus = applier._event_bus  # type: ignore[attr-defined]
    bus.publish_envelope.assert_awaited_once()
    return str(bus.publish_envelope.await_args.kwargs["topic"])


def _applier(contract_path: Path) -> object:
    return build_contract_result_applier(
        event_bus=AsyncMock(spec=ProtocolEventBusLike),
        contract_path=contract_path,
        publish_topics=[COMPLETED_TOPIC, FAILED_TOPIC],
        terminal_event=COMPLETED_TOPIC,
    )


@pytest.mark.unit
def test_factory_derives_both_contract_routing_inputs(tmp_path: Path) -> None:
    """The factory reads BOTH re-route inputs off the contract."""
    contract_path = _write_contract(tmp_path)
    applier = _applier(contract_path)

    assert applier._output_topic == COMPLETED_TOPIC  # type: ignore[attr-defined]
    assert applier._output_topic_map == {  # type: ignore[attr-defined]
        "DemoSkillCompleted": COMPLETED_TOPIC,
        "DemoSkillFailed": FAILED_TOPIC,
    }
    assert applier._failure_terminal_topics == (FAILED_TOPIC,)  # type: ignore[attr-defined]


@pytest.mark.unit
def test_declared_failure_terminal_topics_reads_the_contract(tmp_path: Path) -> None:
    """The path-keyed reader is the one both the kernel and auto-wiring use."""
    contract_path = _write_contract(tmp_path)
    assert declared_failure_terminal_topics(
        contract_path,
        success_topic=COMPLETED_TOPIC,
        publishable_topics=[COMPLETED_TOPIC, FAILED_TOPIC],
    ) == (FAILED_TOPIC,)
    # A topic the contract cannot publish is never a re-route destination.
    assert (
        declared_failure_terminal_topics(
            contract_path,
            success_topic=COMPLETED_TOPIC,
            publishable_topics=[COMPLETED_TOPIC],
        )
        == ()
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_map_miss_failure_verdict_lands_on_the_failure_terminal(
    tmp_path: Path,
) -> None:
    """The live defect: a map-miss return stating a failure must not go to -completed."""
    contract_path = _write_contract(tmp_path)
    correlation_id = uuid4()
    applier = _applier(contract_path)

    topic = await _published_topic(
        applier,
        correlation_id,
        ModelDemoSkillResponse(
            status="failed",
            correlation_id=correlation_id,
            terminal_failure_cause="provider_quota_exhausted",
        ),
    )
    assert topic == FAILED_TOPIC, (
        "a return value stating status='failed' with a typed "
        "terminal_failure_cause was republished onto the SUCCESS terminal — "
        "the exact live record shape read off the .201 dev lane"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_success_verdict_still_lands_on_the_success_terminal(
    tmp_path: Path,
) -> None:
    """Positive control: the guard corrects only the false-success direction."""
    contract_path = _write_contract(tmp_path)
    correlation_id = uuid4()
    applier = _applier(contract_path)

    topic = await _published_topic(
        applier,
        correlation_id,
        ModelDemoSkillResponse(status="completed", correlation_id=correlation_id),
    )
    assert topic == COMPLETED_TOPIC


@pytest.mark.unit
@pytest.mark.asyncio
async def test_declared_failure_class_routes_by_class_identity(
    tmp_path: Path,
) -> None:
    """Class-based routing is alive: an empty published_events map killed it."""
    contract_path = _write_contract(tmp_path)
    correlation_id = uuid4()
    applier = _applier(contract_path)

    topic = await _published_topic(
        applier, correlation_id, ModelDemoSkillFailed(correlation_id=correlation_id)
    )
    assert topic == FAILED_TOPIC


@pytest.mark.unit
@pytest.mark.asyncio
async def test_failure_terminal_record_reads_as_failed_after_the_wire(
    tmp_path: Path,
) -> None:
    """ok must reflect the verdict: the re-routed record derives status=failed."""
    contract_path = _write_contract(tmp_path)
    correlation_id = uuid4()
    applier = _applier(contract_path)
    event = ModelDemoSkillResponse(
        status="failed",
        correlation_id=correlation_id,
        terminal_failure_cause="provider_error",
    )
    topic = await _published_topic(applier, correlation_id, event)

    envelope = applier._event_bus.publish_envelope.await_args.kwargs["envelope"]  # type: ignore[attr-defined]
    decoded = ModelEventEnvelope[object].model_validate_json(
        envelope.model_dump_json().encode("utf-8")
    )
    decoded_payload = decoded.payload
    assert isinstance(decoded_payload, dict)
    assert resolve_terminal_verdict(decoded_payload) is False

    route = ModelRuntimeLocalIngressRoute(
        node_name="node_demo_skill_orchestrator",
        contract_name="node_demo_skill_orchestrator",
        command_topic=COMMAND_TOPIC,
        event_type="omnimarket.demo-skill",
        terminal_event=COMPLETED_TOPIC,
        contract_path=str(contract_path),
        package_name="demopkg",
        terminal_events=(COMPLETED_TOPIC, FAILED_TOPIC),
    )
    assert _status_for_terminal_topic(route, topic, decoded_payload) == "failed"


@pytest.mark.unit
def test_static_factory_requires_the_failure_terminal_answer() -> None:
    """A call site with no contract path must STATE its failure terminals."""
    with pytest.raises(TypeError):
        build_static_result_applier(  # type: ignore[call-arg]
            event_bus=AsyncMock(spec=ProtocolEventBusLike),
            output_topic=COMPLETED_TOPIC,
        )
    applier = build_static_result_applier(
        event_bus=AsyncMock(spec=ProtocolEventBusLike),
        output_topic=COMPLETED_TOPIC,
        failure_terminal_topics=(FAILED_TOPIC,),
    )
    assert applier._failure_terminal_topics == (FAILED_TOPIC,)


@pytest.mark.unit
def test_no_direct_applier_construction_in_src() -> None:
    """Enforcement, not detection: the gate over the real source tree."""
    violations, scanned = scan(Path("src/omnibase_infra"))
    assert scanned > 100, f"vacuous scan: only {scanned} files"
    assert violations == [], [f"{v.path}:{v.line}" for v in violations]


@pytest.mark.unit
def test_validator_positive_control() -> None:
    """A zero-finding scan is only evidence when the scanner can find one."""
    violating = "applier = DispatchResultApplier(event_bus=bus, output_topic='t')\n"
    assert len(scan_source(violating, Path("fake.py"))) == 1
    clean = (
        "applier = build_contract_result_applier(\n"
        "    event_bus=bus, contract_path=p, publish_topics=t\n"
        ")\n"
    )
    assert scan_source(clean, Path("fake.py")) == []
