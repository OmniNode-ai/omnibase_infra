# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cross-boundary seam test for OMN-15468 AC2 — verdict-aware terminal routing.

Drives the REAL definition-B publish seam end to end over a frozen forced-429
capture of ONE accepted command:

    handler return value
      -> DispatchResultApplier (publish-from-return, the artifact that runs)
      -> event bus record + topic
      -> ModelRuntimeLocalIngressRoute terminal set (discovered from the
         contract file, the same reader the Pattern B broker subscribes with)
      -> _status_for_terminal_topic
      -> the /skill ``ok`` derivation (``result.status == "completed"``,
         runtime_host_process.py)

Two independent unit suites — one on ``_apply_failure_terminal_guard``, one on
``_status_for_terminal_topic`` — would NOT satisfy the ticket's AC3; the whole
point is that each half was individually defensible while the pair produced a
false success. This test crosses the boundary.

RED before this change: ``_resolve_mapped_output_topic`` falls back to the
contract's SUCCESS terminal for any returned model whose class name misses the
``published_events`` map, regardless of the verdict on the payload. The
forced-429 terminal — ``status="failed"``,
``terminal_failure_cause="provider_quota_exhausted"``, every attempt refused
with HTTP 429 — was therefore published to ``…-completed.v1``, the broker
derived ``completed`` from the arrival topic, and ``/skill`` returned
``ok=true``. That is the live 2026-07-30 ``.201`` dev-lane reproduction recorded
on the ticket, and PR #2578's own scope note names this locus as the residual it
did not close.

GREEN: the failure-verdict return value is published to the contract's DECLARED
failure terminal, the broker terminalizes it as ``failed``, and the outer ``ok``
is ``False``.

Fixture: ``tests/fixtures/seams/quota_terminal/forced_429_terminal.json`` — the
same path and the same frozen capture used by the consumer-side seam test in
omnimarket (OMN-15503, omnimarket#1995) and the producer-side seam test
(OMN-15469). One capture, three boundaries.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from pydantic import BaseModel

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

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "seams"
    / "quota_terminal"
    / "forced_429_terminal.json"
)

SUCCESS_TOPIC = "onex.evt.omnimarket.delegate-skill-completed.v1"
FAILURE_TOPIC = "onex.evt.omnimarket.delegate-skill-failed.v1"
COMMAND_TOPIC = "onex.cmd.omnimarket.delegate-skill.v1"

_QUOTA_CONTRACT = f"""
name: delegate_skill_seam_demo
version: 1.0.0
node_type: ORCHESTRATOR_GENERIC
description: Frozen forced-429 seam fixture contract (OMN-15468).

runtime_dispatch:
  command_topic: {COMMAND_TOPIC}
  terminal_events:
    success: {SUCCESS_TOPIC}
    failure: {FAILURE_TOPIC}
  default_timeout_ms: 300000

event_bus:
  version: {{major: 1, minor: 0, patch: 0}}
  subscribe_topics:
    - {COMMAND_TOPIC}
  publish_topics:
    - {SUCCESS_TOPIC}
    - {FAILURE_TOPIC}

terminal_event: {SUCCESS_TOPIC}

handler_routing:
  routing_strategy: operation_match
  handlers:
    - operation: delegate-skill.orchestrate
""".strip()


class ModelQuotaTerminalReturn(BaseModel):
    """The definition-B return value, as the producer shapes it.

    Deliberately a class name that is NOT in the contract's ``published_events``
    map — that map-miss is the condition under which the applier falls back to
    the contract's success terminal, and it is the exact condition the live
    defect occurred under. A model that hits the map routes by class and never
    reaches the guard.
    """

    status: str
    correlation_id: UUID
    quality_gate_passed: bool = False
    terminal_failure_cause: str | None = None
    error_message: str = ""


def _load_fixture() -> dict[str, Any]:
    """Load the frozen forced-429 capture, failing loudly if it is absent."""
    if not FIXTURE_PATH.exists():  # pragma: no cover - guard
        raise AssertionError(f"forced-429 seam fixture missing: {FIXTURE_PATH}")
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _lying_outer_terminal(fixture: dict[str, Any]) -> dict[str, Any]:
    """Return the fixture's LAST terminal — the one that declared success.

    The capture records three terminals for one command; the last is the outer
    ``delegate-skill-completed`` claiming ``status=completed`` /
    ``quality_gate_passed=true`` while every inner attempt was 429'd. That
    record is the input this seam has to stop mis-routing.
    """
    return fixture["terminal_events"][-1]


def _write_contract(tmp_path: Path) -> tuple[Path, str]:
    package_root = tmp_path / "quotapkg"
    node_dir = package_root / "nodes" / "node_delegate_skill_seam_demo"
    node_dir.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    contract_path = node_dir / "contract.yaml"
    contract_path.write_text(_QUOTA_CONTRACT, encoding="utf-8")
    return contract_path, str(package_root / "__init__.py")


def _discovered_contract(contract_path: Path) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name="delegate_skill_seam_demo",
        node_type="ORCHESTRATOR_GENERIC",
        contract_version={"major": 1, "minor": 0, "patch": 0},
        contract_path=contract_path,
        entry_point_name="delegate_skill_seam_demo",
        package_name="quotapkg",
        terminal_event=SUCCESS_TOPIC,
        event_bus={
            "subscribe_topics": [COMMAND_TOPIC],
            "publish_topics": [SUCCESS_TOPIC, FAILURE_TOPIC],
        },
    )


@pytest.mark.asyncio
async def test_seam_quota_exhausted_terminal_is_not_reported_as_completed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A forced-429 return value must not terminalize as ``completed``.

    Four hops, all real, no hand-built stand-ins for the artifacts under test:

    1. the wiring reads the contract's declared FAILURE terminal through the
       same reader the broker's subscription set uses;
    2. the applier publishes the returned model — and must not put a
       failure-verdict payload on the success terminal;
    3. route discovery puts both terminals on the route;
    4. ``_status_for_terminal_topic`` + the ``/skill`` ``ok`` derivation report
       the failure.
    """
    fixture = _load_fixture()
    outer = _lying_outer_terminal(fixture)
    correlation_id = UUID(fixture["correlation_id"])
    contract_path, package_init = _write_contract(tmp_path)
    contract = _discovered_contract(contract_path)

    # Hop 1 — declared terminals, read the one canonical way.
    success_topic = _select_dispatch_result_output_topic(contract)
    assert success_topic == SUCCESS_TOPIC
    failure_topics = _declared_failure_terminal_topics(
        contract, success_topic=SUCCESS_TOPIC
    )
    assert failure_topics == (FAILURE_TOPIC,), (
        "the contract declares its failure terminal under "
        f"runtime_dispatch.terminal_events.failure; wiring read {failure_topics!r}"
    )

    # Hop 2 — the real applier, wired exactly as _subscribe_contract_topics wires it.
    event_bus = AsyncMock(spec=ProtocolEventBusLike)
    applier = DispatchResultApplier(
        event_bus=event_bus,
        output_topic=SUCCESS_TOPIC,
        # Empty on purpose: this contract's published_events declares no entry
        # for this return model's class, which is the map-miss fallback the
        # live defect took.
        output_topic_map={},
        allowed_output_topics=[SUCCESS_TOPIC, FAILURE_TOPIC],
        failure_terminal_topics=failure_topics,
    )
    returned = ModelQuotaTerminalReturn(
        # The producer's own claim, carried verbatim from the frozen capture —
        # this is the record that said "completed" over three 429s.
        status="failed",
        correlation_id=correlation_id,
        quality_gate_passed=bool(outer["quality_gate_passed"]),
        terminal_failure_cause=(
            fixture["expected_projection"]["terminal_failure_cause"]
        ),
        error_message=outer["error_message"],
    )
    await applier.apply(
        _make_success_result(correlation_id, returned),
        correlation_id=correlation_id,
    )

    event_bus.publish_envelope.assert_awaited_once()
    published_topic = event_bus.publish_envelope.await_args.kwargs["topic"]
    assert published_topic == FAILURE_TOPIC, (
        "a returned model declaring terminal_failure_cause="
        f"{returned.terminal_failure_cause!r} / status={returned.status!r} was "
        f"published to {published_topic!r}; the contract declares its failure "
        f"terminal at {FAILURE_TOPIC!r}"
    )

    # Hop 3 — route discovery sees both terminals.
    monkeypatch.setattr(
        "omnibase_infra.runtime.runtime_local_ingress.importlib.import_module",
        lambda _name: SimpleNamespace(__file__=package_init),
    )
    route = discover_runtime_local_ingress_routes(("quotapkg",))[
        "delegate_skill_seam_demo"
    ]
    assert FAILURE_TOPIC in route.terminal_events

    # Hop 4 — broker status, and the /skill ok derivation keyed off it
    # (runtime_host_process.py: ok=True iff result.status == "completed").
    status = _status_for_terminal_topic(route, published_topic)
    assert status == "failed", (
        f"terminal arrived on {published_topic!r}; broker reported {status!r}"
    )
    outer_ok = status == "completed"
    assert outer_ok is False, (
        "the /skill response for a forced-429 command must not be byte-"
        "comparable to a success response in its ok/status fields"
    )
    assert fixture["expected_projection"]["terminal_ok"] is False, (
        "fixture precondition: this capture expects a NOT-ok terminal"
    )


@pytest.mark.asyncio
async def test_success_verdict_return_still_publishes_to_success_terminal(
    tmp_path: Path,
) -> None:
    """The guard must not turn honest successes into failures.

    Same contract, same applier wiring, a return value whose verdict is
    positive: routing must be unchanged. A guard that fires on anything other
    than an explicit failure verdict would trade one lie for its mirror image.
    """
    contract_path, _ = _write_contract(tmp_path)
    contract = _discovered_contract(contract_path)
    event_bus = AsyncMock(spec=ProtocolEventBusLike)
    applier = DispatchResultApplier(
        event_bus=event_bus,
        output_topic=SUCCESS_TOPIC,
        output_topic_map={},
        allowed_output_topics=[SUCCESS_TOPIC, FAILURE_TOPIC],
        failure_terminal_topics=_declared_failure_terminal_topics(
            contract, success_topic=SUCCESS_TOPIC
        ),
    )
    correlation_id = UUID("c4f2a1de-0000-4d15-9503-000000000429")
    returned = ModelQuotaTerminalReturn(
        status="completed",
        correlation_id=correlation_id,
        quality_gate_passed=True,
    )

    await applier.apply(
        _make_success_result(correlation_id, returned),
        correlation_id=correlation_id,
    )

    assert event_bus.publish_envelope.await_args.kwargs["topic"] == SUCCESS_TOPIC


@pytest.mark.asyncio
async def test_verdictless_return_routing_is_unchanged(tmp_path: Path) -> None:
    """A model declaring no verdict at all routes exactly as before.

    This is the overwhelming majority of contracts. ``resolve_terminal_verdict``
    returns ``None`` for them and the guard must be a no-op — the change is a
    fail-closed correction for models that STATE a failure, never a guess about
    models that state nothing.
    """

    class ModelNoVerdict(BaseModel):
        value: str = "x"

    contract_path, _ = _write_contract(tmp_path)
    contract = _discovered_contract(contract_path)
    event_bus = AsyncMock(spec=ProtocolEventBusLike)
    applier = DispatchResultApplier(
        event_bus=event_bus,
        output_topic=SUCCESS_TOPIC,
        output_topic_map={},
        allowed_output_topics=[SUCCESS_TOPIC, FAILURE_TOPIC],
        failure_terminal_topics=_declared_failure_terminal_topics(
            contract, success_topic=SUCCESS_TOPIC
        ),
    )
    correlation_id = UUID("c4f2a1de-0000-4d15-9503-000000000429")

    await applier.apply(
        _make_success_result(correlation_id, ModelNoVerdict()),
        correlation_id=correlation_id,
    )

    assert event_bus.publish_envelope.await_args.kwargs["topic"] == SUCCESS_TOPIC


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
