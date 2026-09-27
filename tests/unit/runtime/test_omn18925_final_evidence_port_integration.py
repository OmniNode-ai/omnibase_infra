# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Final K1-K6 dispatch port evidence composition regression."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from uuid import UUID

import pytest

from omnibase_core.models.dispatch.model_dispatch_bus_command import (
    ModelDispatchBusCommand,
)
from omnibase_core.models.dispatch.model_dispatch_bus_terminal_result import (
    ModelDispatchBusTerminalResult,
)
from omnibase_infra.runtime.models.model_delegation_terminal_evidence import (
    ModelDelegationTerminalEvidence,
)
from omnibase_infra.runtime.runtime_local_ingress import ModelRuntimeLocalIngressRoute
from omnibase_infra.runtime.service_delegation_dispatch_port import (
    RuntimeDelegationDispatchPort,
)
from omnibase_infra.runtime.service_pattern_b_broker import TerminalPayload

pytestmark = pytest.mark.unit

_SCRIPTS_DIR = Path(__file__).parents[3] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from finalize_delegation_projection_evidence import (
    build_final_k1_k6_terminal_evidence_sink,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _readback_command(code: str) -> list[str]:
    return [sys.executable, "-c", code, "{correlation_id}"]


def _evidence_inputs(tmp_path: Path, correlation_id: UUID) -> tuple[Path, Path]:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    captured: dict[str, dict[str, str]] = {}
    for name in (
        "request",
        "caller_response",
        "run_readback",
        "persisted_receipt",
        "stdout_receipt",
    ):
        path = evidence_dir / f"{name}.bin"
        payload = f"{name}-actual-issued-bytes".encode()
        path.write_bytes(payload)
        captured[name] = {"path": path.name, "sha256": sha256(payload).hexdigest()}
    _write_json(
        evidence_dir / "manifest.json",
        {
            "run_id": "actual-port-run",
            "correlation_id": str(correlation_id),
            "evidence": captured,
            "tenant_proof": {"state": "pending_authoritative_projection_readback"},
        },
    )
    plan = tmp_path / "plan.json"
    _write_json(
        plan,
        {
            "captured_run_id": "actual-port-run",
            "correlation_id": str(correlation_id),
            "readback_timeout_seconds": 5,
            "projection_readback_command": _readback_command(
                "import json,sys; print(json.dumps({'rows':[{'correlation_id':"
                "sys.argv[1], 'tenant_id':'tenant-k1-k6'}]}))"
            ),
            "issued_lane_tenant_identity_command": _readback_command(
                "import json; print(json.dumps({'tenant_id':'tenant-k1-k6'}))"
            ),
        },
    )
    return evidence_dir, plan


def _route() -> ModelRuntimeLocalIngressRoute:
    return ModelRuntimeLocalIngressRoute(
        node_name="node_delegation_orchestrator",
        contract_name="node_delegation_orchestrator",
        command_topic="onex.cmd.omnimarket.delegation-request.v1",
        event_type="omnimarket.delegation-request",
        terminal_event="onex.evt.omnimarket.delegation-completed.v1",
        terminal_events=(
            "onex.evt.omnimarket.delegation-completed.v1",
            "onex.evt.omnimarket.delegation-failed.v1",
        ),
        contract_path="/contracts/omnimarket/node_delegation_orchestrator.yaml",
        package_name="omnimarket",
    )


@pytest.mark.asyncio
async def test_actual_port_invokes_final_sink_after_raw_terminal_arrival(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    correlation_id = UUID("22222222-2222-2222-2222-222222222222")
    evidence_dir, plan = _evidence_inputs(tmp_path, correlation_id)
    route = _route()
    raw_envelope = json.dumps(
        {
            "event_type": route.terminal_events[0],
            "correlation_id": str(correlation_id),
            "tenant_id": "tenant-k1-k6",
            "payload": {"payload": {"content": "actual terminal"}},
        },
        separators=(",", ":"),
    ).encode()

    class FakeRuntimeBus:
        environment = "test"
        bootstrap_servers = "localhost:9092"

    class FakeBroker:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def dispatch_request(
            self,
            command: ModelDispatchBusCommand,
            *,
            terminal_observer: object | None = None,
        ) -> tuple[object, ModelDispatchBusTerminalResult]:
            terminal = TerminalPayload(
                payload={"payload": {"content": "actual terminal"}},
                topic=route.terminal_events[0],
                raw_envelope=raw_envelope,
                partition=2,
                offset="42",
            )
            assert terminal_observer is not None
            await terminal_observer(terminal)  # type: ignore[operator]
            return route, ModelDispatchBusTerminalResult(
                correlation_id=command.correlation_id,
                status="completed",
                payload=terminal.payload,
                completed_at=datetime.now(UTC),
            )

    monkeypatch.setattr(
        "omnibase_infra.runtime.service_delegation_dispatch_port.RuntimePatternBBroker",
        FakeBroker,
    )
    monkeypatch.setattr(
        "omnibase_infra.runtime.service_delegation_dispatch_port.resolve_bounded_delegation_route",
        lambda **_kwargs: None,
    )
    sink = build_final_k1_k6_terminal_evidence_sink(evidence_dir, plan)
    port = RuntimeDelegationDispatchPort(
        FakeRuntimeBus(),  # type: ignore[arg-type]
        routes={"delegation.orchestrate": route},
        terminal_evidence_sink=sink,
    )

    result = await port.dispatch(
        prompt="actual K1-K6 port dispatch",
        task_type="document",
        correlation_id=correlation_id,
        max_tokens=64,
        source_file_path=None,
        source_session_id=None,
        wait=True,
        tenant_id="tenant-k1-k6",
    )

    assert result["status"] == "completed"
    retained = evidence_dir / "actual-dispatch-terminal"
    assert (retained / "terminal-envelope.json").read_bytes() == raw_envelope
    # The observer only retains the terminal; bounded projection readbacks run
    # after the caller-visible port result so they cannot hold broker progress.
    assert not (evidence_dir / "tenant-projection-proof").exists()
    await sink.finalize_after_dispatch()
    proof = json.loads(
        (evidence_dir / "tenant-projection-proof" / "tenant-proof.json").read_text()
    )
    assert proof["correlation_id"] == str(correlation_id)
    assert proof["tenant_id"] == "tenant-k1-k6"


def _final_run_plan(tmp_path: Path, correlation_id: UUID, evidence_dir: Path) -> Path:
    finalization = tmp_path / "finalization.json"
    _write_json(
        finalization,
        {
            "captured_run_id": "final-k1-k6-run",
            "correlation_id": str(correlation_id),
            "readback_timeout_seconds": 5,
            "projection_readback_command": _readback_command(
                "import json,sys; print(json.dumps({'rows':[{'correlation_id':"
                "sys.argv[1], 'tenant_id':'tenant-k1-k6'}]}))"
            ),
            "issued_lane_tenant_identity_command": _readback_command(
                "import json; print(json.dumps({'tenant_id':'tenant-k1-k6'}))"
            ),
        },
    )
    plan = tmp_path / "dispatch-plan.json"
    _write_json(
        plan,
        {
            "run_id": "final-k1-k6-run",
            "correlation_id": str(correlation_id),
            "tenant_id": "tenant-k1-k6",
            "evidence_dir": str(evidence_dir),
            "finalization_execution_plan": str(finalization),
            "transport": {
                "bus_type": "kafka",
                "kafka_bootstrap_servers": "kafka.prepr.example:9092",
                "environment": "prepr",
                "consumer_group": "omn18925-final-evidence",
            },
            "dispatch": {
                "route_alias": "delegation.orchestrate",
                "correlation_id": str(correlation_id),
                "tenant_id": "tenant-k1-k6",
                "prompt": "actual request",
                "task_type": "delegation",
                "max_tokens": 16,
                "wait": True,
                "execution_timeout_seconds": 30,
                "terminal_delivery_margin_seconds": 5,
            },
        },
    )
    return plan


@pytest.mark.asyncio
async def test_final_composition_entrypoint_binds_port_and_same_dispatch_artifacts(
    tmp_path: Path,
) -> None:
    from run_k1_k6_delegation_evidence import run

    correlation_id = UUID("33333333-3333-3333-3333-333333333333")
    evidence_dir = tmp_path / "final-evidence"
    plan = _final_run_plan(tmp_path, correlation_id, evidence_dir)
    raw_envelope = json.dumps(
        {
            "correlation_id": str(correlation_id),
            "tenant_id": "tenant-k1-k6",
            "payload": {"actual": "terminal"},
        },
        separators=(",", ":"),
    ).encode()

    class FakeBus:
        def __init__(self) -> None:
            self.started = False
            self.closed = False

        async def start(self) -> None:
            self.started = True

        async def close(self) -> None:
            self.closed = True

    bus = FakeBus()

    class FakePort:
        def __init__(
            self, received_bus: object, *, terminal_evidence_sink: object
        ) -> None:
            assert received_bus is bus
            self._sink = terminal_evidence_sink

        async def dispatch(self, **kwargs: object) -> dict[str, object]:
            assert kwargs["correlation_id"] == correlation_id
            assert kwargs["tenant_id"] == "tenant-k1-k6"
            await self._sink(  # type: ignore[operator]
                ModelDelegationTerminalEvidence(
                    correlation_id=correlation_id,
                    tenant_id="tenant-k1-k6",
                    topic="onex.evt.omnimarket.delegation-completed.v1",
                    raw_envelope=raw_envelope,
                    encoding="utf-8",
                    partition=1,
                    offset="9",
                )
            )
            return {"status": "completed", "content": "actual caller output"}

    response = await run(
        plan,
        event_bus_factory=lambda **kwargs: bus,
        port_factory=FakePort,
    )
    assert response["status"] == "completed"
    assert bus.started and bus.closed
    manifest = json.loads((evidence_dir / "manifest.json").read_text())
    assert manifest["capture_kind"] == "runtime_delegation_port"
    assert json.loads(
        (
            evidence_dir / "actual-dispatch-terminal" / "terminal-envelope.json"
        ).read_text()
    ) == json.loads(raw_envelope)
    proof = json.loads(
        (evidence_dir / "tenant-projection-proof" / "tenant-proof.json").read_text()
    )
    assert proof["correlation_id"] == str(correlation_id)
    assert proof["tenant_id"] == "tenant-k1-k6"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("route_alias", "wrong.route", "canonical delegation route"),
        (
            "correlation_id",
            "55555555-5555-5555-5555-555555555555",
            "does not match plan",
        ),
        ("tenant_id", "wrong-tenant", "does not match planned lane tenant"),
    ],
)
def test_final_composition_rejects_mismatched_plan_before_constructing_transport(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    from run_k1_k6_delegation_evidence import run

    correlation_id = UUID("44444444-4444-4444-4444-444444444444")
    plan = _final_run_plan(tmp_path, correlation_id, tmp_path / "final-evidence")
    value_object = json.loads(plan.read_text())
    value_object["dispatch"][field] = value
    _write_json(plan, value_object)
    called = False

    def factory(**kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("transport must not be constructed")

    with pytest.raises(ValueError, match=message):
        import asyncio

        asyncio.run(run(plan, event_bus_factory=factory))
    assert not called


def test_final_composition_rejects_finalizer_identity_before_constructing_transport(
    tmp_path: Path,
) -> None:
    from run_k1_k6_delegation_evidence import run

    correlation_id = UUID("66666666-6666-6666-6666-666666666666")
    plan = _final_run_plan(tmp_path, correlation_id, tmp_path / "final-evidence")
    plan_value = json.loads(plan.read_text())
    finalization = Path(plan_value["finalization_execution_plan"])
    finalization_value = json.loads(finalization.read_text())
    finalization_value["correlation_id"] = "77777777-7777-7777-7777-777777777777"
    _write_json(finalization, finalization_value)
    called = False

    def factory(**kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("transport must not be constructed")

    with pytest.raises(ValueError, match="finalization plan correlation_id"):
        import asyncio

        asyncio.run(run(plan, event_bus_factory=factory))
    assert not called


def test_final_composition_rejects_non_waiting_dispatch_before_constructing_transport(
    tmp_path: Path,
) -> None:
    from run_k1_k6_delegation_evidence import run

    correlation_id = UUID("88888888-8888-8888-8888-888888888888")
    plan = _final_run_plan(tmp_path, correlation_id, tmp_path / "final-evidence")
    plan_value = json.loads(plan.read_text())
    plan_value["dispatch"]["wait"] = False
    _write_json(plan, plan_value)
    called = False

    def factory(**kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("transport must not be constructed")

    with pytest.raises(ValueError, match="requires wait=true"):
        import asyncio

        asyncio.run(run(plan, event_bus_factory=factory))
    assert not called


@pytest.mark.asyncio
async def test_final_composition_closes_constructed_transport_when_start_raises(
    tmp_path: Path,
) -> None:
    from run_k1_k6_delegation_evidence import run

    correlation_id = UUID("99999999-9999-9999-9999-999999999999")
    evidence_dir = tmp_path / "final-evidence"
    plan = _final_run_plan(tmp_path, correlation_id, evidence_dir)

    class PartlyStartedBus:
        closed = False

        async def start(self) -> None:
            raise RuntimeError("startup allocation failed")

        async def close(self) -> None:
            self.closed = True

    bus = PartlyStartedBus()
    with pytest.raises(RuntimeError, match="startup allocation failed"):
        await run(plan, event_bus_factory=lambda **kwargs: bus)
    assert bus.closed
    assert (evidence_dir / "composition-failure.json").is_file()
