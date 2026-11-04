# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Reproduce validation receipts from each handler's own serialized inputs.

The recorder observes a fresh handler import plus execution, as pytest-cov does
in CI. Database responses are explicit replay inputs; no service is contacted.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from pydantic import BaseModel, ConfigDict

from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumCheckSeverity, EnumHandlerTypeCategory
from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.models.validation import (
    ModelPlannedCheck,
    ModelValidationExecutionRequested,
)
from omnibase_infra.models.validation_ledger import ModelPayloadValidationLedgerAppend
from omnibase_infra.runtime.dispatch_envelope_context import bind_dispatch_envelope
from scripts.ci.adequacy_receipt import ModelAdequacyReceipt, build_receipt

ROOT = Path(__file__).resolve().parents[3]
RECEIPTS = ROOT / "scripts/ci/adequacy_receipts"
STAMP = datetime(2026, 10, 31, tzinfo=UTC)


class ModelReplayInput(BaseModel):
    """One request and its explicit dispatch/database boundary conditions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request: ModelValidationExecutionRequested | ModelPayloadValidationLedgerAppend
    bound: bool = True
    route: Literal["handle", "execute"] = "handle"
    database: Literal["insert", "duplicate", "no_result", "no_dsn"] = "insert"


def executor_inputs() -> tuple[ModelReplayInput, ...]:
    request = ModelValidationExecutionRequested(
        plan_id=UUID(int=1),
        candidate_id=UUID(int=2),
        checks=tuple(
            ModelPlannedCheck(
                check_code=f"C-{index}",
                label=f"Check {index}",
                severity=EnumCheckSeverity.REQUIRED,
                enabled=enabled,
                command="true",
            )
            for index, enabled in enumerate((True, False))
        ),
    )
    return (
        ModelReplayInput(request=request),
        ModelReplayInput(
            request=request.model_copy(update={"checks": ()}), bound=False
        ),
    )


def ledger_inputs() -> tuple[ModelReplayInput, ...]:
    request = ModelPayloadValidationLedgerAppend(
        run_id=UUID(int=3),
        repo_id="validation-replay",
        event_type="ValidationChecksCompleted",
        event_version="v1",
        occurred_at=STAMP,
        kafka_topic="validation-replay",
        kafka_partition=0,
        kafka_offset=42,
        envelope_bytes="SGVsbG8=",
        envelope_hash="a" * 64,
        correlation_id=UUID(int=4),
    )
    return (
        ModelReplayInput(request=request),
        ModelReplayInput(request=request, database="duplicate", bound=False),
        ModelReplayInput(
            request=request.model_copy(update={"correlation_id": None}), bound=False
        ),
        ModelReplayInput(request=request.model_copy(update={"correlation_id": None})),
        ModelReplayInput(request=request, database="no_result"),
        ModelReplayInput(request=request, database="no_dsn"),
        ModelReplayInput(request=request.model_copy(update={"envelope_bytes": "!"})),
        ModelReplayInput(request=request, route="execute"),
    )


def replay_executor(case: ModelReplayInput) -> None:
    module = importlib.reload(
        importlib.import_module(
            "omnibase_infra.nodes.node_validation_executor.handlers.handler_run_checks"
        )
    )
    handler = module.HandlerRunChecks()
    assert tuple(inspect.signature(handler.handle).parameters) == ("request",)
    assert handler.handler_id == "handler-run-checks"
    assert handler.handler_type.value == "infra_handler"
    assert handler.handler_category is EnumHandlerTypeCategory.EFFECT
    request = case.request
    assert isinstance(request, ModelValidationExecutionRequested)

    async def run() -> None:
        envelope = ModelEventEnvelope[object](
            payload=request, correlation_id=UUID(int=5)
        )
        with bind_dispatch_envelope(envelope) if case.bound else nullcontext():
            output = await handler.handle(request)
        assert output.plan_id == request.plan_id
        assert output.candidate_id == request.candidate_id
        assert output.score_threshold == request.score_threshold
        assert tuple((r.passed, r.skipped) for r in output.check_results) == tuple(
            (check.enabled, not check.enabled) for check in request.checks
        )

    asyncio.run(run())


def replay_ledger(case: ModelReplayInput) -> None:
    module = importlib.reload(
        importlib.import_module(
            "omnibase_infra.nodes.node_validation_ledger_write_effect.handlers."
            "handler_validation_ledger_append"
        )
    )
    handler = module.HandlerValidationLedgerAppend(MagicMock(spec=ModelONEXContainer))
    assert tuple(inspect.signature(handler.handle).parameters) == ("request",)
    assert handler.handler_type.value == "infra_handler"
    assert handler.handler_category is EnumHandlerTypeCategory.EFFECT
    db = AsyncMock()
    handler._db_handler = db
    rows = [{"id": str(UUID(int=6))}] if case.database == "insert" else []
    db.execute.return_value = MagicMock(
        result=None
        if case.database == "no_result"
        else MagicMock(payload=MagicMock(rows=rows))
    )
    request = case.request
    assert isinstance(request, ModelPayloadValidationLedgerAppend)

    async def run() -> None:
        if case.database == "no_dsn":
            with pytest.raises(RuntimeHostError, match="Missing PostgreSQL DSN"):
                await handler.handle(request)
            await handler.shutdown()
            return
        await handler.initialize({"dsn": "postgresql://replay"})
        envelope = ModelEventEnvelope[object](
            payload=request, correlation_id=UUID(int=5)
        )
        with bind_dispatch_envelope(envelope) if case.bound else nullcontext():
            if case.database == "no_result" or request.envelope_bytes == "!":
                expected = "no result" if case.database == "no_result" else "base64"
                with pytest.raises(RuntimeHostError, match=expected):
                    await handler.handle(request)
            else:
                output = (
                    await handler.handle(request)
                    if case.route == "handle"
                    else await handler.execute(
                        {
                            "payload": request.model_dump(mode="json"),
                            "correlation_id": str(UUID(int=5)),
                        }
                    )
                )
                assert output.result.success is True
                assert output.result.duplicate is (case.database == "duplicate")
                assert output.result.kafka_offset == request.kafka_offset
                assert (
                    db.execute.call_args.args[0]["payload"]["parameters"][8] == b"Hello"
                )
                if case.route == "handle":
                    assert db.execute.call_args.args[0]["correlation_id"] == str(
                        output.correlation_id
                    )
                    if case.bound:
                        assert output.input_envelope_id == envelope.envelope_id
                else:
                    with pytest.raises(RuntimeHostError, match="invalid 'payload'"):
                        await handler.execute({})
        await handler.shutdown()
        await handler.shutdown()

    asyncio.run(run())


def record_receipt(node: str) -> ModelAdequacyReceipt:
    executor = node == "validation_executor"
    module = (
        "omnibase_infra.nodes.node_validation_executor.handlers.handler_run_checks"
        if executor
        else "omnibase_infra.nodes.node_validation_ledger_write_effect.handlers."
        "handler_validation_ledger_append"
    )
    source = ROOT / "src" / Path(*module.split(".")).with_suffix(".py")
    receipt = build_receipt(
        node_id=f"omnibase_infra.nodes.node_{node}",
        handler_module_file=source,
        handler_call=replay_executor if executor else replay_ledger,
        candidates=executor_inputs() if executor else ledger_inputs(),
        source_match=str(source),
        volatile_mask=[
            "total_duration_ms",
            "check_results.executed_at",
            "check_results.duration_ms",
        ]
        if executor
        else ["input_envelope_id", "correlation_id"],
    )
    return receipt.model_copy(update={"handler_module": module})


@pytest.mark.parametrize(
    "node", ["validation_executor", "validation_ledger_write_effect"]
)
def test_replay_typed_request_behavior(node: str) -> None:
    if node == "validation_executor":
        for case in executor_inputs():
            replay_executor(case)
    else:
        for case in ledger_inputs():
            replay_ledger(case)


@pytest.mark.parametrize(
    "node", ["validation_executor", "validation_ledger_write_effect"]
)
def test_receipt_reproduces_handler_inputs(node: str) -> None:
    recorded = json.loads(
        (RECEIPTS / f"omnibase_infra.nodes.node_{node}.json").read_text()
    )
    measured = record_receipt(node).model_dump(mode="json", exclude_none=True)
    measured.pop("recorded_at")
    recorded.pop("recorded_at")
    assert measured["meets_target"] is True
    assert measured == recorded
    proof = json.loads(
        (RECEIPTS / f"omnibase_infra.nodes.node_{node}.handflip.json").read_text()
    )
    assert proof["parity"]["selected_input_hashes"] == measured["selected_input_hashes"]
