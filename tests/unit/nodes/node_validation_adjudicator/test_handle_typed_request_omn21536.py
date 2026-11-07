# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21536: adjudication handles one typed request."""

import inspect
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumCheckSeverity, EnumValidationVerdict
from omnibase_infra.handlers.handler_validation_adjudicate import HandlerAdjudicate
from omnibase_infra.models.validation.model_adjudicator_state import (
    ModelAdjudicatorState,
)
from omnibase_infra.models.validation.model_check_result import ModelCheckResult
from omnibase_infra.models.validation.model_validation_adjudication_requested import (
    ModelValidationAdjudicationRequested,
)
from omnibase_infra.models.validation.model_verdict import ModelVerdict
from omnibase_infra.runtime.dispatch_envelope_context import bind_dispatch_envelope


async def test_handle_typed_request_returns_pass_verdict() -> None:
    request = ModelValidationAdjudicationRequested(
        plan_id=uuid4(),
        candidate_id=uuid4(),
        check_results=(
            ModelCheckResult(
                check_code="CHECK-TEST-001",
                label="Unit tests",
                severity=EnumCheckSeverity.REQUIRED,
                passed=True,
                executed_at=datetime.now(UTC),
            ),
        ),
    )

    result = await HandlerAdjudicate().handle(request)

    assert result.plan_id == request.plan_id
    assert result.candidate_id == request.candidate_id
    assert result.verdict is EnumValidationVerdict.PASS
    assert result.score == 1.0
    assert result.total_checks == result.passed_checks == 1
    assert result.failed_checks == result.skipped_checks == 0
    assert result.blocking_failures == ()
    assert result.adjudicated_at.tzinfo is not None


def test_request_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ModelValidationAdjudicationRequested.model_validate(
            {"plan_id": uuid4(), "candidate_id": uuid4(), "unknown": True}
        )


def test_handle_signature() -> None:
    assert tuple(inspect.signature(HandlerAdjudicate().handle).parameters) == (
        "request",
    )


def _passing_request() -> ModelValidationAdjudicationRequested:
    return ModelValidationAdjudicationRequested(
        plan_id=uuid4(),
        candidate_id=uuid4(),
        check_results=(
            ModelCheckResult(
                check_code="CHECK-TEST-001",
                label="Unit tests",
                severity=EnumCheckSeverity.REQUIRED,
                passed=True,
                executed_at=datetime.now(UTC),
            ),
        ),
    )


def _spy_correlation(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    seen: list[object] = []
    real_started = ModelAdjudicatorState.with_adjudication_started
    real_from_state = ModelVerdict.from_state.__func__

    def started(
        self: ModelAdjudicatorState, event_id: UUID, correlation_id: UUID | None = None
    ) -> ModelAdjudicatorState:
        seen.append(correlation_id)
        return real_started(self, event_id, correlation_id)

    def from_state(cls: type[ModelVerdict], *args: object, **kwargs: object) -> object:
        seen.append(kwargs.get("correlation_id"))
        return real_from_state(cls, *args, **kwargs)

    monkeypatch.setattr(ModelAdjudicatorState, "with_adjudication_started", started)
    monkeypatch.setattr(ModelVerdict, "from_state", classmethod(from_state))
    return seen


async def test_dispatch_envelope_correlation_reaches_fsm_and_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _spy_correlation(monkeypatch)
    request = _passing_request()
    correlation_id = uuid4()
    envelope = ModelEventEnvelope[object](
        payload=request.model_dump(mode="json"), correlation_id=correlation_id
    )

    with bind_dispatch_envelope(envelope):
        result = await HandlerAdjudicate().handle(request)

    assert result.verdict is EnumValidationVerdict.PASS
    assert seen == [correlation_id, correlation_id]


async def test_standalone_call_carries_no_correlation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _spy_correlation(monkeypatch)

    await HandlerAdjudicate().handle(_passing_request())

    assert seen == [None, None]
