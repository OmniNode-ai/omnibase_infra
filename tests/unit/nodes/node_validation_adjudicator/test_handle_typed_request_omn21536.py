# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21536: adjudication handles one typed request."""

import inspect
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.enums import EnumCheckSeverity, EnumValidationVerdict
from omnibase_infra.handlers.handler_validation_adjudicate import HandlerAdjudicate
from omnibase_infra.models.validation.model_check_result import ModelCheckResult
from omnibase_infra.models.validation.model_validation_adjudication_requested import (
    ModelValidationAdjudicationRequested,
)


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
