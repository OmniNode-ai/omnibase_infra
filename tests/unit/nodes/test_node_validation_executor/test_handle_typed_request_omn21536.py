# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed request contract for validation check execution (OMN-21536)."""

from __future__ import annotations

import inspect
import logging
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.enums import EnumCheckSeverity
from omnibase_infra.models.validation import (
    ModelPlannedCheck,
    ModelValidationExecutionRequested,
)
from omnibase_infra.nodes.node_validation_executor.handlers.handler_run_checks import (
    HandlerRunChecks,
)


async def test_handle_typed_request(caplog: pytest.LogCaptureFixture) -> None:
    request = ModelValidationExecutionRequested(
        candidate_id=uuid4(),
        checks=(
            ModelPlannedCheck(
                check_code="C-1",
                label="Enabled check",
                severity=EnumCheckSeverity.REQUIRED,
                enabled=True,
                command="uv run pytest",
            ),
            ModelPlannedCheck(
                check_code="C-2",
                label="Disabled check",
                severity=EnumCheckSeverity.REQUIRED,
                enabled=False,
                command="uv run ruff check",
            ),
        ),
    )
    correlation_id = uuid4()
    handler = HandlerRunChecks(correlation_id=correlation_id)

    with caplog.at_level(logging.INFO, logger=HandlerRunChecks.__module__):
        result = await handler.handle(request)

    start_record = next(
        record
        for record in caplog.records
        if record.msg.startswith("Executing validation plan")
    )
    assert start_record.args[-1] is correlation_id
    assert result.plan_id == request.plan_id
    assert result.candidate_id == request.candidate_id
    assert result.score_threshold == request.score_threshold
    assert result.total_duration_ms >= 0.0
    assert len(result.check_results) == 2
    assert result.check_results[0].passed is True
    assert result.check_results[0].skipped is False
    assert result.check_results[1].passed is False
    assert result.check_results[1].skipped is True
    assert list(inspect.signature(handler.handle).parameters) == ["request"]
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ModelValidationExecutionRequested.model_validate(
            {**request.model_dump(), "unknown_field": True}
        )


async def test_handle_without_correlation_id() -> None:
    request = ModelValidationExecutionRequested(candidate_id=uuid4(), checks=())

    result = await HandlerRunChecks().handle(request)

    assert result.plan_id == request.plan_id
    assert result.check_results == ()
    assert result.pass_count == 0
    assert result.fail_count == 0
