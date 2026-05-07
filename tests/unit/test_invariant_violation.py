# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for InvariantViolation."""

from uuid import uuid4

import pytest

from omnibase_core.enums import EnumCoreErrorCode
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import InvariantViolation, RuntimeHostError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)

pytestmark = pytest.mark.unit


def test_invariant_violation_is_exception() -> None:
    error = InvariantViolation(
        action_name="delete_branch",
        protocol_domain="code_repository",
    )

    assert isinstance(error, Exception)
    assert isinstance(error, RuntimeHostError)
    assert error.model.error_code == EnumCoreErrorCode.CONTRACT_VIOLATION


def test_invariant_violation_includes_action_name() -> None:
    error = InvariantViolation(
        action_name="delete_branch",
        protocol_domain="code_repository",
        allowed_actions=("create_branch", "open_pull_request"),
    )

    assert error.action_name == "delete_branch"
    assert "delete_branch" in str(error)
    assert error.model.context["action_name"] == "delete_branch"
    assert error.model.context["allowed_actions"] == (
        "create_branch",
        "open_pull_request",
    )


def test_invariant_violation_includes_protocol_domain() -> None:
    error = InvariantViolation(
        action_name="post_message",
        protocol_domain="notification",
    )

    assert error.protocol_domain == "notification"
    assert "notification" in str(error)
    assert error.model.context["protocol_domain"] == "notification"
    assert error.model.context["target_name"] == "notification"


def test_invariant_violation_explicit_correlation_id_overrides_context() -> None:
    context_correlation_id = uuid4()
    explicit_correlation_id = uuid4()
    context = ModelInfraErrorContext.with_correlation(
        correlation_id=context_correlation_id,
        transport_type=EnumInfraTransportType.HTTP,
        operation="incoming_request",
        target_name="external_api",
        namespace="tenant-a",
    )

    error = InvariantViolation(
        action_name="delete_branch",
        protocol_domain="code_repository",
        context=context,
        correlation_id=explicit_correlation_id,
    )

    assert error.model.correlation_id == explicit_correlation_id
    assert error.model.context["transport_type"] == EnumInfraTransportType.RUNTIME
    assert error.model.context["operation"] == "validate_allowed_action"
    assert error.model.context["target_name"] == "code_repository"
    assert error.model.context["namespace"] == "tenant-a"
