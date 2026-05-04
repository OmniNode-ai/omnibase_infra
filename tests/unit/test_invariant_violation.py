# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for InvariantViolation."""

from omnibase_core.enums import EnumCoreErrorCode
from omnibase_infra.errors import InvariantViolation, RuntimeHostError


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
