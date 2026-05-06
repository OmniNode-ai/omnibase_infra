# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration coverage for the exported InvariantViolation error."""

import pytest

from omnibase_infra.errors import InvariantViolation

pytestmark = pytest.mark.integration


def test_invariant_violation_export_preserves_structured_context() -> None:
    error = InvariantViolation(
        action_name="delete_branch",
        protocol_domain="code_repository",
        allowed_actions=("create_branch",),
    )

    assert error.model.context["action_name"] == "delete_branch"
    assert error.model.context["protocol_domain"] == "code_repository"
    assert error.model.context["allowed_actions"] == ("create_branch",)
