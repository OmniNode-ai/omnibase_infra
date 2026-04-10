# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for new_employee onboarding policy (OMN-8271)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.onboarding.loader import load_canonical_graph
from omnibase_infra.onboarding.policy_resolver import (
    load_builtin_policies,
    resolve_policy,
)

# Path to the policies directory in this source tree (worktree-aware).
_POLICIES_DIR = (
    Path(__file__).parents[3] / "src" / "omnibase_infra" / "onboarding" / "policies"
)
_NEW_EMPLOYEE_POLICY_FILE = _POLICIES_DIR / "new_employee.yaml"


@pytest.fixture
def canonical_graph():
    """Load the canonical graph for tests."""
    return load_canonical_graph()


@pytest.fixture
def new_employee_policy_data() -> dict:
    """Load new_employee policy directly from source tree (worktree-aware)."""
    assert _NEW_EMPLOYEE_POLICY_FILE.exists(), (
        f"new_employee.yaml not found at {_NEW_EMPLOYEE_POLICY_FILE}. "
        "Run this test from the omnibase_infra worktree."
    )
    with _NEW_EMPLOYEE_POLICY_FILE.open() as f:
        return yaml.safe_load(f)


class TestNewEmployeePolicy:
    """Tests for the new_employee onboarding policy."""

    def test_policy_file_exists(self) -> None:
        """new_employee.yaml must exist in the policies directory."""
        assert _NEW_EMPLOYEE_POLICY_FILE.exists(), (
            f"new_employee.yaml not found at {_NEW_EMPLOYEE_POLICY_FILE}"
        )

    def test_policy_name_is_new_employee(self, new_employee_policy_data) -> None:
        """policy_name field must be 'new_employee'."""
        assert new_employee_policy_data["policy_name"] == "new_employee"

    def test_target_capabilities_present(self, new_employee_policy_data) -> None:
        """Policy must declare target_capabilities as a non-empty list."""
        caps = new_employee_policy_data.get("target_capabilities", [])
        assert isinstance(caps, list)
        assert len(caps) > 0

    def test_all_target_capabilities_exist_in_canonical_graph(
        self, new_employee_policy_data, canonical_graph
    ) -> None:
        """All target capabilities must exist as produces_capabilities entries in canonical.yaml."""
        all_produced = set()
        for step in canonical_graph.steps:
            all_produced.update(step.produces_capabilities)

        for cap in new_employee_policy_data["target_capabilities"]:
            assert cap in all_produced, (
                f"Capability '{cap}' declared in new_employee policy "
                f"but not produced by any step in canonical.yaml. "
                f"Available capabilities: {sorted(all_produced)}"
            )

    def test_resolve_policy_returns_non_empty_steps(
        self, new_employee_policy_data, canonical_graph
    ) -> None:
        """resolve_policy() with new_employee targets must return at least 5 steps."""
        target_capabilities = new_employee_policy_data["target_capabilities"]
        steps = resolve_policy(canonical_graph, target_capabilities, None)
        assert len(steps) >= 5, (
            f"Expected at least 5 resolved steps for new_employee policy, got {len(steps)}: "
            f"{[s.step_key for s in steps]}"
        )

    def test_all_existing_policies_still_load(self) -> None:
        """Regression guard: all builtin policies must still load without error."""
        policies = load_builtin_policies()
        # At minimum the original 3 plus new_employee (4 total after merge)
        assert len(policies) >= 3
        for name in ["standalone_quickstart", "contributor_local", "full_platform"]:
            assert name in policies, (
                f"Existing policy '{name}' missing from load_builtin_policies()"
            )

    def test_resolve_policy_rejects_policy_name_as_positional_arg(
        self, canonical_graph
    ) -> None:
        """resolve_policy() does not accept policy_name as a keyword argument.

        This test enforces correct API usage: callers must call load_builtin_policies()
        first to get target_capabilities, then pass those to resolve_policy().
        """
        with pytest.raises(TypeError):
            resolve_policy(canonical_graph, policy_name="new_employee")  # type: ignore[call-arg]
