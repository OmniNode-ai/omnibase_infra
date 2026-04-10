# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for onboarding policy resolver (OMN-5268)."""

from __future__ import annotations

import pytest

from omnibase_infra.onboarding.loader import load_canonical_graph
from omnibase_infra.onboarding.policy_resolver import (
    UnsatisfiableTargetError,
    load_builtin_policies,
    resolve_policy,
)


@pytest.fixture
def canonical_graph():
    """Load the canonical graph for tests."""
    return load_canonical_graph()


class TestResolvePolicy:
    """Tests for resolve_policy()."""

    def test_standalone_quickstart_produces_5_steps(self, canonical_graph) -> None:
        steps = resolve_policy(
            canonical_graph,
            target_capabilities=["first_node_running"],
        )
        step_keys = [s.step_key for s in steps]
        assert len(steps) == 5
        assert step_keys == [
            "check_python",
            "install_uv",
            "install_core",
            "create_first_node",
            "run_standalone_node",
        ]

    def test_full_platform_produces_all_steps(self, canonical_graph) -> None:
        # Target all terminal capabilities to get all 10 steps
        steps = resolve_policy(
            canonical_graph,
            target_capabilities=[
                "omnidash_running",
                "secrets_configured",
                "first_node_running",
                "event_bus_connected",
            ],
        )
        assert len(steps) == 10

    def test_omnidash_only_produces_6_steps(self, canonical_graph) -> None:
        # Only the infra path is needed for omnidash + secrets
        steps = resolve_policy(
            canonical_graph,
            target_capabilities=["omnidash_running", "secrets_configured"],
        )
        assert len(steps) == 6

    def test_contributor_local(self, canonical_graph) -> None:
        steps = resolve_policy(
            canonical_graph,
            target_capabilities=["event_bus_connected"],
        )
        step_keys = [s.step_key for s in steps]
        assert "check_python" in step_keys
        assert "connect_node_to_bus" in step_keys
        # Should not include omnidash or secrets
        assert "start_omnidash" not in step_keys

    def test_topological_order(self, canonical_graph) -> None:
        steps = resolve_policy(
            canonical_graph,
            target_capabilities=["first_node_running"],
        )
        step_keys = [s.step_key for s in steps]
        # check_python must come before install_uv
        assert step_keys.index("check_python") < step_keys.index("install_uv")

    def test_unsatisfiable_target_raises(self, canonical_graph) -> None:
        with pytest.raises(UnsatisfiableTargetError, match="impossible_cap"):
            resolve_policy(
                canonical_graph,
                target_capabilities=["impossible_cap"],
            )

    def test_skip_steps(self, canonical_graph) -> None:
        steps = resolve_policy(
            canonical_graph,
            target_capabilities=["first_node_running"],
            skip_steps=["create_first_node", "run_standalone_node"],
        )
        step_keys = [s.step_key for s in steps]
        assert "create_first_node" not in step_keys
        assert "run_standalone_node" not in step_keys


class TestLoadBuiltinPolicies:
    """Tests for loading built-in policy YAML files."""

    def test_loads_three_policies(self) -> None:
        policies = load_builtin_policies()
        assert len(policies) == 4
        assert "standalone_quickstart" in policies
        assert "contributor_local" in policies
        assert "full_platform" in policies
        assert "new_employee" in policies

    def test_standalone_targets(self) -> None:
        policies = load_builtin_policies()
        standalone = policies["standalone_quickstart"]
        assert standalone["target_capabilities"] == ["first_node_running"]

    def test_full_platform_targets(self) -> None:
        policies = load_builtin_policies()
        full = policies["full_platform"]
        targets = full["target_capabilities"]
        assert isinstance(targets, list)
        assert "omnidash_running" in targets
        assert "secrets_configured" in targets
