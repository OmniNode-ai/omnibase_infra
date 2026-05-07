# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for omnimarket_quickstart onboarding policy (OMN-10592)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.onboarding.loader import load_canonical_graph
from omnibase_infra.onboarding.policy_resolver import (
    load_builtin_policies,
    resolve_policy,
)

_POLICIES_DIR = (
    Path(__file__).parents[3] / "src" / "omnibase_infra" / "onboarding" / "policies"
)
_POLICY_FILE = _POLICIES_DIR / "omnimarket_quickstart.yaml"


@pytest.fixture
def canonical_graph():
    return load_canonical_graph()


@pytest.fixture
def policy_data() -> dict:
    assert _POLICY_FILE.exists(), (
        f"omnimarket_quickstart.yaml not found at {_POLICY_FILE}. "
        "Run this test from the omnibase_infra worktree."
    )
    with _POLICY_FILE.open() as f:
        return yaml.safe_load(f)


class TestOmnimarketQuickstartPolicy:
    def test_policy_file_exists(self) -> None:
        assert _POLICY_FILE.exists(), (
            f"omnimarket_quickstart.yaml not found at {_POLICY_FILE}"
        )

    def test_policy_name(self, policy_data) -> None:
        assert policy_data["policy_name"] == "omnimarket_quickstart"

    def test_target_capabilities_present(self, policy_data) -> None:
        caps = policy_data.get("target_capabilities", [])
        assert isinstance(caps, list)
        assert len(caps) > 0

    def test_all_target_capabilities_in_canonical_graph(
        self, policy_data, canonical_graph
    ) -> None:
        all_produced = {
            cap for step in canonical_graph.steps for cap in step.produces_capabilities
        }
        for cap in policy_data["target_capabilities"]:
            assert cap in all_produced, (
                f"Capability '{cap}' declared in omnimarket_quickstart policy "
                f"but not produced by any step in canonical.yaml. "
                f"Available capabilities: {sorted(all_produced)}"
            )

    def test_resolve_policy_returns_omnimarket_steps(
        self, policy_data, canonical_graph
    ) -> None:
        steps = resolve_policy(
            canonical_graph, policy_data["target_capabilities"], None
        )
        step_keys = {s.step_key for s in steps}
        assert "install_omnimarket" in step_keys
        assert "validate_omnimarket_config" in step_keys
        assert "register_omnimarket_node" in step_keys

    def test_resolve_policy_includes_prerequisite_steps(
        self, policy_data, canonical_graph
    ) -> None:
        steps = resolve_policy(
            canonical_graph, policy_data["target_capabilities"], None
        )
        step_keys = {s.step_key for s in steps}
        assert "install_uv" in step_keys
        assert "configure_secrets" in step_keys
        assert "connect_node_to_bus" in step_keys

    def test_all_existing_policies_still_load(self) -> None:
        policies = load_builtin_policies()
        for name in [
            "standalone_quickstart",
            "contributor_local",
            "full_platform",
            "new_employee",
            "omnimarket_quickstart",
        ]:
            assert name in policies, (
                f"Policy '{name}' missing from load_builtin_policies()"
            )
