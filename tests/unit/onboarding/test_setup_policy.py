# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for setup onboarding policy (OMN-11051)."""

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
_POLICY_FILE = _POLICIES_DIR / "setup.yaml"

_FORBIDDEN_CAPABILITIES = frozenset(
    {
        "first_node_running",
        "node_created",
        "event_bus_connected",
        "omnidash_running",
        "cloud_deployed",
        "hybrid_deployed",
    }
)


@pytest.fixture
def canonical_graph():
    return load_canonical_graph()


@pytest.fixture
def policy_data() -> dict:
    assert _POLICY_FILE.exists(), (
        f"setup.yaml not found at {_POLICY_FILE}. "
        "Run this test from the omnibase_infra worktree."
    )
    with _POLICY_FILE.open() as f:
        return yaml.safe_load(f)


class TestSetupPolicy:
    def test_policy_file_exists(self) -> None:
        assert _POLICY_FILE.exists(), f"setup.yaml not found at {_POLICY_FILE}"

    def test_policy_name(self, policy_data) -> None:
        assert policy_data["policy_name"] == "setup"

    def test_policy_version(self, policy_data) -> None:
        assert policy_data["policy_version"] == "1.0"

    def test_target_capabilities_present(self, policy_data) -> None:
        caps = policy_data.get("target_capabilities", [])
        assert isinstance(caps, list)
        assert len(caps) > 0

    def test_targets_core_installed_and_secrets_configured(self, policy_data) -> None:
        caps = policy_data["target_capabilities"]
        assert "core_installed" in caps
        assert "secrets_configured" in caps

    def test_all_target_capabilities_in_canonical_graph(
        self, policy_data, canonical_graph
    ) -> None:
        all_produced = {
            cap for step in canonical_graph.steps for cap in step.produces_capabilities
        }
        for cap in policy_data["target_capabilities"]:
            assert cap in all_produced, (
                f"Capability '{cap}' declared in setup policy "
                f"but not produced by any step in canonical.yaml. "
                f"Available capabilities: {sorted(all_produced)}"
            )

    def test_resolve_policy_returns_toolchain_steps(
        self, policy_data, canonical_graph
    ) -> None:
        steps = resolve_policy(
            canonical_graph, policy_data["target_capabilities"], None
        )
        step_keys = {s.step_key for s in steps}
        assert "install_uv" in step_keys
        assert "install_core" in step_keys
        assert "check_secrets" in step_keys

    def test_exclusion_contract_no_node_creation_capabilities(
        self, policy_data, canonical_graph
    ) -> None:
        steps = resolve_policy(
            canonical_graph, policy_data["target_capabilities"], None
        )
        all_caps = {cap for s in steps for cap in s.produces_capabilities}
        violations = _FORBIDDEN_CAPABILITIES & all_caps
        assert not violations, (
            f"Setup policy resolves steps that produce node-creation capabilities: "
            f"{violations}. The setup policy must target environment verification "
            f"only, not node creation."
        )

    def test_setup_policy_registered_in_builtin_policies(self) -> None:
        policies = load_builtin_policies()
        assert "setup" in policies, (
            f"'setup' not found in load_builtin_policies(). "
            f"Available: {sorted(policies)}"
        )
