# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for the contributor_cloud and contributor_hybrid onboarding policies."""

from __future__ import annotations

import pytest

from omnibase_infra.onboarding.loader import load_canonical_graph
from omnibase_infra.onboarding.policy_resolver import (
    load_builtin_policies,
    resolve_policy,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def canonical_graph():
    return load_canonical_graph()


@pytest.fixture
def policies():
    return load_builtin_policies()


class TestCloudPolicy:
    def test_cloud_policy_loaded(self, policies) -> None:
        assert "contributor_cloud" in policies
        assert policies["contributor_cloud"]["target_capabilities"] == [
            "cloud_deployed"
        ]

    def test_cloud_policy_resolves_with_prerequisites(
        self, canonical_graph, policies
    ) -> None:
        cloud = policies["contributor_cloud"]
        steps = resolve_policy(
            canonical_graph,
            target_capabilities=cloud["target_capabilities"],
        )
        step_keys = [s.step_key for s in steps]

        # Cloud-specific steps must be present
        assert "configure_cloud_credentials" in step_keys
        assert "deploy_to_cloud" in step_keys

        # Prerequisite chain must be present
        assert "check_python" in step_keys
        assert "install_uv" in step_keys
        assert "install_core" in step_keys
        assert "check_secrets" in step_keys
        assert "check_node_bus_connection" in step_keys

    def test_cloud_policy_topological_order(self, canonical_graph, policies) -> None:
        cloud = policies["contributor_cloud"]
        steps = resolve_policy(
            canonical_graph,
            target_capabilities=cloud["target_capabilities"],
        )
        step_keys = [s.step_key for s in steps]

        assert step_keys.index("check_secrets") < step_keys.index(
            "configure_cloud_credentials"
        )
        assert step_keys.index("configure_cloud_credentials") < step_keys.index(
            "deploy_to_cloud"
        )
        assert step_keys.index("check_node_bus_connection") < step_keys.index(
            "deploy_to_cloud"
        )

    def test_cloud_policy_excludes_omnidash(self, canonical_graph, policies) -> None:
        cloud = policies["contributor_cloud"]
        steps = resolve_policy(
            canonical_graph,
            target_capabilities=cloud["target_capabilities"],
        )
        step_keys = [s.step_key for s in steps]
        assert "check_omnidash" not in step_keys


class TestHybridPolicy:
    def test_hybrid_policy_loaded(self, policies) -> None:
        assert "contributor_hybrid" in policies
        assert policies["contributor_hybrid"]["target_capabilities"] == [
            "hybrid_deployed"
        ]

    def test_hybrid_policy_resolves_with_prerequisites(
        self, canonical_graph, policies
    ) -> None:
        hybrid = policies["contributor_hybrid"]
        steps = resolve_policy(
            canonical_graph,
            target_capabilities=hybrid["target_capabilities"],
        )
        step_keys = [s.step_key for s in steps]

        # Hybrid-specific steps must be present
        assert "configure_hybrid_split" in step_keys
        assert "deploy_hybrid" in step_keys

        # Hybrid depends on cloud credentials AND local docker infra
        assert "configure_cloud_credentials" in step_keys
        assert "check_docker_infra" in step_keys
        assert "check_node_bus_connection" in step_keys

    def test_hybrid_policy_topological_order(self, canonical_graph, policies) -> None:
        hybrid = policies["contributor_hybrid"]
        steps = resolve_policy(
            canonical_graph,
            target_capabilities=hybrid["target_capabilities"],
        )
        step_keys = [s.step_key for s in steps]

        assert step_keys.index("configure_cloud_credentials") < step_keys.index(
            "configure_hybrid_split"
        )
        assert step_keys.index("check_docker_infra") < step_keys.index(
            "configure_hybrid_split"
        )
        assert step_keys.index("configure_hybrid_split") < step_keys.index(
            "deploy_hybrid"
        )

    def test_hybrid_policy_excludes_pure_cloud_terminal(
        self, canonical_graph, policies
    ) -> None:
        """Hybrid resolves deploy_hybrid, not the standalone deploy_to_cloud terminal."""
        hybrid = policies["contributor_hybrid"]
        steps = resolve_policy(
            canonical_graph,
            target_capabilities=hybrid["target_capabilities"],
        )
        step_keys = [s.step_key for s in steps]
        assert "deploy_to_cloud" not in step_keys


class TestCanonicalGraphCloudHybridSteps:
    """Verify the canonical graph itself has the cloud + hybrid step shape."""

    def test_cloud_credentials_depends_on_secrets(self, canonical_graph) -> None:
        step = next(
            s
            for s in canonical_graph.steps
            if s.step_key == "configure_cloud_credentials"
        )
        assert "check_secrets" in step.depends_on
        assert "cloud_credentials_configured" in step.produces_capabilities

    def test_deploy_to_cloud_depends_on_credentials_and_bus(
        self, canonical_graph
    ) -> None:
        step = next(s for s in canonical_graph.steps if s.step_key == "deploy_to_cloud")
        assert "configure_cloud_credentials" in step.depends_on
        assert "check_node_bus_connection" in step.depends_on
        assert "cloud_deployed" in step.produces_capabilities

    def test_hybrid_split_depends_on_cloud_creds_and_docker(
        self, canonical_graph
    ) -> None:
        step = next(
            s for s in canonical_graph.steps if s.step_key == "configure_hybrid_split"
        )
        assert "configure_cloud_credentials" in step.depends_on
        assert "check_docker_infra" in step.depends_on
        assert "hybrid_split_configured" in step.produces_capabilities

    def test_deploy_hybrid_depends_on_split_and_bus(self, canonical_graph) -> None:
        step = next(s for s in canonical_graph.steps if s.step_key == "deploy_hybrid")
        assert "configure_hybrid_split" in step.depends_on
        assert "check_node_bus_connection" in step.depends_on
        assert "hybrid_deployed" in step.produces_capabilities
