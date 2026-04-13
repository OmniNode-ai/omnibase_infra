# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for topic suffix compliance in contract registration router (OMN-8605).

Verifies that:
1. New SUFFIX_* constants added in OMN-8605 are importable from the topics package.
2. ContractRegistrationEventRouter uses canonical suffix constants (not hardcoded strings).
3. The router's topic patterns match the expected canonical values.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


class TestNewSuffixConstantsExported:
    """Verify all new SUFFIX_* constants added by OMN-8605 are exported from topics."""

    def test_git_hook_suffix_importable(self) -> None:
        from omnibase_infra.topics import SUFFIX_GIT_HOOK

        assert SUFFIX_GIT_HOOK == "onex.evt.git.hook.v1"

    def test_linear_snapshot_suffix_importable(self) -> None:
        from omnibase_infra.topics import SUFFIX_LINEAR_SNAPSHOT

        assert SUFFIX_LINEAR_SNAPSHOT == "onex.evt.linear.snapshot.v1"

    def test_omniclaude_agent_match_suffix_importable(self) -> None:
        from omnibase_infra.topics import SUFFIX_OMNICLAUDE_AGENT_MATCH

        assert SUFFIX_OMNICLAUDE_AGENT_MATCH == "onex.evt.omniclaude.agent-match.v1"

    def test_omniclaude_context_utilization_suffix_importable(self) -> None:
        from omnibase_infra.topics import SUFFIX_OMNICLAUDE_CONTEXT_UTILIZATION

        assert SUFFIX_OMNICLAUDE_CONTEXT_UTILIZATION == (
            "onex.evt.omniclaude.context-utilization.v1"
        )

    def test_omniclaude_latency_breakdown_suffix_importable(self) -> None:
        from omnibase_infra.topics import SUFFIX_OMNICLAUDE_LATENCY_BREAKDOWN

        assert SUFFIX_OMNICLAUDE_LATENCY_BREAKDOWN == (
            "onex.evt.omniclaude.latency-breakdown.v1"
        )

    def test_omniclaude_manifest_injected_suffix_importable(self) -> None:
        from omnibase_infra.topics import SUFFIX_OMNICLAUDE_MANIFEST_INJECTED

        assert SUFFIX_OMNICLAUDE_MANIFEST_INJECTED == (
            "onex.evt.omniclaude.manifest-injected.v1"
        )

    def test_omniclaude_manifest_injection_failed_suffix_importable(self) -> None:
        from omnibase_infra.topics import SUFFIX_OMNICLAUDE_MANIFEST_INJECTION_FAILED

        assert SUFFIX_OMNICLAUDE_MANIFEST_INJECTION_FAILED == (
            "onex.evt.omniclaude.manifest-injection-failed.v1"
        )

    def test_omniclaude_manifest_injection_started_suffix_importable(self) -> None:
        from omnibase_infra.topics import SUFFIX_OMNICLAUDE_MANIFEST_INJECTION_STARTED

        assert SUFFIX_OMNICLAUDE_MANIFEST_INJECTION_STARTED == (
            "onex.evt.omniclaude.manifest-injection-started.v1"
        )

    def test_all_new_suffixes_are_non_empty_strings(self) -> None:
        from omnibase_infra.topics import (
            SUFFIX_GIT_HOOK,
            SUFFIX_LINEAR_SNAPSHOT,
            SUFFIX_OMNICLAUDE_AGENT_MATCH,
            SUFFIX_OMNICLAUDE_CONTEXT_UTILIZATION,
            SUFFIX_OMNICLAUDE_LATENCY_BREAKDOWN,
            SUFFIX_OMNICLAUDE_MANIFEST_INJECTED,
            SUFFIX_OMNICLAUDE_MANIFEST_INJECTION_FAILED,
            SUFFIX_OMNICLAUDE_MANIFEST_INJECTION_STARTED,
        )

        new_suffixes = [
            SUFFIX_GIT_HOOK,
            SUFFIX_LINEAR_SNAPSHOT,
            SUFFIX_OMNICLAUDE_AGENT_MATCH,
            SUFFIX_OMNICLAUDE_CONTEXT_UTILIZATION,
            SUFFIX_OMNICLAUDE_LATENCY_BREAKDOWN,
            SUFFIX_OMNICLAUDE_MANIFEST_INJECTED,
            SUFFIX_OMNICLAUDE_MANIFEST_INJECTION_FAILED,
            SUFFIX_OMNICLAUDE_MANIFEST_INJECTION_STARTED,
        ]
        for suffix in new_suffixes:
            assert isinstance(suffix, str) and suffix, (
                f"suffix must be a non-empty string, got: {suffix!r}"
            )
            assert suffix.startswith("onex."), (
                f"suffix must follow onex. naming convention, got: {suffix!r}"
            )


class TestContractRegistrationRouterTopics:
    """Verify ContractRegistrationEventRouter uses canonical suffix constants."""

    def test_router_topic_patterns_match_canonical_suffixes(self) -> None:
        from omnibase_infra.runtime.contract_registration_event_router import (
            TOPIC_SUFFIX_CONTRACT_DEREGISTERED,
            TOPIC_SUFFIX_CONTRACT_REGISTERED,
            TOPIC_SUFFIX_NODE_HEARTBEAT,
        )
        from omnibase_infra.topics import (
            SUFFIX_CONTRACT_DEREGISTERED,
            SUFFIX_CONTRACT_REGISTERED,
            SUFFIX_NODE_HEARTBEAT,
        )

        assert TOPIC_SUFFIX_CONTRACT_REGISTERED == SUFFIX_CONTRACT_REGISTERED
        assert TOPIC_SUFFIX_CONTRACT_DEREGISTERED == SUFFIX_CONTRACT_DEREGISTERED
        assert TOPIC_SUFFIX_NODE_HEARTBEAT == SUFFIX_NODE_HEARTBEAT

    def test_router_topic_values_are_canonical_strings(self) -> None:
        from omnibase_infra.runtime.contract_registration_event_router import (
            TOPIC_SUFFIX_CONTRACT_DEREGISTERED,
            TOPIC_SUFFIX_CONTRACT_REGISTERED,
            TOPIC_SUFFIX_NODE_HEARTBEAT,
        )

        assert TOPIC_SUFFIX_CONTRACT_REGISTERED == (
            "onex.evt.platform.contract-registered.v1"
        )
        assert TOPIC_SUFFIX_CONTRACT_DEREGISTERED == (
            "onex.evt.platform.contract-deregistered.v1"
        )
        assert TOPIC_SUFFIX_NODE_HEARTBEAT == "onex.evt.platform.node-heartbeat.v1"


__all__: list[str] = [
    "TestNewSuffixConstantsExported",
    "TestContractRegistrationRouterTopics",
]
