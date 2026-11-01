# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Each enum-bound registry suffix is declared by a shipped node contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra import topics
from omnibase_infra.enums import generated
from omnibase_infra.topics import platform_topic_suffixes as registry
from omnibase_infra.topics.contract_topic_extractor import ContractTopicExtractor

pytestmark = pytest.mark.integration

_BINDINGS = (
    (
        "SUFFIX_INTELLIGENCE_LLM_CALL_COMPLETED",
        "EnumOmniintelligenceTopic",
        "EVT_LLM_CALL_COMPLETED_V1",
    ),
    (
        "SUFFIX_OMNIMEMORY_POLICY_STATE_UPDATED",
        "EnumOmnimemoryTopic",
        "EVT_POLICY_STATE_UPDATED_V1",
    ),
    (
        "SUFFIX_VALIDATION_CROSS_REPO_RUN_STARTED",
        "EnumValidationTopic",
        "EVT_CROSS_REPO_RUN_STARTED_V1",
    ),
    (
        "SUFFIX_VALIDATION_CROSS_REPO_VIOLATIONS_BATCH",
        "EnumValidationTopic",
        "EVT_CROSS_REPO_VIOLATIONS_BATCH_V1",
    ),
    (
        "SUFFIX_VALIDATION_CROSS_REPO_RUN_COMPLETED",
        "EnumValidationTopic",
        "EVT_CROSS_REPO_RUN_COMPLETED_V1",
    ),
    (
        "SUFFIX_OMNICLAUDE_NOTIFICATION_BLOCKED",
        "EnumOmniclaudeTopic",
        "EVT_NOTIFICATION_BLOCKED_V1",
    ),
    (
        "SUFFIX_OMNICLAUDE_NOTIFICATION_COMPLETED",
        "EnumOmniclaudeTopic",
        "EVT_NOTIFICATION_COMPLETED_V1",
    ),
)


def test_enum_suffixes_are_declared_by_node_contracts() -> None:
    """Each suffix is its enum member's value and a topic some node contract declares."""
    package_root = Path(registry.__file__).resolve().parents[1]
    declared = {
        extracted.topic
        for extracted in ContractTopicExtractor()
        .scan(package_root / "nodes")
        .all_topics
    }
    for name, enum_name, member_name in _BINDINGS:
        member = getattr(getattr(generated, enum_name), member_name)
        suffix = getattr(topics, name)
        assert type(suffix) is str, name
        assert suffix == member.value, name
        assert suffix in declared, name
