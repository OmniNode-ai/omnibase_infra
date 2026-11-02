# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The suffix registry reads these SUFFIX_* values off the generated topic enums (OMN-21689).

Each constant below used to hold its topic as a string literal, which the
contracts_and_di ``literal_topic`` rule refuses in code. They now name a member of
the per-producer enum generated from contract YAML, so the topic is declared once,
in the owning contract, and the registry keeps the value it always had.
"""

from __future__ import annotations

import ast
from enum import Enum
from pathlib import Path

import pytest

from omnibase_infra import topics as topics_package
from omnibase_infra.enums import generated
from omnibase_infra.topics import platform_topic_suffixes

_REGISTRY_FILE = Path(platform_topic_suffixes.__file__)

# (constant name, "<generated enum>.<member>", the value the constant held before).
_ENUM_BOUND_SUFFIXES: tuple[tuple[str, str, str], ...] = (
    (
        "SUFFIX_INTELLIGENCE_LLM_CALL_COMPLETED",
        "EnumOmniintelligenceTopic.EVT_LLM_CALL_COMPLETED_V1",
        "onex.evt.omniintelligence.llm-call-completed.v1",
    ),
    (
        "SUFFIX_OMNIMEMORY_POLICY_STATE_UPDATED",
        "EnumOmnimemoryTopic.EVT_POLICY_STATE_UPDATED_V1",
        "onex.evt.omnimemory.policy-state-updated.v1",
    ),
    (
        "SUFFIX_VALIDATION_CROSS_REPO_RUN_STARTED",
        "EnumValidationTopic.EVT_CROSS_REPO_RUN_STARTED_V1",
        "onex.evt.validation.cross-repo-run-started.v1",
    ),
    (
        "SUFFIX_VALIDATION_CROSS_REPO_VIOLATIONS_BATCH",
        "EnumValidationTopic.EVT_CROSS_REPO_VIOLATIONS_BATCH_V1",
        "onex.evt.validation.cross-repo-violations-batch.v1",
    ),
    (
        "SUFFIX_VALIDATION_CROSS_REPO_RUN_COMPLETED",
        "EnumValidationTopic.EVT_CROSS_REPO_RUN_COMPLETED_V1",
        "onex.evt.validation.cross-repo-run-completed.v1",
    ),
    (
        "SUFFIX_OMNICLAUDE_NOTIFICATION_BLOCKED",
        "EnumOmniclaudeTopic.EVT_NOTIFICATION_BLOCKED_V1",
        "onex.evt.omniclaude.notification-blocked.v1",
    ),
    (
        "SUFFIX_OMNICLAUDE_NOTIFICATION_COMPLETED",
        "EnumOmniclaudeTopic.EVT_NOTIFICATION_COMPLETED_V1",
        "onex.evt.omniclaude.notification-completed.v1",
    ),
)


def _member(path: str) -> Enum:
    enum_name, member_name = path.split(".")
    enum_class = getattr(generated, enum_name)
    member: Enum = enum_class[member_name]
    return member


@pytest.mark.unit
class TestSuffixRegistryEnumBinding:
    @pytest.mark.parametrize(
        ("name", "member_path", "former_value"), _ENUM_BOUND_SUFFIXES
    )
    def test_constant_keeps_its_value_from_the_generated_enum(
        self, name: str, member_path: str, former_value: str
    ) -> None:
        value = getattr(platform_topic_suffixes, name)
        assert value == _member(member_path).value
        assert value == former_value

    def test_no_enum_bound_constant_is_assigned_a_string_literal(self) -> None:
        tree = ast.parse(_REGISTRY_FILE.read_text(encoding="utf-8"))
        bound = {name for name, _, _ in _ENUM_BOUND_SUFFIXES}
        literal_assignments = {
            node.target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in bound
            and isinstance(node.value, ast.Constant)
        }
        assert literal_assignments == set()

    def test_every_enum_bound_constant_is_exported(self) -> None:
        exported = set(topics_package.__all__)
        assert {name for name, _, _ in _ENUM_BOUND_SUFFIXES} <= exported
