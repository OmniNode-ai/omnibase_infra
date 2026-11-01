# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-backed enum values fail closed and preserve import semantics."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Literal

import pytest

from omnibase_infra.enums.contract_topic_value import contract_topic_value
from omnibase_infra.tools.contract_topic_extractor import ModelContractTopicEntry
from omnibase_infra.tools.topic_enum_generator import TopicEnumGenerator

pytestmark = pytest.mark.unit


def test_topic_comes_from_supplied_contract(tmp_path: Path) -> None:
    source = tmp_path / "contract.yaml"
    source.write_text(
        "event_bus:\n  publish_topics:\n    - onex.evt.example.some-event.v2\n"
    )
    assert (
        contract_topic_value(str(source), "example", "EVT_SOME_EVENT_V2")
        == "onex.evt.example.some-event.v2"
    )
    with pytest.raises(ValueError, match="must bind one contract topic"):
        contract_topic_value(str(source), "example", "EVT_MISSING_V2")


def test_normalization_collision_is_refused(tmp_path: Path) -> None:
    source = tmp_path / "contract.yaml"
    source.write_text(
        "event_bus:\n  publish_topics:\n    - onex.evt.example.some-event.v1\n    - onex.evt.example.some.event.v1\n"
    )
    with pytest.raises(ValueError, match="must bind one contract topic"):
        contract_topic_value(str(source), "example", "EVT_SOME_EVENT_V1")


def test_prose_cannot_replace_a_declaration(tmp_path: Path) -> None:
    source = tmp_path / "contract.yaml"
    source.write_text("description: onex.evt.example.some-event.v1\n")
    with pytest.raises(ValueError, match="must bind one contract topic"):
        contract_topic_value(str(source), "example", "EVT_SOME_EVENT_V1")


@pytest.mark.parametrize(
    "module", ["omnibase_infra.topics", "omnibase_infra.enums.generated"]
)
def test_fresh_import_order(module: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import {module}; from omnibase_infra.topics import SUFFIX_OMNICLAUDE_NOTIFICATION_BLOCKED; assert type(SUFFIX_OMNICLAUDE_NOTIFICATION_BLOCKED) is str",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_unreadable_contract_error_names_source() -> None:
    with pytest.raises(ValueError, match=r"nope/missing\.yaml: cannot read contract"):
        contract_topic_value("nope/missing.yaml", "p", "M")


def test_edited_contract_is_read_again(tmp_path: Path) -> None:
    source = tmp_path / "contract.yaml"
    source.write_text("event_bus:\n  publish_topics: [onex.evt.example.old.v1]\n")
    assert contract_topic_value(str(source), "example", "EVT_OLD_V1").endswith("old.v1")
    source.write_text("event_bus:\n  publish_topics: [onex.evt.example.new.v1]\n")
    assert contract_topic_value(str(source), "example", "EVT_NEW_V1").endswith("new.v1")
    with pytest.raises(ValueError, match="must bind one contract topic"):
        contract_topic_value(str(source), "example", "EVT_OLD_V1")


@pytest.mark.parametrize("contents", ["[invalid yaml", "- not a mapping"])
def test_invalid_contract_error_names_member(tmp_path: Path, contents: str) -> None:
    source = tmp_path / "contract.yaml"
    source.write_text(contents)
    with pytest.raises(ValueError, match=r"example\.EVT_EVENT_V1: .*contract\.yaml"):
        contract_topic_value(str(source), "example", "EVT_EVENT_V1")


def test_unreadable_contract_error_names_member(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"example\.EVT_EVENT_V1: .*missing\.yaml"):
        contract_topic_value(str(tmp_path / "missing.yaml"), "example", "EVT_EVENT_V1")


@pytest.mark.parametrize("kind", ["evt", "cmd", "intent", "dlq", "snapshot"])
@pytest.mark.parametrize("event", ["some-event", "some.event", "some_event"])
def test_runtime_binding_matches_generator(
    tmp_path: Path, kind: str, event: str
) -> None:
    topic = f"onex.{kind}.example.{event}.v12"
    source = tmp_path / "contract.yaml"
    source.write_text(f"event_bus:\n  publish_topics: [{topic}]\n")
    entry = ModelContractTopicEntry(
        topic=topic,
        kind=kind,
        producer="example",
        event_name=event,
        version="v12",
        source_contracts=(source,),
    )
    member = TopicEnumGenerator().render_member_key(entry)
    assert contract_topic_value(str(source), "example", member) == topic


def test_legacy_dlq_binding_matches_generator(tmp_path: Path) -> None:
    source = tmp_path / "topics.yaml"
    source.write_text("topics: [onex.dlq.events.v1]\n")
    assert (
        contract_topic_value(str(source), "dlq", "DLQ_EVENTS_V1")
        == "onex.dlq.events.v1"
    )
