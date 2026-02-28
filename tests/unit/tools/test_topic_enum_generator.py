# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
#
# Tests for TopicEnumGenerator (OMN-2964).
#
# All tests import via installed package — no sys.path manipulation.
# TDD: each test is a genuine assertion on intended behavior.

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.tools.contract_topic_extractor import ModelContractTopicEntry
from omnibase_infra.tools.topic_enum_generator import TopicEnumGenerator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _entry(
    topic: str,
    kind: str,
    producer: str,
    event_name: str,
    version: str,
    source: str = "/a/contract.yaml",
) -> ModelContractTopicEntry:
    """Helper to build a ModelContractTopicEntry for testing."""
    return ModelContractTopicEntry(
        topic=topic,
        kind=kind,  # type: ignore[arg-type]
        producer=producer,
        event_name=event_name,
        version=version,
        source_contracts=(Path(source),),
    )


# ---------------------------------------------------------------------------
# Normalization rules — documented examples
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_normalization_evt_hyphen(tmp_path: Path) -> None:
    """evt kind with hyphenated event-name → EVT_ prefix, underscores, _V1 suffix."""
    gen = TopicEnumGenerator()
    entry = _entry(
        "onex.evt.platform.intent-classified.v1",
        "evt",
        "platform",
        "intent-classified",
        "v1",
    )
    assert gen.render_member_key(entry) == "EVT_INTENT_CLASSIFIED_V1"


@pytest.mark.unit
def test_normalization_cmd_multi_hyphen(tmp_path: Path) -> None:
    """cmd kind with multi-hyphen event-name → CMD_ prefix."""
    gen = TopicEnumGenerator()
    entry = _entry(
        "onex.cmd.platform.intent-query-session.v1",
        "cmd",
        "platform",
        "intent-query-session",
        "v1",
    )
    assert gen.render_member_key(entry) == "CMD_INTENT_QUERY_SESSION_V1"


@pytest.mark.unit
def test_normalization_intent_kind(tmp_path: Path) -> None:
    """intent kind → INTENT_ prefix."""
    gen = TopicEnumGenerator()
    entry = _entry(
        "onex.intent.platform.runtime-tick.v1",
        "intent",
        "platform",
        "runtime-tick",
        "v1",
    )
    assert gen.render_member_key(entry) == "INTENT_RUNTIME_TICK_V1"


@pytest.mark.unit
def test_normalization_version_v2(tmp_path: Path) -> None:
    """Version v2 → _V2 suffix."""
    gen = TopicEnumGenerator()
    entry = _entry(
        "onex.evt.platform.fsm-state-transitions.v2",
        "evt",
        "platform",
        "fsm-state-transitions",
        "v2",
    )
    assert gen.render_member_key(entry) == "EVT_FSM_STATE_TRANSITIONS_V2"


@pytest.mark.unit
def test_normalization_dots_replaced(tmp_path: Path) -> None:
    """Dots in event_name are replaced with underscores."""
    gen = TopicEnumGenerator()
    entry = _entry(
        "onex.evt.platform.node.heartbeat.v1",
        "evt",
        "platform",
        "node.heartbeat",
        "v1",
    )
    assert gen.render_member_key(entry) == "EVT_NODE_HEARTBEAT_V1"


@pytest.mark.unit
def test_normalization_mixed_dots_and_hyphens(tmp_path: Path) -> None:
    """Mixed dots and hyphens in event_name → all replaced with underscores."""
    gen = TopicEnumGenerator()
    entry = _entry(
        "onex.evt.platform.my-node.event.v1",
        "evt",
        "platform",
        "my-node.event",
        "v1",
    )
    assert gen.render_member_key(entry) == "EVT_MY_NODE_EVENT_V1"


# ---------------------------------------------------------------------------
# Class name and filename derivation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_class_name_simple_producer() -> None:
    """Simple producer → title-cased class name."""
    gen = TopicEnumGenerator()
    assert gen.class_name_for_producer("platform") == "EnumPlatformTopic"


@pytest.mark.unit
def test_class_name_hyphenated_producer() -> None:
    """Hyphenated producer → CamelCase class name."""
    gen = TopicEnumGenerator()
    assert gen.class_name_for_producer("my-service") == "EnumMyServiceTopic"


@pytest.mark.unit
def test_filename_simple_producer() -> None:
    """Simple producer → underscored filename."""
    gen = TopicEnumGenerator()
    assert gen.filename_for_producer("platform") == "enum_platform_topic.py"


@pytest.mark.unit
def test_filename_hyphenated_producer() -> None:
    """Hyphenated producer → underscored filename."""
    gen = TopicEnumGenerator()
    assert gen.filename_for_producer("my-service") == "enum_my_service_topic.py"


# ---------------------------------------------------------------------------
# Basic render — single producer
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_render_single_producer_produces_enum_and_init(tmp_path: Path) -> None:
    """Single producer → one enum file + __init__.py."""
    gen = TopicEnumGenerator()
    entries = [
        _entry(
            "onex.evt.platform.intent-classified.v1",
            "evt",
            "platform",
            "intent-classified",
            "v1",
        )
    ]
    result = gen.render(entries, output_dir=tmp_path)

    # Two files: the enum + __init__.py
    assert len(result) == 2
    paths = {p.name for p in result}
    assert "enum_platform_topic.py" in paths
    assert "__init__.py" in paths


@pytest.mark.unit
def test_render_enum_contains_correct_member(tmp_path: Path) -> None:
    """Enum file contains the correctly normalized member."""
    gen = TopicEnumGenerator()
    entries = [
        _entry(
            "onex.evt.platform.intent-classified.v1",
            "evt",
            "platform",
            "intent-classified",
            "v1",
        )
    ]
    result = gen.render(entries, output_dir=tmp_path)

    enum_content = result[tmp_path / "enum_platform_topic.py"]
    assert "EVT_INTENT_CLASSIFIED_V1" in enum_content
    assert "'onex.evt.platform.intent-classified.v1'" in enum_content


@pytest.mark.unit
def test_render_enum_class_name(tmp_path: Path) -> None:
    """Enum file contains the class definition with correct name."""
    gen = TopicEnumGenerator()
    entries = [
        _entry(
            "onex.evt.platform.intent-classified.v1",
            "evt",
            "platform",
            "intent-classified",
            "v1",
        )
    ]
    result = gen.render(entries, output_dir=tmp_path)
    enum_content = result[tmp_path / "enum_platform_topic.py"]
    assert "class EnumPlatformTopic" in enum_content


@pytest.mark.unit
def test_render_enum_inherits_from_str_and_enum(tmp_path: Path) -> None:
    """Enum class inherits from both str and Enum."""
    gen = TopicEnumGenerator()
    entries = [
        _entry(
            "onex.evt.platform.intent-classified.v1",
            "evt",
            "platform",
            "intent-classified",
            "v1",
        )
    ]
    result = gen.render(entries, output_dir=tmp_path)
    enum_content = result[tmp_path / "enum_platform_topic.py"]
    assert "(str, Enum)" in enum_content


# ---------------------------------------------------------------------------
# __init__.py format
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_init_exports_enum_class(tmp_path: Path) -> None:
    """__init__.py exports the enum class via relative import."""
    gen = TopicEnumGenerator()
    entries = [
        _entry(
            "onex.evt.platform.intent-classified.v1",
            "evt",
            "platform",
            "intent-classified",
            "v1",
        )
    ]
    result = gen.render(entries, output_dir=tmp_path)
    init_content = result[tmp_path / "__init__.py"]
    assert "from .enum_platform_topic import EnumPlatformTopic" in init_content


@pytest.mark.unit
def test_init_no_helper_mapping(tmp_path: Path) -> None:
    """__init__.py does NOT export a helper mapping (dict/TOPIC_ENUMS etc.)."""
    gen = TopicEnumGenerator()
    entries = [
        _entry(
            "onex.evt.platform.intent-classified.v1",
            "evt",
            "platform",
            "intent-classified",
            "v1",
        )
    ]
    result = gen.render(entries, output_dir=tmp_path)
    init_content = result[tmp_path / "__init__.py"]
    assert "dict" not in init_content
    assert "mapping" not in init_content.lower()
    assert "TOPIC_ENUMS" not in init_content


# ---------------------------------------------------------------------------
# Multiple producers
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_render_multiple_producers_separate_files(tmp_path: Path) -> None:
    """Multiple producers → separate enum files + one __init__.py."""
    gen = TopicEnumGenerator()
    entries = [
        _entry(
            "onex.evt.platform.intent-classified.v1",
            "evt",
            "platform",
            "intent-classified",
            "v1",
        ),
        _entry(
            "onex.cmd.omniclaude.pipeline-started.v1",
            "cmd",
            "omniclaude",
            "pipeline-started",
            "v1",
        ),
    ]
    result = gen.render(entries, output_dir=tmp_path)

    assert len(result) == 3  # 2 enums + 1 __init__
    paths = {p.name for p in result}
    assert "enum_platform_topic.py" in paths
    assert "enum_omniclaude_topic.py" in paths
    assert "__init__.py" in paths


@pytest.mark.unit
def test_init_exports_all_producers_sorted(tmp_path: Path) -> None:
    """__init__.py exports all producers sorted alphabetically."""
    gen = TopicEnumGenerator()
    entries = [
        _entry(
            "onex.evt.platform.intent-classified.v1",
            "evt",
            "platform",
            "intent-classified",
            "v1",
        ),
        _entry(
            "onex.cmd.omniclaude.pipeline-started.v1",
            "cmd",
            "omniclaude",
            "pipeline-started",
            "v1",
        ),
    ]
    result = gen.render(entries, output_dir=tmp_path)
    init_content = result[tmp_path / "__init__.py"]

    # Both imports present
    assert "from .enum_omniclaude_topic import EnumOmniclaudeTopic" in init_content
    assert "from .enum_platform_topic import EnumPlatformTopic" in init_content

    # Sorted: omniclaude before platform
    omni_pos = init_content.index("from .enum_omniclaude_topic")
    platform_pos = init_content.index("from .enum_platform_topic")
    assert omni_pos < platform_pos


# ---------------------------------------------------------------------------
# Determinism — two runs on same input → byte-identical output
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_render_deterministic_same_input(tmp_path: Path) -> None:
    """Two render() calls on identical input produce byte-identical output."""
    gen = TopicEnumGenerator()
    entries = [
        _entry(
            "onex.evt.platform.intent-classified.v1",
            "evt",
            "platform",
            "intent-classified",
            "v1",
        ),
        _entry(
            "onex.cmd.platform.intent-query-session.v1",
            "cmd",
            "platform",
            "intent-query-session",
            "v1",
        ),
        _entry(
            "onex.evt.platform.intent-stored.v1",
            "evt",
            "platform",
            "intent-stored",
            "v1",
        ),
    ]
    result1 = gen.render(entries, output_dir=tmp_path / "run1")
    result2 = gen.render(entries, output_dir=tmp_path / "run2")

    # Convert to comparable form (relative path → content)
    def normalize(d: dict[Path, str]) -> dict[str, str]:
        return {p.name: c for p, c in d.items()}

    assert normalize(result1) == normalize(result2)


@pytest.mark.unit
def test_render_deterministic_member_order(tmp_path: Path) -> None:
    """Members sorted by (kind, event_name, version) for determinism."""
    gen = TopicEnumGenerator()
    # Provide entries in reverse order — output should still be sorted
    entries = [
        _entry(
            "onex.evt.platform.z-event.v1",
            "evt",
            "platform",
            "z-event",
            "v1",
        ),
        _entry(
            "onex.cmd.platform.a-event.v1",
            "cmd",
            "platform",
            "a-event",
            "v1",
        ),
        _entry(
            "onex.evt.platform.a-event.v1",
            "evt",
            "platform",
            "a-event",
            "v1",
        ),
    ]
    result = gen.render(entries, output_dir=tmp_path)
    enum_content = result[tmp_path / "enum_platform_topic.py"]

    # Extract member lines (lines containing '=')
    member_lines = [
        line.strip()
        for line in enum_content.splitlines()
        if "=" in line and "#" not in line[:4]
    ]
    keys = [line.split("=")[0].strip() for line in member_lines]

    # Sorted by (kind, event_name, version):
    # cmd a-event v1 → CMD_A_EVENT_V1
    # evt a-event v1 → EVT_A_EVENT_V1
    # evt z-event v1 → EVT_Z_EVENT_V1
    assert keys == ["CMD_A_EVENT_V1", "EVT_A_EVENT_V1", "EVT_Z_EVENT_V1"]


# ---------------------------------------------------------------------------
# Collision detection
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_collision_detection_raises_runtime_error(tmp_path: Path) -> None:
    """
    Two entries in the same producer that normalize to the same key →
    RuntimeError with clear message.
    """
    gen = TopicEnumGenerator()
    # Both normalize to EVT_MY_TOPIC_V1 (hyphens vs dots in event_name)
    entries = [
        _entry(
            "onex.evt.platform.my-topic.v1",
            "evt",
            "platform",
            "my-topic",  # → MY_TOPIC
            "v1",
        ),
        _entry(
            "onex.evt.platform.my.topic.v1",
            "evt",
            "platform",
            "my.topic",  # → MY_TOPIC (same after normalization)
            "v1",
        ),
    ]
    with pytest.raises(RuntimeError, match="Enum key collision"):
        gen.render(entries, output_dir=tmp_path)


@pytest.mark.unit
def test_collision_message_contains_both_topics(tmp_path: Path) -> None:
    """Collision error message names both conflicting topics."""
    gen = TopicEnumGenerator()
    entries = [
        _entry(
            "onex.evt.platform.my-topic.v1",
            "evt",
            "platform",
            "my-topic",
            "v1",
        ),
        _entry(
            "onex.evt.platform.my.topic.v1",
            "evt",
            "platform",
            "my.topic",
            "v1",
        ),
    ]
    with pytest.raises(RuntimeError) as exc_info:
        gen.render(entries, output_dir=tmp_path)

    msg = str(exc_info.value)
    assert "EVT_MY_TOPIC_V1" in msg
    assert "my-topic" in msg or "my.topic" in msg


@pytest.mark.unit
def test_no_collision_across_producers(tmp_path: Path) -> None:
    """Same normalized key in different producers is NOT a collision."""
    gen = TopicEnumGenerator()
    entries = [
        _entry(
            "onex.evt.platform.my-topic.v1",
            "evt",
            "platform",
            "my-topic",
            "v1",
        ),
        _entry(
            "onex.evt.omniclaude.my-topic.v1",
            "evt",
            "omniclaude",
            "my-topic",
            "v1",
        ),
    ]
    # Should not raise — different producers get separate files
    result = gen.render(entries, output_dir=tmp_path)
    assert len(result) == 3  # 2 enums + init


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_render_empty_entries_returns_empty(tmp_path: Path) -> None:
    """Empty entry list → empty dict."""
    gen = TopicEnumGenerator()
    result = gen.render([], output_dir=tmp_path)
    assert result == {}


# ---------------------------------------------------------------------------
# Style — no triple-quote docstrings on enum members
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_enum_members_use_line_comments_not_docstrings(tmp_path: Path) -> None:
    """Enum members use # comments, not triple-quote docstrings."""
    gen = TopicEnumGenerator()
    entries = [
        _entry(
            "onex.evt.platform.intent-classified.v1",
            "evt",
            "platform",
            "intent-classified",
            "v1",
        )
    ]
    result = gen.render(entries, output_dir=tmp_path)
    enum_content = result[tmp_path / "enum_platform_topic.py"]

    # No triple-quote docstrings on individual members (class docstring is OK)
    lines = enum_content.splitlines()
    member_line_indices = [
        i
        for i, line_text in enumerate(lines)
        if "EVT_INTENT_CLASSIFIED_V1" in line_text
    ]
    assert len(member_line_indices) == 1
    member_line = lines[member_line_indices[0]]
    # The member line itself uses # not triple quotes
    assert '"""' not in member_line
    assert "'''" not in member_line


# ---------------------------------------------------------------------------
# Generated file has auto-generated header
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_generated_file_has_do_not_edit_header(tmp_path: Path) -> None:
    """Generated enum files contain the DO NOT EDIT header."""
    gen = TopicEnumGenerator()
    entries = [
        _entry(
            "onex.evt.platform.intent-classified.v1",
            "evt",
            "platform",
            "intent-classified",
            "v1",
        )
    ]
    result = gen.render(entries, output_dir=tmp_path)
    enum_content = result[tmp_path / "enum_platform_topic.py"]
    assert "AUTO-GENERATED FILE" in enum_content
    assert "DO NOT EDIT MANUALLY" in enum_content


# ---------------------------------------------------------------------------
# Output path uses the provided output_dir
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_output_paths_use_output_dir(tmp_path: Path) -> None:
    """All returned paths are under the specified output_dir."""
    gen = TopicEnumGenerator()
    custom_dir = tmp_path / "custom" / "output"
    entries = [
        _entry(
            "onex.evt.platform.intent-classified.v1",
            "evt",
            "platform",
            "intent-classified",
            "v1",
        )
    ]
    result = gen.render(entries, output_dir=custom_dir)
    for path in result:
        assert str(path).startswith(str(custom_dir))


# ---------------------------------------------------------------------------
# validate_output_dir
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validate_output_dir_accepts_path(tmp_path: Path) -> None:
    """validate_output_dir returns True for a valid Path."""
    assert TopicEnumGenerator.validate_output_dir(tmp_path) is True


@pytest.mark.unit
def test_validate_output_dir_rejects_string() -> None:
    """validate_output_dir raises TypeError for non-Path input."""
    with pytest.raises(TypeError):
        TopicEnumGenerator.validate_output_dir("not_a_path")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Generator does not call parse/validate on topic strings
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_generator_does_not_validate_topics(tmp_path: Path) -> None:
    """
    Generator accepts entries without re-validating topic strings.
    If caller provides a pre-validated entry (even with an unusual topic),
    the generator renders it as-is without raising.
    """
    gen = TopicEnumGenerator()
    # Entry already validated by extractor — generator renders unconditionally
    entry = ModelContractTopicEntry(
        topic="onex.evt.platform.my-topic.v1",
        kind="evt",
        producer="platform",
        event_name="my-topic",
        version="v1",
        source_contracts=(Path("/a/contract.yaml"),),
    )
    result = gen.render([entry], output_dir=tmp_path)
    assert len(result) == 2  # enum + init
