# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""TDD tests for ContractTopicExtractor skill manifest support.

Tests for:
  - extract_from_skill_manifests(skills_root: Path) -> list[ModelContractTopicEntry]
  - extract_all(contracts_root, skill_manifests_root=None) overload

Ticket: OMN-4593
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from omnibase_infra.tools.contract_topic_extractor import (
    ContractTopicExtractor,
    ModelContractTopicEntry,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def skills_root(tmp_path: Path) -> Path:
    """Create a minimal fake skills root with topics.yaml files."""
    root = tmp_path / "skills"
    root.mkdir()

    # Skill A — 3 valid topics
    skill_a = root / "epic-team"
    skill_a.mkdir()
    (skill_a / "topics.yaml").write_text(
        textwrap.dedent(
            """\
            topics:
              - onex.cmd.omniclaude.epic-team.v1
              - onex.evt.omniclaude.epic-team-completed.v1
              - onex.evt.omniclaude.epic-team-failed.v1
            """
        ),
        encoding="utf-8",
    )

    # Skill B — 3 valid topics
    skill_b = root / "ticket-pipeline"
    skill_b.mkdir()
    (skill_b / "topics.yaml").write_text(
        textwrap.dedent(
            """\
            topics:
              - onex.cmd.omniclaude.ticket-pipeline.v1
              - onex.evt.omniclaude.ticket-pipeline-completed.v1
              - onex.evt.omniclaude.ticket-pipeline-failed.v1
            """
        ),
        encoding="utf-8",
    )

    # Skip dir — should be ignored
    skip_lib = root / "_lib"
    skip_lib.mkdir()
    (skip_lib / "topics.yaml").write_text(
        "topics:\n  - onex.cmd.omniclaude.should-not-appear.v1\n",
        encoding="utf-8",
    )

    return root


@pytest.fixture
def skills_root_with_malformed(tmp_path: Path) -> Path:
    """Skills root containing one malformed topic — should warn+skip, not crash."""
    root = tmp_path / "skills_malformed"
    root.mkdir()

    skill = root / "good-skill"
    skill.mkdir()
    (skill / "topics.yaml").write_text(
        textwrap.dedent(
            """\
            topics:
              - onex.cmd.omniclaude.good-skill.v1
              - onex.INVALID.omniclaude.good-skill-completed.v1
              - onex.evt.omniclaude.good-skill-failed.v1
            """
        ),
        encoding="utf-8",
    )
    return root


# ---------------------------------------------------------------------------
# Test 1: extract_from_skill_manifests returns correct entries from valid files
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_extract_from_skill_manifests_returns_valid_entries(skills_root: Path) -> None:
    """extract_from_skill_manifests discovers topics.yaml in all non-skip skill dirs."""
    extractor = ContractTopicExtractor()
    entries = extractor.extract_from_skill_manifests(skills_root)

    topics = {e.topic for e in entries}

    # Should find 6 topics (3 per skill, 2 non-skip skills)
    assert len(entries) == 6, (
        f"Expected 6 entries, got {len(entries)}: {[e.topic for e in entries]}"
    )

    # Both skills present
    assert "onex.cmd.omniclaude.epic-team.v1" in topics
    assert "onex.evt.omniclaude.epic-team-completed.v1" in topics
    assert "onex.evt.omniclaude.epic-team-failed.v1" in topics
    assert "onex.cmd.omniclaude.ticket-pipeline.v1" in topics
    assert "onex.evt.omniclaude.ticket-pipeline-completed.v1" in topics
    assert "onex.evt.omniclaude.ticket-pipeline-failed.v1" in topics

    # _lib topics must NOT appear
    assert "onex.cmd.omniclaude.should-not-appear.v1" not in topics

    # All entries have correct kind
    for entry in entries:
        assert entry.kind in ("cmd", "evt"), f"Unexpected kind {entry.kind!r}"
        assert entry.producer == "omniclaude"

    # Result is sorted by topic string
    sorted_topics = sorted(e.topic for e in entries)
    assert [e.topic for e in entries] == sorted_topics


# ---------------------------------------------------------------------------
# Test 2: malformed topics are warned and skipped, not raised
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_extract_from_skill_manifests_warns_and_skips_malformed(
    skills_root_with_malformed: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Malformed topics in topics.yaml produce a WARNING on stderr and are skipped."""
    extractor = ContractTopicExtractor()
    entries = extractor.extract_from_skill_manifests(skills_root_with_malformed)

    topics = {e.topic for e in entries}

    # Valid topics should be present
    assert "onex.cmd.omniclaude.good-skill.v1" in topics
    assert "onex.evt.omniclaude.good-skill-failed.v1" in topics

    # Malformed topic must be absent
    assert not any("INVALID" in t for t in topics), "Malformed topic should be skipped"

    # A warning should have been emitted
    captured = capsys.readouterr()
    assert "WARNING" in captured.err or "WARNING" in captured.out, (
        "Expected a WARNING for the malformed topic"
    )


# ---------------------------------------------------------------------------
# Test 3: deduplication — same topic in multiple skills.yaml is merged
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_extract_from_skill_manifests_deduplicates(tmp_path: Path) -> None:
    """Same topic string appearing in two skills is deduplicated into one entry."""
    root = tmp_path / "skills_dup"
    root.mkdir()

    for skill_name in ("skill-a", "skill-b"):
        d = root / skill_name
        d.mkdir()
        (d / "topics.yaml").write_text(
            "topics:\n  - onex.cmd.omniclaude.shared-topic.v1\n",
            encoding="utf-8",
        )

    extractor = ContractTopicExtractor()
    entries = extractor.extract_from_skill_manifests(root)

    shared = [e for e in entries if e.topic == "onex.cmd.omniclaude.shared-topic.v1"]
    assert len(shared) == 1, "Duplicate topic should be deduplicated to 1 entry"
    # Source contracts should reference both files
    assert len(shared[0].source_contracts) == 2, (
        "Deduplicated entry should have both source files"
    )


# ---------------------------------------------------------------------------
# Test 4: extract_all with skill_manifests_root combines contract + skill topics
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_extract_all_with_skill_manifests_root_combines_sources(
    tmp_path: Path,
    skills_root: Path,
) -> None:
    """extract_all(contracts_root, skill_manifests_root=...) unions both sources."""
    # Create a minimal contracts_root with one contract.yaml
    contracts_root = tmp_path / "nodes"
    contracts_root.mkdir()
    node_dir = contracts_root / "node_example_effect"
    node_dir.mkdir()
    (node_dir / "contract.yaml").write_text(
        textwrap.dedent(
            """\
            event_bus:
              publish_topics:
                - onex.evt.platform.example-event.v1
            """
        ),
        encoding="utf-8",
    )

    extractor = ContractTopicExtractor()
    entries = extractor.extract_all(
        contracts_root=contracts_root,
        skill_manifests_root=skills_root,
    )

    topics = {e.topic for e in entries}

    # Contract topic present
    assert "onex.evt.platform.example-event.v1" in topics

    # Skill manifest topics present (from skills_root fixture — 2 skills, 6 topics)
    assert "onex.cmd.omniclaude.epic-team.v1" in topics
    assert "onex.cmd.omniclaude.ticket-pipeline.v1" in topics

    # _lib topic absent
    assert "onex.cmd.omniclaude.should-not-appear.v1" not in topics

    # Sorted and deduplicated
    sorted_topics = sorted(e.topic for e in entries)
    assert [e.topic for e in entries] == sorted_topics

    # When skill_manifests_root=None, only contract topics returned
    entries_no_skills = extractor.extract_all(
        contracts_root=contracts_root,
        skill_manifests_root=None,
    )
    topics_no_skills = {e.topic for e in entries_no_skills}
    assert "onex.evt.platform.example-event.v1" in topics_no_skills
    assert "onex.cmd.omniclaude.epic-team.v1" not in topics_no_skills
