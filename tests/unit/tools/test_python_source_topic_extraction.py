# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
#
# Tests for Python source topic extraction (OMN-3254).
#
# Verifies that ContractTopicExtractor can extract ONEX topic strings
# from Python source files (e.g., topic_constants.py) and merge them
# with contract-derived topics via extract_all().

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from omnibase_infra.tools.contract_topic_extractor import (
    ContractTopicExtractor,
    ModelContractTopicEntry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_python_source(tmp_path: Path, name: str, content: str) -> Path:
    """Write a Python source file under tmp_path."""
    source = tmp_path / name
    source.write_text(textwrap.dedent(content), encoding="utf-8")
    return source


def write_contract(tmp_path: Path, name: str, content: str) -> Path:
    """Write a contract.yaml under a named node directory."""
    node_dir = tmp_path / name
    node_dir.mkdir(parents=True, exist_ok=True)
    contract = node_dir / "contract.yaml"
    contract.write_text(textwrap.dedent(content), encoding="utf-8")
    return contract


# ---------------------------------------------------------------------------
# extract_from_python_sources — basic extraction
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_extract_topics_from_python_source(tmp_path: Path) -> None:
    """Python source with ONEX topic constants yields correct entries."""
    source = write_python_source(
        tmp_path,
        "topics.py",
        """\
        from typing import Final

        TOPIC_A: Final[str] = "onex.evt.omniclaude.session-outcome.v1"
        TOPIC_B: Final[str] = "onex.cmd.omniintelligence.session-outcome.v1"
        """,
    )

    extractor = ContractTopicExtractor()
    entries = extractor.extract_from_python_sources([source])

    assert len(entries) == 2
    topics = {e.topic for e in entries}
    assert "onex.evt.omniclaude.session-outcome.v1" in topics
    assert "onex.cmd.omniintelligence.session-outcome.v1" in topics


@pytest.mark.unit
def test_extract_skips_non_topic_strings(tmp_path: Path) -> None:
    """Non-ONEX strings in Python source are ignored."""
    source = write_python_source(
        tmp_path,
        "mixed.py",
        """\
        from typing import Final

        TOPIC_A: Final[str] = "onex.evt.platform.my-event.v1"
        NOT_A_TOPIC: Final[str] = "some-random-string"
        DLQ_DOMAIN: Final[str] = "dlq"
        REGEX_PATTERN = r"^onex\\.evt\\..*$"
        """,
    )

    extractor = ContractTopicExtractor()
    entries = extractor.extract_from_python_sources([source])

    assert len(entries) == 1
    assert entries[0].topic == "onex.evt.platform.my-event.v1"


@pytest.mark.unit
def test_extract_deduplicates_within_file(tmp_path: Path) -> None:
    """Same topic appearing twice in one file yields a single entry."""
    source = write_python_source(
        tmp_path,
        "dupes.py",
        """\
        from typing import Final

        TOPIC_A: Final[str] = "onex.evt.platform.my-event.v1"
        TOPIC_B: Final[str] = "onex.evt.platform.my-event.v1"
        """,
    )

    extractor = ContractTopicExtractor()
    entries = extractor.extract_from_python_sources([source])

    assert len(entries) == 1


@pytest.mark.unit
def test_extract_multiline_assignment(tmp_path: Path) -> None:
    """Topic in a multi-line parenthesized assignment is extracted."""
    source = write_python_source(
        tmp_path,
        "multiline.py",
        """\
        from typing import Final

        TOPIC_SESSION: Final[str] = (
            "onex.cmd.omniintelligence.session-outcome.v1"
        )
        """,
    )

    extractor = ContractTopicExtractor()
    entries = extractor.extract_from_python_sources([source])

    assert len(entries) == 1
    assert entries[0].topic == "onex.cmd.omniintelligence.session-outcome.v1"
    assert entries[0].kind == "cmd"
    assert entries[0].producer == "omniintelligence"


@pytest.mark.unit
def test_extract_ignores_docstrings_and_comments(tmp_path: Path) -> None:
    """Topics in docstrings or function bodies are not extracted."""
    source = write_python_source(
        tmp_path,
        "docstrings.py",
        '''\
        """Module docstring with onex.evt.platform.fake-topic.v1."""

        from typing import Final

        REAL_TOPIC: Final[str] = "onex.evt.platform.real-topic.v1"

        def some_function():
            """Docstring with onex.evt.platform.another-fake.v1."""
            local_var = "onex.evt.platform.local-only.v1"
        ''',
    )

    extractor = ContractTopicExtractor()
    entries = extractor.extract_from_python_sources([source])

    # Only the module-level constant should be extracted, not docstrings or local vars
    topics = {e.topic for e in entries}
    assert "onex.evt.platform.real-topic.v1" in topics
    # Function body topics should NOT be extracted (they are not module-level assignments)
    assert "onex.evt.platform.local-only.v1" not in topics


@pytest.mark.unit
def test_extract_handles_invalid_python_gracefully(tmp_path: Path) -> None:
    """Invalid Python source is skipped with a warning, not a crash."""
    source = write_python_source(
        tmp_path,
        "invalid.py",
        "def broken(\n",
    )

    extractor = ContractTopicExtractor()
    entries = extractor.extract_from_python_sources([source])

    assert entries == []


@pytest.mark.unit
def test_extract_correct_parsed_fields(tmp_path: Path) -> None:
    """Extracted entries have correct kind, producer, event_name, version."""
    source = write_python_source(
        tmp_path,
        "fields.py",
        """\
        from typing import Final

        TOPIC: Final[str] = "onex.evt.omnibase-infra.effectiveness-data-changed.v1"
        """,
    )

    extractor = ContractTopicExtractor()
    entries = extractor.extract_from_python_sources([source])

    assert len(entries) == 1
    entry = entries[0]
    assert entry.kind == "evt"
    assert entry.producer == "omnibase-infra"
    assert entry.event_name == "effectiveness-data-changed"
    assert entry.version == "v1"
    assert source in entry.source_contracts


# ---------------------------------------------------------------------------
# extract_all — merging contracts + supplementary sources
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_extract_all_merges_contract_and_python_sources(tmp_path: Path) -> None:
    """extract_all combines contract.yaml and Python source topics."""
    contracts_dir = tmp_path / "contracts"
    contracts_dir.mkdir()
    write_contract(
        contracts_dir,
        "node_a",
        """\
        published_events:
          - topic: "onex.evt.platform.intent-classified.v1"
        """,
    )

    source = write_python_source(
        tmp_path,
        "topic_constants.py",
        """\
        from typing import Final

        TOPIC_STATUS: Final[str] = "onex.evt.omniclaude.agent-status.v1"
        """,
    )

    extractor = ContractTopicExtractor()
    entries = extractor.extract_all(
        contracts_dir,
        supplementary_sources=[source],
    )

    topics = {e.topic for e in entries}
    assert "onex.evt.platform.intent-classified.v1" in topics
    assert "onex.evt.omniclaude.agent-status.v1" in topics
    assert len(entries) == 2


@pytest.mark.unit
def test_extract_all_deduplicates_shared_topics(tmp_path: Path) -> None:
    """Topic in both contract and Python source yields merged source_contracts."""
    contracts_dir = tmp_path / "contracts"
    contracts_dir.mkdir()
    contract_path = write_contract(
        contracts_dir,
        "node_a",
        """\
        published_events:
          - topic: "onex.evt.omnimemory.reward-assigned.v1"
        """,
    )

    source = write_python_source(
        tmp_path,
        "topic_constants.py",
        """\
        from typing import Final

        TOPIC_REWARD: Final[str] = "onex.evt.omnimemory.reward-assigned.v1"
        """,
    )

    extractor = ContractTopicExtractor()
    entries = extractor.extract_all(
        contracts_dir,
        supplementary_sources=[source],
    )

    assert len(entries) == 1
    entry = entries[0]
    assert entry.topic == "onex.evt.omnimemory.reward-assigned.v1"
    # Both sources should be in source_contracts
    assert len(entry.source_contracts) == 2


@pytest.mark.unit
def test_extract_all_without_supplementary_equals_extract(tmp_path: Path) -> None:
    """extract_all with no supplementary sources returns same as extract."""
    contracts_dir = tmp_path / "contracts"
    contracts_dir.mkdir()
    write_contract(
        contracts_dir,
        "node_a",
        """\
        published_events:
          - topic: "onex.evt.platform.intent-classified.v1"
        """,
    )

    extractor = ContractTopicExtractor()
    entries_extract = extractor.extract(contracts_dir)
    entries_extract_all = extractor.extract_all(contracts_dir)

    assert len(entries_extract) == len(entries_extract_all)
    assert [e.topic for e in entries_extract] == [e.topic for e in entries_extract_all]


# ---------------------------------------------------------------------------
# Integration: actual topic_constants.py
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_actual_topic_constants_extractable() -> None:
    """The real topic_constants.py yields at least the known topic constants."""
    source = Path(
        "/Volumes/PRO-G40/Code/omni_worktrees/OMN-3254/omnibase_infra"
        "/src/omnibase_infra/event_bus/topic_constants.py"
    )
    if not source.exists():
        pytest.skip("topic_constants.py not found at expected path")

    extractor = ContractTopicExtractor()
    entries = extractor.extract_from_python_sources([source])

    topics = {e.topic for e in entries}

    # These are the known hardcoded topic constants
    expected_topics = {
        "onex.cmd.omniintelligence.session-outcome.v1",
        "onex.evt.omniclaude.session-outcome.v1",
        "onex.evt.omniclaude.context-utilization.v1",
        "onex.evt.omniclaude.agent-match.v1",
        "onex.evt.omniclaude.latency-breakdown.v1",
        "onex.evt.omniintelligence.llm-call-completed.v1",
        "onex.evt.omnibase-infra.effectiveness-data-changed.v1",
        "onex.evt.omniclaude.agent-status.v1",
        "onex.evt.omnimemory.reward-assigned.v1",
        "onex.evt.platform.resolution-decided.v1",
    }

    missing = expected_topics - topics
    assert not missing, f"Missing topics from topic_constants.py: {missing}"
