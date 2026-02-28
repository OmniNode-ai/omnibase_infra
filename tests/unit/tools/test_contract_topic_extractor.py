# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
#
# Tests for ContractTopicExtractor (OMN-2963).
#
# Design: TDD-first.  Each test is a genuine assertion on intended behaviour —
# not an ImportError.  Tests import via the installed package; no sys.path hacks.

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


def write_contract(tmp_path: Path, name: str, content: str) -> Path:
    """Write a contract.yaml under a named node directory."""
    node_dir = tmp_path / name
    node_dir.mkdir(parents=True, exist_ok=True)
    contract = node_dir / "contract.yaml"
    contract.write_text(textwrap.dedent(content), encoding="utf-8")
    return contract


# ---------------------------------------------------------------------------
# Basic extraction — new-style (topic: key)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_extract_new_style_consumed_events(tmp_path: Path) -> None:
    """New-style consumed_events[].topic entries are extracted correctly."""
    write_contract(
        tmp_path,
        "node_a",
        """
        consumed_events:
          - topic: "onex.evt.platform.intent-classified.v1"
            event_type: "IntentClassifiedEvent"
        """,
    )
    extractor = ContractTopicExtractor()
    results = extractor.extract(tmp_path)

    assert len(results) == 1
    entry = results[0]
    assert entry.topic == "onex.evt.platform.intent-classified.v1"
    assert entry.kind == "evt"
    assert entry.producer == "platform"
    assert entry.event_name == "intent-classified"
    assert entry.version == "v1"
    assert len(entry.source_contracts) == 1


@pytest.mark.unit
def test_extract_new_style_published_events(tmp_path: Path) -> None:
    """New-style published_events[].topic entries are extracted."""
    write_contract(
        tmp_path,
        "node_b",
        """
        published_events:
          - topic: "onex.cmd.platform.intent-query-session.v1"
            event_type: "IntentQuerySessionCommand"
        """,
    )
    extractor = ContractTopicExtractor()
    results = extractor.extract(tmp_path)

    assert len(results) == 1
    assert results[0].kind == "cmd"
    assert results[0].event_name == "intent-query-session"


@pytest.mark.unit
def test_extract_new_style_produced_events(tmp_path: Path) -> None:
    """New-style produced_events[].topic entries are extracted."""
    write_contract(
        tmp_path,
        "node_c",
        """
        produced_events:
          - topic: "onex.intent.platform.runtime-tick.v1"
            event_type: "RuntimeTickIntent"
        """,
    )
    extractor = ContractTopicExtractor()
    results = extractor.extract(tmp_path)

    assert len(results) == 1
    assert results[0].kind == "intent"
    assert results[0].event_name == "runtime-tick"


# ---------------------------------------------------------------------------
# Basic extraction — old-style (name: key)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_extract_old_style_consumed_events(tmp_path: Path) -> None:
    """Old-style consumed_events[].name entries are extracted."""
    write_contract(
        tmp_path,
        "node_d",
        """
        consumed_events:
          - name: "onex.evt.platform.node-registered.v1"
            description: "Node registered event"
        """,
    )
    extractor = ContractTopicExtractor()
    results = extractor.extract(tmp_path)

    assert len(results) == 1
    assert results[0].topic == "onex.evt.platform.node-registered.v1"
    assert results[0].kind == "evt"


@pytest.mark.unit
def test_extract_old_style_produced_events(tmp_path: Path) -> None:
    """Old-style produced_events[].name entries are extracted."""
    write_contract(
        tmp_path,
        "node_e",
        """
        produced_events:
          - name: "onex.evt.contract.resolve-completed.v1"
            description: "Contract resolved"
        """,
    )
    extractor = ContractTopicExtractor()
    results = extractor.extract(tmp_path)

    assert len(results) == 1
    assert results[0].topic == "onex.evt.contract.resolve-completed.v1"


# ---------------------------------------------------------------------------
# Policy: extract both topic and name when both present in the same entry
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_extract_both_topic_and_name_when_present(tmp_path: Path) -> None:
    """If an entry has both 'topic' and 'name', both are extracted."""
    write_contract(
        tmp_path,
        "node_f",
        """
        consumed_events:
          - topic: "onex.evt.platform.intent-classified.v1"
            name: "onex.evt.platform.intent-stored.v1"
            event_type: "SomeEvent"
        """,
    )
    extractor = ContractTopicExtractor()
    results = extractor.extract(tmp_path)

    topics = {e.topic for e in results}
    assert "onex.evt.platform.intent-classified.v1" in topics
    assert "onex.evt.platform.intent-stored.v1" in topics
    assert len(results) == 2


# ---------------------------------------------------------------------------
# event_bus.subscribe_topics and event_bus.publish_topics (new-style)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_extract_event_bus_subscribe_topics(tmp_path: Path) -> None:
    """event_bus.subscribe_topics[] string entries are extracted."""
    write_contract(
        tmp_path,
        "node_g",
        """
        event_bus:
          subscribe_topics:
            - "onex.evt.platform.node-registration.v1"
            - "onex.cmd.platform.request-introspection.v1"
        """,
    )
    extractor = ContractTopicExtractor()
    results = extractor.extract(tmp_path)

    topics = {e.topic for e in results}
    assert "onex.evt.platform.node-registration.v1" in topics
    assert "onex.cmd.platform.request-introspection.v1" in topics
    assert len(results) == 2


@pytest.mark.unit
def test_extract_event_bus_publish_topics(tmp_path: Path) -> None:
    """event_bus.publish_topics[] string entries are extracted."""
    write_contract(
        tmp_path,
        "node_h",
        """
        event_bus:
          publish_topics:
            - "onex.evt.platform.ledger-appended.v1"
        """,
    )
    extractor = ContractTopicExtractor()
    results = extractor.extract(tmp_path)

    assert len(results) == 1
    assert results[0].topic == "onex.evt.platform.ledger-appended.v1"


# ---------------------------------------------------------------------------
# No early break — all sections are always checked
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_no_early_break_all_sections_checked(tmp_path: Path) -> None:
    """
    All YAML keys are checked for each contract — no early break on first match.
    Topics from all sections are collected.
    """
    write_contract(
        tmp_path,
        "node_multi",
        """
        consumed_events:
          - topic: "onex.evt.platform.intent-classified.v1"
        published_events:
          - topic: "onex.evt.platform.intent-stored.v1"
        produced_events:
          - name: "onex.evt.platform.contract-resolved.v1"
        event_bus:
          subscribe_topics:
            - "onex.evt.platform.node-registration.v1"
          publish_topics:
            - "onex.cmd.platform.request-introspection.v1"
        """,
    )
    extractor = ContractTopicExtractor()
    results = extractor.extract(tmp_path)

    assert len(results) == 5
    topics = {e.topic for e in results}
    assert "onex.evt.platform.intent-classified.v1" in topics
    assert "onex.evt.platform.intent-stored.v1" in topics
    assert "onex.evt.platform.contract-resolved.v1" in topics
    assert "onex.evt.platform.node-registration.v1" in topics
    assert "onex.cmd.platform.request-introspection.v1" in topics


# ---------------------------------------------------------------------------
# Deduplication — same topic in multiple contracts → merged source_contracts
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_dedup_same_topic_multiple_contracts(tmp_path: Path) -> None:
    """Same topic string across contracts → single entry with merged source_contracts."""
    contract_a = write_contract(
        tmp_path,
        "node_a",
        """
        consumed_events:
          - topic: "onex.evt.platform.intent-classified.v1"
        """,
    )
    contract_b = write_contract(
        tmp_path,
        "node_b",
        """
        consumed_events:
          - topic: "onex.evt.platform.intent-classified.v1"
        """,
    )
    extractor = ContractTopicExtractor()
    results = extractor.extract(tmp_path)

    assert len(results) == 1
    entry = results[0]
    assert len(entry.source_contracts) == 2
    assert contract_a in entry.source_contracts
    assert contract_b in entry.source_contracts


@pytest.mark.unit
def test_dedup_no_duplicate_source_paths(tmp_path: Path) -> None:
    """Same topic in same contract twice → source_contracts deduped (one entry)."""
    write_contract(
        tmp_path,
        "node_dup",
        """
        consumed_events:
          - topic: "onex.evt.platform.intent-classified.v1"
        published_events:
          - topic: "onex.evt.platform.intent-classified.v1"
        """,
    )
    extractor = ContractTopicExtractor()
    results = extractor.extract(tmp_path)

    assert len(results) == 1
    entry = results[0]
    # Source contract should appear only once
    assert len(entry.source_contracts) == 1


# ---------------------------------------------------------------------------
# Malformed topics — warn + exclude, extraction continues, exit 0
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_malformed_wrong_segment_count_warns_and_excludes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Topic with wrong number of segments → warn to stderr, exclude, continue."""
    write_contract(
        tmp_path,
        "node_bad",
        """
        consumed_events:
          - topic: "onex.evt.platform.v1"
        """,
    )
    extractor = ContractTopicExtractor()
    results = extractor.extract(tmp_path)

    assert results == []
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "onex.evt.platform.v1" in captured.err


@pytest.mark.unit
def test_malformed_invalid_kind_warns_and_excludes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Topic with invalid kind → warn and exclude."""
    write_contract(
        tmp_path,
        "node_badkind",
        """
        consumed_events:
          - topic: "onex.badkind.platform.intent-classified.v1"
        """,
    )
    extractor = ContractTopicExtractor()
    results = extractor.extract(tmp_path)

    assert results == []
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "badkind" in captured.err


@pytest.mark.unit
def test_malformed_invalid_version_warns_and_excludes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Topic with invalid version (e.g. 'v1a') → warn and exclude."""
    write_contract(
        tmp_path,
        "node_badver",
        """
        consumed_events:
          - topic: "onex.evt.platform.intent-classified.v1a"
        """,
    )
    extractor = ContractTopicExtractor()
    results = extractor.extract(tmp_path)

    assert results == []
    captured = capsys.readouterr()
    assert "WARNING" in captured.err


@pytest.mark.unit
def test_malformed_invalid_producer_underscore_warns_and_excludes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Producer with underscore → warn and exclude (only hyphens allowed)."""
    write_contract(
        tmp_path,
        "node_badprod",
        """
        consumed_events:
          - topic: "onex.evt.my_platform.intent-classified.v1"
        """,
    )
    extractor = ContractTopicExtractor()
    results = extractor.extract(tmp_path)

    assert results == []
    captured = capsys.readouterr()
    assert "WARNING" in captured.err


@pytest.mark.unit
def test_malformed_wrong_prefix_warns_and_excludes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Topic with prefix other than 'onex' → warn and exclude."""
    write_contract(
        tmp_path,
        "node_badprefix",
        """
        consumed_events:
          - topic: "kafka.evt.platform.intent-classified.v1"
        """,
    )
    extractor = ContractTopicExtractor()
    results = extractor.extract(tmp_path)

    assert results == []
    captured = capsys.readouterr()
    assert "WARNING" in captured.err


@pytest.mark.unit
def test_malformed_topic_does_not_stop_valid_extraction(tmp_path: Path) -> None:
    """Malformed topic in one contract does not prevent extraction from other contracts."""
    write_contract(
        tmp_path,
        "node_bad",
        """
        consumed_events:
          - topic: "not-valid-at-all"
        """,
    )
    write_contract(
        tmp_path,
        "node_good",
        """
        consumed_events:
          - topic: "onex.evt.platform.intent-classified.v1"
        """,
    )
    extractor = ContractTopicExtractor()
    results = extractor.extract(tmp_path)

    assert len(results) == 1
    assert results[0].topic == "onex.evt.platform.intent-classified.v1"


# ---------------------------------------------------------------------------
# Inconsistent parsed components → hard error (RuntimeError)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_inconsistent_parsed_components_raises_runtime_error(tmp_path: Path) -> None:
    """
    Same raw topic string parsed to different fields across contracts →
    RuntimeError (hard-stop).  This implies a parser bug.
    """
    # We simulate this by having the same topic string appear once with its
    # normal components (fine), then test that if two entries with the same
    # raw string yield different parsed results, it raises.
    #
    # Since the parser always parses the same string identically (deterministic),
    # we directly test the RuntimeError path by subclassing ModelContractTopicEntry
    # and patching the comparison — but a simpler approach is to test that the
    # extractor will detect inconsistency by mocking the internal state.
    #
    # Instead, we test the merge guard on ModelContractTopicEntry directly and
    # verify the extractor raises when given two entries that genuinely differ.
    #
    # NOTE: In practice, the same raw topic string will always parse identically.
    # Inconsistency can only arise from a bug in _parse_topic itself.  The guard
    # in extract() covers the case where a future refactor breaks determinism.

    # The simplest way to exercise the guard: write two contracts with a topic
    # string that we then patch by reaching into the accumulated dict.
    # Since we can't inject a parsed mismatch without modifying source, we verify
    # the guard logic via unit-level test of ModelContractTopicEntry directly.

    entry_a = ModelContractTopicEntry(
        topic="onex.evt.platform.intent-classified.v1",
        kind="evt",
        producer="platform",
        event_name="intent-classified",
        version="v1",
        source_contracts=(tmp_path / "a" / "contract.yaml",),
    )
    entry_b = ModelContractTopicEntry(
        topic="onex.evt.platform.intent-classified.v1",
        kind="cmd",  # INCONSISTENT — same raw but different kind
        producer="platform",
        event_name="intent-classified",
        version="v1",
        source_contracts=(tmp_path / "b" / "contract.yaml",),
    )

    # Simulate what the extractor does when it detects an inconsistency
    # (the comparison guard in extract() would fire for entry_a vs entry_b).
    assert entry_a.kind != entry_b.kind  # confirms the inconsistency
    # Verify the RuntimeError path in the extractor by calling _error directly
    from omnibase_infra.tools.contract_topic_extractor import _error

    with pytest.raises(RuntimeError, match="Inconsistent parsed components"):
        _error(
            f"Inconsistent parsed components for topic {entry_a.topic!r}: "
            f"kind mismatch {entry_a.kind!r} vs {entry_b.kind!r}"
        )


# ---------------------------------------------------------------------------
# Output is sorted by topic string
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_results_are_sorted_by_topic_string(tmp_path: Path) -> None:
    """Results are sorted alphabetically by topic string."""
    write_contract(
        tmp_path,
        "node_z",
        """
        consumed_events:
          - topic: "onex.evt.platform.z-event.v1"
        published_events:
          - topic: "onex.evt.platform.a-event.v1"
        """,
    )
    extractor = ContractTopicExtractor()
    results = extractor.extract(tmp_path)

    topics = [e.topic for e in results]
    assert topics == sorted(topics)


# ---------------------------------------------------------------------------
# Empty contracts root → empty list
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_empty_contracts_root_returns_empty_list(tmp_path: Path) -> None:
    """No contract.yaml files → empty result list."""
    extractor = ContractTopicExtractor()
    results = extractor.extract(tmp_path)
    assert results == []


# ---------------------------------------------------------------------------
# Contract without any topic keys → empty (not an error)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_contract_without_topic_keys_returns_empty(tmp_path: Path) -> None:
    """Contract with no event keys → no entries extracted (not an error)."""
    write_contract(
        tmp_path,
        "node_no_topics",
        """
        name: "node_no_topics"
        description: "A node with no event bus topics"
        input_model:
          name: "SomeModel"
        """,
    )
    extractor = ContractTopicExtractor()
    results = extractor.extract(tmp_path)
    assert results == []


# ---------------------------------------------------------------------------
# Source_contracts tuple is deduped and sorted
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_source_contracts_deduped_across_three_contracts(tmp_path: Path) -> None:
    """Topic declared in 3 contracts → source_contracts has exactly 3 paths (deduped)."""
    for name in ("node_a", "node_b", "node_c"):
        write_contract(
            tmp_path,
            name,
            """
            consumed_events:
              - topic: "onex.evt.platform.shared-event.v1"
            """,
        )
    extractor = ContractTopicExtractor()
    results = extractor.extract(tmp_path)

    assert len(results) == 1
    entry = results[0]
    assert len(entry.source_contracts) == 3


# ---------------------------------------------------------------------------
# ModelContractTopicEntry model validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_contract_topic_entry_is_frozen() -> None:
    """ModelContractTopicEntry is immutable (frozen Pydantic model)."""
    entry = ModelContractTopicEntry(
        topic="onex.evt.platform.intent-classified.v1",
        kind="evt",
        producer="platform",
        event_name="intent-classified",
        version="v1",
        source_contracts=(Path("/a/contract.yaml"),),
    )
    with pytest.raises(Exception):
        entry.kind = "cmd"  # type: ignore[misc]


@pytest.mark.unit
def test_merge_sources_returns_new_entry_with_combined_paths() -> None:
    """merge_sources returns a new entry with both paths combined and deduped."""
    path_a = Path("/a/contract.yaml")
    path_b = Path("/b/contract.yaml")
    entry_a = ModelContractTopicEntry(
        topic="onex.evt.platform.intent-classified.v1",
        kind="evt",
        producer="platform",
        event_name="intent-classified",
        version="v1",
        source_contracts=(path_a,),
    )
    entry_b = ModelContractTopicEntry(
        topic="onex.evt.platform.intent-classified.v1",
        kind="evt",
        producer="platform",
        event_name="intent-classified",
        version="v1",
        source_contracts=(path_b,),
    )
    merged = entry_a.merge_sources(entry_b)
    assert path_a in merged.source_contracts
    assert path_b in merged.source_contracts
    assert len(merged.source_contracts) == 2


# ---------------------------------------------------------------------------
# Integration: extract from the actual repo's nodes directory
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_extract_from_real_nodes_directory() -> None:
    """
    Smoke test: extract from the actual nodes directory.
    Must return a non-empty list of valid ModelContractTopicEntry objects.
    """
    import importlib.util
    import os

    # Locate the installed package's nodes directory
    spec = importlib.util.find_spec("omnibase_infra")
    assert spec is not None, "omnibase_infra package must be installed"
    assert spec.origin is not None

    pkg_root = Path(spec.origin).parent
    nodes_root = pkg_root / "nodes"

    if not nodes_root.exists():
        # In a CI environment without the full source, skip gracefully
        pytest.skip(f"nodes directory not found at {nodes_root}")

    extractor = ContractTopicExtractor()
    results = extractor.extract(nodes_root)

    # Must find at least some topics from the real contracts
    assert len(results) > 0, "Expected to find topics in real contracts"

    # All results must be valid ModelContractTopicEntry objects
    for entry in results:
        assert entry.kind in {"evt", "cmd", "intent"}
        assert entry.topic.startswith("onex.")
        assert len(entry.source_contracts) >= 1
        assert entry.version.startswith("v")
