# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
#
# ContractTopicExtractor — single source of topic parsing and validation.
#
# Scans all contract.yaml files under a contracts root directory and returns
# a validated list of ModelContractTopicEntry objects.  All downstream components
# (generator, scripts) consume its output without re-validating.
#
# Also supports extracting topics from Python source files containing
# hardcoded topic constants (e.g., topic_constants.py) to close the
# CONTRACT_DRIFT gap (OMN-3254).
#
# Ticket: OMN-2963, OMN-3254

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Literal, cast

import yaml
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Constants / allowlists
# ---------------------------------------------------------------------------

_VALID_KINDS: frozenset[str] = frozenset({"evt", "cmd", "intent"})
_RE_VERSION = re.compile(r"^v\d+$")
_RE_EVENT_NAME = re.compile(r"^[a-z0-9._-]+$")
_RE_PRODUCER = re.compile(r"^[a-z0-9-]+$")  # no underscores allowed

# Regex to identify ONEX topic string literals in Python source code.
# Matches: onex.<kind>.<producer>.<event-name>.<version>
_RE_ONEX_TOPIC_LITERAL = re.compile(
    r"^onex\.(evt|cmd|intent)\.[a-z0-9-]+\.[a-z0-9._-]+\.v\d+$"
)

# YAML keys to inspect — ordered by spec; always check ALL, no early break.
# Each entry is (top-level-key, sub-key-for-topic, sub-key-for-name).
_EVENT_SECTION_KEYS: tuple[str, ...] = (
    "consumed_events",
    "published_events",
    "produced_events",
)
_EVENT_BUS_SECTION_KEYS: tuple[str, ...] = (
    "subscribe_topics",
    "publish_topics",
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class ModelContractTopicEntry(BaseModel):
    """A single validated topic extracted from one or more contract.yaml files."""

    topic: str
    kind: Literal["evt", "cmd", "intent"]
    producer: str
    event_name: str
    version: str  # e.g. "v1"
    source_contracts: tuple[Path, ...]

    model_config = {"frozen": True}

    def merge_sources(self, other: ModelContractTopicEntry) -> ModelContractTopicEntry:
        """Return a new entry with source_contracts merged (deduped, sorted)."""
        combined = tuple(
            sorted(set(self.source_contracts) | set(other.source_contracts))
        )
        return self.model_copy(update={"source_contracts": combined})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _warn(msg: str) -> None:
    """Emit a warning to stderr (never causes exit != 0 for callers)."""
    print(f"WARNING: {msg}", file=sys.stderr)


def _error(msg: str) -> None:
    """Emit an error message and raise RuntimeError (hard-stop)."""
    print(f"ERROR: {msg}", file=sys.stderr)
    raise RuntimeError(msg)


def _parse_topic(raw: str, source: Path) -> ModelContractTopicEntry | None:
    """
    Parse a raw topic string into a ModelContractTopicEntry.

    Returns None (with a warning) if the topic is malformed.
    Raises RuntimeError if a parser bug (inconsistency) is detected — but
    that check happens at a higher level (deduplication); here we only parse.
    """
    parts = raw.split(".")
    if len(parts) != 5:
        _warn(
            f"Skipping malformed topic (expected 5 segments, got {len(parts)}): "
            f"{raw!r} in {source}"
        )
        return None

    prefix, kind, producer, event_name, version = parts

    if prefix != "onex":
        _warn(
            f"Skipping malformed topic (first segment must be 'onex', got {prefix!r}): "
            f"{raw!r} in {source}"
        )
        return None

    if kind not in _VALID_KINDS:
        _warn(
            f"Skipping malformed topic (invalid kind {kind!r}, "
            f"must be one of {sorted(_VALID_KINDS)}): {raw!r} in {source}"
        )
        return None

    if not _RE_VERSION.match(version):
        _warn(
            f"Skipping malformed topic (invalid version {version!r}, "
            f"must match ^v\\d+$): {raw!r} in {source}"
        )
        return None

    if not _RE_EVENT_NAME.match(event_name):
        _warn(
            f"Skipping malformed topic (invalid event-name {event_name!r}, "
            f"must match ^[a-z0-9._-]+$): {raw!r} in {source}"
        )
        return None

    if not _RE_PRODUCER.match(producer):
        _warn(
            f"Skipping malformed topic (invalid producer {producer!r}, "
            f"must match ^[a-z0-9-]+$ — no underscores): {raw!r} in {source}"
        )
        return None

    return ModelContractTopicEntry(
        topic=raw,
        kind=cast("Literal['evt', 'cmd', 'intent']", kind),
        producer=producer,
        event_name=event_name,
        version=version,
        source_contracts=(source,),
    )


def _extract_raw_topics_from_contract(
    data: dict[str, object], source: Path
) -> list[str]:
    """
    Extract all raw topic strings from a parsed contract YAML dict.

    Checks ALL applicable keys — no early break on first match.
    If a field has both 'topic' and 'name' values, extracts both.
    """
    raw_topics: list[str] = []

    # --- event_bus.subscribe_topics / event_bus.publish_topics (new-style) ---
    event_bus = data.get("event_bus")
    if isinstance(event_bus, dict):
        for key in _EVENT_BUS_SECTION_KEYS:
            topics_list = event_bus.get(key)
            if isinstance(topics_list, list):
                for item in topics_list:
                    if isinstance(item, str) and item:
                        raw_topics.append(item)
                    elif isinstance(item, dict):
                        # topic: "..." format inside subscribe_topics list
                        topic_val = item.get("topic")
                        if isinstance(topic_val, str) and topic_val:
                            raw_topics.append(topic_val)

    # --- consumed_events / published_events / produced_events ---
    for section_key in _EVENT_SECTION_KEYS:
        section = data.get(section_key)
        if not isinstance(section, list):
            continue
        for item in section:
            if not isinstance(item, dict):
                continue
            # new-style: topic key
            topic_val = item.get("topic")
            if isinstance(topic_val, str) and topic_val:
                raw_topics.append(topic_val)
            # old-style: name key — extract both if both present
            name_val = item.get("name")
            if isinstance(name_val, str) and name_val:
                raw_topics.append(name_val)

    return raw_topics


def _extract_topics_from_python_ast(source_path: Path) -> list[str]:
    """Extract ONEX topic string literals from a Python source file using AST.

    Parses the file's AST and collects all string constants that match the
    ONEX topic naming convention (onex.<kind>.<producer>.<event-name>.<version>).

    Only extracts from module-level assignments (Final[str] constants),
    not from docstrings, comments, or function bodies.

    Args:
        source_path: Path to a Python source file.

    Returns:
        Deduplicated list of raw topic strings found in the file.
    """
    try:
        source_text = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source_text, filename=str(source_path))
    except (SyntaxError, OSError) as exc:
        _warn(f"Could not parse Python source {source_path}: {exc} — skipping")
        return []

    topics: list[str] = []
    seen: set[str] = set()

    for node in ast.iter_child_nodes(tree):
        # Only look at module-level assignments (not inside functions/classes)
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue

        # Walk all string constants in the assignment value
        if isinstance(node, ast.AnnAssign):
            if node.value is None:
                continue
            value_node = node.value
        else:
            # ast.Assign.value is always present (not Optional)
            value_node = node.value

        for child in ast.walk(value_node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                val = child.value.strip()
                if _RE_ONEX_TOPIC_LITERAL.match(val) and val not in seen:
                    topics.append(val)
                    seen.add(val)

    return topics


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ContractTopicExtractor:
    """
    Scan contract.yaml files and return validated ModelContractTopicEntry objects.

    This is the *sole* validator in the codebase.  Downstream consumers
    (TopicEnumGenerator, Kafka topic creator script) must not re-validate.
    """

    def extract(self, contracts_root: Path) -> list[ModelContractTopicEntry]:
        """
        Recursively scan *contracts_root* for contract.yaml files, parse every
        topic string, validate each, and return a deduplicated list.

        Malformed topics are warned and excluded; extraction continues.
        Inconsistent parsed components for the same raw topic string are a
        hard error (RuntimeError).

        Args:
            contracts_root: Directory to scan recursively for contract.yaml files.

        Returns:
            Sorted (by topic string) list of ModelContractTopicEntry objects.

        Raises:
            RuntimeError: If the same raw topic string yields inconsistent
                parsed components across contracts (implies a parser bug).
        """
        # accumulated: topic_string -> ModelContractTopicEntry
        accumulated: dict[str, ModelContractTopicEntry] = {}

        contract_files = sorted(contracts_root.rglob("contract.yaml"))

        for contract_path in contract_files:
            try:
                with contract_path.open(encoding="utf-8") as fh:
                    raw_yaml = yaml.safe_load(fh)
            except Exception as exc:
                _warn(f"Could not parse {contract_path}: {exc} — skipping")
                continue

            if not isinstance(raw_yaml, dict):
                _warn(f"contract.yaml is not a mapping: {contract_path} — skipping")
                continue

            raw_topics = _extract_raw_topics_from_contract(raw_yaml, contract_path)

            for raw in raw_topics:
                entry = _parse_topic(raw, contract_path)
                if entry is None:
                    # Warned inside _parse_topic; skip
                    continue

                if raw in accumulated:
                    existing = accumulated[raw]
                    # Consistency check — same raw string must yield identical fields.
                    if (
                        existing.kind != entry.kind
                        or existing.producer != entry.producer
                        or existing.event_name != entry.event_name
                        or existing.version != entry.version
                    ):
                        _error(
                            f"Inconsistent parsed components for topic {raw!r}: "
                            f"first seen as kind={existing.kind!r} producer={existing.producer!r} "
                            f"event_name={existing.event_name!r} version={existing.version!r} "
                            f"(from {existing.source_contracts[0]}), "
                            f"but now parsed as kind={entry.kind!r} producer={entry.producer!r} "
                            f"event_name={entry.event_name!r} version={entry.version!r} "
                            f"(from {contract_path}). This implies a parser bug."
                        )
                    # Merge source_contracts (dedup)
                    accumulated[raw] = existing.merge_sources(entry)
                else:
                    accumulated[raw] = entry

        return sorted(accumulated.values(), key=lambda e: e.topic)

    def extract_from_python_sources(
        self, source_paths: list[Path]
    ) -> list[ModelContractTopicEntry]:
        """Extract and validate ONEX topics from Python source files.

        Parses each file's AST, collects string literals matching the ONEX
        topic naming convention, validates them, and returns deduplicated
        entries. This is used to capture topics defined as hardcoded constants
        (e.g., in topic_constants.py) that are not declared in contract.yaml.

        Args:
            source_paths: List of Python source file paths to scan.

        Returns:
            Sorted (by topic string) list of ModelContractTopicEntry objects.

        Raises:
            RuntimeError: If the same topic string yields inconsistent
                parsed components across files (implies a parser bug).
        """
        accumulated: dict[str, ModelContractTopicEntry] = {}

        for source_path in sorted(source_paths):
            raw_topics = _extract_topics_from_python_ast(source_path)

            for raw in raw_topics:
                entry = _parse_topic(raw, source_path)
                if entry is None:
                    continue

                if raw in accumulated:
                    existing = accumulated[raw]
                    if (
                        existing.kind != entry.kind
                        or existing.producer != entry.producer
                        or existing.event_name != entry.event_name
                        or existing.version != entry.version
                    ):
                        _error(
                            f"Inconsistent parsed components for topic {raw!r}: "
                            f"first seen in {existing.source_contracts[0]}, "
                            f"now in {source_path}. This implies a parser bug."
                        )
                    accumulated[raw] = existing.merge_sources(entry)
                else:
                    accumulated[raw] = entry

        return sorted(accumulated.values(), key=lambda e: e.topic)

    def extract_all(
        self,
        contracts_root: Path,
        supplementary_sources: list[Path] | None = None,
    ) -> list[ModelContractTopicEntry]:
        """Extract topics from contracts AND supplementary Python sources.

        Combines results from contract.yaml scanning and Python source file
        scanning into a single deduplicated list. Topics appearing in both
        sources are merged (source_contracts combined).

        This is the recommended entry point for the generation pipeline
        (OMN-3254) to ensure all topics -- whether declared in contracts
        or hardcoded in Python constants -- appear in the generated enums.

        Args:
            contracts_root: Directory to scan for contract.yaml files.
            supplementary_sources: Optional list of Python files to scan
                for additional topic constants.

        Returns:
            Sorted, deduplicated list of ModelContractTopicEntry objects.

        Raises:
            RuntimeError: On inconsistent parsed components.
        """
        # Start with contract-derived topics
        contract_entries = self.extract(contracts_root)
        accumulated: dict[str, ModelContractTopicEntry] = {
            e.topic: e for e in contract_entries
        }

        # Merge supplementary sources
        if supplementary_sources:
            supplementary_entries = self.extract_from_python_sources(
                supplementary_sources
            )
            for entry in supplementary_entries:
                if entry.topic in accumulated:
                    existing = accumulated[entry.topic]
                    # Same topic from both contract and Python source — merge sources
                    accumulated[entry.topic] = existing.merge_sources(entry)
                else:
                    accumulated[entry.topic] = entry

        return sorted(accumulated.values(), key=lambda e: e.topic)
