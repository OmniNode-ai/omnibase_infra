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
import importlib.metadata
import importlib.resources
import logging
import re
import sys
from pathlib import Path
from typing import Literal, cast

import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / allowlists
# ---------------------------------------------------------------------------

_APPROVED_PACKAGES: tuple[str, ...] = (
    "omnibase_core",
    "omnibase_infra",
    "omnibase_spi",
    "omniintelligence",
    "omnimemory",
    "omniclaude",
    "omnimarket",
)

_VALID_KINDS: frozenset[str] = frozenset({"evt", "cmd", "intent"})
_VALID_DLQ_CATEGORIES: frozenset[str] = frozenset({"intents", "events", "commands"})
_RE_VERSION = re.compile(r"^v\d+$")
_RE_EVENT_NAME = re.compile(r"^[a-z0-9._-]+$")
_RE_PRODUCER = re.compile(r"^[a-z0-9-]+$")  # no underscores allowed

# Regex to identify ONEX topic string literals in Python source code.
# Matches either:
#   - 5-segment standard topics: onex.<kind>.<producer>.<event-name>.<version>
#   - 4-segment DLQ topics: onex.dlq.<category>.<version>
_RE_ONEX_TOPIC_LITERAL = re.compile(
    r"^onex\.(?:"
    r"(?:evt|cmd|intent)\.[a-z0-9-]+\.[a-z0-9._-]+\.v\d+"
    r"|dlq\.[a-z0-9._-]+\.v\d+"
    r")$"
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
    kind: Literal["evt", "cmd", "intent", "dlq"]
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
    if len(parts) == 4:
        prefix, kind, category, version = parts
        if prefix == "onex" and kind == "dlq":
            if category not in _VALID_DLQ_CATEGORIES:
                _warn(
                    "Skipping malformed DLQ topic "
                    f"(invalid category {category!r}, must be one of "
                    f"{sorted(_VALID_DLQ_CATEGORIES)}): {raw!r} in {source}"
                )
                return None
            if not _RE_VERSION.match(version):
                _warn(
                    f"Skipping malformed DLQ topic (invalid version {version!r}, "
                    f"must match ^v\\d+$): {raw!r} in {source}"
                )
                return None
            return ModelContractTopicEntry(
                topic=raw,
                kind="dlq",
                producer="dlq",
                event_name=category,
                version=version,
                source_contracts=(source,),
            )

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
        kind=cast("Literal['evt', 'cmd', 'intent', 'dlq']", kind),
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
# Installed-package discovery helpers
# ---------------------------------------------------------------------------


def _collect_contracts_from_traversable(
    nodes_ref: importlib.resources.abc.Traversable,
) -> list[Path]:
    """Recursively collect contract.yaml files from an importlib Traversable.

    Args:
        nodes_ref: A Traversable pointing to ``<package>.nodes``.

    Returns:
        List of concrete Path objects for each discovered contract.yaml.
    """
    results: list[Path] = []
    try:
        for child in nodes_ref.iterdir():
            if child.is_dir():
                contract = child.joinpath("contract.yaml")
                # Traversable.is_file() is available in Python 3.12+
                try:
                    if contract.is_file():
                        # Convert Traversable to Path for downstream compat
                        # importlib.resources.as_file() is the safe way, but
                        # for editable installs the Traversable IS a Path already
                        concrete = Path(str(contract))
                        if concrete.exists():
                            results.append(concrete)
                except (TypeError, AttributeError):
                    pass
                # Also recurse into subdirectories (nested node structures)
                results.extend(_collect_contracts_from_traversable(child))
    except (OSError, TypeError):
        pass
    return results


def _find_package_root(
    dist: importlib.metadata.Distribution, pkg_name: str
) -> Path | None:
    """Locate the on-disk root directory for a package.

    For editable installs this resolves to the ``src/<pkg_name>`` directory.
    For normal installs it resolves to the ``site-packages/<pkg_name>`` directory.

    Args:
        dist: The importlib.metadata Distribution for the package.
        pkg_name: The package name (used to build the expected subpath).

    Returns:
        Path to the package root directory, or None if not found.
    """
    # Try direct_url.json for editable installs (PEP 610)
    direct_url = dist.read_text("direct_url.json")
    if direct_url is not None:
        import json

        try:
            url_data = json.loads(direct_url)
            url_str: str = url_data.get("url", "")
            if url_str.startswith("file://"):
                base = Path(url_str.removeprefix("file://"))
                # Editable installs: base is the repo root, package is under src/
                candidate = base / "src" / pkg_name
                if candidate.is_dir():
                    return candidate
                # Some packages have flat layout (package at repo root)
                candidate = base / pkg_name
                if candidate.is_dir():
                    return candidate
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    # Fallback: use dist.locate_file() to find the package directory
    # This works for normal pip installs in site-packages
    located = Path(str(dist.locate_file(pkg_name)))
    if located.is_dir():
        return located

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ContractTopicExtractor:
    """
    Scan contract.yaml files and return validated ModelContractTopicEntry objects.

    This is the *sole* validator in the codebase.  Downstream consumers
    (TopicEnumGenerator, Kafka topic creator script) must not re-validate.

    Args:
        include_installed_packages: When True, ``extract_all()`` will also
            discover contracts from approved installed packages via importlib.
    """

    def __init__(self, *, include_installed_packages: bool = False) -> None:
        self._include_installed_packages = include_installed_packages

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
            except Exception as exc:  # noqa: BLE001 — boundary: catch-all for resilience
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

    def extract_from_skill_manifests(
        self, skills_root: Path
    ) -> list[ModelContractTopicEntry]:
        """Extract topics from omniclaude skill topics.yaml manifests.

        Recursively scans *skills_root* for ``topics.yaml`` files in direct
        child directories (one level deep). Skip directories whose names start
        with ``_`` or ``__`` (e.g. ``_lib``, ``_shared``, ``__pycache__``).

        Each ``topics.yaml`` is expected to contain a ``topics:`` list of
        ONEX topic strings. Malformed topics are warned and skipped (no crash).
        Duplicate topics across multiple manifests are deduplicated (sources
        merged).

        Args:
            skills_root: Path to the omniclaude ``plugins/onex/skills/`` directory.

        Returns:
            Sorted (by topic string), deduplicated list of
            :class:`ModelContractTopicEntry` objects.

        Raises:
            RuntimeError: If the same topic string yields inconsistent parsed
                components across files (implies a parser bug).

        Ticket: OMN-4593
        """
        accumulated: dict[str, ModelContractTopicEntry] = {}

        if not skills_root.is_dir():
            _warn(
                f"skills_root {skills_root} is not a directory — "
                "extract_from_skill_manifests returns empty list"
            )
            return []

        # Also check for a root-level topics.yaml (for standalone manifests
        # like cli/topics.yaml or services/topics.yaml where the directory
        # itself IS the producer, not a parent of skill subdirectories).
        root_topics_yaml = skills_root / "topics.yaml"
        if root_topics_yaml.exists():
            self._extract_manifest_file(root_topics_yaml, skills_root.name, accumulated)

        for skill_dir in sorted(skills_root.iterdir()):
            # Only process direct child directories; skip _ / __ prefixes
            if not skill_dir.is_dir():
                continue
            if skill_dir.name.startswith("_") or skill_dir.name.startswith("__"):
                continue

            topics_yaml = skill_dir / "topics.yaml"
            if not topics_yaml.exists():
                continue
            self._extract_manifest_file(topics_yaml, skill_dir.name, accumulated)

        return sorted(accumulated.values(), key=lambda e: e.topic)

    @staticmethod
    def _extract_manifest_file(
        topics_yaml: Path,
        _producer_name: str,
        accumulated: dict[str, ModelContractTopicEntry],
    ) -> None:
        """Parse a single topics.yaml manifest and merge into accumulated dict."""
        try:
            with topics_yaml.open(encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except Exception as exc:  # noqa: BLE001 — boundary: returns degraded response
            _warn(f"Could not parse {topics_yaml}: {exc} — skipping")
            return

        if not isinstance(data, dict):
            _warn(f"topics.yaml is not a mapping: {topics_yaml} — skipping")
            return

        raw_topics = data.get("topics")
        if not isinstance(raw_topics, list):
            _warn(f"topics.yaml missing 'topics' list: {topics_yaml} — skipping")
            return

        for raw in raw_topics:
            if not isinstance(raw, str) or not raw.strip():
                _warn(f"Skipping non-string or empty topic entry in {topics_yaml}")
                continue

            entry = _parse_topic(raw.strip(), topics_yaml)
            if entry is None:
                # Malformed — warned inside _parse_topic
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
                        f"now in {topics_yaml}. This implies a parser bug."
                    )
                accumulated[raw] = existing.merge_sources(entry)
            else:
                accumulated[raw] = entry

    def extract_from_installed_packages(
        self,
        approved_packages: tuple[str, ...] = _APPROVED_PACKAGES,
    ) -> list[ModelContractTopicEntry]:
        """Extract topics from contract.yaml files in approved installed packages.

        Uses ``importlib.metadata`` and ``importlib.resources`` to discover
        contract YAML files bundled inside approved packages.  Works with both
        editable installs (``pip install -e .``) and normal installs.

        The discovery strategy for each package:

        1. Use ``importlib.resources`` to traverse ``<package>.nodes`` for
           ``contract.yaml`` files (works for properly packaged resources).
        2. Fall back to resolving the package install path via
           ``importlib.metadata.Distribution.locate_file()`` and scanning
           ``<install_root>/nodes/**/contract.yaml`` on the filesystem.

        Args:
            approved_packages: Tuple of package names to scan.  Defaults to
                the platform-approved set (omnibase_core, omnibase_infra, etc.).

        Returns:
            Sorted, deduplicated list of ``ModelContractTopicEntry`` objects.

        Raises:
            ValueError: If the same logical package is discovered via multiple
                install paths (duplicate discovery).

        Ticket: OMN-5132
        """
        accumulated: dict[str, ModelContractTopicEntry] = {}
        seen_package_roots: dict[str, Path] = {}

        for pkg_name in approved_packages:
            contract_paths = self._discover_package_contracts(pkg_name)
            if contract_paths is None:
                # Package not installed or has no contracts — non-fatal
                continue

            # Check for duplicate discovery (same package from multiple paths)
            pkg_root = contract_paths[0].parent if contract_paths else None
            if pkg_root is not None:
                # Normalize: walk up to the package-level directory
                # Contract paths look like .../nodes/<node>/contract.yaml
                # We want the top-level package dir
                normalized = self._package_root_from_contract(
                    contract_paths[0], pkg_name
                )
                if normalized is not None:
                    if pkg_name in seen_package_roots:
                        existing_root = seen_package_roots[pkg_name]
                        if existing_root != normalized:
                            raise ValueError(
                                f"Duplicate discovery for package {pkg_name!r}: "
                                f"found at {existing_root} and {normalized}. "
                                f"Only one install of each approved package is "
                                f"allowed."
                            )
                    seen_package_roots[pkg_name] = normalized

            for contract_path in contract_paths:
                try:
                    with contract_path.open(encoding="utf-8") as fh:
                        raw_yaml = yaml.safe_load(fh)
                except Exception as exc:  # noqa: BLE001 — boundary: catch-all for resilience
                    _warn(
                        f"Could not parse {contract_path} from package "
                        f"{pkg_name}: {exc} — skipping"
                    )
                    continue

                if not isinstance(raw_yaml, dict):
                    _warn(
                        f"contract.yaml is not a mapping: {contract_path} "
                        f"(package {pkg_name}) — skipping"
                    )
                    continue

                raw_topics = _extract_raw_topics_from_contract(raw_yaml, contract_path)

                for raw in raw_topics:
                    entry = _parse_topic(raw, contract_path)
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
                                f"Inconsistent parsed components for topic "
                                f"{raw!r} across installed packages: "
                                f"first seen in "
                                f"{existing.source_contracts[0]}, "
                                f"now in {contract_path}."
                            )
                        accumulated[raw] = existing.merge_sources(entry)
                    else:
                        accumulated[raw] = entry

        return sorted(accumulated.values(), key=lambda e: e.topic)

    @staticmethod
    def _package_root_from_contract(contract_path: Path, pkg_name: str) -> Path | None:
        """Resolve the top-level package directory from a contract path.

        Given a contract at ``<root>/<pkg>/nodes/<node>/contract.yaml``,
        return ``<root>/<pkg>``.  Returns None if the path structure does
        not match expectations.
        """
        # Walk up looking for a directory matching the package name
        for parent in contract_path.parents:
            if parent.name == pkg_name:
                return parent
        return None

    @staticmethod
    def _discover_package_contracts(pkg_name: str) -> list[Path] | None:
        """Discover contract.yaml files inside an installed package.

        Tries ``importlib.resources`` traversal first, then falls back to
        filesystem discovery via ``importlib.metadata``.

        Args:
            pkg_name: The Python package name (e.g., ``omnibase_infra``).

        Returns:
            Sorted list of contract.yaml Paths, or None if the package is
            not installed or contains no contracts.
        """
        # Strategy 1: importlib.resources traversal of <package>.nodes
        try:
            nodes_pkg = f"{pkg_name}.nodes"
            nodes_ref = importlib.resources.files(nodes_pkg)
            contracts = _collect_contracts_from_traversable(nodes_ref)
            if contracts:
                logger.debug(
                    "Discovered %d contract(s) in %s via importlib.resources",
                    len(contracts),
                    nodes_pkg,
                )
                return sorted(contracts)
        except (ModuleNotFoundError, TypeError, ValueError):
            # Package or sub-package not found via resources — try fallback
            pass

        # Strategy 2: filesystem discovery via importlib.metadata
        try:
            dist = importlib.metadata.distribution(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            logger.debug(
                "Package %s not installed — skipping installed-package "
                "contract discovery",
                pkg_name,
            )
            return None

        # Locate the package source directory
        # For editable installs, dist.locate_file("") returns the repo src dir
        # For normal installs, it returns the site-packages dir
        pkg_root = _find_package_root(dist, pkg_name)
        if pkg_root is None:
            logger.debug(
                "Could not locate source root for package %s — skipping",
                pkg_name,
            )
            return None

        nodes_dir = pkg_root / "nodes"
        if not nodes_dir.is_dir():
            logger.debug(
                "No nodes/ directory in package %s at %s — skipping",
                pkg_name,
                pkg_root,
            )
            return None

        contracts = sorted(nodes_dir.rglob("contract.yaml"))
        if not contracts:
            logger.debug(
                "No contract.yaml files found in %s/nodes/ — skipping",
                pkg_name,
            )
            return None

        logger.debug(
            "Discovered %d contract(s) in %s via filesystem at %s",
            len(contracts),
            pkg_name,
            nodes_dir,
        )
        return contracts

    def extract_all(
        self,
        contracts_root: Path,
        supplementary_sources: list[Path] | None = None,
        skill_manifests_root: Path | None = None,
        skill_manifests_roots: list[Path] | None = None,
    ) -> list[ModelContractTopicEntry]:
        """Extract topics from all sources: contracts, Python, skills, and installed packages.

        Combines results from contract.yaml scanning, Python source file
        scanning, omniclaude skill manifest scanning, and installed-package
        contract discovery into a single deduplicated list.  Topics appearing
        in multiple sources are merged (source_contracts combined).

        This is the single owner of topic-entry merge/dedup logic. All callers
        (TopicProvisioner, CLI scripts, validation) should use this method
        rather than combining individual extract methods manually.

        Args:
            contracts_root: Directory to scan for contract.yaml files.
            supplementary_sources: Optional list of Python files to scan
                for additional topic constants.
            skill_manifests_root: Optional single path to a skill manifests
                directory. Kept for backwards compatibility with existing
                callers. When set, equivalent to passing it as the first
                element of ``skill_manifests_roots``.
            skill_manifests_roots: Optional list of paths to scan for
                ``topics.yaml`` flat-list manifests. Supports multiple roots
                (e.g., omniclaude skills, infra CLI relays, infra services).
                When both ``skill_manifests_root`` and ``skill_manifests_roots``
                are set, the single root is prepended to the list.

        Returns:
            Sorted, deduplicated list of ModelContractTopicEntry objects.

        Raises:
            RuntimeError: On inconsistent parsed components.
            ValueError: If duplicate package discovery is detected (when
                ``include_installed_packages`` was set at construction).

        Ticket: OMN-4593, OMN-4622, OMN-5132
        """
        # Start with contract-derived topics
        contract_entries = self.extract(contracts_root)
        accumulated: dict[str, ModelContractTopicEntry] = {
            e.topic: e for e in contract_entries
        }

        # Merge supplementary Python sources (legacy path, OMN-3254)
        if supplementary_sources:
            supplementary_entries = self.extract_from_python_sources(
                supplementary_sources
            )
            for entry in supplementary_entries:
                if entry.topic in accumulated:
                    existing = accumulated[entry.topic]
                    accumulated[entry.topic] = existing.merge_sources(entry)
                else:
                    accumulated[entry.topic] = entry

        # Merge skill manifest topics (OMN-4593, OMN-4622)
        # Combine singular root (backwards compat) with plural roots list
        all_roots: list[Path] = []
        if skill_manifests_root is not None:
            all_roots.append(skill_manifests_root)
        if skill_manifests_roots:
            all_roots.extend(skill_manifests_roots)

        for root in all_roots:
            skill_entries = self.extract_from_skill_manifests(root)
            for entry in skill_entries:
                if entry.topic in accumulated:
                    existing = accumulated[entry.topic]
                    accumulated[entry.topic] = existing.merge_sources(entry)
                else:
                    accumulated[entry.topic] = entry

        # Merge installed-package contract topics (OMN-5132)
        if self._include_installed_packages:
            pkg_entries = self.extract_from_installed_packages()
            for entry in pkg_entries:
                if entry.topic in accumulated:
                    existing = accumulated[entry.topic]
                    accumulated[entry.topic] = existing.merge_sources(entry)
                else:
                    accumulated[entry.topic] = entry

        return sorted(accumulated.values(), key=lambda e: e.topic)
