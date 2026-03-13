#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
#
# Topic Completeness Check (OMN-3257).
#
# Scans Python source files for ONEX topic string literals and validates that
# every topic found in source is registered in the topic registry (union of
# contract YAML topics, generated enum values, and manual constants files).
#
# This script catches the case where a developer adds a topic string literal
# to Python code without registering it in the canonical topic registry.
#
# Scanning rules:
#   - File types: .py only
#   - Exclude paths: tests/, docs/, fixtures/, files with '# GENERATED' header
#   - Match: string literals matching onex.(evt|cmd|dlq|intent).* (with version)
#   - Allowlist: completeness-allowlist.yaml for known false positives
#
# Usage:
#   uv run python scripts/validation/check_topic_completeness.py
#   uv run python scripts/validation/check_topic_completeness.py \
#       --src src/ --contracts-root src/omnibase_infra/nodes
#   uv run python scripts/validation/check_topic_completeness.py \
#       --constants-file src/omnibase_infra/event_bus/topic_constants.py
#
# Exit codes:
#   0  all topics are registered (or no unregistered topics found)
#   1  one or more unregistered topics found in source
#   2  runtime error (bad arguments, unreadable files)
#
# NOTE: stdlib only — no third-party dependencies.

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Match ONEX topic literals: onex.(evt|cmd|dlq|intent).<rest>.v<n>
_TOPIC_PATTERN = re.compile(r"^onex\.(evt|cmd|dlq|intent)\.[a-z][a-z0-9._-]*\.v\d+$")

# Directories to exclude from scanning
_EXCLUDED_DIR_SEGMENTS: frozenset[str] = frozenset(
    {"tests", "docs", "fixtures", ".venv", "__pycache__", ".claude"}
)

# YAML keys that contain topic strings in contract files
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
# Registry building: collect all known/registered topics
# ---------------------------------------------------------------------------


def _parse_yaml_minimal(text: str) -> dict[str, object] | None:
    """Minimal YAML parser for contract files (stdlib only).

    Handles the subset of YAML used in ONEX contracts: simple key-value pairs,
    lists, and nested dicts. Returns None if parsing fails.

    This avoids depending on PyYAML for the completeness check script.
    For full YAML parsing, use the yaml module (available in lint_topic_names.py).
    """
    # We use a line-by-line approach for the limited contract YAML format.
    # For robustness, try to import yaml first; fall back to regex extraction.
    try:
        import yaml

        data = yaml.safe_load(text)
        if isinstance(data, dict):
            return data
        return None
    except ImportError:
        pass

    # Fallback: extract topic strings directly via regex from YAML text
    return None


def _extract_topics_from_yaml_text(text: str) -> set[str]:
    """Extract ONEX topic strings from raw YAML text using regex.

    This is the fallback when PyYAML is not available. It finds all strings
    matching the ONEX topic pattern in the YAML content.
    """
    topics: set[str] = set()
    # Match quoted or unquoted topic strings in YAML
    for match in re.finditer(
        r"""(?:["'])(onex\.(?:evt|cmd|dlq|intent)\.[a-z][a-z0-9._-]*\.v\d+)(?:["'])""",
        text,
    ):
        topics.add(match.group(1))
    # Also match unquoted values (common in YAML)
    for match in re.finditer(
        r"(?:^|\s|-\s+)(onex\.(?:evt|cmd|dlq|intent)\.[a-z][a-z0-9._-]*\.v\d+)",
        text,
        re.MULTILINE,
    ):
        topics.add(match.group(1))
    return topics


def collect_registry_from_contracts(contracts_root: Path) -> set[str]:
    """Collect all registered topics from contract.yaml files.

    Scans recursively for contract.yaml files and extracts topic strings
    from event_bus and event section keys.
    """
    topics: set[str] = set()
    if not contracts_root.exists():
        return topics

    for contract_path in sorted(contracts_root.rglob("contract.yaml")):
        try:
            text = contract_path.read_text(encoding="utf-8")
        except OSError:
            continue

        parsed = _parse_yaml_minimal(text)
        if parsed is not None:
            # Extract from parsed YAML dict
            _collect_from_parsed_yaml(parsed, topics)
        else:
            # Fallback: regex extraction from raw text
            topics.update(_extract_topics_from_yaml_text(text))

    return topics


def collect_registry_from_manifests(manifest_roots: list[Path]) -> set[str]:
    """Collect topics from flat-list topics.yaml manifests.

    Scans each root for a root-level topics.yaml and/or child-directory
    topics.yaml files (same format as omniclaude skill manifests).

    Ticket: OMN-4622
    """
    topics: set[str] = set()
    for root in manifest_roots:
        if not root.exists() or not root.is_dir():
            continue
        # Root-level topics.yaml
        root_yaml = root / "topics.yaml"
        if root_yaml.exists():
            topics.update(_extract_topics_from_manifest(root_yaml))
        # Child-directory topics.yaml (omniclaude skill format)
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name.startswith("_"):
                continue
            child_yaml = child / "topics.yaml"
            if child_yaml.exists():
                topics.update(_extract_topics_from_manifest(child_yaml))
    return topics


def _extract_topics_from_manifest(yaml_path: Path) -> set[str]:
    """Extract topics from a flat-list topics.yaml manifest."""
    topics: set[str] = set()
    try:
        text = yaml_path.read_text(encoding="utf-8")
    except OSError:
        return topics

    parsed = _parse_yaml_minimal(text)
    if parsed is not None and isinstance(parsed.get("topics"), list):
        for item in parsed["topics"]:
            if isinstance(item, str) and _TOPIC_PATTERN.match(item):
                topics.add(item)
    else:
        # Fallback: regex
        topics.update(_extract_topics_from_yaml_text(text))
    return topics


def _collect_from_parsed_yaml(data: dict[str, object], topics: set[str]) -> None:
    """Extract topic strings from a parsed contract YAML dict."""
    # event_bus section
    event_bus = data.get("event_bus")
    if isinstance(event_bus, dict):
        for key in _EVENT_BUS_SECTION_KEYS:
            topics_list = event_bus.get(key)
            if isinstance(topics_list, list):
                for item in topics_list:
                    if isinstance(item, str) and _TOPIC_PATTERN.match(item):
                        topics.add(item)
                    elif isinstance(item, dict):
                        topic_val = item.get("topic")
                        if isinstance(topic_val, str) and _TOPIC_PATTERN.match(
                            topic_val
                        ):
                            topics.add(topic_val)

    # consumed_events / published_events / produced_events
    for section_key in _EVENT_SECTION_KEYS:
        section = data.get(section_key)
        if not isinstance(section, list):
            continue
        for item in section:
            if not isinstance(item, dict):
                continue
            for field_key in ("topic", "name"):
                val = item.get(field_key)
                if isinstance(val, str) and _TOPIC_PATTERN.match(val):
                    topics.add(val)


def collect_registry_from_constants_file(constants_path: Path) -> set[str]:
    """Extract registered topics from a Python constants file using AST.

    Parses the file and collects all string literals matching the ONEX topic
    pattern. This covers TOPIC_* constants and enum values.
    """
    topics: set[str] = set()
    if not constants_path.exists():
        return topics

    try:
        source = constants_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(constants_path))
    except (OSError, SyntaxError):
        return topics

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _TOPIC_PATTERN.match(node.value):
                topics.add(node.value)

    return topics


def collect_registry_from_python_enums(src_root: Path) -> set[str]:
    """Collect topics from Python enum/TopicBase files.

    Scans for files that define topic enums (topics.py, platform_topic_suffixes.py,
    topic_constants.py) and extracts their string values.
    """
    topics: set[str] = set()
    canonical_files = [
        "topics.py",
        "topic_constants.py",
        "platform_topic_suffixes.py",
    ]

    if not src_root.exists():
        return topics

    for py_path in sorted(src_root.rglob("*.py")):
        if py_path.name not in canonical_files:
            continue
        # Skip excluded directories
        rel_parts = set(py_path.relative_to(src_root).parts)
        if _EXCLUDED_DIR_SEGMENTS & rel_parts:
            continue
        topics.update(collect_registry_from_constants_file(py_path))

    return topics


# ---------------------------------------------------------------------------
# Source scanning: find topic literals in source code
# ---------------------------------------------------------------------------


def _is_generated_file(path: Path) -> bool:
    """Check if a file has a '# GENERATED' header (first 5 lines)."""
    try:
        with path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= 5:
                    break
                if "# GENERATED" in line:
                    return True
    except OSError:
        pass
    return False


def _should_skip_file(path: Path, src_root: Path) -> bool:
    """Check if a file should be excluded from scanning."""
    try:
        rel_parts = set(path.relative_to(src_root).parts)
    except ValueError:
        rel_parts = set(path.parts)
    return bool(_EXCLUDED_DIR_SEGMENTS & rel_parts)


class _TopicLiteralVisitor(ast.NodeVisitor):
    """AST visitor that collects ONEX topic string literals."""

    def __init__(self, source_lines: list[str]) -> None:
        self.source_lines = source_lines
        self.topics: list[tuple[str, int]] = []  # (topic_value, lineno)

    def _is_in_docstring_or_comment(self, lineno: int) -> bool:
        """Check if a line is a comment (rough heuristic for inline strings)."""
        if lineno < 1 or lineno > len(self.source_lines):
            return False
        line = self.source_lines[lineno - 1].strip()
        return line.startswith("#")

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and _TOPIC_PATTERN.match(node.value):
            if not self._is_in_docstring_or_comment(node.lineno):
                self.topics.append((node.value, node.lineno))
        self.generic_visit(node)


def scan_source_for_topics(
    src_root: Path,
) -> list[tuple[Path, str, int]]:
    """Scan Python source for ONEX topic string literals.

    Returns list of (file_path, topic_string, line_number) tuples.
    """
    results: list[tuple[Path, str, int]] = []
    if not src_root.exists():
        return results

    for py_path in sorted(src_root.rglob("*.py")):
        if _should_skip_file(py_path, src_root):
            continue
        if _is_generated_file(py_path):
            continue

        try:
            source = py_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_path))
        except (OSError, SyntaxError):
            continue

        source_lines = source.splitlines()
        visitor = _TopicLiteralVisitor(source_lines)
        visitor.visit(tree)

        for topic, lineno in visitor.topics:
            results.append((py_path, topic, lineno))

    return results


# ---------------------------------------------------------------------------
# Allowlist loading
# ---------------------------------------------------------------------------


def load_allowlist(allowlist_path: Path) -> set[str]:
    """Load topic allowlist from a YAML file.

    The allowlist is a simple list of topic strings that are known false
    positives (e.g., examples in validators, test fixtures in src/).

    Format (YAML):
        allowlist:
          - onex.evt.example.test-topic.v1
          - onex.cmd.example.another-topic.v1

    Fallback format (plain text, one topic per line):
        onex.evt.example.test-topic.v1
        onex.cmd.example.another-topic.v1
    """
    if not allowlist_path.exists():
        return set()

    try:
        text = allowlist_path.read_text(encoding="utf-8")
    except OSError:
        return set()

    # Try YAML parsing first
    try:
        import yaml

        data = yaml.safe_load(text)
        if isinstance(data, dict) and "allowlist" in data:
            items = data["allowlist"]
            if isinstance(items, list):
                return {str(item) for item in items if item}
    except ImportError:
        pass

    # Fallback: plain text format
    topics: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Skip YAML keys like "allowlist:"
        if line.endswith(":"):
            continue
        # Strip leading "- " for YAML list items
        if line.startswith("- "):
            line = line[2:].strip()
        # Strip quotes
        line = line.strip("'\"")
        if _TOPIC_PATTERN.match(line):
            topics.add(line)

    return topics


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def check_completeness(
    src_root: Path,
    contracts_root: Path | None = None,
    constants_files: list[Path] | None = None,
    allowlist_path: Path | None = None,
    manifest_roots: list[Path] | None = None,
) -> tuple[list[tuple[Path, str, int]], set[str]]:
    """Check that all topic literals in source are in the registry.

    Args:
        src_root: Root directory of Python source to scan.
        contracts_root: Root directory containing contract.yaml files.
        constants_files: Additional Python files defining topic constants.
        allowlist_path: Path to allowlist file for false positives.
        manifest_roots: Additional directories with topics.yaml manifests
            (omniclaude skills, CLI relays, services).

    Returns:
        Tuple of (unregistered_hits, registry) where unregistered_hits is a
        list of (path, topic, lineno) for topics not in the registry.
    """
    # Build the topic registry
    registry: set[str] = set()

    # (a) Contract YAML topics
    if contracts_root is not None:
        registry.update(collect_registry_from_contracts(contracts_root))

    # (b) Generated enum values and canonical topic files
    registry.update(collect_registry_from_python_enums(src_root))

    # (c) Manual constants files
    if constants_files:
        for cf in constants_files:
            registry.update(collect_registry_from_constants_file(cf))

    # (d) Flat-list topics.yaml manifests (skills, CLI, services)
    if manifest_roots:
        registry.update(collect_registry_from_manifests(manifest_roots))

    # Load allowlist
    allowlist: set[str] = set()
    if allowlist_path is not None:
        allowlist = load_allowlist(allowlist_path)

    # Scan source for topic literals
    all_hits = scan_source_for_topics(src_root)

    # Filter: keep only topics NOT in registry and NOT in allowlist
    unregistered: list[tuple[Path, str, int]] = []
    for path, topic, lineno in all_hits:
        if topic not in registry and topic not in allowlist:
            unregistered.append((path, topic, lineno))

    return unregistered, registry


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Topic completeness check — validates that all ONEX topic string "
            "literals in Python source are registered in the topic registry."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--src",
        metavar="DIR",
        type=Path,
        default=None,
        help="Python source root to scan (default: auto-discover src/)",
    )
    parser.add_argument(
        "--contracts-root",
        metavar="DIR",
        type=Path,
        default=None,
        help="Root directory containing contract.yaml files",
    )
    parser.add_argument(
        "--constants-file",
        metavar="FILE",
        type=Path,
        action="append",
        default=None,
        help="Additional Python constants file (can be specified multiple times)",
    )
    parser.add_argument(
        "--allowlist",
        metavar="FILE",
        type=Path,
        default=None,
        help=(
            "Path to topic allowlist file "
            "(default: scripts/validation/completeness-allowlist.yaml)"
        ),
    )
    parser.add_argument(
        "--skills-root",
        metavar="DIR",
        type=Path,
        default=None,
        help="Path to omniclaude skills root (topics.yaml manifests)",
    )
    parser.add_argument(
        "--manifest-root",
        metavar="DIR",
        type=Path,
        action="append",
        default=None,
        help="Additional manifest root with topics.yaml (can be specified multiple times)",
    )
    parser.add_argument(
        "--show-registry",
        action="store_true",
        help="Print the discovered topic registry and exit",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-violation output; only print summary",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 (clean), 1 (violations), or 2 (runtime error)."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Auto-discover project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir
    for _ in range(10):
        if (project_root / "pyproject.toml").exists() or (
            project_root / "src"
        ).exists():
            break
        project_root = project_root.parent

    # Resolve src root
    src_root: Path
    if args.src is not None:
        src_root = args.src.resolve()
    else:
        src_root = project_root / "src"

    if not src_root.exists():
        print(f"ERROR: source root does not exist: {src_root}", file=sys.stderr)
        return 2

    # Resolve contracts root
    contracts_root: Path | None = None
    if args.contracts_root is not None:
        contracts_root = args.contracts_root.resolve()
    else:
        # Default: src/omnibase_infra/nodes (where contract.yaml files live)
        candidate = src_root / "omnibase_infra" / "nodes"
        if candidate.exists():
            contracts_root = candidate

    # Resolve allowlist
    allowlist_path: Path | None = None
    if args.allowlist is not None:
        allowlist_path = args.allowlist.resolve()
    else:
        candidate = script_dir / "completeness-allowlist.yaml"
        if candidate.exists():
            allowlist_path = candidate

    # Resolve manifest roots (skills + any extra --manifest-root args)
    manifest_roots: list[Path] = []
    if args.skills_root is not None:
        manifest_roots.append(args.skills_root.resolve())
    if args.manifest_root:
        manifest_roots.extend(mr.resolve() for mr in args.manifest_root)
    # Auto-discover infra standalone manifests (cli/, services/)
    infra_pkg = src_root / "omnibase_infra"
    for subdir in ("cli", "services"):
        candidate = infra_pkg / subdir
        if candidate.is_dir() and (candidate / "topics.yaml").exists():
            manifest_roots.append(candidate)

    # Show registry mode
    if args.show_registry:
        registry: set[str] = set()
        if contracts_root is not None:
            registry.update(collect_registry_from_contracts(contracts_root))
        registry.update(collect_registry_from_python_enums(src_root))
        if args.constants_file:
            for cf in args.constants_file:
                registry.update(collect_registry_from_constants_file(cf.resolve()))
        if manifest_roots:
            registry.update(collect_registry_from_manifests(manifest_roots))
        print(f"Topic registry ({len(registry)} topics):")
        for topic in sorted(registry):
            print(f"  {topic}")
        return 0

    # Run completeness check
    unregistered, registry = check_completeness(
        src_root=src_root,
        contracts_root=contracts_root,
        constants_files=(
            [cf.resolve() for cf in args.constants_file]
            if args.constants_file
            else None
        ),
        allowlist_path=allowlist_path,
        manifest_roots=manifest_roots if manifest_roots else None,
    )

    if not unregistered:
        if not args.quiet:
            print(
                f"OK: All topic literals in source are registered "
                f"({len(registry)} topics in registry)"
            )
        return 0

    # Report violations
    if not args.quiet:
        print(
            f"FAIL: Found {len(unregistered)} unregistered topic(s) in source:\n",
            file=sys.stderr,
        )
        for path, topic, lineno in sorted(unregistered, key=lambda x: (x[0], x[2])):
            try:
                rel_path = path.relative_to(project_root)
            except ValueError:
                rel_path = path
            print(
                f"  {rel_path}:{lineno}: {topic!r} not in registry",
                file=sys.stderr,
            )
        print(
            "\nTo fix: register the topic in the appropriate contract.yaml or "
            "topic constants file.",
            file=sys.stderr,
        )
        print(
            "If this is a false positive, add it to the allowlist at:"
            "\n  scripts/validation/completeness-allowlist.yaml",
            file=sys.stderr,
        )

    return 1


if __name__ == "__main__":
    sys.exit(main())
