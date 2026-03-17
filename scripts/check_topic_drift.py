#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Check for drift between static topic constants and contract YAML declarations.

Compares the SUFFIX_* constants defined in platform_topic_suffixes.py against
topic strings declared in contract.yaml files across the node tree. Reports:

- Orphaned constants: SUFFIX_* values not referenced by any contract (exit 1).
- Undeclared topics: contract topics not present in constants (warning, exit 0).

DLQ topics (containing '.dlq.') and broadcast topics (containing '.broadcast')
are excluded from comparison since they are infrastructure-scoped.

Usage::

    # Default: scan nodes directory, compare against platform_topic_suffixes.py
    python scripts/check_topic_drift.py --contracts-dir src/omnibase_infra/nodes

    # Custom constants file
    python scripts/check_topic_drift.py \\
        --contracts-dir src/omnibase_infra/nodes \\
        --constants-file src/omnibase_infra/topics/platform_topic_suffixes.py

.. versionadded:: 0.22.0
    OMN-5248: Initial implementation.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

# Allow running as a standalone script (not via uv run) by adjusting sys.path
# when the package is not installed.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from omnibase_infra.topics.contract_topic_extractor import ContractTopicExtractor


def _extract_suffix_values(constants_file: Path) -> dict[str, str]:
    """Extract SUFFIX_* constant values from a Python module using AST parsing.

    Args:
        constants_file: Path to platform_topic_suffixes.py or similar.

    Returns:
        Mapping of constant name to string value (e.g., {"SUFFIX_NODE_REGISTRATION": "onex.evt.platform.node-registration.v1"}).
    """
    source = constants_file.read_text()
    tree = ast.parse(source, filename=str(constants_file))

    suffix_constants: dict[str, str] = {}
    for node in ast.walk(tree):
        # Handle plain assignment: SUFFIX_X = "..."
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                if not target.id.startswith("SUFFIX_"):
                    continue
                value = node.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    suffix_constants[target.id] = value.value

        # Handle annotated assignment: SUFFIX_X: str = "..."
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if not isinstance(target, ast.Name):
                continue
            if not target.id.startswith("SUFFIX_"):
                continue
            value = node.value
            if (
                value is not None
                and isinstance(value, ast.Constant)
                and isinstance(value.value, str)
            ):
                suffix_constants[target.id] = value.value

    return suffix_constants


def _is_infrastructure_topic(topic: str) -> bool:
    """Check if a topic is infrastructure-scoped and should be excluded from drift checks.

    DLQ topics and broadcast topics are infrastructure concerns, not domain
    event routing, so they are intentionally excluded from the contract
    comparison.

    Args:
        topic: Topic string to check.

    Returns:
        True if the topic should be excluded from drift comparison.
    """
    return ".dlq." in topic or "-dlq." in topic or ".broadcast" in topic


def main() -> int:
    """Run the topic drift checker.

    Returns:
        0 if no orphaned constants found, 1 if orphaned constants detected.
    """
    parser = argparse.ArgumentParser(
        description="Check for drift between topic constants and contract YAML declarations."
    )
    parser.add_argument(
        "--contracts-dir",
        type=Path,
        required=True,
        help="Root directory to scan for contract.yaml files.",
    )
    parser.add_argument(
        "--constants-file",
        type=Path,
        default=_REPO_ROOT
        / "src"
        / "omnibase_infra"
        / "topics"
        / "platform_topic_suffixes.py",
        help="Path to platform_topic_suffixes.py (default: auto-detected).",
    )
    parser.add_argument(
        "--warn-only",
        action="store_true",
        default=False,
        help="Report orphaned constants as warnings instead of errors (exit 0).",
    )
    args = parser.parse_args()

    contracts_dir: Path = args.contracts_dir
    constants_file: Path = args.constants_file

    if not contracts_dir.is_dir():
        print(f"ERROR: contracts directory not found: {contracts_dir}", file=sys.stderr)
        return 1

    if not constants_file.is_file():
        print(f"ERROR: constants file not found: {constants_file}", file=sys.stderr)
        return 1

    # Extract SUFFIX_* values from constants file
    suffix_map = _extract_suffix_values(constants_file)
    suffix_values = set(suffix_map.values())

    # Filter out infrastructure-scoped topics
    suffix_values_filtered = {
        t for t in suffix_values if not _is_infrastructure_topic(t)
    }

    # Scan contract YAML files
    extractor = ContractTopicExtractor()
    manifest = extractor.scan(contracts_dir)
    contract_topics = manifest.all_unique_topics

    # Filter out infrastructure-scoped topics from contracts too
    contract_topics_filtered = {
        t for t in contract_topics if not _is_infrastructure_topic(t)
    }

    # Detect orphaned constants (in constants but not in any contract)
    orphaned = suffix_values_filtered - contract_topics_filtered
    # Detect undeclared topics (in contracts but not in constants)
    undeclared = contract_topics_filtered - suffix_values_filtered

    exit_code = 0

    if orphaned:
        level = "WARNING" if args.warn_only else "ERROR"
        print(
            f"\n{level}: {len(orphaned)} orphaned SUFFIX_* constant(s) not in any contract:"
        )
        # Show constant name -> value for clarity
        reverse_map = {v: k for k, v in suffix_map.items()}
        for topic in sorted(orphaned):
            name = reverse_map.get(topic, "???")
            print(f"  {name} = {topic!r}")
        if not args.warn_only:
            exit_code = 1

    if undeclared:
        print(
            f"\nWARNING: {len(undeclared)} contract topic(s) not in SUFFIX_* constants:"
        )
        for topic in sorted(undeclared):
            # Find which node(s) declare this topic
            declaring_nodes = [
                nt.node_name
                for nt in manifest.nodes.values()
                if topic in nt.subscribe_topics or topic in nt.publish_topics
            ]
            print(f"  {topic!r}  (declared by: {', '.join(declaring_nodes)})")

    if exit_code == 0 and not undeclared:
        print(
            f"OK: {len(suffix_values_filtered)} constants and "
            f"{len(contract_topics_filtered)} contract topics are in sync."
        )
    elif exit_code == 0:
        print(
            f"\nOK: No orphaned constants. {len(undeclared)} undeclared topic(s) are warnings only."
        )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
