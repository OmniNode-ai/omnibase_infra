#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""
check-topic-parity.py (OMN-4963, relocated OMN-9286)

Cross-repo topic parity checker. Verifies that omnidash consumer subscription
lists (READ_MODEL_TOPICS in read-model-consumer.ts and EXPECTED_TOPICS in
event-bus-health-poller.ts) cover the canonical topic_registry.yaml.

This script does NOT check shared/topics.ts (that is sync-topic-registry.py's
job, OMN-4962). Instead, it checks that consumer subscription lists reference
topics present in the registry and that critical registry topics are not
missing from subscription lists.

Usage:
    # CI mode: exit 0 if consistent, exit 1 with human-readable diff
    python omnibase_infra/scripts/check-topic-parity.py --check

    # Verbose mode: show all topic lists and their relationships
    python omnibase_infra/scripts/check-topic-parity.py --check --verbose

Paths are resolved relative to the omni_home root. By default the script
assumes it lives at ``<omni_home>/omnibase_infra/scripts/`` and derives
omni_home as ``Path(__file__).resolve().parents[2]``. Override via the
``OMNI_HOME`` env var or the ``--omni-home`` flag (required in CI where
omnibase_infra is cloned standalone).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict

try:
    import yaml
except ImportError:
    print(
        "ERROR: PyYAML is required. Install with: pip install pyyaml",
        file=sys.stderr,
    )
    sys.exit(2)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _default_omni_home() -> Path:
    env_value = os.environ.get("OMNI_HOME")
    if env_value:
        return Path(env_value).resolve()
    # Script lives at <omni_home>/omnibase_infra/scripts/<this file>
    return Path(__file__).resolve().parents[2]


class ModelTopicParityPaths(BaseModel):
    """Resolved filesystem paths required by the topic parity checker."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    omni_home: Path
    registry: Path
    read_model_consumer: Path
    event_bus_health_poller: Path
    topics_ts: Path

    @classmethod
    def from_omni_home(
        cls,
        omni_home: Path,
        registry_override: Path | None = None,
    ) -> ModelTopicParityPaths:
        registry = (
            registry_override.resolve()
            if registry_override is not None
            else (
                omni_home
                / "omniclaude"
                / "src"
                / "omniclaude"
                / "hooks"
                / "topic_registry.yaml"
            )
        )
        return cls(
            omni_home=omni_home,
            registry=registry,
            read_model_consumer=omni_home
            / "omnidash"
            / "server"
            / "read-model-consumer.ts",
            event_bus_health_poller=omni_home
            / "omnidash"
            / "server"
            / "event-bus-health-poller.ts",
            topics_ts=omni_home / "omnidash" / "shared" / "topics.ts",
        )


# ---------------------------------------------------------------------------
# Registry loader
# ---------------------------------------------------------------------------


def load_registry_topics(path: Path) -> set[str]:
    """Load all topic strings from the canonical topic_registry.yaml."""
    if not path.exists():
        print(f"ERROR: Registry not found: {path}", file=sys.stderr)
        sys.exit(2)

    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        print(
            f"ERROR: Registry root must be a mapping, got {type(data).__name__}: {path}",
            file=sys.stderr,
        )
        sys.exit(2)

    topics = data.get("topics", [])
    if not topics:
        print(f"ERROR: No topics found in {path}", file=sys.stderr)
        sys.exit(2)

    return {entry["topic"] for entry in topics}


# ---------------------------------------------------------------------------
# TypeScript topic extractors
# ---------------------------------------------------------------------------

# Match literal topic strings like 'onex.evt.omniclaude.agent-actions.v1'
TOPIC_LITERAL_RE = re.compile(r"'(onex\.[a-z]+\.[a-z_-]+\.[a-z0-9_-]+\.v\d+)'")

# Match SUFFIX_* or TOPIC_* constant references
TOPIC_CONST_RE = re.compile(r"\b(SUFFIX_[A-Z_]+|TOPIC_[A-Z_]+)\b")


def resolve_topic_constants(topics_ts: Path) -> dict[str, str]:
    """Build a map from constant name -> topic string by scanning topics.ts."""
    if not topics_ts.exists():
        print(f"WARNING: topics.ts not found: {topics_ts}", file=sys.stderr)
        return {}

    content = topics_ts.read_text()
    # Match: export const SUFFIX_FOO = 'onex.evt.bar.baz.v1';
    pattern = re.compile(
        r"export\s+const\s+((?:SUFFIX|TOPIC)_[A-Z_]+)\s*=\s*'(onex\.[^']+)'"
    )
    return {m.group(1): m.group(2) for m in pattern.finditer(content)}


def extract_topics_from_array(
    file_path: Path,
    array_name: str,
    const_map: dict[str, str],
    topics_ts: Path,
    *,
    required: bool = False,
) -> set[str]:
    """Extract topic strings from a named TypeScript array in a source file.

    Handles:
    - Literal topic strings: 'onex.evt.omniclaude.foo.v1'
    - Constant references: SUFFIX_FOO, TOPIC_BAR
    - Spread expressions: ...OMNICLAUDE_AGENT_TOPICS (resolved recursively)

    When ``required=True`` (top-level subscription arrays), a missing array
    is fatal — a silent rename would otherwise mask the exact breakage this
    gate exists to catch. Recursive spread lookups pass ``required=False``
    and fall back to searching ``topics.ts``.
    """
    if not file_path.exists():
        print(f"ERROR: Source file not found: {file_path}", file=sys.stderr)
        sys.exit(2)

    content = file_path.read_text()

    # Find the array block
    # Match: const ARRAY_NAME = [ ... ] or export const ARRAY_NAME = [ ... ]
    pattern = re.compile(
        rf"(?:export\s+)?const\s+{re.escape(array_name)}\s*(?::\s*\w+(?:\[\])?)?\s*=\s*\[(.*?)\]",
        re.DOTALL,
    )
    match = pattern.search(content)
    if not match:
        level = "ERROR" if required else "WARNING"
        print(
            f"{level}: Array '{array_name}' not found in {file_path}",
            file=sys.stderr,
        )
        if required:
            print(
                "  A required subscription array is missing or renamed. Topic "
                "parity cannot be verified; failing closed.",
                file=sys.stderr,
            )
            sys.exit(2)
        return set()

    block = match.group(1)
    topics: set[str] = set()

    # Extract literal topic strings
    for m in TOPIC_LITERAL_RE.finditer(block):
        topics.add(m.group(1))

    # Extract constant references and resolve them
    for m in TOPIC_CONST_RE.finditer(block):
        const_name = m.group(1)
        if const_name in const_map:
            topics.add(const_map[const_name])

    # Handle spread expressions like ...OMNICLAUDE_AGENT_TOPICS
    spread_re = re.compile(
        r"\.\.\.((?:SUFFIX|TOPIC|OMNICLAUDE|PLATFORM|INTELLIGENCE)[A-Z_]*)"
    )
    for m in spread_re.finditer(block):
        spread_name = m.group(1)
        # Recursively resolve the spread array from the same file or topics.ts
        spread_topics = extract_topics_from_array(
            file_path, spread_name, const_map, topics_ts
        )
        if not spread_topics:
            spread_topics = extract_topics_from_array(
                topics_ts, spread_name, const_map, topics_ts
            )
        topics.update(spread_topics)

    return topics


# ---------------------------------------------------------------------------
# Parity checker
# ---------------------------------------------------------------------------


def check_parity(paths: ModelTopicParityPaths, verbose: bool = False) -> int:
    """Check topic parity across registry, READ_MODEL_TOPICS, and EXPECTED_TOPICS.

    Returns 0 if all checks pass, 1 if any mismatch is found.
    """
    registry_topics = load_registry_topics(paths.registry)
    const_map = resolve_topic_constants(paths.topics_ts)

    read_model_topics = extract_topics_from_array(
        paths.read_model_consumer,
        "READ_MODEL_TOPICS",
        const_map,
        paths.topics_ts,
        required=True,
    )
    expected_topics = extract_topics_from_array(
        paths.event_bus_health_poller,
        "EXPECTED_TOPICS",
        const_map,
        paths.topics_ts,
        required=True,
    )

    if verbose:
        print(f"Registry topics ({len(registry_topics)}):")
        for t in sorted(registry_topics):
            print(f"  {t}")
        print()
        print(f"READ_MODEL_TOPICS ({len(read_model_topics)}):")
        for t in sorted(read_model_topics):
            print(f"  {t}")
        print()
        print(f"EXPECTED_TOPICS ({len(expected_topics)}):")
        for t in sorted(expected_topics):
            print(f"  {t}")
        print()

    errors: list[str] = []

    # Check 1: EXPECTED_TOPICS should only reference topics in the registry
    # (Allow non-omniclaude topics that come from other producers)
    expected_not_in_registry = expected_topics - registry_topics
    # Filter to only omniclaude topics for this check (other producers may not
    # be in the omniclaude registry)
    omniclaude_expected_not_in_registry = {
        t for t in expected_not_in_registry if ".omniclaude." in t
    }
    if omniclaude_expected_not_in_registry:
        errors.append(
            "EXPECTED_TOPICS references omniclaude topics NOT in topic_registry.yaml:"
        )
        for t in sorted(omniclaude_expected_not_in_registry):
            errors.append(f"  + {t}")

    # Check 2: READ_MODEL_TOPICS should only reference topics that exist in registry
    # (Allow non-omniclaude topics from other producers)
    read_model_not_in_registry = read_model_topics - registry_topics
    omniclaude_read_model_not_in_registry = {
        t for t in read_model_not_in_registry if ".omniclaude." in t
    }
    if omniclaude_read_model_not_in_registry:
        errors.append(
            "READ_MODEL_TOPICS references omniclaude topics NOT in topic_registry.yaml:"
        )
        for t in sorted(omniclaude_read_model_not_in_registry):
            errors.append(f"  + {t}")

    # Check 3: Reverse coverage — every omniclaude evt topic declared in the
    # registry must be covered by both READ_MODEL_TOPICS and EXPECTED_TOPICS.
    # Without this, a newly-declared producer topic can be registered but never
    # wired into the consumer, and the original gate (consumer -> registry)
    # exits 0. This is the exact failure mode OMN-4963 was meant to catch.
    registry_omniclaude_evt = {
        t for t in registry_topics if ".omniclaude." in t and ".evt." in t
    }
    missing_from_read_model = registry_omniclaude_evt - read_model_topics
    missing_from_expected = registry_omniclaude_evt - expected_topics

    if missing_from_read_model:
        errors.append(
            "Registry omniclaude evt topics NOT subscribed in READ_MODEL_TOPICS:"
        )
        for t in sorted(missing_from_read_model):
            errors.append(f"  + {t}")

    if missing_from_expected:
        errors.append("Registry omniclaude evt topics NOT listed in EXPECTED_TOPICS:")
        for t in sorted(missing_from_expected):
            errors.append(f"  + {t}")

    # Check 4: READ_MODEL_TOPICS subscribed omniclaude evt topics should also be
    # in EXPECTED_TOPICS so the health poller can detect missing topics on the
    # broker. Advisory only — not blocking.
    read_model_omniclaude = {
        t for t in read_model_topics if ".omniclaude." in t and ".evt." in t
    }
    expected_omniclaude = {
        t for t in expected_topics if ".omniclaude." in t and ".evt." in t
    }
    subscribed_but_not_expected = read_model_omniclaude - expected_omniclaude
    if subscribed_but_not_expected and verbose:
        print("ADVISORY: READ_MODEL_TOPICS subscribes to these omniclaude evt topics")
        print(
            "that are NOT in EXPECTED_TOPICS (health poller won't detect if missing):"
        )
        for t in sorted(subscribed_but_not_expected):
            print(f"  ~ {t}")
        print()

    if errors:
        print("TOPIC PARITY FAILURE")
        print("=" * 60)
        for line in errors:
            print(line)
        print()
        print("To fix:")
        print("  1. Add missing topics to topic_registry.yaml, OR")
        print("  2. Remove stale references from the consumer/poller files")
        return 1

    # Summary
    print("OK: Topic parity check passed")
    print(f"  Registry:         {len(registry_topics)} topics")
    print(f"  READ_MODEL_TOPICS: {len(read_model_topics)} topics")
    print(f"  EXPECTED_TOPICS:   {len(expected_topics)} topics")

    # Non-omniclaude topics are expected (cross-service topics)
    non_registry_read_model = read_model_topics - registry_topics
    non_registry_expected = expected_topics - registry_topics
    if non_registry_read_model:
        print(
            f"  Non-registry topics in READ_MODEL_TOPICS: {len(non_registry_read_model)} "
            f"(cross-service, OK)"
        )
    if non_registry_expected:
        print(
            f"  Non-registry topics in EXPECTED_TOPICS: {len(non_registry_expected)} "
            f"(cross-service, OK)"
        )

    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-repo topic parity checker (OMN-4963)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        required=True,
        help="Run parity check (CI mode)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show full topic lists",
    )
    parser.add_argument(
        "--omni-home",
        type=Path,
        default=None,
        help="Override omni_home root (default: OMNI_HOME env var, else parents[2])",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Override path to topic_registry.yaml",
    )

    args = parser.parse_args()

    omni_home = args.omni_home.resolve() if args.omni_home else _default_omni_home()
    paths = ModelTopicParityPaths.from_omni_home(
        omni_home, registry_override=args.registry
    )

    return check_parity(paths, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
