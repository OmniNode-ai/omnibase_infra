#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""
Generate a FEATURE_REGISTRY.md by scanning Kafka-integrated ONEX nodes across all repos.

Scans all repos under --root for ONEX nodes with `event_bus:` sections in contract YAMLs.
Outputs a markdown table with columns:
  Repo | Node | Subscribe Topics | Publish Topics | Design Spec | Fixture Present | Manual Review | Contract Path

Also appends an Orphaned Fixtures section for test fixture directories with no matching contract.

Usage:
  python3 scripts/generate_feature_inventory.py
  python3 scripts/generate_feature_inventory.py --root /Volumes/PRO-G40/Code/omni_home \\
      --out /Volumes/PRO-G40/Code/omni_home/docs/integration/FEATURE_REGISTRY.md
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class NodeEntry:
    """A single Kafka-integrated ONEX node discovered from a contract YAML."""

    repo: str
    node_name: str
    subscribe_topics: list[str]
    publish_topics: list[str]
    design_spec: str  # link or "—"
    fixture_present: bool
    manual_review: str  # "—" unless flagged
    contract_path: str  # relative path from repo root


@dataclass
class OrphanedFixture:
    """A test fixture directory that has no matching contract YAML."""

    repo: str
    fixture_path: str  # relative path from repo root


# ---------------------------------------------------------------------------
# YAML loading helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file, returning an empty dict on any parse error."""
    if yaml is not None:
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001 — boundary: returns degraded response
            return {}

    # Fallback: minimal line-by-line parser for subscribe/publish_topics only.
    # This handles the common omnibase_infra / omniintelligence contract format.
    return _parse_yaml_fallback(path)


def _parse_yaml_fallback(path: Path) -> dict[str, Any]:
    """
    Minimal YAML parser that extracts event_bus subscribe_topics / publish_topics.
    Used only when the 'yaml' package is not available.
    """
    lines: list[str] = []
    try:
        lines = path.read_text(errors="replace").splitlines()
    except Exception:  # noqa: BLE001 — boundary: returns degraded response
        return {}

    result: dict[str, Any] = {}
    in_event_bus = False
    in_subscribe_topics = False
    in_publish_topics = False
    in_subscribe_block = False  # for omniclaude-style subscribe: {topic: ...}
    in_publish_block = False  # for omniclaude-style publish: {success_topic: ...}

    subscribe_topics: list[str] = []
    publish_topics: list[str] = []

    for raw_line in lines:
        stripped = raw_line.strip()
        indent = len(raw_line) - len(raw_line.lstrip())

        # Detect top-level sections
        if not raw_line.startswith(" ") and not raw_line.startswith("\t"):
            if stripped.startswith("name:"):
                result["name"] = stripped.split(":", 1)[1].strip().strip('"')
            in_event_bus = stripped == "event_bus:"

        if not in_event_bus:
            continue

        # --- omnibase_infra / omniintelligence format ---
        # event_bus:
        #   subscribe_topics:
        #     - "topic.name"
        #   publish_topics:
        #     - "topic.name"
        if stripped == "subscribe_topics:":
            in_subscribe_topics = True
            in_publish_topics = False
            in_subscribe_block = False
            in_publish_block = False
        elif stripped == "publish_topics:":
            in_publish_topics = True
            in_subscribe_topics = False
            in_subscribe_block = False
            in_publish_block = False
        # --- omniclaude format ---
        # event_bus:
        #   subscribe:
        #     topic: "topic.name"
        #   publish:
        #     success_topic: "..."
        #     failure_topic: "..."
        elif stripped == "subscribe:" and indent >= 2:
            in_subscribe_block = True
            in_publish_block = False
            in_subscribe_topics = False
        elif stripped == "publish:" and indent >= 2:
            in_publish_block = True
            in_subscribe_block = False
            in_publish_topics = False
        elif stripped.startswith("- ") and in_subscribe_topics:
            val = stripped[2:].strip().strip('"').strip("'")
            if val:
                subscribe_topics.append(val)
        elif stripped.startswith("- ") and in_publish_topics:
            val = stripped[2:].strip().strip('"').strip("'")
            if val:
                publish_topics.append(val)
        elif in_subscribe_block and stripped.startswith("topic:"):
            val = stripped.split(":", 1)[1].strip().strip('"').strip("'")
            if val:
                subscribe_topics.append(val)
        elif (in_publish_block and stripped.startswith("success_topic:")) or (
            in_publish_block and stripped.startswith("failure_topic:")
        ):
            val = stripped.split(":", 1)[1].strip().strip('"').strip("'")
            if val:
                publish_topics.append(val)

    if subscribe_topics or publish_topics:
        result["event_bus"] = {
            "_subscribe_topics": subscribe_topics,
            "_publish_topics": publish_topics,
        }

    return result


# ---------------------------------------------------------------------------
# Topic extraction
# ---------------------------------------------------------------------------


def _extract_topics(event_bus: dict[str, Any]) -> tuple[list[str], list[str]]:
    """
    Extract (subscribe_topics, publish_topics) from an event_bus dict.

    Handles three contract formats:
    - omnibase_infra / omniintelligence: subscribe_topics / publish_topics as string lists
    - omniintelligence rich format: lists of {name, topic, description, ...} dicts
    - omniclaude: subscribe.topic + publish.success_topic / publish.failure_topic
    - Fallback parser output: _subscribe_topics / _publish_topics
    """
    subscribe_topics: list[str] = []
    publish_topics: list[str] = []

    if not event_bus:
        return subscribe_topics, publish_topics

    # Fallback parser output
    if "_subscribe_topics" in event_bus or "_publish_topics" in event_bus:
        subscribe_topics = list(event_bus.get("_subscribe_topics", []))
        publish_topics = list(event_bus.get("_publish_topics", []))
        return subscribe_topics, publish_topics

    # omnibase_infra / omniintelligence format
    sub_topics = event_bus.get("subscribe_topics")
    if isinstance(sub_topics, list):
        for t in sub_topics:
            if isinstance(t, dict):
                # Rich format: {name: ..., topic: "...", description: ...}
                topic_val = t.get("topic")
                if topic_val:
                    subscribe_topics.append(str(topic_val))
            elif t:
                subscribe_topics.append(str(t))

    pub_topics = event_bus.get("publish_topics")
    if isinstance(pub_topics, list):
        for t in pub_topics:
            if isinstance(t, dict):
                # Rich format: {name: ..., topic: "...", description: ...}
                topic_val = t.get("topic")
                if topic_val:
                    publish_topics.append(str(topic_val))
            elif t:
                publish_topics.append(str(t))

    # omniclaude format
    subscribe_block = event_bus.get("subscribe")
    if isinstance(subscribe_block, dict):
        topic = subscribe_block.get("topic")
        if topic:
            subscribe_topics.append(str(topic))

    publish_block = event_bus.get("publish")
    if isinstance(publish_block, dict):
        for key in ("success_topic", "failure_topic"):
            val = publish_block.get(key)
            if val:
                publish_topics.append(str(val))

    return subscribe_topics, publish_topics


# ---------------------------------------------------------------------------
# Design spec lookup
# ---------------------------------------------------------------------------

# Map node name fragments to design spec filenames (checked against docs/design/)
_DESIGN_SPEC_KEYWORDS: list[tuple[str, str]] = [
    ("baselines", "DESIGN_BASELINES"),
    ("intent_classifier", "DESIGN_INTENT_CLASSIFIER"),
    ("pattern", "DESIGN_PATTERN"),
    ("routing", "DESIGN_ROUTING"),
    ("embedding", "DESIGN_EMBEDDING"),
    ("ledger_projection", "DESIGN_LEDGER"),
    ("validation_orchestrator", "DESIGN_VALIDATION"),
    ("reward_binder", "DESIGN_REWARD"),
    ("registration", "DESIGN_REGISTRATION"),
    ("contract_resolver", "DESIGN_CONTRACT"),
    ("scoring", "DESIGN_SCORING"),
    ("doc_promotion", "DESIGN_DOC_PROMOTION"),
    ("doc_retrieval", "DESIGN_DOC_RETRIEVAL"),
    ("compliance", "DESIGN_COMPLIANCE"),
    ("policy_state", "DESIGN_POLICY"),
    ("skill", "DESIGN_SKILL"),
]


def find_design_spec(node_name: str, docs_design_dir: Path) -> str:
    """Return a relative link to a matching design spec, or '—'."""
    if not docs_design_dir.exists():
        return "—"

    node_lower = node_name.lower()

    # Try keyword-based lookup first
    for fragment, prefix in _DESIGN_SPEC_KEYWORDS:
        if fragment in node_lower:
            for spec_file in docs_design_dir.iterdir():
                if spec_file.name.upper().startswith(prefix):
                    return f"[{spec_file.name}](../design/{spec_file.name})"

    # Fall back to substring search in spec filenames
    for spec_file in sorted(docs_design_dir.iterdir()):
        parts = (
            node_name.replace("node_", "")
            .replace("_compute", "")
            .replace("_effect", "")
            .replace("_orchestrator", "")
            .replace("_reducer", "")
        )
        if any(p in spec_file.name.lower() for p in parts.split("_") if len(p) > 4):
            return f"[{spec_file.name}](../design/{spec_file.name})"

    return "—"


# ---------------------------------------------------------------------------
# Fixture detection
# ---------------------------------------------------------------------------


def _fixture_paths_for_repo(repo_path: Path) -> set[str]:
    """
    Return a set of node names that have test fixture directories.
    Looks in tests/unit/nodes/<node_name>/ and tests/integration/<node_name>/.
    """
    fixtures: set[str] = set()
    for base in ("tests/unit/nodes", "tests/integration/nodes", "tests/unit", "tests"):
        tests_dir = repo_path / base
        if not tests_dir.exists():
            continue
        for entry in tests_dir.iterdir():
            if entry.is_dir() and entry.name.startswith("node_"):
                fixtures.add(entry.name)
    return fixtures


def _all_node_names_in_repo(repo_path: Path) -> set[str]:
    """
    Return all node names found under src/ with any contract.yaml.
    """
    names: set[str] = set()
    for contract in repo_path.glob("src/**/contract.yaml"):
        # Skip .venv and .claude worktrees
        parts = contract.parts
        if ".venv" in parts or ".claude" in parts:
            continue
        node_dir = contract.parent
        names.add(node_dir.name)
    return names


# ---------------------------------------------------------------------------
# Repo scanning
# ---------------------------------------------------------------------------


def _find_git_repos(root: Path) -> list[Path]:
    """Return direct-child git repos under root."""
    repos: list[Path] = []
    for p in sorted(root.iterdir(), key=lambda x: x.name.lower()):
        if not p.is_dir():
            continue
        if (p / ".git").exists():
            repos.append(p)
    return repos


def scan_repo(
    repo_path: Path,
    docs_design_dir: Path,
) -> tuple[list[NodeEntry], list[OrphanedFixture]]:
    """
    Scan a single repo for Kafka-integrated nodes.

    Returns (nodes, orphaned_fixtures).
    """
    repo_name = repo_path.name
    nodes: list[NodeEntry] = []
    orphaned: list[OrphanedFixture] = []

    # Collect fixture dirs (for "Fixture Present" column)
    fixture_names = _fixture_paths_for_repo(repo_path)
    # Collect all node names (for orphan detection)
    all_node_names = _all_node_names_in_repo(repo_path)

    # Contracts to scan — skip .venv and .claude worktrees
    contracts: list[Path] = []
    for contract in repo_path.glob("src/**/contract.yaml"):
        parts = contract.parts
        if ".venv" in parts or ".claude" in parts:
            continue
        contracts.append(contract)

    # Track which node names have event_bus
    event_bus_nodes: set[str] = set()

    for contract_path in sorted(contracts):
        data = _load_yaml(contract_path)
        event_bus = data.get("event_bus")
        if not event_bus:
            continue

        subscribe_topics, publish_topics = _extract_topics(event_bus)
        if not subscribe_topics and not publish_topics:
            continue

        node_dir = contract_path.parent
        node_name = node_dir.name

        # Use node_name from contract if available, else directory name
        contract_node_name = data.get("name") or data.get("node_name") or node_name

        event_bus_nodes.add(node_name)

        fixture_present = node_name in fixture_names
        design_spec = find_design_spec(node_name, docs_design_dir)

        # Relative contract path from repo root
        try:
            rel_path = str(contract_path.relative_to(repo_path))
        except ValueError:
            rel_path = str(contract_path)

        nodes.append(
            NodeEntry(
                repo=repo_name,
                node_name=contract_node_name,
                subscribe_topics=subscribe_topics,
                publish_topics=publish_topics,
                design_spec=design_spec,
                fixture_present=fixture_present,
                manual_review="—",
                contract_path=rel_path,
            )
        )

    # Detect orphaned fixtures: fixture dirs that have no matching contract.yaml
    for fixture_name in sorted(fixture_names):
        if fixture_name not in all_node_names:
            # Find the actual path
            for base in (
                "tests/unit/nodes",
                "tests/integration/nodes",
                "tests/unit",
                "tests",
            ):
                candidate = repo_path / base / fixture_name
                if candidate.exists():
                    try:
                        rel = str(candidate.relative_to(repo_path))
                    except ValueError:
                        rel = str(candidate)
                    orphaned.append(OrphanedFixture(repo=repo_name, fixture_path=rel))
                    break

    return nodes, orphaned


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _topics_cell(topics: list[str]) -> str:
    if not topics:
        return "—"
    if len(topics) == 1:
        return f"`{topics[0]}`"
    return "<br>".join(f"`{t}`" for t in topics)


def render_registry(
    nodes: list[NodeEntry],
    orphaned: list[OrphanedFixture],
    generated_at: str,
) -> str:
    lines: list[str] = []

    lines.append("# FEATURE_REGISTRY.md")
    lines.append("")
    lines.append(
        "Inventory of all Kafka-integrated ONEX nodes across the OmniNode platform."
    )
    lines.append(
        f"Generated by `docs/tools/generate_feature_inventory.py` on {generated_at}."
    )
    lines.append("")
    lines.append(
        "> **Legend** — Fixture Present: `Y` = test fixture directory exists, `N` = no fixture found.  "
    )
    lines.append("> Manual Review: `—` unless flagged for human attention.  ")
    lines.append("> Design Spec: link to `docs/design/` doc, or `—` if none found.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Table header
    lines.append(
        "| Repo | Node | Subscribe Topics | Publish Topics"
        " | Design Spec | Fixture Present | Manual Review | Contract Path |"
    )
    lines.append(
        "|------|------|-----------------|----------------|"
        "-------------|-----------------|---------------|---------------|"
    )

    for n in nodes:
        fixture_val = "Y" if n.fixture_present else "N"
        subscribe_cell = _topics_cell(n.subscribe_topics)
        publish_cell = _topics_cell(n.publish_topics)
        lines.append(
            f"| {n.repo} | {n.node_name} | {subscribe_cell} | {publish_cell}"
            f" | {n.design_spec} | {fixture_val} | {n.manual_review} | `{n.contract_path}` |"
        )

    lines.append("")
    lines.append(f"**Total Kafka-integrated nodes**: {len(nodes)}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Summary by repo
    lines.append("## Summary by Repo")
    lines.append("")
    repo_counts: dict[str, dict[str, int]] = {}
    for n in nodes:
        if n.repo not in repo_counts:
            repo_counts[n.repo] = {"total": 0, "with_fixture": 0, "with_spec": 0}
        repo_counts[n.repo]["total"] += 1
        if n.fixture_present:
            repo_counts[n.repo]["with_fixture"] += 1
        if n.design_spec != "—":
            repo_counts[n.repo]["with_spec"] += 1

    lines.append("| Repo | Nodes | With Fixture | With Design Spec |")
    lines.append("|------|-------|--------------|-----------------|")
    for repo, counts in sorted(repo_counts.items()):
        lines.append(
            f"| {repo} | {counts['total']}"
            f" | {counts['with_fixture']} ({counts['with_fixture'] * 100 // max(counts['total'], 1)}%)"
            f" | {counts['with_spec']} ({counts['with_spec'] * 100 // max(counts['total'], 1)}%) |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")

    # Orphaned Fixtures section
    lines.append("## Orphaned Fixtures")
    lines.append("")
    lines.append(
        "Test fixture directories that have no matching `contract.yaml` in `src/`."
    )
    lines.append("")
    if orphaned:
        lines.append("| Repo | Fixture Path |")
        lines.append("|------|-------------|")
        for o in orphaned:
            lines.append(f"| {o.repo} | `{o.fixture_path}` |")
    else:
        lines.append("_No orphaned fixtures detected._")
    lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate FEATURE_REGISTRY.md for Kafka-integrated ONEX nodes."
    )
    ap.add_argument(
        "--root",
        type=str,
        default=os.environ.get(
            "OMNI_HOME", str(Path(__file__).resolve().parent.parent.parent)
        ),
        help="Workspace root to scan (direct children only). Defaults to $OMNI_HOME.",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help=("Output path. Defaults to <root>/docs/integration/FEATURE_REGISTRY.md"),
    )
    ap.add_argument(
        "--overrides",
        type=str,
        default=None,
        help="Path to feature_spec_overrides.yaml for manual overrides.",
    )
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()

    # Default output
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        out_path = root / "docs" / "integration" / "FEATURE_REGISTRY.md"

    # Design specs directory (relative to root)
    docs_design_dir = root / "docs" / "design"

    # Load overrides if provided
    overrides: dict[str, Any] = {}
    overrides_path = (
        Path(args.overrides).expanduser().resolve()
        if args.overrides
        else Path(__file__).resolve().parent / "feature_spec_overrides.yaml"
    )
    if overrides_path.exists():
        try:
            overrides = _load_yaml(overrides_path)
        except Exception:  # noqa: BLE001 — boundary: catch-all for resilience
            overrides = {}

    # Scan repos
    repos = _find_git_repos(root)

    all_nodes: list[NodeEntry] = []
    all_orphaned: list[OrphanedFixture] = []

    for repo in repos:
        # Only scan repos we care about (must have a src/ with Python nodes)
        src = repo / "src"
        if not src.exists():
            continue

        nodes, orphaned = scan_repo(repo, docs_design_dir)

        # Apply overrides from feature_spec_overrides.yaml
        for n in nodes:
            key = f"{n.repo}/{n.node_name}"
            if key in overrides:
                ov = overrides[key]
                if "design_spec" in ov:
                    n.design_spec = ov["design_spec"]
                if "manual_review" in ov:
                    n.manual_review = ov["manual_review"]
                if "fixture_present" in ov:
                    n.fixture_present = ov["fixture_present"]

        all_nodes.extend(nodes)
        all_orphaned.extend(orphaned)

    # Sort: by repo, then node_name
    all_nodes.sort(key=lambda n: (n.repo, n.node_name))
    all_orphaned.sort(key=lambda o: (o.repo, o.fixture_path))

    # Timestamp
    import datetime

    generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # Render
    content = render_registry(all_nodes, all_orphaned, generated_at)

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content)
    print(f"Wrote {out_path}")
    print(f"  Total Kafka-integrated nodes: {len(all_nodes)}")
    print(f"  Repos scanned: {len(repos)}")
    print(f"  Orphaned fixtures: {len(all_orphaned)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
