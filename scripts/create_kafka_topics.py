#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
#
# create_kafka_topics.py — contract-driven Kafka topic creator
#
# Reads topics from all contract.yaml files (via ContractTopicExtractor)
# and creates any missing topics on the Kafka broker.  Idempotent and safe
# for repeated runs.
#
# Ticket: OMN-2965
#
# Usage:
#   # Dry-run: print plan, no broker connection
#   uv run python scripts/create_kafka_topics.py --dry-run
#
#   # Create missing topics on broker
#   uv run python scripts/create_kafka_topics.py \
#       --bootstrap-servers localhost:19092
#
#   # Override defaults
#   uv run python scripts/create_kafka_topics.py \
#       --bootstrap-servers localhost:19092 \
#       --partitions 3 \
#       --replication-factor 1 \
#       --contracts-root src/omnibase_infra/nodes/
#
# Exit Codes:
#   0  Success (always in --dry-run; or all topics ensured in non-dry-run)
#   1  Broker or create failure
#   2  Missing --bootstrap-servers in non-dry-run mode
#
# Algorithm:
#   1. Extract topics via ContractTopicExtractor (no broker connection)
#   2. list_topics() from broker
#   3. Diff: determine which topics are missing
#   4. create_topics() for missing topics
#   5. list_topics() again (source of truth — do NOT branch on create_topics return)
#   6. Report final created count based on list_topics() diff
#
# Design decisions:
#   - confluent-kafka (sync): CLI tool — no async event loop needed.
#   - list_topics() is the source of truth, not create_topics() return value.
#   - Repo-root is discovered via Path(__file__).resolve(), not CWD.
#   - --dry-run never attempts a broker connection, even if --bootstrap-servers given.

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo-root discovery
# ---------------------------------------------------------------------------

# This script lives at scripts/create_kafka_topics.py.
# The repo root is one level up.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_DEFAULT_CONTRACTS_ROOT = _REPO_ROOT / "src" / "omnibase_infra" / "nodes"


# ---------------------------------------------------------------------------
# Multi-package discovery via entry points
# ---------------------------------------------------------------------------


def _discover_all_packages(
    *, filter_names: list[str] | None = None, lenient: bool = False
) -> list[tuple[str, Path]]:
    """Return [(name, nodes_dir_path)] for all ``onex.node_package`` entry points.

    Args:
        filter_names: If set, only return entry points whose ``ep.name`` is in
            this list.  Matches are case-sensitive.
        lenient: When ``True``, unloadable entry points emit a warning and are
            skipped.  When ``False`` (default), any load failure causes a hard
            exit (non-zero) because a broken entry point in the canonical path
            indicates a broken environment.

    Returns:
        Sorted list of ``(ep.name, Path)`` tuples.
    """
    eps = importlib.metadata.entry_points(group="onex.node_package")
    result: list[tuple[str, Path]] = []
    for ep in eps:
        if filter_names is not None and ep.name not in filter_names:
            continue
        try:
            pkg = ep.load()
        except Exception as exc:  # noqa: BLE001 — boundary: catch-all for resilience
            msg = f"Could not load entry point {ep.name!r}: {exc}"
            if lenient:
                print(f"WARNING: {msg} — skipping", file=sys.stderr)
                continue
            print(f"ERROR: {msg}", file=sys.stderr)
            sys.exit(1)
        pkg_path = Path(pkg.__path__[0])
        if not pkg_path.exists():
            msg = f"Entry point {ep.name!r} path does not exist: {pkg_path}"
            if lenient:
                print(f"WARNING: {msg} — skipping", file=sys.stderr)
                continue
            print(f"ERROR: {msg}", file=sys.stderr)
            sys.exit(1)
        result.append((ep.name, pkg_path))
    return sorted(result, key=lambda t: t[0])


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="create_kafka_topics.py",
        description=(
            "Contract-driven Kafka topic creator. "
            "Reads topics from contract.yaml files and creates any missing topics "
            "on the Kafka broker. Idempotent — safe to run repeatedly.\n\n"
            "Default mode (no --contracts-root): discovers all installed packages "
            "that declare an onex.node_package entry point and merges their topics. "
            "Use --contracts-root to scan a single directory instead (local dev)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit codes:
  0  Success (always in --dry-run; or all topics ensured in non-dry-run)
  1  Broker or topic creation failure / unloadable entry point
  2  Missing --bootstrap-servers in non-dry-run mode

Examples:
  # Dry-run with multi-package discovery (default)
  uv run python scripts/create_kafka_topics.py --dry-run

  # Dry-run restricted to specific packages
  uv run python scripts/create_kafka_topics.py --dry-run \\
      --packages omnibase_infra,omniclaude

  # Dry-run with lenient mode (skip broken entry points)
  uv run python scripts/create_kafka_topics.py --dry-run --lenient

  # Create missing topics on broker
  uv run python scripts/create_kafka_topics.py \\
      --bootstrap-servers localhost:19092

  # Legacy single-root mode (local dev)
  uv run python scripts/create_kafka_topics.py \\
      --bootstrap-servers localhost:19092 \\
      --contracts-root src/omnibase_infra/nodes/
""",
    )
    parser.add_argument(
        "--bootstrap-servers",
        metavar="HOST:PORT",
        default=None,
        help=(
            "Kafka bootstrap servers (e.g. localhost:19092). "
            "Required in non-dry-run mode. Optional in --dry-run."
        ),
    )
    parser.add_argument(
        "--partitions",
        type=int,
        default=1,
        metavar="N",
        help="Number of partitions for new topics (default: 1).",
    )
    parser.add_argument(
        "--replication-factor",
        type=int,
        default=1,
        metavar="N",
        help="Replication factor for new topics (default: 1).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Print the list of topics that would be ensured without connecting "
            "to the broker or creating any topics."
        ),
    )
    parser.add_argument(
        "--contracts-root",
        metavar="PATH",
        default=None,
        help=(
            "Root directory to scan for contract.yaml files (single-root mode). "
            "When omitted, the script discovers all installed onex.node_package "
            "entry points and scans each package's nodes directory (multi-package "
            "mode). Use this flag for local dev or to override discovery."
        ),
    )
    parser.add_argument(
        "--skills-root",
        metavar="PATH",
        default=None,
        help=(
            "Path to omniclaude plugins/onex/skills/ directory. "
            "When set, topics.yaml manifests from each skill are discovered "
            "and merged with contract-extracted topics. Enables cross-repo "
            "topic discovery in CI. Optional — omitted in contract-only runs."
        ),
    )
    parser.add_argument(
        "--packages",
        metavar="PKG1,PKG2",
        default=None,
        help=(
            "Comma-separated list of onex.node_package entry point names to "
            "scan (e.g. --packages omnibase_infra,omniclaude). "
            "Only effective in multi-package mode (when --contracts-root is "
            "not given). Matches against ep.name."
        ),
    )
    parser.add_argument(
        "--lenient",
        action="store_true",
        default=False,
        help=(
            "In multi-package mode, skip unloadable entry points with a "
            "warning instead of exiting non-zero. Useful for debugging "
            "partial environments."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _run_dry(
    topics: list[str],
    bootstrap_servers: str | None,
    contracts_root: Path | None,
    *,
    topic_sources: dict[str, list[str]] | None = None,
) -> int:
    """
    Execute dry-run: print plan without any broker connection.

    Always exits 0 (unless internal error).

    Args:
        topics: Sorted list of topic strings.
        bootstrap_servers: Broker address (display only).
        contracts_root: Single contracts root (legacy mode) or None.
        topic_sources: Optional mapping of topic -> list of package names
            that declared it.  Used in multi-package mode for diagnostics.
    """
    bs_display = bootstrap_servers if bootstrap_servers else "<unset>"
    print(f"Bootstrap servers: {bs_display}")
    if contracts_root is not None:
        print(f"Contracts root: {contracts_root}")
    if topic_sources:
        # Report per-topic provenance in multi-package mode
        multi = {t: pkgs for t, pkgs in topic_sources.items() if len(pkgs) > 1}
        if multi:
            print(f"\nTopics declared by multiple packages ({len(multi)}):")
            for t in sorted(multi):
                print(f"  [WARN] {t} declared by: {', '.join(sorted(multi[t]))}")
    print(f"\nTopics to ensure exist ({len(topics)}):")
    for topic in sorted(topics):
        src = ""
        if topic_sources and topic in topic_sources:
            src = f"  ({', '.join(sorted(topic_sources[topic]))})"
        print(f"  - {topic}{src}")
    return 0


def _run_live(
    topics: list[str],
    bootstrap_servers: str,
    partitions: int,
    replication_factor: int,
    contracts_root: Path,
) -> int:
    """
    Connect to broker, diff existing topics, create missing ones.

    Returns 0 on success, 1 on broker or creation failure.
    """
    try:
        from confluent_kafka.admin import (  # type: ignore[attr-defined]
            AdminClient,
            NewTopic,
        )
    except ImportError as import_exc:
        print(
            f"ERROR: confluent-kafka not available: {import_exc}\n"
            "Install with: pip install confluent-kafka",
            file=sys.stderr,
        )
        return 1

    admin: AdminClient | None = None
    try:
        print(f"Connecting to broker: {bootstrap_servers}")
        admin = AdminClient({"bootstrap.servers": bootstrap_servers})

        # Step 1: List existing topics (source of truth — before)
        print("Listing existing topics...")
        cluster_metadata = admin.list_topics(timeout=10)
        existing_topics: set[str] = set(cluster_metadata.topics.keys())

        # Step 2: Diff — missing topics only
        topic_set = set(topics)
        missing = sorted(topic_set - existing_topics)

        if not missing:
            print(f"All {len(topic_set)} topics already exist. Nothing to create.")
            return 0

        print(f"Topics to create ({len(missing)}):")
        for t in missing:
            print(f"  + {t}")

        # Step 3: Create missing topics
        new_topics = [
            NewTopic(
                t,
                num_partitions=partitions,
                replication_factor=replication_factor,
            )
            for t in missing
        ]
        futures = admin.create_topics(new_topics)

        # Collect create results (best-effort: log per-topic errors)
        create_errors: list[str] = []
        for topic_name, future in futures.items():
            topic_exc = future.exception()
            if topic_exc is not None:
                # TOPIC_ALREADY_EXISTS is not a real error (race condition)
                from confluent_kafka import KafkaException

                if isinstance(topic_exc, KafkaException):
                    kafka_err = topic_exc.args[0]
                    if hasattr(kafka_err, "code") and "TOPIC_ALREADY_EXISTS" in str(
                        kafka_err.code()
                    ):
                        # Harmless — topic was created concurrently
                        continue
                create_errors.append(f"  {topic_name}: {topic_exc}")

        if create_errors:
            print("WARNING: Some topics failed to create:", file=sys.stderr)
            for err in create_errors:
                print(err, file=sys.stderr)

        # Step 4: list_topics() is the source of truth — re-check after create
        cluster_metadata_after = admin.list_topics(timeout=10)
        existing_after: set[str] = set(cluster_metadata_after.topics.keys())
        # Topics from our set that now exist but didn't before (newly_present is the truth)
        newly_present = sorted((topic_set & existing_after) - existing_topics)

        print(f"\nResult: {len(newly_present)} topics created successfully.")
        if newly_present:
            for t in newly_present:
                print(f"  + {t}")

        # If topics we needed are still missing after create, that's an error
        still_missing = sorted(topic_set - existing_after)
        if still_missing:
            print(
                f"\nERROR: {len(still_missing)} topics are still missing after creation:",
                file=sys.stderr,
            )
            for t in still_missing:
                print(f"  - {t}", file=sys.stderr)
            return 1

        return 0

    except Exception as exc:  # noqa: BLE001 — boundary: prints error and degrades
        print(f"ERROR: Broker operation failed: {exc}", file=sys.stderr)
        return 1
    finally:
        # confluent_kafka AdminClient does not have a close() method in all versions
        # The GC handles cleanup; nothing to do here explicitly.
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Main entry point. Returns exit code."""
    parser = _build_parser()
    args = parser.parse_args()

    # In non-dry-run mode, --bootstrap-servers is required
    if not args.dry_run and not args.bootstrap_servers:
        print(
            "ERROR: --bootstrap-servers is required in non-dry-run mode.\n"
            "Use --dry-run to print the topic plan without connecting to a broker.",
            file=sys.stderr,
        )
        return 2

    # Resolve optional skills root (--skills-root)
    skill_manifests_root: Path | None = None
    if args.skills_root is not None:
        skill_manifests_root = Path(args.skills_root).resolve()
        if not skill_manifests_root.exists():
            print(
                f"WARNING: --skills-root does not exist: {skill_manifests_root} — "
                "skill topic discovery will be skipped.",
                file=sys.stderr,
            )
            skill_manifests_root = None

    # Import here so the script fails fast if omnibase_infra is not installed
    from omnibase_infra.tools.contract_topic_extractor import ContractTopicExtractor

    extractor = ContractTopicExtractor()

    # -----------------------------------------------------------------------
    # Mode selection: single-root (legacy) vs multi-package (new default)
    # -----------------------------------------------------------------------
    use_single_root = args.contracts_root is not None

    if use_single_root:
        # Legacy single-root mode (--contracts-root given)
        contracts_root = Path(args.contracts_root).resolve()
        if not contracts_root.exists():
            print(
                f"ERROR: contracts root does not exist: {contracts_root}",
                file=sys.stderr,
            )
            return 1

        try:
            if skill_manifests_root is not None:
                entries = extractor.extract_all(
                    contracts_root=contracts_root,
                    skill_manifests_root=skill_manifests_root,
                )
            else:
                entries = extractor.extract(contracts_root)
        except Exception as exc:  # noqa: BLE001 — boundary: prints error and degrades
            print(
                f"ERROR: Failed to extract topics from contracts: {exc}",
                file=sys.stderr,
            )
            return 1

        if not entries:
            print(
                f"WARNING: No topics found in contracts root: {contracts_root}",
                file=sys.stderr,
            )
            return 0

        topics = [e.topic for e in entries]

        if args.dry_run:
            if skill_manifests_root is not None:
                print(f"Skills root: {skill_manifests_root}")
            return _run_dry(topics, args.bootstrap_servers, contracts_root)

        return _run_live(
            topics,
            bootstrap_servers=args.bootstrap_servers,
            partitions=args.partitions,
            replication_factor=args.replication_factor,
            contracts_root=contracts_root,
        )

    # -------------------------------------------------------------------
    # Multi-package mode (default when --contracts-root is omitted)
    # -------------------------------------------------------------------
    filter_names: list[str] | None = None
    if args.packages:
        filter_names = [n.strip() for n in args.packages.split(",") if n.strip()]

    packages = _discover_all_packages(filter_names=filter_names, lenient=args.lenient)

    if not packages:
        print(
            "WARNING: No onex.node_package entry points found. "
            "Install app packages or use --contracts-root for local dev.",
            file=sys.stderr,
        )
        return 0

    print(f"Discovered {len(packages)} onex.node_package entry points:")
    for name, path in packages:
        print(f"  {name} -> {path}")

    # Extract and merge topics from all packages
    # Track which package(s) declared each topic for diagnostics
    from omnibase_infra.tools.contract_topic_extractor import ModelContractTopicEntry

    topic_sources: dict[str, list[str]] = {}
    all_entries: list[ModelContractTopicEntry] = []

    try:
        for pkg_name, pkg_path in packages:
            if skill_manifests_root is not None:
                pkg_entries = extractor.extract_all(
                    contracts_root=pkg_path,
                    skill_manifests_root=skill_manifests_root,
                )
            else:
                pkg_entries = extractor.extract(pkg_path)

            print(f"  {pkg_name}: {len(pkg_entries)} topics")
            for entry in pkg_entries:
                topic_sources.setdefault(entry.topic, []).append(pkg_name)
            all_entries.extend(pkg_entries)
    except Exception as exc:  # noqa: BLE001 — boundary: prints error and degrades
        print(
            f"ERROR: Failed to extract topics: {exc}",
            file=sys.stderr,
        )
        return 1

    # Deduplicate topics across packages
    unique_topics = sorted({e.topic for e in all_entries})

    if not unique_topics:
        print("WARNING: No topics found across any packages.", file=sys.stderr)
        return 0

    if args.dry_run:
        if skill_manifests_root is not None:
            print(f"Skills root: {skill_manifests_root}")
        return _run_dry(
            unique_topics,
            args.bootstrap_servers,
            None,
            topic_sources=topic_sources,
        )

    return _run_live(
        unique_topics,
        bootstrap_servers=args.bootstrap_servers,
        partitions=args.partitions,
        replication_factor=args.replication_factor,
        contracts_root=packages[0][1],  # display first package path
    )


if __name__ == "__main__":
    sys.exit(main())
