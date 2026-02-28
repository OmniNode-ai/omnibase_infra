#!/usr/bin/env python3
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
#       --bootstrap-servers 192.168.86.200:29092
#
#   # Override defaults
#   uv run python scripts/create_kafka_topics.py \
#       --bootstrap-servers 192.168.86.200:29092 \
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
# CLI argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="create_kafka_topics.py",
        description=(
            "Contract-driven Kafka topic creator. "
            "Reads topics from contract.yaml files and creates any missing topics "
            "on the Kafka broker. Idempotent — safe to run repeatedly."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit codes:
  0  Success (always in --dry-run; or all topics ensured in non-dry-run)
  1  Broker or topic creation failure
  2  Missing --bootstrap-servers in non-dry-run mode

Examples:
  # Dry-run: print plan without connecting to broker
  uv run python scripts/create_kafka_topics.py --dry-run

  # Create missing topics on broker
  uv run python scripts/create_kafka_topics.py \\
      --bootstrap-servers 192.168.86.200:29092

  # Custom partitions and contracts root
  uv run python scripts/create_kafka_topics.py \\
      --bootstrap-servers 192.168.86.200:29092 \\
      --partitions 3 \\
      --contracts-root src/omnibase_infra/nodes/
""",
    )
    parser.add_argument(
        "--bootstrap-servers",
        metavar="HOST:PORT",
        default=None,
        help=(
            "Kafka bootstrap servers (e.g. 192.168.86.200:29092). "
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
            f"Root directory to scan for contract.yaml files. "
            f"Default: {_DEFAULT_CONTRACTS_ROOT}"
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _run_dry(
    topics: list[str], bootstrap_servers: str | None, contracts_root: Path
) -> int:
    """
    Execute dry-run: print plan without any broker connection.

    Always exits 0 (unless internal error).
    """
    bs_display = bootstrap_servers if bootstrap_servers else "<unset>"
    print(f"Bootstrap servers: {bs_display}")
    print(f"Contracts root: {contracts_root}")
    print(f"Topics to ensure exist ({len(topics)}):")
    for topic in sorted(topics):
        print(f"  - {topic}")
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

    except Exception as exc:
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

    # Resolve contracts root
    contracts_root: Path
    if args.contracts_root is not None:
        contracts_root = Path(args.contracts_root).resolve()
    else:
        contracts_root = _DEFAULT_CONTRACTS_ROOT

    if not contracts_root.exists():
        print(
            f"ERROR: contracts root does not exist: {contracts_root}",
            file=sys.stderr,
        )
        return 1

    # In non-dry-run mode, --bootstrap-servers is required
    if not args.dry_run and not args.bootstrap_servers:
        print(
            "ERROR: --bootstrap-servers is required in non-dry-run mode.\n"
            "Use --dry-run to print the topic plan without connecting to a broker.",
            file=sys.stderr,
        )
        return 2

    # Extract topics from contract files
    try:
        # Import here so the script fails fast if omnibase_infra is not installed
        from omnibase_infra.tools.contract_topic_extractor import ContractTopicExtractor

        extractor = ContractTopicExtractor()
        entries = extractor.extract(contracts_root)
    except Exception as exc:
        print(f"ERROR: Failed to extract topics from contracts: {exc}", file=sys.stderr)
        return 1

    if not entries:
        print(
            f"WARNING: No topics found in contracts root: {contracts_root}",
            file=sys.stderr,
        )
        # Not a failure — contracts may legitimately have no topics yet
        return 0

    topics = [e.topic for e in entries]

    if args.dry_run:
        return _run_dry(topics, args.bootstrap_servers, contracts_root)

    return _run_live(
        topics,
        bootstrap_servers=args.bootstrap_servers,
        partitions=args.partitions,
        replication_factor=args.replication_factor,
        contracts_root=contracts_root,
    )


if __name__ == "__main__":
    sys.exit(main())
