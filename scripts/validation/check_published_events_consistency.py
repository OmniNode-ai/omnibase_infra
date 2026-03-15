#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Validate that published_events topics are declared in event_bus.publish_topics.

For every contract.yaml found in src/omnibase_infra/nodes/:
  - Every published_events[].topic must appear in event_bus.publish_topics
  - Exits 1 if any mismatch is found, 0 on success

Usage:
    python scripts/validation/check_published_events_consistency.py
    python scripts/validation/check_published_events_consistency.py --contracts-root src/omnibase_infra/nodes

OMN-4885 / OMN-4880
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


class ContractConsistencyError(Exception):
    """Raised when a contract.yaml has inconsistent published_events vs event_bus.publish_topics."""


def check_contract_consistency(contract: dict[str, object], contract_path: str) -> None:
    """Raise ContractConsistencyError if published_events topics are not in event_bus.publish_topics.

    Args:
        contract: Dict loaded from a contract.yaml file via yaml.safe_load.
        contract_path: Path string used in error messages.

    Raises:
        ContractConsistencyError: If the contract is malformed or has inconsistent topics.
    """
    if not isinstance(contract, dict):
        raise ContractConsistencyError(
            f"{contract_path}: contract is not a dict (got {type(contract).__name__}). "
            "YAML file may be malformed."
        )

    published_events = contract.get("published_events", [])
    if not isinstance(published_events, list):
        raise ContractConsistencyError(
            f"{contract_path}: published_events is not a list "
            f"(got {type(published_events).__name__})"
        )

    event_bus = contract.get("event_bus") or {}
    if not isinstance(event_bus, dict):
        raise ContractConsistencyError(
            f"{contract_path}: event_bus is not a dict (got {type(event_bus).__name__})"
        )

    publish_topics_raw = event_bus.get("publish_topics", []) or []
    if not isinstance(publish_topics_raw, list):
        publish_topics_set: set[str] = set()
    else:
        publish_topics_set = {t for t in publish_topics_raw if isinstance(t, str)}

    errors: list[str] = []

    for entry in published_events:
        if not isinstance(entry, dict):
            continue
        topic = entry.get("topic")
        event_type = entry.get("event_type", "<unknown>")
        if isinstance(topic, str) and topic and topic not in publish_topics_set:
            errors.append(
                f"  published_events entry '{event_type}' declares topic '{topic}' "
                f"but it is missing from event_bus.publish_topics"
            )

    if errors:
        raise ContractConsistencyError(
            f"{contract_path}: {len(errors)} consistency error(s):\n"
            + "\n".join(errors)
        )


def main(contracts_root: str = "src/omnibase_infra/nodes") -> int:
    """Scan all contract.yaml files under contracts_root and check consistency.

    Returns:
        0 on success, 1 if any errors found.
    """
    root = Path(contracts_root)
    contract_files = sorted(root.rglob("contract.yaml"))
    errors: list[str] = []

    for contract_path in contract_files:
        try:
            contract_data = yaml.safe_load(contract_path.read_text())
            check_contract_consistency(contract_data, str(contract_path))
        except ContractConsistencyError as e:
            errors.append(str(e))

    if errors:
        print("published_events consistency check FAILED:\n")
        for err in errors:
            print(err)
            print()
        return 1

    print(
        f"published_events consistency check passed ({len(contract_files)} contracts scanned)"
    )
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate published_events topics appear in event_bus.publish_topics"
    )
    parser.add_argument(
        "--contracts-root",
        default="src/omnibase_infra/nodes",
        help="Root directory to scan for contract.yaml files",
    )
    args = parser.parse_args()
    sys.exit(main(args.contracts_root))
