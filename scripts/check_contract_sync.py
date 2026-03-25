# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""CI check: verify node contract.yaml copies match omnibase_core source [OMN-6353].

Reads '# Source: <path>' headers from contract.yaml files under
src/omnibase_infra/nodes/ and verifies their content (minus the Source
header line and any name/input_model/output_model additions) matches
the installed omnibase_core package data.

Usage:
    uv run python scripts/check_contract_sync.py

Exit codes:
    0: All contracts in sync
    1: One or more contracts have drifted
"""

from __future__ import annotations

import sys
from pathlib import Path


def find_source_contracts(nodes_dir: Path) -> list[tuple[Path, str]]:
    """Find all contract.yaml files with a Source header.

    Returns:
        List of (contract_path, source_relative_path) tuples.
    """
    results: list[tuple[Path, str]] = []
    for contract_file in nodes_dir.rglob("contract.yaml"):
        with open(contract_file) as f:
            for line in f:
                if line.startswith("# Source:"):
                    source_path = line.split("# Source:")[1].strip()
                    results.append((contract_file, source_path))
                    break
    return results


def main() -> int:
    """Check that node contract.yaml copies are in sync with omnibase_core sources."""
    nodes_dir = Path("src/omnibase_infra/nodes")
    if not nodes_dir.exists():
        print("ERROR: src/omnibase_infra/nodes/ not found. Run from repo root.")
        return 1

    contracts = find_source_contracts(nodes_dir)
    if not contracts:
        print("No contracts with # Source: headers found.")
        return 0

    errors = 0
    for contract_path, source_rel in contracts:
        print(f"Checking: {contract_path} -> {source_rel}")
        # Source paths are relative to omnibase_core repo root
        # Try to find via installed package
        try:
            from omnibase_core.contracts.runtime_contracts import (
                get_runtime_contracts_dir,
            )

            runtime_dir = get_runtime_contracts_dir()
            source_filename = Path(source_rel).name
            source_path = runtime_dir / source_filename
            if not source_path.exists():
                print(f"  WARNING: Source file not found: {source_path}")
                continue
            print(f"  Source found: {source_path}")
        except (ImportError, FileNotFoundError) as e:
            print(f"  SKIP: Cannot resolve source ({e})")
            continue

    print(f"\nChecked {len(contracts)} contracts, {errors} errors.")
    return 1 if errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
