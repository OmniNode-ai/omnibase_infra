#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Batch-validate all node contract.yaml files via model_validate().

Walks all nodes/*/contract.yaml files under a given directory and runs
ModelNodeTypeProbe.model_validate() (the OMN-9746 choke-point for node_type)
on each one. Any contract that fails node_type validation, YAML parsing, or
structure checks causes a non-zero exit.

Usage:
    uv run python scripts/batch_validate_node_contracts.py
    uv run python scripts/batch_validate_node_contracts.py --directory src/omnibase_infra/nodes/
    uv run python scripts/batch_validate_node_contracts.py --verbose

Exit Codes:
    0: All contracts passed model_validate()
    1: One or more contracts failed validation
    2: Runtime error (directory not found, import failure)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _discover_contract_paths(directory: Path, recursive: bool = True) -> list[Path]:
    """Return sorted list of contract.yaml paths under directory."""
    if not directory.is_dir():
        return []
    if recursive:
        return sorted(directory.rglob("contract.yaml"))
    return sorted(directory.glob("*/contract.yaml"))


def _validate_contract_node_type(contract_path: Path) -> str:
    """Validate a single contract.yaml and return the node_type value.

    Uses ModelNodeTypeProbe.model_validate() — the OMN-9746 choke-point — to
    validate node_type without requiring handler_routing (which is optional for
    COMPUTE/REDUCER nodes that use FSM contracts instead of handler routing).

    Args:
        contract_path: Path to the contract.yaml file.

    Returns:
        The validated node_type string (e.g. "COMPUTE_GENERIC").

    Raises:
        ValueError: If YAML is invalid, empty, not a dict, or node_type is
            missing/unrecognised.
    """
    import yaml
    from pydantic import ValidationError

    from omnibase_infra.runtime.contract_loaders.model_node_type_probe import (
        ModelNodeTypeProbe,
    )

    try:
        raw = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML parse error: {exc}") from exc

    if raw is None:
        raise ValueError("Contract file is empty")
    if not isinstance(raw, dict):
        raise ValueError(f"Contract YAML root is not a dict: got {type(raw).__name__}")

    node_type_str = raw.get("node_type", "")
    if not node_type_str:
        raise ValueError(f"Missing or empty node_type in contract: {list(raw.keys())}")

    try:
        probe = ModelNodeTypeProbe.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(
            f"node_type validation failed ({node_type_str!r}): {exc}"
        ) from exc

    return probe.node_type.value


def run_batch_validation(
    directory: Path,
    *,
    verbose: bool = False,
    recursive: bool = True,
) -> tuple[int, int, list[tuple[Path, str]]]:
    """Validate all contract.yaml files under directory.

    Returns:
        (passed_count, failed_count, failures) where failures is a list of
        (path, error_message) tuples.
    """
    contract_paths = _discover_contract_paths(directory, recursive=recursive)
    if not contract_paths:
        print(f"WARNING: No contract.yaml files found under {directory}")
        return 0, 0, []

    passed = 0
    failures: list[tuple[Path, str]] = []

    for path in contract_paths:
        try:
            node_type = _validate_contract_node_type(path)
            passed += 1
            if verbose:
                print(f"  PASS  {path}  (node_type={node_type})")
        except ImportError:
            raise
        except Exception as exc:  # noqa: BLE001
            failures.append((path, str(exc)))
            if verbose:
                print(f"  FAIL  {path}")
                print(f"        {exc}")
            if not verbose:
                print(str(exc), file=sys.stderr)

    return passed, len(failures), failures


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Batch-validate all node contract.yaml files via model_validate().",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default="src/omnibase_infra/nodes/",
        help="Root directory to search for contract.yaml files (default: src/omnibase_infra/nodes/)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print each contract path and result",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Only check contract.yaml one level deep (not recursive)",
    )
    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.is_dir():
        print(
            f"ERROR: Directory not found or not a directory: {directory}",
            file=sys.stderr,
        )
        return 2

    try:
        passed, failed, failures = run_batch_validation(
            directory,
            verbose=args.verbose,
            recursive=not args.non_recursive,
        )
    except ImportError as exc:
        print(
            f"ERROR: Import failure — ensure omnibase_infra is installed: {exc}",
            file=sys.stderr,
        )
        return 2

    total = passed + failed
    print(f"Batch contract validation: {passed}/{total} passed, {failed} failed")

    if failures:
        print("\nFailed contracts:")
        for path, msg in failures:
            print(f"  {path}")
            print(f"    {msg}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
