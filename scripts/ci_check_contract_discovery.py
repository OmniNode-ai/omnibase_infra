#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""CI invariant: node contracts discoverable from package root (OMN-3900).

Verifies that ``ContractConfigExtractor`` finds non-zero config requirements
when scanning the installed ``omnibase_infra`` package root.  This catches the
``/app/contracts`` vs ``nodes/`` path mismatch that caused Infisical prefetch
to silently skip all config requirements.

Design decisions:
    - Checks a **minimum baseline** of transport types (``{"database"}``),
      not an exhaustive frozen list.  New transports can be added without
      updating this script.
    - Intent: catch gross regressions (e.g., no contracts found at all,
      or a path change that makes contracts invisible to the extractor).

Usage::

    # Run as CI check (exit 0 = pass, exit 1 = fail)
    python scripts/ci_check_contract_discovery.py

    # Verbose mode for debugging
    python scripts/ci_check_contract_discovery.py --verbose

Exit codes:
    0 -- All checks passed
    1 -- One or more checks failed (diagnostic output on stderr)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Baseline expectations
# ---------------------------------------------------------------------------
# Minimum transport types that MUST appear in extracted requirements.
# This is intentionally a small set -- we only want to catch gross
# regressions, not enforce an exhaustive transport catalog.
# NOTE: uses enum *values* (e.g. "db"), not aliases (e.g. "database").
_BASELINE_TRANSPORTS: frozenset[str] = frozenset({"db"})


def _derive_package_root() -> Path:
    """Derive the ``omnibase_infra`` package root from its ``__file__``.

    Returns:
        Path to the directory containing the ``omnibase_infra`` package
        (i.e., ``<prefix>/omnibase_infra/``).
    """
    import omnibase_infra

    return Path(omnibase_infra.__file__).resolve().parent


def check_contract_discovery(*, verbose: bool = False) -> int:
    """Run all contract discovery checks.

    Args:
        verbose: If True, print detailed diagnostic output to stdout.

    Returns:
        Exit code: 0 if all checks pass, 1 if any fail.
    """
    from omnibase_infra.runtime.config_discovery import ContractConfigExtractor

    package_root = _derive_package_root()
    nodes_dir = package_root / "nodes"

    if verbose:
        print(f"Package root: {package_root}")
        print(f"Nodes dir:    {nodes_dir}")

    # ------------------------------------------------------------------
    # Check 1: nodes directory exists
    # ------------------------------------------------------------------
    if not nodes_dir.is_dir():
        print(
            f"FAILED: nodes directory does not exist: {nodes_dir}",
            file=sys.stderr,
        )
        return 1

    # ------------------------------------------------------------------
    # Check 2: extractor finds non-zero requirements
    # ------------------------------------------------------------------
    extractor = ContractConfigExtractor()
    result = extractor.extract_from_paths([nodes_dir])

    if verbose:
        print(f"Contracts scanned: {len(result.contract_paths)}")
        print(f"Requirements found: {len(result.requirements)}")
        print(f"Transport types: {sorted(t.value for t in result.transport_types)}")
        if result.errors:
            print(f"Extraction errors: {len(result.errors)}")
            for err in result.errors:
                print(f"  - {err}")

    if len(result.contract_paths) == 0:
        print(
            f"FAILED: no contract YAML files found under {nodes_dir}",
            file=sys.stderr,
        )
        return 1

    if len(result.requirements) == 0:
        print(
            "FAILED: extractor found 0 config requirements. "
            "This likely means the contract scan path is wrong "
            "or contracts have no transport_type / handler_type fields.",
            file=sys.stderr,
        )
        return 1

    # ------------------------------------------------------------------
    # Check 3: baseline transport types present
    # ------------------------------------------------------------------
    discovered_transport_values: set[str] = {t.value for t in result.transport_types}

    missing = _BASELINE_TRANSPORTS - discovered_transport_values
    if missing:
        print(
            f"FAILED: baseline transport types missing from discovery: "
            f"{sorted(missing)}. "
            f"Discovered: {sorted(discovered_transport_values)}",
            file=sys.stderr,
        )
        return 1

    # ------------------------------------------------------------------
    # All checks passed
    # ------------------------------------------------------------------
    print(
        f"OK: contract discovery found {len(result.requirements)} "
        f"requirements from {len(result.contract_paths)} contracts "
        f"({len(result.transport_types)} transport types)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(
        prog="ci_check_contract_discovery",
        description=(
            "CI invariant: verify node contracts are discoverable "
            "from the omnibase_infra package root (OMN-3900)."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed diagnostic output.",
    )
    args = parser.parse_args(argv)

    try:
        return check_contract_discovery(verbose=args.verbose)
    except Exception as exc:  # noqa: BLE001 — boundary: prints error and degrades
        print(
            f"FAILED: unexpected error during contract discovery check: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
