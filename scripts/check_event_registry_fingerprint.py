#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""CI twin for event registry fingerprint validation (OMN-2149).

Thin wrapper around the existing ``event_registry`` verify CLI that
provides a uniform interface alongside the schema fingerprint CI twin.

This is the CI-side mirror of the B3 runtime assertion. The runtime calls
``validate_event_registry_fingerprint()`` at startup to compare the live
event registry (built from ``ALL_EVENT_REGISTRATIONS``) against the committed
artifact at ``event_registry_fingerprint.json``. This script runs the same
comparison in CI, failing the build when the artifact is stale.

Usage::

    # Verify: compare live registry to committed artifact
    python scripts/check_event_registry_fingerprint.py verify

    # Stamp: regenerate the artifact from ALL_EVENT_REGISTRATIONS
    python scripts/check_event_registry_fingerprint.py stamp

    # Dry-run: compute fingerprint without writing
    python scripts/check_event_registry_fingerprint.py stamp --dry-run

Exit codes:
    0 -- Artifact is current (verify) or stamp succeeded
    1 -- Usage error or unexpected failure
    2 -- Fingerprint mismatch (artifact is stale)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Default artifact path (co-located with event_registry module)
_DEFAULT_ARTIFACT = str(
    Path(__file__).resolve().parent.parent
    / "src"
    / "omnibase_infra"
    / "runtime"
    / "emit_daemon"
    / "event_registry_fingerprint.json"
)


def cmd_verify(artifact_path: str) -> int:
    """Verify committed artifact matches live event registrations.

    Returns:
        Exit code: 0 if matching, 2 if stale or missing.
    """
    from omnibase_infra.errors.error_event_registry_fingerprint import (
        EventRegistryFingerprintMismatchError,
        EventRegistryFingerprintMissingError,
    )
    from omnibase_infra.runtime.emit_daemon.event_registry import (
        validate_event_registry_fingerprint,
    )

    try:
        validate_event_registry_fingerprint(artifact_path=artifact_path)
        print("Event registry fingerprint OK")
        return 0
    except EventRegistryFingerprintMismatchError as exc:
        print(f"FAILED: {exc.message}", file=sys.stderr)
        print(
            "\nEvent registrations have changed but the artifact was not regenerated.",
            file=sys.stderr,
        )
        print(
            "Run: python scripts/check_event_registry_fingerprint.py stamp",
            file=sys.stderr,
        )
        return 2
    except EventRegistryFingerprintMissingError as exc:
        print(f"FAILED: {exc.message}", file=sys.stderr)
        print(
            "Run: python scripts/check_event_registry_fingerprint.py stamp",
            file=sys.stderr,
        )
        return 2


def cmd_stamp(artifact_path: str, *, dry_run: bool = False) -> int:
    """Regenerate the fingerprint artifact from ALL_EVENT_REGISTRATIONS.

    Returns:
        Exit code: 0 on success.
    """
    from omnibase_infra.runtime.emit_daemon.event_registry import _cli_stamp

    _cli_stamp(artifact_path, dry_run=dry_run)
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="check_event_registry_fingerprint",
        description="CI twin: event registry fingerprint drift detection (OMN-2149).",
    )
    sub = parser.add_subparsers(dest="command")

    stamp_parser = sub.add_parser(
        "stamp",
        help="Regenerate the fingerprint artifact from event registrations.",
    )
    stamp_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute fingerprint without writing the artifact.",
    )
    stamp_parser.add_argument(
        "--artifact",
        default=_DEFAULT_ARTIFACT,
        help=f"Path to fingerprint artifact (default: {_DEFAULT_ARTIFACT}).",
    )

    verify_parser = sub.add_parser(
        "verify",
        help="Verify committed artifact matches live event registrations.",
    )
    verify_parser.add_argument(
        "--artifact",
        default=_DEFAULT_ARTIFACT,
        help=f"Path to fingerprint artifact (default: {_DEFAULT_ARTIFACT}).",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "stamp":
        return cmd_stamp(args.artifact, dry_run=args.dry_run)
    elif args.command == "verify":
        return cmd_verify(args.artifact)

    return 1


if __name__ == "__main__":
    sys.exit(main())
