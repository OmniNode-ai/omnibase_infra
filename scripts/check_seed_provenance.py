#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Advisory lint: check that seed/demo scripts set data_provenance in event payloads.

Scans scripts/ for Python files that both:
  1. Match seed/demo naming patterns (heuristic: filename contains 'seed' or 'demo'),
  2. Publish Kafka events (heuristic: contain publish/produce/send_event/emit patterns).

For each matched script, warns if the word ``data_provenance`` does not appear
anywhere in the file content.

This is ADVISORY — exit 0 always. CI uses ``continue-on-error: true``.

Usage:
    uv run python scripts/check_seed_provenance.py [--scripts-dir <path>]

OMN-11208
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_EVENT_PATTERNS = re.compile(
    r"\b(publish|produce|send_event|emit|send_and_wait|AIOKafkaProducer)\b",
    re.MULTILINE,
)
_PROVENANCE_PATTERN = re.compile(r"\bdata_provenance\b", re.MULTILINE)
_SEED_DEMO_PATTERN = re.compile(r"(seed|demo)", re.IGNORECASE)


def _is_seed_or_demo(path: Path) -> bool:
    return bool(_SEED_DEMO_PATTERN.search(path.stem))


def _publishes_events(content: str) -> bool:
    return bool(_EVENT_PATTERNS.search(content))


def _has_provenance(content: str) -> bool:
    return bool(_PROVENANCE_PATTERN.search(content))


def check_scripts(scripts_dir: Path) -> list[str]:
    """Return list of warning strings for scripts missing provenance."""
    warnings: list[str] = []

    candidates = sorted(scripts_dir.glob("*.py"))
    for path in candidates:
        if not _is_seed_or_demo(path):
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue

        if not _publishes_events(content):
            continue

        if not _has_provenance(content):
            warnings.append(
                f"WARNING: {path.name} publishes events but does not set "
                "data_provenance in any payload. "
                'Consider adding data_provenance="demo_seeded" to event payloads.'
            )

    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scripts-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to scan (default: scripts/)",
    )
    args = parser.parse_args()

    warnings = check_scripts(args.scripts_dir)

    if warnings:
        print("=== Seed Provenance Advisory Check ===")
        for w in warnings:
            print(w)
        print(
            f"\n{len(warnings)} script(s) may be missing data_provenance. "
            "This is advisory — no action required to pass CI."
        )
    else:
        print("=== Seed Provenance Advisory Check: clean ===")

    return 0


if __name__ == "__main__":
    sys.exit(main())
