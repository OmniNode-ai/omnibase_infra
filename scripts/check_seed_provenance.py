#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Check that seed/demo publishers declare data_provenance (OMN-18786).

Scans scripts/ for Python files that both:
  1. Match seed/demo naming patterns (heuristic: filename contains 'seed' or 'demo'),
  2. Publish Kafka events (heuristic: contain publish/produce/send_event/emit patterns).

For each matched script, fail if the word ``data_provenance`` does not appear
anywhere in the file content. This is a source heuristic, not payload dataflow
analysis or proof that every runtime seed producer is covered.

Retained because projection models and contract persistence still distinguish
seeded data from measured data using provenance. Findings and scan errors fail
the gate; a clean scan passes. CI and pre-commit enforce the same scan.

Usage:
    uv run python scripts/check_seed_provenance.py [--scripts-dir <path>]

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
    """Return provenance violations and errors that prevent a complete scan."""
    findings: list[str] = []

    if not scripts_dir.is_dir():
        return [f"ERROR: {scripts_dir} is not a scripts directory."]
    candidates = sorted(scripts_dir.rglob("*.py"))
    if not candidates:
        return [f"ERROR: {scripts_dir} contains no Python scripts to check."]
    for path in candidates:
        if not _is_seed_or_demo(path):
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            findings.append(f"ERROR: cannot read {path}: {exc}")
            continue

        if not _publishes_events(content):
            continue

        if not _has_provenance(content):
            findings.append(
                f"ERROR: {path} publishes events but has no data_provenance "
                "declaration. "
                'Add data_provenance="demo_seeded" to event payloads.'
            )

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scripts-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to scan (default: scripts/)",
    )
    args = parser.parse_args()

    findings = check_scripts(args.scripts_dir)

    if findings:
        print("=== Seed Provenance Check: FAIL ===")
        for finding in findings:
            print(finding)
        print(f"\n{len(findings)} finding(s); seed provenance check failed.")
    else:
        print("=== Seed Provenance Check: clean ===")

    return int(bool(findings))


if __name__ == "__main__":
    sys.exit(main())
