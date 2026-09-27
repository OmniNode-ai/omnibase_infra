#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Validate config/lab_proof_profiles.yaml (OMN-19565).

The registry is the per-repository proof-profile data behind the operator ruling
of 2026-09-25T13:07:40Z (every code PR gets a lab proof before merge). This is the
gate that keeps it honest, run by pre-commit and by CI:

  1. the file parses into the typed registry (every row-shape rule, including
     one row per registry repository and no ``enforce: true``, lives in the
     models under src/omnibase_infra/lab_proof/);
  2. every step a row declares names a node directory that exists in this
     repository and an operation that node's contract routes.

Exit 0 when both hold; exit 1 naming every problem otherwise.

``--self-test`` proves the gate is not vacuous: it removes each row in turn from
an in-memory copy of the registry and exits 0 only if every copy is refused,
naming the repository whose row is missing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from omnibase_infra.lab_proof.lab_proof_profile_registry import (
    load_lab_proof_profile_registry,
    validate_steps_against_repo,
)
from omnibase_infra.lab_proof.model_lab_proof_profile_registry import (
    ModelLabProofProfileRegistry,
)

DEFAULT_REGISTRY = REPO_ROOT / "config" / "lab_proof_profiles.yaml"


def self_test(registry_path: Path) -> int:
    """Every one-row-removed copy of the registry must be refused, naming the repo."""
    raw = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    rows = raw["profiles"]
    not_refused: list[str] = []
    for index, row in enumerate(rows):
        copy = {**raw, "profiles": rows[:index] + rows[index + 1 :]}
        try:
            ModelLabProofProfileRegistry.model_validate(copy)
        except ValidationError as exc:
            if row["repo"] in str(exc):
                continue
        not_refused.append(row["repo"])
    if not_refused:
        print(
            "self-test FAILED: a copy missing these rows was accepted: "
            + ", ".join(not_refused)
        )
        return 1
    print(f"self-test OK: each of {len(rows)} one-row-removed copies was refused")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test(args.registry)
    try:
        registry = load_lab_proof_profile_registry(args.registry)
    except (ValidationError, ValueError) as exc:
        print(f"lab proof registry INVALID: {args.registry}\n{exc}")
        return 1
    errors = validate_steps_against_repo(registry, REPO_ROOT)
    if errors:
        print(f"lab proof registry INVALID: {args.registry}")
        for error in errors:
            print(f"  - {error}")
        return 1
    runnable = sum(
        1
        for profile in registry.profiles
        for variant in profile.variants
        if variant.status in ("live", "pilot")
    )
    print(
        f"lab proof registry OK: {len(registry.profiles)} repositories, "
        f"{sum(len(p.variants) for p in registry.profiles)} variants, "
        f"{runnable} runnable, 0 enforced"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
