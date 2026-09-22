# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Validate one captured cohort key or compare two captured cohort keys.

Input JSON is a serialized ``ModelDelegationCohortKey``. A report producer must
populate it from captured request, runtime, and route evidence; this command
validates completeness and equality but cannot authenticate those sources.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from omnibase_infra.models.delegation import ModelDelegationCohortKey


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON key file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"key file must contain a JSON object: {path}")
    if "cohort_key" in payload:
        payload = payload["cohort_key"]
    elif "observed_key_fields" in payload:
        payload = payload["observed_key_fields"]
    if not isinstance(payload, dict):
        raise ValueError(f"key payload must be a JSON object: {path}")
    return payload


def _load_key(path: Path) -> ModelDelegationCohortKey:
    return ModelDelegationCohortKey.model_validate_json(json.dumps(_load_payload(path)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="captured cohort-key JSON")
    parser.add_argument(
        "candidate",
        type=Path,
        nargs="?",
        help="second captured cohort-key JSON to compare",
    )
    args = parser.parse_args(argv)

    try:
        baseline = _load_key(args.baseline)
        candidate = _load_key(args.candidate) if args.candidate else None
    except (ValueError, ValidationError) as exc:
        print(f"INVALID_COHORT_KEY: {exc}", file=sys.stderr)
        return 2

    if candidate is None:
        print(f"VALID_COHORT_KEY: {baseline.key_sha256}")
        return 0

    changed = baseline.changed_dimensions(candidate)
    if changed:
        print(f"CROSS_COHORT: {', '.join(changed)}")
        return 1
    print(f"SAME_COHORT: {baseline.key_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
