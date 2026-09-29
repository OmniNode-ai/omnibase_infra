# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Enforce a bijection between contract board checks and criterion coverage.

Run with ``uv run python -m
omnibase_infra.nodes.node_board_probe_effect.coverage_validator`` in CI or
pre-commit. The CLI also resolves each declared operation against routing.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from omnibase_infra.nodes.node_board_probe_effect.models.model_board_check_coverage import (
    ModelBoardCheckCoverage,
)


def validate_coverage(
    board_checks: Sequence[Mapping[str, object]],
    coverage_rows: Sequence[Mapping[str, object]],
) -> None:
    """Refuse empty coverage, phantom/missing handlers, or unexplained deferral.

    ``board_checks`` is the parsed contract's executable-check declarations;
    their ``operation`` names the handler route. No I/O occurs in this function.
    """
    if not coverage_rows:
        raise ValueError("coverage file is empty")
    declared: dict[str, Mapping[str, object]] = {}
    for check in board_checks:
        check_id = check.get("check_id")
        operation = check.get("operation")
        if (
            not isinstance(check_id, str)
            or not isinstance(operation, str)
            or not operation.strip()
        ):
            raise ValueError(f"check {check_id!r} has no handler operation")
        if check_id in declared:
            raise ValueError(f"duplicate contract check {check_id}")
        declared[check_id] = check
    seen: set[str] = set()
    for raw in coverage_rows:
        row = ModelBoardCheckCoverage.model_validate(raw)
        if row.check not in declared:
            raise ValueError(
                f"coverage check {row.check} has no handler declared in contract"
            )
        if row.check in seen:
            raise ValueError(f"duplicate coverage check {row.check}")
        if row.surface_class != declared[row.check].get("surface_class"):
            raise ValueError(f"surface class mismatch for {row.check}")
        seen.add(row.check)
    missing = sorted(declared.keys() - seen)
    if missing:
        raise ValueError(f"contract handler checks have no coverage row: {missing}")


def main(argv: Sequence[str] | None = None) -> int:
    """Load real declarations and exit nonzero on any coverage refusal."""
    node = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=node / "contract.yaml")
    parser.add_argument(
        "--coverage", type=Path, default=node / "board_check_coverage.yaml"
    )
    args = parser.parse_args(argv)
    try:
        contract = yaml.safe_load(args.contract.read_text(encoding="utf-8"))
        coverage = yaml.safe_load(args.coverage.read_text(encoding="utf-8"))
        rows = coverage.get("coverage") if isinstance(coverage, dict) else None
        if not isinstance(contract, dict) or not isinstance(
            contract.get("board_checks"), list
        ):
            raise ValueError("contract must declare board_checks")
        if not isinstance(rows, list) or not rows:
            raise ValueError("coverage file is empty or is not a list")
        if any(not isinstance(row, dict) for row in rows + contract["board_checks"]):
            raise ValueError("coverage and board checks must be mappings")
        routes = {
            route["operation"]
            for route in contract["handler_routing"]["handlers"]
            if route.get("handler", {}).get("name")
            and route.get("handler", {}).get("module")
        }
        for check in contract["board_checks"]:
            if check.get("operation") not in routes:
                raise ValueError(f"check {check.get('check_id')} has no handler route")
        validate_coverage(contract["board_checks"], rows)
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        yaml.YAMLError,
    ) as exc:
        sys.stderr.write(f"board check coverage refused: {exc}\n")
        return 1
    sys.stdout.write(f"board check coverage valid: {len(rows)} checks\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
