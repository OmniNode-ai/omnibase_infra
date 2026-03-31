# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""CLI entry point for runtime contract compliance verification.

Usage:
    python -m omnibase_infra.verification --contract-path <path>
    python -m omnibase_infra.verification --all
    python -m omnibase_infra.verification --registration-only

Exit codes:
    0 = PASS
    1 = FAIL
    2 = QUARANTINE
"""

from __future__ import annotations

import argparse
import importlib.resources
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

from omnibase_infra.enums.enum_validation_verdict import EnumValidationVerdict
from omnibase_infra.verification.models import ModelContractVerificationReport
from omnibase_infra.verification.orchestrator import (
    VerificationConfig,
    run_contract_verification,
)

# Exit code mapping
_EXIT_CODES: dict[EnumValidationVerdict, int] = {
    EnumValidationVerdict.PASS: 0,
    EnumValidationVerdict.FAIL: 1,
    EnumValidationVerdict.QUARANTINE: 2,
}

# Registration contract directory names
_REGISTRATION_NODE_NAMES: tuple[str, ...] = (
    "node_registration_orchestrator",
    "node_registration_reducer",
    "node_registration_storage_effect",
)


def _default_contracts_dir() -> Path:
    """Resolve the default contracts directory via importlib.resources."""
    try:
        nodes_pkg = importlib.resources.files("omnibase_infra") / "nodes"
        return Path(str(nodes_pkg))
    except (ModuleNotFoundError, TypeError):
        # Fallback to relative path from this file
        return Path(__file__).resolve().parents[1] / "nodes"


def _find_all_contracts(contracts_dir: Path) -> list[Path]:
    """Walk a directory tree and find all contract.yaml files."""
    contracts: list[Path] = []
    if not contracts_dir.is_dir():
        return contracts
    for contract_path in sorted(contracts_dir.rglob("contract.yaml")):
        contracts.append(contract_path)
    return contracts


def _find_registration_contracts(contracts_dir: Path) -> list[Path]:
    """Find the 3 registration contracts."""
    contracts: list[Path] = []
    for node_name in _REGISTRATION_NODE_NAMES:
        candidate = contracts_dir / node_name / "contract.yaml"
        if candidate.is_file():
            contracts.append(candidate)
    return contracts


def _aggregate_verdict(
    reports: list[ModelContractVerificationReport],
) -> EnumValidationVerdict:
    """Compute overall verdict across multiple reports: FAIL > QUARANTINE > PASS."""
    if any(r.overall_verdict == EnumValidationVerdict.FAIL for r in reports):
        return EnumValidationVerdict.FAIL
    if any(r.overall_verdict == EnumValidationVerdict.QUARANTINE for r in reports):
        return EnumValidationVerdict.QUARANTINE
    return EnumValidationVerdict.PASS


def _run_registration_only(
    contracts_dir: Path,
    output_path: Path | None,
) -> int:
    """Run verify_registration_contract() for the 3 registration contracts."""
    from omnibase_infra.verification.verify_registration import (
        verify_registration_contract,
    )

    contracts = _find_registration_contracts(contracts_dir)
    if not contracts:
        logger.error("No registration contracts found in %s", contracts_dir)
        return 1

    # For registration-only, we use placeholder fns that produce QUARANTINE
    # since the real infra may not be available. The verify_registration_contract
    # function has its own dependency injection.
    def _noop_db(sql: str) -> list[dict[str, str]]:
        return []

    def _noop_kafka() -> set[str]:
        return set()

    def _noop_watermark(topic: str) -> tuple[int, int]:
        return (0, 0)

    reports: list[ModelContractVerificationReport] = []
    for contract_path in contracts:
        report = verify_registration_contract(
            db_query_fn=_noop_db,
            kafka_admin_fn=_noop_kafka,
            watermark_fn=_noop_watermark,
            contract_path=contract_path,
        )
        reports.append(report)

    return _output_reports(reports, output_path)


def _output_reports(
    reports: list[ModelContractVerificationReport],
    output_path: Path | None,
) -> int:
    """Serialize reports to JSON and write to stdout or file. Returns exit code."""
    output_data = [r.model_dump(mode="json") for r in reports]
    json_str = json.dumps(output_data, indent=2, default=str)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_str)
        logger.info("Report written to %s", output_path)
    else:
        sys.stdout.write(json_str + "\n")

    overall = _aggregate_verdict(reports)
    return _EXIT_CODES.get(overall, 1)


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the verification CLI."""
    parser = argparse.ArgumentParser(
        prog="omnibase_infra.verification",
        description="Runtime contract compliance verification for ONEX nodes.",
    )
    parser.add_argument(
        "--contract-path",
        type=Path,
        help="Path to a single contract.yaml to verify.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="verify_all",
        help="Walk all contracts in --contracts-dir and verify each.",
    )
    parser.add_argument(
        "--registration-only",
        action="store_true",
        help="Run verification for the 3 registration contracts only.",
    )
    parser.add_argument(
        "--contracts-dir",
        type=Path,
        default=None,
        help="Directory containing node contracts. Defaults to installed package nodes/.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=True,
        dest="json_output",
        help="Machine-readable JSON output (default).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Write report to file instead of stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns exit code: 0=PASS, 1=FAIL, 2=QUARANTINE."""
    parser = build_parser()
    args = parser.parse_args(argv)

    contracts_dir = args.contracts_dir or _default_contracts_dir()

    # Validate mutually exclusive flags
    mode_count = sum(
        [
            args.contract_path is not None,
            args.verify_all,
            args.registration_only,
        ]
    )
    if mode_count == 0:
        parser.error(
            "One of --contract-path, --all, or --registration-only is required."
        )
    if mode_count > 1:
        parser.error(
            "Only one of --contract-path, --all, or --registration-only may be specified."
        )

    # Registration-only mode
    if args.registration_only:
        return _run_registration_only(contracts_dir, args.output_path)

    # Single contract mode
    if args.contract_path:
        if not args.contract_path.is_file():
            logger.error("Contract not found: %s", args.contract_path)
            return 1
        config = VerificationConfig()
        report = run_contract_verification(args.contract_path, config)
        return _output_reports([report], args.output_path)

    # All contracts mode
    if args.verify_all:
        contract_paths = _find_all_contracts(contracts_dir)
        if not contract_paths:
            logger.error("No contracts found in %s", contracts_dir)
            return 1

        config = VerificationConfig()
        reports: list[ModelContractVerificationReport] = []
        for path in contract_paths:
            report = run_contract_verification(path, config)
            reports.append(report)

        return _output_reports(reports, args.output_path)

    return 1  # unreachable


if __name__ == "__main__":
    sys.exit(main())


__all__: list[str] = [
    "build_parser",
    "main",
]
