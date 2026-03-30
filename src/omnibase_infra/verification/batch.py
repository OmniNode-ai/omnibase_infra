# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Multi-contract batch verification.

Walks all contract.yaml files under a contracts directory and runs the
appropriate verification probes per contract class (node_type). Results
are aggregated into per-contract reports with an overall summary.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from omnibase_infra.enums.enum_check_severity import EnumCheckSeverity
from omnibase_infra.enums.enum_contract_check_type import EnumContractCheckType
from omnibase_infra.enums.enum_validation_verdict import EnumValidationVerdict
from omnibase_infra.verification.contract_parser import (
    ModelParsedContractForVerification,
    parse_contract_for_verification,
)
from omnibase_infra.verification.models import (
    ModelContractCheckResult,
    ModelContractVerificationReport,
)

logger = logging.getLogger(__name__)

# ONEX_EXCLUDE: any_type - DB rows are untyped dicts from SQL queries
DbQueryFn = Callable[[str], list[dict[str, Any]]]
KafkaAdminFn = Callable[[str], set[str]]
WatermarkFn = Callable[[str], tuple[int, int]]


class BatchVerificationConfig:
    """Configuration for batch verification runs.

    Attributes:
        db_query_fn: Callable(sql) -> list[dict] for database queries.
        kafka_admin_fn: Callable(group_id) -> set[str] of subscribed topics.
        watermark_fn: Callable(topic) -> (low, high) watermarks.
        runtime_target: Runtime endpoint label for reports.
    """

    def __init__(
        self,
        db_query_fn: DbQueryFn | None = None,
        kafka_admin_fn: KafkaAdminFn | None = None,
        watermark_fn: WatermarkFn | None = None,
        runtime_target: str = "localhost:8085",
    ) -> None:
        self.db_query_fn = db_query_fn
        self.kafka_admin_fn = kafka_admin_fn
        self.watermark_fn = watermark_fn
        self.runtime_target = runtime_target


# -- Node-type to probe requirement matrix --
# Values: REQUIRED, RECOMMENDED, SKIP
# Probe keys map to EnumContractCheckType members.

_PROBE_MATRIX: dict[str, dict[str, str]] = {
    "ORCHESTRATOR_GENERIC": {
        "REGISTRATION": "REQUIRED",
        "SUBSCRIPTION": "REQUIRED",
        "PUBLICATION": "REQUIRED",
        "HANDLER_EXECUTION": "OPTIONAL",
        "PROJECTION_STATE": "REQUIRED",
    },
    "REDUCER_GENERIC": {
        "REGISTRATION": "SKIP",
        "SUBSCRIPTION": "REQUIRED",
        "PUBLICATION": "REQUIRED",
        "HANDLER_EXECUTION": "OPTIONAL",
        "PROJECTION_STATE": "SKIP",
    },
    "EFFECT_GENERIC": {
        "REGISTRATION": "SKIP",
        "SUBSCRIPTION": "REQUIRED",
        "PUBLICATION": "RECOMMENDED",
        "HANDLER_EXECUTION": "OPTIONAL",
        "PROJECTION_STATE": "REQUIRED",
    },
    "COMPUTE_GENERIC": {
        "REGISTRATION": "SKIP",
        "SUBSCRIPTION": "SKIP",
        "PUBLICATION": "SKIP",
        "HANDLER_EXECUTION": "SKIP",
        "PROJECTION_STATE": "SKIP",
    },
}


def _get_probe_requirement(node_type: str, probe_key: str) -> str:
    """Look up the probe requirement for a node type.

    Falls back to COMPUTE_GENERIC (all SKIP) for unknown types.
    """
    matrix = _PROBE_MATRIX.get(node_type, _PROBE_MATRIX["COMPUTE_GENERIC"])
    return matrix.get(probe_key, "SKIP")


def _compute_overall_verdict(
    checks: tuple[ModelContractCheckResult, ...],
) -> EnumValidationVerdict:
    """Compute aggregated verdict: FAIL > QUARANTINE > PASS."""
    has_fail = any(
        c.verdict == EnumValidationVerdict.FAIL
        and c.severity == EnumCheckSeverity.REQUIRED
        for c in checks
    )
    if has_fail:
        return EnumValidationVerdict.FAIL

    has_quarantine = any(c.verdict == EnumValidationVerdict.QUARANTINE for c in checks)
    if has_quarantine:
        return EnumValidationVerdict.QUARANTINE

    return EnumValidationVerdict.PASS


def _run_registration_probe(
    parsed: ModelParsedContractForVerification,
    config: BatchVerificationConfig,
) -> list[ModelContractCheckResult]:
    """Run registration probe if db_query_fn is available."""
    if config.db_query_fn is None:
        return [
            ModelContractCheckResult(
                check_type=EnumContractCheckType.REGISTRATION,
                severity=EnumCheckSeverity.REQUIRED,
                verdict=EnumValidationVerdict.QUARANTINE,
                evidence="No db_query_fn configured; cannot verify registration.",
                contract_name=parsed.name,
                message="Registration check quarantined: no DB access.",
            )
        ]

    # Contract names are internal identifiers from YAML files (not user input).
    _base = "SELECT node_name, current_state FROM registration_projections WHERE node_name = "
    sql = _base + "'" + parsed.name + "'"
    try:
        rows = config.db_query_fn(sql)
    # ONEX_EXCLUDE: blind_except - boundary probe must not crash on infra errors
    except Exception as exc:  # noqa: BLE001
        logger.warning("Registration probe DB error for %s: %s", parsed.name, exc)
        return [
            ModelContractCheckResult(
                check_type=EnumContractCheckType.REGISTRATION,
                severity=EnumCheckSeverity.REQUIRED,
                verdict=EnumValidationVerdict.QUARANTINE,
                evidence=f"DB query failed: {exc}",
                contract_name=parsed.name,
                message="Registration check quarantined: DB error.",
            )
        ]

    if rows:
        return [
            ModelContractCheckResult(
                check_type=EnumContractCheckType.REGISTRATION,
                severity=EnumCheckSeverity.REQUIRED,
                verdict=EnumValidationVerdict.PASS,
                evidence=f"Found {len(rows)} registration row(s) for '{parsed.name}'.",
                contract_name=parsed.name,
                message=f"Registration check passed for '{parsed.name}'.",
            )
        ]
    return [
        ModelContractCheckResult(
            check_type=EnumContractCheckType.REGISTRATION,
            severity=EnumCheckSeverity.REQUIRED,
            verdict=EnumValidationVerdict.FAIL,
            evidence=f"No registration rows found for '{parsed.name}'.",
            contract_name=parsed.name,
            message=f"Registration check failed for '{parsed.name}'.",
        )
    ]


def _run_subscription_probe(
    parsed: ModelParsedContractForVerification,
    config: BatchVerificationConfig,
) -> list[ModelContractCheckResult]:
    """Run subscription probe via the subscription probe module."""
    from omnibase_infra.verification.probes.probe_subscription import (
        check_subscriptions,
    )

    return check_subscriptions(parsed, kafka_admin_fn=config.kafka_admin_fn)


def _run_publication_probe(
    parsed: ModelParsedContractForVerification,
    config: BatchVerificationConfig,
) -> list[ModelContractCheckResult]:
    """Run publication probe via the publication probe module."""
    from omnibase_infra.verification.probes.probe_publication import (
        check_publications,
    )

    return check_publications(parsed, watermark_fn=config.watermark_fn)


def _run_projection_probe(
    parsed: ModelParsedContractForVerification,
    config: BatchVerificationConfig,
) -> list[ModelContractCheckResult]:
    """Run projection state probe if db_query_fn is available."""
    from omnibase_infra.verification.probes.probe_projection import (
        check_projection_state,
    )

    if config.db_query_fn is None:
        return [
            ModelContractCheckResult(
                check_type=EnumContractCheckType.PROJECTION_STATE,
                severity=EnumCheckSeverity.REQUIRED,
                verdict=EnumValidationVerdict.QUARANTINE,
                evidence="No db_query_fn configured; cannot verify projection state.",
                contract_name=parsed.name,
                message="Projection check quarantined: no DB access.",
            )
        ]

    sql = (
        "SELECT node_name, current_state FROM registration_projections "
        "WHERE current_state IS NOT NULL"
    )
    try:
        rows = config.db_query_fn(sql)
    # ONEX_EXCLUDE: blind_except - boundary probe must not crash on infra errors
    except Exception as exc:  # noqa: BLE001
        logger.warning("Projection probe DB error for %s: %s", parsed.name, exc)
        return [
            ModelContractCheckResult(
                check_type=EnumContractCheckType.PROJECTION_STATE,
                severity=EnumCheckSeverity.REQUIRED,
                verdict=EnumValidationVerdict.QUARANTINE,
                evidence=f"DB query failed: {exc}",
                contract_name=parsed.name,
                message="Projection check quarantined: DB error.",
            )
        ]

    return [check_projection_state(parsed.name, rows)]


# Map probe keys to runner functions.
_PROBE_RUNNERS: dict[
    str,
    Callable[
        [ModelParsedContractForVerification, BatchVerificationConfig],
        list[ModelContractCheckResult],
    ],
] = {
    "REGISTRATION": _run_registration_probe,
    "SUBSCRIPTION": _run_subscription_probe,
    "PUBLICATION": _run_publication_probe,
    "PROJECTION_STATE": _run_projection_probe,
}


def _verify_single_contract(
    parsed: ModelParsedContractForVerification,
    config: BatchVerificationConfig,
) -> ModelContractVerificationReport:
    """Run all applicable probes for a single contract and return a report."""
    start_ms = time.monotonic_ns() // 1_000_000
    all_checks: list[ModelContractCheckResult] = []
    degraded: list[str] = []

    for probe_key, runner in _PROBE_RUNNERS.items():
        requirement = _get_probe_requirement(parsed.node_type, probe_key)

        if requirement == "SKIP":
            continue

        # OPTIONAL probes are skipped in batch mode (handler_execution needs
        # publish/consume fns we don't inject here).
        if requirement == "OPTIONAL":
            continue

        try:
            results = runner(parsed, config)
            all_checks.extend(results)
        # ONEX_EXCLUDE: blind_except - one probe failing must not stop the batch
        except Exception as exc:  # noqa: BLE001
            logger.warning("Probe %s failed for %s: %s", probe_key, parsed.name, exc)
            degraded.append(probe_key)
            severity = (
                EnumCheckSeverity.REQUIRED
                if requirement == "REQUIRED"
                else EnumCheckSeverity.RECOMMENDED
            )
            all_checks.append(
                ModelContractCheckResult(
                    check_type=EnumContractCheckType(probe_key.lower()),
                    severity=severity,
                    verdict=EnumValidationVerdict.QUARANTINE,
                    evidence=f"Probe {probe_key} threw: {exc}",
                    contract_name=parsed.name,
                    message=f"{probe_key} probe error: quarantined.",
                )
            )

    checks_tuple = tuple(all_checks)
    duration_ms = (time.monotonic_ns() // 1_000_000) - start_ms

    return ModelContractVerificationReport(
        contract_name=parsed.name,
        node_type=parsed.node_type,
        checks=checks_tuple,
        overall_verdict=_compute_overall_verdict(checks_tuple),
        probe_mode="fallback" if degraded else "primary",
        degraded_probes=tuple(degraded),
        runtime_target=config.runtime_target,
        duration_ms=duration_ms,
        report_fingerprint=ModelContractVerificationReport.compute_fingerprint(
            checks_tuple
        ),
    )


def _discover_contracts(contracts_dir: Path) -> list[Path]:
    """Walk contracts_dir/*/contract.yaml and return sorted paths."""
    if not contracts_dir.is_dir():
        logger.warning("Contracts directory does not exist: %s", contracts_dir)
        return []

    paths = sorted(contracts_dir.glob("*/contract.yaml"))
    logger.info("Discovered %d contract(s) in %s", len(paths), contracts_dir)
    return paths


def run_batch_verification(
    contracts_dir: Path,
    config: BatchVerificationConfig,
) -> list[ModelContractVerificationReport]:
    """Walk all contracts and run appropriate probes per contract class.

    Args:
        contracts_dir: Directory containing node_*/contract.yaml files.
        config: BatchVerificationConfig with injectable dependencies.

    Returns:
        List of ModelContractVerificationReport, one per contract.
        Contracts that fail to parse are logged and skipped.
    """
    contract_paths = _discover_contracts(contracts_dir)
    reports: list[ModelContractVerificationReport] = []

    for path in contract_paths:
        try:
            parsed = parse_contract_for_verification(path)
        # ONEX_EXCLUDE: blind_except - bad YAML must not stop the entire batch
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to parse %s: %s", path, exc)
            continue

        report = _verify_single_contract(parsed, config)
        reports.append(report)
        logger.info(
            "Verified %s (%s): %s",
            parsed.name,
            parsed.node_type,
            report.overall_verdict.value,
        )

    return reports


__all__: list[str] = [
    "BatchVerificationConfig",
    "run_batch_verification",
]
