# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Verification orchestrator for running all probes against a contract.

Coordinates subscription, publication, and projection probes, aggregates
results into a ModelContractVerificationReport with overall verdict using
FAIL > QUARANTINE > PASS precedence.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from omnibase_infra.enums.enum_check_severity import EnumCheckSeverity
from omnibase_infra.enums.enum_validation_verdict import EnumValidationVerdict
from omnibase_infra.verification.contract_parser import (
    ModelParsedContractForVerification,
    parse_contract_for_verification,
)
from omnibase_infra.verification.models import (
    ModelContractCheckResult,
    ModelContractVerificationReport,
)
from omnibase_infra.verification.probes.probe_publication import (
    WatermarkFn,
    check_publications,
)
from omnibase_infra.verification.probes.probe_subscription import (
    KafkaAdminFn,
    check_subscriptions,
)

logger = logging.getLogger(__name__)

# Type alias for DB query function used by projection probe
DbQueryFn = Callable[[str], list[dict[str, str]]]


@dataclass(frozen=True)
class VerificationConfig:
    """Configuration for the verification orchestrator.

    Attributes:
        db_query_fn: Callable(sql: str) -> list[dict] for database queries.
        kafka_admin_fn: Callable(group_id) -> set[str] for subscription checks.
        watermark_fn: Callable(topic) -> (low, high) for publication checks.
        runtime_target: Runtime endpoint being verified.
        probe_timeout: Per-probe timeout in seconds.
    """

    db_query_fn: DbQueryFn | None = None
    kafka_admin_fn: KafkaAdminFn | None = None
    watermark_fn: WatermarkFn | None = None
    runtime_target: str = "localhost:8085"
    probe_timeout: float = 5.0


def _compute_overall_verdict(
    checks: tuple[ModelContractCheckResult, ...],
) -> EnumValidationVerdict:
    """Compute aggregated verdict: FAIL > QUARANTINE > PASS.

    A REQUIRED check with FAIL verdict produces overall FAIL.
    A RECOMMENDED check with FAIL or any QUARANTINE produces overall QUARANTINE.
    Otherwise PASS.
    """
    has_required_fail = any(
        c.verdict == EnumValidationVerdict.FAIL
        and c.severity == EnumCheckSeverity.REQUIRED
        for c in checks
    )
    if has_required_fail:
        return EnumValidationVerdict.FAIL

    has_quarantine = any(c.verdict == EnumValidationVerdict.QUARANTINE for c in checks)
    has_recommended_fail = any(
        c.verdict == EnumValidationVerdict.FAIL
        and c.severity == EnumCheckSeverity.RECOMMENDED
        for c in checks
    )
    if has_quarantine or has_recommended_fail:
        return EnumValidationVerdict.QUARANTINE

    return EnumValidationVerdict.PASS


def _run_probe_safe(
    probe_name: str,
    probe_fn: Callable[[], list[ModelContractCheckResult]],
    contract_name: str,
    degraded_probes: list[str],
) -> list[ModelContractCheckResult]:
    """Run a probe with exception handling for graceful degradation.

    On exception, returns a QUARANTINE result and records the probe as degraded.
    """
    try:
        return probe_fn()
    # ONEX_EXCLUDE: blind_except - orchestrator must not crash on probe errors
    except Exception as exc:  # noqa: BLE001
        logger.warning("Probe %s failed for %s: %s", probe_name, contract_name, exc)
        degraded_probes.append(probe_name)
        from omnibase_infra.enums.enum_contract_check_type import (
            EnumContractCheckType,
        )

        check_type_map = {
            "subscription": EnumContractCheckType.SUBSCRIPTION,
            "publication": EnumContractCheckType.PUBLICATION,
            "projection": EnumContractCheckType.PROJECTION_STATE,
        }
        return [
            ModelContractCheckResult(
                check_type=check_type_map.get(
                    probe_name, EnumContractCheckType.PROJECTION_STATE
                ),
                severity=EnumCheckSeverity.REQUIRED,
                verdict=EnumValidationVerdict.QUARANTINE,
                evidence=f"Probe '{probe_name}' raised exception: {exc}",
                contract_name=contract_name,
                message=f"Probe '{probe_name}' degraded: {exc}",
            )
        ]


def _run_projection_probe(
    parsed: ModelParsedContractForVerification,
    config: VerificationConfig,
) -> list[ModelContractCheckResult]:
    """Run the projection state probe via DB query."""
    from omnibase_infra.verification.probes.probe_projection import (
        check_projection_state,
    )

    if config.db_query_fn is None:
        from omnibase_infra.enums.enum_contract_check_type import (
            EnumContractCheckType,
        )

        return [
            ModelContractCheckResult(
                check_type=EnumContractCheckType.PROJECTION_STATE,
                severity=EnumCheckSeverity.RECOMMENDED,
                verdict=EnumValidationVerdict.QUARANTINE,
                evidence="No db_query_fn provided; projection probe skipped.",
                contract_name=parsed.name,
                message="Projection probe skipped: no DB query function.",
            )
        ]

    rows = config.db_query_fn(
        "SELECT node_name, current_state FROM registration_projections"
    )
    return [check_projection_state(parsed.name, rows)]


def run_contract_verification(
    contract_path: Path,
    config: VerificationConfig | None = None,
) -> ModelContractVerificationReport:
    """Run all verification probes against a single contract.

    Orchestrates subscription, publication, and projection probes, then
    aggregates results into a single report.

    Args:
        contract_path: Path to the contract.yaml file.
        config: Verification configuration. Uses defaults if None.

    Returns:
        A ModelContractVerificationReport with all probe results.
    """
    start_ms = time.monotonic_ns() // 1_000_000
    cfg = config or VerificationConfig()

    parsed = parse_contract_for_verification(contract_path)

    degraded_probes: list[str] = []
    all_checks: list[ModelContractCheckResult] = []

    # Probe 1: Subscription
    sub_results = _run_probe_safe(
        "subscription",
        lambda: check_subscriptions(parsed, cfg.kafka_admin_fn),
        parsed.name,
        degraded_probes,
    )
    all_checks.extend(sub_results)

    # Probe 2: Publication
    pub_results = _run_probe_safe(
        "publication",
        lambda: check_publications(parsed, cfg.watermark_fn),
        parsed.name,
        degraded_probes,
    )
    all_checks.extend(pub_results)

    # Probe 3: Projection state
    proj_results = _run_probe_safe(
        "projection",
        lambda: _run_projection_probe(parsed, cfg),
        parsed.name,
        degraded_probes,
    )
    all_checks.extend(proj_results)

    checks = tuple(all_checks)
    overall = _compute_overall_verdict(checks)
    duration_ms = (time.monotonic_ns() // 1_000_000) - start_ms

    probe_mode = "degraded" if degraded_probes else "primary"

    return ModelContractVerificationReport(
        contract_name=parsed.name,
        node_type=parsed.node_type,
        checks=checks,
        overall_verdict=overall,
        probe_mode=probe_mode,
        degraded_probes=tuple(degraded_probes),
        runtime_target=cfg.runtime_target,
        duration_ms=duration_ms,
        report_fingerprint=ModelContractVerificationReport.compute_fingerprint(checks),
    )


__all__: list[str] = [
    "DbQueryFn",
    "VerificationConfig",
    "run_contract_verification",
]
