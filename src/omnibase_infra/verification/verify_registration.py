# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""End-to-end verification of the registration orchestrator contract.

Runs four checks against the live runtime:
1. REGISTRATION  - orchestrator node exists in registration_projections
2. SUBSCRIPTION  - consumer group is subscribed to declared topics
3. PUBLICATION   - core publish topics have non-zero watermark offsets
4. PROJECTION_STATE - at least one node in terminal FSM state
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

import yaml

from omnibase_infra.enums.enum_check_severity import EnumCheckSeverity
from omnibase_infra.enums.enum_contract_check_type import EnumContractCheckType
from omnibase_infra.enums.enum_validation_verdict import EnumValidationVerdict
from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.verification.contract_parser import (
    ModelParsedContractForVerification,
    parse_contract_for_verification,
)
from omnibase_infra.verification.models import (
    ModelContractCheckResult,
    ModelContractVerificationReport,
)
from omnibase_infra.verification.probes.probe_projection import (
    check_projection_state,
)
from omnibase_infra.verification.probes.probe_subscription import (
    check_subscriptions,
)

logger = logging.getLogger(__name__)

# Type aliases for the injectable dependencies
DbQueryFn = Callable[[str], list[dict[str, str]]]
KafkaAdminFn = Callable[[], set[str]]
WatermarkFn = Callable[[str], tuple[int, int]]

# Path to the registration orchestrator contract
_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "nodes"
    / "node_registration_orchestrator"
    / "contract.yaml"
)
_RUNTIME_CONFIG_PATH = (
    Path(__file__).resolve().parents[3]
    / "contracts"
    / "runtime"
    / "runtime_config.yaml"
)


def _check_registration(
    contract_name: str,
    db_query_fn: DbQueryFn,
) -> ModelContractCheckResult:
    """Check 1: Verify orchestrator node exists in registration_projections.

    Uses node_type='orchestrator' (not nonexistent node_name column).
    Validates schema via information_schema before querying data.

    Args:
        contract_name: Name of the contract being verified.
        db_query_fn: Callable that accepts a SQL string and returns list[dict].
    """
    try:
        schema_rows = db_query_fn(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'public' "
            "AND table_name = 'registration_projections' "
            "ORDER BY ordinal_position"
        )
    # ONEX_EXCLUDE: blind_except - boundary probe must not crash on infra errors
    except Exception as exc:  # noqa: BLE001
        return ModelContractCheckResult(
            check_type=EnumContractCheckType.REGISTRATION,
            severity=EnumCheckSeverity.REQUIRED,
            verdict=EnumValidationVerdict.FAIL,
            evidence=f"Schema introspection failed: {exc}",
            contract_name=contract_name,
            message="Registration check failed: could not introspect schema.",
        )

    actual_columns = {row.get("column_name", "") for row in schema_rows}
    required_columns = {"entity_id", "current_state", "node_type"}
    missing = required_columns - actual_columns
    if missing:
        return ModelContractCheckResult(
            check_type=EnumContractCheckType.REGISTRATION,
            severity=EnumCheckSeverity.REQUIRED,
            verdict=EnumValidationVerdict.QUARANTINE,
            evidence=(
                f"Schema mismatch: columns {sorted(missing)} not found in "
                f"registration_projections. Actual columns: {sorted(actual_columns)}"
            ),
            contract_name=contract_name,
            message="Registration check quarantined: column assumptions invalid.",
        )

    try:
        rows = db_query_fn(
            "SELECT entity_id, current_state, node_type "
            "FROM registration_projections "
            "WHERE node_type = 'orchestrator' LIMIT 1"
        )
    # ONEX_EXCLUDE: blind_except - boundary probe must not crash on infra errors
    except Exception as exc:  # noqa: BLE001
        return ModelContractCheckResult(
            check_type=EnumContractCheckType.REGISTRATION,
            severity=EnumCheckSeverity.REQUIRED,
            verdict=EnumValidationVerdict.FAIL,
            evidence=f"DB query failed: {exc}",
            contract_name=contract_name,
            message="Registration check failed: could not query database.",
        )

    if rows:
        state = rows[0].get("current_state", "unknown")
        if state.lower() != "idle":
            return ModelContractCheckResult(
                check_type=EnumContractCheckType.REGISTRATION,
                severity=EnumCheckSeverity.REQUIRED,
                verdict=EnumValidationVerdict.PASS,
                evidence=f"Orchestrator found with state={state}.",
                contract_name=contract_name,
                message=f"Registration check passed: orchestrator state={state}.",
            )
        return ModelContractCheckResult(
            check_type=EnumContractCheckType.REGISTRATION,
            severity=EnumCheckSeverity.REQUIRED,
            verdict=EnumValidationVerdict.FAIL,
            evidence=f"Orchestrator found but state is IDLE (state={state}).",
            contract_name=contract_name,
            message="Registration check failed: orchestrator stuck in IDLE.",
        )

    return ModelContractCheckResult(
        check_type=EnumContractCheckType.REGISTRATION,
        severity=EnumCheckSeverity.REQUIRED,
        verdict=EnumValidationVerdict.FAIL,
        evidence=(
            "No orchestrator row found in registration_projections "
            "(node_type='orchestrator')."
        ),
        contract_name=contract_name,
        message="Registration check failed: orchestrator not registered.",
    )


def _check_subscription(
    contract_name: str,
    subscribe_topics: tuple[str, ...],
    kafka_admin_fn: KafkaAdminFn,
    *,
    identity: ModelNodeIdentity | None = None,
) -> ModelContractCheckResult:
    """Check 2: Verify consumer group subscribes to declared topics.

    Args:
        contract_name: Name of the contract.
        subscribe_topics: Topics from the parsed contract.
        kafka_admin_fn: Callable that returns a set of subscribed topics for
            the consumer group, or raises on failure.
    """
    parsed = ModelParsedContractForVerification(
        name=contract_name,
        node_type="ORCHESTRATOR_GENERIC",
        subscribe_topics=subscribe_topics,
    )
    per_topic_results = check_subscriptions(
        parsed,
        kafka_admin_fn=lambda group_id: kafka_admin_fn(),
        identity=identity,
    )

    if all(r.verdict == EnumValidationVerdict.PASS for r in per_topic_results):
        return ModelContractCheckResult(
            check_type=EnumContractCheckType.SUBSCRIPTION,
            severity=EnumCheckSeverity.REQUIRED,
            verdict=EnumValidationVerdict.PASS,
            evidence=per_topic_results[0].evidence,
            contract_name=contract_name,
            message="Subscription check passed.",
        )

    fail_results = [
        r for r in per_topic_results if r.verdict == EnumValidationVerdict.FAIL
    ]
    if fail_results:
        missing_count = len(fail_results)
        total_count = len(subscribe_topics)
        return ModelContractCheckResult(
            check_type=EnumContractCheckType.SUBSCRIPTION,
            severity=EnumCheckSeverity.REQUIRED,
            verdict=EnumValidationVerdict.FAIL,
            evidence=(
                f"Missing {missing_count}/{total_count} subscriptions: "
                f"{' | '.join(r.evidence for r in fail_results)}"
            ),
            contract_name=contract_name,
            message=f"Subscription check failed: {missing_count} topics not subscribed.",
        )

    return ModelContractCheckResult(
        check_type=EnumContractCheckType.SUBSCRIPTION,
        severity=EnumCheckSeverity.REQUIRED,
        verdict=EnumValidationVerdict.QUARANTINE,
        evidence=" | ".join(r.evidence for r in per_topic_results),
        contract_name=contract_name,
        message="Subscription check quarantined: consumer group not authoritatively grounded.",
    )


def _load_registration_runtime_identity(
    runtime_config_path: Path | None = None,
) -> ModelNodeIdentity | None:
    """Load the runtime identity used by the registration plugin, if available."""
    path = runtime_config_path or _RUNTIME_CONFIG_PATH
    if not path.is_file():
        return None

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None

    if not isinstance(raw, dict):
        return None

    event_bus = raw.get("event_bus", {})
    if not isinstance(event_bus, dict):
        event_bus = {}

    environment = event_bus.get("environment")
    service = raw.get("name")
    version = raw.get("contract_version")
    if not all(isinstance(value, str) and value.strip() for value in (environment, service, version)):
        return None

    return ModelNodeIdentity(
        env=environment,
        service=service,
        node_name="registration-orchestrator",
        version=version,
    )


def _check_publication(
    contract_name: str,
    publish_topics: tuple[str, ...],
    watermark_fn: WatermarkFn,
) -> ModelContractCheckResult:
    """Check 3: Verify core publish topics have non-zero offsets.

    Args:
        contract_name: Name of the contract.
        publish_topics: Topics from the parsed contract.
        watermark_fn: Callable(topic) -> (low, high) watermark offsets.
    """
    if not publish_topics:
        return ModelContractCheckResult(
            check_type=EnumContractCheckType.PUBLICATION,
            severity=EnumCheckSeverity.REQUIRED,
            verdict=EnumValidationVerdict.PASS,
            evidence="No publish_topics declared; nothing to verify.",
            contract_name=contract_name,
            message="Publication check passed: no topics declared.",
        )

    active_topics: list[str] = []
    empty_topics: list[str] = []
    error_topics: list[str] = []

    for topic in publish_topics:
        try:
            _low, high = watermark_fn(topic)
            if high > 0:
                active_topics.append(topic)
            else:
                empty_topics.append(topic)
        # ONEX_EXCLUDE: blind_except - boundary probe must not crash on infra errors
        except Exception:  # noqa: BLE001
            error_topics.append(topic)

    total = len(publish_topics)
    # PASS if at least one core topic has data
    if active_topics:
        return ModelContractCheckResult(
            check_type=EnumContractCheckType.PUBLICATION,
            severity=EnumCheckSeverity.REQUIRED,
            verdict=EnumValidationVerdict.PASS,
            evidence=(
                f"{len(active_topics)}/{total} topics have non-zero offsets. "
                f"Empty: {len(empty_topics)}. Errors: {len(error_topics)}."
            ),
            contract_name=contract_name,
            message=f"Publication check passed: {len(active_topics)} active topics.",
        )

    return ModelContractCheckResult(
        check_type=EnumContractCheckType.PUBLICATION,
        severity=EnumCheckSeverity.REQUIRED,
        verdict=EnumValidationVerdict.FAIL,
        evidence=(
            f"0/{total} topics have non-zero offsets. "
            f"Empty: {len(empty_topics)}. Errors: {len(error_topics)}. "
            f"Error topics: {sorted(error_topics)}"
        ),
        contract_name=contract_name,
        message="Publication check failed: no topics have data.",
    )


def _check_projection_state_via_db(
    contract_name: str,
    db_query_fn: DbQueryFn,
) -> ModelContractCheckResult:
    """Check 4: Query registration_projections for any ACTIVE node.

    Validates schema assumptions via information_schema before querying data.
    Introspection failure = QUARANTINE (consistent with _check_registration).

    Args:
        contract_name: Name of the contract.
        db_query_fn: Callable that accepts SQL and returns list[dict].
    """
    try:
        schema_rows = db_query_fn(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'public' "
            "AND table_name = 'registration_projections' "
            "ORDER BY ordinal_position"
        )
    # ONEX_EXCLUDE: blind_except - boundary probe must not crash on infra errors
    except Exception as exc:  # noqa: BLE001
        return ModelContractCheckResult(
            check_type=EnumContractCheckType.PROJECTION_STATE,
            severity=EnumCheckSeverity.REQUIRED,
            verdict=EnumValidationVerdict.QUARANTINE,
            evidence=f"Schema introspection failed: {exc}",
            contract_name=contract_name,
            message="Projection state check quarantined: could not introspect schema.",
        )

    if schema_rows:
        actual_columns = {row.get("column_name", "") for row in schema_rows}
        required = {"current_state"}
        missing = required - actual_columns
        if missing:
            return ModelContractCheckResult(
                check_type=EnumContractCheckType.PROJECTION_STATE,
                severity=EnumCheckSeverity.REQUIRED,
                verdict=EnumValidationVerdict.QUARANTINE,
                evidence=(
                    f"Schema mismatch: columns {sorted(missing)} not found. "
                    f"Actual: {sorted(actual_columns)}"
                ),
                contract_name=contract_name,
                message="Projection state check quarantined: schema assumptions invalid.",
            )

    try:
        rows = db_query_fn(
            "SELECT entity_id, current_state, node_type FROM registration_projections"
        )
    # ONEX_EXCLUDE: blind_except - boundary probe must not crash on infra errors
    except Exception as exc:  # noqa: BLE001
        return ModelContractCheckResult(
            check_type=EnumContractCheckType.PROJECTION_STATE,
            severity=EnumCheckSeverity.REQUIRED,
            verdict=EnumValidationVerdict.FAIL,
            evidence=f"DB query failed: {exc}",
            contract_name=contract_name,
            message="Projection state check failed: could not query database.",
        )

    return check_projection_state(contract_name, rows)


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


def verify_registration_contract(
    db_query_fn: DbQueryFn,
    kafka_admin_fn: KafkaAdminFn,
    watermark_fn: WatermarkFn,
    contract_path: Path | None = None,
    runtime_config_path: Path | None = None,
) -> ModelContractVerificationReport:
    """Run all four verification checks against the registration orchestrator.

    Args:
        db_query_fn: Callable(sql: str) -> list[dict] for database queries.
        kafka_admin_fn: Callable() -> set[str] returning subscribed topics.
        watermark_fn: Callable(topic: str) -> tuple[int, int] returning
            (low, high) watermark offsets for a topic.
        contract_path: Optional override for the contract.yaml path.
        runtime_config_path: Optional override for runtime_config.yaml used to
            derive the exact registration runtime identity.

    Returns:
        A ModelContractVerificationReport with all check results.
    """
    start_ms = time.monotonic_ns() // 1_000_000
    path = contract_path or _CONTRACT_PATH
    parsed = parse_contract_for_verification(path)
    runtime_identity = _load_registration_runtime_identity(runtime_config_path)

    check_registration = _check_registration(parsed.name, db_query_fn)
    check_subscription = _check_subscription(
        parsed.name,
        parsed.subscribe_topics,
        kafka_admin_fn,
        identity=runtime_identity,
    )
    check_publication = _check_publication(
        parsed.name, parsed.publish_topics, watermark_fn
    )
    check_projection = _check_projection_state_via_db(parsed.name, db_query_fn)

    checks = (
        check_registration,
        check_subscription,
        check_publication,
        check_projection,
    )

    overall = _compute_overall_verdict(checks)
    duration_ms = (time.monotonic_ns() // 1_000_000) - start_ms

    return ModelContractVerificationReport(
        contract_name=parsed.name,
        node_type=parsed.node_type,
        checks=checks,
        overall_verdict=overall,
        duration_ms=duration_ms,
        report_fingerprint=ModelContractVerificationReport.compute_fingerprint(checks),
    )


__all__: list[str] = [
    "DbQueryFn",
    "KafkaAdminFn",
    "WatermarkFn",
    "verify_registration_contract",
]
