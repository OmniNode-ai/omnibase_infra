# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Cross-contract chain verification for the registration trio.

Static analysis that verifies the three registration contracts
(orchestrator -> reducer -> storage_effect) work together by checking:

1. Topic connectivity: orchestrator publish topics match reducer subscribe topics
2. Intent routing: event types flow correctly between contracts
3. Projection consistency: published events align with consumed events

This is pure YAML parsing -- no Docker or runtime required. Safe to run as
a unit test.
"""

from __future__ import annotations

import time
from pathlib import Path

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


def _check_topic_connectivity(
    contracts: list[ModelParsedContractForVerification],
) -> ModelContractCheckResult:
    """Check that publish topics of upstream contracts connect to subscribe topics of downstream.

    For every pair (A, B) in the ordered contract list, at least one of A's
    publish_topics should appear in B's subscribe_topics.

    Args:
        contracts: Ordered list of parsed contracts (upstream first).

    Returns:
        ModelContractCheckResult with PASS if all adjacent pairs are connected.
    """
    chain_name = " -> ".join(c.name for c in contracts)
    if len(contracts) < 2:
        return ModelContractCheckResult(
            check_type=EnumContractCheckType.SUBSCRIPTION,
            severity=EnumCheckSeverity.REQUIRED,
            verdict=EnumValidationVerdict.PASS,
            evidence="Fewer than 2 contracts; no connectivity to check.",
            contract_name=chain_name,
            message="Topic connectivity: trivially passes (single contract).",
        )

    gaps: list[str] = []
    connected_pairs: list[str] = []

    for i in range(len(contracts) - 1):
        upstream = contracts[i]
        downstream = contracts[i + 1]
        shared = set(upstream.publish_topics) & set(downstream.subscribe_topics)
        if shared:
            connected_pairs.append(
                f"{upstream.name} -> {downstream.name} via {sorted(shared)}"
            )
        else:
            gaps.append(
                f"{upstream.name} publishes {sorted(upstream.publish_topics)} "
                f"but {downstream.name} subscribes to "
                f"{sorted(downstream.subscribe_topics)}"
            )

    if not gaps:
        return ModelContractCheckResult(
            check_type=EnumContractCheckType.SUBSCRIPTION,
            severity=EnumCheckSeverity.REQUIRED,
            verdict=EnumValidationVerdict.PASS,
            evidence=(
                f"All {len(connected_pairs)} adjacent pairs connected: "
                f"{'; '.join(connected_pairs)}"
            ),
            contract_name=chain_name,
            message=f"Topic connectivity passed: {len(connected_pairs)} links verified.",
        )

    return ModelContractCheckResult(
        check_type=EnumContractCheckType.SUBSCRIPTION,
        severity=EnumCheckSeverity.REQUIRED,
        verdict=EnumValidationVerdict.FAIL,
        evidence=f"Topic gaps: {'; '.join(gaps)}",
        contract_name=chain_name,
        message=f"Topic connectivity failed: {len(gaps)} gap(s) found.",
    )


def _check_event_type_routing(
    contracts: list[ModelParsedContractForVerification],
) -> ModelContractCheckResult:
    """Check that published event types from upstream match consumed event types downstream.

    Args:
        contracts: Ordered list of parsed contracts.

    Returns:
        ModelContractCheckResult with PASS if event types flow correctly.
    """
    chain_name = " -> ".join(c.name for c in contracts)
    if len(contracts) < 2:
        return ModelContractCheckResult(
            check_type=EnumContractCheckType.HANDLER_EXECUTION,
            severity=EnumCheckSeverity.RECOMMENDED,
            verdict=EnumValidationVerdict.PASS,
            evidence="Fewer than 2 contracts; no event routing to check.",
            contract_name=chain_name,
            message="Event type routing: trivially passes (single contract).",
        )

    gaps: list[str] = []
    connected_pairs: list[str] = []

    for i in range(len(contracts) - 1):
        upstream = contracts[i]
        downstream = contracts[i + 1]

        # If neither declares event types, skip this pair (not all contracts use them)
        if not upstream.published_events and not downstream.consumed_events:
            continue

        shared = set(upstream.published_events) & set(downstream.consumed_events)
        if shared:
            connected_pairs.append(
                f"{upstream.name} -> {downstream.name} via events {sorted(shared)}"
            )
        elif upstream.published_events and downstream.consumed_events:
            gaps.append(
                f"{upstream.name} publishes events {sorted(upstream.published_events)} "
                f"but {downstream.name} consumes "
                f"{sorted(downstream.consumed_events)}"
            )

    if gaps:
        return ModelContractCheckResult(
            check_type=EnumContractCheckType.HANDLER_EXECUTION,
            severity=EnumCheckSeverity.RECOMMENDED,
            verdict=EnumValidationVerdict.FAIL,
            evidence=f"Event routing gaps: {'; '.join(gaps)}",
            contract_name=chain_name,
            message=f"Event type routing failed: {len(gaps)} gap(s).",
        )

    evidence = (
        f"{len(connected_pairs)} event routing link(s) verified."
        if connected_pairs
        else "No event type declarations to verify (contracts may use topic-only routing)."
    )
    return ModelContractCheckResult(
        check_type=EnumContractCheckType.HANDLER_EXECUTION,
        severity=EnumCheckSeverity.RECOMMENDED,
        verdict=EnumValidationVerdict.PASS,
        evidence=evidence,
        contract_name=chain_name,
        message="Event type routing passed.",
    )


def _check_projection_consistency(
    contracts: list[ModelParsedContractForVerification],
) -> ModelContractCheckResult:
    """Check that downstream contracts consume all event types the chain publishes.

    Looks at the full chain: the first contract's consumed_events should be
    a subset of what's available externally, and the last contract's
    published_events represent the chain's output.

    Args:
        contracts: Ordered list of parsed contracts.

    Returns:
        ModelContractCheckResult with PASS if chain is internally consistent.
    """
    chain_name = " -> ".join(c.name for c in contracts)

    # Collect all published and consumed event types across the chain
    all_published: set[str] = set()
    all_consumed: set[str] = set()
    for contract in contracts:
        all_published.update(contract.published_events)
        all_consumed.update(contract.consumed_events)

    # Internal events: consumed by chain members that are also published by chain members
    internal_events = all_published & all_consumed
    # Dangling consumed: events consumed but never published by any chain member
    # (these must come from outside the chain -- valid, just informational)
    external_inputs = all_consumed - all_published
    # Chain outputs: events published but not consumed within the chain
    chain_outputs = all_published - all_consumed

    evidence_parts: list[str] = []
    if internal_events:
        evidence_parts.append(f"Internal flow: {sorted(internal_events)}")
    if external_inputs:
        evidence_parts.append(f"External inputs: {sorted(external_inputs)}")
    if chain_outputs:
        evidence_parts.append(f"Chain outputs: {sorted(chain_outputs)}")

    if not all_published and not all_consumed:
        return ModelContractCheckResult(
            check_type=EnumContractCheckType.PROJECTION_STATE,
            severity=EnumCheckSeverity.INFORMATIONAL,
            verdict=EnumValidationVerdict.PASS,
            evidence="No event types declared across chain; topic-only routing assumed.",
            contract_name=chain_name,
            message="Projection consistency: no event types to verify.",
        )

    return ModelContractCheckResult(
        check_type=EnumContractCheckType.PROJECTION_STATE,
        severity=EnumCheckSeverity.INFORMATIONAL,
        verdict=EnumValidationVerdict.PASS,
        evidence="; ".join(evidence_parts) if evidence_parts else "No events declared.",
        contract_name=chain_name,
        message=(
            f"Projection consistency passed: {len(internal_events)} internal, "
            f"{len(external_inputs)} external, {len(chain_outputs)} output events."
        ),
    )


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


def verify_contract_chain(
    contract_paths: list[Path],
) -> ModelContractVerificationReport:
    """Verify a chain of contracts work together via static YAML analysis.

    Parses each contract and runs three cross-contract checks:
    1. Topic connectivity between adjacent contracts
    2. Event type routing between adjacent contracts
    3. Projection consistency across the full chain

    Args:
        contract_paths: Ordered list of contract.yaml paths (upstream first).
            Typical usage: [orchestrator, reducer, storage_effect].

    Returns:
        ModelContractVerificationReport with cross-contract check results.
    """
    start_ms = time.monotonic_ns() // 1_000_000

    contracts = [parse_contract_for_verification(p) for p in contract_paths]
    chain_name = " -> ".join(c.name for c in contracts)

    check_topics = _check_topic_connectivity(contracts)
    check_events = _check_event_type_routing(contracts)
    check_projection = _check_projection_consistency(contracts)

    checks = (check_topics, check_events, check_projection)
    overall = _compute_overall_verdict(checks)
    duration_ms = (time.monotonic_ns() // 1_000_000) - start_ms

    return ModelContractVerificationReport(
        contract_name=chain_name,
        node_type="CHAIN_VERIFICATION",
        checks=checks,
        overall_verdict=overall,
        probe_mode="static",
        duration_ms=duration_ms,
        report_fingerprint=ModelContractVerificationReport.compute_fingerprint(checks),
    )


__all__: list[str] = ["verify_contract_chain"]
