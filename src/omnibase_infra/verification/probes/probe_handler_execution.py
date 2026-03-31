# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Handler execution verification probe.

Verifies that contract-declared handlers actually execute by publishing a
synthetic test event and checking for expected output on the declared publish
topics. Only runs when --probe-handlers flag is set.

Usage:
    The probe requires injectable publish and consume callables so it can be
    tested with mocks (unit) or real Kafka (integration). Synthetic events
    use a correlation_id prefix of "verify-<timestamp>" to distinguish them
    from real traffic.

    Timeout is configurable (default 10s). The probe returns PASS if an
    expected event is received within the timeout, FAIL otherwise.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from omnibase_infra.enums.enum_check_severity import EnumCheckSeverity
from omnibase_infra.enums.enum_contract_check_type import EnumContractCheckType
from omnibase_infra.enums.enum_validation_verdict import EnumValidationVerdict
from omnibase_infra.verification.contract_parser import (
    ModelParsedContractForVerification,
)
from omnibase_infra.verification.models.model_contract_check_result import (
    ModelContractCheckResult,
)

# Type aliases for injectable I/O functions.
# publish_fn(topic, payload_dict) -> None
PublishFn = Callable[[str, dict[str, str]], None]
# consume_fn(topic, timeout_seconds) -> list[dict[str, str]]
ConsumeFn = Callable[[str, float], list[dict[str, str]]]

# Correlation ID prefix for synthetic verification events.
VERIFY_CORRELATION_PREFIX = "verify-"


def _build_synthetic_event(contract_name: str) -> dict[str, str]:
    """Build a synthetic introspection event for handler verification.

    Args:
        contract_name: The contract name to embed in the event payload.

    Returns:
        A dict payload with a verify-prefixed correlation_id.
    """
    correlation_id = f"{VERIFY_CORRELATION_PREFIX}{int(time.time() * 1000)}"
    return {
        "correlation_id": correlation_id,
        "event_type": "ModelNodeIntrospectionEvent",
        "node_name": contract_name,
        "action": "verify_handler_execution",
    }


def check_handler_execution(
    parsed_contract: ModelParsedContractForVerification,
    publish_fn: PublishFn,
    consume_fn: ConsumeFn,
    timeout_seconds: float = 10.0,
) -> ModelContractCheckResult:
    """Verify contract-declared handlers execute by round-tripping a test event.

    Publishes a synthetic ModelNodeIntrospectionEvent to the first declared
    subscribe_topic, then consumes from the first declared publish_topic
    looking for a response with a matching verify- correlation_id.

    Args:
        parsed_contract: The parsed contract with topic and handler metadata.
        publish_fn: Callable(topic, payload) that publishes an event.
        consume_fn: Callable(topic, timeout) that returns consumed messages.
        timeout_seconds: Max seconds to wait for a response (default 10).

    Returns:
        ModelContractCheckResult with PASS if a verify-correlated event is
        received on the publish topic within timeout, FAIL otherwise.
    """
    contract_name = parsed_contract.name

    # Guard: need at least one subscribe and one publish topic
    if not parsed_contract.subscribe_topics:
        return ModelContractCheckResult(
            check_type=EnumContractCheckType.HANDLER_EXECUTION,
            severity=EnumCheckSeverity.RECOMMENDED,
            verdict=EnumValidationVerdict.FAIL,
            evidence="No subscribe_topics declared; cannot inject test event.",
            contract_name=contract_name,
            message="Handler execution check skipped: no subscribe topics.",
        )

    if not parsed_contract.publish_topics:
        return ModelContractCheckResult(
            check_type=EnumContractCheckType.HANDLER_EXECUTION,
            severity=EnumCheckSeverity.RECOMMENDED,
            verdict=EnumValidationVerdict.FAIL,
            evidence="No publish_topics declared; cannot verify handler output.",
            contract_name=contract_name,
            message="Handler execution check skipped: no publish topics.",
        )

    inject_topic = parsed_contract.subscribe_topics[0]
    observe_topic = parsed_contract.publish_topics[0]
    synthetic_event = _build_synthetic_event(contract_name)
    correlation_id = synthetic_event["correlation_id"]

    # Publish the synthetic event
    try:
        publish_fn(inject_topic, synthetic_event)
    # ONEX_EXCLUDE: blind_except - boundary probe must not crash on infra errors
    except Exception as exc:  # noqa: BLE001
        return ModelContractCheckResult(
            check_type=EnumContractCheckType.HANDLER_EXECUTION,
            severity=EnumCheckSeverity.RECOMMENDED,
            verdict=EnumValidationVerdict.FAIL,
            evidence=f"Failed to publish synthetic event to {inject_topic}: {exc}",
            contract_name=contract_name,
            message="Handler execution check failed: publish error.",
        )

    # Consume from the publish topic looking for our correlation_id
    try:
        messages = consume_fn(observe_topic, timeout_seconds)
    # ONEX_EXCLUDE: blind_except - boundary probe must not crash on infra errors
    except Exception as exc:  # noqa: BLE001
        return ModelContractCheckResult(
            check_type=EnumContractCheckType.HANDLER_EXECUTION,
            severity=EnumCheckSeverity.RECOMMENDED,
            verdict=EnumValidationVerdict.FAIL,
            evidence=f"Failed to consume from {observe_topic}: {exc}",
            contract_name=contract_name,
            message="Handler execution check failed: consume error.",
        )

    # Look for a message with our correlation_id
    matched = [
        msg
        for msg in messages
        if msg.get("correlation_id", "").startswith(VERIFY_CORRELATION_PREFIX)
        and msg.get("correlation_id", "") == correlation_id
    ]

    if matched:
        return ModelContractCheckResult(
            check_type=EnumContractCheckType.HANDLER_EXECUTION,
            severity=EnumCheckSeverity.RECOMMENDED,
            verdict=EnumValidationVerdict.PASS,
            evidence=(
                f"Handler produced response on {observe_topic} with "
                f"correlation_id={correlation_id}. "
                f"Received {len(matched)} matching message(s) "
                f"out of {len(messages)} total."
            ),
            contract_name=contract_name,
            message="Handler execution check passed: response received.",
        )

    return ModelContractCheckResult(
        check_type=EnumContractCheckType.HANDLER_EXECUTION,
        severity=EnumCheckSeverity.RECOMMENDED,
        verdict=EnumValidationVerdict.FAIL,
        evidence=(
            f"No response with correlation_id={correlation_id} received on "
            f"{observe_topic} within {timeout_seconds}s. "
            f"Total messages consumed: {len(messages)}."
        ),
        contract_name=contract_name,
        message=(
            f"Handler execution check failed: no response within "
            f"{timeout_seconds}s timeout."
        ),
    )


__all__: list[str] = [
    "PublishFn",
    "ConsumeFn",
    "VERIFY_CORRELATION_PREFIX",
    "check_handler_execution",
]
