# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Verification report dashboard event emission.

Emits verification results as Kafka events so downstream consumers
(omnidash, alerting) can react to contract compliance state changes.
Topic is read from this module's own contract.yaml -- never hardcoded.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from omnibase_infra.verification.models import ModelContractVerificationReport

logger = logging.getLogger(__name__)

# Path to this module's contract.yaml (co-located).
_CONTRACT_PATH = Path(__file__).parent / "contract.yaml"

# ONEX_EXCLUDE: any_type - event payloads are heterogeneous dicts for Kafka serialization
PublishFn = Callable[[str, dict[str, Any]], None]


def _load_publish_topic() -> str:
    """Load the publish topic from the co-located contract.yaml.

    Raises:
        FileNotFoundError: If the contract.yaml is missing.
        KeyError: If the contract has no publish_topics.
    """
    with open(_CONTRACT_PATH) as f:
        data = yaml.safe_load(f) or {}
    topics = data.get("event_bus", {}).get("publish_topics", [])
    if not topics:
        msg = f"No publish_topics in {_CONTRACT_PATH}"
        raise KeyError(msg)
    return topics[0]


# ONEX_EXCLUDE: any_type - event payloads are heterogeneous dicts for Kafka serialization
def _build_event_payload(
    report: ModelContractVerificationReport,
) -> dict[str, Any]:
    """Build the event payload from a verification report.

    Includes freshness fields (checked_at, fingerprint) so consumers
    can detect staleness.
    """
    return {
        "contract_name": report.contract_name,
        "node_type": report.node_type,
        "overall_verdict": report.overall_verdict.value,
        "check_count": len(report.checks),
        "fail_count": sum(1 for c in report.checks if c.verdict.value == "fail"),
        "checked_at": report.checked_at.isoformat(),
        "report_fingerprint": report.report_fingerprint,
        "duration_ms": report.duration_ms,
        "emitted_at": datetime.now(UTC).isoformat(),
    }


def emit_verification_result(
    report: ModelContractVerificationReport,
    publish_fn: PublishFn,
) -> None:
    """Emit a verification result as a Kafka event.

    Fire-and-forget: logs errors but does not raise, so verification
    is never blocked by event emission failures.

    Args:
        report: The verification report to emit.
        publish_fn: Callable(topic, payload_dict) that publishes the event.
    """
    try:
        topic = _load_publish_topic()
        payload = _build_event_payload(report)
        publish_fn(topic, payload)
        logger.info(
            "Emitted verification event for %s to %s",
            report.contract_name,
            topic,
        )
    # ONEX_EXCLUDE: blind_except - emission must never block verification
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to emit verification event for %s: %s",
            report.contract_name,
            exc,
        )


__all__: list[str] = [
    "PublishFn",
    "emit_verification_result",
]
