# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Publication verification probe.

Verifies that a contract's declared publish_topics have non-zero offsets,
indicating that messages have been published. Existence-oriented only --
does NOT check freshness (timestamp-based recency).
"""

from __future__ import annotations

import logging
import subprocess
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

logger = logging.getLogger(__name__)

# Type alias: given a topic name, return (low, high) watermark offsets.
# Raises on infrastructure failure.
WatermarkFn = Callable[[str], tuple[int, int]]

# Suffixes that identify core registration topics. Topics containing these
# substrings get REQUIRED severity; all others get RECOMMENDED.
# Pattern-based matching avoids hardcoding full topic strings (OMN-5256).
CORE_REGISTRATION_SUFFIXES: tuple[str, ...] = (
    "node-registration-result",
    "node-registration-initiated",
)


def _rpk_watermark_fallback(topic: str) -> tuple[int, int]:
    """Fallback: query topic watermark offsets via rpk CLI.

    Args:
        topic: Topic name to query.

    Returns:
        Tuple of (low_watermark, high_watermark) across all partitions.

    Raises:
        RuntimeError: If rpk is not available or returns an error.
    """
    try:
        result = subprocess.run(
            [
                "rpk",
                "topic",
                "describe",
                topic,
                "--print-partitions",
                "--format",
                "json",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("rpk not found on PATH") from exc

    if result.returncode != 0:
        raise RuntimeError(f"rpk topic describe failed: {result.stderr.strip()}")

    import json

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"rpk returned invalid JSON: {result.stdout[:200]}") from exc

    # Aggregate across partitions
    partitions = data.get("partitions", [])
    if not partitions:
        return (0, 0)

    total_low = 0
    total_high = 0
    for partition in partitions:
        total_low += partition.get("log_start_offset", 0)
        total_high += partition.get("high_watermark", 0)
    return (total_low, total_high)


def _is_core_registration_topic(topic: str) -> bool:
    """Check if a topic matches a core registration suffix pattern."""
    return any(suffix in topic for suffix in CORE_REGISTRATION_SUFFIXES)


def _severity_for_topic(topic: str) -> EnumCheckSeverity:
    """Determine severity based on whether the topic is a core registration topic."""
    if _is_core_registration_topic(topic):
        return EnumCheckSeverity.REQUIRED
    return EnumCheckSeverity.RECOMMENDED


def check_publications(
    parsed_contract: ModelParsedContractForVerification,
    watermark_fn: WatermarkFn | None = None,
) -> list[ModelContractCheckResult]:
    """Verify a contract's declared publish_topics have non-zero offsets.

    Args:
        parsed_contract: Parsed contract with publish_topics.
        watermark_fn: Injectable callable(topic) -> (low, high) watermark.
            If None, uses rpk fallback.

    Returns:
        One ModelContractCheckResult per publish_topic.
    """
    if not parsed_contract.publish_topics:
        return [
            ModelContractCheckResult(
                check_type=EnumContractCheckType.PUBLICATION,
                severity=EnumCheckSeverity.REQUIRED,
                verdict=EnumValidationVerdict.PASS,
                evidence="No publish_topics declared; nothing to verify.",
                contract_name=parsed_contract.name,
                message="Publication check passed: no topics declared.",
            )
        ]

    wm_fn: WatermarkFn = watermark_fn or _rpk_watermark_fallback

    results: list[ModelContractCheckResult] = []
    for topic in parsed_contract.publish_topics:
        severity = _severity_for_topic(topic)

        try:
            _low, high = wm_fn(topic)
        # ONEX_EXCLUDE: blind_except - boundary probe must not crash on infra errors
        except Exception as exc:  # noqa: BLE001
            logger.warning("Watermark query failed for topic %s: %s", topic, exc)
            results.append(
                ModelContractCheckResult(
                    check_type=EnumContractCheckType.PUBLICATION,
                    severity=severity,
                    verdict=EnumValidationVerdict.QUARANTINE,
                    evidence=(f"Watermark query failed for topic '{topic}': {exc}"),
                    contract_name=parsed_contract.name,
                    message=f"Publication check quarantined for '{topic}': query failed.",
                )
            )
            continue

        if high > 0:
            results.append(
                ModelContractCheckResult(
                    check_type=EnumContractCheckType.PUBLICATION,
                    severity=severity,
                    verdict=EnumValidationVerdict.PASS,
                    evidence=(f"Topic '{topic}' has high watermark offset={high}."),
                    contract_name=parsed_contract.name,
                    message=f"Publication check passed for '{topic}'.",
                )
            )
        else:
            results.append(
                ModelContractCheckResult(
                    check_type=EnumContractCheckType.PUBLICATION,
                    severity=severity,
                    verdict=EnumValidationVerdict.FAIL,
                    evidence=(
                        f"Topic '{topic}' has zero offset (high_watermark=0). "
                        f"No messages have been published."
                    ),
                    contract_name=parsed_contract.name,
                    message=f"Publication check failed for '{topic}': no data.",
                )
            )

    return results


__all__: list[str] = [
    "CORE_REGISTRATION_SUFFIXES",
    "WatermarkFn",
    "check_publications",
]
