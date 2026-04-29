# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Analyzer for tokens spent on failed tool-backed calls."""

from __future__ import annotations

from datetime import datetime

from omnibase_infra.nodes.node_waste_detection_compute.handlers.analyzer_utils import (
    build_finding,
    first_attribution,
)
from omnibase_infra.nodes.node_waste_detection_compute.models import (
    ModelWasteCall,
    ModelWasteFinding,
)

RULE_ID = "tool_failure_waste"
FAILED_STATUSES = {"failed", "failure", "error", "errored", "timeout"}


def analyze_tool_failure_waste(
    calls: tuple[ModelWasteCall, ...], detected_at: datetime
) -> tuple[ModelWasteFinding, ...]:
    """Detect calls whose tool execution failed after spending tokens."""
    failed = tuple(
        call
        for call in calls
        if (call.status or "").lower() in FAILED_STATUSES or call.error_type
    )
    if not failed:
        return ()

    waste_tokens = sum(call.total_tokens for call in failed)
    waste_cost_usd = sum(call.cost_usd for call in failed)
    repo_name, machine_id = first_attribution(failed)
    evidence = {
        "failed_call_count": len(failed),
        "correlation_ids": sorted(
            call.correlation_id for call in failed if call.correlation_id
        ),
        "error_types": sorted({call.error_type for call in failed if call.error_type}),
    }
    severity = "HIGH" if waste_cost_usd >= 0.1 or waste_tokens >= 10000 else "MEDIUM"
    return (
        build_finding(
            session_id=failed[0].session_id,
            rule_id=RULE_ID,
            severity=severity,
            waste_tokens=waste_tokens,
            waste_cost_usd=waste_cost_usd,
            evidence=evidence,
            recommendation="Fix failing tool calls before retrying LLM-backed work.",
            repo_name=repo_name,
            machine_id=machine_id,
            detected_at=detected_at,
        ),
    )
