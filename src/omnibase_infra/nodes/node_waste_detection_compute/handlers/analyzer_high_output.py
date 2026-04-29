# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Analyzer for unusually large output-token spend."""

from __future__ import annotations

from datetime import datetime

from omnibase_infra.nodes.node_waste_detection_compute.handlers.analyzer_utils import (
    build_finding,
    cost_for_tokens,
    first_attribution,
)
from omnibase_infra.nodes.node_waste_detection_compute.models import (
    ModelWasteCall,
    ModelWasteFinding,
)

RULE_ID = "high_output"
OUTPUT_TOKEN_THRESHOLD = 4000
OUTPUT_RATIO_THRESHOLD = 0.8


def analyze_high_output(
    calls: tuple[ModelWasteCall, ...], detected_at: datetime
) -> tuple[ModelWasteFinding, ...]:
    """Detect calls dominated by excessive completion tokens."""
    excessive = tuple(
        call
        for call in calls
        if call.output_tokens >= OUTPUT_TOKEN_THRESHOLD
        or (
            call.total_tokens > 0
            and call.output_tokens / call.total_tokens >= OUTPUT_RATIO_THRESHOLD
            and call.output_tokens >= 1000
        )
    )
    if not excessive:
        return ()

    waste_tokens = sum(max(call.output_tokens - 1000, 0) for call in excessive)
    waste_cost_usd = sum(
        cost_for_tokens(call, max(call.output_tokens - 1000, 0)) for call in excessive
    )
    repo_name, machine_id = first_attribution(excessive)
    evidence = {
        "call_count": len(excessive),
        "max_output_tokens": max(call.output_tokens for call in excessive),
        "correlation_ids": [
            call.correlation_id for call in excessive if call.correlation_id
        ],
    }
    return (
        build_finding(
            session_id=excessive[0].session_id,
            rule_id=RULE_ID,
            severity="HIGH" if waste_tokens >= 8000 else "MEDIUM",
            waste_tokens=waste_tokens,
            waste_cost_usd=waste_cost_usd,
            evidence=evidence,
            recommendation="Constrain response length or request structured, bounded output.",
            repo_name=repo_name,
            machine_id=machine_id,
            detected_at=detected_at,
        ),
    )
