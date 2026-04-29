# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Analyzer for using premium models on small/simple requests."""

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

RULE_ID = "model_overkill"
PREMIUM_MODEL_MARKERS = ("opus", "gpt-4", "o3")
SIMPLE_REQUEST_TYPES = {"classification", "embedding", "routing", "summarization"}


def analyze_model_overkill(
    calls: tuple[ModelWasteCall, ...], detected_at: datetime
) -> tuple[ModelWasteFinding, ...]:
    """Detect premium model use on small/simple work."""
    overkill = tuple(
        call
        for call in calls
        if any(marker in call.model_id.lower() for marker in PREMIUM_MODEL_MARKERS)
        and (
            call.request_type.lower() in SIMPLE_REQUEST_TYPES
            or call.total_tokens <= 600
        )
        and call.cost_usd > 0
    )
    if not overkill:
        return ()

    waste_tokens = sum(call.total_tokens for call in overkill)
    waste_cost_usd = sum(call.cost_usd * 0.5 for call in overkill)
    repo_name, machine_id = first_attribution(overkill)
    evidence = {
        "call_count": len(overkill),
        "models": sorted({call.model_id for call in overkill}),
        "request_types": sorted({call.request_type for call in overkill}),
        "correlation_ids": [
            call.correlation_id for call in overkill if call.correlation_id
        ],
    }
    return (
        build_finding(
            session_id=overkill[0].session_id,
            rule_id=RULE_ID,
            severity="LOW" if waste_cost_usd < 0.02 else "MEDIUM",
            waste_tokens=waste_tokens,
            waste_cost_usd=waste_cost_usd,
            evidence=evidence,
            recommendation="Route small deterministic requests to a cheaper model tier.",
            repo_name=repo_name,
            machine_id=machine_id,
            detected_at=detected_at,
        ),
    )
