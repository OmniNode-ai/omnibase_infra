# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Analyzer for cache-eligible sessions with low cache reads."""

from __future__ import annotations

from datetime import datetime

from omnibase_infra.nodes.node_waste_detection_compute.analyzers.analyzer_utils import (
    build_finding,
    cost_for_tokens,
    first_attribution,
    sorted_calls,
)
from omnibase_infra.nodes.node_waste_detection_compute.models import (
    ModelWasteCall,
    ModelWasteFinding,
)

RULE_ID = "low_cache"
MIN_INPUT_TOKENS = 6000
MAX_CACHE_READ_RATIO = 0.1


def analyze_low_cache(
    calls: tuple[ModelWasteCall, ...], detected_at: datetime
) -> tuple[ModelWasteFinding, ...]:
    """Detect large prompt sessions with little cache reuse."""
    eligible = tuple(
        call
        for call in sorted_calls(calls)
        if call.input_tokens >= MIN_INPUT_TOKENS and call.cache_read_tokens is not None
    )
    low_cache = tuple(
        call
        for call in eligible
        if call.input_tokens > 0
        and (call.cache_read_tokens or 0) / call.input_tokens < MAX_CACHE_READ_RATIO
    )
    if not low_cache:
        return ()

    waste_tokens = sum(
        max(int(call.input_tokens * 0.5) - (call.cache_read_tokens or 0), 0)
        for call in low_cache
    )
    waste_cost_usd = sum(
        cost_for_tokens(
            call, max(int(call.input_tokens * 0.5) - (call.cache_read_tokens or 0), 0)
        )
        for call in low_cache
    )
    repo_name, machine_id = first_attribution(low_cache)
    evidence = {
        "call_count": len(low_cache),
        "total_input_tokens": sum(call.input_tokens for call in low_cache),
        "total_cache_read_tokens": sum(
            call.cache_read_tokens or 0 for call in low_cache
        ),
        "correlation_ids": sorted(
            call.correlation_id for call in low_cache if call.correlation_id
        ),
    }
    return (
        build_finding(
            session_id=low_cache[0].session_id,
            rule_id=RULE_ID,
            severity="HIGH" if waste_tokens >= 12000 else "MEDIUM",
            waste_tokens=waste_tokens,
            waste_cost_usd=waste_cost_usd,
            evidence=evidence,
            recommendation="Enable prompt caching or avoid resending stable context.",
            repo_name=repo_name,
            machine_id=machine_id,
            detected_at=detected_at,
        ),
    )
