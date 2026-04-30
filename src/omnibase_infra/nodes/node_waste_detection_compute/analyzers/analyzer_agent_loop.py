# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Analyzer for repeated identical agent/tool actions."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from omnibase_infra.nodes.node_waste_detection_compute.analyzers.analyzer_utils import (
    build_finding,
    first_attribution,
    sorted_calls,
)
from omnibase_infra.nodes.node_waste_detection_compute.models import (
    ModelWasteCall,
    ModelWasteFinding,
)

RULE_ID = "agent_loop"
WINDOW_SECONDS = 10
MIN_REPEATS = 3


def analyze_agent_loop(
    calls: tuple[ModelWasteCall, ...], detected_at: datetime
) -> tuple[ModelWasteFinding, ...]:
    """Detect three or more identical actions inside a short time window."""
    by_signature: dict[str, list[ModelWasteCall]] = defaultdict(list)
    for call in sorted_calls(calls):
        by_signature[call.request_signature].append(call)

    findings: list[ModelWasteFinding] = []
    for signature, group in sorted(by_signature.items()):
        if len(group) < MIN_REPEATS:
            continue
        for index in range(len(group) - MIN_REPEATS + 1):
            start_call = group[index]
            window = [
                call
                for call in group[index:]
                if (call.emitted_at - start_call.emitted_at).total_seconds()
                <= WINDOW_SECONDS
            ]
            if len(window) < MIN_REPEATS:
                continue
            elapsed = (window[-1].emitted_at - start_call.emitted_at).total_seconds()
            repeated = window[1:]
            waste_tokens = sum(call.total_tokens for call in repeated)
            waste_cost_usd = sum(call.cost_usd for call in repeated)
            repo_name, machine_id = first_attribution(window)
            evidence = {
                "signature": signature,
                "repeat_count": len(window),
                "window_seconds": int(elapsed),
                "correlation_ids": [
                    call.correlation_id for call in window if call.correlation_id
                ],
            }
            findings.append(
                build_finding(
                    session_id=window[0].session_id,
                    rule_id=RULE_ID,
                    severity="HIGH" if len(window) >= 5 else "MEDIUM",
                    waste_tokens=waste_tokens,
                    waste_cost_usd=waste_cost_usd,
                    evidence=evidence,
                    recommendation="Collapse repeated agent actions into one call or add loop termination.",
                    repo_name=repo_name,
                    machine_id=machine_id,
                    detected_at=detected_at,
                )
            )
            break
    return tuple(findings)
