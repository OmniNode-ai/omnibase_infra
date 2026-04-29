# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Analyzer for duplicate retries that repeat the same request."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from omnibase_infra.nodes.node_waste_detection_compute.handlers.analyzer_utils import (
    build_finding,
    first_attribution,
    sorted_calls,
)
from omnibase_infra.nodes.node_waste_detection_compute.models import (
    ModelWasteCall,
    ModelWasteFinding,
)

RULE_ID = "retry_waste"
WINDOW_SECONDS = 60


def analyze_retry_waste(
    calls: tuple[ModelWasteCall, ...], detected_at: datetime
) -> tuple[ModelWasteFinding, ...]:
    """Detect repeated successful requests inside a retry window."""
    by_signature: dict[str, list[ModelWasteCall]] = defaultdict(list)
    for call in sorted_calls(calls):
        if (call.status or "success").lower() not in {"success", "ok", "completed"}:
            continue
        by_signature[f"{call.model_id}:{call.request_signature}"].append(call)

    findings: list[ModelWasteFinding] = []
    for signature, group in sorted(by_signature.items()):
        if len(group) < 2:
            continue
        first = group[0]
        retries = [
            call
            for call in group[1:]
            if (call.emitted_at - first.emitted_at).total_seconds() <= WINDOW_SECONDS
        ]
        if not retries:
            continue
        waste_tokens = sum(call.total_tokens for call in retries)
        waste_cost_usd = sum(call.cost_usd for call in retries)
        repo_name, machine_id = first_attribution(group)
        evidence = {
            "signature": signature,
            "retry_count": len(retries),
            "first_correlation_id": first.correlation_id,
            "retry_correlation_ids": [
                call.correlation_id for call in retries if call.correlation_id
            ],
        }
        findings.append(
            build_finding(
                session_id=first.session_id,
                rule_id=RULE_ID,
                severity="MEDIUM" if len(retries) == 1 else "HIGH",
                waste_tokens=waste_tokens,
                waste_cost_usd=waste_cost_usd,
                evidence=evidence,
                recommendation="Use idempotency and retry budgets before resubmitting identical LLM requests.",
                repo_name=repo_name,
                machine_id=machine_id,
                detected_at=detected_at,
            )
        )
    return tuple(findings)
