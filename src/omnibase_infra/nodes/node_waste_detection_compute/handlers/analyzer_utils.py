# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Shared helpers for pure waste analyzer functions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import datetime

from omnibase_infra.nodes.node_waste_detection_compute.models import (
    ModelWasteCall,
    ModelWasteFinding,
    WasteSeverity,
)


def canonical_hash(evidence: dict[str, object]) -> str:
    """Return a deterministic SHA-256 hash for evidence."""
    payload = json.dumps(evidence, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_finding(
    *,
    session_id: str,
    rule_id: str,
    severity: WasteSeverity,
    waste_tokens: int,
    waste_cost_usd: float,
    evidence: dict[str, object],
    recommendation: str,
    detected_at: datetime,
    repo_name: str | None = None,
    machine_id: str | None = None,
) -> ModelWasteFinding:
    """Build a deterministic finding with the required dedup key."""
    evidence_hash = canonical_hash(evidence)
    return ModelWasteFinding(
        session_id=session_id,
        rule_id=rule_id,
        severity=severity,
        waste_tokens=waste_tokens,
        waste_cost_usd=round(waste_cost_usd, 6),
        evidence=evidence,
        evidence_hash=evidence_hash,
        dedup_key=f"{session_id}:{rule_id}:{evidence_hash}",
        recommendation=recommendation,
        repo_name=repo_name,
        machine_id=machine_id,
        detected_at=detected_at,
    )


def first_attribution(calls: Iterable[ModelWasteCall]) -> tuple[str | None, str | None]:
    """Return the first repo and machine attribution present in a call set."""
    repo_name = None
    machine_id = None
    for call in calls:
        repo_name = repo_name or call.repo_name
        machine_id = machine_id or call.machine_id
        if repo_name is not None and machine_id is not None:
            break
    return repo_name, machine_id


def cost_for_tokens(call: ModelWasteCall, tokens: int) -> float:
    """Allocate call cost proportionally to token count."""
    if call.total_tokens <= 0:
        return 0.0
    return call.cost_usd * min(tokens, call.total_tokens) / call.total_tokens


def sorted_calls(calls: tuple[ModelWasteCall, ...]) -> list[ModelWasteCall]:
    """Sort calls by event time for deterministic analysis."""
    return sorted(calls, key=lambda call: (call.emitted_at, call.correlation_id or ""))
