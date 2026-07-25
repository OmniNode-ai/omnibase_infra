# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``terminal_event_hash`` mirror (OMN-15095).

Byte-for-byte mirror of ``omninode_infra``'s
``workflow_receipt_renderer._canonical_json`` / ``_sha256`` (OMN-15094):
``json.dumps(payload, sort_keys=True, default=str)`` encoded UTF-8, sha256
hexdigest. Cannot be a literal shared import for the same cross-repo/
dependency-conflict reason documented in ``wire_shape.py`` -- pinned exactly
to keep this a mirror, not an independent reimplementation that could drift.
A parity test (``test_receipt_hash_matches_renderer.py``) asserts this
produces the same hash as OMN-15094's fixture-recorded example.
"""

from __future__ import annotations

import hashlib
import json


def canonical_json(payload: dict[str, object]) -> str:
    """Stable JSON encoding so identical payloads always hash identically."""
    return json.dumps(payload, sort_keys=True, default=str)


def sha256_of(payload: dict[str, object]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def terminal_event_payload(
    *,
    correlation_id: str,
    status: str,
    terminal_model_used: str,
    terminal_total_tokens: int,
    terminal_latency_ms: int,
) -> dict[str, object]:
    """Build the exact dict shape OMN-15094 hashes for ``terminal_event_hash``."""
    return {
        "correlation_id": correlation_id,
        "status": status,
        "terminal_model_used": terminal_model_used,
        "terminal_total_tokens": terminal_total_tokens,
        "terminal_latency_ms": terminal_latency_ms,
    }


__all__ = ["canonical_json", "sha256_of", "terminal_event_payload"]
