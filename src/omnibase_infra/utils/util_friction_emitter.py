# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lightweight friction event emitter for build loop handlers.

Writes NDJSON friction events to ``$ONEX_STATE_DIR/friction/build-loop.ndjson``.
This module has zero dependencies outside the standard library so that any
omnibase_infra handler can emit friction without importing omniclaude.

Design:
- Best-effort: all errors are swallowed and logged.
- Non-blocking: file append only, no network I/O.
- Deterministic: timestamps must be injected by callers.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

logger = logging.getLogger(__name__)

_REDACT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"password\s*[=:]\s*\S+", re.IGNORECASE), "password=[REDACTED]"),
    (re.compile(r"://[^:@/\s]+:[^@\s]+@"), "://[REDACTED]@"),
    (re.compile(r"api[_-]?key\s*[=:]\s*\S+", re.IGNORECASE), "api_key=[REDACTED]"),
    (
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        "[EMAIL_REDACTED]",
    ),
]


def _sanitize_error_message(msg: str) -> str:
    """Redact credentials and PII from error messages before writing to disk."""
    for pattern, replacement in _REDACT_PATTERNS:
        msg = pattern.sub(replacement, msg)
    return msg


def emit_build_loop_friction(
    *,
    phase: str,
    correlation_id: UUID,
    severity: str = "high",
    description: str,
    error_message: str | None = None,
    timestamp: datetime | None = None,
) -> bool:
    """Append a friction event to the build loop friction log.

    Args:
        phase: Build loop phase where friction occurred.
        correlation_id: Cycle correlation ID.
        severity: One of "critical", "high", "medium", "low".
        description: Human-readable friction description.
        error_message: Optional underlying error message.
        timestamp: Event timestamp (defaults to now if not provided).

    Returns:
        True if the event was written, False on any error.
    """
    try:
        state_dir = os.environ.get("ONEX_STATE_DIR", "")
        if not state_dir:
            logger.debug("ONEX_STATE_DIR not set — friction emission skipped")
            return False

        friction_dir = Path(state_dir) / "friction"
        friction_dir.mkdir(parents=True, exist_ok=True)

        ts = timestamp or datetime.now(UTC)

        record = {
            "skill": "build_loop",
            "surface": f"build_loop/{phase}",
            "severity": severity,
            "description": description,
            "error_message": _sanitize_error_message(error_message)
            if error_message
            else "",
            "correlation_id": str(correlation_id),
            "phase": phase,
            "timestamp": ts.isoformat(),
        }

        friction_path = friction_dir / "build-loop.ndjson"
        with friction_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

        logger.info(
            "Friction emitted: phase=%s severity=%s correlation_id=%s",
            phase,
            severity,
            correlation_id,
        )
        return True
    except Exception:  # noqa: BLE001 — boundary: best-effort friction emission must never crash callers
        logger.debug("Failed to emit friction event", exc_info=True)
        return False
