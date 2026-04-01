# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Eval-completed event emitter.

Emits an eval-completed event to Kafka after a full A/B eval run.
Topic: onex.evt.omnibase-infra.eval-completed.v1

Related:
    - OMN-6779: Emit eval-completed event to Kafka
    - OMN-6776: Eval orchestrator skill
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from onex_change_control.models.model_eval_report import ModelEvalReport

from omnibase_infra.topics import SUFFIX_EVAL_COMPLETED

logger = logging.getLogger(__name__)

# Topic declared in platform_topic_suffixes.py (OMN-6779)
EVAL_COMPLETED_TOPIC = SUFFIX_EVAL_COMPLETED


def build_eval_completed_payload(report: ModelEvalReport) -> dict[str, object]:
    """Build the Kafka event payload from an eval report.

    Returns a dict suitable for JSON serialization and Kafka production.
    """
    return {
        "event_type": "eval_completed",
        "topic": EVAL_COMPLETED_TOPIC,
        "timestamp": datetime.now(UTC).isoformat(),
        "report_id": report.report_id,
        "suite_id": report.suite_id,
        "suite_version": report.suite_version,
        "generated_at": report.generated_at.isoformat(),
        "summary": {
            "total_tasks": report.summary.total_tasks,
            "onex_better_count": report.summary.onex_better_count,
            "onex_worse_count": report.summary.onex_worse_count,
            "neutral_count": report.summary.neutral_count,
            "avg_latency_delta_ms": report.summary.avg_latency_delta_ms,
            "avg_token_delta": report.summary.avg_token_delta,
            "avg_success_rate_on": report.summary.avg_success_rate_on,
            "avg_success_rate_off": report.summary.avg_success_rate_off,
            "pattern_hit_rate_on": report.summary.pattern_hit_rate_on,
        },
    }


def serialize_eval_event(report: ModelEvalReport) -> bytes:
    """Serialize the eval-completed event as JSON bytes for Kafka."""
    payload = build_eval_completed_payload(report)
    return json.dumps(payload, default=str).encode("utf-8")


__all__: list[str] = [
    "EVAL_COMPLETED_TOPIC",
    "build_eval_completed_payload",
    "serialize_eval_event",
]
