# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Wiring health alert payload model for Slack webhook integration.

This module provides the data model for wiring health alerts that are
emitted when topics exceed the mismatch threshold. The alert payload
is designed to be consumed by a Slack webhook handler (OMN-1905).

Design Rationale:
    - Immutable (frozen=True) for thread-safety
    - Contains all context needed for actionable alerts
    - Slack-ready formatting with precomputed message
    - Used as ModelIntent payload for alert emission

See Also:
    - OMN-1895: Wiring health monitor implementation
    - OMN-1905: Slack webhook handler (consumer of this payload)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ModelWiringHealthAlert(BaseModel):
    """Alert payload for wiring health threshold violations.

    Contains all information needed to generate an actionable Slack alert
    when one or more topics exceed the mismatch threshold.

    Attributes:
        alert_type: Literal identifier for alert routing.
        correlation_id: Correlation ID for tracing.
        timestamp: When the alert was generated.
        environment: Environment where the alert originated.
        unhealthy_topics: List of topics exceeding threshold.
        threshold: The mismatch threshold that was exceeded.
        summary: Human-readable summary for Slack message.
        details: Detailed topic-by-topic breakdown.

    Example:
        >>> alert = ModelWiringHealthAlert(
        ...     environment="prod",
        ...     unhealthy_topics=["session-outcome"],
        ...     threshold=0.05,
        ...     summary="1 topic exceeds mismatch threshold",
        ...     details=[{"topic": "session-outcome", "emit": 100, "consume": 85}],
        ... )
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    alert_type: Literal["wiring_health_mismatch"] = Field(
        default="wiring_health_mismatch", description="Alert type identifier"
    )
    correlation_id: UUID = Field(
        default_factory=uuid4, description="Correlation ID for tracing"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Alert timestamp"
    )
    environment: str = Field(..., description="Environment (dev, prod, etc.)")
    unhealthy_topics: tuple[str, ...] = Field(
        ..., description="Topics exceeding threshold"
    )
    threshold: float = Field(..., ge=0.0, description="Mismatch threshold exceeded")
    summary: str = Field(..., description="Human-readable summary")
    details: tuple[dict[str, object], ...] = Field(
        default_factory=tuple, description="Per-topic breakdown"
    )

    @classmethod
    def from_metrics(
        cls,
        metrics: object,  # ModelWiringHealthMetrics - runtime type check below
        environment: str,
        correlation_id: UUID | None = None,
    ) -> ModelWiringHealthAlert | None:
        """Create alert from wiring health metrics if unhealthy.

        Args:
            metrics: The computed wiring health metrics.
            environment: Environment identifier.
            correlation_id: Optional correlation ID (generates new if None).

        Returns:
            ModelWiringHealthAlert if any topics are unhealthy, None otherwise.
        """
        # Avoid circular import
        from omnibase_infra.observability.wiring_health.model_wiring_health_metrics import (
            ModelWiringHealthMetrics as _Metrics,
        )

        if not isinstance(metrics, _Metrics):
            raise TypeError(f"Expected ModelWiringHealthMetrics, got {type(metrics)}")

        # No alert needed if healthy
        if metrics.overall_healthy:
            return None

        unhealthy = [t for t in metrics.topics if not t.is_healthy]
        unhealthy_topics = tuple(t.topic for t in unhealthy)

        # Build summary
        count = len(unhealthy)
        summary = (
            f"{count} topic{'s' if count != 1 else ''} "
            f"exceed{'s' if count == 1 else ''} "
            f"{metrics.threshold:.1%} mismatch threshold"
        )

        # Build details
        details = tuple(
            {
                "topic": t.topic,
                "emit_count": t.emit_count,
                "consume_count": t.consume_count,
                "mismatch_ratio": f"{t.mismatch_ratio:.2%}",
            }
            for t in unhealthy
        )

        return cls(
            correlation_id=correlation_id or uuid4(),
            environment=environment,
            unhealthy_topics=unhealthy_topics,
            threshold=metrics.threshold,
            summary=summary,
            details=details,
        )

    def to_slack_message(self) -> dict[str, object]:
        """Format alert as Slack message payload.

        Returns:
            Dictionary suitable for Slack webhook POST body.
        """
        # Build topic details as bullet list
        topic_lines = "\n".join(
            f"• *{d['topic']}*: {d['emit_count']} emit / {d['consume_count']} "
            f"consume ({d['mismatch_ratio']} mismatch)"
            for d in self.details
        )

        return {
            "text": f":warning: Wiring Health Alert - {self.environment}",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"⚠️ Wiring Health Alert - {self.environment}",
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": self.summary,
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Unhealthy Topics:*\n{topic_lines}",
                    },
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"Threshold: {self.threshold:.1%} | "
                            f"Correlation: `{self.correlation_id}` | "
                            f"Time: {self.timestamp.isoformat()}",
                        }
                    ],
                },
            ],
        }


__all__ = ["ModelWiringHealthAlert"]
