# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Contract-declared alert bounds for the DLQ monitor (OMN-16769 AC3).

Every number here is an EXPERIMENT DEFAULT, stated explicitly rather than
buried in a script. They are pinned in the node contract and are expected
to be re-ratified against measured traffic — the measure-and-ratify
pattern, not a permanent claim of correctness.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.nodes.node_dlq_depth_evaluate_compute.models.model_dlq_topic_threshold_override import (
    ModelDlqTopicThresholdOverride,
)


class ModelDlqThresholdPolicy(BaseModel):
    """Bounds the evaluator applies. Contract-pinned, never script-hardcoded."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    window_seconds: int = Field(
        default=1800,
        ge=60,
        le=86400,
        description=(
            "Evaluation window. Defaults to 1800s to match the 30-minute "
            "scheduled cadence, so consecutive runs tile the timeline with "
            "neither gaps nor double-counting."
        ),
    )
    default_max_arrivals_per_window: int = Field(
        default=0,
        ge=0,
        description=(
            "EXPERIMENT DEFAULT: 0 — ANY arrival into a dead-letter sink is "
            "alertable. This is deliberately the strictest possible bound. A "
            "DLQ is by definition the place messages go when the system could "
            "not handle them; a steady trickle is not a healthy baseline to be "
            "tuned around, it is an unfixed defect. Topics that genuinely "
            "carry standing traffic today are listed in `overrides` WITH their "
            "reason, so the allowance is visible instead of averaged away. "
            "Chosen over a rate-per-minute bound because the OMN-16767 "
            "reproduction (16 quarantined commands in ~1 minute) must trip the "
            "alert, and 16 arrivals spread over a 30-minute window is only "
            "0.53/min — a per-minute bound low enough to catch it would be "
            "indistinguishable from this absolute-count bound anyway."
        ),
    )
    max_retained_depth: int | None = Field(
        default=None,
        ge=0,
        description=(
            "SECONDARY, DISABLED BY DEFAULT (None). A standing-backlog depth "
            "bound cannot be the primary signal: onex.dlq.omnibase-infra."
            "quarantine.v1 already holds ~8.88M retained records, so ANY "
            "finite depth bound is either already breached (alerting forever, "
            "which AC4 explicitly rejects as 'not an alert') or set above 8.88M "
            "(alerting never). Depth is therefore REPORTED as context on every "
            "row but does not gate the run unless an operator opts in."
        ),
    )
    overrides: tuple[ModelDlqTopicThresholdOverride, ...] = Field(
        default=(),
        description="Per-topic allowances, each carrying its own justification.",
    )

    @model_validator(mode="after")
    def _reject_duplicate_override_topics(self) -> ModelDlqThresholdPolicy:
        """Two overrides for one topic is ambiguous — fail rather than pick one."""
        seen = [override.topic for override in self.overrides]
        duplicates = sorted({topic for topic in seen if seen.count(topic) > 1})
        if duplicates:
            raise ValueError(
                "Duplicate threshold overrides for topic(s): "
                f"{', '.join(duplicates)} — refusing to silently pick one."
            )
        return self

    def bound_for(self, topic: str) -> int:
        """Resolve the arrival bound for one topic (override, else default)."""
        for override in self.overrides:
            if override.topic == topic:
                return override.max_arrivals_per_window
        return self.default_max_arrivals_per_window


__all__ = ["ModelDlqThresholdPolicy"]
