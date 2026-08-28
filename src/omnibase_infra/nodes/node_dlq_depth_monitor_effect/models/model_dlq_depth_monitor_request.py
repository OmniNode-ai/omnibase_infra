# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Request model for the read-only DLQ depth probe (OMN-16769)."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# Every ONEX DLQ topic starts with this, for both the omnibase-infra producer
# segment and the per-node omnimarket ones (verified live on the .201 dev lane
# 2026-08-27: 60 topics matched, spanning onex.dlq.omnibase-infra.* and
# onex.dlq.omnimarket.*).
DEFAULT_DLQ_TOPIC_PREFIX = "onex.dlq."


class ModelDlqDepthMonitorRequest(BaseModel):
    """Configuration for one read-only DLQ sweep of the broker."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Probe run correlation ID.")
    topic_prefix: str = Field(
        default=DEFAULT_DLQ_TOPIC_PREFIX,
        min_length=1,
        description="Only topics starting with this prefix are probed.",
    )
    window_seconds: int = Field(
        default=1800,
        ge=60,
        le=86400,
        description=(
            "Arrival-measurement window, ending at evaluated_at. Defaults to "
            "the 30-minute scheduled cadence so consecutive runs tile the "
            "timeline exactly."
        ),
    )
    default_max_arrivals_per_window: int = Field(
        default=0,
        ge=0,
        description="Arrival bound applied to every topic without an override.",
    )
    max_retained_depth: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Opt-in secondary depth bound. Left unset the monitor never gates "
            "on depth — see the contract's alert_policy block for why a depth "
            "bound cannot be the primary signal on a topic holding ~8.88M "
            "standing records."
        ),
    )
    suppress_alert_exit: bool = Field(
        default=False,
        description=(
            "Default False: an alerting run RAISES, so the scheduled workflow "
            "goes RED — a red run IS the alert surface. Set True to collect "
            "the histogram without gating, which is what characterization and "
            "baseline runs use.\n\n"
            "Phrased as suppress-rather-than-enable on purpose: `onex skill` "
            "booleans are presence-only flags (cli_skill.py sets True on "
            "sight and rejects any unknown `--no-*` token), so a "
            "default-True `fail_on_alert` would have no way to be turned off "
            "from the CLI. Framing the flag negatively keeps the SAFE value "
            "(alerts gate the run) as the default that needs no flag at all."
        ),
    )


__all__ = ["DEFAULT_DLQ_TOPIC_PREFIX", "ModelDlqDepthMonitorRequest"]
