# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-declared canary probe config for the gateway path-verifying healthcheck.

OMN-15741 (G1): the container healthcheck must exercise the same transport and
credentials as real traffic — not a sentinel file — and must be able to tell
"local leg healthy, cloud leg dead" apart from "both legs healthy" (the exact
failure mode of the 2026-08-04 outage: the ready-file was present and the
process was up for four days while forwarding 0%). This model is the sole
authority for the canary topic, cadence, and per-leg deadlines; the probe
process consults it instead of hardcoding any of those values.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_topic_transform import (
    validate_canonical_topic,
)


class ModelGatewayCanaryConfig(BaseModel):
    """Dedicated canary topic + cadence + deadlines for the path-verifying probe.

    The canary topic is deliberately separate from ``mirror_topics`` — it never
    carries tenant traffic, so probing it can never be mistaken for (or
    interfere with) the real forwarding path. ``cadence_seconds`` bounds how
    often the probe is allowed to perform a real produce+readback round trip
    against the live brokers; between real checks the probe process may serve
    a cached result so the healthcheck does not spam either leg.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    topic: str = Field(..., min_length=1)
    cadence_seconds: int = Field(..., ge=1)
    produce_deadline_seconds: float = Field(..., gt=0)
    readback_deadline_seconds: float = Field(..., gt=0)

    @field_validator("topic")
    @classmethod
    def _validate_topic(cls, value: str) -> str:
        return validate_canonical_topic(value)

    @property
    def total_deadline_seconds(self) -> float:
        """Upper bound on one full real check across a single leg."""
        return self.produce_deadline_seconds + self.readback_deadline_seconds


__all__ = ["ModelGatewayCanaryConfig"]
