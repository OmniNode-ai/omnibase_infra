# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Bifrost shadow mode configuration.

Shadow mode runs a learned routing policy in parallel with the static
rule-based routing, without affecting the actual routing decision.

Shadow Mode Mechanism:
    - Shadow mode is enabled/disabled via ``ModelBifrostShadowConfig.enabled``.
    - The learned policy checkpoint is loaded from ``checkpoint_path`` at
      gateway construction time. To update the checkpoint, the gateway must
      be reconstructed (restart). Hot-reload is NOT supported because:
        1. Checkpoint files may be large (100s of MB for RL policies)
        2. Atomic swap of in-memory policy requires careful locking
        3. Restart-based deployment is the standard ONEX pattern
    - When enabled, the shadow policy is evaluated async after the static
      routing decision is made, adding < 5ms latency.
    - Shadow decisions are logged to the ``routing_shadow_decisions`` table
      in the omnidash read-model database via Kafka events.

Related Tickets:
    - OMN-5570: Shadow Mode + Comparison Dashboard
    - OMN-5556: Learned Decision Optimization Platform (epic)
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelBifrostShadowConfig(BaseModel):
    """Configuration for bifrost shadow mode.

    Controls whether a learned routing policy runs in parallel with
    static routing rules. The shadow policy's decisions are logged
    but never affect the actual routing outcome.

    Attributes:
        enabled: Whether shadow mode is active. When False, no shadow
            computation occurs and no latency is added.
        checkpoint_path: Filesystem path to the learned policy checkpoint
            file. Required when ``enabled=True``. The checkpoint format
            is policy-specific (e.g. ONNX, pickle, safetensors).
        log_sample_rate: Fraction of requests to log shadow decisions for.
            1.0 means log 100% of requests (required for promotion gate
            evaluation). Values < 1.0 are useful for high-traffic scenarios.
        comparison_logging_enabled: Whether to emit Kafka events for each
            shadow decision comparison. When False, shadow decisions are
            still computed but not persisted.
        max_shadow_latency_ms: Maximum allowed latency for shadow policy
            evaluation in milliseconds. If the shadow computation exceeds
            this timeout, it is cancelled and a timeout is logged.
        policy_version: Human-readable version identifier for the loaded
            policy checkpoint. Recorded in every shadow decision log for
            A/B tracking across policy iterations.

    Example:
        >>> config = ModelBifrostShadowConfig(
        ...     enabled=True,
        ...     checkpoint_path="/models/bifrost_policy_v1.onnx",
        ...     policy_version="v1.0.0-alpha",
        ... )
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    enabled: bool = Field(
        default=False,
        description="Whether shadow mode is active.",
    )
    checkpoint_path: str | None = Field(
        default=None,
        description=(
            "Filesystem path to the learned policy checkpoint. "
            "Required when enabled=True."
        ),
    )
    log_sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Fraction of requests to log shadow decisions for (1.0 = 100%).",
    )
    comparison_logging_enabled: bool = Field(
        default=True,
        description="Whether to emit Kafka events for shadow decision comparisons.",
    )
    max_shadow_latency_ms: float = Field(
        default=5.0,
        ge=0.1,
        le=100.0,
        description=(
            "Maximum allowed latency for shadow policy evaluation (ms). "
            "Shadow computation is cancelled if it exceeds this timeout."
        ),
    )
    policy_version: str = Field(
        default="unknown",
        max_length=128,
        description="Human-readable version of the loaded policy checkpoint.",
    )


__all__: list[str] = ["ModelBifrostShadowConfig"]
