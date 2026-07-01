# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Bounded readiness-confirm knobs for the boot interleave (OMN-13237).

Named knobs, not magic numbers (§3.7, §3.9). The metadata readiness poll is
bounded by a total budget, a poll cadence, and a max attempt count; the boot
interleave is allowed bounded parallelism across contracts via
``max_concurrent_contract_attach`` while each contract keeps the
provision->ready->attach order invariant (§3.9).

This model is pure (no env reads). The runtime kernel — the approved
overlay-resolution boundary — resolves any operator overrides and constructs the
config explicitly; see ``service_kernel.resolve_topic_readiness_config``.

Related Tickets:
    - OMN-13237: Per-contract scoped topic provisioning at runtime boot.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Conservative defaults: the per-contract confirm replaces the 30s consumer-start
# retry budget doubling as a provisioning wait, so the budget here is the actual
# metadata-convergence wait, not a worst-case consumer group join.
DEFAULT_READINESS_TIMEOUT_SECONDS: float = 30.0
DEFAULT_READINESS_POLL_INTERVAL_MS: int = 500
DEFAULT_READINESS_MAX_ATTEMPTS: int = 60
# Conservative default: per-contract order is the invariant, not global
# serialization. Leave the knob in place so a future boot-time regression can
# widen it without a redesign (§3.9).
DEFAULT_MAX_CONCURRENT_CONTRACT_ATTACH: int = 4


class ModelTopicReadinessConfig(BaseModel):
    """Bounded behavior knobs for the readiness confirm + interleave.

    Attributes:
        readiness_timeout_seconds: Total budget per contract's topic-set confirm.
        readiness_poll_interval_ms: Metadata poll cadence.
        max_attempts: Bounded retry count for the metadata poll.
        max_concurrent_contract_attach: Bounded parallelism across contracts
            (each contract still does provision->ready->attach in order).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    readiness_timeout_seconds: float = Field(
        default=DEFAULT_READINESS_TIMEOUT_SECONDS, gt=0.0
    )
    readiness_poll_interval_ms: int = Field(
        default=DEFAULT_READINESS_POLL_INTERVAL_MS, ge=1
    )
    max_attempts: int = Field(default=DEFAULT_READINESS_MAX_ATTEMPTS, ge=1)
    max_concurrent_contract_attach: int = Field(
        default=DEFAULT_MAX_CONCURRENT_CONTRACT_ATTACH, ge=1
    )


__all__: list[str] = [
    "DEFAULT_MAX_CONCURRENT_CONTRACT_ATTACH",
    "DEFAULT_READINESS_MAX_ATTEMPTS",
    "DEFAULT_READINESS_POLL_INTERVAL_MS",
    "DEFAULT_READINESS_TIMEOUT_SECONDS",
    "ModelTopicReadinessConfig",
]
