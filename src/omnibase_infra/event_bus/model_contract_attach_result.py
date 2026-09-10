# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Per-contract boot interleave attach result (OMN-13237).

Related Tickets:
    - OMN-13237: Per-contract scoped topic provisioning at runtime boot.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.event_bus.enum_contract_attach_status import (
    EnumContractAttachStatus,
)
from omnibase_infra.event_bus.model_topic_set_readiness import (
    ModelTopicSetReadiness,
)


class ModelContractAttachResult(BaseModel):
    """Per-contract result of the boot interleave (provision->ready->attach).

    Attributes:
        contract_name: The wired contract's node name.
        status: Whether the contract's consumer attached, was skipped as
            not-ready, or failed during attach.
        dispatcher_ids: Exact dispatcher scope owned by the contract's
            subscription callbacks. Preserved on every non-attached result so a
            later reconciliation attempt cannot fall back to process-global
            fan-out.
        topics_subscribed: Topics whose consumers were actually attached.
        readiness: The readiness confirm outcome for the contract's topics.
        detail: Human-readable detail for non-attached outcomes (no secrets).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    contract_name: str
    status: EnumContractAttachStatus
    dispatcher_ids: tuple[str, ...] = Field(default_factory=tuple)
    topics_subscribed: tuple[str, ...] = Field(default_factory=tuple)
    readiness: ModelTopicSetReadiness | None = Field(default=None)
    detail: str = Field(default="")

    @property
    def needs_reattach(self) -> bool:
        """Whether this contract has no live consumer and must be re-attempted.

        OMN-18110. The ONE definition of the boot reconciliation's input set,
        read by both the kernel that selects it and the wiring seam that
        re-validates it, so the two cannot disagree about which contracts get
        retried.

        The predicate is "did NOT attach", never "is NOT_READY". Both
        non-attached statuses leave the contract with zero consumer groups and
        both are re-attemptable by the same idempotent
        provision -> confirm-ready -> attach interleave; the only thing that
        distinguishes them is how far the boot got before giving up. Filtering
        on ``NOT_READY`` alone meant a contract whose Kafka group-join timed
        out after readiness PASSED was recorded once and never revisited, so a
        transient broker blip stranded it for the whole process lifetime
        (live: four contracts on the ``.201`` dev lane, boot
        2026-09-10T00:03:45Z, each ``status=failed detail=InfraTimeoutError``
        over a ``readiness.status=ready`` with no failures).

        Written as the negation of ATTACHED rather than as a list of the
        retryable statuses so a status added later is retried by default: a
        new non-attached outcome that is silently NOT retried reproduces this
        defect, while one that is retried costs at most a bounded, idempotent
        re-attempt.
        """
        return self.status is not EnumContractAttachStatus.ATTACHED


__all__: list[str] = ["ModelContractAttachResult"]
