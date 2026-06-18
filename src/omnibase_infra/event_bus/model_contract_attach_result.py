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
        topics_subscribed: Topics whose consumers were actually attached.
        readiness: The readiness confirm outcome for the contract's topics.
        detail: Human-readable detail for non-attached outcomes (no secrets).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    contract_name: str
    status: EnumContractAttachStatus
    topics_subscribed: tuple[str, ...] = Field(default_factory=tuple)
    readiness: ModelTopicSetReadiness | None = Field(default=None)
    detail: str = Field(default="")


__all__: list[str] = ["ModelContractAttachResult"]
