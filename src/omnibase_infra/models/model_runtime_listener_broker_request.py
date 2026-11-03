# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One request to the listener broker (OMN-22379).

Wire shape (the T1 runtime client and the T4/T5 wiring lanes reconcile here):
one JSON object per line (UTF-8, newline terminated) over the unix socket
`${DEV_RUNTIME_HANDOVER_DIR}/broker.sock`. A client line carries `op`
(`get`, `attach`, `confirmed`, `detach_request`), `pool` (`runtime-main` or
`runtime-effects`) and, on `attach`, `instance` (the runtime's
KAFKA_INSTANCE_ID). The session is the connection: the broker assigns
`session_id` itself, a client-supplied value is refused.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_runtime_listener_broker_op import (
    EnumRuntimeListenerBrokerOp,
)
from omnibase_infra.enums.enum_runtime_listener_pool import EnumRuntimeListenerPool


class ModelRuntimeListenerBrokerRequest(BaseModel):
    """A broker operation for one pool, on behalf of one session."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    op: EnumRuntimeListenerBrokerOp = Field(description="The operation requested.")
    pool: EnumRuntimeListenerPool = Field(description="The pool the request is for.")
    instance: str = Field(default="", description="Runtime instance id, on attach.")
    session_id: UUID | None = Field(
        default=None,
        description="Set by the broker from the connection, never by a client.",
    )


__all__: list[str] = ["ModelRuntimeListenerBrokerRequest"]
