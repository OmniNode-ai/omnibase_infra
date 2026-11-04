# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One answer of the listener broker (OMN-22379).

Wire shape: one JSON line, `ok` (bool), `op` (echo), `pool` (echo), and
either `error` (a short code: `not_installed`, `no_session`, `not_confirmed`,
`bad_request`) or the result: `state` (`provisional` or `confirmed`) on
`attach`/`confirmed`, `verdict` (`granted`, `wait`, `alone`) on
`detach_request`. The answer to `get` is this line with the listening
socket attached as one SCM_RIGHTS descriptor.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelRuntimeListenerBrokerResponse(BaseModel):
    """The broker's answer to one request."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    ok: bool = Field(description="Whether the request was served.")
    op: str = Field(description="The operation answered.")
    pool: str = Field(default="", description="The pool answered for.")
    error: str = Field(default="", description="Refusal code; empty when ok.")
    state: str = Field(default="", description="Session state after the request.")
    verdict: str = Field(default="", description="Detach verdict, on detach_request.")
    session_id: UUID | None = Field(default=None, description="The session, on attach.")


__all__: list[str] = ["ModelRuntimeListenerBrokerResponse"]
