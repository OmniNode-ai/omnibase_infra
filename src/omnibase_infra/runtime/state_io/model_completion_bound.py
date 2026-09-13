# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-declared completion bound for a ``state_io`` workflow (OMN-18296).

A ``state_io`` contract persists per-request FSM state so a workflow survives a
cold-process replay. It did not, until this module, declare how long that
workflow is allowed to stay non-terminal, nor what the runtime owes the caller
when the process that owned an in-flight leg goes away and the leg is simply
lost.

That gap was measured, not theorised. On 2026-09-13 the lab lane's
``omninode-runtime-effects`` pod was recreated at 09:57:40Z while delegation
correlation ``a2fe0848-4b4b-462e-b633-c5f9559afee5`` (submitted 09:55:04Z) had
an inference command in flight. The command's consumer offset was already
committed, so the record was never redelivered; no inference response, success
or failure, was ever published. The FSM row stayed ``ROUTED`` with
``in_flight = TRUE`` indefinitely, and because the gateway row can only leave
``published`` when a REAL terminal event is consumed, the customer's delegation
never terminalised and the client polled with nothing to find.

Two numbers governed that run and neither was declared anywhere a reader could
find them: the runtime's give-up TTL was an environment-variable default inside
``state_store_adapter`` and the client's patience was a hardcoded CLI default,
unrelated to it. This block makes the bound one contract-declared fact that the
runtime enforces and the client reads.

Shape::

    completion_bound:
      max_wall_seconds: 900
      on_runtime_restart: terminalise_failed
      failure_class: runtime_restart_during_delegation
      failure_code: ONEX_MARKET_DELEGATION_RUNTIME_RESTART

``failure_class`` is the node's own failure vocabulary token (for delegation,
a member of ``EnumDelegationFailureClass``); this package does not own that
vocabulary and deliberately does not enumerate it here. ``failure_code`` is the
optional canonical ``ONEX_…`` token carried alongside it on the wire.
"""

from __future__ import annotations

import re
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.enums.enum_runtime_restart_policy import (
    EnumRuntimeRestartPolicy,
)

_FAILURE_CODE_PATTERN: Final[re.Pattern[str]] = re.compile(r"^ONEX_[A-Z0-9_]+$")


class ModelCompletionBound(BaseModel):
    """The wall-clock bound a ``state_io`` workflow must terminalise within."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_wall_seconds: int = Field(
        ...,
        gt=0,
        description=(
            "Wall-clock seconds from the row's last advance after which a "
            "still-in-flight, non-terminal workflow is given up on. Measured "
            "from updated_at, so a workflow that is still making progress is "
            "never swept."
        ),
    )
    on_runtime_restart: EnumRuntimeRestartPolicy = Field(
        ...,
        description="The runtime's recovery behaviour for an abandoned row.",
    )
    failure_class: str = Field(
        ...,
        min_length=1,
        description=(
            "The node's own failure-vocabulary token recorded on the terminal "
            "event, naming why the workflow was given up on."
        ),
    )
    failure_code: str | None = Field(
        default=None,
        description=(
            "Optional canonical ONEX_ token carried with the failure class on "
            "the wire. Absent is reported as absent, never invented."
        ),
    )

    @field_validator("failure_code")
    @classmethod
    def validate_canonical_code(cls, value: str | None) -> str | None:
        """Reject a failure code that is not a canonical ONEX_ token.

        Fails at wiring time rather than at emit time: a malformed code would
        otherwise reach the gateway's attribution grammar, match nothing, and
        be reported as carrying no class at all — a silent downgrade of the one
        fact this terminal exists to carry.
        """
        if value is not None and not _FAILURE_CODE_PATTERN.match(value):
            msg = (
                f"completion_bound.failure_code {value!r} is not a canonical "
                "ONEX_ token (^ONEX_[A-Z0-9_]+$)"
            )
            raise ValueError(msg)
        return value


__all__: list[str] = ["ModelCompletionBound"]
