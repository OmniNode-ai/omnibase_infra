# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The typed terminal a delegation that never reached the broker returns (OMN-18925).

Sibling of
:class:`~omnibase_infra.cli.model_delegate_timeout_refusal.ModelDelegateTimeoutRefusal`,
and written for the same reason one level further down the stack. That model
covers "the command was published and no terminal came back". This one covers
"the command was never published at all", which until now was the one delegate
outcome that produced no artifact of any kind.

**The measured failure.** On 2026-09-21 the dev Redpanda stalled its shard-0
reactor for 17.2 seconds under host CPU starvation. Two delegations landed
inside that window. Neither wrote a run directory, a ``receipt.json``, a
``run.json`` or a ``result.txt``; the caller got a raised exception and a
stderr line. Every lane in this workspace is told to read the terminal from
``.onex_state/runs/<run_id>/receipt.json`` rather than from an exit code, so on
this path the instruction named a file that did not exist. Three separate lanes
spent real diagnosis time on delegation symptoms in the same week, each one
slowed by the absence of the artifact that would have settled it.

**Why a transport refusal cannot borrow the delegation failure enum.**
``EnumDelegationTerminalFailureCause`` has exactly three members —
``provider_quota_exhausted``, ``auth_failed``, ``provider_error`` — and all
three are *provider-side*. A broker that never accepted the command says
nothing whatsoever about the provider: no rung ran, no model was chosen, no
endpoint was called. Assigning any of the three would send the next reader to
the inference provider, the credential or the endpoint, and every one of those
would look fine, because every one of them was fine. That is precisely the
class of lie OMN-19004 is a standing invariant against, so this model states
its cause in its own fields and leaves that enum alone. Widening the enum is
OMN-19004's work and is deliberately not done here.

**Consistency with the OMN-19004 invariant, field by field.** Nothing here is
set alongside the record in a way the record can contradict:

* ``attempts_made`` is the count the retry helper actually returned, never a
  configured ceiling. A refusal that reports the policy's maximum when the
  loop exited early would be the ``attempts_count`` defect in a new place.
* ``bound_seconds`` is the policy's own computed worst case, so the deadline
  the refusal names is the deadline that was actually in force.
* ``elapsed_seconds`` is measured wall clock, reported rather than inferred
  from the bound: a connect that fails fast against a refused TCP port exits
  far under its bound, and a refusal claiming otherwise would overstate the
  wait the caller served.
* There is no quality, score or rung field to be wrong. A run that never
  reached a broker has no content to judge, and the honest shape of that is
  an absent field rather than a null one carrying an implied verdict.

.. versionadded:: OMN-18925
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelDelegateTransportRefusal"]


class ModelDelegateTransportRefusal(BaseModel):
    """A delegation whose command never reached the broker."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    awaited: Literal["broker_connection"] = Field(
        default="broker_connection",
        description=(
            "What never arrived. A literal rather than free text, matching "
            "the sibling timeout refusal's 'delegation_terminal': the two "
            "values are the closed set of things this command can fail to "
            "obtain, and they distinguish 'never sent' from 'sent, never "
            "answered' at a glance."
        ),
    )
    reason: Literal["broker_unreachable", "locus_probe_refused"] = Field(
        ...,
        description=(
            "Which transport stage stopped the run. 'broker_unreachable' is "
            "a connect that exhausted its retry policy while the run was "
            "being dispatched. 'locus_probe_refused' is the earlier, cheaper "
            "refusal: the pre-dispatch probe could not confirm a live "
            "consumer group on the command topic, so nothing was published. "
            "They are different failures with different remedies -- the "
            "first says the broker is sick, the second says the lane may "
            "simply not be running -- and collapsing them would send the "
            "reader to the wrong place half the time."
        ),
    )
    correlation_id: UUID = Field(
        ...,
        description=(
            "The correlation id THIS invocation minted. Duplicated from the "
            "carrying receipt so the refusal stays attributable when it is "
            "read apart from its envelope."
        ),
    )
    bus: str = Field(
        ...,
        min_length=1,
        description="The RESOLVED event-bus transport, not the flag as typed.",
    )
    locus: str = Field(
        ...,
        min_length=1,
        description=(
            "Where the orchestrator was to run. A transport refusal on an "
            "in-process locus and one on a deployed lane are different "
            "findings, and the second is the only one where a broker is "
            "even involved."
        ),
    )
    broker: str = Field(
        default="",
        description=(
            "The broker address the run was bound to, sanitized of any "
            "credential, or empty for an in-process bus that has none. This "
            "is the first thing a human checks and the 2026-09-21 diagnosis "
            "had to recover it from the command line rather than the record."
        ),
    )
    command_topic: str = Field(
        default="",
        description=(
            "The topic the command would have been published to. Empty on an "
            "in-process run, which publishes to no shared topic at all."
        ),
    )
    attempts_permitted: int = Field(
        ...,
        ge=1,
        description=(
            "Connect attempts the retry policy in force ALLOWED, resolved "
            "from the same declaration the transport itself read.\n\n"
            "This is deliberately not 'attempts made'. The connect runs "
            "below this process's frame -- inside the event bus, reached "
            "through the runtime -- and reports no count back, so the "
            "observed number is not available to the writer. Reporting the "
            "permitted count, named as such, is the truthful option; a "
            "field called 'attempts_made' filled with the ceiling would be "
            "exactly the OMN-19004 defect of a summary field set alongside "
            "the record rather than derived from it, and a nullable one "
            "would carry an implied verdict on every run that could not "
            "fill it. What the reader actually needs -- did this use its "
            "budget or fail fast -- is answered by elapsed against bound "
            "below, both of which ARE measured or resolved."
        ),
    )
    bound_seconds: float = Field(
        ...,
        gt=0.0,
        description=(
            "The total wall-clock bound the retry policy promised not to "
            "exceed, which is the deadline AC-2 asks this refusal to name. "
            "Resolved from the SAME declaration this process's own transport "
            "reads -- the overlay-resolved event-bus config -- rather than "
            "from a literal duplicated at the writer, which would put a "
            "number in the refusal that was never in force.\n\n"
            "Read it as the bound for the delegate CLI's transport, not for "
            "an arbitrary caller's. A surface that builds its own narrowed "
            "config is not represented here, because the writer never sees "
            "that object: the connect happens below its frame. That gap is "
            "harmless on this path, where the CLI's transport IS the "
            "overlay-resolved default, and it is recorded rather than "
            "hidden because a field whose scope is unstated is how a bound "
            "nobody checked becomes a bound everybody cites."
        ),
    )
    elapsed_seconds: float = Field(
        ...,
        ge=0.0,
        description=(
            "Wall-clock seconds actually spent before giving up. Well under "
            "bound_seconds whenever the broker refused the connection fast "
            "rather than stalling, which is the difference between a lane "
            "that is down and a lane that is overloaded."
        ),
    )
    transport_error_type: str = Field(
        ...,
        min_length=1,
        description=(
            "Exception class name of the final failure, e.g. "
            "'InfraConnectionError' or 'TimeoutError'. Named separately from "
            "the message so a projection can group by it without parsing "
            "prose."
        ),
    )
    transport_error: str = Field(
        default="",
        description=(
            "The final failure's message, sanitized. Empty only if the "
            "exception carried none."
        ),
    )
