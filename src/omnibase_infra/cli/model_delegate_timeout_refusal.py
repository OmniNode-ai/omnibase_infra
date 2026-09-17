# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The typed refusal a delegation that never terminalized returns (OMN-17516).

``onex delegate`` documents one contract on stdout: exactly ONE
``ModelSkillResult`` JSON, always. Every terminal outcome honours it — a
completed run, a failed rung, an unattributed run, a run whose customer
artifacts could not be written. The hard-timeout backstop (OMN-14397) did not:
it wrote prose to stderr, returned exit 1, and left **stdout empty**.

That empty stdout is the whole of the OMN-17516 report. The 2026-08-26
observation — *"hung past 120 seconds and returned no result at all. Not a
wrong result, not a typed timeout surfaced to the caller: no terminal"* — is a
description of this path. A caller parsing the documented contract received
nothing, and nothing is exactly what a run that is still working also looks
like, so there was no way to tell "will never answer" from "not answered yet",
nothing to retry against, and nothing to report. A hang with no typed result is
the worst available failure shape precisely because it is indistinguishable
from patience.

**What this model does NOT do, stated so the next reader does not have to
re-derive it.** It does not make a missing terminal arrive, does not extend or
invent a timeout, and does not paper over the absence with a synthesized
success. The run genuinely did not terminalize; this is the honest report of
that, and the ``status`` on the receipt carrying it is
:attr:`~omnibase_core.enums.enum_skill_result_status.EnumSkillResultStatus.FAILED`
with a non-zero exit code.

**Why the fields are what they are.** AC4 asks for a refusal "naming what was
waited on and for how long". Both halves are load-bearing and neither is
inferable from the other:

* *what was waited on* — the delegation terminal, on a named ``terminal_topic``,
  over a named ``bus`` at a named ``locus`` (and, on a shared bus, a named
  ``broker``). The 2026-08-26 diagnosis cost days precisely because none of
  that was on the record, so "which transport did the failing run actually
  resolve?" became OMN-17516's own AC1 rather than something the failure said
  for itself.
* *for how long* — the ``declared_timeout_seconds`` the caller asked for, the
  ``grace_seconds`` added before the backstop fires, and the
  ``elapsed_seconds`` actually spent. The three differ in practice: a
  ``SIGALRM`` raised inside a non-cooperative blocking call only propagates at
  the next bytecode boundary, so elapsed routinely exceeds declared + grace.
  Reporting only the declared bound would understate the wait the caller
  actually served, which is the number a human is trying to explain.

``correlation_id`` is on the refusal as well as on its carrying receipt because
a refusal is read in logs where it has been separated from its envelope, and
two concurrent hung runs are otherwise indistinguishable — the OMN-17295 defect
in the failure direction.

.. versionadded:: OMN-17516
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelDelegateTimeoutRefusal"]


class ModelDelegateTimeoutRefusal(BaseModel):
    """A delegation whose terminal did not arrive inside the caller's bound."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    awaited: Literal["delegation_terminal"] = Field(
        default="delegation_terminal",
        description=(
            "What never arrived. A literal rather than free text: the set of "
            "things this command waits on is closed, and a reader filtering "
            "logs should not have to match prose."
        ),
    )
    reason: Literal["hard_timeout_backstop"] = Field(
        default="hard_timeout_backstop",
        description=(
            "Which bound stopped the run. 'hard_timeout_backstop' is the "
            "CLI-level SIGALRM guard (OMN-14397), which fires only when the "
            "runtime's own cooperative asyncio.wait_for did NOT preempt first "
            "— so seeing this value says the hang was in non-cooperative "
            "blocking work, not in the terminal wait itself."
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
    declared_timeout_seconds: int = Field(
        ...,
        ge=1,
        description="The --timeout the caller asked for, in seconds.",
    )
    grace_seconds: int = Field(
        ...,
        ge=0,
        description=(
            "Seconds added to the declared timeout before the backstop fires, "
            "giving the runtime's cooperative timeout the first chance to exit "
            "cleanly with a receipt of its own."
        ),
    )
    elapsed_seconds: float = Field(
        ...,
        ge=0.0,
        description=(
            "Wall-clock seconds actually spent. Exceeds declared + grace "
            "whenever the SIGALRM landed inside a blocking call, which is the "
            "case this backstop exists for; reported rather than assumed."
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
            "Where the orchestrator was to run: in-process here, or a deployed "
            "lane. Which one it was decides where a human looks next, and it "
            "is a resolved property of the run (OMN-17295/OMN-17304), never "
            "inferable from the command line alone."
        ),
    )
    terminal_topic: str = Field(
        default="",
        description=(
            "The topic the delegation terminal was to arrive on, read from the "
            "contract. Empty when the contract declares none — reported as "
            "empty rather than omitted, because 'the contract names no "
            "terminal topic' is itself a finding a reader wants."
        ),
    )
    command_topic: str = Field(
        default="",
        description=(
            "The topic the command was published to. Empty on an in-process "
            "run, where nothing was published to a shared topic at all."
        ),
    )
    broker: str = Field(
        default="",
        description=(
            "The broker address the run was bound to, or empty for an "
            "in-process bus that has none."
        ),
    )
