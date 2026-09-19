# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Where one ``onex delegate`` run was addressed, for the files it writes.

OMN-18810. ``run.json`` and ``receipt.json`` are the two artifacts a customer
reads, and until now neither recorded the transport, the lane, or the locus.
The only key that looked like it did was ``run.json``'s ``lane``, which held
the accepted attempt's ROUTING TIER — so a run dispatched to the ``dev`` lane
wrote ``lane: "local"``, and a lane reading it reported the flags as silently
ignored against a path that had honoured every one of them.

The confusion is a naming one, and it is the same one
:mod:`omnibase_core.runtime.runtime_local` already went out of its way to
avoid when it named ``handler_locus`` rather than ``execution_locus``: a
receipt whose words for two different questions are the same word "is how a
receipt ends up asserting a lane it never touched". This model carries the
four addressing facts under names that answer exactly one question each, so
the tier can go back to being spelled ``routing_tier`` in both files.

Resolved by the CLI before dispatch, from the same values that decide whether
handlers are hosted — not reconstructed afterwards from the terminal, which
would only ever restate what the run already assumed.

.. versionadded:: OMN-18810
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus

__all__ = ["ModelDelegateRunAddressing"]


class ModelDelegateRunAddressing(BaseModel):
    """The transport, lane and locus one delegation run resolved to."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    locus: EnumDelegateLocus = Field(
        ...,
        description=(
            "Resolved locus — ``in-process`` when this CLI hosted the "
            "orchestrator, ``deployed-lane`` when it published the command "
            "and hosted nothing. Never ``auto``: resolution always yields a "
            "concrete choice, and this is the value that decided whether "
            "handlers were hosted at all."
        ),
    )
    bus: str = Field(
        ...,
        description=(
            "Resolved event-bus transport (``kafka`` / ``inmemory``). The "
            "TRANSPORT only — it does not say where the work ran, which is "
            "what ``locus`` is for and what conflating the two broke."
        ),
    )
    lane: str | None = Field(
        default=None,
        description=(
            "The lane this delegation was addressed to, as declared in the "
            "checked-in lane declaration — e.g. ``dev``. ``None`` when no "
            "lane was selected, which is every in-process run and any run "
            "stating its broker directly with ``--kafka-bootstrap``. This is "
            "the ``--lane`` flag's own value and nothing else; the accepted "
            "rung's tier is ``routing_tier``."
        ),
    )
    dispatch_target: str | None = Field(
        default=None,
        description=(
            "``'<command topic> via <broker>'`` for a dispatched run — the "
            "topic the command was published to and the broker it went to, "
            "both taken from the locus decision that was proven viable "
            "before the publish. ``None`` for an in-process run, which "
            "dispatches to nobody."
        ),
    )

    def as_run_file_fields(self) -> dict[str, str | None]:
        """The four keys, as they appear in ``run.json`` and ``receipt.json``.

        One spelling for both files. Two writers that each built their own
        dict is how the tier came to be called ``lane`` in one file and
        ``routing_tier`` in the other while holding the same value.
        """
        return {
            "locus": self.locus.value,
            "bus": self.bus,
            "lane": self.lane,
            "dispatch_target": self.dispatch_target,
        }
