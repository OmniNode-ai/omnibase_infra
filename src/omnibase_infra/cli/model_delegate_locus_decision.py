# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Resolved execution-locus decision for one ``onex delegate`` run (OMN-17295).

Everything a reader needs to answer "which orchestrator resolved this run?"
without cross-checking container logs by hand: the locus, how it was decided,
the contract and the installed distribution the contract came from, and — for
a dispatched run — the broker, the exact command topic, and the live consumer
groups observed on that topic at the moment of dispatch.

The consumer-group list is the load-bearing field. A version string is a
label; a ``STABLE`` group bound to the topic the command is about to land on
is a process that was joined and consuming when this run was admitted.

.. versionadded:: OMN-17304
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus

__all__ = ["ModelDelegateLocusDecision"]


class ModelDelegateLocusDecision(BaseModel):
    """The locus this delegation resolved to, and the evidence behind it."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    locus: EnumDelegateLocus = Field(
        ...,
        description=(
            "Resolved locus. Never AUTO — resolution always produces a "
            "concrete choice, so a receipt cannot report an unmade decision."
        ),
    )
    resolved_from: str = Field(
        ...,
        description=(
            "How the locus was decided — the resolved transport, or an "
            "explicit --locus naming itself as an override. A receipt that "
            "cannot distinguish 'the authority chose this' from 'a human "
            "typed it' is the same class of instrument defect as OMN-17295."
        ),
    )
    orchestrator_contract: str = Field(
        ...,
        description="Filesystem path of the contract that was dispatched.",
    )
    orchestrator_distribution: str = Field(
        ...,
        description=(
            "Installed distribution the contract resolved out of, as "
            "'<name> <version> (<location>)'. For an in-process run this IS "
            "the orchestrator that decided; for a dispatched run it is only "
            "the source of the wire description, and the deciding code lives "
            "in whatever consumed the command topic."
        ),
    )
    command_topic: str = Field(
        ...,
        description=(
            "Topic the typed command is published to, read from the "
            "contract's event_bus.subscribe_topics — never a constant in this "
            "file."
        ),
    )
    broker: str = Field(
        ...,
        description=(
            "Broker the command is published through. Empty for an in-process "
            "run on the in-memory bus, which involves no broker at all."
        ),
    )
    lane_consumer_groups: tuple[str, ...] = Field(
        ...,
        description=(
            "Consumer groups observed STABLE on command_topic at dispatch "
            "time. Empty for an in-process run. Non-empty is the fail-closed "
            "precondition for a dispatched run: with nothing bound there is "
            "no deployed orchestrator to decide."
        ),
    )
