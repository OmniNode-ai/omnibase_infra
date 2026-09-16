# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""One hop of the chain the node contracts DECLARE must happen (OMN-18419).

Why this model exists at all
----------------------------
``chain_topology`` was an ordered list of topic strings. An ordered list can
say "these hops, in this order"; it cannot say "this hop was caused by that
one". The replay therefore had no declared parent to check against and
re-derived each hop's expected parent from whatever sat at ``index - 1``.

That re-derivation is correct only for a LINE, and the delegation chain is a
TREE. Measured read-only on the .201 compose dev lane, correlation
``41235987-425c-481b-b2e3-8970083ce512`` (chain-canary run 35037024216):
``onex.evt.omnimarket.delegate-skill-completed.v1`` records its parent as the
``onex.cmd.omnimarket.delegate-skill.v1`` envelope, not as the routing
decision that preceded it in time — because the terminal is a consequence of
consuming the delegate-skill COMMAND, not of the routing decision. The
recorded edge was right and the re-derivation was wrong, so a correct chain
graded red.

The fix is to make the declaration say what it means. Each hop names the
declared topic of the hop whose consumption caused it, and the replay grades
the recorded edge against THAT hop's observed envelope id. A branch is then
expressible, and nothing about this canary is special-cased: a line is simply
a tree in which every parent happens to be the preceding hop.

``parent = None`` is the checkable statement "this hop is the chain HEAD",
the same claim an absent ``parent_message_id`` makes on the wire
(``envelope_header_identity``). It is a claim, not a gap: the replay refuses a
declared head that records a parent.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelDeclaredChainHop(BaseModel):
    """A declared hop: the topic, and the declared topic that causes it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    topic: str = Field(
        min_length=1,
        description="The topic this hop is declared to be observed on.",
    )
    parent: str | None = Field(
        default=None,
        description=(
            "The DECLARED TOPIC of the hop whose consumption causes this one. "
            "None means this hop is declared to be the chain head, which the "
            "replay checks rather than assumes: a declared head that records "
            "a parent envelope is a replay failure, not a tolerated extra."
        ),
    )

    @model_validator(mode="after")
    def _refuse_self_causation(self) -> ModelDeclaredChainHop:
        """A hop cannot declare itself as its own cause.

        The envelope model already refuses a self-edge on the wire
        (``parent_envelope_id == envelope_id``). Refusing it in the
        DECLARATION too means a contract can never ask the replay to check an
        edge the transport would reject.
        """
        if self.parent is not None and self.parent == self.topic:
            raise ValueError(
                f"declared hop {self.topic!r} names itself as its own parent; "
                "a self-causing hop is not a chain"
            )
        return self


__all__ = ["ModelDeclaredChainHop"]
