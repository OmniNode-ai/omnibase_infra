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

``alternatives`` and why a sixth hop was the wrong answer (OMN-18937)
---------------------------------------------------------------------
A delegation has TWO terminals -- ``delegate-skill-completed.v1`` and
``delegate-skill-failed.v1`` -- and exactly one of them is observed per
chain. That is ONE hop with two possible topics, not two hops.

Declaring the failure terminal as a sixth hop was considered and rejected on
two independent grounds. Tier 2 grades POSITIONALLY: on a failed delegation
the terminal is observed at index 4, so a sixth declared entry puts
``...-completed`` at ``declared_chain[4]`` and the terminal row grades FAIL --
and the canary reads the whole chain's tier-2 verdict from the LAST row. And
anything that counts declared hops (the chain-canary workflow's
``EXPECTED_LEDGER_HOPS``) would read five-of-six on every successful run
forever. A check that cannot pass is worse than one that cannot fail.

So the hop stays one entry and gains ``alternatives``. ``topic`` remains the
hop's CANONICAL name -- the name a ``parent`` citation resolves against and
the name position-counting reads -- and ``alternatives`` names the other
topics the same hop may legitimately be observed on. Five entries stay five.
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
    alternatives: tuple[str, ...] = Field(
        default=(),
        description=(
            "Other topics THIS SAME hop may be observed on. Exactly one of "
            "`topic` and `alternatives` is observed per chain. `topic` stays "
            "the canonical name: a `parent` citation resolves against it, and "
            "the positional tier-2 grade counts entries, not aliases."
        ),
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

    @property
    def topics(self) -> tuple[str, ...]:
        """Every topic this hop may be observed on, canonical name first.

        One accessor rather than two call-site unions, so no reader can hold
        a narrower idea of what this hop IS than another reader does.
        """
        return (self.topic, *self.alternatives)

    @model_validator(mode="after")
    def _refuse_self_causation(self) -> ModelDeclaredChainHop:
        """A hop cannot declare itself as its own cause, by any of its names.

        The envelope model already refuses a self-edge on the wire
        (``parent_envelope_id == envelope_id``). Refusing it in the
        DECLARATION too means a contract can never ask the replay to check an
        edge the transport would reject. ``alternatives`` widens what "itself"
        means: a parent citing an alias of this hop is the same self-edge
        spelled differently (OMN-18937).
        """
        if self.parent is not None and self.parent in self.topics:
            raise ValueError(
                f"declared hop {self.topic!r} names itself as its own parent "
                f"(as {self.parent!r}); a self-causing hop is not a chain"
            )
        if any(not alternative for alternative in self.alternatives):
            raise ValueError(
                f"declared hop {self.topic!r} carries an empty alternative "
                "topic; an unnamed alias cannot be matched against anything"
            )
        if self.topic in self.alternatives:
            raise ValueError(
                f"declared hop {self.topic!r} repeats its own canonical topic "
                "in `alternatives`; the canonical name is already accepted"
            )
        if len(set(self.alternatives)) != len(self.alternatives):
            raise ValueError(
                f"declared hop {self.topic!r} repeats an alternative topic; a "
                "duplicated alias cannot say which occurrence a match means"
            )
        return self


__all__ = ["ModelDeclaredChainHop"]
