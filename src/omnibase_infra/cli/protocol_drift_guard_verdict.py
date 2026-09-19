# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""What every omnimarket drift-guard verdict owes its two consumers
(OMN-18814).

The guard can hand back more than one KIND of verdict -- the off-registry
check (OMN-17255) on a machine with no canonical clone, and the ancestor-lag
stamp (OMN-18814) on a registry machine running behind the tip -- and the
number of kinds is not fixed. What does not vary is the contract: every
verdict renders to one structured stderr line for the run, and to one receipt
fragment that outlives the terminal, from the same object, so the two cannot
disagree.

This protocol names that contract so callers can be annotated by what they
need rather than by an enumeration of what currently satisfies it. The
alternative -- widening three signatures to a union each time a verdict kind
is added -- makes every consumer a place the next author has to remember to
edit, and the compiler only catches the ones that break.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["ProtocolDriftGuardVerdict"]


@runtime_checkable
class ProtocolDriftGuardVerdict(Protocol):
    """One drift-guard verdict, renderable for a terminal and for a receipt."""

    @property
    def line(self) -> str:
        """The one structured line for stderr, stating this verdict."""
        ...

    def as_receipt_fields(self) -> dict[str, object]:
        """The same facts, JSON-serializable, for a run receipt."""
        ...
