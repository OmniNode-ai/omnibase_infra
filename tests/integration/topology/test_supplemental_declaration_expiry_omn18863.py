# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Interim supplemental declarations must expire, loudly (OMN-18863).

A supplemental entry in ``LEGACY_MIGRATION_TABLE_DECLARATIONS`` exists to close
the window between the two halves of a deliberate infra-first cross-repo
ordering: this repo vendors a relation's migrations first, because omnimarket's
node-migration-vendor-parity gate refuses the producing pull request until the
vendored counterpart is on this repo's ``dev``. During that window no pinned
contract declares the relation, so the derivation cannot reproduce the shipped
grant and the enforcement gate reds every open pull request at once.

The entry closes the window. Nothing closes the ENTRY.

**A stale supplemental entry is silent by construction.** The derivation unions
by relation, so once the pinned contract declares the same table the entry
contributes byte-identical output: nothing fails, nothing warns, and the only
trace is a hand-maintained declaration that no longer has a reason to exist.
Measured 2026-09-20 on the tuple this test guards: five of its eight entries
were already redundant against the pin, the oldest for weeks. That is the same
silent-success shape this ticket's sibling defect has on the deploy agent's
terminal event -- a surface reporting "fine" because it cannot tell "correct"
from "never checked".

So the expiry is asserted rather than remembered. The moment the pinned
contract set declares one of these relations, this test goes RED and names the
entry to delete, which lands in the pin-advance that made it redundant instead
of accumulating.

Scoped to the two entries this ticket added, deliberately. The other five are a
real cleanup with a real risk of deleting something still load-bearing, and
adopting them here under a red-dev fix would be exactly the unreviewed widening
this repo's gates exist to refuse. They are named in the ticket instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
    load_contract_declarations,
)

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[3]
# The cross-repo checkout the OMN-15361 enforcement job provides, resolved at
# the committed pin. Absent in a bare local run, which is why every test here
# skips rather than passing vacuously -- a green that means "I could not look"
# is the failure mode this whole module is about.
_PINNED_CONTRACTS = (
    _REPO_ROOT / ".proof-dependencies" / "omnimarket" / "src" / ("omnimarket") / "nodes"
)

# The relations this ticket declared by hand, each with the omnimarket pull
# request whose merge plus pin advance retires it.
_INTERIM_ENTRIES: dict[str, str] = {
    "runtime_error_fingerprints": "omnimarket#2664",
    "lab_lane_health": "omnimarket#2674",
}

_SKIP_REASON = (
    "requires the pinned omnimarket checkout at .proof-dependencies/omnimarket, "
    "which the OMN-15361 enforcement job provides and a bare local run does not"
)


def _pinned_relation_names() -> set[str]:
    return {
        declaration.table.name
        for declaration in load_contract_declarations(_PINNED_CONTRACTS)
    }


def _supplemental_relation_names() -> set[str]:
    return {
        declaration.table.name for declaration in LEGACY_MIGRATION_TABLE_DECLARATIONS
    }


class TestTheInterimEntriesAreStillCarried:
    """The premise. Without these the expiry assertions below are about nothing."""

    @pytest.mark.parametrize("relation", sorted(_INTERIM_ENTRIES))
    def test_the_entry_exists(self, relation: str) -> None:
        assert relation in _supplemental_relation_names(), (
            f"{relation} is no longer declared in "
            "LEGACY_MIGRATION_TABLE_DECLARATIONS. If the pin now declares it, "
            "delete this parametrisation entry too -- the pair is retired "
            "together or not at all"
        )


@pytest.mark.skipif(not _PINNED_CONTRACTS.is_dir(), reason=_SKIP_REASON)
class TestEachEntryExpiresWhenThePinDeclaresIt:
    @pytest.mark.parametrize(
        ("relation", "source_pr"), sorted(_INTERIM_ENTRIES.items())
    )
    def test_the_pin_does_not_yet_declare_it(
        self, relation: str, source_pr: str
    ) -> None:
        """RED the moment the pin catches up. That redness is the whole point.

        This is not a check that the pin is behind. It is an instruction,
        delivered at the only moment anyone can act on it cheaply: the
        pin-advance commit that makes the hand entry redundant.
        """
        assert relation not in _pinned_relation_names(), (
            f"the pinned omnimarket contracts now declare {relation!r}, so the "
            "supplemental LEGACY_MIGRATION_TABLE_DECLARATIONS entry for it is "
            f"redundant and must be DELETED in this same change ({source_pr} "
            "has merged and the pin has advanced past it). Remove the entry "
            "from src/omnibase_infra/topology/table_grant_derivation.py, remove "
            "it from _INTERIM_ENTRIES here, and regenerate -- the instances "
            "must come back byte-identical, because the contract now derives "
            "what the entry used to"
        )

    def test_the_pin_is_readable_and_non_empty(self) -> None:
        """Positive control: prove the lookup can see contracts at all.

        Without this, a checkout that resolved to an empty tree would make
        every expiry assertion above pass for the wrong reason -- the exact
        vacuous-green this module exists to refuse.
        """
        pinned = _pinned_relation_names()
        assert len(pinned) > 50, (
            f"the pinned contract set yielded only {len(pinned)} relations; "
            "the expiry assertions above are vacuous against a tree this small"
        )
