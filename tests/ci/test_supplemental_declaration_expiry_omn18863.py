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

The entry closes the window. Nothing closed the ENTRY, until this.

**A stale supplemental entry is silent by construction.** The derivation unions
by relation, so once the pinned contract declares the same table the entry
contributes byte-identical output: nothing fails, nothing warns, and the only
trace is a hand-maintained declaration that no longer has a reason to exist.
Measured across one night: five of the tuple's eight entries were already
redundant at 01:55Z, six by 02:46Z when a pin advance expired the newest one
thirty minutes after it merged. Nobody was careless. There was no signal.

So the expiry is asserted rather than remembered. It has now fired twice on
real handovers rather than synthetic ones -- ``runtime_error_fingerprints`` on
the pin advance to ``ac35d56338b3``, and ``lab_lane_health`` on the advance to
``e1c4c8f61a1f``, each time naming the entry on the bot's own pull request so
the deletion rides the commit that made it redundant.

**Where this runs, stated rather than implied.** It needs the pinned contract
set, which only the ``Application Database Domain Enforcement (OMN-15361)`` job
checks out, so it lives here in ``tests/ci/`` and is named in that job's own
pytest list. It is therefore real coverage on every pull request and no
coverage at all in the ordinary test splits, where it is recorded in
``config/skip_count_baseline.yaml`` with provenance. That is a conditional
coverage and the honest limit is that one workflow edit could silence it; the
alternative considered was answering the question from in-repo artifacts alone,
and nothing committed here encodes whether a relation is contract-declared.

**Adding an entry is one line.** The assertions loop over the map rather than
parametrising on it, so a new bridge costs a map entry and moves no node ids
and no baseline number. That is deliberate: the previous shape charged a
baseline edit for every bridge, which is friction pointed at exactly the person
doing the right thing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
    load_contract_declarations,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PINNED_CONTRACTS = (
    _REPO_ROOT / ".proof-dependencies" / "omnimarket" / "src" / "omnimarket" / "nodes"
)

# Relations this repo declares by hand during an infra-first window, each with
# the omnimarket pull request whose merge plus pin advance retires it.
#
# EMPTY IS THE GOAL STATE, not a gap. Both entries this ticket added have been
# retired by the mechanism below. A relation belongs here only while it is
# declared in the shipped topology instances and derivable from no pinned
# contract; add it in the same pull request that vendors it, and this module
# will tell you when to take it out.
_INTERIM_ENTRIES: dict[str, str] = {}

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


class TestTheMapMatchesTheManifest:
    """Runs everywhere. Needs no foreign tree, so it is never skipped."""

    def test_every_mapped_relation_is_actually_carried(self) -> None:
        """A map entry for a relation nobody declares asserts nothing.

        This is the premise the expiry assertion rests on: if an entry were
        removed from the manifest but left in the map, the expiry check would
        keep watching a relation this repo no longer bridges, and would read
        green for the wrong reason.
        """
        orphaned = sorted(set(_INTERIM_ENTRIES) - _supplemental_relation_names())
        assert not orphaned, (
            f"{orphaned} appear in _INTERIM_ENTRIES but in no "
            "LEGACY_MIGRATION_TABLE_DECLARATIONS entry. Either the bridge was "
            "deleted and the map was not, or the name is misspelled"
        )


@pytest.mark.skipif(not _PINNED_CONTRACTS.is_dir(), reason=_SKIP_REASON)
class TestEachEntryExpiresWhenThePinDeclaresIt:
    def test_no_interim_entry_is_redundant(self) -> None:
        """RED the moment the pin catches up. That redness is the whole point.

        This is not a check that the pin is behind. It is an instruction,
        delivered at the only moment anyone can act on it cheaply: the
        pin-advance commit that makes the hand entry redundant. It has fired
        twice on real advances, and both times the deletion landed on the bot's
        own pull request.

        Vacuous when the map is empty, which is the goal state and is why the
        positive control below is not optional.
        """
        pinned = _pinned_relation_names()
        redundant = sorted(name for name in _INTERIM_ENTRIES if name in pinned)
        assert not redundant, (
            f"the pinned omnimarket contracts now declare {redundant}, so the "
            "supplemental LEGACY_MIGRATION_TABLE_DECLARATIONS entries for them "
            "are redundant and must be DELETED in this same change (source: "
            + ", ".join(f"{name} -> {_INTERIM_ENTRIES[name]}" for name in redundant)
            + "). Remove each entry from "
            "src/omnibase_infra/topology/table_grant_derivation.py, remove it "
            "from _INTERIM_ENTRIES here, and regenerate -- the instances must "
            "come back byte-identical, because the contract now derives what "
            "the entry used to"
        )

    def test_the_pin_is_readable_and_non_empty(self) -> None:
        """Positive control: prove the lookup can see contracts at all.

        Without this, a checkout that resolved to an empty tree would make the
        assertion above pass for the wrong reason -- the exact vacuous-green
        this module exists to refuse. It matters more once the map is empty,
        because then this is the only thing distinguishing "nothing to retire"
        from "could not look".
        """
        pinned = _pinned_relation_names()
        assert len(pinned) > 50, (
            f"the pinned contract set yielded only {len(pinned)} relations; "
            "the expiry assertion above is vacuous against a tree this small"
        )
