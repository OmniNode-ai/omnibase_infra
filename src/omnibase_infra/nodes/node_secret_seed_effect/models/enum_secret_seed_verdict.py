# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Terminal verdict for one headless secret-seeding run (OMN-16897)."""

from __future__ import annotations

from enum import StrEnum


class EnumSecretSeedVerdict(StrEnum):
    """Exactly one verdict is recorded per seed run.

    Only ``SEEDED`` and ``DRY_RUN`` are non-failing. Everything else is a
    failure, including ``NO_KEYS``: a seed run that seeded nothing is not a
    success, and reporting it as one is how a "green" run that changed
    nothing gets mistaken for a key that landed.

    The failure values are deliberately not collapsed into a single "red".
    ``AUTH_UNAVAILABLE`` and ``STORE_UNREACHABLE`` send an operator to two
    different places (the machine identity vs the instance address), and
    ``VERIFY_FAILED`` says something much more specific than
    ``WRITE_FAILED`` — the write was accepted and the name still did not
    appear on readback.
    """

    # Every requested name was written and confirmed present by NAME
    # readback. Values are never compared — this node never reads one.
    SEEDED = "seeded"
    # ``dry_run`` was set. The plan was computed from a name listing only;
    # zero writes were issued.
    DRY_RUN = "dry_run"
    # The machine-identity auth material was not resolvable. Fail-fast per
    # CLAUDE.md Rule 8 — no silent fallback to an ambient identity, and no
    # write attempted. The detail names the MISSING VARIABLE NAMES only.
    AUTH_UNAVAILABLE = "auth_unavailable"
    # The source file could not be read or parsed. The detail carries a line
    # NUMBER, never line content: a malformed line in a secrets file is
    # itself likely to be a secret.
    SOURCE_UNREADABLE = "source_unreadable"
    # The source parsed but yielded nothing to seed — an empty file, or a
    # ``--keys`` allowlist naming keys the source does not contain. Failing,
    # not passing: the caller asked for a write and got none.
    NO_KEYS = "no_keys"
    # The store could not be reached or its name listing failed. Distinct
    # from AUTH_UNAVAILABLE: the identity may be fine and the instance down,
    # or ``infisical_host`` may name an instance that is not the one meant.
    STORE_UNREACHABLE = "store_unreachable"
    # At least one ``set_secret`` call failed. Partial success is reported
    # in full — the run does not stop at the first failure, because a
    # half-finished seed the operator cannot see the shape of is worse than
    # a complete report of what landed and what did not.
    WRITE_FAILED = "write_failed"
    # Writes were accepted but at least one name was absent from the
    # post-write name listing. Fails closed: an unconfirmed write is not a
    # confirmed one.
    VERIFY_FAILED = "verify_failed"


__all__ = ["EnumSecretSeedVerdict"]
