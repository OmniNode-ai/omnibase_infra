#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Refuse a trailered vendoring PR whose derivation only holds on its own branch.

OMN-18863. ``Application Database Domain Enforcement (OMN-15361)`` derives the
application-database TABLE grants from omnimarket node contracts. Which
omnimarket tree it derives from is NOT fixed: when a pull request body declares
``Node-Migration-Source-*`` trailers, ``resolve_node_migration_source_ref.py``
reads them out of the ``pull_request`` event payload and the job checks the
dependency out at that BRANCH. A push to ``dev`` carries no pull-request
payload, so the same job falls back to the committed pin.

That asymmetry is the defect. A vendoring pull request adds a relation to the
shipped topology instances, passes the gate because its trailer tree declares
that relation, merges, and the push to ``dev`` then refuses the byte-identical
tree because the pin does not declare it yet. Measured twice on 2026-09-19/20,
nine minutes apart:

* ``#3795`` head ``831644aa2613`` GREEN 23:39:57Z, dev squash ``30dec562ab19``
  RED 23:55:39Z. Same tree: both ``fd7b35d2b8f139331184733e927768a385e2da81``.
* ``#3861`` cleared dev 02:16:49Z; ``#3821`` merged 02:25:15Z and RED 02:25:52Z.

Each occurrence blocked every open pull request in the repository until someone
unrelated diagnosed it and hand-landed a supplemental declaration. The cost
landed on every other lane rather than on the change that created the window,
which is what this check moves.

WHAT THIS DOES. On a pull request that declares the trailers, the enforcement
job now derives TWICE: once against the trailer-resolved tree, which is the
existing check and is unchanged, and once against the committed pin, exactly as
the push to ``dev`` will. Both must hold. A vendoring pull request therefore
cannot merge into a state its own required gate rejects on the next push; it
must carry the self-expiring supplemental declaration that bridges the window.

WHAT THIS DELIBERATELY DOES NOT DO.

* It does not weaken the push-side check. That check is the backstop and stays
  exactly as strict.
* It does not teach the push side to read trailers out of the squash message.
  That was considered and rejected: a squash message is editable after review
  by anyone with write access, so it is forgeable in a way a pull-request body
  read at PR time is not, and it would make the push-side verdict depend on
  prose. Strictly worse than making the PR carry its own bridge.
* It does not run on a pull request without the trailers. Such a pull request
  already derives from the pin on both sides, so there is nothing asymmetric
  to check and running it would only add a second identical derivation.

HONEST LIMIT. This enforces that a declaration exists when it is needed. It
does not judge whether the vendoring is right on the merits, and it cannot stop
a relation being declared that should never have been. What it removes is the
silent case: a merge that is green on its own head and red for everybody else
one push later.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GENERATOR = _REPO_ROOT / "scripts" / "generate_application_database_table_grants.py"

#: Printed verbatim on refusal. It names the remedy rather than the symptom,
#: because three separate lanes in one night read the sibling check's "the pin
#: needs advancing, which is the bot's job" text and concluded that a plain
#: regeneration was the fix. It is not: regenerating against a pin that does
#: not declare the relation DELETES the declaration from three instances and
#: nine rendered catalogs while the vendored migration still grants it, which
#: trips the OMN-18768 reverse ratchet and, per that gate's own text, refuses
#: the projection binding at boot and takes the whole runtime process down.
REMEDY = """\
The derivation holds against the omnimarket tree your trailers name, and does
NOT hold against the committed pin. That is the OMN-18863 window: this pull
request would be green on its own head and would red `dev` for every other open
pull request on the next push, because a push carries no pull-request payload
and derives from the pin.

DO NOT regenerate the grants to make this pass. If a relation is declared in
the shipped topology instances and derivable from neither the pin nor a
supplemental entry, regeneration DELETES it while the vendored migration still
grants it -- which arms the OMN-18768 boot crash rather than fixing anything.

Add a self-expiring supplemental declaration in this same pull request:

  1. a `ContractTableDeclaration` in `LEGACY_MIGRATION_TABLE_DECLARATIONS`
     (`src/omnibase_infra/topology/table_grant_derivation.py`), following the
     `OMN-18159` / `OMN-17426` precedent blocks already in that tuple; and
  2. one line in `_INTERIM_ENTRIES`
     (`tests/ci/test_supplemental_declaration_expiry_omn18863.py`) naming the
     relation and the omnimarket pull request that retires it.

Step 2 is what deletes step 1 again: that module goes red on the pin advance
that makes the bridge redundant, so the entry is removed by the commit that
caused it rather than accumulating. Five of eight entries in that tuple had
gone silently redundant before it existed.

Then re-run. A correct bridge makes THIS check pass and writes no generated
diff, because the declaration reproduces grants the instances already carry.
"""


def _run_check(contracts_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(_GENERATOR),
            "--contracts-root",
            str(contracts_root),
            "--check",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=_REPO_ROOT,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pin-contracts-root",
        required=True,
        type=Path,
        help=(
            "omnimarket node contracts checked out at the COMMITTED pin, which "
            "is what a push to dev will derive from"
        ),
    )
    parser.add_argument(
        "--trailer-ref",
        default="",
        help="the trailer-resolved ref, for the message only",
    )
    args = parser.parse_args(argv)

    root: Path = args.pin_contracts_root
    if not root.is_dir():
        # Fail closed. An absent checkout here is indistinguishable from a
        # passing derivation if we let it through, and that is the exact
        # vacuous-green this whole ticket is about.
        print(
            "::error::OMN-18863: the committed-pin contracts root "
            f"{root} does not exist, so the push-side derivation could not be "
            "evaluated. This check fails closed: a derivation that was not run "
            "is not a derivation that passed.",
            file=sys.stderr,
        )
        return 1

    result = _run_check(root)
    if result.returncode == 0:
        print(
            "OMN-18863: the derivation holds against the committed pin as well "
            "as the trailer-resolved tree"
            + (f" ({args.trailer_ref})" if args.trailer_ref else "")
            + "; this pull request will not red dev on merge."
        )
        return 0

    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    print(
        "::error::OMN-18863: this pull request derives its grants from the "
        "omnimarket tree named by its Node-Migration-Source trailers, and the "
        "same derivation FAILS against the committed pin. See the remedy below.",
        file=sys.stderr,
    )
    print(REMEDY, file=sys.stderr)
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
