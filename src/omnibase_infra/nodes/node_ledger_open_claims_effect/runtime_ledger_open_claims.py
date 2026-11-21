# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Command line of the open-claims read, for scripts that used to grep the ledger file (OMN-20547).

``snapshot`` reads the open claims once and writes them to a file; ``match`` prints
``<stamp> lane=<lane>`` for each open claim in a snapshot whose row names a key as a
whole token; ``owner`` prints the lane of the newest open claim naming a repo and one of
its paths, or ``unowned``. Exit 0 answered, 3 refused (the reason on stderr): a reader
that cannot answer is never read around.

    python -m omnibase_infra.nodes.node_ledger_open_claims_effect.runtime_ledger_open_claims \\
        snapshot --reader-command <cmd> --reader <name> --window-hours <n> --out <file>
    python -m omnibase_infra.nodes.node_ledger_open_claims_effect.runtime_ledger_open_claims \\
        match <snapshot-file> -- <token> [<token> ...]
    python -m omnibase_infra.nodes.node_ledger_open_claims_effect.runtime_ledger_open_claims \\
        owner --reader-command <cmd> --reader <name> --window-hours <n> --repo <name> --path <path>
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from omnibase_infra.handlers.handler_ledger_open_claims import (
    LedgerReadRefusedError,
    matching,
    owner_of,
    read_open_claims,
)
from omnibase_infra.models.model_ledger_open_claim import ModelLedgerOpenClaim
from omnibase_infra.models.model_ledger_open_claims_request import (
    ModelLedgerOpenClaimsRequest,
)

EXIT_OK = 0
EXIT_REFUSED = 3


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="runtime_ledger_open_claims")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("snapshot", "owner"):
        read = commands.add_parser(name)
        read.add_argument("--reader-command", required=True)
        read.add_argument("--reader", required=True)
        read.add_argument("--window-hours", type=int, required=True)
    commands.choices["snapshot"].add_argument("--out", type=Path, required=True)
    commands.choices["owner"].add_argument("--repo", required=True)
    commands.choices["owner"].add_argument("--path", action="append", required=True)
    match = commands.add_parser("match")
    match.add_argument("snapshot", type=Path)
    match.add_argument("keys", nargs="+")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand; a refusal is exit 3 with its reason on stderr."""
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "match":
            lines = args.snapshot.read_text(encoding="utf-8").splitlines()
            claims = [ModelLedgerOpenClaim.model_validate_json(x) for x in lines]
            for claim in matching(claims, args.keys):
                sys.stdout.write(f"{claim.stamp} lane={claim.lane}\n")
            return EXIT_OK
        request = ModelLedgerOpenClaimsRequest(
            reader_command=args.reader_command,
            reader=args.reader,
            window_hours=args.window_hours,
        )
        claims = list(read_open_claims(request).claims)
    except (LedgerReadRefusedError, OSError, ValidationError) as exc:
        sys.stderr.write(f"runtime_ledger_open_claims: REFUSED: {exc}\n")
        return EXIT_REFUSED
    if args.command == "snapshot":
        args.out.write_text(
            "".join(f"{c.model_dump_json()}\n" for c in claims), encoding="utf-8"
        )
    else:
        sys.stdout.write(f"{owner_of(claims, args.repo, args.path) or 'unowned'}\n")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
