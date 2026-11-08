# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Command line of the dev-lane code-ref pin (OMN-23039).

Register, show or release the pin that keeps the dev deploy agent's code on a branch.
The result is one stdout line; the exit code is 0 when it worked, 1 when ``show`` finds
no live pin and 2 when ``register`` refuses a record::

    python -m omnibase_infra.nodes.node_dev_lane_code_ref_pin_effect.runtime_code_ref_pin \\
        register --state-dir D --branch B --head SHA --pr N --ticket OMN-n \\
        --registered-by LANE --expires-in-hours 48
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import ValidationError

from omnibase_infra.handlers.handler_dev_lane_code_ref_pin import (
    read_active_pin,
    release_pin,
    write_pin,
)
from omnibase_infra.models.model_dev_lane_code_ref_pin import ModelDevLaneCodeRefPin


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="runtime_code_ref_pin")
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register")
    register.add_argument("--state-dir", required=True, type=Path)
    register.add_argument("--branch", required=True)
    register.add_argument("--head", required=True)
    register.add_argument("--pr", required=True, type=int)
    register.add_argument("--ticket", required=True)
    register.add_argument("--registered-by", required=True)
    register.add_argument("--expires-in-hours", required=True, type=float)
    for name in ("show", "release"):
        commands.add_parser(name).add_argument("--state-dir", required=True, type=Path)
    return parser


def main(
    argv: Sequence[str] | None = None, *, clock: Callable[[], datetime] | None = None
) -> int:
    """Run one subcommand; the result is one stdout line."""
    args = _build_parser().parse_args(argv)
    now = (clock or (lambda: datetime.now(UTC)))()
    if args.command == "register":
        try:
            pin = ModelDevLaneCodeRefPin(
                branch=args.branch,
                head=args.head,
                pr=args.pr,
                ticket=args.ticket,
                registered_by=args.registered_by,
                registered_at=now,
                expires_at=now + timedelta(hours=args.expires_in_hours),
            )
        except ValidationError as exc:
            sys.stderr.write(f"code-ref-pin refused: {exc}\n")
            raise SystemExit(2) from exc
        target = write_pin(args.state_dir, pin)
        sys.stdout.write(
            f"code-ref-pin registered: branch={pin.branch} head={pin.head} "
            f"pr={pin.pr} ticket={pin.ticket} expires={pin.expires_at.isoformat()} "
            f"file={target}\n"
        )
        return 0
    if args.command == "show":
        live = read_active_pin(args.state_dir, now)
        sys.stdout.write(
            live.model_dump_json(indent=2) + "\n"
            if live
            else "code-ref-pin: none live\n"
        )
        return 0 if live else 1
    target = release_pin(args.state_dir)
    sys.stdout.write(f"code-ref-pin released: {target}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
