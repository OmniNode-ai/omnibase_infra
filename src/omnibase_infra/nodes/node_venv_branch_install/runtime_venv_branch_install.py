# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Command line of the omnimarket branch install (OMN-26102).

    install --host H --venv V --state-dir S --pr N --pr-state FILE --unit U [--unit U ...]
    revert  --host H --venv V --state-dir S

``FILE`` is the pull request watcher record of omnimarket#N
(``pr_state_local.py --pr OmniNode-ai/omnimarket#N --json``). The head sha is read
from it and is not an option: a head somebody typed is never installed.

Output lines: ``installed=<host>:<sha>``, ``receipt=<path>``, ``replaced=<sha>``.
The exit code is the handler's: 0 ok, 3 refused, 5 failed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_infra.handlers.handler_venv_branch_install import (
    HandlerVenvBranchInstall,
    head_from_watcher_facts,
)
from omnibase_infra.models.model_venv_branch_install_request import (
    ModelVenvBranchInstallRequest,
)

_EXIT_USAGE = 2


async def _bootstrap() -> ModelONEXContainer:
    """Bind the handler at the composition root."""
    container = ModelONEXContainer()
    await container.service_registry.register_instance(
        HandlerVenvBranchInstall, HandlerVenvBranchInstall(container)
    )
    return container


def main(argv: list[str] | None = None) -> int:
    """Parse one command, run it and print its record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["install", "revert"])
    parser.add_argument("--host", required=True)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--pr", type=int, default=0)
    parser.add_argument("--pr-state", type=Path)
    parser.add_argument("--unit", action="append", default=[])
    args = parser.parse_args(argv)
    head = ""
    if args.command == "install":
        if args.pr_state is None:
            sys.stderr.write("an install reads the head from --pr-state\n")
            return _EXIT_USAGE
        try:
            record = json.loads(args.pr_state.read_text(encoding="utf-8"))
            head = head_from_watcher_facts(record, repo="omnimarket", number=args.pr)
        except (OSError, ValueError) as exc:
            sys.stderr.write(f"cannot read the head from {args.pr_state}: {exc}\n")
            return _EXIT_USAGE
    try:
        request = ModelVenvBranchInstallRequest(
            command=args.command,
            host=args.host,
            venv=args.venv,
            state_dir=args.state_dir,
            pr_number=args.pr,
            head_sha=head,
            units=tuple(args.unit),
        )
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return _EXIT_USAGE
    container = asyncio.run(_bootstrap())
    result = container.get_service(HandlerVenvBranchInstall).handle(request)
    sys.stdout.write(result.output)
    for key, value in (
        ("status", result.status),
        ("installed", result.installed.removeprefix("installed=")),
        ("receipt", result.receipt_path),
        ("replaced", result.replaced_revision),
    ):
        if value:
            sys.stdout.write(f"{key}={value}\n")
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
