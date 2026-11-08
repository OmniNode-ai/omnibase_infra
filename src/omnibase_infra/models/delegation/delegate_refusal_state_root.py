# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Where ``onex delegate`` files a refusal receipt when arguments did not parse.

Moved out of ``omnibase_infra.cli.cli_delegate`` (OMN-23163) so the command line
module only shrinks while it gains the traffic flags.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import click

from omnibase_infra.cli.receipt_mode import resolve_state_root
from omnibase_infra.errors import ProtocolConfigurationError

__all__ = ["funnel_state_root", "state_root_from_arguments"]

STATE_ROOT_FLAG = "--state-root"


def funnel_state_root(explicit: Path | None, *, quiet: bool = False) -> Path | None:
    """The root a refusal receipt is filed under: the command's own rule.

    The flag is the operator's own choice and is used as given; without it the
    root resolves exactly as the command body resolves it (OMN-19232), never
    against the working directory. ``None`` when no root can be resolved (a
    relative ``ONEX_STATE_DIR``, a root under ``~/.claude``): there is then no
    place the receipt belongs, and the cause goes to stderr instead. ``quiet``
    (``--result-only``) leaves stderr alone: the command body refuses on the same
    unresolvable root and that refusal is the one line the caller reads.
    """
    if explicit is not None:
        return explicit
    try:
        return resolve_state_root(None)
    except ProtocolConfigurationError as exc:
        if not quiet:
            click.echo(
                f"onex delegate: no refusal receipt was written: {exc.message}",
                err=True,
            )
        return None


def state_root_from_arguments(arguments: Sequence[str]) -> Path | None:
    """Read ``--state-root`` out of raw arguments that did not parse.

    Argument parsing may have failed before the option was read, and the
    receipt belongs where the caller said state lives.
    """
    for index, argument in enumerate(arguments):
        if argument == STATE_ROOT_FLAG and index + 1 < len(arguments):
            return Path(arguments[index + 1])
        if argument.startswith(f"{STATE_ROOT_FLAG}="):
            return Path(argument.partition("=")[2])
    return None
