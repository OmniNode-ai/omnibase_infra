# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""No internal-only parameter on the ``onex`` commands this package ships (OMN-19197).

## What this pins

Every command this distribution registers under the ``onex.cli`` entry-point
group is part of the ``onex`` a customer installs. OMN-16852 moved the
workspace-root option's ENV binding to the product name ``OMNIBASE_PATH`` but
left the option spelled ``--omni-home`` (dest ``omni_home``) on ``onex
delegate``, ``onex node`` and ``onex skill``, and left ``$OMNI_HOME`` in the
help text of ``onex delegate --lane``. The 2026-08-28 boundary ruling under
OMN-16849 puts "CLI params a customer sets" on the rename side, and the C17
customer-surface probe reported all five as findings on a clean install.

The rule applied here is the one the C17 probe applies to the installed tree:
a parameter is rendered in its shell-expansion identities (``--omni-home``
and its dest ``omni_home`` both become ``$OMNI_HOME``; an envvar binding
becomes ``$<VAR>``), and every rendered identity plus every help string is
scanned for the maintainer-workspace name. The forbidden forms are spelled in
this file rather than read from another repository, so the check runs
wherever this repository's tests run.

## Why the tree is read from ``pyproject.toml``

The entry-point table is the list a customer's ``onex`` loads. Reading it from
the project file rather than from installed metadata means a stale editable
install cannot hide a newly registered command, and a floor on the enumerated
commands means an import failure cannot shrink the audit to nothing.

## The positive control

A zero from a scan that looked nowhere reads identically to a clean one, so
``test_the_scan_catches_a_seeded_internal_parameter`` seeds the exact shape
that shipped (an option ``--omni-home``, a help string naming
``$OMNI_HOME``, an ``OMNI_HOME`` envvar binding) and requires every one to be
reported.
"""

from __future__ import annotations

import importlib
import re
import tomllib
from pathlib import Path

import click
import pytest

_PYPROJECT = Path(__file__).resolve().parents[3] / "pyproject.toml"

#: The maintainer-workspace name. ``$OMNI_HOME`` and ``${OMNI_HOME`` are the
#: two shell spellings a customer would read or type.
_FORBIDDEN = re.compile(r"\$\{?OMNI_HOME\b")

#: Commands the enumeration must reach, or the audit is not an audit.
_REQUIRED_COMMANDS = frozenset({"onex delegate", "onex node", "onex skill"})


def _canonical(name: str) -> str:
    return name.lstrip("-").replace("-", "_").upper()


def _rendered_identities(param: click.Parameter) -> list[str]:
    """A parameter's names in the shell-expansion form a customer reads."""
    names = [*param.opts, *param.secondary_opts, param.name or ""]
    lines = [f"${_canonical(n)}" for n in names if n and _canonical(n)]
    envvar = getattr(param, "envvar", None)
    envvars = [envvar] if isinstance(envvar, str) else list(envvar or [])
    lines += [f"${v}" for v in envvars]
    return list(dict.fromkeys(lines))


def _walk(path: str, command: click.Command) -> list[tuple[str, click.Command]]:
    out = [(path, command)]
    if isinstance(command, click.Group):
        for name, sub in sorted(command.commands.items()):
            out += _walk(f"{path} {name}", sub)
    return out


def scan(commands: list[tuple[str, click.Command]]) -> list[str]:
    """Every place an internal-only parameter shows on these commands."""
    findings: list[str] = []
    for path, command in commands:
        for text in (command.help, command.short_help, command.epilog):
            if text and _FORBIDDEN.search(text):
                findings.append(f"{path} --help")
        for param in command.params:
            label = f"{path} {'/'.join(param.opts) or param.name}"
            if any(_FORBIDDEN.search(line) for line in _rendered_identities(param)):
                findings.append(label)
            help_text = getattr(param, "help", None)
            if help_text and _FORBIDDEN.search(help_text):
                findings.append(f"{label} (help)")
    return sorted(set(findings))


def _shipped_commands() -> list[tuple[str, click.Command]]:
    table = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    entry_points: dict[str, str] = table["project"]["entry-points"]["onex.cli"]
    commands: list[tuple[str, click.Command]] = []
    for name, target in sorted(entry_points.items()):
        module_name, _, attr = target.partition(":")
        command = getattr(importlib.import_module(module_name), attr)
        assert isinstance(command, click.Command), f"{target} is not a click command"
        commands += _walk(f"onex {name}", command)
    return commands


@pytest.mark.unit
def test_the_scan_catches_a_seeded_internal_parameter() -> None:
    """Positive control: the shape that shipped is reported, every part of it."""

    @click.command(help="Reads the lane declaration under $OMNI_HOME.")
    @click.option("--omni-home", type=str, default=None)
    @click.option("--lane", type=str, default=None, help="Root: ${OMNI_HOME}.")
    @click.option("--root", type=str, envvar="OMNI_HOME", default=None)
    @click.option("--omnibase-path", type=str, envvar="OMNIBASE_PATH", default=None)
    def seeded(
        omni_home: str | None,
        lane: str | None,
        root: str | None,
        omnibase_path: str | None,
    ) -> None:
        """Never invoked."""

    assert scan([("onex seeded", seeded)]) == [
        "onex seeded --help",
        "onex seeded --lane (help)",
        "onex seeded --omni-home",
        "onex seeded --root",
    ]


@pytest.mark.unit
def test_no_shipped_onex_command_exposes_an_internal_parameter() -> None:
    """AC1/AC2: no option spelled OMNI_HOME, no ``$OMNI_HOME`` in any help."""
    commands = _shipped_commands()
    reached = {path for path, _ in commands}
    missing = sorted(_REQUIRED_COMMANDS - reached)
    assert missing == [], f"the audit did not reach {missing}; it is not an audit"

    assert scan(commands) == [], (
        "an internal-only parameter is on a customer `onex` surface. The "
        "workspace root is --omnibase-path / $OMNIBASE_PATH (OMN-16849 "
        "boundary ruling, OMN-19197); OMNI_HOME names the maintainer's own "
        "checkout and never appears on a command a customer runs."
    )
