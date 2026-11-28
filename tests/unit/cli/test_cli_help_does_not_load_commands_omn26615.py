# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""AC3 (OMN-26615): rendering ``onex --help`` does not import extension command modules.

The mechanism: ``_LazyCoreCommandGroup.format_commands`` reads one-line help from
``onex.cli.help`` entry-point metadata and never calls ``_load_extension_command``
for a command that has a help entry.  This test proves that mechanism in a fresh
interpreter via subprocess, so no prior import contaminates ``sys.modules``.

Failure modes covered:
- the help page imports an omnibase_infra command module (the speed-up is broken)
- removing the help entry does NOT cause the import (the control is broken, so a
  passing green would be meaningless — VF-16)
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_PROBE_MODULE = "omnibase_infra.cli.cli_delegate"
_PROBE_COMMAND = "delegate"
_PYTHON = sys.executable


def _run_probe(*, remove_help_entry: bool) -> dict:
    """Run a fresh interpreter that renders onex --help and reports sys.modules."""
    script = textwrap.dedent(f"""\
        import json, sys

        remove_help = {remove_help_entry!r}
        probe_module = {_PROBE_MODULE!r}
        probe_command = {_PROBE_COMMAND!r}

        if remove_help:
            import importlib.metadata as _md
            _real_ep = _md.entry_points

            class _Stripped:
                def __init__(self, orig):
                    self._orig = orig

                def select(self, **kw):
                    eps = self._orig.select(**kw)
                    if kw.get("group") == "onex.cli.help":
                        return type(eps)(e for e in eps if e.name != probe_command)
                    return eps

                def __iter__(self):
                    return iter(self._orig)

                def __getattr__(self, name):
                    return getattr(self._orig, name)

            _md.entry_points = lambda **kw: _Stripped(_real_ep()).select(**kw) if kw else _Stripped(_real_ep())

        from omnibase_core.cli.cli_commands import cli
        import click

        ctx = click.Context(cli)
        cli.get_help(ctx)

        loaded = probe_module in sys.modules
        json.dump({{"loaded": loaded, "remove_help": remove_help}}, sys.stdout)
    """)
    result = subprocess.run(
        [_PYTHON, "-I", "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            f"Subprocess exited {result.returncode}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return json.loads(result.stdout)


def test_a_fresh_interpreter_help_page_leaves_the_declared_extension_unimported() -> (
    None
):
    """With the help entry present, rendering --help must NOT import the command module."""
    report = _run_probe(remove_help_entry=False)
    assert not report["loaded"], (
        f"{_PROBE_MODULE} was imported by onex --help even though "
        f"onex.cli.help declares the {_PROBE_COMMAND!r} command's one-line help"
    )


def test_control_removing_the_help_entry_causes_the_import() -> None:
    """Without the help entry, rendering --help MUST import the command module.

    This is the positive control (VF-16, CORE-02 check 6): if this test passes,
    the instrument works — a zero in the test above is a real finding.
    """
    report = _run_probe(remove_help_entry=True)
    assert report["loaded"], (
        f"{_PROBE_MODULE} was NOT imported when the help entry for "
        f"{_PROBE_COMMAND!r} was removed — the control is broken"
    )
