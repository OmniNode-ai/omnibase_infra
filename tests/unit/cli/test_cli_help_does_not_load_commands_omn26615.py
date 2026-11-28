# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""AC3 (OMN-26615): rendering ``onex --help`` does not import extension command modules.

The mechanism: ``_LazyCoreCommandGroup.format_commands`` reads one-line help from
``onex.cli.help`` entry-point metadata and never calls ``_load_extension_command``
for a command that has a help entry.  This test proves that mechanism in a fresh
interpreter via subprocess, so no prior import contaminates ``sys.modules``.

CI builds with ``uv sync --frozen --all-extras --all-groups --no-install-project``,
so the repo's own entry points (``onex.cli`` and ``onex.cli.help``) are never
registered.  A synthetic dist-info stub built from ``pyproject.toml`` makes them
discoverable without installing the package.

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
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_PROBE_MODULE = "omnibase_infra.cli.cli_delegate"
_PROBE_COMMAND = "delegate"
_PYTHON = sys.executable
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _build_synthetic_ep(base: Path) -> Path:
    """Build a dist-info stub with the repo's onex.cli and onex.cli.help entry points."""
    with open(_REPO_ROOT / "pyproject.toml", "rb") as f:
        cfg = tomllib.load(f)
    eps = cfg["project"]["entry-points"]

    dist = base / "omnibase_infra_ep_stub-0.0.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.3\nName: omnibase-infra-ep-stub\nVersion: 0.0.0\n"
    )

    lines: list[str] = []
    for group in ("onex.cli", "onex.cli.help"):
        entries = eps.get(group, {})
        if entries:
            lines.append(f"[{group}]")
            for name, value in entries.items():
                lines.append(f"{name} = {value}")
            lines.append("")
    (dist / "entry_points.txt").write_text("\n".join(lines))
    return base


def _run_probe(*, remove_help_entry: bool, ep_base: str) -> dict:
    """Run a fresh interpreter that renders onex --help and reports sys.modules."""
    src_path = str(_REPO_ROOT / "src")
    script = textwrap.dedent(f"""\
        import importlib, importlib.metadata, json, sys

        ep_base = {ep_base!r}
        src_path = {src_path!r}
        probe_module = {_PROBE_MODULE!r}
        probe_command = {_PROBE_COMMAND!r}
        remove_help = {remove_help_entry!r}

        _existing = {{e.name for e in importlib.metadata.entry_points(group="onex.cli")}}
        if probe_command not in _existing:
            sys.path.insert(0, ep_base)
            sys.path.insert(0, src_path)
            importlib.invalidate_caches()

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
                    return (e for e in self._orig if not (e.group == "onex.cli.help" and e.name == probe_command))

                def __getattr__(self, name):
                    return getattr(self._orig, name)

            _md.entry_points = lambda **kw: _Stripped(_real_ep()).select(**kw) if kw else _Stripped(_real_ep())

        from omnibase_core.cli.cli_commands import cli
        import click

        ctx = click.Context(cli)
        help_text = cli.get_help(ctx)

        loaded = probe_module in sys.modules
        json.dump({{"loaded": loaded, "remove_help": remove_help, "help_text": help_text}}, sys.stdout)
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


def test_a_fresh_interpreter_help_page_leaves_the_declared_extension_unimported(
    tmp_path: Path,
) -> None:
    """With the help entry present, rendering --help must NOT import the command module."""
    ep_base = _build_synthetic_ep(tmp_path)
    report = _run_probe(remove_help_entry=False, ep_base=str(ep_base))
    assert _PROBE_COMMAND in report["help_text"], (
        f"{_PROBE_COMMAND!r} does not appear in the rendered help page"
    )
    assert not report["loaded"], (
        f"{_PROBE_MODULE} was imported by onex --help even though "
        f"onex.cli.help declares the {_PROBE_COMMAND!r} command's one-line help"
    )


def test_control_removing_the_help_entry_causes_the_import(
    tmp_path: Path,
) -> None:
    """Without the help entry, rendering --help MUST import the command module.

    This is the positive control (VF-16, CORE-02 check 6): if this test passes,
    the instrument works — a zero in the test above is a real finding.
    """
    ep_base = _build_synthetic_ep(tmp_path)
    report = _run_probe(remove_help_entry=True, ep_base=str(ep_base))
    assert report["loaded"], (
        f"{_PROBE_MODULE} was NOT imported when the help entry for "
        f"{_PROBE_COMMAND!r} was removed — the control is broken"
    )
