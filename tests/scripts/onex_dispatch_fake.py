# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Developer wrapper fake for the dispatch venv interpreter (OMN-25070).

The developer wrapper ``scripts/onex`` runs omnibase-core's CLI module with
the dispatch venv's interpreter (OMN-25070), so wrapper tests stand in for
that interpreter with a strict fake that refuses any other invocation; a
test that passes through it has therefore also proved the module form.
"""

from __future__ import annotations

from pathlib import Path

CORE_CLI_MODULE = "omnibase_core.cli.cli_commands"


def install_fake_core_cli(venv: Path, body: str) -> Path:
    """A fake dispatch interpreter at <venv>/bin/python that runs ``body``.

    It runs ``body`` with the CLI's own arguments as ``$@``. It exits 98
    and records WRONG_ENTRY unless called as ``-m omnibase_core.cli.cli_commands``,
    and it marks omnibase-core's CLI as installed in the venv's site-packages,
    which is what the wrapper checks before it execs.
    """
    bin_dir = venv / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    python = bin_dir / "python"
    text = (
        "#!/usr/bin/env bash\n"
        'if [[ "${1:-}" != "-m" || "${2:-}" != "' + CORE_CLI_MODULE + '" ]]; then\n'
        '  printf "WRONG_ENTRY %s\\n" "$*" >> "${WITNESS:-/dev/null}"\n'
        "  exit 98\n"
        "fi\n"
        "shift 2\n" + body
    )
    python.write_text(text, encoding="utf-8")
    python.chmod(0o755)
    marker = (
        venv
        / "lib"
        / "python3.12"
        / "site-packages"
        / "omnibase_core"
        / "cli"
        / "cli_commands.py"
    )
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("", encoding="utf-8")
    return python


def remove_core_cli(venv: Path) -> None:
    """Leave the interpreter but take omnibase-core's CLI out of the venv."""
    marker = (
        venv
        / "lib"
        / "python3.12"
        / "site-packages"
        / "omnibase_core"
        / "cli"
        / "cli_commands.py"
    )
    if marker.exists():
        marker.unlink()
