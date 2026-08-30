# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for ``scripts/onex`` -- the sanctioned CLI entry point (OMN-17190).

Fully hermetic: every run points ``OMNI_HOME`` at a ``tmp_path`` skeleton, so
no test can reach (or sync) the real workspace venv. The "entry point" and the
"reconciler" are shell stubs that record their argv.

## What this pins down, and why it exists

``onex`` was documented only as an interactive shell alias
(``alias onex='uv run --project $OMNI_HOME/omnibase_infra onex'``). An alias is
not an invocation contract: it does not exist in a non-interactive shell, so
scripts, hooks, Makefiles and agent tool calls resolve ``onex`` through PATH
instead. Measured on the author's Mac 2026-08-30, with the workspace venv
confirmed ``IN_SYNC``, ``bash -lc 'onex skill ...'`` resolved a ``uv tool``
shim holding omnibase_infra 0.38.11 (pre-self-heal) and a PyPI omnimarket, and
emitted the pre-OMN-17190 hard refusal. A second sibling at
``/opt/homebrew/bin/onex`` ran the self-heal, repaired the workspace venv, and
refused anyway -- it was never running in the venv it had just repaired.

So the contract asserted here is: ONE interpreter, resolved deterministically,
with no PATH lookup and no silent fallback to a sibling install.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "onex"

_EX_CONFIG = 78
_EX_UNAVAILABLE = 69


def _write_exec(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/usr/bin/env bash\n{body}\n", encoding="utf-8")
    path.chmod(0o755)


def _skeleton(root: Path, *, with_entrypoint: bool, with_reconciler: bool) -> Path:
    """Build a throwaway $OMNI_HOME with the two collaborators as stubs."""
    infra = root / "omnibase_infra"
    if with_entrypoint:
        _write_exec(
            infra / ".venv" / "bin" / "onex",
            f'printf "%s\\n" "WORKSPACE-ONEX $*" > "{root}/entrypoint.argv"',
        )
    if with_reconciler:
        # A reconciler that actually creates the entry point, the way the real
        # one does by building the venv.
        _write_exec(
            infra / "scripts" / "reconcile-workspace-venvs.sh",
            f'printf "%s\\n" "RECONCILED $*" >> "{root}/reconcile.argv"\n'
            f'mkdir -p "{infra}/.venv/bin"\n'
            f'printf "#!/usr/bin/env bash\\nprintf \'%%s\\\\n\' \\"WORKSPACE-ONEX $*\\" '
            f'> \\"{root}/entrypoint.argv\\"\\n" > "{infra}/.venv/bin/onex"\n'
            f'chmod +x "{infra}/.venv/bin/onex"',
        )
    return root


def _run(root: Path | None, *args: str, path_decoy: Path | None = None):
    env = dict(os.environ)
    env.pop("OMNI_HOME", None)
    if root is not None:
        env["OMNI_HOME"] = str(root)
    if path_decoy is not None:
        # A hostile PATH carrying a DIFFERENT `onex`, exactly like the sibling
        # installs measured on the author's machine.
        env["PATH"] = f"{path_decoy}:{env.get('PATH', '')}"
    return subprocess.run(
        ["bash", str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_execs_the_workspace_entrypoint_and_forwards_argv(tmp_path: Path) -> None:
    root = _skeleton(tmp_path, with_entrypoint=True, with_reconciler=True)

    result = _run(root, "skill", "merge_sweep", "--dry-run")

    assert result.returncode == 0, result.stderr
    assert (
        root / "entrypoint.argv"
    ).read_text().strip() == "WORKSPACE-ONEX skill merge_sweep --dry-run"
    # Nothing was reconciled: the entry point already existed.
    assert not (root / "reconcile.argv").exists()


def test_a_sibling_onex_earlier_on_path_is_never_used(tmp_path: Path) -> None:
    """The load-bearing assertion.

    This is the failure that made the OMN-17190 self-heal read as flaky: a
    PATH-resolved sibling install answering to the name `onex`, running a
    different interpreter with its own omnimarket state. The wrapper must
    resolve by PATH for NOTHING.
    """
    root = _skeleton(tmp_path, with_entrypoint=True, with_reconciler=True)
    decoy_dir = tmp_path / "decoy_bin"
    _write_exec(decoy_dir / "onex", f'printf "SIBLING\\n" > "{root}/entrypoint.argv"')

    result = _run(root, "delegate", "hello", path_decoy=decoy_dir)

    assert result.returncode == 0, result.stderr
    assert (root / "entrypoint.argv").read_text().startswith("WORKSPACE-ONEX")


def test_missing_entrypoint_is_bootstrapped_once_then_exec(tmp_path: Path) -> None:
    """Bootstrap, not fallback: build the venv, then run IT -- never a sibling."""
    root = _skeleton(tmp_path, with_entrypoint=False, with_reconciler=True)
    decoy_dir = tmp_path / "decoy_bin"
    _write_exec(decoy_dir / "onex", f'printf "SIBLING\\n" > "{root}/entrypoint.argv"')

    result = _run(root, "node", "merge_sweep", path_decoy=decoy_dir)

    assert result.returncode == 0, result.stderr
    reconcile_calls = (root / "reconcile.argv").read_text().strip().splitlines()
    assert len(reconcile_calls) == 1, "bootstrap must run the reconciler exactly once"
    assert f"--omni-home {root}" in reconcile_calls[0]
    assert (root / "entrypoint.argv").read_text().startswith("WORKSPACE-ONEX")


def test_unset_omni_home_refuses_and_never_guesses(tmp_path: Path) -> None:
    """No default root (CLAUDE.md rule 8): a guess dispatches another checkout."""
    result = _run(None, "skill", "x")

    assert result.returncode == _EX_CONFIG
    assert "OMNI_HOME is not set" in result.stderr


def test_unbuildable_venv_refuses_and_names_the_command(tmp_path: Path) -> None:
    """A reconciler that cannot produce the entry point is a refusal, not a fallback."""
    root = _skeleton(tmp_path, with_entrypoint=False, with_reconciler=False)
    _write_exec(
        root / "omnibase_infra" / "scripts" / "reconcile-workspace-venvs.sh",
        "exit 2",
    )
    decoy_dir = tmp_path / "decoy_bin"
    _write_exec(decoy_dir / "onex", 'printf "SIBLING\\n"')

    result = _run(root, "skill", "x", path_decoy=decoy_dir)

    assert result.returncode == _EX_UNAVAILABLE
    assert "reconcile-workspace-venvs.sh" in result.stderr
    assert "SIBLING" not in result.stdout


def test_missing_reconciler_and_missing_venv_refuses(tmp_path: Path) -> None:
    root = _skeleton(tmp_path, with_entrypoint=False, with_reconciler=False)

    result = _run(root, "skill", "x")

    assert result.returncode == _EX_UNAVAILABLE
    assert str(root / "omnibase_infra" / ".venv" / "bin" / "onex") in result.stderr


def test_exit_status_of_the_workspace_cli_is_propagated(tmp_path: Path) -> None:
    """`exec` semantics: the caller sees the CLI's own status, not the wrapper's."""
    root = _skeleton(tmp_path, with_entrypoint=False, with_reconciler=False)
    _write_exec(root / "omnibase_infra" / ".venv" / "bin" / "onex", "exit 42")

    assert _run(root, "anything").returncode == 42
