# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16932: the dispatch venv and the proven floor move together.

The default (repair, not --proven) run installs the clone head into the
dispatch venv. Until OMN-16932 it left ``$OMNI_HOME/.onex-workspace-floor.json``
at the old commit until a later step stamped it, so the real ``scripts/onex``
wrapper refused every evidence command in between (exit 3). The reconciler now
advances the floor inside the same staging-lock critical section as the venv
swap, so a reader outside the lock never sees the two disagree.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tests.scripts.test_reconcile_workspace_venvs import (
    _EXIT_FAILED,
    _EXIT_OK,
    _advance,
    _Workspace,
)

pytestmark = pytest.mark.unit


def _install_shim_writing_dist_info(ws: _Workspace) -> None:
    """Replace the install shim with one that also writes the dist-info.

    The plain shim only logs ``OMNIMARKET_REF`` to ``$INSTALL_SHIM_LOG``. The
    floor reader (``reconcile_verify_movement.py floor-from-venv``) reads the
    installed commit from ``omnimarket-*.dist-info/direct_url.json``, so a shim
    that does not write that file leaves the floor unable to advance.
    """
    (ws.dispatch_venv / "lib" / "python3.12" / "site-packages").mkdir(
        parents=True, exist_ok=True
    )
    # The installer is called as `install-node-skill-package.sh --execute <python>`:
    # the venv it writes into is the one that interpreter belongs to, which during
    # a staged pass is the staging sibling and not the live venv.
    dist = '"$site/omnimarket-0.4.11.dist-info"'
    shim = (
        "#!/usr/bin/env bash\n"
        "set -uo pipefail\n"
        'printf "%s\\n" "${OMNIMARKET_REF:-<unset>}" >> "$INSTALL_SHIM_LOG"\n'
        'site="$(dirname "$(dirname "$2")")/lib/python3.12/site-packages"\n'
        f"mkdir -p {dist}\n"
        f'printf "Name: omnimarket\\nVersion: 0.4.11\\n" > {dist}/METADATA\n'
        'printf \'{"vcs_info": {"commit_id": "%s"}}\' "$OMNIMARKET_REF"'
        f" > {dist}/direct_url.json\n"
        "exit 0\n"
    )
    ws.install_script.write_text(shim, encoding="utf-8")
    ws.install_script.chmod(0o755)


@pytest.fixture
def ws(tmp_path: Path) -> _Workspace:
    return _Workspace(tmp_path)


def test_a_clone_advance_moves_the_floor_with_the_venv(ws: _Workspace) -> None:
    """The floor advances with the venv, so the wrapper no longer refuses."""
    floor = ws.root / ".onex-workspace-floor.json"
    floor.write_text(
        json.dumps({"omnimarket_commit": ws.market_head, "distributions": {}}),
        encoding="utf-8",
    )
    _install_shim_writing_dist_info(ws)
    newer = _advance(ws.omnimarket, "advance")
    result = ws.run()
    assert result.returncode == _EXIT_OK, result.stdout + result.stderr

    data = json.loads(floor.read_text(encoding="utf-8"))
    assert data["omnimarket_commit"] == newer, result.stdout + result.stderr

    # The seam between the reconciler's floor writer and the wrapper's floor
    # reader: with the floor advanced, the real wrapper runs the entrypoint
    # instead of refusing it.
    wrapper_src = Path(__file__).resolve().parents[2] / "scripts" / "onex"
    wrapper_dst = ws.infra / "scripts" / "onex"
    wrapper_dst.parent.mkdir(parents=True, exist_ok=True)
    wrapper_dst.write_text(wrapper_src.read_text(encoding="utf-8"), encoding="utf-8")
    wrapper_dst.chmod(0o755)
    entrypoint = ws.dispatch_venv / "bin" / "onex"
    entrypoint.parent.mkdir(parents=True, exist_ok=True)
    entrypoint.write_text("#!/usr/bin/env bash\nexit 41\n", encoding="utf-8")
    entrypoint.chmod(0o755)

    wrapped = subprocess.run(
        ["bash", str(wrapper_dst), "skill", "merge_sweep"],
        capture_output=True,
        text=True,
        env={"OMNIBASE_PATH": str(ws.root), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
        check=False,
    )
    assert wrapped.returncode == 41, wrapped.stdout + wrapped.stderr
    assert "REFUSED" not in wrapped.stderr


def test_no_floor_means_no_first_floor_is_created(ws: _Workspace) -> None:
    """No floor file: the run succeeds and never creates the first floor."""
    floor = ws.root / ".onex-workspace-floor.json"
    assert not floor.exists()
    _install_shim_writing_dist_info(ws)
    result = ws.run()
    assert result.returncode == _EXIT_OK, result.stdout + result.stderr
    assert not floor.exists()


def test_proven_never_advances_the_floor(ws: _Workspace) -> None:
    """--proven installs the proven commit and never moves the floor."""
    floor = ws.root / ".onex-workspace-floor.json"
    proven = ws.market_head
    floor.write_text(
        json.dumps({"omnimarket_commit": proven, "distributions": {}}),
        encoding="utf-8",
    )
    before = floor.read_bytes()
    _install_shim_writing_dist_info(ws)
    _advance(ws.omnimarket, "advance")
    result = ws.run("--proven")
    assert result.returncode == _EXIT_OK, result.stdout + result.stderr
    assert floor.read_bytes() == before


def test_a_floor_that_cannot_be_advanced_fails_the_run(ws: _Workspace) -> None:
    """A floor that cannot be advanced (no dist-info written) fails the run."""
    floor = ws.root / ".onex-workspace-floor.json"
    floor.write_text(
        json.dumps({"omnimarket_commit": ws.market_head, "distributions": {}}),
        encoding="utf-8",
    )
    _advance(ws.omnimarket, "advance")
    result = ws.run()
    assert result.returncode == _EXIT_FAILED, result.stdout + result.stderr
    assert "floor" in result.stdout
