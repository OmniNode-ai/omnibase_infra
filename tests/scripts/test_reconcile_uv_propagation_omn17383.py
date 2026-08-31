# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The resolved ``uv`` must reach the co-install child, not just the parent (OMN-17383).

THE INCIDENT

OMN-17335 gave ``reconcile-workspace-venvs.sh`` an ordered ``resolve_uv()`` and
routed its own ``uv sync`` calls through the resolved ``$UV_BIN``. It then shells
out to ``install-node-skill-package.sh``, which calls **bare** ``uv`` and inherits
the caller's ``PATH``. Under the `.201` cron that ``PATH`` is::

    /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

which cannot reach ``~/.local/bin/uv`` -- the very install ``resolve_uv()`` had
just found. So the parent resolved uv successfully and the child still died::

    [reconcile] cli venv: reconciling provider layer to omnimarket 893d16ebc267
    == step 1: install omnimarket + omni-internal leaf deps (--no-deps) ==
    .../install-node-skill-package.sh: line 131: uv: command not found
    [reconcile] FAILED: provider co-install did not complete
    [reconcile-host] VERDICT: FAILED -- 2 surface(s) could not be proven at target.

WHY THE OMN-17335 GATE DID NOT CATCH IT

``check_reconciler_privilege.py`` discovers scripts by the glob
``scripts/reconcile*.sh``. ``install-node-skill-package.sh`` matches neither
pattern, so the one file that still had a bare ``uv`` was the one file never
scanned -- a gate whose scope stopped at a filename giving false assurance.

The fix propagates the parent's resolved interpreter down the ``PATH`` rather
than letting the child resolve its own: two independent resolutions can disagree,
and then the binary the parent proved usable is not the one that runs.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tests.scripts.test_reconcile_workspace_venvs import _make_uv_shim, _Workspace

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "reconcile-workspace-venvs.sh"


def _install_shim_recording_uv(path: Path, probe_log: Path) -> None:
    """Stand-in for the co-install that records whether IT can find ``uv``.

    This is the assertion that matters. Checking that the reconciler passed a
    ``PATH`` string would only prove the parent's intent; running ``command -v
    uv`` inside the child proves the child can actually reach the binary, which
    is the thing that was false on `.201`.
    """
    path.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "CHILD_UV=%s\\n" "$(command -v uv || echo NONE)" >> "{probe_log}"\n'
        f'printf "CHILD_PATH=%s\\n" "$PATH" >> "{probe_log}"\n'
        'printf "%s\\n" "${OMNIMARKET_REF:-<unset>}" >> "$INSTALL_SHIM_LOG"\n'
        'printf "install %s\\n" "${OMNIMARKET_REF:-<unset>}" >> "$ORDER_LOG"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_the_co_install_child_can_find_uv_when_path_alone_cannot(
    tmp_path: Path,
) -> None:
    """The `.201` condition: uv reachable only via the override, never via PATH.

    The workspace is put in provider drift so the co-install actually runs, then
    the child is asked -- in its own process -- whether ``uv`` resolves. Before
    this fix it answered ``NONE``.
    """
    workspace = _Workspace(tmp_path / "omni_home")
    # Provider drift: installed commit differs from the clone HEAD, which is what
    # makes the reconciler run the co-install at all.
    workspace.set_installed_commit("0" * 40)

    probe_log = tmp_path / "child.log"
    _install_shim_recording_uv(workspace.install_script, probe_log)

    override_bin = tmp_path / "override"
    _make_uv_shim(override_bin)

    env = workspace.env()
    # No uv anywhere on PATH, and a HOME with none either: the override is the
    # only route to a usable uv, exactly as a user-local install is under cron.
    env["PATH"] = "/usr/bin:/bin"
    env["HOME"] = str(tmp_path / "emptyhome")
    (tmp_path / "emptyhome").mkdir()
    env["ONEX_RECONCILE_UV_BIN"] = str(override_bin / "uv")

    result = subprocess.run(
        ["bash", str(_SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert probe_log.exists(), "the co-install never ran"
    recorded = probe_log.read_text(encoding="utf-8")
    assert "CHILD_UV=NONE" not in recorded, (
        "the co-install child could not find uv. The parent resolved it and did "
        "not pass it down -- OMN-17383, which is how the .201 reconcile died on "
        "'uv: command not found' after resolve_uv() had already succeeded."
    )
    assert str(override_bin / "uv") in recorded


def test_the_child_gets_the_same_uv_the_parent_resolved(tmp_path: Path) -> None:
    """Not merely *a* uv -- the one the parent proved usable.

    A child that finds a different uv on some other PATH entry would satisfy a
    naive "can it find uv" check while running a version nobody verified. The
    parent's choice is the contract.
    """
    workspace = _Workspace(tmp_path / "omni_home")
    workspace.set_installed_commit("0" * 40)

    probe_log = tmp_path / "child.log"
    _install_shim_recording_uv(workspace.install_script, probe_log)

    override_bin = tmp_path / "override"
    _make_uv_shim(override_bin)

    env = workspace.env()
    env["PATH"] = "/usr/bin:/bin"
    env["HOME"] = str(tmp_path / "emptyhome")
    (tmp_path / "emptyhome").mkdir()
    env["ONEX_RECONCILE_UV_BIN"] = str(override_bin / "uv")

    subprocess.run(
        ["bash", str(_SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    recorded = probe_log.read_text(encoding="utf-8")
    child_uv = next(
        line.split("=", 1)[1]
        for line in recorded.splitlines()
        if line.startswith("CHILD_UV=")
    )
    assert Path(child_uv).resolve() == (override_bin / "uv").resolve()


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #
def _gate(repo_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "python3",
            str(_REPO_ROOT / "scripts" / "check_reconciler_privilege.py"),
            "--repo-root",
            str(repo_root),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_gate_passes_on_the_real_repository() -> None:
    result = _gate(_REPO_ROOT)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_gate_rejects_a_co_install_invocation_without_path_propagation(
    tmp_path: Path,
) -> None:
    """Mutation proof on the real shipped file.

    Removing the one line that hands the child its interpreter must fail the
    build -- otherwise the gate extension is decoration and the next person to
    "tidy" that invocation reintroduces the `.201` outage.
    """
    scratch = tmp_path / "repo"
    (scratch / "scripts").mkdir(parents=True)
    source = _SCRIPT.read_text(encoding="utf-8")
    mutated = source.replace('          PATH="$(dirname "$UV_BIN"):$PATH" \\\n', "")
    assert mutated != source, "the mutation target moved; update this test"
    (scratch / "scripts" / "reconcile-workspace-venvs.sh").write_text(
        mutated, encoding="utf-8"
    )

    result = _gate(scratch)

    assert result.returncode == 1
    assert "without putting the resolved uv on the child's PATH" in result.stderr


def test_the_gate_is_wired_into_precommit_and_ci() -> None:
    """The extension rides the hook and CI step OMN-17335 already wired."""
    precommit = (_REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "check-reconciler-privilege" in precommit

    ci = (_REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "scripts/check_reconciler_privilege.py" in ci
