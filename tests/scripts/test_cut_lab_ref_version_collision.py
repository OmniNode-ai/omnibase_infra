# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-14562: cut-lab-ref.sh must survive a same-version redeploy collision.

The bug (found exercising OMN-14438's RT-1/RT-2 live on .201 for the first
time): ``deploy-runtime.sh``'s version-directory collision guard
(``guard_existing_deployment()``, Phase 5) runs BEFORE ``sync_files()``'s
``stage_workspace_if_needed()`` (Phase 6) -- the function that engages RT-1's
clean-ref checkout. Because the lab fast lane redeploys the SAME package
version at new SHAs constantly (version bumps are infrequent/manual), this
guard fires on nearly every ordinary lab redeploy, and RT-1's checkout never
gets a chance to run. ``cut-lab-ref.sh`` never passed ``--force`` through, so
there was no way to clear the guard from the wrapper.

These tests prove, through the REAL ``cut-lab-ref.sh`` wrapper and the REAL
RT-1 mechanism (``stage_workspace.sh`` / ``deploy_source_ref.py`` -- not a
stub of RT-1 itself):

* RED: with a pre-existing ``deployed/{version}/`` directory and no
  ``--force``, the collision blocks BEFORE the RT-1 checkout ever runs (the
  sibling clone never moves off its stale checked-out commit, and no
  expected-refs manifest is written).
* GREEN: ``cut-lab-ref.sh`` (fixed) reaches the SAME collision with
  ``--force`` and the RT-1 checkout actually runs -- the sibling clone is
  checked out to the new ref SHA (not the stale one) and the vendored-SHA
  manifest proves it.

Only deploy-runtime.sh's Docker-build phases (Phase 5 guard onward through
image build) are represented by a small fixture stub, so these tests do not
require a running Docker daemon -- consistent with every other
``tests/scripts/test_deploy_runtime_*.py`` test in this repo, none of which
exercise deploy-runtime.sh's real Docker phases either. The stub's guard
logic is a direct, log-string-faithful port of
``guard_existing_deployment()``; ``test_deploy_runtime_guard_precedes_staging``
below locks the REAL script's call order so the fixture cannot silently drift
from production.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CUT_LAB_REF = REPO_ROOT / "scripts" / "runtime_build" / "cut-lab-ref.sh"
STAGE_WORKSPACE = REPO_ROOT / "scripts" / "runtime_build" / "stage_workspace.sh"
DEPLOY_RUNTIME_SCRIPT = REPO_ROOT / "scripts" / "deploy-runtime.sh"

SIBLING_REPOS = (
    "omnibase_core",
    "omnibase_compat",
    "onex_change_control",
    "omnimarket",
)
LAB_REF_REPOS = ("omnibase_infra", *SIBLING_REPOS)
_DIST_NAME = {
    "omnibase_core": "omnibase-core",
    "omnibase_compat": "omnibase-compat",
    "onex_change_control": "onex-change-control",
    "omnimarket": "omnimarket",
    "omnibase_infra": "omnibase-infra",
    "omnibase_spi": "omnibase-spi",
}
_PIN_VERSION = "9.9.9"
_FAKE_APP_VERSION = "0.99.0"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    ).stdout.strip()


def _init_repo(path: Path, dist: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q", "-b", "dev")
    _git(path, "config", "user.email", "t@t.t")
    _git(path, "config", "user.name", "t")
    (path / "pyproject.toml").write_text(
        f"[project]\nname = '{dist}'\nversion = '{_PIN_VERSION}'\n", encoding="utf-8"
    )
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "init")


def _write_consumer_lock(omni_home: Path) -> None:
    blocks = [
        f'[[package]]\nname = "{dist}"\nversion = "{_PIN_VERSION}"\n'
        for dist in _DIST_NAME.values()
    ]
    market = omni_home / "omnimarket"
    (market / "uv.lock").write_text("\n".join(blocks), encoding="utf-8")
    _git(market, "add", "uv.lock")
    _git(market, "commit", "-q", "-m", "add uv.lock")


def _make_omni_home(tmp_path: Path) -> Path:
    omni_home = tmp_path / "omni_home"
    for repo in LAB_REF_REPOS:
        _init_repo(omni_home / repo, _DIST_NAME[repo])
    _init_repo(omni_home / "omnibase_spi", _DIST_NAME["omnibase_spi"])
    _write_consumer_lock(omni_home)
    return omni_home


def _advance_dev_leave_behind(repo: Path) -> tuple[str, str]:
    """Advance ``dev`` by a commit, then leave the clone checked out (detached)
    at the OLD commit -- exactly the ambient-behind-clone disease OMN-14438
    targeted. Returns (old_sha, new_dev_sha)."""
    old_sha = _git(repo, "rev-parse", "HEAD")
    (repo / "marker.txt").write_text("advanced\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "advance dev")
    new_sha = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "--detach", old_sha)
    return old_sha, new_sha


def _write_deploy_runtime_stub(
    path: Path, *, fake_deploy_root: Path, fake_version: str
) -> None:
    """A fixture stub standing in for deploy-runtime.sh's guard + Docker phases
    ONLY. It replicates guard_existing_deployment()'s exact log strings and
    Phase 5 (guard) -> Phase 6 (stage_workspace_if_needed) ordering, then
    delegates the real RT-1 work to the REAL stage_workspace.sh -- RT-1 itself
    is never stubbed.
    """
    deploy_target = fake_deploy_root / "deployed" / fake_version
    path.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
FORCE=false
for a in "$@"; do
    [[ "$a" == "--force" ]] && FORCE=true
done
DEPLOY_TARGET="{deploy_target}"
if [[ -d "${{DEPLOY_TARGET}}" ]]; then
    if [[ "${{FORCE}}" != true ]]; then
        echo "[deploy] ERROR: Deployment directory already exists:" >&2
        echo "[deploy] ERROR:   ${{DEPLOY_TARGET}}" >&2
        echo "[deploy] ERROR: This version has already been deployed." >&2
        exit 1
    fi
    rm -rf "${{DEPLOY_TARGET}}.bak"
    mv "${{DEPLOY_TARGET}}" "${{DEPLOY_TARGET}}.bak"
fi
mkdir -p "${{DEPLOY_TARGET}}/workspace"
cd "${{DEPLOY_TARGET}}"
bash "{STAGE_WORKSPACE}"
echo "[deploy] stub: RT-1 staging complete" >&2
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


@pytest.mark.unit
def test_collision_without_force_blocks_before_rt1_checkout(tmp_path: Path) -> None:
    """RED: a pre-existing deployed/{version}/ directory blocks the whole
    pipeline before RT-1's clean-ref checkout ever runs -- reproducing the
    OMN-14562 bug directly (no cut-lab-ref.sh involved; this isolates the
    collision mechanism itself)."""
    omni_home = _make_omni_home(tmp_path)
    old_sha, new_sha = _advance_dev_leave_behind(omni_home / "omnimarket")
    assert old_sha != new_sha
    assert _git(omni_home / "omnimarket", "rev-parse", "HEAD") == old_sha

    fake_deploy_root = tmp_path / "deploy-root"
    deploy_target = fake_deploy_root / "deployed" / _FAKE_APP_VERSION
    deploy_target.mkdir(parents=True)
    (deploy_target / "stale-marker.txt").write_text("pre-existing\n", encoding="utf-8")

    stub = tmp_path / "stub-deploy-runtime.sh"
    _write_deploy_runtime_stub(
        stub, fake_deploy_root=fake_deploy_root, fake_version=_FAKE_APP_VERSION
    )

    result = subprocess.run(
        ["bash", str(stub), "--execute", "--restart"],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "OMNI_HOME": str(omni_home),
            "DEPLOY_REF": "dev",
            "CONSUMER_LOCK": str(omni_home / "omnimarket" / "uv.lock"),
        },
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Deployment directory already exists" in result.stderr
    assert "already been deployed" in result.stderr

    # RT-1 never ran: the sibling clone is still detached at the OLD (behind)
    # SHA, and no expected-refs manifest was written into the collision dir.
    assert _git(omni_home / "omnimarket", "rev-parse", "HEAD") == old_sha
    assert not (deploy_target / "workspace" / "deploy-source-refs.json").exists()
    assert (deploy_target / "stale-marker.txt").exists()  # untouched


@pytest.mark.unit
def test_cut_lab_ref_execute_overwrites_collision_and_runs_rt1_checkout(
    tmp_path: Path,
) -> None:
    """GREEN: the REAL (fixed) cut-lab-ref.sh reaches the SAME collision with
    --force, and RT-1's clean-ref checkout actually runs and vendors the
    advanced ref -- not the stale ambient one."""
    omni_home = _make_omni_home(tmp_path)
    old_sha, new_sha = _advance_dev_leave_behind(omni_home / "omnimarket")
    assert old_sha != new_sha

    fake_deploy_root = tmp_path / "deploy-root"
    deploy_target = fake_deploy_root / "deployed" / _FAKE_APP_VERSION
    deploy_target.mkdir(parents=True)
    (deploy_target / "stale-marker.txt").write_text("pre-existing\n", encoding="utf-8")

    stub = tmp_path / "stub-deploy-runtime.sh"
    _write_deploy_runtime_stub(
        stub, fake_deploy_root=fake_deploy_root, fake_version=_FAKE_APP_VERSION
    )

    result = subprocess.run(
        ["bash", str(CUT_LAB_REF), "--ref", "dev", "--lane", "dev", "--execute"],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "OMNI_HOME": str(omni_home),
            "DEPLOY_RUNTIME": str(stub),
            "CONSUMER_LOCK": str(omni_home / "omnimarket" / "uv.lock"),
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--force" in result.stderr
    assert "stub: RT-1 staging complete" in result.stderr

    # RT-1 actually ran and moved the sibling clone to the NEW (advanced) SHA --
    # a no-op checkout would have left it detached at old_sha.
    assert _git(omni_home / "omnimarket", "rev-parse", "HEAD") == new_sha

    expected_refs = deploy_target / "workspace" / "deploy-source-refs.json"
    provenance = deploy_target / "workspace" / "sibling-vcs-provenance.json"
    assert expected_refs.exists(), "RT-1 expected-refs manifest was never written"
    assert provenance.exists(), "RT-1 vendored-SHA provenance was never written"

    exp = json.loads(expected_refs.read_text(encoding="utf-8"))
    assert exp["repos"]["omnimarket"]["expected_sha"] == new_sha

    vcs = json.loads(provenance.read_text(encoding="utf-8"))
    assert vcs["siblings"]["omnimarket"]["vcs_ref"] == new_sha
    assert vcs["siblings"]["omnimarket"]["vcs_ref"] != old_sha


# ---------------------------------------------------------------------------
# Structural lock: the REAL deploy-runtime.sh's guard-before-stage ordering
# ---------------------------------------------------------------------------


def _function_body(name: str) -> str:
    """Return the source of a top-level shell function ``name() { ... }`` from
    the REAL deploy-runtime.sh (mirrors the helper in
    test_deploy_runtime_cold_full_bringup.py)."""
    text = DEPLOY_RUNTIME_SCRIPT.read_text(encoding="utf-8")
    anchored = text.find(f"\n{name}() {{")
    start = 0 if text.startswith(f"{name}() {{") else anchored + 1
    assert anchored != -1 or text.startswith(f"{name}() {{"), (
        f"function {name}() not found in deploy-runtime.sh"
    )
    rest = text[start:]
    end_rel = rest.find("\n}\n")
    assert end_rel != -1, f"could not find end of function {name}()"
    return rest[: end_rel + 3]


@pytest.mark.unit
def test_deploy_runtime_guard_precedes_staging() -> None:
    """Pins the ROOT CAUSE: main() must call guard_existing_deployment()
    before sync_files() (which is what engages RT-1's clean-ref checkout via
    stage_workspace_if_needed()). This is why a same-version lab redeploy
    needs --force -- RT-1 never gets a chance to run otherwise. If a future
    change reorders this, the fixture stub above (and this test) will drift
    from production; this lock exists precisely to catch that."""
    body = _function_body("main")
    guard = body.find('guard_existing_deployment "${deploy_target}"')
    sync = body.find('sync_files "${repo_root}" "${deploy_target}"')
    assert guard != -1, "guard_existing_deployment not called in main()"
    assert sync != -1, "sync_files not called in main()"
    assert guard < sync, (
        "guard_existing_deployment must run BEFORE sync_files (OMN-14562): "
        "the version-directory collision guard blocks the RT-1 clean-ref "
        "checkout (inside sync_files -> stage_workspace_if_needed) from ever "
        "running on a same-version redeploy."
    )
