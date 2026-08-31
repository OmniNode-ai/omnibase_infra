# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-17291: the `.201` deploy-source clone sync must fail closed.

Three proven field defects, one test module:

  1. A canonical clone can carry ``core.bare=true`` while still having a full
     working tree on disk. In that shape ``git fetch`` exits 0 (and advances
     ``origin/<branch>``) while ``git checkout`` exits 128 -- so any sync loop
     that treats a clean fetch as progress reports success forever while HEAD
     never moves. ``test_bare_clone_fetch_succeeds_while_head_never_moves``
     pins that hazard down as a characterization fact; the reconciler must turn
     it into a NAMED failure, never a no-op.

  2. Nothing reconciles ``/data/omninode/omni_home`` on ``.201`` -- the tree
     every lane image is built from. The reconciler added here fetches,
     fast-forwards, refuses loudly on a dirty or diverged clone, VERIFIES HEAD
     actually landed on the fetched tip, and emits a repo/old->new receipt.

  3. ``stage_workspace.sh`` with ``DEPLOY_REF`` unset printed a warning and
     built the ambient tree anyway. A warning in a 4000-line deploy log is not
     a gate: the unpinned path is now a refusal with a named opt-in.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_SOURCE_REF = REPO_ROOT / "scripts" / "runtime_build" / "deploy_source_ref.py"
RECONCILE_SCRIPT = (
    REPO_ROOT / "scripts" / "runtime_build" / "reconcile_deploy_clones.sh"
)
STAGE_SCRIPT = REPO_ROOT / "scripts" / "runtime_build" / "stage_workspace.sh"

_DIST_NAME = {
    "omnibase_core": "omnibase-core",
    "omnibase_compat": "omnibase-compat",
    "omnimarket": "omnimarket",
    "omnibase_infra": "omnibase-infra",
    "omnibase_spi": "omnibase-spi",
}
_STAGED_SIBLINGS = ("omnibase_core", "omnibase_compat", "omnimarket")
_PIN_VERSION = "9.9.9"


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    return result.stdout.strip()


def _make_origin(path: Path, dist: str) -> Path:
    """Create an upstream repo on branch ``dev`` with one commit."""
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q", "-b", "dev")
    _git(path, "config", "user.email", "t@t.t")
    _git(path, "config", "user.name", "t")
    (path / "pyproject.toml").write_text(
        f"[project]\nname = '{dist}'\nversion = '{_PIN_VERSION}'\n", encoding="utf-8"
    )
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "init")
    return path


def _advance_origin(path: Path, marker: str = "advanced") -> str:
    (path / f"{marker}.txt").write_text(f"{marker}\n", encoding="utf-8")
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", f"advance dev ({marker})")
    return _git(path, "rev-parse", "HEAD")


def _clone(origin: Path, dest: Path) -> Path:
    subprocess.run(
        ["git", "clone", "-q", str(origin), str(dest)],
        check=True,
        capture_output=True,
        text=True,
    )
    _git(dest, "config", "user.email", "t@t.t")
    _git(dest, "config", "user.name", "t")
    return dest


def _make_deploy_tree(tmp_path: Path, repos: tuple[str, ...]) -> tuple[Path, Path]:
    """Build (origins_root, omni_home) with one origin + one clone per repo."""
    origins = tmp_path / "origins"
    omni_home = tmp_path / "omni_home"
    omni_home.mkdir(parents=True, exist_ok=True)
    for repo in repos:
        _make_origin(origins / repo, _DIST_NAME[repo])
        _clone(origins / repo, omni_home / repo)
    return origins, omni_home


def _run_reconcile(
    omni_home: Path,
    repos: tuple[str, ...],
    *,
    receipt: Path,
    branch: str = "dev",
) -> subprocess.CompletedProcess[str]:
    args = [
        "python3",
        str(DEPLOY_SOURCE_REF),
        "reconcile",
        "--branch",
        branch,
        "--output",
        str(receipt),
    ]
    for repo in repos:
        args += ["--repo", f"{repo}={omni_home / repo}"]
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )


# ---------------------------------------------------------------------------
# Defect 1 -- core.bare=true on a clone that HAS a working tree
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_bare_clone_fetch_succeeds_while_head_never_moves(tmp_path: Path) -> None:
    """Characterization of the field defect: on a ``core.bare=true`` clone that
    still has a working tree, ``git fetch`` exits 0 and advances the remote ref
    while ``git checkout`` exits non-zero and HEAD stays put.

    This is why "fetch exited 0" can never be evidence of sync.
    """
    origins, omni_home = _make_deploy_tree(tmp_path, ("omnibase_core",))
    clone = omni_home / "omnibase_core"
    before = _git(clone, "rev-parse", "HEAD")
    new_sha = _advance_origin(origins / "omnibase_core")
    assert before != new_sha

    # The exact corruption found on .201: config key flipped, working tree intact.
    _git(clone, "config", "core.bare", "true")
    assert (clone / "pyproject.toml").exists()
    assert (clone / ".git").is_dir()

    fetch = subprocess.run(
        ["git", "-C", str(clone), "fetch", "--prune", "origin", "dev"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert fetch.returncode == 0, fetch.stderr

    checkout = subprocess.run(
        ["git", "-C", str(clone), "checkout", "--force", new_sha],
        capture_output=True,
        text=True,
        check=False,
    )
    assert checkout.returncode != 0

    # HEAD never moved -- a sync loop reading only the fetch status reports success.
    assert _git(clone, "rev-parse", "HEAD") == before


@pytest.mark.unit
def test_reconcile_rejects_bare_clone_with_working_tree_by_name(
    tmp_path: Path,
) -> None:
    """AC2: a clone whose checkout cannot succeed is a FAILURE naming the repo
    and the cause, never a silent no-op."""
    origins, omni_home = _make_deploy_tree(tmp_path, ("omnibase_core", "omnimarket"))
    _advance_origin(origins / "omnibase_core")
    _git(omni_home / "omnibase_core", "config", "core.bare", "true")

    receipt = tmp_path / "receipt.json"
    result = _run_reconcile(omni_home, ("omnibase_core", "omnimarket"), receipt=receipt)

    assert result.returncode != 0, result.stdout + result.stderr
    combined = result.stdout + result.stderr
    assert "omnibase_core" in combined
    assert "core.bare" in combined
    # The remediation is named, not left as an opaque git fatal.
    assert "config core.bare false" in combined


@pytest.mark.unit
def test_clean_checkout_names_core_bare_rather_than_opaque_git_error(
    tmp_path: Path,
) -> None:
    """RT-1's own checkout path must diagnose the bare-with-working-tree shape
    instead of surfacing git's generic "must be run in a work tree" fatal."""
    origins, omni_home = _make_deploy_tree(tmp_path, ("omnibase_core",))
    _advance_origin(origins / "omnibase_core")
    _git(omni_home / "omnibase_core", "config", "core.bare", "true")

    result = subprocess.run(
        [
            "python3",
            str(DEPLOY_SOURCE_REF),
            "checkout",
            "--repo",
            f"omnibase_core={omni_home / 'omnibase_core'}",
            "--ref",
            "origin/dev",
            "--output",
            str(tmp_path / "refs.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "core.bare" in combined
    assert "omnibase_core" in combined


# ---------------------------------------------------------------------------
# Defect 2 -- nothing reconciles the .201 deploy-source clones
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_reconcile_fast_forwards_behind_clone_and_receipts_old_to_new(
    tmp_path: Path,
) -> None:
    """AC1: fetch, fast-forward, and emit a receipt naming repo and old->new."""
    origins, omni_home = _make_deploy_tree(tmp_path, ("omnibase_core", "omnimarket"))
    core = omni_home / "omnibase_core"
    before = _git(core, "rev-parse", "HEAD")
    new_sha = _advance_origin(origins / "omnibase_core")

    receipt = tmp_path / "receipt.json"
    result = _run_reconcile(omni_home, ("omnibase_core", "omnimarket"), receipt=receipt)
    assert result.returncode == 0, result.stdout + result.stderr

    assert _git(core, "rev-parse", "HEAD") == new_sha
    assert _git(core, "rev-parse", "--abbrev-ref", "HEAD") == "dev"

    doc = json.loads(receipt.read_text(encoding="utf-8"))
    row = doc["repos"]["omnibase_core"]
    assert row["before_sha"] == before
    assert row["after_sha"] == new_sha
    assert row["target_sha"] == new_sha
    assert row["moved"] is True
    assert doc["repos"]["omnimarket"]["moved"] is False


@pytest.mark.unit
def test_reconcile_recovers_a_detached_behind_clone(tmp_path: Path) -> None:
    """The .201 clones were DETACHED and behind (omnimarket, omnibase_core).
    A reconciler that only handles the on-branch case leaves them stale."""
    origins, omni_home = _make_deploy_tree(tmp_path, ("omnimarket",))
    clone = omni_home / "omnimarket"
    before = _git(clone, "rev-parse", "HEAD")
    _git(clone, "checkout", "-q", "--detach", before)
    new_sha = _advance_origin(origins / "omnimarket")

    receipt = tmp_path / "receipt.json"
    result = _run_reconcile(omni_home, ("omnimarket",), receipt=receipt)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _git(clone, "rev-parse", "HEAD") == new_sha
    assert _git(clone, "rev-parse", "--abbrev-ref", "HEAD") == "dev"


@pytest.mark.unit
def test_reconcile_refuses_dirty_clone_naming_repo(tmp_path: Path) -> None:
    """AC1: refuse loudly, with the repo named, on a dirty clone -- never
    clobber uncommitted work on the deploy host."""
    origins, omni_home = _make_deploy_tree(tmp_path, ("omnimarket",))
    clone = omni_home / "omnimarket"
    _advance_origin(origins / "omnimarket")
    (clone / "pyproject.toml").write_text("dirty\n", encoding="utf-8")

    receipt = tmp_path / "receipt.json"
    result = _run_reconcile(omni_home, ("omnimarket",), receipt=receipt)
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "omnimarket" in combined
    assert "dirty" in combined.lower()
    # The dirty edit survives -- the refusal did not reset the tree.
    assert (clone / "pyproject.toml").read_text(encoding="utf-8") == "dirty\n"


@pytest.mark.unit
def test_reconcile_tolerates_the_builds_own_scratch_and_still_advances(
    tmp_path: Path,
) -> None:
    """The build dirties its OWN build context on every run, so a reconciler
    that refuses on any dirt refuses forever on the clone it most needs to cover.

    Fixture reproduces the exact `.201` omnibase_infra state measured
    2026-08-31: two TRACKED workspace/*.json build outputs modified, one
    untracked workspace/*.json, and a stray untracked `origin/` directory.
    """
    origin = _make_origin(tmp_path / "origins" / "omnibase_infra", "omnibase-infra")
    # workspace/*.json are committed placeholders the build overwrites in place.
    origin_workspace = origin / "workspace"
    origin_workspace.mkdir()
    (origin_workspace / "sibling-pin-comparison.json").write_text(
        "{}\n", encoding="utf-8"
    )
    (origin_workspace / "sibling-vcs-provenance.json").write_text(
        "{}\n", encoding="utf-8"
    )
    _git(origin, "add", "-A")
    _git(origin, "commit", "-q", "-m", "commit workspace placeholders")

    omni_home = tmp_path / "omni_home"
    omni_home.mkdir(parents=True, exist_ok=True)
    clone = _clone(origin, omni_home / "omnibase_infra")
    workspace = clone / "workspace"
    before = _git(clone, "rev-parse", "HEAD")
    new_sha = _advance_origin(origin)

    # Now reproduce the live dirt.
    (workspace / "sibling-pin-comparison.json").write_text(
        '{"built": true}\n', encoding="utf-8"
    )
    (workspace / "sibling-vcs-provenance.json").write_text(
        '{"built": true}\n', encoding="utf-8"
    )
    (workspace / "deploy-source-refs.json").write_text("{}\n", encoding="utf-8")
    (clone / "origin").mkdir()
    (clone / "origin" / "stray").write_text("stray\n", encoding="utf-8")
    assert _git(clone, "status", "--porcelain") != ""

    receipt = tmp_path / "receipt.json"
    result = _run_reconcile(omni_home, ("omnibase_infra",), receipt=receipt)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _git(clone, "rev-parse", "HEAD") == new_sha

    doc = json.loads(receipt.read_text(encoding="utf-8"))
    assert doc["repos"]["omnibase_infra"]["before_sha"] == before
    assert doc["repos"]["omnibase_infra"]["after_sha"] == new_sha
    # The stray untracked directory is never destroyed by the reconcile.
    assert (clone / "origin" / "stray").exists()


@pytest.mark.unit
def test_reconcile_refuses_tracked_dirt_outside_the_build_scratch(
    tmp_path: Path,
) -> None:
    """The build-scratch tolerance is narrow: a tracked modification anywhere
    else still blocks, and is never silently discarded."""
    origins, omni_home = _make_deploy_tree(tmp_path, ("omnibase_infra",))
    clone = omni_home / "omnibase_infra"
    _advance_origin(origins / "omnibase_infra")
    (clone / "pyproject.toml").write_text("operator edit\n", encoding="utf-8")

    receipt = tmp_path / "receipt.json"
    result = _run_reconcile(omni_home, ("omnibase_infra",), receipt=receipt)
    assert result.returncode != 0
    assert "pyproject.toml" in result.stdout + result.stderr
    assert (clone / "pyproject.toml").read_text(encoding="utf-8") == "operator edit\n"


@pytest.mark.unit
def test_reconcile_refuses_diverged_clone_naming_repo(tmp_path: Path) -> None:
    """A clone carrying local commits is not fast-forwardable; refuse rather
    than silently discard them."""
    origins, omni_home = _make_deploy_tree(tmp_path, ("omnibase_core",))
    clone = omni_home / "omnibase_core"
    (clone / "local.txt").write_text("local\n", encoding="utf-8")
    _git(clone, "add", "-A")
    _git(clone, "commit", "-q", "-m", "local only")
    local_sha = _git(clone, "rev-parse", "HEAD")
    _advance_origin(origins / "omnibase_core")

    receipt = tmp_path / "receipt.json"
    result = _run_reconcile(omni_home, ("omnibase_core",), receipt=receipt)
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "omnibase_core" in combined
    assert "fast-forward" in combined.lower()
    assert _git(clone, "rev-parse", "HEAD") == local_sha


@pytest.mark.unit
def test_reconcile_wrapper_sources_the_shared_clone_manifest(tmp_path: Path) -> None:
    """The .201 entry point must derive its repo set from
    sibling_clone_manifest.sh -- not a third hand-maintained copy of the list
    (the OMN-15137 drift this repo already paid for once)."""
    assert RECONCILE_SCRIPT.exists(), (
        f"missing reconciler entry point: {RECONCILE_SCRIPT}"
    )
    body = RECONCILE_SCRIPT.read_text(encoding="utf-8")
    assert "sibling_clone_manifest.sh" in body
    assert "SIBLING_CLONE_MANIFEST" in body


@pytest.mark.unit
def test_reconcile_wrapper_runs_end_to_end_over_the_manifest(tmp_path: Path) -> None:
    """The wrapper reconciles every manifest repo present under OMNI_HOME and
    writes the receipt."""
    manifest_repos = (
        "omnibase_infra",
        "omnibase_core",
        "omnibase_spi",
        "omnibase_compat",
        "omnimarket",
    )
    origins, omni_home = _make_deploy_tree(tmp_path, manifest_repos)
    new_sha = _advance_origin(origins / "omnibase_spi")

    receipt = tmp_path / "reconcile-receipt.json"
    result = subprocess.run(
        ["bash", str(RECONCILE_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "OMNI_HOME": str(omni_home),
            "RECONCILE_RECEIPT": str(receipt),
            "GIT_TERMINAL_PROMPT": "0",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    doc = json.loads(receipt.read_text(encoding="utf-8"))
    assert set(doc["repos"]) == set(manifest_repos)
    assert doc["repos"]["omnibase_spi"]["after_sha"] == new_sha
    assert doc["repos"]["omnibase_spi"]["moved"] is True


# ---------------------------------------------------------------------------
# Defect 3 -- unset DEPLOY_REF warned and built the ambient tree anyway
# ---------------------------------------------------------------------------


def _write_consumer_lock(omni_home: Path) -> None:
    blocks = [
        f'[[package]]\nname = "{dist}"\nversion = "{_PIN_VERSION}"\n'
        for dist in _DIST_NAME.values()
    ]
    market = omni_home / "omnimarket"
    (market / "uv.lock").write_text("\n".join(blocks), encoding="utf-8")
    _git(market, "add", "uv.lock")
    _git(market, "commit", "-q", "-m", "add uv.lock")


def _stage_omni_home(tmp_path: Path) -> Path:
    omni_home = tmp_path / "omni_home"
    omni_home.mkdir(parents=True, exist_ok=True)
    for repo in (
        "omnibase_core",
        "omnibase_compat",
        "omnimarket",
        "omnibase_infra",
        "omnibase_spi",
    ):
        _make_origin(omni_home / repo, _DIST_NAME[repo])
    _write_consumer_lock(omni_home)
    return omni_home


def _run_stage(
    omni_home: Path, build_ctx: Path, **env_extra: str
) -> subprocess.CompletedProcess[str]:
    (build_ctx / "workspace").mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "OMNI_HOME": str(omni_home),
        "CONSUMER_LOCK": str(omni_home / "omnimarket" / "uv.lock"),
        **env_extra,
    }
    env.pop("DEPLOY_REF", None)
    env.pop("DEPLOY_HOTPATCH", None)
    env.update(env_extra)
    return subprocess.run(
        ["bash", str(STAGE_SCRIPT)],
        cwd=build_ctx,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.unit
def test_stage_workspace_refuses_unpinned_build_when_deploy_ref_unset(
    tmp_path: Path,
) -> None:
    """AC3: an unasserted source ref must not be able to produce a build. The
    old behaviour printed a warning and exited 0."""
    omni_home = _stage_omni_home(tmp_path)
    build_ctx = tmp_path / "ctx"
    result = _run_stage(omni_home, build_ctx)

    assert result.returncode != 0, result.stdout + result.stderr
    assert "DEPLOY_REF" in result.stderr
    # Nothing was staged: a refused build leaves no provenance claiming success.
    assert not (build_ctx / "workspace" / "sibling-vcs-provenance.json").exists()
    assert not (build_ctx / "workspace" / "sibling-repos" / "omnibase_core").exists()


@pytest.mark.unit
def test_stage_workspace_unpinned_build_requires_named_opt_in(
    tmp_path: Path,
) -> None:
    """The ambient-tree build stays reachable, but only behind an explicit,
    recorded opt-in -- never as the silent default."""
    omni_home = _stage_omni_home(tmp_path)
    build_ctx = tmp_path / "ctx"
    result = _run_stage(omni_home, build_ctx, ALLOW_UNPINNED_DEPLOY_SOURCE="1")

    assert result.returncode == 0, result.stdout + result.stderr
    assert (build_ctx / "workspace" / "sibling-vcs-provenance.json").exists()
    assert "ALLOW_UNPINNED_DEPLOY_SOURCE" in result.stderr
