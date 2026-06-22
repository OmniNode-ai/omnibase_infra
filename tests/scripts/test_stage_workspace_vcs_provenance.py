# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-13030: per-repo VCS provenance in build-provenance.json.

Covers the staging-time capture of {vcs_ref, vcs_dirty, vcs_branch} for each
sibling repo, the fail-non-zero behavior when a sibling has no git history, and
the manifest fold-in (infra_vcs_ref rename + per_repo_vcs_provenance block).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE_SCRIPT = REPO_ROOT / "scripts" / "runtime_build" / "stage_workspace.sh"
PROVENANCE_SCRIPT = (
    REPO_ROOT / "scripts" / "runtime_build" / "compute_workspace_provenance.py"
)
SCHEMA_FILE = REPO_ROOT / "scripts" / "runtime_build" / "build-provenance-schema.json"

SIBLING_REPOS = (
    "omnibase_core",
    "omnibase_compat",
    "onex_change_control",
    "omnimarket",
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )


# Distribution names (as they appear in uv.lock / pyproject) keyed by repo dir.
_DIST_NAME = {
    "omnibase_core": "omnibase-core",
    "omnibase_compat": "omnibase-compat",
    "onex_change_control": "onex-change-control",
    "omnimarket": "omnimarket",
    "omnibase_infra": "omnibase-infra",
    "omnibase_spi": "omnibase-spi",
}
_PIN_VERSION = "9.9.9"


def _init_repo(path: Path, dist: str, *, dirty: bool = False) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q", "-b", "dev")
    _git(path, "config", "user.email", "t@t.t")
    _git(path, "config", "user.name", "t")
    # Version matches the synthetic lock pin so the preflight resolves clean.
    (path / "pyproject.toml").write_text(
        f"[project]\nname = '{dist}'\nversion = '{_PIN_VERSION}'\n",
        encoding="utf-8",
    )
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "init")
    if dirty:
        (path / "pyproject.toml").write_text(
            f"[project]\nname = '{dist}'\nversion = '{_PIN_VERSION}'\n# dirty\n",
            encoding="utf-8",
        )


def _write_consumer_lock(omni_home: Path) -> None:
    """Write a uv.lock pinning every vendored sibling at _PIN_VERSION."""
    blocks = []
    for dist in _DIST_NAME.values():
        blocks.append(f'[[package]]\nname = "{dist}"\nversion = "{_PIN_VERSION}"\n')
    (omni_home / "omnimarket" / "uv.lock").write_text(
        "\n".join(blocks), encoding="utf-8"
    )


def _make_omni_home(tmp_path: Path, *, dirty_repo: str | None = None) -> Path:
    omni_home = tmp_path / "omni_home"
    for repo in SIBLING_REPOS:
        _init_repo(omni_home / repo, _DIST_NAME[repo], dirty=(repo == dirty_repo))
    # omnibase_infra (the building repo) + spi must exist for the preflight.
    for repo in ("omnibase_infra", "omnibase_spi"):
        _init_repo(omni_home / repo, _DIST_NAME[repo])
    _write_consumer_lock(omni_home)
    return omni_home


def _run_stage(omni_home: Path, build_ctx: Path) -> subprocess.CompletedProcess[str]:
    (build_ctx / "workspace").mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "OMNI_HOME": str(omni_home),
        # Synthetic clones are pinned at _PIN_VERSION matching the consumer lock,
        # so the preflight resolves clean without an allow-drift override.
        "CONSUMER_LOCK": str(omni_home / "omnimarket" / "uv.lock"),
    }
    return subprocess.run(
        ["bash", str(STAGE_SCRIPT)],
        cwd=build_ctx,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.unit
def test_stage_writes_per_repo_vcs_provenance(tmp_path: Path) -> None:
    omni_home = _make_omni_home(tmp_path, dirty_repo="omnimarket")
    build_ctx = tmp_path / "ctx"
    result = _run_stage(omni_home, build_ctx)
    assert result.returncode == 0, result.stderr

    manifest_path = build_ctx / "workspace" / "sibling-vcs-provenance.json"
    assert manifest_path.exists(), result.stderr
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    siblings = manifest["siblings"]
    assert set(siblings) == set(SIBLING_REPOS)
    for repo in SIBLING_REPOS:
        entry = siblings[repo]
        assert len(entry["vcs_ref"]) == 40  # full SHA
        assert entry["vcs_branch"] == "dev"
        assert isinstance(entry["vcs_dirty"], bool)
    # The dirty repo is flagged; clean repos are not.
    assert siblings["omnimarket"]["vcs_dirty"] is True
    assert siblings["omnibase_core"]["vcs_dirty"] is False


@pytest.mark.unit
def test_stage_aborts_when_sibling_has_no_git_history(tmp_path: Path) -> None:
    omni_home = _make_omni_home(tmp_path)
    # Destroy one sibling's git history -> unverifiable tree -> must abort.
    subprocess.run(["rm", "-rf", str(omni_home / "omnibase_core" / ".git")], check=True)
    build_ctx = tmp_path / "ctx"
    result = _run_stage(omni_home, build_ctx)
    # DoD (OMN-13030): a missing sibling git history must fail the build non-zero.
    # The failure may surface in the pin preflight (which also git-resolves clone
    # SHAs) or in the staging loop's vcs-provenance capture; either is a hard
    # abort — never a silent "unknown" stamp.
    assert result.returncode != 0
    assert (
        "cannot resolve clone pin" in result.stderr
        or "unverifiable tree" in result.stderr
    )
    # The unverifiable tree must NOT have produced a VCS provenance manifest.
    assert not (build_ctx / "workspace" / "sibling-vcs-provenance.json").exists()


@pytest.mark.unit
def test_manifest_uses_infra_vcs_ref_and_folds_vcs_provenance(tmp_path: Path) -> None:
    """compute_workspace_provenance folds the staged VCS provenance and renames
    the ambiguous top-level vcs_ref -> infra_vcs_ref."""
    # Build a synthetic image-layout tree the provenance script expects.
    sib_dir = tmp_path / "sibling-repos"
    venv_dir = tmp_path / "app" / ".venv"
    app_dir = tmp_path / "app"
    app_dir.mkdir(parents=True)
    for repo in (
        "omnibase_core",
        "omnibase_compat",
        "onex_change_control",
        "omnimarket",
    ):
        (sib_dir / repo).mkdir(parents=True)
        (sib_dir / repo / "f.py").write_text("x=1\n", encoding="utf-8")
    (sib_dir / "omnimarket" / "uv.lock").write_text("", encoding="utf-8")

    vcs_prov = tmp_path / "sibling-vcs-provenance.json"
    vcs_prov.write_text(
        json.dumps(
            {
                "siblings": {
                    "omnibase_core": {
                        "vcs_ref": "a" * 40,
                        "vcs_dirty": False,
                        "vcs_branch": "dev",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    # Patch the module-level paths so the script runs against the temp layout
    # without needing the in-image /workspace + /app absolute paths.
    import importlib.util

    spec = importlib.util.spec_from_file_location("cwp_under_test", PROVENANCE_SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mod.SIBLING_REPOS_DIR = sib_dir
    mod.VENV_DIR = venv_dir
    mod.OUTPUT_MANIFEST = app_dir / "build-provenance.json"
    mod.PIN_COMPARISON_PATH = tmp_path / "no-such-pin.json"
    mod.VCS_PROVENANCE_PATH = vcs_prov

    os.environ["VCS_REF"] = "deadbeef"
    rc = mod.main()
    # Install proofs will fail (packages not installed) -> rc 1, but the manifest
    # is still written with the renamed field + folded VCS provenance.
    manifest = json.loads(mod.OUTPUT_MANIFEST.read_text(encoding="utf-8"))
    assert "vcs_ref" not in manifest
    assert manifest["infra_vcs_ref"] == "deadbeef"
    assert (
        manifest["per_repo_vcs_provenance"]["siblings"]["omnibase_core"]["vcs_ref"]
        == "a" * 40
    )
    assert rc in (0, 1)  # rc depends on install proofs, not on this ticket's fields


@pytest.mark.unit
def test_schema_declares_per_repo_vcs_provenance_and_infra_vcs_ref() -> None:
    schema = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    assert "infra_vcs_ref" in schema["properties"]
    sibling_schema = schema["properties"]["per_repo_vcs_provenance"]["properties"][
        "siblings"
    ]["additionalProperties"]
    assert set(sibling_schema["required"]) == {"vcs_ref", "vcs_dirty", "vcs_branch"}
