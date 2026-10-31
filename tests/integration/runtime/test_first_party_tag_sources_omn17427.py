# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Stage real Git origins, install their wheels, and read back both proofs.

Only disposable origins and build contexts are used. No Docker daemon, broker,
credentials, package index, or deployed lane is needed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / "scripts" / "runtime_build"
CHECKER = ROOT / "src/omnibase_infra/validation/checker_first_party_sources.py"
SIBLINGS = ("omnibase_core", "omnibase_spi", "omnimarket")
INSTALLED = (*SIBLINGS[:2], "omnibase_infra")
pytestmark = pytest.mark.integration


def _run(
    *args: str, cwd: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def _git(repo: Path, *args: str) -> str:
    done = _run("git", "-C", str(repo), *args, cwd=repo, env=scrub_git_location_env())
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout.strip()


def _sources(
    tmp_path: Path, missing_tag: str = ""
) -> tuple[Path, dict[str, str], dict[str, str]]:
    home = tmp_path / "clones"
    home.mkdir()
    tags, commits = {}, {}
    for repo in (*SIBLINGS, "omnibase_infra"):
        origin = tmp_path / "origins" / repo
        origin.mkdir(parents=True)
        _git(origin, "init", "-q", "-b", "dev")
        _git(origin, "config", "user.name", "test")
        _git(origin, "config", "user.email", "test@example.com")
        (origin / "pyproject.toml").write_text(
            '[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n'
            f'[project]\nname = "{repo.replace("_", "-")}"\nversion = "9.9.9"\n',
            encoding="utf-8",
        )
        package = origin / "src" / repo
        package.mkdir(parents=True)
        (package / "__init__.py").write_text('SOURCE = "tagged"\n', encoding="utf-8")
        if repo == "omnimarket":
            (package / "configs").mkdir()
            (package / "configs/routing_tiers.yaml").write_text(
                "tiers: []\n", encoding="utf-8"
            )
            (origin / "uv.lock").write_text(
                "\n".join(
                    f'[[package]]\nname = "{name.replace("_", "-")}"\nversion = "9.9.9"\n'
                    for name in (*SIBLINGS, "omnibase_infra")
                ),
                encoding="utf-8",
            )
            _git(origin, "add", "-f", "uv.lock")
        _git(origin, "add", ".")
        _git(origin, "commit", "-qm", "tagged package")
        commits[repo] = _git(origin, "rev-parse", "HEAD")
        tags[repo] = f"dev-20261030150000-{commits[repo][:12]}"
        if repo != missing_tag:
            _git(origin, "tag", tags[repo])
        if repo != "omnibase_infra":
            # The ambient branch has different bytes from the tag being built.
            (package / "__init__.py").write_text(
                'SOURCE = "branch"\n', encoding="utf-8"
            )
            _git(origin, "add", ".")
            _git(origin, "commit", "-qm", "untagged branch advance")
        _git(home, "clone", "-q", str(origin), str(home / repo))
    return home, tags, commits


def _stage(home: Path, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    env = {
        **scrub_git_location_env(),
        "OMNI_HOME": str(home),
        "DEPLOY_SOURCE_CLONE_ROOT": str(home),
        "DEPLOY_REF": "origin/dev",
        "DEPLOY_HOTPATCH": "0",
        "ALLOW_UNPINNED_DEPLOY_SOURCE": "0",
        "ALLOW_SIBLING_PIN_DRIFT": "0",
        "CONSUMER_LOCK": str(home / "omnimarket/uv.lock"),
        "DEPLOY_SOURCE_REFS_OUT": str(tmp_path / "refs.json"),
        "DEPLOY_SOURCE_WORKTREE_ROOT": str(tmp_path / "trees"),
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}/usr/bin{os.pathsep}/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    return _run(
        "bash", str(BUILD / "stage_workspace.sh"), cwd=home / "omnibase_infra", env=env
    )


@pytest.mark.parametrize("missing_tag", SIBLINGS)
def test_stage_refuses_each_sibling_without_a_dev_tag(
    tmp_path: Path, missing_tag: str
) -> None:
    home, _, _ = _sources(tmp_path, missing_tag)
    done = _stage(home, tmp_path)
    assert done.returncode == 4, done.stdout + done.stderr
    assert f"{missing_tag}: origin has no refs/tags/dev-* tag" in done.stderr
    assert not (home / "omnibase_infra/workspace/sibling-vcs-provenance.json").exists()


def _provenance(
    site: Path, workspace: Path, manifest: Path
) -> subprocess.CompletedProcess[str]:
    code = (
        "import importlib.util as u, sys; from pathlib import Path; "
        f"sys.path.insert(0, {str(site)!r}); "
        f"s = u.spec_from_file_location('provenance', {str(BUILD / 'compute_workspace_provenance.py')!r}); "
        "m = u.module_from_spec(s); s.loader.exec_module(m); "
        f"m.SIBLING_REPOS_DIR = Path({str(workspace / 'sibling-repos')!r}); "
        f"m.PIN_COMPARISON_PATH = Path({str(workspace / 'sibling-pin-comparison.json')!r}); "
        f"m.VCS_PROVENANCE_PATH = Path({str(workspace / 'sibling-vcs-provenance.json')!r}); "
        f"m.OUTPUT_MANIFEST = Path({str(manifest)!r}); sys.exit(m.main())"
    )
    return _run(sys.executable, "-c", code, cwd=workspace)


def _audit(
    site: Path, manifest: Path, staging: Path | None = None
) -> subprocess.CompletedProcess[str]:
    code = (
        "import runpy, sys; from pathlib import Path; "
        f"m = runpy.run_path({str(CHECKER)!r}); "
        f"sys.exit(m['run_audit'](Path({str(staging or manifest)!r}), "
        f"record_into=Path({str(manifest)!r}), search_paths=[{str(site)!r}], "
        f"require={[name.replace('_', '-') for name in INSTALLED]!r}))"
    )
    return _run(sys.executable, "-c", code, cwd=site)


def test_staged_wheels_have_tag_identity_and_content_parity(tmp_path: Path) -> None:
    home, tags, commits = _sources(tmp_path)
    before = {repo: _git(home / repo, "rev-parse", "HEAD") for repo in SIBLINGS}
    done = _stage(home, tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    workspace = home / "omnibase_infra/workspace"
    record = json.loads((workspace / "sibling-vcs-provenance.json").read_text())
    for repo in SIBLINGS:
        entry = record["siblings"][repo]
        assert entry["vcs_ref"] == commits[repo] != before[repo]
        assert entry["dev_merge_tag"] == tags[repo]
        assert entry["vcs_dirty"] is False
        assert _git(home / repo, "rev-parse", "HEAD") == before[repo]
        staged = workspace / "sibling-repos" / repo
        assert (staged / ".build-sha").read_text().strip() == commits[repo]
        assert (
            staged / "src" / repo / "__init__.py"
        ).read_text() == 'SOURCE = "tagged"\n'
    assert (
        record["build_context"]["omnibase_infra"]["dev_merge_tag"]
        == tags["omnibase_infra"]
    )

    # Build actual wheels from the staged trees. --target isolates the install
    # from pytest's environment; hatchling is a declared development dependency.
    uv = shutil.which("uv")
    assert uv is not None
    site = tmp_path / "installed"
    site.mkdir()
    for repo in INSTALLED:
        source = (
            home / repo
            if repo == "omnibase_infra"
            else workspace / "sibling-repos" / repo
        )
        installed = _run(
            uv,
            "pip",
            "install",
            "--python",
            sys.executable,
            "--target",
            str(site),
            "--no-deps",
            "--no-build-isolation",
            "--no-cache",
            "--no-sources",
            f"{repo.replace('_', '-')} @ {source.as_uri()}",
            cwd=tmp_path,
        )
        assert installed.returncode == 0, installed.stdout + installed.stderr

    manifest = tmp_path / "build-provenance.json"
    proof = _provenance(site, workspace, manifest)
    assert proof.returncode == 0, proof.stdout + proof.stderr
    audited = _audit(site, manifest, workspace / "sibling-vcs-provenance.json")
    assert audited.returncode == 0, audited.stdout + audited.stderr
    written = json.loads(manifest.read_text())
    assert {row["repo"]: row["tag"] for row in written["first_party_sources"]} == {
        repo: tags[repo] for repo in INSTALLED
    }
    assert {row["repo"] for row in written["proofs"]} == set(SIBLINGS[:2])

    # A running environment is re-audited from the folded manifest alone.
    assert _audit(site, manifest).returncode == 0
    written["per_repo_vcs_provenance"]["siblings"]["omnibase_spi"]["dev_merge_tag"] = (
        tags["omnibase_core"]
    )
    manifest.write_text(json.dumps(written), encoding="utf-8")
    refused = _audit(site, manifest)
    assert refused.returncode == 1
    assert "FAIL omnibase-spi: tag-mismatch" in refused.stdout

    # The independent content proof must reject a stale installed wheel too.
    (site / "omnibase_spi/__init__.py").write_text(
        'SOURCE = "stale"\n', encoding="utf-8"
    )
    stale = _provenance(site, workspace, manifest)
    assert stale.returncode == 1
    assert "omnibase-spi" in stale.stderr
    assert "__init__.py" in stale.stderr
