# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Run pool cloning and real workspace staging against disposable Git origins."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
PROVE = ROOT / "scripts/runtime_build/prepr_pool_prove.sh"
SIBLINGS = ("omnibase_core", "omnibase_spi", "omnimarket")
REPOS = ("omnibase_infra", *SIBLINGS)


def _run(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args, env=env, capture_output=True, text=True, timeout=60, check=False
    )


def _git(env: dict[str, str], repo: Path, *args: str) -> str:
    done = _run(env, "git", "-C", str(repo), *args)
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout.strip()


def _origins(tmp_path: Path) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    # All GitHub clone URLs are redirected by this process-local Git config.
    # No shell Git stub, host Git config, credentials or network is involved.
    home = tmp_path / "home"
    home.mkdir()
    config = tmp_path / "gitconfig"
    config.write_text(
        f'[url "{(tmp_path / "origins").as_uri()}/"]\n'
        "    insteadOf = https://github.com/OmniNode-ai/\n",
        encoding="utf-8",
    )
    env = {
        **scrub_git_location_env(),
        "HOME": str(home),
        "GIT_CONFIG_GLOBAL": str(config),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}/usr/bin{os.pathsep}/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "DEPLOY_SOURCE_CLONE_ROOT": str(tmp_path / "work/root"),
        "DEPLOY_SOURCE_WORKTREE_ROOT": str(tmp_path / "source-trees"),
        "DEPLOY_HOTPATCH": "0",
        "DEPLOY_SIBLING_REFS": "",
        "ALLOW_UNPINNED_DEPLOY_SOURCE": "0",
        "ALLOW_SIBLING_PIN_DRIFT": "0",
    }
    env.pop("MODEL_ENDPOINT", None)
    tags, commits = {}, {}
    for repo in REPOS:
        origin = tmp_path / "origins" / f"{repo}.git"
        origin.mkdir(parents=True)
        _git(env, origin, "init", "-q", "-b", "dev")
        _git(env, origin, "config", "user.name", "test")
        _git(env, origin, "config", "user.email", "test@example.com")
        (origin / "pyproject.toml").write_text(
            f'[project]\nname = "{repo.replace("_", "-")}"\nversion = "9.9.9"\n',
            encoding="utf-8",
        )
        (origin / "source.txt").write_text("tagged\n", encoding="utf-8")
        if repo == "omnibase_infra":
            build = origin / "scripts/runtime_build"
            build.mkdir(parents=True)
            for name in (
                "stage_workspace.sh",
                "sibling_clone_manifest.sh",
                "deploy_source_ref.py",
                "check_sibling_lock_pins.py",
            ):
                shutil.copy2(ROOT / "scripts/runtime_build" / name, build / name)
            validation = origin / "src/omnibase_infra/validation"
            validation.mkdir(parents=True)
            for name in ("checker_first_party_sources.py", "first_party_sources.yaml"):
                shutil.copy2(
                    ROOT / "src/omnibase_infra/validation" / name, validation / name
                )
        if repo == "omnimarket":
            ladder = origin / "src/omnimarket/configs/routing_tiers.yaml"
            ladder.parent.mkdir(parents=True)
            ladder.write_text("tiers: []\n", encoding="utf-8")
            (origin / "uv.lock").write_text(
                "\n".join(
                    f'[[package]]\nname = "{name.replace("_", "-")}"\nversion = "9.9.9"\n'
                    for name in REPOS
                ),
                encoding="utf-8",
            )
        _git(env, origin, "add", "-f", ".")
        _git(env, origin, "commit", "-qm", "tagged source")
        commits[repo] = _git(env, origin, "rev-parse", "HEAD")
        tags[repo] = f"dev-20261030150000-{commits[repo][:12]}"
        _git(env, origin, "tag", tags[repo])
        (origin / "source.txt").write_text("branch\n", encoding="utf-8")
        _git(env, origin, "add", ".")
        _git(env, origin, "commit", "-qm", "untagged branch advance")
    return env, tags, commits


def _params(tmp_path: Path, **overrides: str) -> Path:
    values = dict.fromkeys(
        (
            "INFRA_PR",
            "INFRA_GROUP",
            "INFRA_REF_SHA",
            "INFRA_DEV_TAG",
            "MARKET_PR",
            "MARKET_GROUP",
            "MARKET_REF_SHA",
            "CORE_REF",
            "SPI_REF",
            "STAGE_DEV_TAGS",
        ),
        "",
    )
    # A nonempty retired pin must be ignored, even though it cannot resolve.
    values.update(
        TAG="stage0-test",
        W=str(tmp_path / "work"),
        SLOT_KIND="isolated",
        COMPAT_REF="retired-compat-pin-must-be-ignored",
    )
    values.update(overrides)
    params = tmp_path / "params.env"
    params.write_text(
        "".join(f"{key}={shlex.quote(value)}\n" for key, value in values.items())
        # The real build has a required MODEL_ENDPOINT check immediately after
        # local-env. Let that check stop it normally before any container call.
        + f'make() {{ printf "%s\\n" "$*" >> {shlex.quote(str(tmp_path / "make.log"))}; }}\n'
        + f"docker() {{ echo unexpected-docker >> {shlex.quote(str(tmp_path / 'docker.log'))}; return 99; }}\n",
        encoding="utf-8",
    )
    return params


@pytest.mark.parametrize("use_tag", [False, True])
def test_pool_clone_selects_infra_tag_and_only_current_siblings(
    tmp_path: Path, use_tag: bool
) -> None:
    env, tags, commits = _origins(tmp_path)
    params = _params(
        tmp_path,
        INFRA_DEV_TAG=tags["omnibase_infra"] if use_tag else "",
        CORE_REF=tags["omnibase_core"],
        SPI_REF=tags["omnibase_spi"],
    )
    done = _run(env, "bash", str(PROVE), str(params), "clone")
    assert done.returncode == 0, done.stdout + done.stderr
    root = tmp_path / "work/root"
    assert {path.name for path in root.iterdir()} == set(REPOS)
    assert "omnibase_compat" not in done.stdout + done.stderr
    expected = (
        commits["omnibase_infra"]
        if use_tag
        else _git(env, tmp_path / "origins/omnibase_infra.git", "rev-parse", "dev")
    )
    assert _git(env, root / "omnibase_infra", "rev-parse", "HEAD") == expected
    for repo in SIBLINGS[:2]:
        assert _git(env, root / repo, "rev-parse", "HEAD") == commits[repo]
        assert (root / repo / "source.txt").read_text() == "tagged\n"


def test_pool_clone_fails_when_the_infra_dev_tag_cannot_be_cloned(
    tmp_path: Path,
) -> None:
    env, _, _ = _origins(tmp_path)
    params = _params(tmp_path, INFRA_DEV_TAG="dev-missing-tag")
    done = _run(env, "bash", str(PROVE), str(params), "clone")
    assert done.returncode == 1, done.stdout + done.stderr
    assert "dev-missing-tag" in done.stderr
    assert not (tmp_path / "work/root/omnimarket").exists()
    assert "clone done" not in done.stdout


@pytest.mark.parametrize("stage_tags", ["", "1"])
def test_pool_staging_passes_spi_pins_or_selects_dev_tags(
    tmp_path: Path, stage_tags: str
) -> None:
    env, tags, commits = _origins(tmp_path)
    params = _params(
        tmp_path, INFRA_DEV_TAG=tags["omnibase_infra"], STAGE_DEV_TAGS=stage_tags
    )
    cloned = _run(env, "bash", str(PROVE), str(params), "clone")
    assert cloned.returncode == 0, cloned.stdout + cloned.stderr
    root = tmp_path / "work/root"
    before = {repo: _git(env, root / repo, "rev-parse", "HEAD") for repo in SIBLINGS}
    done = _run(env, "bash", str(PROVE), str(params), "build")
    assert done.returncode == 1, done.stdout + done.stderr
    assert "MODEL_ENDPOINT is required" in done.stderr
    assert "stage rc=0" in done.stdout
    assert (tmp_path / "make.log").read_text().startswith("local-env ")
    assert not (tmp_path / "docker.log").exists()
    refs = json.loads((tmp_path / "work/refs.json").read_text())
    workspace = root / "omnibase_infra/workspace"
    record = json.loads((workspace / "sibling-vcs-provenance.json").read_text())
    assert set(record["siblings"]) == set(SIBLINGS)
    assert set(refs["repos"]) == set(SIBLINGS)
    assert "omnibase_compat" not in cloned.stdout + cloned.stderr
    assert "omnibase_compat" not in done.stdout + done.stderr
    assert "omnibase_compat" not in json.dumps(refs) + json.dumps(record)
    assert "compat" not in (tmp_path / "make.log").read_text()
    for repo in SIBLINGS:
        expected = commits[repo] if stage_tags else before[repo]
        entry = record["siblings"][repo]
        assert entry["vcs_ref"] == expected
        assert entry["vcs_dirty"] is False
        staged = workspace / "sibling-repos" / repo
        assert (staged / ".build-sha").read_text().strip() == expected
        assert (staged / "source.txt").read_text() == (
            "tagged\n" if stage_tags else "branch\n"
        )
        assert _git(env, root / repo, "rev-parse", "HEAD") == before[repo]
        assert refs["repos"][repo]["ref"] == (
            tags[repo] if stage_tags else before[repo]
        )
        assert refs["repos"][repo]["dev_tag_from"] == (
            "origin/dev" if stage_tags else ""
        )
        if stage_tags:
            assert entry["dev_merge_tag"] == tags[repo]
    assert (workspace / "routing_tiers.yaml").read_text() == "tiers: []\n"
