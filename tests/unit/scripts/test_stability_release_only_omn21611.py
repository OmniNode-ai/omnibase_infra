# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21611: a stability refresh runs release tags only.

The stability lane runs a pinned release set, never a dev ref. A refresh
asked for a dev tag, a bare sha, a branch or a dev omnimarket ref is refused,
naming ``release-tag-only``, before it reads the lane, tags an image, checks a
clone out, builds or restarts anything. PATH shims record every docker, curl,
jq and git call the script makes, so "before it changes anything" is measured
from what ran, not asserted from the source.

The fixtures are real git clones under a temporary OMNI_HOME, each with a
release tag and a dev tag, and a pin file under a temporary workspace config
root, the way ``ONEX_WORKSPACE_CONFIG_ROOT`` names it on a lane host.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_BUILD = REPO_ROOT / "scripts" / "runtime_build"
REFRESH = RUNTIME_BUILD / "refresh_stability_lane.sh"
STAGE = RUNTIME_BUILD / "stage_workspace.sh"

#: The lane id the script hardcodes; the pin file is named for it. Read from
#: the script so no lane id is spelled in this public repository.
_LANE_MATCH = re.search(r'^readonly LANE="([^"]+)"$', REFRESH.read_text(), re.M)
assert _LANE_MATCH is not None
LANE = _LANE_MATCH.group(1)
DEV_TAG = "dev-20261025205951-5bb5f11404ae"
RELEASE = {
    "omnibase_infra": "v0.38.74",
    "omnibase_core": "v0.47.46",
    "omnibase_compat": "v0.5.7",
    "omnimarket": "v0.4.316",
}


def _env() -> dict[str, str]:
    env = scrub_git_location_env(os.environ)
    return {k: v for k, v in env.items() if not k.startswith("GIT_")}


def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        [
            "git",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "user.name=fixture",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "tag.gpgsign=false",
            *args,
        ],
        cwd=cwd,
        env=scrub_git_location_env(os.environ),
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


@pytest.fixture
def clones(tmp_path: Path) -> Path:
    home = tmp_path / "clones"
    for repo, tag in RELEASE.items():
        clone = home / repo
        clone.mkdir(parents=True)
        _git(clone, "init", "-q", "-b", "dev")
        (clone / "README").write_text(repo)
        _git(clone, "add", "-A")
        _git(clone, "commit", "-q", "-m", "release")
        _git(clone, "tag", tag)
        (clone / "DEV").write_text("dev\n")
        _git(clone, "add", "-A")
        _git(clone, "commit", "-q", "-m", "dev")
        _git(clone, "tag", DEV_TAG)
    return home


def _write_pin(config_root: Path, pin: dict[str, str]) -> Path:
    path = config_root / "config" / "lab" / "lane-overlays" / f"{LANE}.release-pin.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "schema_version: 1",
        "date: 2026-10-30",
        "pin:",
        *(f"  {repo}: {tag}" for repo, tag in pin.items()),
        "previous: null",
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def config_root(tmp_path: Path) -> Path:
    root = tmp_path / "workspace-config"
    _write_pin(root, RELEASE)
    return root


@pytest.fixture
def shims(tmp_path: Path) -> Path:
    """docker, curl, jq and the deploy command record and succeed; git records and runs."""
    shim_dir = tmp_path / "shims"
    shim_dir.mkdir()
    log = tmp_path / "calls.log"
    real_git = shutil.which("git")
    assert real_git
    for name in ("docker", "curl", "jq", "onex-runtime-deploy"):
        shim = shim_dir / name
        shim.write_text(f'#!/usr/bin/env bash\necho "{name} $*" >> "{log}"\nexit 0\n')
        shim.chmod(0o755)
    git_shim = shim_dir / "git"
    git_shim.write_text(
        f'#!/usr/bin/env bash\necho "git $*" >> "{log}"\nexec "{real_git}" "$@"\n'
    )
    git_shim.chmod(0o755)
    log.write_text("")
    return shim_dir


def _run_refresh(
    tmp_path: Path,
    clones: Path,
    config_root: Path | None,
    shims: Path,
    *args: str,
) -> subprocess.CompletedProcess[str]:
    env = _env()
    for key in (
        "OMNIBASE_INFRA_COMPOSE_PROJECT",
        "DEPLOY_SOURCE_CLONE_ROOT",
        "ONEX_WORKSPACE_CONFIG_ROOT",
    ):
        env.pop(key, None)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env.update(
        {
            "HOME": str(home),
            "OMNI_HOME": str(clones),
            "PATH": f"{shims}{os.pathsep}{env.get('PATH', '')}",
            "DEPLOY_RUNTIME": str(shims / "onex-runtime-deploy"),
            "TMPDIR": str(tmp_path),
        }
    )
    if config_root is not None:
        env["ONEX_WORKSPACE_CONFIG_ROOT"] = str(config_root)
    return subprocess.run(
        ["bash", str(REFRESH), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _assert_refused_before_any_change(
    result: subprocess.CompletedProcess[str], tmp_path: Path, named: str
) -> None:
    assert result.returncode == 64, result.stderr
    assert "release-tag-only" in result.stderr
    assert named in result.stderr
    calls = [c for c in (tmp_path / "calls.log").read_text().splitlines() if c]
    assert not [c for c in calls if c.startswith(("docker", "curl", "jq", "onex-"))]
    for verb in ("checkout", "fetch", "reset", "tag", "worktree", "clean"):
        assert not [c for c in calls if f" {verb}" in c], calls
    # No state directory, receipt or lock was written.
    assert not (tmp_path / "home" / ".omnibase").exists()


def test_stability_release_only_refuses_a_dev_tag(
    tmp_path: Path, clones: Path, config_root: Path, shims: Path
) -> None:
    result = _run_refresh(tmp_path, clones, config_root, shims, "--ref", DEV_TAG)
    _assert_refused_before_any_change(
        result, tmp_path, f"omnibase_infra ref '{DEV_TAG}'"
    )


def test_stability_release_only_refuses_a_bare_sha(
    tmp_path: Path, clones: Path, config_root: Path, shims: Path
) -> None:
    sha = _git(clones / "omnibase_infra", "rev-parse", "HEAD")
    result = _run_refresh(tmp_path, clones, config_root, shims, "--ref", sha)
    _assert_refused_before_any_change(result, tmp_path, f"omnibase_infra ref '{sha}'")


@pytest.mark.parametrize(
    "ref", ["origin/dev", "dev", "lab/stability/20260727T002130Z-5bb0c25d9b71"]
)
def test_stability_release_only_refuses_a_branch_or_lab_tag(
    tmp_path: Path, clones: Path, config_root: Path, shims: Path, ref: str
) -> None:
    result = _run_refresh(tmp_path, clones, config_root, shims, "--ref", ref)
    _assert_refused_before_any_change(result, tmp_path, f"omnibase_infra ref '{ref}'")


@pytest.mark.parametrize("omnimarket_ref", ["dev", "origin/dev", DEV_TAG])
def test_stability_release_only_refuses_a_dev_omnimarket_ref(
    tmp_path: Path, clones: Path, shims: Path, omnimarket_ref: str
) -> None:
    root = tmp_path / "dev-omnimarket-config"
    _write_pin(root, {**RELEASE, "omnimarket": omnimarket_ref})
    result = _run_refresh(tmp_path, clones, root, shims)
    _assert_refused_before_any_change(
        result, tmp_path, f"omnimarket ref '{omnimarket_ref}'"
    )


def test_stability_release_only_refuses_a_pin_naming_a_dev_sibling(
    tmp_path: Path, clones: Path, shims: Path
) -> None:
    root = tmp_path / "dev-core-config"
    _write_pin(root, {**RELEASE, "omnibase_core": DEV_TAG})
    result = _run_refresh(tmp_path, clones, root, shims)
    _assert_refused_before_any_change(
        result, tmp_path, f"omnibase_core ref '{DEV_TAG}'"
    )


def test_stability_release_only_has_no_dev_default(
    tmp_path: Path, clones: Path, shims: Path
) -> None:
    """No pin file and no --ref is refused; it no longer means origin/dev."""
    result = _run_refresh(tmp_path, clones, None, shims)
    _assert_refused_before_any_change(result, tmp_path, "omnibase_infra ref '<none>'")


def test_stability_release_only_refuses_a_missing_pin_file(
    tmp_path: Path, clones: Path, shims: Path
) -> None:
    result = _run_refresh(
        tmp_path, clones, None, shims, "--pin-file", str(tmp_path / "absent.yaml")
    )
    _assert_refused_before_any_change(result, tmp_path, "does not exist")


def test_stability_release_only_refuses_a_tag_missing_from_its_clone(
    tmp_path: Path, clones: Path, config_root: Path, shims: Path
) -> None:
    result = _run_refresh(tmp_path, clones, config_root, shims, "--ref", "v9.9.9")
    _assert_refused_before_any_change(result, tmp_path, "omnibase_infra ref 'v9.9.9'")


def test_stability_release_only_plans_the_pinned_release_set(
    tmp_path: Path, clones: Path, config_root: Path, shims: Path
) -> None:
    """The pinned set passes the guard; the plan names the infra tag and each sibling tag."""
    result = _run_refresh(tmp_path, clones, config_root, shims)
    assert "release-tag-only" not in result.stderr
    assert (
        "ref             : v0.38.74 (siblings: omnibase_core=v0.47.46 "
        "omnibase_compat=v0.5.7 omnimarket=v0.4.316)"
    ) in result.stderr


def test_stability_release_only_ref_overrides_the_pinned_infra_tag(
    tmp_path: Path, clones: Path, config_root: Path, shims: Path
) -> None:
    _git(clones / "omnibase_infra", "tag", "v0.38.76")
    result = _run_refresh(tmp_path, clones, config_root, shims, "--ref", "v0.38.76")
    assert "release-tag-only" not in result.stderr
    assert "ref             : v0.38.76 (siblings:" in result.stderr


def test_stability_release_only_guard_runs_before_every_mutation() -> None:
    text = REFRESH.read_text()
    guard = text.index("# --- release-only guard (OMN-21611)")
    for later in (
        'mkdir -p "${HISTORY_DIR}"',
        "lane_lock_acquire ",
        "ATTRIBUTION_PREFLIGHT=",
        'bash "${SCRIPT_DIR}/ensure_runner_clones.sh"',
        'docker tag "${image_tag}:latest"',
        "checkout --force --detach",
    ):
        assert guard < text.index(later), later
    assert 'REF="origin/dev"' not in text
    assert 'export DEPLOY_FOLLOW_DEV_TAG_REPOS="omnimarket"' not in text
    assert "export DEPLOY_SIBLING_REFS" in text


def test_stability_release_only_stage_workspace_pins_each_sibling() -> None:
    text = STAGE.read_text()
    assert 'read -r -a sibling_refs <<< "${DEPLOY_SIBLING_REFS:-}"' in text
    assert 'checkout_args+=(--repo-ref "${sibling_ref}")' in text


# ---------------------------------------------------------------------------
# Staging: each sibling is checked out at its pinned tag, never the fallback
# branch, and an explicit tag is never replaced by omnimarket's newest dev tag.
# These run the real stage_workspace.sh and deploy_source_ref.py.
# ---------------------------------------------------------------------------
_STAGE_DISTS = {
    "omnibase_core": "omnibase-core",
    "omnibase_compat": "omnibase-compat",
    "omnimarket": "omnimarket",
    "omnibase_infra": "omnibase-infra",
    "omnibase_spi": "omnibase-spi",
}
_STAGE_VERSION = "9.9.9"


def _stage_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env={**scrub_git_location_env(), "GIT_TERMINAL_PROMPT": "0"},
    )
    return result.stdout.strip()


def _stage_clones(tmp_path: Path) -> tuple[Path, dict[str, str], dict[str, str]]:
    """Clones whose dev branch moved past each sibling's release tag.

    Returns the clone root, each sibling's tag and the commit that tag names.
    """
    root = tmp_path / "stage_clones"
    tags: dict[str, str] = {}
    tagged: dict[str, str] = {}
    for repo, dist in _STAGE_DISTS.items():
        clone = root / repo
        clone.mkdir(parents=True)
        _stage_git(clone, "init", "-q", "-b", "dev")
        _stage_git(clone, "config", "user.email", "t@t.t")
        _stage_git(clone, "config", "user.name", "t")
        _stage_git(clone, "config", "tag.gpgsign", "false")
        _stage_git(clone, "config", "commit.gpgsign", "false")
        (clone / "pyproject.toml").write_text(
            f"[project]\nname = '{dist}'\nversion = '{_STAGE_VERSION}'\n"
        )
        if repo == "omnimarket":
            ladder = clone / "src" / "omnimarket" / "configs"
            ladder.mkdir(parents=True)
            (ladder / "routing_tiers.yaml").write_text("tiers: []\n")
            (clone / "uv.lock").write_text(
                "\n".join(
                    f'[[package]]\nname = "{d}"\nversion = "{_STAGE_VERSION}"\n'
                    for d in _STAGE_DISTS.values()
                )
            )
        _stage_git(clone, "add", "-A")
        _stage_git(clone, "commit", "-q", "-m", "release")
        tags[repo] = RELEASE.get(repo, "v0.23.8")
        _stage_git(clone, "tag", tags[repo])
        tagged[repo] = _stage_git(clone, "rev-parse", "HEAD")
        (clone / "after-release.txt").write_text("dev moved on\n")
        _stage_git(clone, "add", "-A")
        _stage_git(clone, "commit", "-q", "-m", "dev after the release")
        _stage_git(clone, "tag", DEV_TAG)
    return root, tags, tagged


def _run_stage(
    root: Path, build_ctx: Path, extra_env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    (build_ctx / "workspace").mkdir(parents=True, exist_ok=True)
    env = {
        **scrub_git_location_env(),
        "OMNI_HOME": str(root),
        "CONSUMER_LOCK": str(root / "omnimarket" / "uv.lock"),
        "DEPLOY_SOURCE_REFS_OUT": str(build_ctx.parent / "refs" / "expected.json"),
        "DEPLOY_SOURCE_WORKTREE_ROOT": str(build_ctx.parent / "source-trees"),
        "DEPLOY_SIBLING_FALLBACK_REF": "dev",
        **extra_env,
    }
    for key in ("DEPLOY_HOTPATCH", "ALLOW_UNPINNED_DEPLOY_SOURCE"):
        env.pop(key, None)
    return subprocess.run(
        ["bash", str(STAGE)],
        cwd=build_ctx,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _sibling_refs(tags: dict[str, str]) -> str:
    return " ".join(
        f"{repo}={tags[repo]}"
        for repo in ("omnibase_core", "omnibase_compat", "omnimarket")
    )


def test_stability_release_only_stage_checks_each_sibling_out_at_its_tag(
    tmp_path: Path,
) -> None:
    root, tags, tagged = _stage_clones(tmp_path)
    build_ctx = tmp_path / "ctx"
    result = _run_stage(
        root,
        build_ctx,
        {
            "DEPLOY_REF": tags["omnibase_infra"],
            "DEPLOY_SIBLING_REFS": _sibling_refs(tags),
        },
    )
    assert result.returncode == 0, result.stderr
    vcs = json.loads(
        (build_ctx / "workspace" / "sibling-vcs-provenance.json").read_text()
    )
    expected = json.loads((tmp_path / "refs" / "expected.json").read_text())
    for repo in ("omnibase_core", "omnibase_compat", "omnimarket"):
        assert vcs["siblings"][repo]["vcs_ref"] == tagged[repo], repo
        row = expected["repos"][repo]
        assert row["ref"] == tags[repo]
        assert row["expected_sha"] == tagged[repo]
        assert not row.get("fallback_from"), row


def test_stability_release_only_stage_keeps_omnimarket_tag_over_dev_follow(
    tmp_path: Path,
) -> None:
    """A dev-tag follow list cannot replace a pinned omnimarket release tag."""
    root, tags, tagged = _stage_clones(tmp_path)
    build_ctx = tmp_path / "ctx"
    result = _run_stage(
        root,
        build_ctx,
        {
            "DEPLOY_REF": tags["omnibase_infra"],
            "DEPLOY_SIBLING_REFS": _sibling_refs(tags),
            "DEPLOY_FOLLOW_DEV_TAG_REPOS": "omnimarket",
        },
    )
    assert result.returncode == 0, result.stderr
    vcs = json.loads(
        (build_ctx / "workspace" / "sibling-vcs-provenance.json").read_text()
    )
    assert vcs["siblings"]["omnimarket"]["vcs_ref"] == tagged["omnimarket"]


def test_stability_release_only_stage_without_sibling_refs_falls_back(
    tmp_path: Path,
) -> None:
    """Control: with no sibling refs the same staging lands on the dev branch."""
    root, tags, tagged = _stage_clones(tmp_path)
    build_ctx = tmp_path / "ctx"
    result = _run_stage(root, build_ctx, {"DEPLOY_REF": tags["omnibase_infra"]})
    assert result.returncode == 0, result.stderr
    vcs = json.loads(
        (build_ctx / "workspace" / "sibling-vcs-provenance.json").read_text()
    )
    assert vcs["siblings"]["omnimarket"]["vcs_ref"] != tagged["omnimarket"]
