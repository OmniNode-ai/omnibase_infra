# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for HandlerActionArchiveCacheFill (OMN-24299)."""

from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.handlers.handler_action_archive_cache_fill import (
    HandlerActionArchiveCacheFill,
)
from omnibase_infra.models.model_action_archive_cache_fill_request import (
    ModelActionArchiveCacheFillRequest,
)
from omnibase_infra.nodes.node_deploy_runners.runtime_action_archive_cache_fill import (
    main as entry_main,
)


def _git(*args: str, cwd: Path | None = None) -> str:
    done = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=True,
        cwd=cwd,
        env=scrub_git_location_env(os.environ),
    )
    return done.stdout.strip()


def _make_bare_remote(tmp: Path, owner: str, repo: str, tag: str) -> Path:
    """Create a bare git repo with a tagged commit containing a couple of files."""
    work = tmp / f"work-{owner}-{repo}"
    work.mkdir()
    _git("init", "-b", "main", cwd=work)
    _git("config", "user.email", "t@t", cwd=work)
    _git("config", "user.name", "t", cwd=work)
    (work / "a.txt").write_text("alpha")
    (work / "b.txt").write_text("beta")
    _git("add", ".", cwd=work)
    _git("commit", "-m", "init", cwd=work)
    _git("tag", tag, cwd=work)
    bare = tmp / owner / f"{repo}.git"
    bare.parent.mkdir(parents=True, exist_ok=True)
    _git("clone", "--bare", str(work), str(bare))
    return bare


def _make_clone(tmp: Path, name: str, workflow_text: str) -> Path:
    """Create a canonical clone with a .github/workflows/ci.yml."""
    clone = tmp / name
    clone.mkdir(parents=True)
    _git("init", "-b", "main", cwd=clone)
    _git("config", "user.email", "t@t", cwd=clone)
    _git("config", "user.name", "t", cwd=clone)
    wf_dir = clone / ".github" / "workflows"
    wf_dir.mkdir(parents=True)
    (wf_dir / "ci.yml").write_text(workflow_text)
    _git("add", ".", cwd=clone)
    _git("commit", "-m", "init", cwd=clone)
    # Set origin/HEAD symbolic ref
    _git("remote", "add", "origin", str(clone), cwd=clone)
    _git("symbolic-ref", "refs/remotes/origin/HEAD", "refs/heads/main", cwd=clone)
    return clone


def _make_fleet_yaml(
    tmp: Path, staging_root: Path, clone_root: Path, local_owner: str = "OmniNode-ai"
) -> Path:
    """Write a minimal valid runner_fleet.yaml with action_archive_cache block."""
    cfg = {
        "version": "1",
        "github_org": "test",
        "runner_host": "test-host",
        "runner_group": "test",
        "runner_name_prefix": "test-runner",
        "expected_count": 1,
        "git_mirror": {
            "active": False,
            "bind_address": "127.0.0.1",
            "docker_network": "n",
            "port": 9418,
            "mirror_root": "/m",
            "refresh_interval_seconds": 60,
            "serialized": True,
            "repos": ["omnibase_infra"],
            "kill_switch_env": "K",
        },
        "action_archive_cache": {
            "active": True,
            "staging_root": str(staging_root),
            "container_path": "/opt/onex-action-archives",
            "local_owner": local_owner,
            "remote_url_template": f"file://{tmp}/{{repo}}.git",
            "fetch_timeout_seconds": 30,
            "refresh_interval_seconds": 3600,
        },
    }
    path = tmp / "runner_fleet.yaml"
    path.write_text(yaml.dump(cfg))
    return path


def _setup_fixtures(tmp: Path) -> dict[str, Any]:
    """Build all fixture repos and return a dict of paths."""
    clone_root = tmp / "clones"
    staging_root = tmp / "staging"
    staging_root.mkdir()

    # Third-party bare remotes
    checkout_bare = _make_bare_remote(tmp, "actions", "checkout", "v4")
    _make_bare_remote(tmp, "astral-sh", "setup-uv", "v7")
    _make_bare_remote(tmp, "actions", "cache", "v6")

    # Local owner clone (OmniNode-ai/localaction)
    localaction = _make_clone(tmp / "clones", "localaction", "")

    # Main workflow source clone
    workflow_text = """
name: CI
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/checkout@{sha}
      - uses: astral-sh/setup-uv@v7
      - uses: actions/cache/restore@v6
      - uses: ./local
      - uses: docker://x
      - uses: OmniNode-ai/x/.github/workflows/y.yml@abc
      - uses: OmniNode-ai/localaction@main
      - uses: OmniNode-ai/localaction/.github/actions/y@<sha>
"""
    # The pinned form of actions/checkout: the sha of the commit v4 points at.
    sha = _git("rev-parse", "v4^{commit}", cwd=checkout_bare)
    workflow_text = workflow_text.replace("{sha}", sha)

    main_clone = _make_clone(tmp / "clones", "omnibase_infra", workflow_text)

    return {
        "clone_root": clone_root,
        "staging_root": staging_root,
        "main_clone": main_clone,
        "pinned_sha": sha,
        "expected": {
            f"actions/checkout@{sha}",
            f"astral-sh/setup-uv@{_git('rev-parse', 'v7^{commit}', cwd=tmp / 'astral-sh' / 'setup-uv.git')}",
            f"actions/cache@{_git('rev-parse', 'v6^{commit}', cwd=tmp / 'actions' / 'cache.git')}",
            f"OmniNode-ai/localaction@{_git('rev-parse', 'main', cwd=localaction)}",
        },
    }


def _fill(
    tmp: Path, fx: dict[str, Any], *, dry_run: bool = False, config: Path | None = None
):
    return HandlerActionArchiveCacheFill().handle(
        ModelActionArchiveCacheFillRequest(
            config_path=str(
                config or _make_fleet_yaml(tmp, fx["staging_root"], fx["clone_root"])
            ),
            clone_root=str(fx["clone_root"]),
            dry_run=dry_run,
        )
    )


def _runner_lookup_path(staging: Path, key: str) -> Path:
    """Where the runner looks: <cache>/<owner>_<repo>/<resolved sha>.tar.gz."""
    owner_repo, sha = key.split("@", 1)
    return staging / owner_repo.replace("/", "_") / f"{sha}.tar.gz"


def test_golden_first_run_fills_one_archive_per_resolved_sha(tmp_path: Path) -> None:
    fx = _setup_fixtures(tmp_path)
    result = _fill(tmp_path, fx)

    assert result.success is True
    assert result.failed == ()
    # The tag v4 and the pinned sha are one commit: one archive, the second ref finds it.
    assert set(result.filled) | set(result.present) == fx["expected"]
    assert len(result.filled) == 4
    assert (
        result.present == (f"actions/checkout@{fx['pinned_sha']}",)
        or result.present == ()
    )
    # Skipped references never become archives.
    for key in fx["expected"]:
        assert (
            "/.github/workflows/" not in key and "docker" not in key and "./" not in key
        )
    for key in result.filled:
        archive = _runner_lookup_path(fx["staging_root"], key)
        owner_repo, sha = key.split("@", 1)
        with tarfile.open(archive) as tf:
            names = tf.getnames()
        assert {n.split("/")[0] for n in names} == {
            f"{owner_repo.replace('/', '-')}-{sha[:7]}"
        }
        if not key.startswith("OmniNode-ai/"):
            assert any(n.endswith("/a.txt") for n in names)


def test_second_run_finds_everything_present_and_writes_nothing(tmp_path: Path) -> None:
    fx = _setup_fixtures(tmp_path)
    assert _fill(tmp_path, fx).success is True
    before = {p: p.stat().st_mtime_ns for p in fx["staging_root"].rglob("*.tar.gz")}
    again = _fill(tmp_path, fx)
    assert again.success is True
    assert again.filled == ()
    assert set(again.present) == fx["expected"]
    assert {
        p: p.stat().st_mtime_ns for p in fx["staging_root"].rglob("*.tar.gz")
    } == before


def test_seam_every_ref_lands_where_the_runner_looks(tmp_path: Path) -> None:
    """The workflow ref -> archive seam: sha read independently of the handler."""
    fx = _setup_fixtures(tmp_path)
    assert _fill(tmp_path, fx).success is True
    for key in fx["expected"]:
        assert _runner_lookup_path(fx["staging_root"], key).is_file(), key
    layout = sorted(
        p.name for p in fx["staging_root"].iterdir() if not p.name.startswith(".")
    )
    assert layout == [
        "OmniNode-ai_localaction",
        "actions_cache",
        "actions_checkout",
        "astral-sh_setup-uv",
    ]


def test_error_chain_unreachable_source_names_the_ref_and_the_rest_is_filled(
    tmp_path: Path,
) -> None:
    fx = _setup_fixtures(tmp_path)
    config = _make_fleet_yaml(tmp_path, fx["staging_root"], fx["clone_root"])
    data = yaml.safe_load(config.read_text())
    data["action_archive_cache"]["remote_url_template"] = (
        f"file://{tmp_path}/nowhere/{{repo}}.git"
    )
    config.write_text(yaml.dump(data))

    result = _fill(tmp_path, fx, config=config)

    assert result.success is False
    unreachable = {
        f.ref for f in result.failed if f.reason_code == "SOURCE_UNREACHABLE"
    }
    assert "actions/checkout@v4" in unreachable
    assert {f.reason_code for f in result.failed} == {"SOURCE_UNREACHABLE"}
    # The clone-backed ref is still filled, and no partial file is left behind.
    assert [k for k in result.filled if k.startswith("OmniNode-ai/localaction@")]
    assert list(fx["staging_root"].rglob("*.partial")) == []


def test_error_chain_unknown_tag_is_ref_unresolved(tmp_path: Path) -> None:
    fx = _setup_fixtures(tmp_path)
    workflow = fx["main_clone"] / ".github" / "workflows" / "ci.yml"
    workflow.write_text(
        workflow.read_text().replace("checkout@v4", "checkout@no-such-tag")
    )
    _git("add", ".", cwd=fx["main_clone"])
    _git("commit", "-m", "update", cwd=fx["main_clone"])

    result = _fill(tmp_path, fx)

    assert result.success is False
    assert [(f.ref, f.reason_code) for f in result.failed] == [
        ("actions/checkout@no-such-tag", "REF_UNRESOLVED")
    ]
    assert len(result.filled) + len(result.present) == 4


def test_error_chain_missing_workflow_source_is_named(tmp_path: Path) -> None:
    fx = _setup_fixtures(tmp_path)
    shutil.rmtree(fx["clone_root"] / "omnibase_infra")
    result = _fill(tmp_path, fx)
    assert result.success is False
    assert [(f.ref, f.reason_code) for f in result.failed] == [
        ("omnibase_infra", "WORKFLOW_SOURCE_MISSING")
    ]
    assert result.filled == ()


def test_error_chain_config_without_the_block_is_refused(tmp_path: Path) -> None:
    fx = _setup_fixtures(tmp_path)
    config = tmp_path / "bare_fleet.yaml"
    config.write_text(
        yaml.dump(
            {
                "version": "1",
                "github_org": "test",
                "runner_host": "test-host",
                "runner_group": "test",
                "runner_name_prefix": "test-runner",
                "expected_count": 1,
            }
        )
    )
    result = _fill(tmp_path, fx, config=config)
    assert result.success is False
    assert [f.reason_code for f in result.failed] == ["CONFIG_ABSENT"]
    assert list(fx["staging_root"].iterdir()) == []


def test_dry_run_lists_what_it_would_write_and_writes_nothing(tmp_path: Path) -> None:
    fx = _setup_fixtures(tmp_path)
    result = _fill(tmp_path, fx, dry_run=True)
    assert result.success is True
    assert set(result.filled) | set(result.present) == fx["expected"]
    assert list(fx["staging_root"].rglob("*.tar.gz")) == []


def test_failing_control_wrong_layout_file_is_not_a_cache_hit(tmp_path: Path) -> None:
    """A <sha>.tar.gz directly under the root is not where the runner looks."""
    fx = _setup_fixtures(tmp_path)
    (fx["staging_root"] / f"{fx['pinned_sha']}.tar.gz").write_bytes(b"wrong layout")
    result = _fill(tmp_path, fx)
    assert f"actions/checkout@{fx['pinned_sha']}" in result.filled
    assert _runner_lookup_path(
        fx["staging_root"], f"actions/checkout@{fx['pinned_sha']}"
    ).is_file()


def test_entry_exit_codes_golden_error_and_absent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fx = _setup_fixtures(tmp_path)
    config = _make_fleet_yaml(tmp_path, fx["staging_root"], fx["clone_root"])
    argv = ["--config", str(config), "--clone-root", str(fx["clone_root"])]
    assert entry_main(argv) == 0
    assert capsys.readouterr().out.startswith("[action-archive-cache-fill] ")
    shutil.rmtree(fx["clone_root"] / "omnibase_infra")
    assert entry_main(argv) == 1
    assert (
        entry_main(
            [
                "--config",
                str(tmp_path / "missing.yaml"),
                "--clone-root",
                str(fx["clone_root"]),
            ]
        )
        == 2
    )
