# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Run the pool's clone phase against local Git refs (OMN-21433)."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from tests.unit.scripts.test_prepr_runtime_pool_omn18893 import (
    CFG,
    DEV_FAILS,
    NOW,
    FakeHost,
    FakeTransport,
    pool,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
PROVE = REPO_ROOT / "scripts" / "runtime_build" / "prepr_pool_prove.sh"


def _git(repo: Path, *args: str, env: dict[str, str]) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        env=scrub_git_location_env(env),
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return result.stdout.strip()


@pytest.mark.parametrize(
    ("repo_var", "repo"), [("INFRA", "omnibase_infra"), ("MARKET", "omnimarket")]
)
@pytest.mark.parametrize("ref_kind", ["tag", "pull", "wrong-head"])
def test_clone_fetches_the_selected_ref_into_build_and_test_trees(
    tmp_path: Path, repo_var: str, repo: str, ref_kind: str
) -> None:
    # Every GitHub URL resolves to a disposable local repository. No git or shell
    # function is mocked: the real clone case runs fetch_pr and both expansions.
    home = tmp_path / "home"
    home.mkdir()
    env = scrub_git_location_env(os.environ)
    env.update(
        HOME=str(home),
        GIT_CONFIG_GLOBAL=str(home / ".gitconfig"),
        GIT_CONFIG_NOSYSTEM="1",
        GIT_AUTHOR_NAME="Pool Test",
        GIT_AUTHOR_EMAIL="pool-test@example.invalid",
        GIT_COMMITTER_NAME="Pool Test",
        GIT_COMMITTER_EMAIL="pool-test@example.invalid",
    )
    remote = tmp_path / "remote"
    remote.mkdir()
    _git(remote, "init", "-q", "--initial-branch=dev", env=env)
    (remote / "base.txt").write_text("dev\n", encoding="utf-8")
    _git(remote, "add", ".", env=env)
    _git(remote, "commit", "-qm", "base", env=env)
    base = _git(remote, "rev-parse", "HEAD", env=env)
    _git(remote, "branch", "main", env=env)
    _git(remote, "switch", "-qc", "candidate", env=env)
    (remote / "candidate.txt").write_text("candidate\n", encoding="utf-8")
    _git(remote, "add", ".", env=env)
    _git(remote, "commit", "-qm", "candidate", env=env)
    head = _git(remote, "rev-parse", "HEAD", env=env)
    tag = "lab/pr-example-42-" + head[:12]
    _git(remote, "tag", tag, env=env)
    # With a tag selected, a stale pull ref must not win in either clone.
    _git(
        remote,
        "update-ref",
        "refs/pull/42/head",
        head if ref_kind == "pull" else base,
        env=env,
    )
    _git(remote, "switch", "-q", "dev", env=env)
    for name in (
        "omnibase_infra",
        "omnimarket",
        "omnibase_compat",
        "omnibase_core",
        "omnibase_spi",
    ):
        _git(
            remote,
            "config",
            "--global",
            f"url.{remote}.insteadOf",
            f"https://github.com/OmniNode-ai/{name}.git",
            "--add",
            env=env,
        )
    work = tmp_path / "work"
    expected = base if ref_kind == "wrong-head" else head
    params = {
        "TAG": "local-tag-test",
        "W": str(work),
        f"{repo_var}_PR": "42",
        f"{repo_var}_HEAD": expected,
    }
    if ref_kind != "pull":
        params[f"{repo_var}_TAG"] = tag
    params_path = tmp_path / "params.env"
    params_path.write_text(
        "".join(f"{key}={shlex.quote(value)}\n" for key, value in params.items()),
        encoding="utf-8",
    )
    done = subprocess.run(
        ["bash", str(PROVE), str(params_path), "clone"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    match = "NO" if ref_kind == "wrong-head" else "yes"
    assert (
        f"{repo}#42 fetched head {head} expected {expected} match={match}"
        in done.stdout
    )
    assert "clone done" in done.stdout
    # The test tree is the raw selected head; the build tree merges that same
    # head onto dev. These reads fail if a fetch is absent or chooses pull/42.
    assert _git(work / "test" / repo, "rev-parse", "HEAD", env=env) == head
    assert _git(work / "root" / repo, "rev-parse", "HEAD^1", env=env) == base
    assert _git(work / "root" / repo, "rev-parse", "HEAD^2", env=env) == head
    assert (work / "root" / repo / "candidate.txt").read_text() == "candidate\n"
    readback = pool.judge({"clone": done.stdout})
    assert readback.checks["head_matches"] is (ref_kind != "wrong-head")


def test_the_base_control_drops_tags_and_pr_identity_before_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    params_path = tmp_path / "params.env"
    params_path.write_text(
        "INFRA_PR=42\nINFRA_HEAD=abc\nINFRA_TAG=lab/infra\n"
        "MARKET_PR=43\nMARKET_HEAD=def\nMARKET_TAG=lab/market\n"
        "CORE_REF=v0.47.31\nTESTS=omnibase_infra:tests/unit/test_x.py\n",
        encoding="utf-8",
    )
    host = FakeHost()
    host.phases["probe"] = DEV_FAILS
    host.base_phases["probe"] = DEV_FAILS
    calls: list[dict[str, str]] = []
    run_stack = pool._run_stack

    def record(*args: Any, **kwargs: Any) -> dict[str, str]:
        calls.append(dict(args[3]))
        return run_stack(*args, **kwargs)

    monkeypatch.setattr(pool, "_run_stack", record)
    code, text = pool.run_proof(
        CFG,
        FakeTransport({"pool-a": host}),
        params_path,
        "me",
        90,
        "pool-a",
        [],
        now_fn=lambda: NOW,
        log=lambda _: None,
        with_base_control=True,
    )
    assert code == pool.EXIT_PASS, text
    assert len(calls) == 2
    assert calls[0]["INFRA_TAG"] == "lab/infra"
    assert calls[0]["MARKET_TAG"] == "lab/market"
    assert calls[1] == {"CORE_REF": "v0.47.31"}
    assert "base control run at dev" in text
    assert host.lease is None
