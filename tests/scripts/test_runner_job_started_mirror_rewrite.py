# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Regression coverage for the C2/C2b/C2c fetch-only mirror rewrite mechanisms
in `docker/runners/runner-job-started.sh` (OMN-16063, OMN-16114).

OMN-16114 extends the existing uv git-dependency rewrite
(`wire_uv_git_mirror_rewrite`, C2b) to sibling `actions/checkout` steps
(`wire_sibling_checkout_mirror_rewrite`, C2c): a second checkout in the same
job with an explicit `repository:` different from the job's own repo, e.g.
dispatcher-route-coverage.yml's "Checkout omnimarket (sibling)" step. Those
previously went straight to github.com with zero mirror acceleration and were
the confirmed root cause of ~27% of `dispatcher-route-coverage` runs timing
out at the 30-minute job budget (see OMN-16114).

`insteadOf` has no server-side fallback: once a URL is rewritten, a fetch the
mirror cannot serve fails outright rather than retrying the un-rewritten URL
(documented in docker/runners/README-c2b-uv-git-mirror.md, re-verified for
the sibling-checkout case by test_exact_sha_absent_from_mirror_fails_open_not_a_silent_fallthrough
below). Every test here proves either (a) the rewrite is installed only when
the mirror can actually serve every ref the discovered job would request, or
(b) it is correctly withheld (fail-open, job runs on github.com exactly as it
does today) when that cannot be proven.

TEST STRATEGY. Two tiers:

  1. Fast, portable, function-level tests (the bulk of this file) that source
     only the function-definitions PREFIX of the hook script (everything
     before the `wire_pypi_cache || true` main-body marker) and call
     `wire_uv_git_mirror_rewrite` / `wire_sibling_checkout_mirror_rewrite` /
     `_c2_rewrite_flush` directly against a real local `git daemon`. These
     never touch the workspace-reset logic at the bottom of the script, so
     they are NOT gated on GNU `realpath -m` and run on Linux CI, `.200`, and
     a bare macOS dev box alike.
  2. One end-to-end subprocess test that runs the REAL top-level script
     exactly as production invokes it, proving the two new call sites are
     actually wired into the success path. This one DOES exercise the
     GNU-`realpath -m`-gated workspace-reset logic and is skipped on a BSD
     `realpath` host (same skip guard as
     test_runner_job_started_root_owned_debris.py) -- it runs for real on
     Linux CI and is not the surface these fail-open/gating tests depend on.

All mirror "hosts" here are a real `git daemon --export-all` serving local
bare repos over 127.0.0.1, mirroring exactly how the production daemon is
started (docker/runners/systemd/omninode-git-mirror-daemon.service): no
mocking of git itself, so the negative-probe assertions (missing SHA, missing
branch) exercise the real git wire protocol, not a stand-in.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK_SCRIPT = REPO_ROOT / "docker" / "runners" / "runner-job-started.sh"
_MAIN_BODY_MARKER = "\nwire_pypi_cache || true\n"

pytestmark = [pytest.mark.unit]


def _has_gnu_realpath_m() -> bool:
    try:
        result = subprocess.run(
            ["realpath", "-m", "."],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def _functions_only_script() -> str:
    """Everything in the hook script before the main body starts running.

    Sourcing this (instead of the whole file) lets tests call the mirror
    functions directly without the script's own `exit 0`/`exit 1` at the
    bottom terminating the test process, and without needing GNU
    `realpath -m` (only the main body's workspace-confinement check uses it).
    """
    text = HOOK_SCRIPT.read_text()
    idx = text.index(_MAIN_BODY_MARKER)
    assert idx > 0, (
        "expected marker 'wire_pypi_cache || true' not found in "
        f"{HOOK_SCRIPT} -- the function-defs/main-body boundary moved; "
        "update _MAIN_BODY_MARKER"
    )
    return text[:idx]


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _run_git(*args: str, cwd: Path, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
        env=env,
    )
    return result.stdout.strip()


def _make_mirror_repo(
    mirror_root: Path,
    repo: str,
    *,
    branch: str = "dev",
    files: dict[str, str] | None = None,
) -> str:
    """Creates or updates `${mirror_root}/${repo}.git`, a bare repo with one
    new commit on `branch`, serving-config applied exactly like
    `apply_mirror_serving_config` in git-mirror-refresh.sh (allowFilter /
    allowAnySHA1InWant -- without these the by-SHA probe this component
    depends on cannot succeed against a real daemon). Returns the commit SHA.

    Safe to call more than once for the same `repo` (e.g. to seed a second
    commit that references the first commit's SHA): each call uses its own
    scratch work dir and either clones a fresh bare repo or pushes into the
    existing one.
    """
    work = Path(tempfile.mkdtemp(prefix=f"{repo}-work-", dir=mirror_root))
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
    }
    _run_git("init", "--quiet", f"--initial-branch={branch}", cwd=work, env=env)
    for name, content in (files or {"README.md": f"{repo}\n"}).items():
        path = work / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        _run_git("add", name, cwd=work, env=env)
    _run_git("commit", "--quiet", "-m", "seed", cwd=work, env=env)
    sha = _run_git("rev-parse", "HEAD", cwd=work, env=env)

    bare = mirror_root / f"{repo}.git"
    if bare.exists():
        _run_git(
            "push",
            "--quiet",
            "--force",
            str(bare),
            f"{branch}:{branch}",
            cwd=work,
            env=env,
        )
    else:
        _run_git(
            "clone", "--quiet", "--bare", str(work), str(bare), cwd=mirror_root, env=env
        )
        _run_git("config", "uploadpack.allowFilter", "true", cwd=bare, env=env)
        _run_git("config", "uploadpack.allowAnySHA1InWant", "true", cwd=bare, env=env)
    return sha


@pytest.fixture
def mirror_daemon(tmp_path: Path) -> Iterator[tuple[str, int, Path]]:
    """A real `git daemon --export-all`, matching
    omninode-git-mirror-daemon.service's invocation shape, serving
    `tmp_path/mirror` on an ephemeral 127.0.0.1 port. Repos are added to the
    returned root by the individual tests via `_make_mirror_repo`."""
    if shutil.which("git-daemon") is None and shutil.which("git") is None:
        pytest.skip("git not available")

    mirror_root = tmp_path / "mirror"
    mirror_root.mkdir()
    port = _find_free_port()

    proc = subprocess.Popen(
        [
            "git",
            "daemon",
            "--verbose",
            "--reuseaddr",
            "--export-all",
            f"--base-path={mirror_root}",
            "--listen=127.0.0.1",
            f"--port={port}",
            str(mirror_root),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        # Readiness: the daemon logs "Ready to rock" once listening, but
        # polling the actual port is a more direct readiness signal and
        # avoids depending on a specific log string across git versions.
        deadline = time.monotonic() + 10
        ready = False
        while time.monotonic() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.2)
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    ready = True
                    break
            time.sleep(0.1)
        if not ready:
            proc.terminate()
            pytest.fail(f"git daemon did not start listening on 127.0.0.1:{port}")
        yield "127.0.0.1", port, mirror_root
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _base_env(
    tmp_path: Path,
    host: str,
    port: int,
    *,
    own_repo: str = "omnibase_infra",
    github_sha: str,
) -> dict[str, str]:
    github_env = tmp_path / "github_env"
    github_env.write_text("")
    return {
        **os.environ,
        "OMNI_GIT_MIRROR_HOST": host,
        "OMNI_GIT_MIRROR_PORT": str(port),
        "GITHUB_ENV": str(github_env),
        "GITHUB_REPOSITORY": f"OmniNode-ai/{own_repo}",
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_SHA": github_sha,
        "GITHUB_JOB": "test-job",
        "GITHUB_WORKFLOW_REF": f"OmniNode-ai/{own_repo}/.github/workflows/test.yml@refs/heads/dev",
        # Isolate from the uv-dependency mechanism unless a test opts in.
        "OMNI_GIT_MIRROR_REWRITE_DISABLE": "1",
    }


def _github_env_pairs(github_env_path: Path) -> tuple[int, list[tuple[str, str]]]:
    """Parses the flushed GIT_CONFIG_COUNT / GIT_CONFIG_KEY_i / VALUE_i lines
    into (count, [(key_string, value_string), ...])."""
    keys: dict[int, str] = {}
    values: dict[int, str] = {}
    count = 0
    for line in github_env_path.read_text().splitlines():
        if not line or "=" not in line:
            continue
        name, _, val = line.partition("=")
        if name == "GIT_CONFIG_COUNT":
            count = int(val)
        elif name.startswith("GIT_CONFIG_KEY_"):
            keys[int(name.removeprefix("GIT_CONFIG_KEY_"))] = val
        elif name.startswith("GIT_CONFIG_VALUE_"):
            values[int(name.removeprefix("GIT_CONFIG_VALUE_"))] = val
    pairs = [(keys[i], values[i]) for i in sorted(keys)]
    return count, pairs


def _run_functions(
    *,
    workspace_dir: Path,
    env: dict[str, str],
    calls: str,
    tmp_path: Path,
) -> subprocess.CompletedProcess[str]:
    """Sources the function-defs prefix of the real hook script, then runs
    `calls` (e.g. 'wire_sibling_checkout_mirror_rewrite "$1"; _c2_rewrite_flush')
    against it, with `workspace_dir` as $1."""
    driver = tmp_path / "driver.sh"
    driver.write_text(_functions_only_script() + "\n" + calls + "\n")
    return subprocess.run(
        ["bash", str(driver), str(workspace_dir)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _seed_own_repo_workspace(
    tmp_path: Path, host: str, port: int, own_repo: str, workflow_content: str
) -> tuple[Path, str]:
    """Builds the own-repo mirror (containing the workflow file at a known
    commit) and an empty local workspace git repo pointed at it, matching
    what `seed_workspace_from_mirror` would have left behind: a `.git` dir
    that can delta-fetch GITHUB_SHA from the own-repo mirror."""
    mirror_root = tmp_path / "mirror"
    mirror_root.mkdir(exist_ok=True)
    sha = _make_mirror_repo(
        mirror_root,
        own_repo,
        branch="dev",
        files={".github/workflows/test.yml": workflow_content},
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _run_git("init", "--quiet", cwd=workspace)
    return workspace, sha


# ---------------------------------------------------------------------------
# wire_sibling_checkout_mirror_rewrite (OMN-16114 C2c)
# ---------------------------------------------------------------------------


def test_branch_ref_present_on_mirror_installs_rewrite(tmp_path: Path) -> None:
    """The dominant real-world case: dispatcher-route-coverage.yml's
    `repository: OmniNode-ai/omnimarket` / `ref: dev` sibling checkout. When
    the mirror has that branch, the fetch-only rewrite must be installed."""
    host, port = "127.0.0.1", _find_free_port()
    mirror_root = tmp_path / "mirror"
    mirror_root.mkdir()

    proc = subprocess.Popen(
        [
            "git",
            "daemon",
            "--reuseaddr",
            "--export-all",
            f"--base-path={mirror_root}",
            f"--listen={host}",
            f"--port={port}",
            str(mirror_root),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.2)
                if s.connect_ex((host, port)) == 0:
                    break
            time.sleep(0.1)

        workflow = (
            "jobs:\n"
            "  test-job:\n"
            "    steps:\n"
            "      - name: Checkout omnimarket (sibling)\n"
            "        uses: actions/checkout@pin\n"
            "        with:\n"
            "          repository: OmniNode-ai/omnimarket\n"
            "          ref: dev\n"
            "          path: omnimarket\n"
        )
        workspace, sha = _seed_own_repo_workspace(
            tmp_path, host, port, "omnibase_infra", workflow
        )
        _make_mirror_repo(mirror_root, "omnimarket", branch="dev")

        env = _base_env(tmp_path, host, port, github_sha=sha)
        result = _run_functions(
            workspace_dir=workspace,
            env=env,
            calls='wire_sibling_checkout_mirror_rewrite "$1"; _c2_rewrite_flush',
            tmp_path=tmp_path,
        )
        assert result.returncode == 0, result.stderr

        count, pairs = _github_env_pairs(Path(env["GITHUB_ENV"]))
        assert count == 2, f"expected one fetch+push pair; got {result.stderr}"
        fetch_key = f"url.git://{host}:{port}/omnimarket.git.insteadOf"
        assert (fetch_key, "https://github.com/OmniNode-ai/omnimarket") in pairs
        push_key = "url.https://github.com/OmniNode-ai/omnimarket.pushInsteadOf"
        assert (push_key, "https://github.com/OmniNode-ai/omnimarket") in pairs
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_branch_ref_absent_from_mirror_fails_open(
    mirror_daemon: tuple[str, int, Path], tmp_path: Path
) -> None:
    """The mirror is reachable but has never heard of this repo (not in
    git_mirror.repos, or not yet mirrored) -- no rewrite, job stays on
    github.com exactly as it does today."""
    host, port, _mirror_root = mirror_daemon
    workflow = (
        "jobs:\n"
        "  test-job:\n"
        "    steps:\n"
        "      - name: Checkout unmirrored sibling\n"
        "        uses: actions/checkout@pin\n"
        "        with:\n"
        "          repository: OmniNode-ai/never-mirrored\n"
        "          ref: dev\n"
    )
    workspace, sha = _seed_own_repo_workspace(
        tmp_path, host, port, "omnibase_infra", workflow
    )
    # Deliberately do NOT create a "never-mirrored.git" bare repo on the daemon.

    env = _base_env(tmp_path, host, port, github_sha=sha)
    result = _run_functions(
        workspace_dir=workspace,
        env=env,
        calls='wire_sibling_checkout_mirror_rewrite "$1"; _c2_rewrite_flush',
        tmp_path=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    count, pairs = _github_env_pairs(Path(env["GITHUB_ENV"]))
    assert count == 0
    assert pairs == []
    assert "not advertised by" in result.stdout + result.stderr


def test_exact_sha_present_on_mirror_installs_rewrite(
    mirror_daemon: tuple[str, int, Path], tmp_path: Path
) -> None:
    host, port, mirror_root = mirror_daemon
    pinned_sha = _make_mirror_repo(mirror_root, "onex_change_control", branch="dev")
    workflow = (
        "jobs:\n"
        "  test-job:\n"
        "    steps:\n"
        "      - name: Checkout OCC exact pin\n"
        "        uses: actions/checkout@pin\n"
        "        with:\n"
        "          repository: OmniNode-ai/onex_change_control\n"
        f"          ref: {pinned_sha}\n"
    )
    workspace, sha = _seed_own_repo_workspace(
        tmp_path, host, port, "omnibase_infra", workflow
    )

    env = _base_env(tmp_path, host, port, github_sha=sha)
    result = _run_functions(
        workspace_dir=workspace,
        env=env,
        calls='wire_sibling_checkout_mirror_rewrite "$1"; _c2_rewrite_flush',
        tmp_path=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    count, pairs = _github_env_pairs(Path(env["GITHUB_ENV"]))
    assert count == 2, result.stderr
    fetch_key = f"url.git://{host}:{port}/onex_change_control.git.insteadOf"
    assert (fetch_key, "https://github.com/OmniNode-ai/onex_change_control") in pairs


def test_exact_sha_absent_from_mirror_fails_open_not_a_silent_fallthrough(
    mirror_daemon: tuple[str, int, Path], tmp_path: Path
) -> None:
    """OMN-16114 explicitly asked this to be VERIFIED, not assumed: does an
    exact-SHA checkout of a mirrored repo "fall through to origin" when the
    mirror lacks that one commit? It does not (insteadOf has no server-side
    fallback -- see README-c2b-uv-git-mirror.md). This test proves the
    MECHANISM's own defense against that fact: a SHA the mirror cannot serve
    must never receive a rewrite in the first place, because if it did, the
    checkout step would hard-fail with no recovery."""
    host, port, mirror_root = mirror_daemon
    _make_mirror_repo(mirror_root, "onex_change_control", branch="dev")
    missing_sha = "f" * 40  # syntactically valid, guaranteed absent from a fresh repo
    workflow = (
        "jobs:\n"
        "  test-job:\n"
        "    steps:\n"
        "      - name: Checkout OCC exact pin\n"
        "        uses: actions/checkout@pin\n"
        "        with:\n"
        "          repository: OmniNode-ai/onex_change_control\n"
        f"          ref: {missing_sha}\n"
    )
    workspace, sha = _seed_own_repo_workspace(
        tmp_path, host, port, "omnibase_infra", workflow
    )

    env = _base_env(tmp_path, host, port, github_sha=sha)
    result = _run_functions(
        workspace_dir=workspace,
        env=env,
        calls='wire_sibling_checkout_mirror_rewrite "$1"; _c2_rewrite_flush',
        tmp_path=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    count, pairs = _github_env_pairs(Path(env["GITHUB_ENV"]))
    assert count == 0, (
        "a SHA absent from the mirror must never receive a rewrite -- "
        f"insteadOf has no fallback, this would hard-fail the checkout step. got: {pairs}"
    )
    assert "not served by" in result.stdout + result.stderr


def test_conjunctive_gating_one_missing_sha_disqualifies_whole_repo(
    mirror_daemon: tuple[str, int, Path], tmp_path: Path
) -> None:
    """ci.yml's application-database-domain-enforcement job checks out
    omnibase_infra TWICE in the same job at two different pinned SHAs
    (.proof-dependencies/legacy-fixture, .proof-dependencies/domain-adapter).
    `insteadOf` rewrites the whole URL, not one ref -- so if only ONE of the
    two pins is servable, installing the rewrite would still redirect BOTH
    checkouts, hard-failing the unservable one. Neither may be rewritten
    unless the mirror can serve every pin discovered for that repo."""
    host, port, mirror_root = mirror_daemon
    present_sha = _make_mirror_repo(mirror_root, "omnibase_core", branch="dev")
    absent_sha = "a" * 40
    workflow = (
        "jobs:\n"
        "  test-job:\n"
        "    steps:\n"
        "      - name: Checkout dependency A\n"
        "        uses: actions/checkout@pin\n"
        "        with:\n"
        "          repository: OmniNode-ai/omnibase_core\n"
        f"          ref: {present_sha}\n"
        "          path: dep-a\n"
        "      - name: Checkout dependency B\n"
        "        uses: actions/checkout@pin\n"
        "        with:\n"
        "          repository: OmniNode-ai/omnibase_core\n"
        f"          ref: {absent_sha}\n"
        "          path: dep-b\n"
    )
    workspace, sha = _seed_own_repo_workspace(
        tmp_path, host, port, "omnibase_infra", workflow
    )

    env = _base_env(tmp_path, host, port, github_sha=sha)
    result = _run_functions(
        workspace_dir=workspace,
        env=env,
        calls='wire_sibling_checkout_mirror_rewrite "$1"; _c2_rewrite_flush',
        tmp_path=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    count, _pairs = _github_env_pairs(Path(env["GITHUB_ENV"]))
    assert count == 0, (
        "one unservable pin must disqualify the whole repo, not just that occurrence"
    )


def test_dynamic_gha_expression_ref_disqualifies_repo(
    mirror_daemon: tuple[str, int, Path], tmp_path: Path
) -> None:
    """ci.yml's application-database-domain-enforcement job resolves
    omnimarket's ref via an earlier step's output
    (`${{ steps.resolve-omnimarket-ref... }}`). That ref is unknowable at
    job-start scan time -- must never be treated as servable on a guess."""
    host, port, mirror_root = mirror_daemon
    _make_mirror_repo(mirror_root, "omnimarket", branch="dev")
    workflow = (
        "jobs:\n"
        "  test-job:\n"
        "    steps:\n"
        "      - name: Checkout exact typed registry dependency\n"
        "        uses: actions/checkout@pin\n"
        "        with:\n"
        "          repository: OmniNode-ai/omnimarket\n"
        "          ref: ${{ steps.resolve-omnimarket-ref.outputs.ref }}\n"
    )
    workspace, sha = _seed_own_repo_workspace(
        tmp_path, host, port, "omnibase_infra", workflow
    )

    env = _base_env(tmp_path, host, port, github_sha=sha)
    result = _run_functions(
        workspace_dir=workspace,
        env=env,
        calls='wire_sibling_checkout_mirror_rewrite "$1"; _c2_rewrite_flush',
        tmp_path=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    count, _pairs = _github_env_pairs(Path(env["GITHUB_ENV"]))
    assert count == 0


def test_own_repository_never_rewritten_even_if_self_referenced(
    mirror_daemon: tuple[str, int, Path], tmp_path: Path
) -> None:
    """ci.yml's application-database-domain-enforcement job also checks out
    omnibase_infra-the-repo (its OWN repo) twice more at older pinned SHAs,
    from a job running inside omnibase_infra. The primary checkout uses the
    identical no-.git-suffix URL form; a rewrite keyed on that URL cannot
    tell the two apart, and the primary checkout fetches GITHUB_SHA (a
    possibly seconds-old head commit the mirror may not have caught up to
    yet) with no fallback. own_repo must be excluded unconditionally,
    regardless of whether its pins would otherwise probe clean."""
    host, port, mirror_root = mirror_daemon
    own_repo = "omnibase_infra"
    # Build the own-repo mirror directly (not via _seed_own_repo_workspace,
    # since that also writes the workflow file at the SAME commit this
    # occurrence pins to -- proving even a servable, present pin of own_repo
    # is still excluded).
    workflow = (
        "jobs:\n"
        "  test-job:\n"
        "    steps:\n"
        "      - name: Checkout self at an older pin\n"
        "        uses: actions/checkout@pin\n"
        "        with:\n"
        "          repository: OmniNode-ai/omnibase_infra\n"
        "          ref: SELF_SHA_PLACEHOLDER\n"
        "          path: .proof-dependencies/legacy-fixture\n"
    )
    mirror_root.mkdir(exist_ok=True)
    sha = _make_mirror_repo(
        mirror_root,
        own_repo,
        branch="dev",
        files={".github/workflows/test.yml": workflow},
    )
    # The pin references a real, present, servable commit on own_repo's own
    # mirror (its own seed commit) -- if own_repo exclusion were broken, this
    # occurrence would probe clean and a rewrite would be installed.
    workflow_final = workflow.replace("SELF_SHA_PLACEHOLDER", sha)
    sha2 = _make_mirror_repo(
        mirror_root,
        own_repo,
        branch="dev",
        files={".github/workflows/test.yml": workflow_final},
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _run_git("init", "--quiet", cwd=workspace)

    env = _base_env(tmp_path, host, port, own_repo=own_repo, github_sha=sha2)
    result = _run_functions(
        workspace_dir=workspace,
        env=env,
        calls='wire_sibling_checkout_mirror_rewrite "$1"; _c2_rewrite_flush',
        tmp_path=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    count, _pairs = _github_env_pairs(Path(env["GITHUB_ENV"]))
    assert count == 0, "own_repo must never be a sibling-checkout rewrite target"


def test_kill_switch_disables_without_breaking_the_job(
    mirror_daemon: tuple[str, int, Path], tmp_path: Path
) -> None:
    host, port, mirror_root = mirror_daemon
    _make_mirror_repo(mirror_root, "omnimarket", branch="dev")
    workflow = (
        "jobs:\n"
        "  test-job:\n"
        "    steps:\n"
        "      - name: Checkout omnimarket (sibling)\n"
        "        uses: actions/checkout@pin\n"
        "        with:\n"
        "          repository: OmniNode-ai/omnimarket\n"
        "          ref: dev\n"
    )
    workspace, sha = _seed_own_repo_workspace(
        tmp_path, host, port, "omnibase_infra", workflow
    )

    env = _base_env(tmp_path, host, port, github_sha=sha)
    env["OMNI_GIT_MIRROR_CHECKOUT_REWRITE_DISABLE"] = "1"
    result = _run_functions(
        workspace_dir=workspace,
        env=env,
        calls='wire_sibling_checkout_mirror_rewrite "$1"; _c2_rewrite_flush',
        tmp_path=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    count, _pairs = _github_env_pairs(Path(env["GITHUB_ENV"]))
    assert count == 0


# ---------------------------------------------------------------------------
# Shared accumulator (both mechanisms flush through one GIT_CONFIG_COUNT)
# ---------------------------------------------------------------------------


def test_shared_accumulator_merges_uv_and_sibling_rewrites_without_clobbering(
    mirror_daemon: tuple[str, int, Path], tmp_path: Path
) -> None:
    """GIT_CONFIG_COUNT is a single flat namespace. If wire_uv_git_mirror_rewrite
    and wire_sibling_checkout_mirror_rewrite each wrote their own
    GIT_CONFIG_COUNT independently, the second write would silently drop the
    first mechanism's KEY_0/VALUE_0 -- this is the regression the shared
    _c2_rewrite_add_pair/_c2_rewrite_flush accumulator exists to prevent."""
    host, port, mirror_root = mirror_daemon
    occ_sha = _make_mirror_repo(mirror_root, "onex_change_control", branch="dev")
    _make_mirror_repo(mirror_root, "omnimarket", branch="dev")

    workflow = (
        "jobs:\n"
        "  test-job:\n"
        "    steps:\n"
        "      - name: Checkout omnimarket (sibling)\n"
        "        uses: actions/checkout@pin\n"
        "        with:\n"
        "          repository: OmniNode-ai/omnimarket\n"
        "          ref: dev\n"
    )
    uv_lock = f"""[[package]]
name = "onex-change-control"
source = {{ git = "https://github.com/OmniNode-ai/onex_change_control.git?rev={occ_sha}#{occ_sha}" }}
"""
    mirror_root.mkdir(exist_ok=True)
    sha = _make_mirror_repo(
        mirror_root,
        "omnibase_infra",
        branch="dev",
        files={
            ".github/workflows/test.yml": workflow,
            "uv.lock": uv_lock,
        },
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _run_git("init", "--quiet", cwd=workspace)

    env = _base_env(tmp_path, host, port, github_sha=sha)
    del env["OMNI_GIT_MIRROR_REWRITE_DISABLE"]  # opt the uv mechanism back in
    result = _run_functions(
        workspace_dir=workspace,
        env=env,
        calls=(
            'wire_uv_git_mirror_rewrite "$1"; '
            'wire_sibling_checkout_mirror_rewrite "$1"; '
            "_c2_rewrite_flush"
        ),
        tmp_path=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    count, pairs = _github_env_pairs(Path(env["GITHUB_ENV"]))
    assert count == 4, (
        f"expected 2 pairs (uv + sibling) = 4 entries, got {count}: {result.stderr}"
    )
    occ_fetch_key = f"url.git://{host}:{port}/onex_change_control.git.insteadOf"
    market_fetch_key = f"url.git://{host}:{port}/omnimarket.git.insteadOf"
    keys = [k for k, _ in pairs]
    assert occ_fetch_key in keys, (
        "uv-mechanism entry was clobbered by the sibling-checkout flush"
    )
    assert market_fetch_key in keys, (
        "sibling-mechanism entry was clobbered by the uv flush"
    )


# ---------------------------------------------------------------------------
# End-to-end wiring: the real top-level script actually calls both functions
# ---------------------------------------------------------------------------


def _has_realpath_gate() -> bool:
    return _has_gnu_realpath_m()


@pytest.mark.skipif(
    not _has_realpath_gate(),
    reason="full-script run needs GNU `realpath -m` (Ubuntu 22.04 runner image "
    "and Linux CI; absent on BSD/macOS) -- same gate as "
    "test_runner_job_started_root_owned_debris.py",
)
def test_full_script_success_path_installs_sibling_checkout_rewrite(
    mirror_daemon: tuple[str, int, Path], tmp_path: Path
) -> None:
    """Runs the REAL script (not the function-defs prefix) end to end,
    proving the two new call sites (wire_sibling_checkout_mirror_rewrite +
    _c2_rewrite_flush) are actually reached on the normal success path, after
    the workspace-reset logic this file's sibling test covers."""
    host, port, mirror_root = mirror_daemon
    _make_mirror_repo(mirror_root, "omnimarket", branch="dev")
    workflow = (
        "jobs:\n"
        "  test-job:\n"
        "    steps:\n"
        "      - name: Checkout omnimarket (sibling)\n"
        "        uses: actions/checkout@pin\n"
        "        with:\n"
        "          repository: OmniNode-ai/omnimarket\n"
        "          ref: dev\n"
    )
    mirror_root.mkdir(exist_ok=True)
    sha = _make_mirror_repo(
        mirror_root,
        "omnibase_infra",
        branch="dev",
        files={".github/workflows/test.yml": workflow},
    )

    runner_home = tmp_path / "actions-runner"
    workspace = runner_home / "_work" / "omnibase_infra" / "omnibase_infra"
    workspace.mkdir(parents=True)

    github_env = tmp_path / "github_env"
    github_env.write_text("")
    env = {
        **os.environ,
        "RUNNER_HOME": str(runner_home),
        "GITHUB_WORKSPACE": str(workspace),
        "OMNI_GIT_MIRROR_HOST": host,
        "OMNI_GIT_MIRROR_PORT": str(port),
        "GITHUB_ENV": str(github_env),
        "GITHUB_REPOSITORY": "OmniNode-ai/omnibase_infra",
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_SHA": sha,
        "GITHUB_JOB": "test-job",
        "GITHUB_WORKFLOW_REF": "OmniNode-ai/omnibase_infra/.github/workflows/test.yml@refs/heads/dev",
        "OMNI_GIT_MIRROR_REWRITE_DISABLE": "1",
    }
    result = subprocess.run(
        ["bash", str(HOOK_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    count, pairs = _github_env_pairs(github_env)
    assert count == 2, result.stderr
    fetch_key = f"url.git://{host}:{port}/omnimarket.git.insteadOf"
    assert (fetch_key, "https://github.com/OmniNode-ai/omnimarket") in pairs
