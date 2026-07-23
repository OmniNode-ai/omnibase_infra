# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Seeded-failure tests for the OMN-14900 deploy-runner fix.

Live failure being eliminated (five red release-train-lab.yml runs,
2026-07-21T03:59-04:19Z): the omninode-deploy-runner container bind-mounted
the SHARED /data/omninode/omni_home clones, which are owned by a different
uid. Every git touch from the runner failed in one of three modes:

  mode 1: ``fatal: detected dubious ownership in repository at ...`` (128)
  mode 2: ``error: cannot open .git/FETCH_HEAD: Permission denied`` (255)
  mode 3: ``fatal: 'dev' is already checked out at .../runtime-sync-worktrees``

The mode-1 relief applied live was an exec'd ``git config --global --add
safe.directory`` inside the running container -- which dies on the next
``--force-recreate`` because /home/runner is not a volume. That non-durable
relief is exactly what these tests exist to prevent coming back.

The committed, recreate-durable fix has three legs, each guarded here:

  1. PRIVATE OMNI_HOME: the deploy runner mounts its own uid-owned clones
     (``DEPLOY_RUNNER_OMNI_HOME``) at an identical container path, and the
     shared ``${OMNI_HOME}`` bind is REMOVED from the service -- writes into
     the shared clones become structurally impossible, not a discipline.
     Guarded by the compose config-as-data tests (which FAIL against the
     pre-fix compose file that still carries the shared bind).
  2. Scoped ``git -c safe.directory=<clone>`` on every git invocation the
     runner executes in scripts/runtime_build/ -- defense in depth that needs
     no container-global state. Guarded by static source tests (which FAIL
     against the pre-fix refresh_stability_lane.sh, cut_release_train_tag.sh,
     stage_workspace.sh, deploy_source_ref.py, check_sibling_lock_pins.py)
     and by a behavioral RED/GREEN pair driven with git's own
     ``GIT_TEST_ASSUME_DIFFERENT_OWNER`` ownership-mismatch knob.
  3. Automatic provisioning: ensure_runner_clones.sh creates the 5 private
     clones idempotently at the top of every entry script (behavioral tests
     against file:// bare fixtures).

Per reference_git_env_vars_override_c_and_cwd: strip GIT_DIR/GIT_INDEX_FILE/
GIT_WORK_TREE from subprocess env so an inherited pre-push hook export cannot
redirect these git operations onto the real worktree.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_BUILD = REPO_ROOT / "scripts" / "runtime_build"
COMPOSE_FILE = REPO_ROOT / "docker" / "docker-compose.runners.yml"
ENTRYPOINT = REPO_ROOT / "docker" / "runners" / "entrypoint.sh"
ENSURE_SCRIPT = RUNTIME_BUILD / "ensure_runner_clones.sh"

# Every shell script the deploy runner executes against OMNI_HOME clones.
RUNNER_SHELL_SCRIPTS = (
    "refresh_stability_lane.sh",
    "refresh_dev_lane.sh",
    "cut_release_train_tag.sh",
    "stage_workspace.sh",
    "ensure_runner_clones.sh",
)

_HERMETIC_ENV = {
    k: v
    for k, v in os.environ.items()
    if k not in {"GIT_DIR", "GIT_INDEX_FILE", "GIT_WORK_TREE"}
}


def _run(
    args: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        env=env if env is not None else _HERMETIC_ENV,
        capture_output=True,
        text=True,
        check=check,
    )


def _git(
    args: list[str], cwd: Path, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return _run(["git", *args], cwd=cwd, check=check)


# ---------------------------------------------------------------------------
# 1. Static source guards: no bare `git -C` against OMNI_HOME clones
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script_name", RUNNER_SHELL_SCRIPTS)
def test_no_bare_git_dash_c_in_runner_shell_scripts(script_name: str) -> None:
    """FAILS pre-fix: refresh_stability_lane.sh, cut_release_train_tag.sh and
    stage_workspace.sh all ran bare ``git -C`` against ``${OMNI_HOME}`` clones,
    which dies with 'dubious ownership' the moment the invoking uid differs
    from the clone owner. Every git call must go through the scoped
    ``git -c safe.directory=<clone> -C <clone>`` shape (the git_clone()
    wrapper, or an inline ``-c safe.directory=`` injection).
    """
    script = RUNTIME_BUILD / script_name
    assert script.is_file(), f"missing script: {script}"
    text = script.read_text()
    assert "git -C" not in text, (
        f"{script_name} contains a bare `git -C` invocation -- it will fail "
        f"with 'detected dubious ownership' whenever the invoking uid does not "
        f"own the clone. Route it through the scoped "
        f"`git -c safe.directory=<clone> -C <clone>` shape (OMN-14900)."
    )


@pytest.mark.parametrize(
    "script_name",
    ["refresh_stability_lane.sh", "refresh_dev_lane.sh", "cut_release_train_tag.sh"],
)
def test_entry_scripts_define_scoped_git_clone_wrapper(script_name: str) -> None:
    """Each entry script must carry the scoped git wrapper and the automatic
    private-clone provisioning call (ensure_runner_clones.sh) -- provisioning
    is a committed step in the script, never a remembered manual host step.
    """
    text = (RUNTIME_BUILD / script_name).read_text()
    assert "git_clone()" in text, f"{script_name}: missing git_clone() wrapper"
    assert '-c "safe.directory=${clone}"' in text, (
        f"{script_name}: git_clone() wrapper lost its safe.directory scoping"
    )
    assert "ensure_runner_clones.sh" in text, (
        f"{script_name}: missing the ensure_runner_clones.sh provisioning call"
    )


def test_stage_workspace_probes_are_safe_directory_scoped() -> None:
    """stage_workspace.sh's own vcs-provenance probes (rev-parse HEAD /
    --abbrev-ref / status --porcelain) must carry the inline scoping.
    """
    text = (RUNTIME_BUILD / "stage_workspace.sh").read_text()
    assert text.count('git -c "safe.directory=${src}" -C "${src}"') >= 3, (
        "stage_workspace.sh: the three vcs-provenance git probes must be "
        "safe.directory-scoped (OMN-14900)"
    )


@pytest.mark.parametrize(
    "script_name", ["deploy_source_ref.py", "check_sibling_lock_pins.py"]
)
def test_python_git_helpers_inject_safe_directory(script_name: str) -> None:
    """FAILS pre-fix: deploy_source_ref.py's _git()/_resolve_commit() (the RT-1
    write surface stage_workspace.sh drives) and check_sibling_lock_pins.py's
    probes built bare ``["git", "-C", ...]`` argv -- they only worked in the
    live container via the exec-applied global gitconfig / container-env
    GIT_CONFIG_* pass-through, both of which a subprocess env override or a
    container recreate silently drops.
    """
    text = (RUNTIME_BUILD / script_name).read_text()
    normalized = "".join(text.split())
    assert '["git","-C"' not in normalized, (
        f"{script_name}: bare git -C argv found -- inject "
        f'"-c", f"safe.directory={{...}}" before "-C" (OMN-14900)'
    )
    assert "safe.directory=" in text, (
        f"{script_name}: no safe.directory injection found"
    )


def test_entrypoint_runnergroup_is_conditional() -> None:
    """The runner entrypoint must only pass --runnergroup when RUNNER_GROUP is
    non-empty: the deploy runner is registered at the REPOSITORY level, where
    ``config.sh --runnergroup`` hard-fails, bricking re-registration after a
    recreate (OMN-14900 recon finding).
    """
    text = ENTRYPOINT.read_text()
    assert 'if [[ -n "${RUNNER_GROUP}" ]]' in text, (
        "entrypoint.sh: --runnergroup must be conditional on RUNNER_GROUP "
        "being non-empty (repo-scoped registration rejects it)"
    )
    # No-colon default: an explicitly-set empty RUNNER_GROUP must NOT be
    # replaced by the omnibase-ci default (that is the opt-out signal).
    assert 'RUNNER_GROUP="${RUNNER_GROUP-omnibase-ci}"' in text, (
        "entrypoint.sh: RUNNER_GROUP default must use ${VAR-default} (no "
        "colon) so an explicit empty value survives as the repo-scoped opt-out"
    )


# ---------------------------------------------------------------------------
# 2. Behavioral mode-1 RED/GREEN with real git
# ---------------------------------------------------------------------------


def _ownership_knob_supported(repo: Path) -> bool:
    """Probe whether this git honours GIT_TEST_ASSUME_DIFFERENT_OWNER."""
    env = {**_HERMETIC_ENV, "GIT_TEST_ASSUME_DIFFERENT_OWNER": "1"}
    probe = _run(["git", "-C", str(repo), "status"], env=env)
    return probe.returncode != 0 and "dubious ownership" in probe.stderr.lower()


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "clone"
    _git(["init", "-q", "-b", "dev", str(repo)], cwd=tmp_path)
    _git(
        [
            "-c",
            "user.email=t@t.com",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "c1",
        ],
        cwd=repo,
    )
    return repo


def test_mode1_red_bare_git_fails_on_ownership_mismatch(tmp_path: Path) -> None:
    """RED: the exact live failure -- a bare ``git -C <clone>`` dies with
    'detected dubious ownership' when the invoking uid does not own the clone
    (forced here via git's own GIT_TEST_ASSUME_DIFFERENT_OWNER test knob,
    which needs no root/chown).
    """
    repo = _make_repo(tmp_path)
    if not _ownership_knob_supported(repo):
        pytest.skip(
            "this git does not honour GIT_TEST_ASSUME_DIFFERENT_OWNER; "
            "static source guards remain the floor"
        )
    env = {**_HERMETIC_ENV, "GIT_TEST_ASSUME_DIFFERENT_OWNER": "1"}
    result = _run(["git", "-C", str(repo), "rev-parse", "HEAD"], env=env)
    assert result.returncode != 0
    assert "dubious ownership" in result.stderr.lower()


def test_mode1_green_scoped_safe_directory_succeeds(tmp_path: Path) -> None:
    """GREEN: the exact command shape the fixed scripts use --
    ``git -c safe.directory=<clone> -C <clone>`` -- succeeds under the
    identical ownership-mismatch condition that fails above, with no global
    gitconfig write and no container state.
    """
    repo = _make_repo(tmp_path)
    if not _ownership_knob_supported(repo):
        pytest.skip(
            "this git does not honour GIT_TEST_ASSUME_DIFFERENT_OWNER; "
            "static source guards remain the floor"
        )
    env = {**_HERMETIC_ENV, "GIT_TEST_ASSUME_DIFFERENT_OWNER": "1"}
    result = _run(
        [
            "git",
            "-c",
            f"safe.directory={repo}",
            "-C",
            str(repo),
            "rev-parse",
            "HEAD",
        ],
        env=env,
    )
    assert result.returncode == 0, (
        f"scoped safe.directory invocation must survive an ownership "
        f"mismatch: {result.stderr!r}"
    )
    assert result.stdout.strip()


# ---------------------------------------------------------------------------
# 3. Compose config-as-data: recreate-durability lives in the COMMITTED file
# ---------------------------------------------------------------------------


def _deploy_runner_service() -> dict[str, Any]:
    data = yaml.safe_load(COMPOSE_FILE.read_text())
    svc = data["services"]["omninode-deploy-runner"]
    assert isinstance(svc, dict)
    return svc


def test_compose_deploy_runner_mounts_private_omni_home_only() -> None:
    """FAILS pre-fix: the service bind-mounted the SHARED ``${OMNI_HOME}``
    clones. The committed service must mount ONLY the private
    DEPLOY_RUNNER_OMNI_HOME (identical container path, required for
    docker-outside-of-docker path resolution) -- removing the shared mount is
    what makes shared-clone writes structurally impossible.
    """
    svc = _deploy_runner_service()
    volumes = [str(v) for v in svc["volumes"]]
    shared = [v for v in volumes if "${OMNI_HOME}" in v]
    assert not shared, (
        f"deploy runner still mounts the SHARED OMNI_HOME clones: {shared} -- "
        f"this reopens the uid-mismatch write surface OMN-14900 closed"
    )
    private = [v for v in volumes if "DEPLOY_RUNNER_OMNI_HOME" in v]
    assert len(private) == 1, (
        f"expected exactly one DEPLOY_RUNNER_OMNI_HOME bind, got: {private}"
    )
    # Identical host:container path (fail-fast interpolation on both sides).
    host, sep, container = private[0].rpartition(":${")
    assert sep, f"unparseable private bind: {private[0]}"
    assert host.startswith("${DEPLOY_RUNNER_OMNI_HOME"), private[0]
    assert container.startswith("DEPLOY_RUNNER_OMNI_HOME"), private[0]


def test_compose_deploy_runner_env_is_fail_fast_and_repo_scoped() -> None:
    """The committed env must carry the live-verified reality so a recreate
    from THIS file cannot silently drop it: repo-scoped GITHUB_ORG_URL (creds
    volume was seeded repo-scoped), empty RUNNER_GROUP (repo-scoped
    registration rejects --runnergroup), fail-fast OMNI_HOME interpolation,
    and the defense-in-depth GIT_CONFIG_* safe.directory entries for all 5
    private clones.
    """
    svc = _deploy_runner_service()
    env = svc["environment"]
    assert isinstance(env, dict)
    assert env["GITHUB_ORG_URL"] == "https://github.com/OmniNode-ai/omnibase_infra", (
        "GITHUB_ORG_URL must be repo-scoped: the live runner registration and "
        "the seeded omninode-deploy-runner-creds volume are repository-level; "
        "an org-level URL invalidates the credential cache key and re-registers "
        "into the wrong scope on recreate"
    )
    assert env.get("RUNNER_GROUP", None) == "", (
        "RUNNER_GROUP must be present-and-empty: repo-scoped config.sh "
        "hard-fails on --runnergroup, and the entrypoint only omits the flag "
        "for an explicitly empty value"
    )
    omni_home = str(env["OMNI_HOME"])
    assert omni_home.startswith("${DEPLOY_RUNNER_OMNI_HOME:?"), (
        f"OMNI_HOME must fail-fast interpolate from DEPLOY_RUNNER_OMNI_HOME "
        f"(got {omni_home!r}) -- a silent empty default produces broken binds"
    )
    assert str(env.get("GIT_CONFIG_COUNT")) == "5"
    values = [str(env[f"GIT_CONFIG_VALUE_{i}"]) for i in range(5)]
    for repo in (
        "omnibase_infra",
        "omnibase_core",
        "omnibase_compat",
        "onex_change_control",
        "omnimarket",
    ):
        assert any(v.endswith(f"/{repo}") for v in values), (
            f"GIT_CONFIG_* safe.directory entries must cover {repo}"
        )
        assert all("DEPLOY_RUNNER_OMNI_HOME" in v for v in values), (
            "safe.directory values must point at the PRIVATE clones"
        )
    for i in range(5):
        assert env[f"GIT_CONFIG_KEY_{i}"] == "safe.directory"


def test_compose_deploy_runner_has_no_shared_gid_and_has_init_wrapper() -> None:
    """No group_add 1000 (that was the live hand-edit's shared-clone write
    grant -- with no shared mount there is nothing it should reach), and the
    committed root-phase init wrapper must own the private OMNI_HOME dir
    before exec'ing the stock entrypoint.
    """
    svc = _deploy_runner_service()
    group_add = [str(g) for g in svc.get("group_add", [])]
    assert "1000" not in group_add, (
        "group_add 1000 grants host-uid-1000 group write access; the private "
        "OMNI_HOME design removes the need for it"
    )
    assert "984" in group_add, "docker socket GID membership must be preserved"
    entrypoint = svc.get("entrypoint")
    assert entrypoint is not None, (
        "deploy runner must carry the committed root-phase init entrypoint "
        "wrapper (mkdir/chown the private OMNI_HOME) -- the live container's "
        "wrapper was an UNCOMMITTED hand-edit that a recreate silently drops"
    )
    wrapper = " ".join(str(part) for part in entrypoint)
    assert "chown" in wrapper
    assert "exec /usr/local/bin/entrypoint.sh" in wrapper


# ---------------------------------------------------------------------------
# 4. ensure_runner_clones.sh behavioral (file:// bare fixtures)
# ---------------------------------------------------------------------------

_ENSURE_REPOS = (
    "omnibase_infra",
    "omnibase_core",
    "omnibase_compat",
    "onex_change_control",
    "omnimarket",
)


def _make_bare_fixtures(tmp_path: Path) -> Path:
    """One seed commit, five bare <repo>.git fixtures cloneable via file://."""
    seed = tmp_path / "seed"
    _git(["init", "-q", "-b", "dev", str(seed)], cwd=tmp_path)
    (seed / "README.md").write_text("fixture\n")
    _git(["add", "README.md"], cwd=seed)
    _git(
        ["-c", "user.email=t@t.com", "-c", "user.name=t", "commit", "-q", "-m", "c1"],
        cwd=seed,
    )
    base = tmp_path / "origins"
    base.mkdir()
    for repo in _ENSURE_REPOS:
        _git(
            ["clone", "-q", "--bare", str(seed), str(base / f"{repo}.git")],
            cwd=tmp_path,
        )
    return base


def _run_ensure(
    omni_home: Path | None, base_url: str, unset_omni_home: bool = False
) -> subprocess.CompletedProcess[str]:
    env = {**_HERMETIC_ENV, "RUNNER_CLONE_BASE_URL": base_url}
    env.pop("OMNI_HOME", None)
    if not unset_omni_home and omni_home is not None:
        env["OMNI_HOME"] = str(omni_home)
    return _run(["bash", str(ENSURE_SCRIPT)], env=env)


def test_ensure_runner_clones_provisions_all_five_and_is_idempotent(
    tmp_path: Path,
) -> None:
    base = _make_bare_fixtures(tmp_path)
    omni_home = tmp_path / "private_home"
    omni_home.mkdir()

    first = _run_ensure(omni_home, f"file://{base}")
    assert first.returncode == 0, first.stderr
    for repo in _ENSURE_REPOS:
        assert (omni_home / repo / ".git").exists(), f"{repo} was not cloned"
        head = _git(["rev-parse", "HEAD"], cwd=omni_home / repo)
        assert head.stdout.strip()

    marker = omni_home / "omnibase_infra" / "local-marker.txt"
    marker.write_text("must survive a re-run\n")
    second = _run_ensure(omni_home, f"file://{base}")
    assert second.returncode == 0, second.stderr
    assert marker.exists(), "idempotent re-run must not re-clone existing repos"


def test_ensure_runner_clones_fails_closed_without_omni_home(tmp_path: Path) -> None:
    base = _make_bare_fixtures(tmp_path)
    result = _run_ensure(None, f"file://{base}", unset_omni_home=True)
    assert result.returncode == 64
    assert "OMNI_HOME must be set" in result.stderr


def test_ensure_runner_clones_fails_closed_on_missing_directory(
    tmp_path: Path,
) -> None:
    base = _make_bare_fixtures(tmp_path)
    result = _run_ensure(tmp_path / "does_not_exist", f"file://{base}")
    assert result.returncode == 64
    assert "does not exist" in result.stderr
