# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Deploy-runtime regression coverage for Docker build context paths."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-runtime.sh"
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile.runtime"

# Matches a COPY directive's argument list. We discard `--from=<stage>` lines
# (those copy from a prior build stage, not the host build context) and any
# remaining `--flag` tokens (e.g. `--chown=...`), then keep the source operands
# that pull from the workspace/ tree.
_COPY_LINE_RE = re.compile(r"^COPY\s+(?P<args>.+)$", re.MULTILINE)


def _dockerfile_workspace_copy_sources() -> list[str]:
    """Every Dockerfile.runtime COPY source that pulls from the workspace/ tree.

    The deployed build context is assembled by deploy-runtime.sh's sync_files;
    each of these paths must be rsynced (or generated) into that context or the
    workspace-mode `docker build` fails with "failed to calculate checksum ...:
    not found" (the OMN-12987 regression). This list is derived from the live
    Dockerfile so a future COPY workspace/<x> without a matching rsync is caught.
    """
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    sources: list[str] = []
    for match in _COPY_LINE_RE.finditer(dockerfile):
        tokens = match.group("args").split()
        # `--from=<stage>` copies from a build stage, never the host context.
        if any(tok.startswith("--from=") for tok in tokens):
            continue
        # Drop flag tokens (--chown=, --link, ...); the last operand is the
        # destination, everything before it is a build-context source.
        operands = [tok for tok in tokens if not tok.startswith("--")]
        sources.extend(src for src in operands[:-1] if src.startswith("workspace/"))
    return sources


@pytest.mark.unit
def test_deploy_runtime_syncs_runtime_dockerfile_copy_sources() -> None:
    """deploy-runtime.sh must ship paths copied by Dockerfile.runtime."""
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    deploy_script = DEPLOY_SCRIPT.read_text(encoding="utf-8")

    required_sources = (
        "workspace/sibling-repos/",
        "scripts/runtime_build/compute_workspace_provenance.py",
    )
    for source in required_sources:
        assert source in dockerfile

    assert '"${repo_root}/workspace/sibling-repos/"' in deploy_script
    assert '"${repo_root}/scripts/runtime_build/"' in deploy_script


@pytest.mark.unit
def test_deploy_runtime_stages_every_workspace_copy_source() -> None:
    """Every `COPY workspace/<x>` in Dockerfile.runtime must be staged.

    Regression guard for OMN-12987: Dockerfile.runtime COPYs
    workspace/sibling-pin-comparison.json (and workspace/sibling-repos/), but an
    earlier deploy-runtime.sh only rsynced workspace/sibling-repos/ into the
    deployed build context. The root-level comparison file was never carried
    over, so every workspace-mode `docker build` failed with "failed to
    calculate checksum of ref ...:/workspace/sibling-pin-comparison.json: not
    found". The dev compose build masked this because it runs from the repo root
    where the committed placeholder exists.

    This test derives the workspace/ COPY sources from the live Dockerfile and
    asserts deploy-runtime.sh references each as an rsync source for the deployed
    context, so a future Dockerfile COPY without a matching rsync fails CI.
    """
    deploy_script = DEPLOY_SCRIPT.read_text(encoding="utf-8")

    workspace_sources = _dockerfile_workspace_copy_sources()
    # The Dockerfile must at minimum COPY the two known workspace/ paths; a regex
    # that silently matched nothing would make this guard vacuously pass.
    assert "workspace/sibling-repos/" in workspace_sources
    assert "workspace/sibling-pin-comparison.json" in workspace_sources

    missing: list[str] = []
    for source in workspace_sources:
        # A directory source (trailing slash) and a file source both appear in
        # deploy-runtime.sh as an rsync argument quoted under ${repo_root}.
        staged = f'"${{repo_root}}/{source}"' in deploy_script
        if not staged:
            missing.append(source)

    assert not missing, (
        "Dockerfile.runtime COPYs these workspace/ paths but deploy-runtime.sh "
        f"does not stage them into the deployed build context: {missing}. Add an "
        "rsync of each into sync_files() or workspace-mode `docker build` will "
        "fail with 'failed to calculate checksum ...: not found' (OMN-12987)."
    )


def _init_git_repo(path: Path, marker: str) -> str:
    path.mkdir(parents=True)
    (path / "marker.txt").write_text(marker, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "add", "marker.txt"], cwd=path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=deploy-runtime-test",
            "-c",
            "user.email=deploy-runtime-test@example.com",
            "commit",
            "-q",
            "-m",
            "init",
        ],
        cwd=path,
        check=True,
    )
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_fake_docker(bin_dir: Path) -> None:
    docker = bin_dir / "docker"
    docker.write_text(
        """#!/usr/bin/env sh
set -eu
if [ "$1" = "compose" ] && [ "$2" = "version" ] && [ "$3" = "--short" ]; then
  printf '2.27.0\\n'
  exit 0
fi
printf 'unexpected docker invocation: %s\\n' "$*" >&2
exit 1
""",
        encoding="utf-8",
    )
    docker.chmod(0o755)


@pytest.mark.unit
def test_workspace_printed_build_command_uses_operator_omni_home(
    tmp_path: Path,
) -> None:
    """Operator OMNI_HOME must survive ~/.omnibase/.env during workspace builds."""
    operator_home = tmp_path / "operator" / "omni_home"
    env_home_root = tmp_path / "env-file" / "omni_home"
    operator_omnimarket_sha = _init_git_repo(
        operator_home / "omnimarket", "operator-root"
    )
    env_omnimarket_sha = _init_git_repo(env_home_root / "omnimarket", "env-root")

    home = tmp_path / "home"
    omnibase_dir = home / ".omnibase"
    omnibase_dir.mkdir(parents=True)
    (omnibase_dir / ".env").write_text(
        f"INFRA_HOST=127.0.0.1\nOMNI_HOME={env_home_root}\n",
        encoding="utf-8",
    )

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_docker(bin_dir)

    env = os.environ.copy()
    env.update(
        {
            "BUILD_SOURCE": "workspace",
            "HOME": str(home),
            "OMNI_HOME": str(operator_home),
            "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
        }
    )
    env.pop("EXPECTED_BUILD_SOURCE", None)

    result = subprocess.run(
        ["bash", str(DEPLOY_SCRIPT), "--print-compose-cmd"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert f"--build-arg OMNI_HOME={operator_home}" in result.stdout
    assert "--build-arg BUILD_SOURCE=workspace" in result.stdout
    assert "--build-arg EXPECTED_BUILD_SOURCE=workspace" in result.stdout
    assert f"--build-arg OMNIMARKET_REF={operator_omnimarket_sha}" in result.stdout
    assert env_omnimarket_sha not in result.stdout


@pytest.mark.unit
def test_deploy_runtime_stages_workspace_and_passes_omni_home_arg() -> None:
    """Workspace mode must stage sibling repos and pass OMNI_HOME to Dockerfile."""
    deploy_script = DEPLOY_SCRIPT.read_text(encoding="utf-8")

    assert 'stage_workspace_if_needed "${repo_root}"' in deploy_script
    assert '--build-arg "BUILD_SOURCE=${build_source}"' in deploy_script
    assert (
        '--build-arg "EXPECTED_BUILD_SOURCE=${expected_build_source}"' in deploy_script
    )
    assert '--build-arg "OMNI_HOME=${omni_home}"' in deploy_script


@pytest.mark.unit
def test_deploy_runtime_runs_sibling_lock_pin_preflight() -> None:
    """Workspace staging must run the OMN-12987 lock-pin preflight before build.

    Recurrence guard: the 2026-06-11 stability crash shipped a 13-day-stale
    infra 0.37.0 because the build ignored the omnimarket dev lock. The deploy
    script must invoke check_sibling_lock_pins after staging and abort on
    failure.
    """
    deploy_script = DEPLOY_SCRIPT.read_text(encoding="utf-8")

    # The preflight is called from stage_workspace_if_needed after staging.
    assert 'check_sibling_lock_pins "${repo_root}" "${omni_home}"' in deploy_script
    # The preflight function references the guard script and aborts on failure.
    assert "scripts/runtime_build/check_sibling_lock_pins.py" in deploy_script
    assert "Refusing to build a stale image." in deploy_script


@pytest.mark.unit
def test_deploy_runtime_uses_current_lock_pin_preflight_interface() -> None:
    """The preflight caller must match check_sibling_lock_pins.py's current CLI.

    Regression guard: OMN-12977/12987 replaced the original ``--provenance-out``
    flag with ``--lock`` (required pin authority), repeatable ``--repo
    PACKAGE=PATH`` (the canonical clones the build vendors), and ``--output``
    (where to write the comparison JSON). The deploy-runtime.sh caller was left
    pinned to the removed ``--provenance-out`` flag, so EVERY workspace
    ``--execute`` deploy failed at argparse (``the following arguments are
    required: --lock``) before any build started. This test pins the corrected
    interface so the stale-flag invocation can never come back.
    """
    deploy_script = DEPLOY_SCRIPT.read_text(encoding="utf-8")

    # Removed flag must be gone — its presence is the exact regression we hit.
    assert "--provenance-out" not in deploy_script

    # Current required + supported flags must be wired into the guard_args.
    assert '--lock "${lock_path}"' in deploy_script
    assert "--repo " in deploy_script
    assert '--output "${provenance_out}"' in deploy_script

    # The consuming repo's omnimarket uv.lock is the pin authority, and every
    # vendored sibling must be passed as a --repo PACKAGE=PATH entry.
    assert 'lock_path="${omni_home}/omnimarket/uv.lock"' in deploy_script
    for package in (
        "omnibase-infra",
        "omnibase-core",
        "omnibase-spi",
        "omnibase-compat",
        "onex-change-control",
    ):
        assert f'--repo "{package}=' in deploy_script

    # The OMN-12977 operator override must be honored via --allow-drift.
    assert "ALLOW_SIBLING_PIN_DRIFT" in deploy_script
    assert "--allow-drift" in deploy_script


@pytest.mark.unit
def test_stage_workspace_emits_build_sha_marker() -> None:
    """stage_workspace.sh must record each sibling's HEAD SHA (OMN-12987).

    rsync drops .git, so without a .build-sha marker the staged tree has no
    recoverable SHA and the lock-pin preflight cannot verify the vendored commit.
    """
    stage_script = (
        DEPLOY_SCRIPT.parent / "runtime_build" / "stage_workspace.sh"
    ).read_text(encoding="utf-8")

    assert 'git -C "${src}" rev-parse HEAD > "${dst}/.build-sha"' in stage_script
