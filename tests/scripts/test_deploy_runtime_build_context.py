# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Deploy-runtime regression coverage for Docker build context paths."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-runtime.sh"
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile.runtime"


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
