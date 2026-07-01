# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""deploy-runtime.sh must pass PROMOTION_CLASS + NON_MAIN_LINEAGE build-args (OMN-13669).

The Dockerfile.runtime guard (OMN-13656 / #2116) exits 64 when a workspace build
does not stamp PROMOTION_CLASS=stability-candidate + NON_MAIN_LINEAGE=true.
Before OMN-13669, deploy-runtime.sh only passed BUILD_SOURCE/EXPECTED_BUILD_SOURCE
and never set the two provenance args, so every workspace redeploy failed the
Dockerfile guard immediately.

These tests verify:
  - resolve_promotion_class / resolve_non_main_lineage functions are defined,
  - the build-arg array in build_images() carries both args,
  - the print_compose_commands() echo path also carries both args,
  - a workspace invocation of --print-compose-cmd produces
    PROMOTION_CLASS=stability-candidate and NON_MAIN_LINEAGE=true in its output,
  - a release invocation produces PROMOTION_CLASS=clean-main and NON_MAIN_LINEAGE=false.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-runtime.sh"


def _script_text() -> str:
    return DEPLOY_SCRIPT.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Static text assertions (no subprocess needed)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_resolve_promotion_class_function_defined() -> None:
    """deploy-runtime.sh must define resolve_promotion_class()."""
    text = _script_text()
    assert re.search(r"^resolve_promotion_class\s*\(\)", text, re.MULTILINE), (
        "deploy-runtime.sh must define resolve_promotion_class()"
    )


@pytest.mark.unit
def test_resolve_non_main_lineage_function_defined() -> None:
    """deploy-runtime.sh must define resolve_non_main_lineage()."""
    text = _script_text()
    assert re.search(r"^resolve_non_main_lineage\s*\(\)", text, re.MULTILINE), (
        "deploy-runtime.sh must define resolve_non_main_lineage()"
    )


@pytest.mark.unit
def test_build_images_passes_promotion_class_build_arg() -> None:
    """build_images() must pass --build-arg PROMOTION_CLASS to docker build."""
    text = _script_text()
    assert '--build-arg "PROMOTION_CLASS=${promotion_class}"' in text, (
        "build_images() must pass --build-arg PROMOTION_CLASS=... to docker compose build"
    )


@pytest.mark.unit
def test_build_images_passes_non_main_lineage_build_arg() -> None:
    """build_images() must pass --build-arg NON_MAIN_LINEAGE to docker build."""
    text = _script_text()
    assert '--build-arg "NON_MAIN_LINEAGE=${non_main_lineage}"' in text, (
        "build_images() must pass --build-arg NON_MAIN_LINEAGE=... to docker compose build"
    )


@pytest.mark.unit
def test_print_compose_commands_includes_promotion_class() -> None:
    """print_compose_commands() echo path must include PROMOTION_CLASS."""
    text = _script_text()
    assert "--build-arg PROMOTION_CLASS=${promotion_class}" in text, (
        "print_compose_commands() must echo --build-arg PROMOTION_CLASS=..."
    )


@pytest.mark.unit
def test_print_compose_commands_includes_non_main_lineage() -> None:
    """print_compose_commands() echo path must include NON_MAIN_LINEAGE."""
    text = _script_text()
    assert "--build-arg NON_MAIN_LINEAGE=${non_main_lineage}" in text, (
        "print_compose_commands() must echo --build-arg NON_MAIN_LINEAGE=..."
    )


@pytest.mark.unit
def test_workspace_resolves_to_stability_candidate() -> None:
    """resolve_promotion_class must map workspace => stability-candidate in script text."""
    text = _script_text()
    # The function body must contain the stability-candidate literal.
    func_match = re.search(
        r"resolve_promotion_class\s*\(\)\s*\{(.*?)\n\}",
        text,
        re.DOTALL,
    )
    assert func_match is not None, "resolve_promotion_class function not found"
    body = func_match.group(1)
    assert "stability-candidate" in body, (
        "resolve_promotion_class must return stability-candidate for workspace builds"
    )
    assert "workspace" in body, (
        "resolve_promotion_class must branch on workspace build_source"
    )


@pytest.mark.unit
def test_release_resolves_to_clean_main() -> None:
    """resolve_promotion_class must map non-workspace => clean-main in script text."""
    text = _script_text()
    func_match = re.search(
        r"resolve_promotion_class\s*\(\)\s*\{(.*?)\n\}",
        text,
        re.DOTALL,
    )
    assert func_match is not None, "resolve_promotion_class function not found"
    body = func_match.group(1)
    assert "clean-main" in body, (
        "resolve_promotion_class must return clean-main for release builds"
    )


# ---------------------------------------------------------------------------
# Subprocess assertions (dry-run via --print-compose-cmd, no real docker build)
# ---------------------------------------------------------------------------


def _write_fake_docker(bin_dir: Path) -> None:
    """Write a minimal docker stub that answers `docker compose version --short`."""
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


def _setup_home_env(tmp_path: Path, extra_env: str = "") -> tuple[Path, Path]:
    """Create a minimal HOME with ~/.omnibase/.env (INFRA_HOST required by script).

    Returns (home_dir, bin_dir).  bin_dir contains a fake docker stub.
    """
    home = tmp_path / "home"
    omnibase_dir = home / ".omnibase"
    omnibase_dir.mkdir(parents=True)
    (omnibase_dir / ".env").write_text(
        f"INFRA_HOST=127.0.0.1\n{extra_env}\n",
        encoding="utf-8",
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_docker(bin_dir)
    return home, bin_dir


@pytest.mark.unit
def test_print_compose_cmd_workspace_yields_stability_candidate(
    tmp_path: Path,
) -> None:
    """--print-compose-cmd with BUILD_SOURCE=workspace must emit stability-candidate args."""
    # Minimal OMNI_HOME so the script satisfies the workspace path check.
    omni_home = tmp_path / "omni_home"
    omni_home.mkdir()
    home, bin_dir = _setup_home_env(tmp_path)

    env = os.environ.copy()
    env.update(
        {
            "BUILD_SOURCE": "workspace",
            "OMNI_HOME": str(omni_home),
            "HOME": str(home),
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

    assert "--build-arg PROMOTION_CLASS=stability-candidate" in result.stdout, (
        f"workspace --print-compose-cmd must show PROMOTION_CLASS=stability-candidate; "
        f"got stdout={result.stdout!r}"
    )
    assert "--build-arg NON_MAIN_LINEAGE=true" in result.stdout, (
        f"workspace --print-compose-cmd must show NON_MAIN_LINEAGE=true; "
        f"got stdout={result.stdout!r}"
    )


@pytest.mark.unit
def test_print_compose_cmd_release_yields_clean_main(
    tmp_path: Path,
) -> None:
    """--print-compose-cmd with BUILD_SOURCE=release must emit clean-main args."""
    home, bin_dir = _setup_home_env(tmp_path)

    env = os.environ.copy()
    env.update(
        {
            "BUILD_SOURCE": "release",
            "HOME": str(home),
            "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
        }
    )
    env.pop("EXPECTED_BUILD_SOURCE", None)
    env.pop("OMNI_HOME", None)

    result = subprocess.run(
        ["bash", str(DEPLOY_SCRIPT), "--print-compose-cmd"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--build-arg PROMOTION_CLASS=clean-main" in result.stdout, (
        f"release --print-compose-cmd must show PROMOTION_CLASS=clean-main; "
        f"got stdout={result.stdout!r}"
    )
    assert "--build-arg NON_MAIN_LINEAGE=false" in result.stdout, (
        f"release --print-compose-cmd must show NON_MAIN_LINEAGE=false; "
        f"got stdout={result.stdout!r}"
    )
