# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22552 AC3: exposure-reader-coverage validator exits 0 at pinned shas and exits 1 naming local-environment.v1 and run-trace.v1 with omnidash at e0dadfc6."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# The omnidash pin before OMN-22552: it has no reader for local-environment.v1 or run-trace.v1.
OLD_OMNIDASH_SHA = "e0dadfc6eadef5f5151f24f7f13bdc3628011c66"


def _fetch_repo_at_sha(
    tmp_path_factory: pytest.TempPathFactory,
    name: str,
    sha: str,
) -> pathlib.Path:
    """Fetch a repository at a specific SHA into a temporary directory."""
    target = tmp_path_factory.mktemp(f"repo_{name}_{sha[:8]}")
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    commands: list[list[str]] = [
        ["git", "init"],
        [
            "git",
            "remote",
            "add",
            "origin",
            f"https://github.com/OmniNode-ai/{name}.git",
        ],
        ["git", "fetch", "--depth", "1", "origin", sha],
        ["git", "checkout", "--detach", "FETCH_HEAD"],
    ]
    for cmd in commands:
        try:
            subprocess.run(
                cmd,
                cwd=target,
                check=True,
                timeout=600,
                env=env,
                capture_output=True,
                text=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            pytest.skip(f"Could not fetch {name} at {sha}: {exc}")
    return target


@pytest.fixture(scope="session")
def pinned_shas() -> dict[str, str]:
    """Read pinned SHAs from .github/sibling-pins.yaml."""
    pins_path = REPO_ROOT / ".github" / "sibling-pins.yaml"
    with pins_path.open() as f:
        data: dict[str, Any] = yaml.safe_load(f)
    pins = data["pins"]
    return {
        "omnimarket": pins["omnimarket"],
        "omnidash": pins["omnidash"],
    }


@pytest.fixture(scope="session")
def omnimarket_repo(
    tmp_path_factory: pytest.TempPathFactory,
    pinned_shas: dict[str, str],
) -> pathlib.Path:
    """Fetch omnimarket at the pinned SHA."""
    return _fetch_repo_at_sha(tmp_path_factory, "omnimarket", pinned_shas["omnimarket"])


@pytest.fixture(scope="session")
def omnidash_repo(
    tmp_path_factory: pytest.TempPathFactory,
    pinned_shas: dict[str, str],
) -> pathlib.Path:
    """Fetch omnidash at the pinned SHA."""
    return _fetch_repo_at_sha(tmp_path_factory, "omnidash", pinned_shas["omnidash"])


def run_validator(
    market: pathlib.Path,
    dash: pathlib.Path,
) -> subprocess.CompletedProcess[str]:
    """Run the bus-backed exposure readers validator."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "omnibase_infra.validators.bus_backed_exposure_readers",
            "src/omnibase_infra/nodes",
            "--extra-contracts-dir",
            str(market / "src" / "omnimarket" / "nodes"),
            "--registry",
            str(dash / "src" / "registry" / "component-registry.json"),
            "--layouts-dir",
            str(dash / "src" / "templates"),
            "--local-pages-dir",
            str(dash / "src" / "pages" / "local"),
            "--backend-reader-surface",
            str(market / "src" / "omnimarket" / "projection"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def test_validator_passes_at_the_pinned_shas(
    omnimarket_repo: pathlib.Path,
    omnidash_repo: pathlib.Path,
) -> None:
    """Validator exits 0 over the omnimarket and omnidash trees at pinned shas."""
    result = run_validator(omnimarket_repo, omnidash_repo)
    assert result.returncode == 0, (
        f"Validator failed at pinned shas:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_validator_names_both_exposures_with_omnidash_at_e0dadfc6(
    tmp_path_factory: pytest.TempPathFactory,
    omnimarket_repo: pathlib.Path,
) -> None:
    """Validator exits 1 naming local-environment.v1 and run-trace.v1 with omnidash at e0dadfc6."""
    old_dash = _fetch_repo_at_sha(tmp_path_factory, "omnidash", OLD_OMNIDASH_SHA)
    result = run_validator(omnimarket_repo, old_dash)
    combined = result.stdout + result.stderr
    assert result.returncode == 1, (
        f"Validator should exit 1:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "local-environment.v1" in combined, (
        f"local-environment.v1 not in output:\n{combined}"
    )
    assert "run-trace.v1" in combined, f"run-trace.v1 not in output:\n{combined}"
