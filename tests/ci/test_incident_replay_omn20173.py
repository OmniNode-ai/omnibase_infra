# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Incident replay for the Coding Plan endpoint guard (OMN-20173).

The artifact is ``docker/docker-compose.judge.yml`` exactly as it stood on
``dev`` at 3a12fa83d on 2026-09-30, read from the git object and committed
unmodified. Its ``LLM_GLM_URL`` default addressed the z.ai GLM Coding Plan path,
so the judge lane (and the dogfood and lakshman lanes that carried the same
default) sent system traffic to an endpoint whose terms bar direct API calls
from our own systems. Nothing refused it: the old endpoint test in this repo
pinned every committed z.ai URL TO the Coding Plan path, the opposite of the
rule.

The discriminator is load-bearing: a guard that refused every compose file
would replay this perfectly, so the same guard is run over the real tracked
tree on this branch and must accept it.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = _ROOT / "tests/fixtures/omn20173/docker-compose.judge.yml.captured"
_SHA256 = "8b13e93fe66e7ffe3fb87efc5c09ebeb25eb2b6861a60f58d45091e2ef1eacfd"
_GUARD = _ROOT / "scripts/check_no_coding_plan_endpoint.py"


def _run_guard(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_GUARD), "--root", str(root)],
        capture_output=True,
        text=True,
        check=False,
        env=scrub_git_location_env(os.environ),
    )


def test_the_fixture_is_the_captured_bytes() -> None:
    assert hashlib.sha256(_FIXTURE.read_bytes()).hexdigest() == _SHA256


def test_the_real_guard_refuses_the_compose_default_that_addressed_the_coding_plan(
    tmp_path: Path,
) -> None:
    target = tmp_path / "docker" / "docker-compose.judge.yml"
    target.parent.mkdir(parents=True)
    target.write_bytes(_FIXTURE.read_bytes())
    subprocess.run(
        ["git", "init", "-q", str(tmp_path)],
        check=True,
        env=scrub_git_location_env(os.environ),
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "--", "docker/docker-compose.judge.yml"],
        check=True,
        env=scrub_git_location_env(os.environ),
    )
    result = _run_guard(tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "docker/docker-compose.judge.yml:145" in result.stderr


def test_the_same_guard_accepts_the_tracked_tree_on_the_fixing_branch() -> None:
    result = _run_guard(_ROOT)
    assert result.returncode == 0, result.stderr
