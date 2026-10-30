# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exercise the shipped Bash dispatcher and Python reader across real git I/O."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)
from omnibase_infra.runtime.service_kernel import workspace_runtime_config_root
from tests.ci._prepush_private_table import (
    PRIVATE_TABLE_REL,
    SYNTHETIC_TABLE,
    write_hook_repo,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
LIBRARY = REPO_ROOT / "scripts/hooks/prepush_dispatch.sh"


@pytest.mark.parametrize("root_source", ["explicit", "registry-sibling"])
def test_dispatcher_and_guard_share_committed_table_and_refuse_drift(
    tmp_path: Path, root_source: str
) -> None:
    """Real subprocesses resolve one authority, then refuse a forged identity."""
    hook_repo = write_hook_repo(tmp_path, REPO_ROOT)
    registry = tmp_path / "registry"
    registry.mkdir()
    env = {
        key: value
        for key, value in scrub_git_location_env(os.environ).items()
        if not key.startswith("PREPUSH_")
        and key not in {"ONEX_WORKSPACE_CONFIG_ROOT", "OMNI_HOME"}
    }
    env.update(
        OMNI_HOME=str(registry),
        REPO_ROOT=str(hook_repo),
        PREPUSH_PRIVATE_TABLE_REL=PRIVATE_TABLE_REL,
        PYTHONDONTWRITEBYTECODE="1",
    )
    if root_source == "explicit":
        env["ONEX_WORKSPACE_CONFIG_ROOT"] = str(tmp_path / "declared-config")
    root = workspace_runtime_config_root(
        registry, environment=env, allow_developer_config=False
    )
    target = root / PRIVATE_TABLE_REL
    target.parent.mkdir(parents=True)
    target.write_text(SYNTHETIC_TABLE, encoding="utf-8")
    for args in (
        ["init", "-q"],
        ["add", PRIVATE_TABLE_REL],
        [
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "user.name=Test",
            "commit",
            "-qm",
            "Synthetic deployment table",
        ],
    ):
        subprocess.run(
            ["git", *args],
            cwd=root,
            env=scrub_git_location_env(env),
            check=True,
            capture_output=True,
            timeout=30,
        )

    def shell(command: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", "-c", f"source {shlex.quote(str(LIBRARY))}\n{command}"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )

    def guard() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.hooks.pytest_full_suite_host_guard",
                "--private-table",
                PRIVATE_TABLE_REL,
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )

    cli = guard()
    dispatcher = shell("prepush_private_table_text")
    assert cli.returncode == dispatcher.returncode == 0, (
        cli.stderr,
        dispatcher.stderr,
    )
    assert cli.stdout == dispatcher.stdout == SYNTHETIC_TABLE
    rows = shell("prepush_table_rows")
    assert rows.returncode == 0, rows.stderr
    assert rows.stdout.splitlines()[0].split("\t")[0] == "local"
    assert rows.stdout.splitlines()[1:] == SYNTHETIC_TABLE.splitlines()[1:]

    target.write_text(
        SYNTHETIC_TABLE + "# uncommitted identity edit\n", encoding="utf-8"
    )
    for result in (guard(), shell("prepush_private_table_text")):
        assert result.returncode == 1
        assert result.stdout == ""
        assert "differs between the working tree and HEAD" in result.stderr

    target.unlink()
    for result in (guard(), shell("prepush_private_table_text")):
        assert result.returncode == 2
        assert result.stdout == ""
    neutral = shell("prepush_table_rows")
    assert neutral.returncode == 0, neutral.stderr
    assert [row.split("\t")[0] for row in neutral.stdout.splitlines()] == ["local"]
