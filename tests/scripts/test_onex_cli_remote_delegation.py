# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for infra shell wrapper remote-target guards (OMN-7465)."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "onex-cli.sh"


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o755)


def _run_infra_function(
    tmp_path: Path,
    env_text: str,
    invocation: str,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    infra_dir = tmp_path / "omnibase_infra"
    infra_dir.mkdir()
    (infra_dir / "scripts").mkdir()
    _write_executable(
        infra_dir / "scripts" / "check-stale-images.sh",
        "#!/usr/bin/env bash\nexit 0\n",
    )

    env_file = tmp_path / ".env"
    env_file.write_text(env_text)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "uv",
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "$*" >> "$ONEX_TEST_UV_LOG"\n',
    )
    _write_executable(
        bin_dir / "ssh",
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$ONEX_TEST_SSH_LOG"\n',
    )

    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "OMNIBASE_ENV_FILE": str(env_file),
        "OMNIBASE_INFRA_DIR": str(infra_dir),
        "ONEX_TEST_SSH_LOG": str(tmp_path / "ssh.log"),
        "ONEX_TEST_UV_LOG": str(tmp_path / "uv.log"),
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
    }
    env.pop("POSTGRES_HOST", None)
    env.pop("KAFKA_BOOTSTRAP_SERVERS", None)
    if extra_env:
        env.update(extra_env)

    return subprocess.run(
        ["bash", "-c", f"source {shlex.quote(str(SCRIPT))}; {invocation}"],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=env,
        check=False,
        timeout=30,
    )


@pytest.mark.unit
def test_infra_up_delegates_to_remote_host_from_postgres_env(tmp_path: Path) -> None:
    result = _run_infra_function(
        tmp_path,
        "POSTGRES_HOST=192.168.86.201\n",
        "infra-up --build",
    )

    assert result.returncode == 0, result.stderr
    assert "delegating infra-up over SSH" in result.stderr
    assert not (tmp_path / "uv.log").exists()

    ssh_args = (tmp_path / "ssh.log").read_text().splitlines()
    assert ssh_args[0] == "192.168.86.201"
    assert "zsh -lic" in ssh_args[1]
    assert "ONEX_INFRA_REMOTE_DELEGATED=1" in ssh_args[1]
    assert "infra-up" in ssh_args[1]
    assert "--build" in ssh_args[1]


@pytest.mark.unit
def test_remote_fail_mode_blocks_local_compose(tmp_path: Path) -> None:
    result = _run_infra_function(
        tmp_path,
        "POSTGRES_HOST=192.168.86.201\n",
        "infra-up-runtime",
        {"ONEX_INFRA_REMOTE_BEHAVIOR": "fail"},
    )

    assert result.returncode == 1
    assert "Infrastructure is configured to run on 192.168.86.201" in result.stderr
    assert "ssh 192.168.86.201 'infra-up-runtime'" in result.stderr
    assert not (tmp_path / "ssh.log").exists()
    assert not (tmp_path / "uv.log").exists()


@pytest.mark.unit
def test_local_postgres_host_starts_local_onex_bundle(tmp_path: Path) -> None:
    result = _run_infra_function(
        tmp_path,
        "POSTGRES_HOST=localhost\n",
        "infra-up --build",
    )

    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "ssh.log").exists()
    assert (tmp_path / "uv.log").read_text() == (
        "run python -m omnibase_infra.docker.catalog.cli up --build core\n"
    )


@pytest.mark.unit
def test_delegated_marker_allows_remote_host_to_run_locally(tmp_path: Path) -> None:
    result = _run_infra_function(
        tmp_path,
        "POSTGRES_HOST=192.168.86.201\n",
        "infra-status",
        {"ONEX_INFRA_REMOTE_DELEGATED": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "ssh.log").exists()
    assert (tmp_path / "uv.log").read_text() == (
        "run python -m omnibase_infra.docker.catalog.cli status\n"
    )


@pytest.mark.unit
def test_kafka_bootstrap_host_delegates_when_postgres_is_unset(tmp_path: Path) -> None:
    result = _run_infra_function(
        tmp_path,
        "KAFKA_BOOTSTRAP_SERVERS=192.168.86.201:19092,localhost:19092\n",  # kafka-fallback-ok
        "infra-up-memory",
        {"ONEX_INFRA_REMOTE_USER": "jonah"},
    )

    assert result.returncode == 0, result.stderr
    ssh_args = (tmp_path / "ssh.log").read_text().splitlines()
    assert ssh_args[0] == "jonah@192.168.86.201"
    assert "infra-up-memory" in ssh_args[1]
    assert not (tmp_path / "uv.log").exists()
