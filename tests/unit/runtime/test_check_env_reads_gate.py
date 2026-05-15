# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for scripts/check-env-reads.sh CI anti-regression gate (OMN-11069)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[3] / "scripts" / "check-env-reads.sh"


def _run_in_git_repo(tmp_path: Path, staged_files: dict[str, str]) -> tuple[int, str]:
    """Set up a minimal git repo, stage files, run check-env-reads.sh --staged."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    # Initial empty commit so staged diff has a base
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    for rel_path, content in staged_files.items():
        full_path = tmp_path / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        subprocess.run(
            ["git", "add", rel_path],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    result = subprocess.run(
        ["bash", str(SCRIPT), "--staged"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, result.stdout + result.stderr


@pytest.mark.unit
class TestCheckEnvReadsGate:
    def test_blocked_on_new_os_environ_read(self, tmp_path: Path) -> None:
        code, output = _run_in_git_repo(
            tmp_path,
            {"src/some_module.py": 'x = os.environ["FOO"]\n'},
        )
        assert code == 1
        assert "BLOCKED" in output

    def test_blocked_on_os_environ_get(self, tmp_path: Path) -> None:
        code, output = _run_in_git_repo(
            tmp_path,
            {"src/some_module.py": 'x = os.environ.get("FOO", "default")\n'},
        )
        assert code == 1
        assert "BLOCKED" in output

    def test_blocked_on_os_getenv(self, tmp_path: Path) -> None:
        code, output = _run_in_git_repo(
            tmp_path,
            {"src/some_module.py": 'x = os.getenv("FOO")\n'},
        )
        assert code == 1
        assert "BLOCKED" in output

    def test_blocked_on_from_os_import_environ(self, tmp_path: Path) -> None:
        code, output = _run_in_git_repo(
            tmp_path,
            {"src/some_module.py": "from os import environ\n"},
        )
        assert code == 1
        assert "BLOCKED" in output

    def test_blocked_on_from_os_import_getenv(self, tmp_path: Path) -> None:
        code, output = _run_in_git_repo(
            tmp_path,
            {"src/some_module.py": "from os import getenv\n"},
        )
        assert code == 1
        assert "BLOCKED" in output

    def test_allowed_in_tests_directory(self, tmp_path: Path) -> None:
        code, _ = _run_in_git_repo(
            tmp_path,
            {"tests/unit/test_something.py": 'x = os.environ["FOO"]\n'},
        )
        assert code == 0

    def test_allowed_in_scripts_directory(self, tmp_path: Path) -> None:
        code, _ = _run_in_git_repo(
            tmp_path,
            {"scripts/some_script.py": 'x = os.getenv("FOO")\n'},
        )
        assert code == 0

    def test_allowed_in_service_kernel(self, tmp_path: Path) -> None:
        code, _ = _run_in_git_repo(
            tmp_path,
            {
                "src/omnibase_infra/runtime/service_kernel.py": (
                    'x = os.environ.get("FOO")\n'
                )
            },
        )
        assert code == 0

    def test_allowed_in_overlay_directory(self, tmp_path: Path) -> None:
        code, _ = _run_in_git_repo(
            tmp_path,
            {
                "src/omnibase_infra/runtime/overlay/boot_overlay.py": (
                    'x = os.environ.get("ONEX_OVERLAY_PATH")\n'
                )
            },
        )
        assert code == 0

    def test_allowed_in_config_prefetcher(self, tmp_path: Path) -> None:
        code, _ = _run_in_git_repo(
            tmp_path,
            {
                "src/omnibase_infra/runtime/config_discovery/config_prefetcher.py": (
                    'os.environ["KEY"] = "value"\n'
                )
            },
        )
        assert code == 0

    def test_clean_file_exits_zero(self, tmp_path: Path) -> None:
        code, _ = _run_in_git_repo(
            tmp_path,
            {"src/some_module.py": "x = 42\n"},
        )
        assert code == 0

    def test_error_message_names_alternative(self, tmp_path: Path) -> None:
        _, output = _run_in_git_repo(
            tmp_path,
            {"src/some_module.py": 'x = os.environ["FOO"]\n'},
        )
        assert "overlay" in output.lower() or "config" in output.lower()
