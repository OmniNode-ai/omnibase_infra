# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for scripts/check-env-reads.sh CI anti-regression gate (OMN-11069)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[3] / "scripts" / "check-env-reads.sh"


def _hermetic_git_env() -> dict[str, str]:
    """Environment with inherited ``GIT_*`` vars stripped.

    When this suite runs inside a git hook (e.g. the local ``pre-push``
    smart-tests hook), git exports ``GIT_DIR`` / ``GIT_INDEX_FILE`` /
    ``GIT_PREFIX`` into the environment. Those leak into the subprocess ``git``
    commands below and retarget them at the ambient repo instead of
    ``tmp_path`` (``git commit --allow-empty`` then fails against the parent's
    locked index). Stripping every ``GIT_*`` var makes the temp-repo operations
    hermetic regardless of the caller's git context.
    """
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def _run_in_git_repo(tmp_path: Path, staged_files: dict[str, str]) -> tuple[int, str]:
    """Set up a minimal git repo, stage files, run check-env-reads.sh --staged."""
    git_env = _hermetic_git_env()
    subprocess.run(
        ["git", "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        env=git_env,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        env=git_env,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        env=git_env,
    )
    # Initial empty commit so staged diff has a base
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        env=git_env,
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
            env=git_env,
        )

    result = subprocess.run(
        ["bash", str(SCRIPT), "--staged"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env=git_env,
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
        # OMN-14951 gap 2: renamed the placeholder from the literal name
        # "KEY" to "ONEX_OVERLAY_PATH" -- "KEY" is itself secret-ish under
        # the new declared-name check below (SECRET_ISH matches the bare
        # word KEY), so the original placeholder now legitimately trips that
        # check. This fixture's intent is unchanged: prove Check 1 (path
        # boundary) allows a non-secret-ish raw env read in an approved
        # boundary file.
        code, _ = _run_in_git_repo(
            tmp_path,
            {
                "src/omnibase_infra/runtime/config_discovery/config_prefetcher.py": (
                    'x = os.environ.get("ONEX_OVERLAY_PATH")\n'
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


@pytest.mark.unit
class TestCheckEnvReadsSecretNameDeclarationGate:
    """OMN-14951 gap 2: an undeclared secret-ish name read via
    os.environ[...]/os.environ.get(...)/os.getenv(...)/get_secret(...) in a
    boundary file fails the gate; a declared one (bootstrap allowlist, or a
    required_secrets/bootstrap_secrets list element in the same file) passes.
    Scoped to the same boundary infix file set as Check 1's path allowlist --
    tests/ and scripts/ are test-double/tooling code, not a deployable's
    consumed env surface, and stay exempt from this check (as they already
    are from Check 1).
    """

    BOUNDARY_FILE = "src/omnibase_infra/runtime/config_discovery/config_prefetcher.py"

    def test_blocked_on_undeclared_secret_ish_name(self, tmp_path: Path) -> None:
        code, output = _run_in_git_repo(
            tmp_path,
            {self.BOUNDARY_FILE: 'x = os.environ.get("STRIPE_SECRET_KEY")\n'},
        )
        assert code == 1
        assert "BLOCKED" in output
        assert "STRIPE_SECRET_KEY" in output

    def test_blocked_on_undeclared_name_via_os_environ_bracket(
        self, tmp_path: Path
    ) -> None:
        code, output = _run_in_git_repo(
            tmp_path,
            {self.BOUNDARY_FILE: 'x = os.environ["DB_PASSWORD"]\n'},
        )
        assert code == 1
        assert "DB_PASSWORD" in output

    def test_blocked_on_undeclared_name_via_get_secret(self, tmp_path: Path) -> None:
        code, output = _run_in_git_repo(
            tmp_path,
            {self.BOUNDARY_FILE: 'x = get_secret("payments.stripe.secret_key")\n'},
        )
        assert code == 1
        assert "payments.stripe.secret_key" in output

    def test_allowed_on_bootstrap_allowlist_name(self, tmp_path: Path) -> None:
        # INFISICAL_CLIENT_SECRET is secret-ish but on the finite, named
        # bootstrap allowlist -- a keyring cannot unlock itself.
        code, _ = _run_in_git_repo(
            tmp_path,
            {self.BOUNDARY_FILE: ('x = os.environ.get("INFISICAL_CLIENT_SECRET")\n')},
        )
        assert code == 0

    def test_allowed_when_self_declared_in_required_secrets(
        self, tmp_path: Path
    ) -> None:
        code, _ = _run_in_git_repo(
            tmp_path,
            {
                self.BOUNDARY_FILE: (
                    'REQUIRED = ["STRIPE_SECRET_KEY"]\n'
                    "required_secrets = REQUIRED\n"
                    'x = os.environ.get("STRIPE_SECRET_KEY")\n'
                )
            },
        )
        assert code == 0

    def test_allowed_when_self_declared_in_bootstrap_secrets(
        self, tmp_path: Path
    ) -> None:
        code, _ = _run_in_git_repo(
            tmp_path,
            {
                self.BOUNDARY_FILE: (
                    'bootstrap_secrets = ["MY_APP_TOKEN"]\n'
                    'x = os.environ.get("MY_APP_TOKEN")\n'
                )
            },
        )
        assert code == 0

    def test_non_secret_ish_name_never_blocked_by_this_check(
        self, tmp_path: Path
    ) -> None:
        # RUNTIME_PROFILE-style names never match the secret-ish heuristic,
        # so they pass regardless of declaration.
        code, _ = _run_in_git_repo(
            tmp_path,
            {self.BOUNDARY_FILE: 'x = os.environ.get("RUNTIME_PROFILE")\n'},
        )
        assert code == 0

    def test_scripts_directory_exempt_from_secret_name_check(
        self, tmp_path: Path
    ) -> None:
        # scripts/ is outside the boundary-infix target set for Check 2
        # (it is tooling, not a deployable's env surface) -- an undeclared
        # secret-ish read there is still blocked by Check 1 (any new raw env
        # read outside the approved set)... except scripts/ IS in Check 1's
        # own prefix allowlist too, so this must pass cleanly, proving Check
        # 2 does not silently widen enforcement into already-exempt tooling
        # paths.
        code, _ = _run_in_git_repo(
            tmp_path,
            {"scripts/some_script.py": 'x = os.environ.get("SOME_API_KEY")\n'},
        )
        assert code == 0

    def test_tests_directory_exempt_from_secret_name_check(
        self, tmp_path: Path
    ) -> None:
        code, _ = _run_in_git_repo(
            tmp_path,
            {"tests/unit/test_something.py": ('x = os.environ.get("SOME_API_KEY")\n')},
        )
        assert code == 0
