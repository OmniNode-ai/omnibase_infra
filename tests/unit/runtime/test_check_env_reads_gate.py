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


# ---------------------------------------------------------------------------
# OMN-15234: per-name grandfathering regression matrix
# ---------------------------------------------------------------------------
# The pre-OMN-15234 matcher was `grep -qE "^\+.*(os\.environ|...)"` -- it fired
# on ANY added line containing an env read. Editing the default value of a
# pre-existing, already-grandfathered read re-adds that line, so maintenance of
# an existing read was indistinguishable from introducing a new one. Every test
# below drives the REAL script as a bash subprocess against a REAL two-commit
# git repo; none of them inspects the script's source text.


def _init_repo(tmp_path: Path) -> dict[str, str]:
    git_env = _hermetic_git_env()
    for args in (
        ["git", "init", "-b", "base"],
        ["git", "config", "user.email", "test@example.com"],
        ["git", "config", "user.name", "Test"],
    ):
        subprocess.run(args, cwd=tmp_path, check=True, capture_output=True, env=git_env)
    return git_env


def _write(tmp_path: Path, files: dict[str, str]) -> None:
    for rel_path, content in files.items():
        full_path = tmp_path / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)


def _run_change(
    tmp_path: Path,
    base_files: dict[str, str],
    changed_files: dict[str, str],
    *,
    deleted: tuple[str, ...] = (),
    mode: str = "--staged",
) -> tuple[int, str]:
    """Commit ``base_files``, apply ``changed_files``, run the gate.

    ``mode="--staged"`` mirrors the pre-commit hook (change staged, uncommitted).
    ``mode="--base"`` mirrors the CI workflow (change committed on top of the
    base commit, gate run against that base SHA) -- the same script, the same
    checks, the other endpoint pair.
    """
    git_env = _init_repo(tmp_path)
    _write(tmp_path, base_files)
    subprocess.run(
        ["git", "add", "-A"], cwd=tmp_path, check=True, capture_output=True, env=git_env
    )
    subprocess.run(
        ["git", "commit", "-m", "base"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        env=git_env,
    )
    base_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        env=git_env,
    ).stdout.strip()

    _write(tmp_path, changed_files)
    for rel_path in deleted:
        (tmp_path / rel_path).unlink()
    subprocess.run(
        ["git", "add", "-A"], cwd=tmp_path, check=True, capture_output=True, env=git_env
    )

    argv = [str(SCRIPT), mode]
    if mode == "--base":
        subprocess.run(
            ["git", "commit", "-m", "change"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            env=git_env,
        )
        argv.append(base_sha)

    result = subprocess.run(
        ["bash", *argv],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env=git_env,
    )
    return result.returncode, result.stdout + result.stderr


# The exact file and read OMN-15233 could not edit without an allowlist entry.
HANDLER = (
    "src/omnibase_infra/nodes/node_runner_fleet_health_compute/handlers/"
    "handler_runner_fleet_health_evaluate.py"
)
HANDLER_AT_900 = (
    "import os\n"
    "_RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS = int(\n"
    '    os.environ.get("RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS", "900")\n'
    ")\n"
)
HANDLER_AT_4500 = HANDLER_AT_900.replace('"900"', '"4500"')


@pytest.mark.unit
class TestCheckEnvReadsGrandfatheredReadEdits:
    """OMN-15234 four-case matrix, plus the cases that must stay blocked."""

    def test_default_value_edit_on_grandfathered_read_is_allowed(
        self, tmp_path: Path
    ) -> None:
        """THE regression: 900 -> 4500 on a read that already exists in this file."""
        code, output = _run_change(
            tmp_path,
            {HANDLER: HANDLER_AT_900},
            {HANDLER: HANDLER_AT_4500},
        )
        assert code == 0, output

    def test_new_name_in_the_same_handler_file_is_still_blocked(
        self, tmp_path: Path
    ) -> None:
        """Bidirectional proof that the ALLOW above is narrowing, not an allowlist.

        This is the same path as the test above. If ``handler_runner_fleet_
        health_evaluate.py`` were still in ``APPROVED_INFIX_PATTERNS`` (the
        OMN-15233 entry this ticket removed), a brand-new env name here would
        pass -- so this test fails the moment the allowlist entry comes back.
        """
        code, output = _run_change(
            tmp_path,
            {HANDLER: HANDLER_AT_900},
            {
                HANDLER: HANDLER_AT_900
                + '_NEW = os.environ.get("RUNNER_HEALTH_BRAND_NEW_KNOB", "1")\n'
            },
        )
        assert code == 1
        assert "BLOCKED" in output
        assert "RUNNER_HEALTH_BRAND_NEW_KNOB" in output
        # The grandfathered name must NOT be reported.
        assert "'RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS' is not read" not in output

    def test_default_value_edit_on_a_never_allowlisted_file_is_allowed(
        self, tmp_path: Path
    ) -> None:
        """The same edit on a path that was never on any allowlist.

        ``HANDLER`` was allowlisted at OMN-15233, so an ALLOW there could in
        principle come from the allowlist rather than the narrowed matcher.
        ``src/mod.py`` never was, so this isolates the narrowing itself.
        """
        code, output = _run_change(
            tmp_path,
            {"src/mod.py": 'import os\nA = os.environ.get("EXISTING_NAME", "900")\n'},
            {"src/mod.py": 'import os\nA = os.environ.get("EXISTING_NAME", "4500")\n'},
        )
        assert code == 0, output

    def test_new_name_in_an_existing_file_is_blocked(self, tmp_path: Path) -> None:
        code, output = _run_change(
            tmp_path,
            {"src/mod.py": 'import os\nA = os.environ.get("EXISTING_NAME")\n'},
            {
                "src/mod.py": (
                    "import os\n"
                    'A = os.environ.get("EXISTING_NAME")\n'
                    'B = os.environ["A_BRAND_NEW_NAME"]\n'
                )
            },
        )
        assert code == 1
        assert "A_BRAND_NEW_NAME" in output

    def test_new_name_in_a_new_file_is_blocked(self, tmp_path: Path) -> None:
        code, output = _run_change(
            tmp_path,
            {"src/mod.py": 'import os\nA = os.environ.get("EXISTING_NAME")\n'},
            {"src/other.py": 'import os\nB = os.environ.get("A_BRAND_NEW_NAME")\n'},
        )
        assert code == 1
        assert "src/other.py" in output
        assert "A_BRAND_NEW_NAME" in output

    def test_existing_read_moved_to_a_different_file_is_blocked(
        self, tmp_path: Path
    ) -> None:
        """The boundary moved: the destination file's base has no such read.

        Grandfathering is per (file, name) -- otherwise the gate could be
        defeated by relocating a read into any module.
        """
        code, output = _run_change(
            tmp_path,
            {
                "src/mod.py": 'import os\nA = os.environ.get("EXISTING_NAME")\n',
                "src/other.py": "VALUE = 1\n",
            },
            {
                "src/mod.py": "import os\n",
                "src/other.py": 'import os\nA = os.environ.get("EXISTING_NAME")\n',
            },
        )
        assert code == 1
        assert "src/other.py" in output
        assert "EXISTING_NAME" in output

    def test_existing_read_relocated_within_the_same_file_is_allowed(
        self, tmp_path: Path
    ) -> None:
        """Same file, same name -- the read moved lines, not boundaries."""
        code, output = _run_change(
            tmp_path,
            {
                "src/mod.py": (
                    "import os\n"
                    'A = os.environ.get("EXISTING_NAME")\n'
                    "\n\ndef f():\n    return A\n"
                )
            },
            {
                "src/mod.py": (
                    "import os\n"
                    "\n\ndef f():\n"
                    '    return os.environ.get("EXISTING_NAME")\n'
                )
            },
        )
        assert code == 0, output

    def test_removing_a_read_is_allowed(self, tmp_path: Path) -> None:
        code, output = _run_change(
            tmp_path,
            {"src/mod.py": ('import os\nA = os.environ.get("EXISTING_NAME")\nB = 2\n')},
            {"src/mod.py": "B = 2\n"},
        )
        assert code == 0, output

    def test_deleting_the_whole_file_is_allowed(self, tmp_path: Path) -> None:
        code, output = _run_change(
            tmp_path,
            {
                "src/mod.py": 'import os\nA = os.environ.get("EXISTING_NAME")\n',
                "src/keep.py": "B = 2\n",
            },
            {"src/keep.py": "B = 3\n"},
            deleted=("src/mod.py",),
        )
        assert code == 0, output

    def test_dynamic_read_of_a_grandfathered_name_is_blocked(
        self, tmp_path: Path
    ) -> None:
        """Fail closed: no literal name means nothing to grandfather against."""
        code, output = _run_change(
            tmp_path,
            {"src/mod.py": 'import os\nA = os.environ.get("EXISTING_NAME")\n'},
            {
                "src/mod.py": (
                    "import os\n"
                    'A = os.environ.get("EXISTING_NAME")\n'
                    'NAME = "EXISTING_NAME"\n'
                    "B = os.environ.get(NAME)\n"
                )
            },
        )
        assert code == 1
        assert "BLOCKED" in output

    def test_bare_from_os_import_environ_stays_blocked_in_an_env_reading_file(
        self, tmp_path: Path
    ) -> None:
        code, output = _run_change(
            tmp_path,
            {"src/mod.py": 'import os\nA = os.environ.get("EXISTING_NAME")\n'},
            {
                "src/mod.py": (
                    "import os\n"
                    "from os import environ\n"
                    'A = os.environ.get("EXISTING_NAME")\n'
                )
            },
        )
        assert code == 1
        assert "BLOCKED" in output

    def test_a_second_name_added_alongside_a_grandfathered_one_on_one_line(
        self, tmp_path: Path
    ) -> None:
        """One added line can carry both a grandfathered and a new name."""
        code, output = _run_change(
            tmp_path,
            {"src/mod.py": 'import os\nA = os.environ.get("EXISTING_NAME")\n'},
            {
                "src/mod.py": (
                    "import os\n"
                    'A = os.environ.get("EXISTING_NAME") or os.environ.get("NEW_NAME")\n'
                )
            },
        )
        assert code == 1
        assert "NEW_NAME" in output


@pytest.mark.unit
class TestCheckEnvReadsBaseModeParity:
    """The CI surface (`--base <sha>`), not just the pre-commit surface.

    check-env-reads-gate.yml runs `scripts/check-env-reads.sh --base <base_sha>`.
    Grandfathering resolves the base blob from a DIFFERENT endpoint in that
    mode (merge-base of the ref and HEAD, matching the `REF...HEAD` three-dot
    diff), so both endpoints get their own proof.
    """

    def test_default_value_edit_is_allowed_in_base_mode(self, tmp_path: Path) -> None:
        code, output = _run_change(
            tmp_path,
            {HANDLER: HANDLER_AT_900},
            {HANDLER: HANDLER_AT_4500},
            mode="--base",
        )
        assert code == 0, output

    def test_new_name_is_blocked_in_base_mode(self, tmp_path: Path) -> None:
        code, output = _run_change(
            tmp_path,
            {HANDLER: HANDLER_AT_900},
            {
                HANDLER: HANDLER_AT_900
                + '_NEW = os.environ.get("RUNNER_HEALTH_BRAND_NEW_KNOB", "1")\n'
            },
            mode="--base",
        )
        assert code == 1
        assert "RUNNER_HEALTH_BRAND_NEW_KNOB" in output

    def test_new_file_is_blocked_in_base_mode(self, tmp_path: Path) -> None:
        code, output = _run_change(
            tmp_path,
            {"src/mod.py": "VALUE = 1\n"},
            {"src/other.py": 'import os\nB = os.getenv("A_BRAND_NEW_NAME")\n'},
            mode="--base",
        )
        assert code == 1
        assert "A_BRAND_NEW_NAME" in output
