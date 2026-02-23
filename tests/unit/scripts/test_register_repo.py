# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for register-repo.py script (OMN-2287).

Covers:
- cmd_seed_shared pre-flight validation (INFISICAL_ADDR, INFISICAL_PROJECT_ID)
  runs BEFORE the dry-run gate (even --dry-run exits non-zero when unset)
- _service_override_required empty-list semantics
- _upsert_secret bare SDK exception wrapping
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "scripts"
# Accepted risk: sys.path.append() at module import time persists for the entire
# test session (there is no cleanup).  This is a deliberate decision because
# hyphenated module names (like 'register-repo') are not valid Python identifiers
# and therefore cannot be imported by any other module via a normal `import`
# statement.  They cannot shadow stdlib, installed packages, or any other
# importable name in sys.path.  The risk of import confusion is negligible.
# append() (not insert(0, ...)) is used so scripts/ gets lowest priority in
# module resolution, further reducing any hypothetical conflict.
sys.path.append(str(_SCRIPTS_DIR))


def _import_register_repo() -> object:
    """Import register-repo module (hyphenated name requires importlib)."""
    return importlib.import_module("register-repo")


# Import once at module level so all test classes share the same module object.
_module = _import_register_repo()


def _make_dry_run_args(env_file: str) -> argparse.Namespace:
    """Build a minimal Namespace that mimics --dry-run (no --execute)."""
    return argparse.Namespace(
        env_file=env_file,
        execute=False,
        overwrite=False,
    )


# ---------------------------------------------------------------------------
# Issue 1: cmd_seed_shared pre-flight validation fires even in dry-run mode
# ---------------------------------------------------------------------------


class TestCmdSeedSharedPreflightValidation:
    """cmd_seed_shared validates INFISICAL_ADDR before the dry-run gate.

    Validation fires before the dry-run gate as a usability guard (early
    failure so the operator sees a misconfiguration immediately).  Dry-run
    itself never contacts Infisical — it exits after printing the plan
    without any network calls.  Operators with INFISICAL_ADDR unset must
    receive a clear error, not a silent zero exit.
    """

    @pytest.mark.unit
    def test_dry_run_exits_nonzero_when_infisical_addr_unset(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Should exit non-zero with an informative message when INFISICAL_ADDR is
        unset, even when --execute is not passed (dry-run mode)."""
        rr = _module

        env_file = tmp_path / ".env"
        env_file.write_text(
            "POSTGRES_HOST=192.168.86.200\n"
            "POSTGRES_PORT=5436\n"
            "CONSUL_HOST=192.168.86.200\n"
        )

        args = _make_dry_run_args(str(env_file))

        # Strip INFISICAL_ADDR from the environment entirely.
        env_without_addr = {
            k: v for k, v in os.environ.items() if k != "INFISICAL_ADDR"
        }
        env_without_addr.pop("INFISICAL_PROJECT_ID", None)

        with patch.dict("os.environ", env_without_addr, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                rr.cmd_seed_shared(args)  # type: ignore[attr-defined]

        # Must exit non-zero.
        assert exc_info.value.code != 0, (
            "cmd_seed_shared must exit non-zero when INFISICAL_ADDR is unset"
        )

        # Must print an informative message to stderr.
        captured = capsys.readouterr()
        assert "INFISICAL_ADDR" in captured.err, (
            "Error output must mention INFISICAL_ADDR so the operator knows what to fix; "
            f"got stderr: {captured.err!r}"
        )

    @pytest.mark.unit
    def test_dry_run_exits_nonzero_when_infisical_project_id_unset(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Should exit non-zero when INFISICAL_ADDR is set but INFISICAL_PROJECT_ID is
        missing, even in dry-run mode."""
        rr = _module

        env_file = tmp_path / ".env"
        env_file.write_text("POSTGRES_HOST=192.168.86.200\n")

        args = _make_dry_run_args(str(env_file))

        env_with_addr_no_project = {
            k: v for k, v in os.environ.items() if k not in ("INFISICAL_PROJECT_ID",)
        }
        env_with_addr_no_project["INFISICAL_ADDR"] = "http://localhost:8880"

        with patch.dict("os.environ", env_with_addr_no_project, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                rr.cmd_seed_shared(args)  # type: ignore[attr-defined]

        assert exc_info.value.code == 1, (
            "cmd_seed_shared must exit with integer code 1 when INFISICAL_PROJECT_ID is unset; "
            f"got {exc_info.value.code!r}"
        )

        captured = capsys.readouterr()
        assert "INFISICAL_PROJECT_ID" in captured.err, (
            "Error output must mention INFISICAL_PROJECT_ID so the operator knows what to fix; "
            f"got stderr: {captured.err!r}"
        )

    @pytest.mark.unit
    def test_dry_run_exits_nonzero_when_infisical_addr_invalid_scheme(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Should exit non-zero when INFISICAL_ADDR is set but has no http/https scheme."""
        rr = _module

        env_file = tmp_path / ".env"
        env_file.write_text("POSTGRES_HOST=192.168.86.200\n")

        args = _make_dry_run_args(str(env_file))

        bad_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("INFISICAL_ADDR", "INFISICAL_PROJECT_ID")
        }
        bad_env["INFISICAL_ADDR"] = "localhost:8880"  # missing scheme

        with patch.dict("os.environ", bad_env, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                rr.cmd_seed_shared(args)  # type: ignore[attr-defined]

        assert exc_info.value.code != 0, (
            "cmd_seed_shared must exit non-zero when INFISICAL_ADDR lacks http/https scheme"
        )

        captured = capsys.readouterr()
        assert "INFISICAL_ADDR" in captured.err, (
            f"Error output must mention INFISICAL_ADDR; got stderr: {captured.err!r}"
        )

    @pytest.mark.unit
    def test_dry_run_exits_zero_when_all_preflight_vars_present(
        self, tmp_path: Path
    ) -> None:
        """Should return 0 (dry-run success) when INFISICAL_ADDR and
        INFISICAL_PROJECT_ID are both valid — confirming that preflight passes
        and execution stops at the dry-run gate."""
        rr = _module

        env_file = tmp_path / ".env"
        # Provide at least one real-looking value so env_values is non-empty.
        env_file.write_text("POSTGRES_HOST=192.168.86.200\n")

        args = _make_dry_run_args(str(env_file))

        valid_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("INFISICAL_ADDR", "INFISICAL_PROJECT_ID")
        }
        valid_env["INFISICAL_ADDR"] = "http://localhost:8880"
        valid_env["INFISICAL_PROJECT_ID"] = "00000000-0000-0000-0000-000000000001"

        # Patch _read_registry_data to return a minimal valid registry so the
        # test does not require the real shared_key_registry.yaml to be present.
        minimal_registry: dict[str, object] = {
            "shared": {
                "/shared/db/": ["POSTGRES_HOST"],
            },
            "bootstrap_only": ["POSTGRES_PASSWORD"],
            "identity_defaults": ["POSTGRES_DATABASE"],
            "service_override_required": [],
        }

        with (
            patch.dict("os.environ", valid_env, clear=True),
            patch.object(
                rr,  # type: ignore[arg-type]
                "_read_registry_data",
                return_value=minimal_registry,
            ),
        ):
            result = rr.cmd_seed_shared(args)  # type: ignore[attr-defined]

        assert result == 0, (
            "cmd_seed_shared should return 0 in dry-run when pre-flight passes; "
            f"got {result}"
        )


# ---------------------------------------------------------------------------
# Issue 2: _service_override_required treats empty list as absent section
# ---------------------------------------------------------------------------


class TestServiceOverrideRequired:
    """_service_override_required treats [] identically to an absent section."""

    @pytest.mark.unit
    def test_empty_list_returns_empty_frozenset(self) -> None:
        """An empty service_override_required list should return frozenset(), not raise."""
        rr = _module

        data: dict[str, object] = {
            "shared": {"/shared/db/": ["POSTGRES_HOST"]},
            "bootstrap_only": ["POSTGRES_PASSWORD"],
            "identity_defaults": ["POSTGRES_DATABASE"],
            "service_override_required": [],
        }

        result = rr._service_override_required(data)  # type: ignore[attr-defined]
        assert result == frozenset(), (
            "Empty service_override_required list should return frozenset() "
            f"(same as absent section), got {result!r}"
        )

    @pytest.mark.unit
    def test_absent_section_returns_empty_frozenset(self) -> None:
        """A missing service_override_required section should return frozenset()."""
        rr = _module

        data: dict[str, object] = {
            "shared": {"/shared/db/": ["POSTGRES_HOST"]},
            "bootstrap_only": ["POSTGRES_PASSWORD"],
            "identity_defaults": ["POSTGRES_DATABASE"],
        }

        result = rr._service_override_required(data)  # type: ignore[attr-defined]
        assert result == frozenset()

    @pytest.mark.unit
    def test_non_empty_list_returns_frozenset_of_keys(self) -> None:
        """A non-empty list should return the expected frozenset."""
        rr = _module

        data: dict[str, object] = {
            "service_override_required": ["KAFKA_GROUP_ID", "POSTGRES_DSN"],
        }

        result = rr._service_override_required(data)  # type: ignore[attr-defined]
        assert result == frozenset({"KAFKA_GROUP_ID", "POSTGRES_DSN"})

    @pytest.mark.unit
    def test_non_list_type_raises_value_error(self) -> None:
        """A non-list value for service_override_required should still raise ValueError."""
        rr = _module

        data: dict[str, object] = {
            "service_override_required": "KAFKA_GROUP_ID",  # string, not list
        }

        with pytest.raises(ValueError, match="must be a list"):
            rr._service_override_required(data)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Issue 4: _upsert_secret wraps bare SDK exceptions in InfraConnectionError
# ---------------------------------------------------------------------------


class TestUpsertSecretBareExceptionWrapping:
    """Bare SDK exceptions from get_secret are wrapped in InfraConnectionError."""

    def _make_mock_adapter(self) -> MagicMock:
        adapter = MagicMock()
        adapter.create_secret.return_value = None
        adapter.update_secret.return_value = None
        return adapter

    @pytest.mark.unit
    def test_bare_sdk_exception_wraps_as_infra_connection_error(self) -> None:
        """A bare Exception from get_secret (not RuntimeHostError) should be
        re-raised as InfraConnectionError so the outer loop's _is_abort_error
        check correctly triggers an abort."""
        adapter = self._make_mock_adapter()
        adapter.get_secret.side_effect = ConnectionError("SDK connection refused")

        # Patch the omnibase_infra imports inside _upsert_secret.
        mock_runtime_host_error = type("RuntimeHostError", (Exception,), {})
        mock_secret_resolution_error = type(
            "SecretResolutionError", (mock_runtime_host_error,), {}
        )
        mock_infra_connection_error = type(
            "InfraConnectionError", (mock_runtime_host_error,), {}
        )

        mock_errors_module = MagicMock(
            RuntimeHostError=mock_runtime_host_error,
            SecretResolutionError=mock_secret_resolution_error,
            InfraConnectionError=mock_infra_connection_error,
        )

        # patch.dict restores sys.modules to its original state on exit,
        # regardless of whether the body raises or not (it uses a try/finally
        # internally).  The "register-repo": _module entry seeds the original
        # module object back so that on exit, sys.modules["register-repo"] is
        # restored to _module rather than the reloaded object produced by the
        # importlib.reload() call below.  This means _module remains the
        # canonical reference for all other tests even if this test fails with
        # an assertion error or unexpected exception — no explicit finally or
        # backup/restore is required.
        with patch.dict(
            "sys.modules",
            {
                "omnibase_infra.errors": mock_errors_module,
                "register-repo": _module,
            },
        ):
            # Reload inside the patch so the function body re-executes its
            # local `from omnibase_infra.errors import ...` against the mock.
            rr2 = importlib.reload(importlib.import_module("register-repo"))
            with pytest.raises(mock_infra_connection_error) as exc_info:
                rr2._upsert_secret(  # type: ignore[attr-defined]
                    adapter,
                    "MY_KEY",
                    "value",
                    "/shared/db/",
                    overwrite=False,
                )

        # The cause chain should preserve the original exception.
        assert exc_info.value.__cause__ is not None
        assert "SDK connection refused" in str(exc_info.value.__cause__)

    @pytest.mark.unit
    def test_bare_sdk_exception_is_abort_error(self) -> None:
        """After wrapping, the outer loop's _is_abort_error should return True
        because InfraConnectionError is a RuntimeHostError subclass."""
        rr = _module

        # _is_abort_error checks isinstance(exc, RuntimeHostError).
        # Simulate what happens when the wrapped exception reaches the outer loop.
        from omnibase_infra.errors import InfraConnectionError, RuntimeHostError

        wrapped = InfraConnectionError(
            "SDK raised unexpected error fetching secret /shared/db/MY_KEY: test"
        )
        assert rr._is_abort_error(wrapped) is True  # type: ignore[attr-defined]
        assert isinstance(wrapped, RuntimeHostError)


# ---------------------------------------------------------------------------
# cmd_onboard_repo tests
# ---------------------------------------------------------------------------


def _make_onboard_args(
    repo: str, env_file: str, execute: bool = False
) -> argparse.Namespace:
    """Build a minimal Namespace for cmd_onboard_repo."""
    return argparse.Namespace(
        repo=repo,
        env_file=env_file,
        execute=execute,
        overwrite=False,
    )


class TestCmdOnboardRepoDryRun:
    """cmd_onboard_repo dry-run behaviour: succeeds without Infisical credentials.

    Unlike cmd_seed_shared, cmd_onboard_repo validates INFISICAL_ADDR *after*
    the dry-run gate — so --dry-run (no --execute) can succeed even when no
    Infisical credentials are present in the environment.
    """

    @pytest.mark.unit
    def test_dry_run_prints_plan_without_credentials(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--dry-run should return 0 and print the onboarding plan without
        requiring INFISICAL_ADDR or INFISICAL_PROJECT_ID."""
        rr = _module

        env_file = tmp_path / ".env"
        # Provide one plausible repo-specific key so the plan is non-empty.
        env_file.write_text("POSTGRES_DATABASE=myrepo\n")

        args = _make_onboard_args("myrepo", str(env_file), execute=False)

        # Strip Infisical credentials completely to confirm they are not needed.
        env_no_creds = {
            k: v
            for k, v in os.environ.items()
            if k
            not in (
                "INFISICAL_ADDR",
                "INFISICAL_PROJECT_ID",
                "INFISICAL_CLIENT_ID",
                "INFISICAL_CLIENT_SECRET",
            )
        }

        minimal_registry: dict[str, object] = {
            "shared": {"/shared/db/": ["POSTGRES_HOST"]},
            "bootstrap_only": ["POSTGRES_PASSWORD", "KAFKA_BOOTSTRAP_SERVERS"],
            "identity_defaults": ["POSTGRES_DATABASE"],
            "service_override_required": [],
        }

        with (
            patch.dict("os.environ", env_no_creds, clear=True),
            patch.object(
                rr,  # type: ignore[arg-type]
                "_read_registry_data",
                return_value=minimal_registry,
            ),
        ):
            result = rr.cmd_onboard_repo(args)  # type: ignore[attr-defined]

        assert result == 0, (
            "cmd_onboard_repo should return 0 in dry-run regardless of Infisical credentials; "
            f"got {result}"
        )

        captured = capsys.readouterr()
        # The plan header must appear in stdout.
        assert "onboard-repo" in captured.out, (
            f"Expected plan output on stdout; got {captured.out!r}"
        )
        assert "dry-run" in captured.out.lower(), (
            f"Expected dry-run notice on stdout; got {captured.out!r}"
        )


class TestCmdOnboardRepoRequiredArgs:
    """cmd_onboard_repo enforces required arguments at the argparse level."""

    @pytest.mark.unit
    def test_exits_nonzero_when_repo_not_provided(self, tmp_path: Path) -> None:
        """Invoking the CLI via main() without --repo should exit non-zero.

        argparse marks --repo as required=True for the onboard-repo subcommand,
        so omitting it produces a non-zero exit before cmd_onboard_repo is called.
        """
        rr = _module

        env_file = tmp_path / ".env"
        env_file.write_text("POSTGRES_DATABASE=myrepo\n")

        # Drive through main() with argv patched to omit --repo.
        with patch(
            "sys.argv",
            ["register-repo", "onboard-repo", "--env-file", str(env_file)],
        ):
            with pytest.raises(SystemExit) as exc_info:
                rr.main()  # type: ignore[attr-defined]

        assert exc_info.value.code != 0, (
            "main() must exit non-zero when --repo is not provided to onboard-repo; "
            f"got exit code {exc_info.value.code!r}"
        )
