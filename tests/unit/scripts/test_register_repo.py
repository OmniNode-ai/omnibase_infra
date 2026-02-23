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
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))


def _import_register_repo() -> object:
    """Import register-repo module (hyphenated name requires importlib)."""
    return importlib.import_module("register-repo")


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

    The comment at line ~623 of register-repo.py explicitly documents that
    seed-shared requires a live Infisical instance even for dry-run preview
    (to read back existing keys).  Operators with INFISICAL_ADDR unset must
    receive a clear error, not a silent zero exit.
    """

    @pytest.mark.unit
    def test_dry_run_exits_nonzero_when_infisical_addr_unset(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Should exit non-zero with an informative message when INFISICAL_ADDR is
        unset, even when --execute is not passed (dry-run mode)."""
        rr = _import_register_repo()

        env_file = tmp_path / ".env"
        env_file.write_text(
            "POSTGRES_HOST=192.168.86.200\n"
            "POSTGRES_PORT=5436\n"
            "CONSUL_HOST=192.168.86.200\n"
        )

        args = _make_dry_run_args(str(env_file))

        # Strip INFISICAL_ADDR from the environment entirely.
        env_without_addr = {
            k: v for k, v in __import__("os").environ.items() if k != "INFISICAL_ADDR"
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
        self, tmp_path: Path
    ) -> None:
        """Should exit non-zero when INFISICAL_ADDR is set but INFISICAL_PROJECT_ID is
        missing, even in dry-run mode."""
        rr = _import_register_repo()

        env_file = tmp_path / ".env"
        env_file.write_text("POSTGRES_HOST=192.168.86.200\n")

        args = _make_dry_run_args(str(env_file))

        env_with_addr_no_project = {
            k: v
            for k, v in __import__("os").environ.items()
            if k not in ("INFISICAL_PROJECT_ID",)
        }
        env_with_addr_no_project["INFISICAL_ADDR"] = "http://localhost:8880"

        with patch.dict("os.environ", env_with_addr_no_project, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                rr.cmd_seed_shared(args)  # type: ignore[attr-defined]

        assert exc_info.value.code != 0, (
            "cmd_seed_shared must exit non-zero when INFISICAL_PROJECT_ID is unset"
        )

    @pytest.mark.unit
    def test_dry_run_exits_nonzero_when_infisical_addr_invalid_scheme(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Should exit non-zero when INFISICAL_ADDR is set but has no http/https scheme."""
        rr = _import_register_repo()

        env_file = tmp_path / ".env"
        env_file.write_text("POSTGRES_HOST=192.168.86.200\n")

        args = _make_dry_run_args(str(env_file))

        bad_env = {
            k: v
            for k, v in __import__("os").environ.items()
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
        rr = _import_register_repo()

        env_file = tmp_path / ".env"
        # Provide at least one real-looking value so env_values is non-empty.
        env_file.write_text("POSTGRES_HOST=192.168.86.200\n")

        args = _make_dry_run_args(str(env_file))

        valid_env = {
            k: v
            for k, v in __import__("os").environ.items()
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
        rr = _import_register_repo()

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
        rr = _import_register_repo()

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
        rr = _import_register_repo()

        data: dict[str, object] = {
            "service_override_required": ["KAFKA_GROUP_ID", "POSTGRES_DSN"],
        }

        result = rr._service_override_required(data)  # type: ignore[attr-defined]
        assert result == frozenset({"KAFKA_GROUP_ID", "POSTGRES_DSN"})

    @pytest.mark.unit
    def test_non_list_type_raises_value_error(self) -> None:
        """A non-list value for service_override_required should still raise ValueError."""
        rr = _import_register_repo()

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
        rr = _import_register_repo()

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

        with patch.dict(
            "sys.modules",
            {
                "omnibase_infra.errors": MagicMock(
                    RuntimeHostError=mock_runtime_host_error,
                    SecretResolutionError=mock_secret_resolution_error,
                    InfraConnectionError=mock_infra_connection_error,
                ),
            },
        ):
            # Re-import after patching so the function picks up the mock errors.
            rr2 = importlib.reload(importlib.import_module("register-repo"))
            with pytest.raises(mock_infra_connection_error) as exc_info:
                rr2._upsert_secret(  # type: ignore[attr-defined]
                    adapter,
                    "MY_KEY",
                    "value",
                    "/shared/db/",
                    overwrite=False,
                    sanitize=str,
                )

        # The cause chain should preserve the original exception.
        assert exc_info.value.__cause__ is not None
        assert "SDK connection refused" in str(exc_info.value.__cause__)

    @pytest.mark.unit
    def test_bare_sdk_exception_is_abort_error(self) -> None:
        """After wrapping, the outer loop's _is_abort_error should return True
        because InfraConnectionError is a RuntimeHostError subclass."""
        rr = _import_register_repo()

        # _is_abort_error checks isinstance(exc, RuntimeHostError).
        # Simulate what happens when the wrapped exception reaches the outer loop.
        from omnibase_infra.errors import InfraConnectionError, RuntimeHostError

        wrapped = InfraConnectionError(
            "SDK raised unexpected error fetching secret /shared/db/MY_KEY: test"
        )
        assert rr._is_abort_error(wrapped) is True  # type: ignore[attr-defined]
        assert isinstance(wrapped, RuntimeHostError)
