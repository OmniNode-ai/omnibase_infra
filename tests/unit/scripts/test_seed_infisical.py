# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for seed-infisical.py script (OMN-2287)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Script has hyphenated name so cannot use normal import; add scripts dir
# to sys.path and use importlib.import_module("seed-infisical") instead.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Shared fixtures and helpers for _do_seed tests
# ---------------------------------------------------------------------------

_VALID_ENV = {
    "INFISICAL_ADDR": "http://localhost:8880",
    "INFISICAL_CLIENT_ID": "test-client-id",
    "INFISICAL_CLIENT_SECRET": "test-client-secret",
    "INFISICAL_PROJECT_ID": "00000000-0000-0000-0000-000000000001",
}

_SAMPLE_REQUIREMENT: dict[str, str] = {
    "key": "POSTGRES_DSN",
    "transport_type": "db",
    "folder": "/shared/db/",
    "source": "transport",
}


def _make_mock_adapter(*, existing_secret: MagicMock | None = None) -> MagicMock:
    """Return a fresh AdapterInfisical mock.

    Args:
        existing_secret: The value returned by ``get_secret``. Pass ``None``
            to simulate a key that does not exist in Infisical.
    """
    adapter = MagicMock()
    adapter.get_secret.return_value = existing_secret
    adapter.create_secret.return_value = None
    adapter.update_secret.return_value = None
    adapter.initialize.return_value = None
    adapter.shutdown.return_value = None
    return adapter


class TestParseEnvFile:
    """Tests for _parse_env_file function."""

    def test_parse_simple_env(self, tmp_path: Path) -> None:
        """Should parse simple KEY=VALUE pairs."""
        from importlib import import_module

        # Reload to get fresh module
        seed = import_module("seed-infisical")
        env_file = tmp_path / ".env"
        env_file.write_text("DB_HOST=localhost\nDB_PORT=5432\nDB_NAME=test\n")
        values = seed._parse_env_file(env_file)
        assert values["DB_HOST"] == "localhost"
        assert values["DB_PORT"] == "5432"
        assert values["DB_NAME"] == "test"

    def test_parse_comments_and_empty_lines(self, tmp_path: Path) -> None:
        """Should skip comments and empty lines."""
        from importlib import import_module

        seed = import_module("seed-infisical")
        env_file = tmp_path / ".env"
        env_file.write_text(
            "# This is a comment\n\nKEY1=value1\n# Another comment\nKEY2=value2\n"
        )
        values = seed._parse_env_file(env_file)
        assert len(values) == 2
        assert "KEY1" in values
        assert "KEY2" in values

    def test_parse_quoted_values(self, tmp_path: Path) -> None:
        """Should strip quotes from values."""
        from importlib import import_module

        seed = import_module("seed-infisical")
        env_file = tmp_path / ".env"
        env_file.write_text("SINGLE='value1'\nDOUBLE=\"value2\"\nNONE=value3\n")
        values = seed._parse_env_file(env_file)
        assert values["SINGLE"] == "value1"
        assert values["DOUBLE"] == "value2"
        assert values["NONE"] == "value3"

    def test_parse_nonexistent_file(self, tmp_path: Path) -> None:
        """Should return empty dict for nonexistent file."""
        from importlib import import_module

        seed = import_module("seed-infisical")
        values = seed._parse_env_file(tmp_path / "nonexistent")
        assert values == {}


class TestExtractRequirements:
    """Tests for _extract_requirements function."""

    def test_extract_from_contracts(self, tmp_path: Path) -> None:
        """Should extract requirements from contract files."""
        from importlib import import_module

        seed = import_module("seed-infisical")

        # Create a contract
        (tmp_path / "handlers" / "db").mkdir(parents=True)
        (tmp_path / "handlers" / "db" / "contract.yaml").write_text(
            """
name: "handler_db"
metadata:
  transport_type: "database"
"""
        )

        reqs, _errors = seed._extract_requirements(tmp_path)
        assert len(reqs) > 0
        assert any(r["key"] == "POSTGRES_HOST" for r in reqs)
        assert any(r["transport_type"] == "db" for r in reqs)

    def test_extract_empty_dir(self, tmp_path: Path) -> None:
        """Should handle empty directory gracefully."""
        from importlib import import_module

        seed = import_module("seed-infisical")
        reqs, _errors = seed._extract_requirements(tmp_path)
        assert len(reqs) == 0


class TestPrintDiffSummary:
    """Tests for _print_diff_summary function."""

    def test_diff_summary_runs(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Should print diff summary without error."""
        from importlib import import_module

        seed = import_module("seed-infisical")
        requirements = [
            {
                "key": "POSTGRES_DSN",
                "transport_type": "db",
                "folder": "/shared/db/",
                "source": "transport",
            }
        ]
        env_values = {"POSTGRES_DSN": "postgresql://test"}

        seed._print_diff_summary(
            requirements,
            env_values,
            create_missing=True,
            set_values=False,
            overwrite_existing=False,
        )

        captured = capsys.readouterr()
        assert "POSTGRES_DSN" in captured.out
        assert "Seed Diff Summary" in captured.out


class TestMainEntryPoint:
    """Tests for main() function."""

    def test_main_dry_run(self, tmp_path: Path) -> None:
        """Should run in dry-run mode without errors."""
        from importlib import import_module

        seed = import_module("seed-infisical")

        # Create minimal contract
        contracts_dir = tmp_path / "nodes"
        (contracts_dir / "db").mkdir(parents=True)
        (contracts_dir / "db" / "contract.yaml").write_text(
            'name: "test"\nmetadata:\n  transport_type: "database"\n'
        )

        with patch(
            "sys.argv",
            [
                "seed-infisical.py",
                "--contracts-dir",
                str(contracts_dir),
                "--dry-run",
            ],
        ):
            result = seed.main()
            assert result == 0


class TestDoSeed:
    """Unit tests for _do_seed() -- the actual Infisical write path."""

    # ------------------------------------------------------------------
    # Internal helper: run _do_seed with mocked adapter and credentials.
    # ------------------------------------------------------------------

    def _run_do_seed(
        self,
        seed: object,
        *,
        requirements: list[dict[str, str]],
        env_values: dict[str, str],
        create_missing: bool = True,
        set_values: bool = False,
        overwrite_existing: bool = False,
        mock_adapter: MagicMock | None = None,
    ) -> tuple[int, int, int, int]:
        """Invoke ``_do_seed`` with fully mocked Infisical infrastructure.

        All dynamic imports inside ``_do_seed`` are patched so no real adapter
        or Infisical connection is needed.
        """
        if mock_adapter is None:
            mock_adapter = _make_mock_adapter()

        adapter_cls_mock = MagicMock(return_value=mock_adapter)
        config_cls_mock = MagicMock()
        secret_str_mock = MagicMock(side_effect=lambda v: v)

        # _do_seed imports these inside the function body; patch them.
        with (
            patch.dict("os.environ", _VALID_ENV),
            patch.dict(
                "sys.modules",
                {
                    "pydantic": MagicMock(SecretStr=secret_str_mock),
                    "omnibase_infra.adapters._internal.adapter_infisical": MagicMock(
                        AdapterInfisical=adapter_cls_mock
                    ),
                    "omnibase_infra.adapters.models.model_infisical_config": MagicMock(
                        ModelInfisicalAdapterConfig=config_cls_mock
                    ),
                    "omnibase_infra.errors": MagicMock(
                        InfraConnectionError=Exception,
                        InfraUnavailableError=Exception,
                    ),
                    "omnibase_infra.utils.util_error_sanitization": MagicMock(
                        sanitize_error_message=str
                    ),
                },
            ),
        ):
            return seed._do_seed(  # type: ignore[attr-defined]
                requirements,
                env_values,
                create_missing=create_missing,
                set_values=set_values,
                overwrite_existing=overwrite_existing,
            )

    # ------------------------------------------------------------------
    # (a) Create path: key is missing in Infisical and create_missing=True
    # ------------------------------------------------------------------

    @pytest.mark.unit
    def test_do_seed_creates_missing_key(self) -> None:
        """Should call create_secret when key does not exist and create_missing=True."""
        from importlib import import_module

        seed = import_module("seed-infisical")

        mock_adapter = _make_mock_adapter(existing_secret=None)
        created, updated, skipped, errors = self._run_do_seed(
            seed,
            requirements=[_SAMPLE_REQUIREMENT],
            env_values={"POSTGRES_DSN": "postgresql://test"},
            create_missing=True,
            set_values=True,
            overwrite_existing=False,
            mock_adapter=mock_adapter,
        )

        assert created == 1
        assert updated == 0
        assert skipped == 0
        assert errors == 0
        mock_adapter.create_secret.assert_called_once()

    # ------------------------------------------------------------------
    # (b) Update path: key exists and overwrite_existing=True
    # ------------------------------------------------------------------

    @pytest.mark.unit
    def test_do_seed_updates_existing_key_when_overwrite_true(self) -> None:
        """Should call update_secret when key already exists and overwrite_existing=True."""
        from importlib import import_module

        seed = import_module("seed-infisical")

        existing = MagicMock()  # simulate an existing secret object
        mock_adapter = _make_mock_adapter(existing_secret=existing)
        created, updated, skipped, errors = self._run_do_seed(
            seed,
            requirements=[_SAMPLE_REQUIREMENT],
            env_values={"POSTGRES_DSN": "postgresql://test"},
            create_missing=True,
            set_values=True,
            overwrite_existing=True,
            mock_adapter=mock_adapter,
        )

        assert created == 0
        assert updated == 1
        assert skipped == 0
        assert errors == 0
        mock_adapter.update_secret.assert_called_once()
        mock_adapter.create_secret.assert_not_called()

    # ------------------------------------------------------------------
    # (c) Skip path: key exists and overwrite_existing=False
    # ------------------------------------------------------------------

    @pytest.mark.unit
    def test_do_seed_skips_existing_key_when_overwrite_false(self) -> None:
        """Should skip key that already exists when overwrite_existing=False."""
        from importlib import import_module

        seed = import_module("seed-infisical")

        existing = MagicMock()
        mock_adapter = _make_mock_adapter(existing_secret=existing)
        created, updated, skipped, errors = self._run_do_seed(
            seed,
            requirements=[_SAMPLE_REQUIREMENT],
            env_values={"POSTGRES_DSN": "postgresql://test"},
            create_missing=True,
            set_values=True,
            overwrite_existing=False,
            mock_adapter=mock_adapter,
        )

        assert created == 0
        assert updated == 0
        assert skipped == 1
        assert errors == 0
        mock_adapter.create_secret.assert_not_called()
        mock_adapter.update_secret.assert_not_called()

    # ------------------------------------------------------------------
    # (d) Skip path: key missing and create_missing=False
    # ------------------------------------------------------------------

    @pytest.mark.unit
    def test_do_seed_skips_missing_key_when_create_missing_false(self) -> None:
        """Should skip key that does not exist when create_missing=False."""
        from importlib import import_module

        seed = import_module("seed-infisical")

        mock_adapter = _make_mock_adapter(existing_secret=None)
        created, updated, skipped, errors = self._run_do_seed(
            seed,
            requirements=[_SAMPLE_REQUIREMENT],
            env_values={"POSTGRES_DSN": "postgresql://test"},
            create_missing=False,
            set_values=False,
            overwrite_existing=False,
            mock_adapter=mock_adapter,
        )

        assert created == 0
        assert updated == 0
        assert skipped == 1
        assert errors == 0
        mock_adapter.create_secret.assert_not_called()
        mock_adapter.update_secret.assert_not_called()

    # ------------------------------------------------------------------
    # (e) Error path: create_secret raises, error_count increments
    # ------------------------------------------------------------------

    @pytest.mark.unit
    def test_do_seed_increments_error_count_when_create_secret_raises(self) -> None:
        """Should increment error_count and not created_count when create_secret raises."""
        from importlib import import_module

        seed = import_module("seed-infisical")

        mock_adapter = _make_mock_adapter(existing_secret=None)
        mock_adapter.create_secret.side_effect = RuntimeError("connection refused")

        created, updated, skipped, errors = self._run_do_seed(
            seed,
            requirements=[_SAMPLE_REQUIREMENT],
            env_values={"POSTGRES_DSN": "postgresql://test"},
            create_missing=True,
            set_values=True,
            overwrite_existing=False,
            mock_adapter=mock_adapter,
        )

        assert errors == 1
        assert created == 0
        assert updated == 0
        assert skipped == 0
        mock_adapter.create_secret.assert_called_once()

    # ------------------------------------------------------------------
    # Extra: multiple requirements -- verify per-key routing is independent
    # ------------------------------------------------------------------

    @pytest.mark.unit
    def test_do_seed_handles_multiple_requirements_independently(self) -> None:
        """Should route each requirement independently through create/update/skip."""
        from importlib import import_module

        seed = import_module("seed-infisical")

        existing = MagicMock()

        # First call (POSTGRES_DSN) -- key does not exist.
        # Second call (REDIS_URL)    -- key exists.
        mock_adapter = _make_mock_adapter()
        mock_adapter.get_secret.side_effect = [None, existing]

        requirements = [
            {
                "key": "POSTGRES_DSN",
                "folder": "/shared/db/",
                "transport_type": "db",
                "source": "transport",
            },
            {
                "key": "REDIS_URL",
                "folder": "/shared/cache/",
                "transport_type": "cache",
                "source": "transport",
            },
        ]
        created, updated, skipped, errors = self._run_do_seed(
            seed,
            requirements=requirements,
            env_values={"REDIS_URL": "redis://localhost:6379"},
            create_missing=True,
            set_values=True,
            overwrite_existing=True,
            mock_adapter=mock_adapter,
        )

        # POSTGRES_DSN: missing → create; REDIS_URL: exists + overwrite → update
        assert created == 1
        assert updated == 1
        assert skipped == 0
        assert errors == 0


class TestDoExport:
    """Unit tests for _do_export() -- the --export/--reveal path."""

    # ------------------------------------------------------------------
    # Internal helper: run _do_export with mocked adapter and credentials.
    # ------------------------------------------------------------------

    def _run_do_export(
        self,
        seed: object,
        *,
        reveal: bool = False,
        mock_adapter: MagicMock | None = None,
        list_secrets_side_effect: object = None,
    ) -> bool:
        """Invoke ``_do_export`` with fully mocked Infisical infrastructure.

        All dynamic imports inside ``_do_export`` are patched so no real
        adapter or Infisical connection is needed.
        """
        if mock_adapter is None:
            mock_adapter = MagicMock()
            mock_adapter.initialize.return_value = None
            mock_adapter.shutdown.return_value = None
            if list_secrets_side_effect is not None:
                mock_adapter.list_secrets.side_effect = list_secrets_side_effect
            else:
                mock_adapter.list_secrets.return_value = []

        adapter_cls_mock = MagicMock(return_value=mock_adapter)
        config_cls_mock = MagicMock()
        secret_str_mock = MagicMock(side_effect=lambda v: v)

        with (
            patch.dict("os.environ", _VALID_ENV),
            patch.dict(
                "sys.modules",
                {
                    "pydantic": MagicMock(SecretStr=secret_str_mock),
                    "omnibase_infra.adapters._internal.adapter_infisical": MagicMock(
                        AdapterInfisical=adapter_cls_mock
                    ),
                    "omnibase_infra.adapters.models.model_infisical_config": MagicMock(
                        ModelInfisicalAdapterConfig=config_cls_mock
                    ),
                },
            ),
        ):
            return seed._do_export(reveal=reveal)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # (a) Success with masked output (reveal=False, default)
    # ------------------------------------------------------------------

    @pytest.mark.unit
    def test_do_export_masked_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Should print key=**** for each secret when reveal=False."""
        from importlib import import_module

        seed = import_module("seed-infisical")

        # Build two fake secret objects with .key and .value attributes.
        secret_a = MagicMock()
        secret_a.key = "POSTGRES_DSN"

        secret_b = MagicMock()
        secret_b.key = "REDIS_URL"

        mock_adapter = MagicMock()
        mock_adapter.initialize.return_value = None
        mock_adapter.shutdown.return_value = None
        mock_adapter.list_secrets.return_value = [secret_a, secret_b]

        ok = self._run_do_export(seed, reveal=False, mock_adapter=mock_adapter)

        assert ok is True
        captured = capsys.readouterr()
        assert "POSTGRES_DSN=****" in captured.out
        assert "REDIS_URL=****" in captured.out
        # No actual values should appear in stdout when masked
        assert "get_secret_value" not in captured.out

    # ------------------------------------------------------------------
    # (b) Success with revealed values (reveal=True)
    # ------------------------------------------------------------------

    @pytest.mark.unit
    def test_do_export_revealed_output(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Should print key=value in plaintext when reveal=True."""
        from importlib import import_module

        seed = import_module("seed-infisical")

        secret_a = MagicMock()
        secret_a.key = "POSTGRES_DSN"
        secret_a.value.get_secret_value.return_value = "postgresql://localhost/test"

        mock_adapter = MagicMock()
        mock_adapter.initialize.return_value = None
        mock_adapter.shutdown.return_value = None
        mock_adapter.list_secrets.return_value = [secret_a]

        ok = self._run_do_export(seed, reveal=True, mock_adapter=mock_adapter)

        assert ok is True
        captured = capsys.readouterr()
        assert "POSTGRES_DSN=postgresql://localhost/test" in captured.out
        # Warning must go to stderr, not stdout
        assert "WARNING" in captured.err

    # ------------------------------------------------------------------
    # (c) Failure when list_secrets raises
    # ------------------------------------------------------------------

    @pytest.mark.unit
    def test_do_export_returns_false_on_list_secrets_error(self) -> None:
        """Should return False when list_secrets raises an exception."""
        from importlib import import_module

        seed = import_module("seed-infisical")

        ok = self._run_do_export(
            seed,
            reveal=False,
            list_secrets_side_effect=RuntimeError("connection refused"),
        )

        assert ok is False
