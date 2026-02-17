# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for seed-infisical.py script (OMN-2287)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add scripts dir to path so we can import
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))


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
        assert any(r["key"] == "POSTGRES_DSN" for r in reqs)
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
