"""Tests for scripts/run-migrations.py migration runner."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[3] / "scripts" / "run-migrations.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("run_migrations", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.unit
class TestSequenceNumberExtraction:
    def test_extracts_number_from_docker_format(self):
        runner = load_runner()
        assert runner.extract_sequence_number("036_create_schema_migrations.sql") == 36

    def test_extracts_number_from_src_format(self):
        runner = load_runner()
        assert runner.extract_sequence_number("007_create_skill_executions.sql") == 7

    def test_raises_on_no_leading_number(self):
        runner = load_runner()
        with pytest.raises(ValueError, match="no leading sequence number"):
            runner.extract_sequence_number("init.sql")

    def test_detects_duplicate_sequence_numbers(self):
        runner = load_runner()
        files = [
            Path("006_create_manifest_injection_lifecycle_table.sql"),
            Path("006_create_skill_executions.sql"),
        ]
        with pytest.raises(ValueError, match="duplicate sequence number 6"):
            runner.validate_no_duplicates(files)
