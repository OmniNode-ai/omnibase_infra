# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

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


@pytest.mark.unit
class TestSplitSqlStatements:
    def test_splits_simple_statements(self):
        runner = load_runner()
        sql = "CREATE TABLE a (id INT);\nCREATE TABLE b (id INT);"
        stmts = runner.split_sql_statements(sql)
        assert len(stmts) == 2
        assert "CREATE TABLE a" in stmts[0]
        assert "CREATE TABLE b" in stmts[1]

    def test_preserves_dollar_quoted_blocks(self):
        runner = load_runner()
        sql = "DO $$ BEGIN RAISE NOTICE 'hi;there'; END $$;\nCREATE INDEX CONCURRENTLY idx ON t (c);"
        stmts = runner.split_sql_statements(sql)
        assert len(stmts) == 2
        assert "DO $$" in stmts[0]
        assert "RAISE NOTICE 'hi;there'" in stmts[0]
        assert "CREATE INDEX CONCURRENTLY" in stmts[1]

    def test_skips_comment_only_fragments(self):
        runner = load_runner()
        sql = "-- just a comment\nCREATE TABLE x (id INT);"
        stmts = runner.split_sql_statements(sql)
        assert len(stmts) == 1
        assert "CREATE TABLE x" in stmts[0]


@pytest.mark.unit
class TestConnectDirective:
    def test_extracts_leading_connect_directive(self):
        runner = load_runner()
        sql = "-- comment\n\n\\connect omnidash_analytics\nCREATE TABLE x (id INT);"

        database, cleaned = runner.parse_connect_directive(sql)

        assert database == "omnidash_analytics"
        assert "\\connect" not in cleaned
        assert "CREATE TABLE x" in cleaned

    def test_rejects_unsupported_meta_command(self):
        runner = load_runner()
        with pytest.raises(ValueError, match="unsupported psql meta-command"):
            runner.parse_connect_directive("CREATE TABLE x (id INT);\n\\dt\n")

    def test_builds_database_specific_url(self):
        runner = load_runner()
        db_url = (
            "postgresql://user:pass@example.test:5432/omnibase_infra?sslmode=prefer"
        )

        assert (
            runner.database_url_for_database(db_url, "omnidash_analytics")
            == "postgresql://user:pass@example.test:5432/omnidash_analytics?sslmode=prefer"
        )
