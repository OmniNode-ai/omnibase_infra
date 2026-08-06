# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for scripts/check_node_migration_declarations.py (OMN-15717).

Pre-merge static gate mirroring the deploy-time
validate_application_migration_manifest() "exactly one declaration or block
per vendored node migration file" invariant.
"""

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).parents[3] / "scripts" / "check_node_migration_declarations.py"
)


def load_checker():
    spec = importlib.util.spec_from_file_location(
        "check_node_migration_declarations", SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_sql(root: Path, node: str, filename: str) -> Path:
    node_dir = root / node
    node_dir.mkdir(parents=True, exist_ok=True)
    sql_file = node_dir / filename
    sql_file.write_text("-- migration content\n")
    return sql_file


@pytest.mark.unit
class TestFindUndeclaredMigrations:
    def test_no_node_migrations_dir_returns_empty(self, tmp_path: Path):
        checker = load_checker()
        problems = checker.find_undeclared_migrations(
            tmp_path / "nowhere",
            tmp_path / "manifest.tsv",
            tmp_path / "blocks.tsv",
        )
        assert problems == []

    def test_declared_migration_is_not_flagged(self, tmp_path: Path):
        checker = load_checker()
        node_dir = tmp_path / "nodes"
        _write_sql(node_dir, "node_example", "0001_create_thing.sql")
        manifest = tmp_path / "manifest.tsv"
        manifest.write_text(
            "nodes/node_example/0001_create_thing.sql\tnode:node_example\t"
            "node:node_example\tomninode_internal\t"
            "node:node_example:0001_create_thing.sql\t" + ("a" * 64) + "\n"
        )
        blocks = tmp_path / "blocks.tsv"
        blocks.write_text("")

        problems = checker.find_undeclared_migrations(node_dir, manifest, blocks)
        assert problems == []

    def test_blocked_migration_is_not_flagged(self, tmp_path: Path):
        checker = load_checker()
        node_dir = tmp_path / "nodes"
        _write_sql(node_dir, "node_example", "0001_create_thing.sql")
        manifest = tmp_path / "manifest.tsv"
        manifest.write_text("")
        blocks = tmp_path / "blocks.tsv"
        blocks.write_text(
            "nodes/node_example/0001_create_thing.sql\t"
            "node:node_example:0001_create_thing.sql\t"
            + ("a" * 64)
            + "\tOMN-1\treason\n"
        )

        problems = checker.find_undeclared_migrations(node_dir, manifest, blocks)
        assert problems == []

    def test_undeclared_migration_is_flagged(self, tmp_path: Path):
        """RED case: this is the exact OMN-15717 defect shape -- a vendored
        migration file with zero rows in either declaration surface."""
        checker = load_checker()
        node_dir = tmp_path / "nodes"
        _write_sql(
            node_dir, "node_pr_review_bot", "001_create_review_bot_bypass_log.sql"
        )
        manifest = tmp_path / "manifest.tsv"
        manifest.write_text("")
        blocks = tmp_path / "blocks.tsv"
        blocks.write_text("")

        problems = checker.find_undeclared_migrations(node_dir, manifest, blocks)
        assert problems == [
            "nodes/node_pr_review_bot/001_create_review_bot_bypass_log.sql"
        ]

    def test_declared_and_blocked_is_flagged_as_ambiguous(self, tmp_path: Path):
        checker = load_checker()
        node_dir = tmp_path / "nodes"
        _write_sql(node_dir, "node_example", "0001_create_thing.sql")
        manifest = tmp_path / "manifest.tsv"
        manifest.write_text(
            "nodes/node_example/0001_create_thing.sql\tnode:node_example\t"
            "node:node_example\tomninode_internal\t"
            "node:node_example:0001_create_thing.sql\t" + ("a" * 64) + "\n"
        )
        blocks = tmp_path / "blocks.tsv"
        blocks.write_text(
            "nodes/node_example/0001_create_thing.sql\t"
            "node:node_example:0001_create_thing.sql\t"
            + ("a" * 64)
            + "\tOMN-1\treason\n"
        )

        problems = checker.find_undeclared_migrations(node_dir, manifest, blocks)
        assert problems == ["nodes/node_example/0001_create_thing.sql"]

    def test_missing_manifest_file_raises(self, tmp_path: Path):
        checker = load_checker()
        node_dir = tmp_path / "nodes"
        _write_sql(node_dir, "node_example", "0001_create_thing.sql")

        with pytest.raises(FileNotFoundError):
            checker.find_undeclared_migrations(
                node_dir,
                tmp_path / "missing-manifest.tsv",
                tmp_path / "missing-blocks.tsv",
            )


@pytest.mark.unit
class TestLiveRepoTree:
    """Guards the actual checked-in tree, not a synthetic fixture -- this is
    the test that would have caught OMN-15717 pre-merge."""

    def test_every_vendored_node_migration_is_declared(self):
        checker = load_checker()
        problems = checker.find_undeclared_migrations(
            checker.DEFAULT_NODE_MIGRATIONS_DIR,
            checker.DEFAULT_MANIFEST_PATH,
            checker.DEFAULT_BLOCKS_PATH,
        )
        assert problems == [], (
            f"undeclared node migration(s) found: {problems} -- add a row to "
            "docker/migrations/forward/_ledger/application-migrations.tsv "
            "(or an explicit block) before merge"
        )


CAPTURED_FIXTURE = (
    Path(__file__).parents[2]
    / "fixtures"
    / "omn15717"
    / "001_create_review_bot_bypass_log.sql.captured"
)


@pytest.mark.unit
class TestOmn15717IncidentReplay:
    """Incident replay case (OMN-15547 registry): drives the REAL guard
    against the exact captured bytes of node_pr_review_bot's
    001_create_review_bot_bypass_log.sql (git-object:OmniNode-ai/omnimarket@
    cedd24311ed320d34cc5ab5f8f79f5b04e9abf25:src/omnimarket/nodes/
    node_pr_review_bot/migrations/001_create_review_bot_bypass_log.sql),
    staged with NO manifest declaration -- reproducing the exact pre-OMN-15717
    tree shape that a guardless CI silently accepted (false_green) and that
    live bootstrap.sql later rejected at deploy time with "unknown migration
    stream/domain: adopted node version
    node:node_pr_review_bot:001_create_review_bot_bypass_log.sql has no
    checked-in declaration"."""

    def test_captured_undeclared_file_is_rejected(self, tmp_path: Path):
        assert CAPTURED_FIXTURE.is_file(), (
            f"incident replay fixture missing: {CAPTURED_FIXTURE}"
        )
        checker = load_checker()
        node_dir = tmp_path / "nodes"
        dest_dir = node_dir / "node_pr_review_bot"
        dest_dir.mkdir(parents=True)
        (dest_dir / "001_create_review_bot_bypass_log.sql").write_bytes(
            CAPTURED_FIXTURE.read_bytes()
        )
        manifest = tmp_path / "manifest.tsv"
        manifest.write_text("")
        blocks = tmp_path / "blocks.tsv"
        blocks.write_text("")

        problems = checker.find_undeclared_migrations(node_dir, manifest, blocks)

        assert problems == [
            "nodes/node_pr_review_bot/001_create_review_bot_bypass_log.sql"
        ], "the real guard must reject the captured OMN-15717 artifact when undeclared"
