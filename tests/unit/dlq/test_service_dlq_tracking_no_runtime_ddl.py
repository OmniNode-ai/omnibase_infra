# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests confirming ServiceDlqTracking contains no runtime DDL (OMN-12633).

Architecture invariant: the dlq_replay_history table is owned by
node_dlq_replay_effect and provisioned by the canonical forward migration
runner.  ServiceDlqTracking must NOT contain CREATE TABLE IF NOT EXISTS or
any _ensure_table_exists method — those are imperative Service* patterns
that violate CONTRACT/NODE/HANDLER (OMN-12525).

These tests are purely static (no DB required) and act as a ratchet: they
fail if the imperative DDL is re-introduced.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path


def _migration_sql_path() -> Path:
    here = Path(__file__).parent
    repo_root = here
    for _ in range(10):
        if (repo_root / "docker" / "migrations").exists():
            break
        repo_root = repo_root.parent
    return (
        repo_root
        / "docker"
        / "migrations"
        / "forward"
        / "086_create_dlq_replay_history.sql"
    )


def _schema_sql_path() -> Path:
    here = Path(__file__).parent
    repo_root = here
    for _ in range(10):
        if (repo_root / "src").exists():
            break
        repo_root = repo_root.parent
    return (
        repo_root
        / "src"
        / "omnibase_infra"
        / "schemas"
        / "schema_dlq_replay_history.sql"
    )


class TestServiceDlqTrackingNoRuntimeDdl:
    """ServiceDlqTracking must not own or execute table DDL (OMN-12633)."""

    def test_no_create_table_in_function_bodies(self) -> None:
        """CREATE TABLE must not appear in string literals inside function bodies.

        Module-level and class-level docstrings may reference the schema for
        documentation purposes.  Only string literals inside method bodies
        (FunctionDef nodes) are the imperative DDL surface we guard against.
        """
        from omnibase_infra.dlq import service_dlq_tracking

        source_file = Path(inspect.getfile(service_dlq_tracking))
        tree = ast.parse(source_file.read_text())

        offending: list[tuple[int, str]] = []
        for func_node in ast.walk(tree):
            if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(func_node):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if "CREATE TABLE" in node.value.upper():
                        offending.append(
                            (node.lineno, textwrap.shorten(node.value, 80))
                        )

        assert not offending, (
            "ServiceDlqTracking contains CREATE TABLE inside a method body "
            f"(OMN-12633): {offending}. "
            "Imperative DDL must be removed; table provisioning belongs in "
            "the canonical forward migration runner."
        )

    def test_no_ensure_table_exists_method(self) -> None:
        """_ensure_table_exists() must not exist on ServiceDlqTracking (OMN-12633)."""
        from omnibase_infra.dlq.service_dlq_tracking import ServiceDlqTracking

        assert not hasattr(ServiceDlqTracking, "_ensure_table_exists"), (
            "ServiceDlqTracking._ensure_table_exists() still exists — "
            "the runtime table-creation method must be removed (OMN-12633)."
        )

    def test_initialize_does_not_call_ensure_table(self) -> None:
        """ServiceDlqTracking.initialize() must not call _ensure_table_exists()."""
        from omnibase_infra.dlq.service_dlq_tracking import ServiceDlqTracking

        source = inspect.getsource(ServiceDlqTracking.initialize)
        assert "_ensure_table_exists" not in source, (
            "ServiceDlqTracking.initialize() still calls _ensure_table_exists() — "
            "that call must be removed (OMN-12633)."
        )

    def test_forward_migration_file_exists(self) -> None:
        """The canonical forward migration for dlq_replay_history must exist."""
        migration_path = _migration_sql_path()
        assert migration_path.exists(), (
            f"Forward migration not found at {migration_path} — "
            "create docker/migrations/forward/086_create_dlq_replay_history.sql (OMN-12633)."
        )

    def test_forward_migration_contains_create_table(self) -> None:
        """The forward migration must define the dlq_replay_history table."""
        migration_path = _migration_sql_path()
        if not migration_path.exists():
            return  # covered by test_forward_migration_file_exists
        sql = migration_path.read_text()
        assert "CREATE TABLE IF NOT EXISTS dlq_replay_history" in sql, (
            f"Migration {migration_path} does not define dlq_replay_history table."
        )

    def test_schema_sql_file_exists(self) -> None:
        """The node-owned schema SQL must exist in src/omnibase_infra/schemas/."""
        schema_path = _schema_sql_path()
        assert schema_path.exists(), (
            f"Schema SQL not found at {schema_path} — "
            "create src/omnibase_infra/schemas/schema_dlq_replay_history.sql (OMN-12633)."
        )
