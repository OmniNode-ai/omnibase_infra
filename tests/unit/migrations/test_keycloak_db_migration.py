# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Regression guard for Keycloak warm-volume database provisioning."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
KEYCLOAK_MIGRATION = (
    REPO_ROOT / "docker" / "migrations" / "forward" / "042_create_keycloak_db.sql"
)

pytestmark = pytest.mark.unit


def test_keycloak_database_migration_is_executable_not_placeholder() -> None:
    sql = KEYCLOAK_MIGRATION.read_text(encoding="utf-8")

    assert "CREATE DATABASE keycloak" in sql
    assert "WHERE NOT EXISTS" in sql
    assert "\\gexec" in sql
    assert "no-op" not in sql.lower()
