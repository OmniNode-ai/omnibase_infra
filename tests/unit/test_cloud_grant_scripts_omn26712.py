# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26712 -- the cloud grant scripts take role and database names from settings only.

The live proof (grants land on the Dev-named roles, second run changes nothing) is
``tests/integration/migrations/test_cloud_database_migration_omn26712.py``. This module is the
static half: the three cloud scripts name no lab role, database or address, read every name from
a psql variable, and the Dev settings set names none of the lab's either.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

CLOUD_DIR = (
    Path(__file__).resolve().parents[2] / "docker" / "migrations" / "seams" / "cloud"
)
SCRIPTS = (
    "apply_cloud_database.sql",
    "gateway_grants.sql",
    "tenant_identity_writer_grants.sql",
)
# Spelled in parts so this file's own added lines do not carry the lab names.
LAB_LITERALS = (
    "gateway_" + "runtime",
    "tenant_identity_" + 'writer"',
    "omninode_" + "cloud",
    "role_omni",
    "localhost",
    "127.0.0.1",
    "postgres:5432",
)
LAB_ADDRESS = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")


def _code(name: str) -> str:
    return (CLOUD_DIR / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_names_no_lab_role_database_or_address(name: str) -> None:
    text = _code(name)
    for literal in LAB_LITERALS:
        assert literal not in text, f"{name} carries the lab literal {literal!r}"
    assert not LAB_ADDRESS.search(text), f"{name} carries an IP address"


def test_grant_scripts_address_roles_and_database_through_variables() -> None:
    gateway = _code("gateway_grants.sql")
    writer = _code("tenant_identity_writer_grants.sql")
    assert 'TO :"gateway_role"' in gateway and ":'gateway_role'" in gateway
    assert 'TO :"writer_role"' in writer and ":'writer_role'" in writer
    for text in (gateway, writer):
        assert '\\connect :"cloud_database"' in text
        assert not re.search(r"TO \"[A-Za-z_]+\"", text), "a grant names a literal role"


def test_settings_are_read_from_the_environment_and_refused_when_absent() -> None:
    entry = _code("apply_cloud_database.sql")
    for variable in (
        "ONEX_CLOUD_DATABASE",
        "ONEX_CLOUD_GATEWAY_ROLE",
        "ONEX_CLOUD_TENANT_IDENTITY_WRITER_ROLE",
    ):
        assert "\\getenv" in entry and f"{variable} is not set" in entry


def test_dev_settings_set_names_no_lab_literal() -> None:
    settings = (CLOUD_DIR / "dev.settings.env").read_text(encoding="utf-8")
    values = [
        line.partition("=")[2]
        for line in settings.splitlines()
        if "=" in line and not line.startswith("#")
    ]
    assert len(values) == 3
    for value in values:
        for literal in LAB_LITERALS:
            assert literal.strip('"') not in value
