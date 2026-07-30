# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the rebuilt PostgreSQL application ACL proof."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import cast

import psycopg2.extensions
import pytest

_SCRIPT = (
    Path(__file__).parents[3] / "scripts" / "ci" / "prove_application_database_acl.py"
)


def _load_proof(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.setenv(
        "ADMIN_DSN",
        "postgresql://proof_admin:proof-secret@postgres:5432/omnidash_analytics"
        "?application_name=acl-proof",  # pragma: allowlist secret
    )
    spec = importlib.util.spec_from_file_location(
        "test_prove_application_database_acl_module",
        _SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_admin_dsn_for_database_preserves_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof = _load_proof(monkeypatch)
    dsn_for_database = cast(
        "Callable[[str], str]",
        proof._admin_dsn_for_database,
    )

    parameters = psycopg2.extensions.parse_dsn(dsn_for_database("acl_scaffold_probe"))

    assert parameters == {
        "application_name": "acl-proof",
        "dbname": "acl_scaffold_probe",
        "host": "postgres",
        "password": "proof-secret",  # pragma: allowlist secret
        "port": "5432",
        "user": "proof_admin",
    }
