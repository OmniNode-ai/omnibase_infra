# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19977 vendor identity for the metering-summary projection migration."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker" / "migrations" / "forward"
_NODE = "node_projection_metering_summary"
_CREATE = "0000_create_metering_summary.sql"
_GRANT = "0001_grant_tenant_projection_writer_metering_summary.sql"
_VENDORED = _FORWARD / "nodes" / _NODE / _CREATE
_VENDORED_GRANT = _FORWARD / "nodes" / _NODE / _GRANT
_MANIFEST = _FORWARD / "_ledger" / "application-migrations.tsv"
_CLASSES = _ROOT / "config" / "migration_classes.yaml"
_SHA256 = "040e7246125719fae9fc031976d89c6e0ad610c024a94f93e1ec3eb23ae18149"
_GRANT_SHA256 = "56aeaea91083bc4421813a23f43a83d0f96d157938cd5914bec6344ad47ba1e3"


def test_vendored_bytes_and_manifest_binding_are_exact() -> None:
    artifact_path = f"nodes/{_NODE}/{_CREATE}"
    assert hashlib.sha256(_VENDORED.read_bytes()).hexdigest() == _SHA256
    rows = {
        row.split("\t", 1)[0]: row.split("\t")
        for row in _MANIFEST.read_text(encoding="utf-8").splitlines()
        if row.strip()
    }
    assert rows[artifact_path] == [
        artifact_path,
        f"node:{_NODE}",
        f"node:{_NODE}",
        "omninode_internal",
        f"node:{_NODE}:{_CREATE}",
        _SHA256,
    ]
    grant_path = f"nodes/{_NODE}/{_GRANT}"
    assert hashlib.sha256(_VENDORED_GRANT.read_bytes()).hexdigest() == _GRANT_SHA256
    assert rows[grant_path] == [
        grant_path,
        f"node:{_NODE}",
        f"node:{_NODE}",
        "omninode_internal",
        f"node:{_NODE}:{_GRANT}",
        _GRANT_SHA256,
    ]


def test_migration_class_matches_the_unique_index_classifier() -> None:
    classes = yaml.safe_load(_CLASSES.read_text(encoding="utf-8"))["migrations"]
    assert classes[f"forward/nodes/{_NODE}/{_CREATE}"] == "forward-only"
    assert classes[f"forward/nodes/{_NODE}/{_GRANT}"] == "expand-only"


def test_vendored_migration_creates_the_table_and_its_unique_key_without_grants() -> (
    None
):
    sql = _VENDORED.read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS public.metering_summary" in sql
    assert "CREATE UNIQUE INDEX IF NOT EXISTS metering_summary_key" in sql
    assert "GRANT" not in sql
    grant_sql = _VENDORED_GRANT.read_text(encoding="utf-8")
    assert (
        "GRANT SELECT, INSERT, UPDATE ON public.metering_summary "
        "TO tenant_projection_writer;"
    ) in grant_sql
