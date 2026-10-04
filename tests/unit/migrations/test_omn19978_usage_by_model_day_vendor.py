# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19978 vendor identity for the usage-by-model-day node migrations."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker" / "migrations" / "forward"
_NODE = "node_projection_usage_by_model_day"
_VENDOR = _FORWARD / "nodes" / _NODE
_MANIFEST = _FORWARD / "_ledger" / "application-migrations.tsv"
_CLASSES = _ROOT / "config" / "migration_classes.yaml"
_CREATE = "0000_create_usage_by_model_day.sql"
_GRANT = "0001_grant_usage_by_model_day.sql"
# OMN-20006: measured cost apart from estimates, additive columns only.
_MEASURED = "0002_usage_by_model_day_measured_cost.sql"
_TABLES = ("usage_by_model_day_calls", "usage_by_model_day")
_SHA256 = {
    _CREATE: "0763b51e2d8fc794001d3ab1f9a39618a281fdf55c9acf89677a7e84ef0e9683",
    _GRANT: "2bf2ec13f194a2dfa0738cb57f65a765b730a1739416d40e7126678254fe979f",
    # Byte-identical to omnimarket's node migration of the same name.
    _MEASURED: "193470741af2e3e7e773a32583564f91bce266cfe8a194a2ae9cf8debd444605",
}


def _sql(filename: str) -> str:
    return (_VENDOR / filename).read_text(encoding="utf-8")


def _statements(filename: str) -> str:
    return "\n".join(
        line
        for line in _sql(filename).splitlines()
        if not line.lstrip().startswith("--")
    )


def _manifest_rows() -> dict[str, list[str]]:
    return {
        row.split("\t", 1)[0]: row.split("\t")
        for row in _MANIFEST.read_text(encoding="utf-8").splitlines()
        if row.strip()
    }


@pytest.mark.parametrize("filename", [_CREATE, _GRANT, _MEASURED])
def test_vendor_bytes_and_manifest_binding_are_exact(filename: str) -> None:
    artifact_path = f"nodes/{_NODE}/{filename}"
    assert (
        hashlib.sha256((_VENDOR / filename).read_bytes()).hexdigest()
        == _SHA256[filename]
    )
    assert _manifest_rows()[artifact_path] == [
        artifact_path,
        f"node:{_NODE}",
        f"node:{_NODE}",
        "tenant",
        f"node:{_NODE}:{filename}",
        _SHA256[filename],
    ]


def test_migration_classes_match_the_classifier() -> None:
    classes = yaml.safe_load(_CLASSES.read_text(encoding="utf-8"))["migrations"]
    for filename in (_CREATE, _GRANT, _MEASURED):
        assert classes[f"forward/nodes/{_NODE}/{filename}"] == "forward-only"


def test_the_measured_cost_migration_only_adds_columns() -> None:
    """OMN-20006: old rows read unknown / NULL / 0 and nothing is rewritten, so
    it applies on every lane without a fence and old code ignores it."""
    measured = _statements(_MEASURED)
    assert re.findall(
        r"ALTER TABLE public\.(\w+) ADD COLUMN IF NOT EXISTS (\w+)", measured
    ) == [
        ("usage_by_model_day_calls", "usage_source"),
        ("usage_by_model_day", "measured_cost_usd"),
        ("usage_by_model_day", "unmeasured_call_count"),
    ]
    for forbidden in ("DROP", "UPDATE", "DELETE", "TRUNCATE", "ROW LEVEL"):
        assert forbidden not in measured.upper()
    fenced = yaml.safe_load(
        (_FORWARD / "fenced-node-migrations.yaml").read_text(encoding="utf-8")
    )
    assert f"node:{_NODE}:{_MEASURED}" not in {
        entry["id"] for entry in fenced["fenced_node_migrations"]
    }


def test_tenant_tables_and_writer_grants_are_exact() -> None:
    create = _statements(_CREATE)
    grant = _statements(_GRANT)
    for table in _TABLES:
        assert re.search(rf"CREATE TABLE IF NOT EXISTS public\.{table}\s*\(", create)
        assert f"public.{table}" in grant
    assert "GRANT USAGE ON SCHEMA public TO tenant_projection_writer;" in grant
    assert re.search(
        r"GRANT SELECT, INSERT, UPDATE\s+ON public\.usage_by_model_day_calls, "
        r"public\.usage_by_model_day\s+TO tenant_projection_writer;",
        grant,
    )
    assert not re.search(
        r"GRANT\s+[^;]*\bDELETE\b[^;]*\bTO\s+tenant_projection_writer\b",
        grant,
        re.IGNORECASE,
    )


@pytest.mark.parametrize("profile", ["local", "onex-dev", "onex-prod"])
def test_the_shipped_tenant_writer_grants_are_read_write(profile: str) -> None:
    # The interim LEGACY_MIGRATION_TABLE_DECLARATIONS bridges for these relations
    # were retired when the omnimarket contract pin reached omnimarket#3073, which
    # declares both (tests/integration/topology/
    # test_vendored_bridges_retired_omn17292.py). What must not change is the
    # grant each shipped topology instance carries.
    from omnibase_infra.topology.table_grant_derivation import (
        LEGACY_MIGRATION_TABLE_DECLARATIONS,
    )

    assert not any(
        entry.table.name in _TABLES for entry in LEGACY_MIGRATION_TABLE_DECLARATIONS
    )
    instance = yaml.safe_load(
        (
            _ROOT / "src/omnibase_infra/topology/instances" / f"{profile}.yaml"
        ).read_text()
    )
    for table in _TABLES:
        shipped = [
            grant
            for grant in instance["databases"]["application"]["principals"][
                "tenant_projection_writer"
            ]["grants"]
            if grant["object_type"] == "TABLE"
            and grant["schema"] == "public"
            and table in grant["objects"]
        ]
        assert len(shipped) == 1
        assert set(shipped[0]["privileges"]) == {"SELECT", "INSERT", "UPDATE"}
