# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21920 vendor identity and grants for the agent-actions projection migrations."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker" / "migrations" / "forward"
_NODE = "node_projection_agent_actions"
_VENDOR = _FORWARD / "nodes" / _NODE
_MANIFEST = _FORWARD / "_ledger" / "application-migrations.tsv"
_CLASSES = _ROOT / "config" / "migration_classes.yaml"
_INSTANCES = _ROOT / "src" / "omnibase_infra" / "topology" / "instances"
_CATALOGS = _ROOT / "docker" / "catalog" / "database-topology"
_CREATE = "0000_create_agent_observability_tables.sql"
_GRANT = "0001_grant_tenant_projection_writer_agent_observability_tables.sql"
_SHA256 = {
    _CREATE: "a2782f589a32ab0b1597b1ba0c8959d0060c006d4962322f044d21ff1f7307b1",
    _GRANT: "7056fa7b3aa27974e4866348e9c0f5c936f8635962b859ad3bbe4cec0ae36346",
}
# The six tables node_projection_agent_actions writes. The seventh table of the
# retired consumer, agent_routing_decisions, is projected by
# node_projection_routing_decision and was granted before this change.
_TABLES = (
    "agent_actions",
    "agent_transformation_events",
    "router_performance_metrics",
    "agent_detection_failures",
    "agent_execution_logs",
    "agent_status_events",
)
_SHIPPED_PROFILES = sorted(path.stem for path in _INSTANCES.glob("*.yaml"))
_CATALOG_PROFILES = sorted(path.stem for path in _CATALOGS.glob("*.yaml"))
_WRITER_TABLES = {*_TABLES, "agent_routing_decisions"}


def _statements(filename: str) -> str:
    return "\n".join(
        line
        for line in (_VENDOR / filename).read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("--")
    )


def _writer_tables(instance: Path) -> set[str]:
    topology = yaml.safe_load(instance.read_text(encoding="utf-8"))
    grants = topology["databases"]["application"]["principals"][
        "tenant_projection_writer"
    ]["grants"]
    return {
        table
        for grant in grants
        if grant["object_type"] == "TABLE"
        and grant.get("schema") == "public"
        and set(grant["privileges"]) == {"INSERT", "SELECT", "UPDATE"}
        for table in grant["objects"]
    }


@pytest.mark.parametrize("filename", [_CREATE, _GRANT])
def test_vendor_bytes_and_manifest_binding_are_exact(filename: str) -> None:
    artifact_path = f"nodes/{_NODE}/{filename}"
    assert (
        hashlib.sha256((_VENDOR / filename).read_bytes()).hexdigest()
        == _SHA256[filename]
    )
    rows = {
        row.split("\t", 1)[0]: row.split("\t")
        for row in _MANIFEST.read_text(encoding="utf-8").splitlines()
        if row.strip()
    }
    assert rows[artifact_path] == [
        artifact_path,
        f"node:{_NODE}",
        f"node:{_NODE}",
        "tenant",
        f"node:{_NODE}:{filename}",
        _SHA256[filename],
    ]


def test_migration_classes_are_declared() -> None:
    classes = yaml.safe_load(_CLASSES.read_text(encoding="utf-8"))["migrations"]
    assert classes[f"forward/nodes/{_NODE}/{_CREATE}"] == "forward-only"
    assert classes[f"forward/nodes/{_NODE}/{_GRANT}"] == "expand-only"


def test_the_create_migration_builds_the_six_tables() -> None:
    create = _statements(_CREATE)
    for table in _TABLES:
        assert re.search(rf"CREATE TABLE IF NOT EXISTS public\.{table}\s*\(", create), (
            table
        )


def test_the_grant_migration_grants_exactly_the_six_tables_and_no_delete() -> None:
    grants = _statements(_GRANT)
    assert "GRANT USAGE ON SCHEMA public TO tenant_projection_writer;" in grants
    granted = re.search(
        r"GRANT\s+SELECT,\s*INSERT,\s*UPDATE\s+ON\s+(.*?)\s+TO\s+tenant_projection_writer;",
        grants,
        re.DOTALL,
    )
    assert granted is not None
    assert re.findall(r"public\.(\w+)", granted.group(1)) == list(_TABLES)
    assert not re.search(
        r"GRANT\s+[^;]*\bDELETE\b[^;]*\bTO\s+tenant_projection_writer\b",
        grants,
        re.IGNORECASE,
    )
    assert "has_table_privilege(" in grants


@pytest.mark.parametrize("profile", _SHIPPED_PROFILES)
def test_every_shipped_instance_grants_the_writer_all_six_tables(profile: str) -> None:
    assert _writer_tables(_INSTANCES / f"{profile}.yaml") >= _WRITER_TABLES


def test_the_shipped_instances_are_found() -> None:
    """The parametrised grant check above ran on something."""
    assert len(_SHIPPED_PROFILES) >= 3


@pytest.mark.parametrize("profile", _CATALOG_PROFILES)
def test_every_rendered_catalog_carries_the_seven_grants(profile: str) -> None:
    path = _CATALOGS / f"{profile}.yaml"
    assert _writer_tables(path) >= _WRITER_TABLES, profile
    catalog = yaml.safe_load(path.read_text(encoding="utf-8"))
    source = catalog["source"]
    assert (
        source["sha256"]
        == hashlib.sha256((_ROOT / source["path"]).read_bytes()).hexdigest()
    )
    assert (
        source["profile_catalog_sha256"]
        == hashlib.sha256(
            (_ROOT / source["profile_catalog_path"]).read_bytes()
        ).hexdigest()
    )


def test_all_shipped_catalog_profiles_are_checked() -> None:
    from omnibase_infra.topology import SUPPORTED_TOPOLOGY_PROFILES

    assert set(_CATALOG_PROFILES) == SUPPORTED_TOPOLOGY_PROFILES


def test_the_grant_check_is_live_positive_control(tmp_path: Path) -> None:
    """Remove a relation from the document before exercising the extractor."""
    topology = yaml.safe_load((_INSTANCES / "local.yaml").read_text(encoding="utf-8"))
    grants = topology["databases"]["application"]["principals"][
        "tenant_projection_writer"
    ]["grants"]
    for grant in grants:
        if "agent_status_events" in grant.get("objects", []):
            grant["objects"].remove("agent_status_events")
    path = tmp_path / "local.yaml"
    path.write_text(yaml.safe_dump(topology), encoding="utf-8")
    assert "agent_status_events" not in _writer_tables(path)
    assert not _writer_tables(path) >= _WRITER_TABLES


@pytest.mark.parametrize("filename", [_CREATE, _GRANT])
@pytest.mark.parametrize("profile", _SHIPPED_PROFILES)
def test_migrations_pass_the_application_database_sql_gate(
    filename: str, profile: str
) -> None:
    from omnibase_infra.topology.application_database import load_topology_profile
    from omnibase_infra.validation.application_database_domain_enforcement import (
        lint_application_database_sql,
    )

    sql = (_VENDOR / filename).read_text(encoding="utf-8")
    violations = lint_application_database_sql(sql, load_topology_profile(profile))
    assert violations == (), f"{profile}/{filename}: {violations}"


def test_the_sql_gate_is_live_positive_control() -> None:
    from omnibase_infra.topology.application_database import load_topology_profile
    from omnibase_infra.validation.application_database_domain_enforcement import (
        lint_application_database_sql,
    )

    sql = (_VENDOR / _GRANT).read_text(encoding="utf-8")
    broken = sql.replace("public.agent_actions", "tenant.agent_actions")
    assert broken != sql
    assert lint_application_database_sql(broken, load_topology_profile("local")) != ()


@pytest.mark.parametrize(
    "mutation", ["principal", "privileges", "schema", "source_hash"]
)
def test_catalog_grant_check_refuses_a_misplaced_or_stale_grant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    """Names elsewhere in a catalog do not establish a valid writer grant."""
    catalog = yaml.safe_load((_CATALOGS / "local.yaml").read_text(encoding="utf-8"))
    principals = catalog["databases"]["application"]["principals"]
    writer = principals["tenant_projection_writer"]
    if mutation == "principal":
        principals["unrelated_writer"] = principals.pop("tenant_projection_writer")
    elif mutation == "privileges":
        for grant in writer["grants"]:
            if "agent_actions" in grant.get("objects", []):
                grant["privileges"] = ["SELECT"]
    elif mutation == "schema":
        for grant in writer["grants"]:
            if "agent_actions" in grant.get("objects", []):
                grant["schema"] = "omninode_internal"
    else:
        catalog["source"]["sha256"] = "0" * 64
    (tmp_path / "local.yaml").write_text(yaml.safe_dump(catalog), encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "_CATALOGS", tmp_path)
    with pytest.raises((AssertionError, KeyError)):
        test_every_rendered_catalog_carries_the_seven_grants("local")
