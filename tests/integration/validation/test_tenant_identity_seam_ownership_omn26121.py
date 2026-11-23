# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage: the tenant identity seam's SQL targets resolve to owners.

Drives the real seam file through the production resolution path
(``lint_application_database_sql``, ``application_database_sql_target_requirements``
and ``load_application_database_ownership_identities``), the three calls the
Application Database Domain Enforcement gate makes for changed deployable SQL.
The ownership manifest here declares the eight relations at the logical schemas
the cloud ownership manifest declares them at: six in ``public``, the two plan
tables at ``platform_catalog`` and the outbox at ``omninode_internal``. The seam
names all eight at ``public`` because the cloud database holds them there.
"""

from pathlib import Path

import pytest

from omnibase_infra.topology.application_database import load_topology_profile
from omnibase_infra.validation.application_database_domain_enforcement import (
    application_database_sql_target_requirements,
    lint_application_database_sql,
    load_application_database_ownership_identities,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SEAM = _REPO_ROOT / "docker" / "migrations" / "seams" / "tenant_identity_schema.sql"

_LOGICAL_SCHEMA_BY_TABLE = {
    "tenants": "public",
    "tenant_api_keys": "public",
    "tenant_plans": "public",
    "gateway_workflows": "public",
    "gateway_rate_limit_events": "public",
    "plan_tiers": "platform_catalog",
    "plan_entitlements": "platform_catalog",
    "tenant_event_outbox": "omninode_internal",
}


def _write_manifest(tmp_path: Path, tables: dict[str, str]) -> Path:
    lines = [
        "schema_version: '1.0'",
        "service: omn26121_seam_fixture",
        "target_database_ref: application",
        "db_io:",
        "  db_tables:",
    ]
    for name, schema in tables.items():
        lines += [
            f"    - name: {name}",
            "      database_ref: application",
            f"      schema: {schema}",
            "      migration: fixture.sql",
            "      access: read_write",
            f"      role: omn26121_seam_fixture_{name}",
        ]
    manifest = tmp_path / "ownership.yaml"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def _unowned_targets(manifest: Path) -> set[str]:
    sql = _SEAM.read_text(encoding="utf-8")
    topology = load_topology_profile("local")
    identities = load_application_database_ownership_identities((manifest,))
    unowned: set[str] = set()
    for requirement in application_database_sql_target_requirements(sql, topology):
        located = [
            identity
            for identity in identities
            if identity.schema == requirement.schema
            and identity.name == requirement.name
            and identity.kind in requirement.allowed_kinds
        ]
        if not located:
            unowned.add(f"{requirement.schema}.{requirement.name}")
    return unowned


@pytest.mark.integration
def test_every_seam_target_resolves_to_a_declared_owner(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, _LOGICAL_SCHEMA_BY_TABLE)

    sql = _SEAM.read_text(encoding="utf-8")
    assert not lint_application_database_sql(sql, load_topology_profile("local"))
    assert _unowned_targets(manifest) == set()


@pytest.mark.integration
def test_seam_target_without_a_declaration_is_reported_unowned(
    tmp_path: Path,
) -> None:
    """Positive control: the resolution is not vacuous, an undeclared table fails."""
    declared = {
        name: schema
        for name, schema in _LOGICAL_SCHEMA_BY_TABLE.items()
        if name != "tenant_event_outbox"
    }
    manifest = _write_manifest(tmp_path, declared)

    assert _unowned_targets(manifest) == {"public.tenant_event_outbox"}
