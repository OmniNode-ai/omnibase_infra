# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A sequence declared as a bare name under its domain resolves for its table's schema.

The cloud ownership manifest lists ``tenant_event_outbox_id_seq`` in
``database_objects`` as a bare name (``kind: sequence``, ``domain:
OMNINODE_INTERNAL``, ``dependencies: [tenant_event_outbox]``) with no
``database_ref`` or ``schema``. The loader skipped every such entry, so deployable
SQL naming ``public.tenant_event_outbox_id_seq`` was refused with ``requires exactly
one ownership declaration`` although the manifest declares it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.topology.application_database import load_topology_profile
from omnibase_infra.validation.application_database_domain_enforcement import (
    application_database_sql_target_requirements,
    load_application_database_ownership_identities,
)
from omnibase_infra.validation.enums.enum_application_inventory_object_kind import (
    EnumApplicationInventoryObjectKind,
)

pytestmark = pytest.mark.unit

_SEQUENCE = "tenant_event_outbox_id_seq"
_GRANT = f'GRANT USAGE ON SEQUENCE public.{_SEQUENCE} TO "tenant_identity_writer";\n'

_HEADER = """\
schema_version: "1.0"
service: onex_api
target_database_ref: application
db_io:
  db_tables:
    - name: tenant_event_outbox
      database_ref: application
      schema: omninode_internal
      migration: 20260816_tenant_event_outbox.sql
      access: read_write
      role: tenant_lifecycle_outbox
    - name: tenants
      database_ref: application
      schema: public
      migration: scripts/00_baseline_schema.sql
      access: read_write
      role: tenant_identity_root
"""

_OUTBOX_SEQUENCE = """\
  - name: tenant_event_outbox_id_seq
    kind: sequence
    domain: OMNINODE_INTERNAL
    owner_declaration: service:onex_api
    producer: omninode-cloud-migrate
    authoritative_sources: [db/migrations/20260816_tenant_event_outbox.sql]
    dependencies: [tenant_event_outbox]
"""

_OUTBOX_EVIDENCE = """\
relation_evidence:
  - name: tenant_event_outbox
    kind: table
    domain: {domain}
    owner_declaration: service:onex_api
    dependent_objects: [tenant_event_outbox_id_seq]
"""


def _manifest(
    tmp_path: Path,
    *,
    sequences: str = _OUTBOX_SEQUENCE,
    evidence: str = "",
) -> Path:
    manifest = tmp_path / "ownership.yaml"
    manifest.write_text(
        f"{_HEADER}{evidence}database_objects:\n{sequences}", encoding="utf-8"
    )
    return manifest


def _sequence_locations(manifest: Path) -> set[tuple[str, str]]:
    return {
        (identity.schema, identity.name)
        for identity in load_application_database_ownership_identities((manifest,))
        if identity.kind is EnumApplicationInventoryObjectKind.SEQUENCE
    }


def _unowned_grant_targets(manifest: Path) -> set[str]:
    identities = load_application_database_ownership_identities((manifest,))
    unowned: set[str] = set()
    for requirement in application_database_sql_target_requirements(
        _GRANT, load_topology_profile("local")
    ):
        if not any(
            identity.schema == requirement.schema
            and identity.name == requirement.name
            and identity.kind in requirement.allowed_kinds
            for identity in identities
        ):
            unowned.add(f"{requirement.schema}.{requirement.name}")
    return unowned


def test_bare_sequence_resolves_for_the_schema_its_table_occupies(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)

    assert ("public", _SEQUENCE) in _sequence_locations(manifest)


def test_grant_on_the_public_sequence_has_a_declared_owner(tmp_path: Path) -> None:
    assert _unowned_grant_targets(_manifest(tmp_path)) == set()


def test_sequence_without_an_owning_table_in_the_manifest_stays_unowned(
    tmp_path: Path,
) -> None:
    """Positive control: resolution follows the owning table, it is not a blanket pass."""
    orphan = _OUTBOX_SEQUENCE.replace(
        "dependencies: [tenant_event_outbox]", "dependencies: [absent_table]"
    )

    assert _unowned_grant_targets(_manifest(tmp_path, sequences=orphan)) == {
        f"public.{_SEQUENCE}"
    }


def test_bare_sequence_declared_twice_still_fails_exactly_one(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path, sequences=_OUTBOX_SEQUENCE + _OUTBOX_SEQUENCE)

    with pytest.raises(ValueError, match="exactly one ownership declaration"):
        load_application_database_ownership_identities((manifest,))


def test_bare_sequence_whose_domain_differs_from_its_table_is_refused(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path, evidence=_OUTBOX_EVIDENCE.format(domain="TENANT"))

    with pytest.raises(ValueError, match="domain"):
        load_application_database_ownership_identities((manifest,))


def test_bare_sequence_whose_domain_matches_its_table_is_accepted(
    tmp_path: Path,
) -> None:
    manifest = _manifest(
        tmp_path, evidence=_OUTBOX_EVIDENCE.format(domain="OMNINODE_INTERNAL")
    )

    assert ("public", _SEQUENCE) in _sequence_locations(manifest)
