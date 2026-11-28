# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The omninode_cloud database is declared in every shipped topology profile (OMN-26471 AC6).

The omninode_cloud ledger is ``public.schema_migrations(version, applied_at,
checksum)`` with ``PRIMARY KEY (version)`` -- read-only on the dev lane's
omninode_cloud database. It carries no stream or domain column, so the entry
declares ledger shape ``version_checksum`` (omnibase_core#2378) rather than
inventing the two columns.
"""

from __future__ import annotations

import pytest

from omnibase_core.enums.enum_migration_ledger_shape import EnumMigrationLedgerShape
from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.application_database import SUPPORTED_TOPOLOGY_PROFILES

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
def test_omninode_cloud_database_declared_with_version_checksum_ledger(
    profile: str,
) -> None:
    database = load_topology_profile(profile).databases["omninode_cloud"]

    assert database.physical_name == "omninode_cloud"
    ledger = database.checksum_ledgers[database.checksum_ledger]
    assert ledger.shape is EnumMigrationLedgerShape.VERSION_CHECKSUM
    assert ledger.schema_name == "public"
    assert ledger.relation == "schema_migrations"
    assert ledger.version_column == "version"
    assert ledger.checksum_column == "checksum"
    assert ledger.stream_column is None
    assert ledger.domain_column is None


@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
def test_omninode_cloud_binds_the_tenant_identity_writer_login(profile: str) -> None:
    database = load_topology_profile(profile).databases["omninode_cloud"]

    binding = database.bindings["tenant_identity_writer"]
    assert binding.dsn_env == "TENANT_IDENTITY_DATABASE_DSN"
    principal = database.principals[binding.principal]
    assert binding.principal == "tenant_identity_writer"
    assert principal.login is True
    assert principal.bypass_rls is False
