# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The lease projection migration passes the application database SQL gate.

The lease projection migration (flat 112 at the time, now
node_projection_lease/0001) from the OMN-21424 lease stack failed "Application Database Domain
Enforcement (OMN-15361)" twice over: the table was not schema-qualified, and no
ownership declaration named it. Both halves are proved here over the real file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.topology.application_database import load_topology_profile
from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
    STATE_IO_TABLE_DECLARATIONS,
)
from omnibase_infra.validation.application_database_domain_enforcement import (
    application_database_created_catalog_identities,
    lint_application_database_sql,
)
from scripts.ci.check_application_database_sql import (
    _supplemental_ownership_identities,
)

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).parents[2]
_MIGRATION = (
    _ROOT
    / "docker/migrations/forward/nodes/node_projection_lease"
    / "0001_create_lease_projection.sql"
)


def test_lease_projection_is_schema_qualified() -> None:
    sql = _MIGRATION.read_text(encoding="utf-8")
    assert lint_application_database_sql(sql, load_topology_profile("local")) == ()


def test_lease_projection_ownership_is_declared() -> None:
    sql = _MIGRATION.read_text(encoding="utf-8")
    created = {
        identity.identity
        for identity in application_database_created_catalog_identities(sql)
    }
    assert created, "the migration creates a catalog object"
    declared = {
        identity.identity
        for identity in _supplemental_ownership_identities(
            _ROOT,
            (),
            declarations=STATE_IO_TABLE_DECLARATIONS
            + LEGACY_MIGRATION_TABLE_DECLARATIONS,
        )
    }
    assert created <= declared
