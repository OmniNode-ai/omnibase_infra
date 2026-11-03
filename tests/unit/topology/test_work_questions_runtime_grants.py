# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Work-question projections need runtime write admission in every profile."""

import pytest

from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _resolve_projection_database_target,
)
from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.application_database import SUPPORTED_TOPOLOGY_PROFILES

pytestmark = pytest.mark.unit

_TABLES = (
    "work_outcome_units",
    "orchestration_style_outcomes",
    "lane_drift_and_stalls",
    "brief_quality_vs_outcome",
    "subagent_waste",
)


@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
@pytest.mark.parametrize("table", _TABLES)
def test_work_question_writer_has_exact_runtime_privileges(
    profile: str, table: str
) -> None:
    topology = load_topology_profile(profile)
    declaration = ModelDbTableDeclaration(
        name=table,
        database_ref="application",
        schema="omninode_internal",
        migration=(
            "docker/migrations/forward/nodes/node_projection_work_questions/"
            "0000_create_work_questions.sql"
        ),
        access="read_write",
        role=table,
    )
    _resolve_projection_database_target((declaration,), topology)
    principals = topology.databases["application"].principals
    actual = {
        (principal_name, grant.schema, privilege.value)
        for principal_name, principal in principals.items()
        for grant in principal.grants
        if table in grant.objects
        for privilege in grant.privileges
    }
    assert actual == {
        ("omninode_runtime", "omninode_internal", privilege)
        for privilege in ("SELECT", "INSERT", "UPDATE")
    }
