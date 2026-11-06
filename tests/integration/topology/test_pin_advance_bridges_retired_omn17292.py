# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Six supplemental table bridges retired by the omnimarket pin advance (OMN-17292).

The pinned omnimarket contracts now declare these relations, so the hand-authored
``LEGACY_MIGRATION_TABLE_DECLARATIONS`` entries that carried them are deleted.
The vendored migration lineage must survive. This proof uses committed
repository state alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
)

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MIGRATIONS_ROOT = _REPO_ROOT / "docker" / "migrations" / "forward" / "nodes"

_RETIRED_RELATIONS = (
    ("delegation_lab_jobs", "node_llm_delegation_call_effect"),
    ("delegation_lab_job_outbox", "node_llm_delegation_call_effect"),
    ("delegation_budget_readings", None),
    ("compaction_receipts", "node_projection_compaction_receipts"),
    ("delegation_capability_cells", "node_projection_delegation_capability_map"),
    ("local_environment", "node_projection_local_environment"),
)


@pytest.mark.parametrize(
    ("relation", "package"), _RETIRED_RELATIONS, ids=[r[0] for r in _RETIRED_RELATIONS]
)
class TestTheBridgeWasRetired:
    def test_no_supplemental_bridge_remains(
        self, relation: str, package: str | None
    ) -> None:
        carried = {
            declaration.table.name
            for declaration in LEGACY_MIGRATION_TABLE_DECLARATIONS
        }
        assert relation not in carried

    def test_the_migration_lineage_is_still_in_the_tree(
        self, relation: str, package: str | None
    ) -> None:
        if package is None:
            pytest.skip("no vendored node package named for this relation")
        directory = _MIGRATIONS_ROOT / package
        assert directory.is_dir()
        assert any(directory.glob("*.sql"))
