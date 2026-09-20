# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Both views migration 089 grants must be declared together (OMN-18862).

`089_savings_aggregate_views_per_tenant.sql` grants `SELECT` to
`tenant_projection_writer` on two read views, in one statement block, on
ADJACENT lines: `projection_delegation_savings` at `:716` and
`projection_cost_savings_overview` at `:717`. The OMN-17426 supplemental
declaration covered the second and missed the first, so from the day 089 landed
one of the pair was granted by this corpus and declared by nothing.

WHY THAT IS NOT COSMETIC
------------------------
The runtime resolves a projection binding against the DECLARATION, not against
the database. A relation the corpus has already granted is still refused at
boot with `principal ... lacks declared read`, and auto-wiring is fail-closed at
the process level, so one refusal crash-loops the whole runtime rather than
disabling one handler. That is the OMN-18768 outage, which ran four and a half
hours on a different relation with exactly this shape.

WHY THIS PAIR HAS NO EXPIRY TEST
--------------------------------
The OMN-18863 expiry suite tracks supplemental entries that BRIDGE a window
until an omnimarket contract declares the relation, keyed to the pull request
that will retire each one. These two are views, and a view is never a
`db_io.db_tables` entry, so no contract will ever declare them: the supplemental
declaration is the steady state rather than a bridge, and there is no retiring
pull request to name. Asserting an expiry that will never arrive would be a
test that can only ever pass.

What replaces it is the PAIRING. Both views come from one statement block in
one migration, so declaring one and not the other is the defect itself, and
that is what this asserts -- per instance, because the three files are kept
identical only by the generator.

Deliberately reads only files committed to this repository, so it runs in every
test job rather than solely where the cross-repo checkout exists. An assertion
that runs in one job is one workflow edit away from running in none (OMN-18863
hit exactly that against the skip-count ratchet).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]

# The two relations 089 grants to tenant_projection_writer, named rather than
# counted: a count stays stable if one is dropped while another is added.
_MIGRATION_089_VIEWS = (
    "projection_delegation_savings",
    "projection_cost_savings_overview",
)

_MIGRATION_089 = (
    "docker/migrations/forward/nodes/node_projection_savings/"
    "089_savings_aggregate_views_per_tenant.sql"
)

_INSTANCES = (
    "src/omnibase_infra/topology/instances/local.yaml",
    "src/omnibase_infra/topology/instances/onex-dev.yaml",
    "src/omnibase_infra/topology/instances/onex-prod.yaml",
)


def _declared_relations(instance_path: Path) -> set[str]:
    document = yaml.safe_load(instance_path.read_text(encoding="utf-8"))
    relations: set[str] = set()
    for database in document["databases"].values():
        for principal in database.get("principals", {}).values():
            for grant in principal.get("grants") or ():
                if grant.get("object_type") == "TABLE":
                    relations.update(grant.get("objects") or ())
    return relations


@pytest.mark.integration
@pytest.mark.parametrize("relation", _MIGRATION_089_VIEWS)
def test_migration_089_grants_the_view(relation: str) -> None:
    """The premise: 089 really does grant both. If it stops, this pair is moot."""
    sql = (REPO_ROOT / _MIGRATION_089).read_text(encoding="utf-8")
    assert relation in sql, (
        f"{relation} is no longer granted by {_MIGRATION_089}. If the grant was "
        "removed deliberately, remove its supplemental declaration in "
        "src/omnibase_infra/topology/table_grant_derivation.py too, and lower "
        "MAX_UNDECLARED if the residual moves."
    )


@pytest.mark.integration
@pytest.mark.parametrize("instance", _INSTANCES)
def test_both_089_views_are_declared_in_every_instance(instance: str) -> None:
    """Declared together, in all three instances, or the gap crash-loops a boot.

    Checked per instance rather than on ``local.yaml`` alone: the three files
    are kept identical only by the generator, and a drifted instance leaves
    whichever lane a rebuild targets refusing a binding the others accept.
    """
    declared = _declared_relations(REPO_ROOT / instance)
    missing = sorted(set(_MIGRATION_089_VIEWS) - declared)
    assert not missing, (
        f"{missing} are granted by {_MIGRATION_089} and declared by no principal "
        f"in {instance}. The runtime resolves a projection binding against the "
        "declaration, so this refuses auto-wiring at boot and takes the whole "
        "process down (OMN-18768). Both views are granted by one statement "
        "block; declare them together. Do not hand-edit the generated block -- "
        "add the entry to LEGACY_MIGRATION_TABLE_DECLARATIONS and regenerate."
    )
