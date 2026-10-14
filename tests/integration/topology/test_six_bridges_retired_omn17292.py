# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Six supplemental table bridges were retired by the pin advance (OMN-17292).

Infra vendored these migrations before the omnimarket node contracts that
declare their relations landed, so hand-authored
``LEGACY_MIGRATION_TABLE_DECLARATIONS`` entries carried them in the interim:

- ``lab_job_state`` and ``lab_job_transitions`` (omnimarket#3434)
- ``automation_liveness_state``, ``automation_run_history`` and
  ``automation_alarm_episodes`` (OMN-20802)
- ``run_trace`` (OMN-19987)

The omnimarket contract pin advancing to 464634262791 carries all three, so the
bridges are deleted. The vendored migration lineage and the exact shipped
grants must survive. This proof uses committed repository state alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
)

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MIGRATIONS_ROOT = Path("docker/migrations/forward/nodes")
_INSTANCES_ROOT = _REPO_ROOT / "src" / "omnibase_infra" / "topology" / "instances"
_SHIPPED_INSTANCES = tuple(sorted(path.stem for path in _INSTANCES_ROOT.glob("*.yaml")))
_REQUIRED_PRIVILEGES = frozenset({"INSERT", "SELECT", "UPDATE"})

_LAB_JOB = (
    "node_projection_lab_job",
    ("0000_create_lab_job.sql", "0001_grant_omninode_runtime_lab_job.sql"),
)
_AUTOMATION_LIVENESS = (
    "node_projection_automation_liveness",
    (
        "0000_create_automation_liveness.sql",
        "0001_grant_omninode_runtime_automation_liveness.sql",
    ),
)

# (relation, schema, principal, node package, migrations)
_RETIRED_BRIDGES = (
    ("lab_job_state", "omninode_internal", "omninode_runtime", *_LAB_JOB),
    ("lab_job_transitions", "omninode_internal", "omninode_runtime", *_LAB_JOB),
    (
        "automation_liveness_state",
        "omninode_internal",
        "omninode_runtime",
        *_AUTOMATION_LIVENESS,
    ),
    (
        "automation_run_history",
        "omninode_internal",
        "omninode_runtime",
        *_AUTOMATION_LIVENESS,
    ),
    (
        "automation_alarm_episodes",
        "omninode_internal",
        "omninode_runtime",
        *_AUTOMATION_LIVENESS,
    ),
    (
        "run_trace",
        "public",
        "tenant_projection_writer",
        "node_projection_run_trace",
        (
            "0000_create_run_trace.sql",
            "0001_grant_tenant_projection_writer_run_trace.sql",
        ),
    ),
)


def _committed_privileges(
    profile: str, relation: str, schema: str, principal: str
) -> frozenset[str]:
    path = _INSTANCES_ROOT / f"{profile}.yaml"
    document: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    granted: set[str] = set()
    for database in document.get("databases", {}).values():
        entry = database.get("principals", {}).get(principal)
        if entry is None:
            continue
        for grant in entry.get("grants", ()):
            if (
                grant.get("object_type") == "TABLE"
                and grant.get("schema") == schema
                and relation in tuple(grant.get("objects") or ())
            ):
                granted.update(grant.get("privileges", ()))
    return frozenset(granted)


@pytest.mark.parametrize(
    ("relation", "schema", "principal", "package", "migrations"),
    _RETIRED_BRIDGES,
    ids=[bridge[0] for bridge in _RETIRED_BRIDGES],
)
class TestTheBridgeWasRetired:
    def test_every_shipped_instance_is_checked(
        self,
        relation: str,
        schema: str,
        principal: str,
        package: str,
        migrations: tuple[str, ...],
    ) -> None:
        assert len(_SHIPPED_INSTANCES) >= 3

    def test_no_supplemental_bridge_remains(
        self,
        relation: str,
        schema: str,
        principal: str,
        package: str,
        migrations: tuple[str, ...],
    ) -> None:
        carried = {
            declaration.table.name
            for declaration in LEGACY_MIGRATION_TABLE_DECLARATIONS
        }
        assert relation not in carried

    def test_the_migration_lineage_is_still_in_the_tree(
        self,
        relation: str,
        schema: str,
        principal: str,
        package: str,
        migrations: tuple[str, ...],
    ) -> None:
        for name in migrations:
            assert (_REPO_ROOT / _MIGRATIONS_ROOT / package / name).is_file()

    @pytest.mark.parametrize("instance", _SHIPPED_INSTANCES)
    def test_the_shipped_grant_survived(
        self,
        relation: str,
        schema: str,
        principal: str,
        package: str,
        migrations: tuple[str, ...],
        instance: str,
    ) -> None:
        granted = _committed_privileges(instance, relation, schema, principal)
        assert granted >= _REQUIRED_PRIVILEGES
