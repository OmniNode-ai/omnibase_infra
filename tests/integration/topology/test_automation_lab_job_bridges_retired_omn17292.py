# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The pinned omnimarket contracts replaced five more supplemental bridges.

The omnimarket contract pin advance made the supplemental declarations for the
automation liveness and lab job tables redundant. Retiring a bridge removes
only its hand-authored declaration. The vendored migration lineage and the
exact grants in every shipped instance must survive. This proof uses committed
repository state and needs no foreign checkout.
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
_SCHEMA = "omninode_internal"
_PRINCIPAL = "omninode_runtime"
_REQUIRED_PRIVILEGES = frozenset({"SELECT", "INSERT", "UPDATE"})

_AUTOMATION_LIVENESS_MIGRATIONS = (
    "0000_create_automation_liveness.sql",
    "0001_grant_omninode_runtime_automation_liveness.sql",
)
_LAB_JOB_MIGRATIONS = (
    "0000_create_lab_job.sql",
    "0001_grant_omninode_runtime_lab_job.sql",
)

_RETIRED_BRIDGES = (
    (
        "automation_alarm_episodes",
        "node_projection_automation_liveness",
        _AUTOMATION_LIVENESS_MIGRATIONS,
    ),
    (
        "automation_liveness_state",
        "node_projection_automation_liveness",
        _AUTOMATION_LIVENESS_MIGRATIONS,
    ),
    (
        "automation_run_history",
        "node_projection_automation_liveness",
        _AUTOMATION_LIVENESS_MIGRATIONS,
    ),
    ("lab_job_state", "node_projection_lab_job", _LAB_JOB_MIGRATIONS),
    ("lab_job_transitions", "node_projection_lab_job", _LAB_JOB_MIGRATIONS),
)


def _instance_document(profile: str) -> dict[str, Any]:
    path = _INSTANCES_ROOT / f"{profile}.yaml"
    document: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict), f"{path} is not a topology mapping"
    return document


def _shipped_privileges(profile: str, relation: str) -> frozenset[str]:
    application = _instance_document(profile)["databases"]["application"]
    grants = application["principals"][_PRINCIPAL]["grants"]
    matching = [
        grant
        for grant in grants
        if grant["object_type"] == "TABLE"
        and grant["schema"] == _SCHEMA
        and relation in grant["objects"]
    ]
    assert len(matching) == 1, (
        f"{profile} ships {len(matching)} TABLE grants for {_SCHEMA}.{relation} "
        f"to {_PRINCIPAL}; expected exactly one"
    )
    return frozenset(matching[0]["privileges"])


@pytest.mark.parametrize(("relation", "node", "migrations"), _RETIRED_BRIDGES)
class TestTheBridgesWereRetired:
    def test_no_supplemental_bridge_remains(
        self, relation: str, node: str, migrations: tuple[str, ...]
    ) -> None:
        carried = {
            declaration.table.name
            for declaration in LEGACY_MIGRATION_TABLE_DECLARATIONS
        }
        assert relation not in carried, (
            f"{relation} still has a supplemental declaration, although the "
            "pinned omnimarket contract now declares it"
        )

    def test_vendored_migration_lineage_remains(
        self, relation: str, node: str, migrations: tuple[str, ...]
    ) -> None:
        for filename in migrations:
            migration = _MIGRATIONS_ROOT / node / filename
            assert (_REPO_ROOT / migration).is_file(), (
                f"{migration} is missing; retiring {relation}'s bridge must "
                "leave its vendored migration in the tree"
            )


def test_shipped_instances_are_found() -> None:
    assert _SHIPPED_INSTANCES, f"no shipped topology instance under {_INSTANCES_ROOT}"


@pytest.mark.parametrize("profile", _SHIPPED_INSTANCES)
@pytest.mark.parametrize(("relation", "node", "migrations"), _RETIRED_BRIDGES)
def test_shipped_instance_retains_exact_writer_privileges(
    profile: str, relation: str, node: str, migrations: tuple[str, ...]
) -> None:
    granted = _shipped_privileges(profile, relation)
    assert granted == _REQUIRED_PRIVILEGES, (
        f"{profile} grants {sorted(granted)} on {_SCHEMA}.{relation} to "
        f"{_PRINCIPAL}; expected exactly {sorted(_REQUIRED_PRIVILEGES)}"
    )
