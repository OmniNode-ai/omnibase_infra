# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Activation persistence must pass the same binding validator as the runtime."""

from pathlib import Path

import pytest

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _resolve_projection_database_target,
)
from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.application_database import TOPOLOGY_PROFILE_INSTANCE_MAP
from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
)

pytestmark = pytest.mark.unit

# Source bytes from the OMN-21115 omnimarket activation-store migrations.
_EXPECTED = {
    "0001_grant_compaction_activations.sql": "98459e8b8dab226d460eb70f44fe0df3b797de15e97442198919d523ae664561",
    "0000_create_compaction_activations.sql": "5f997b73125f6e08688146dcac94290e6684aadb9a0fca86c83e07d3adad8a59",
}


@pytest.mark.parametrize(
    "profile", tuple(sorted(set(TOPOLOGY_PROFILE_INSTANCE_MAP.values())))
)
def test_activation_write_uses_only_the_runtime_table_grant(profile):
    topology = load_topology_profile(profile)
    (declaration,) = (
        d
        for d in LEGACY_MIGRATION_TABLE_DECLARATIONS
        if d.table.name == "compaction_activations"
    )
    _resolve_projection_database_target((declaration.table,), topology)
    principal = topology.databases["application"].principals["omninode_runtime"]
    grants = [g for g in principal.grants if "compaction_activations" in g.objects]
    assert len(grants) == 1
    assert grants[0].schema == "omninode_internal"
    assert {p.value for p in grants[0].privileges} == {"INSERT", "SELECT", "UPDATE"}
    for name, other in topology.databases["application"].principals.items():
        if name != "omninode_runtime":
            assert not any("compaction_activations" in g.objects for g in other.grants)


def test_interim_declaration_has_delivered_migrations():
    declarations = [
        d
        for d in LEGACY_MIGRATION_TABLE_DECLARATIONS
        if d.table.name == "compaction_activations"
    ]
    assert len(declarations) == 1
    declaration = declarations[0]
    assert declaration.table.access == "read_write"
    assert declaration.table.role == "compaction_activations"
    root = Path(__file__).resolve().parents[3]
    create = root / declaration.table.migration
    assert create.is_file()
    grant = create.parent / "0001_grant_compaction_activations.sql"
    # Compare the complete executable SQL, so a second GRANT widening the
    # privileges or admitting another principal cannot hide after a substring.
    statements = " ".join(
        line.split("--", 1)[0].strip() for line in grant.read_text().splitlines()
    ).split(";")
    assert [s.strip() for s in statements if s.strip()] == [
        "GRANT USAGE ON SCHEMA omninode_internal TO omninode_runtime",
        "GRANT SELECT, INSERT, UPDATE ON omninode_internal.compaction_activations TO omninode_runtime",
    ]


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("0000_create_compaction_activations.sql", "forward-only"),
        ("0001_grant_compaction_activations.sql", "expand-only"),
    ],
)
def test_vendor_ledger_binds_live_bytes_and_migration_class(name, kind):
    import hashlib

    import yaml

    root = Path(__file__).resolve().parents[3]
    node = "node_context_install_confirm_trigger_effect"
    relative = f"nodes/{node}/{name}"
    artifact = root / "docker/migrations/forward" / relative
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert digest == _EXPECTED[name]
    ledger = root / "docker/migrations/forward/_ledger/application-migrations.tsv"
    rows = [row.split("\t") for row in ledger.read_text().splitlines()]
    assert [row for row in rows if row[0] == relative] == [
        [
            relative,
            f"node:{node}",
            f"node:{node}",
            "omninode_internal",
            f"node:{node}:{name}",
            digest,
        ]
    ]
    classes = yaml.safe_load((root / "config/migration_classes.yaml").read_text())
    assert classes["migrations"][f"forward/{relative}"] == kind
    fences = yaml.safe_load(
        (root / "docker/migrations/forward/fenced-node-migrations.yaml").read_text()
    )
    assert f"node:{node}:{name}" not in {
        fence["id"] for fence in fences["fenced_node_migrations"]
    }
