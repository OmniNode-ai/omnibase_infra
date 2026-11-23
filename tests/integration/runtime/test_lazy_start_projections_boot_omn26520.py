# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The real discovered manifest keeps every declared projection in the boot set (OMN-26520)."""

from __future__ import annotations

import pytest

from omnibase_core.models.contracts.subcontracts.model_db_ownership_subcontract import (
    ModelDbOwnershipSubcontract,
)
from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_infra.handlers.lazy_start_boot_split import split_boot_manifest
from omnibase_infra.models.lazy_start import ModelLazyStartPolicy
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts
from omnibase_infra.runtime.health.projection_liveness import (
    select_projection_contracts,
)

pytestmark = pytest.mark.integration


def test_the_real_manifest_split_keeps_every_contract_and_boots_a_projection() -> None:
    manifest = discover_contracts()
    policy = ModelLazyStartPolicy(ingress_contract="node_lazy_start_ingress")
    base_boot, base_dormant = split_boot_manifest(manifest, policy)
    assert len(base_boot.contracts) + len(base_dormant) == len(manifest.contracts)
    template = base_dormant[0]
    projection = template.model_copy(
        update={
            "name": "projection_omn26520_probe",
            "db_io": ModelDbOwnershipSubcontract(
                db_tables=[
                    ModelDbTableDeclaration(
                        name="omn26520_probe_rows",
                        database_ref="application",
                        schema="omninode_internal",
                        migration="0001_init.sql",
                        access="read_write",
                        role="projection_target",
                    )
                ]
            ),
        }
    )
    with_projection = manifest.model_copy(
        update={"contracts": (*manifest.contracts, projection)}
    )
    declared = {ref.name for ref in select_projection_contracts(with_projection)}
    boot, dormant = split_boot_manifest(with_projection, policy)
    assert "projection_omn26520_probe" in declared
    assert declared <= {c.name for c in boot.contracts}
    assert not declared & {c.name for c in dormant}
