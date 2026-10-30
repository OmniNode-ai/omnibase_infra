# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The table declarations that sit beside the pinned omnimarket contracts (OMN-21424).

Table grants and the SQL ownership gate derive from ``db_io.db_tables``
declarations. Those of the pinned omnimarket contracts are read by the caller;
this supplies the rest: the state_io and legacy migration manifests, and the
``db_io.db_tables`` of this repository's own node contracts, so an infra node
declares its table in its contract and not in a hand-kept manifest row.
"""

from __future__ import annotations

from pathlib import Path

from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
    STATE_IO_TABLE_DECLARATIONS,
    ContractTableDeclaration,
    load_contract_declarations,
)

_INFRA_NODES = Path(__file__).resolve().parents[1] / "nodes"


def supplemental_table_declarations() -> tuple[ContractTableDeclaration, ...]:
    """The state_io, legacy migration and in-repo node contract declarations."""
    return (
        STATE_IO_TABLE_DECLARATIONS
        + LEGACY_MIGRATION_TABLE_DECLARATIONS
        + load_contract_declarations(_INFRA_NODES)
    )


__all__ = ["supplemental_table_declarations"]
