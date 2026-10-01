# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The pinned omnimarket contracts declare ``delegation_dispositions`` (OMN-17292).

The supplemental ``legacy_migration:delegation_dispositions`` declaration was
deleted because the advanced pin declares the relation itself. This asserts the
derivation still carries it from the pinned contracts alone, so removing the
bridge cannot silently drop the grant.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
    load_contract_declarations,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROOF = _REPO_ROOT / ".proof-dependencies"
_SUFFIX = ("src", "omnimarket", "nodes")
_PIN_ROOT = next(
    (
        root
        for root in (
            _PROOF.joinpath("omnimarket-pin", *_SUFFIX),
            _PROOF.joinpath("omnimarket", *_SUFFIX),
        )
        if root.is_dir()
    ),
    None,
)

_NEEDS_PIN = pytest.mark.skipif(
    _PIN_ROOT is None,
    reason="requires the pinned omnimarket checkout under .proof-dependencies",
)


@_NEEDS_PIN
def test_pinned_contracts_declare_delegation_dispositions() -> None:
    assert _PIN_ROOT is not None
    pinned = {d.table.name for d in load_contract_declarations(_PIN_ROOT)}
    assert "delegation_dispositions" in pinned


def test_no_supplemental_bridge_remains_for_delegation_dispositions() -> None:
    bridged = {d.table.name for d in LEGACY_MIGRATION_TABLE_DECLARATIONS}
    assert "delegation_dispositions" not in bridged
