# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20945: every ticket contract in contracts/ loads through the strict loader.

The dod-verify node reads product-repo contracts through omnibase_core's
``load_yaml_mapping_no_duplicates``, so a contract with a duplicate mapping key is a
FAILED verdict even though ``yaml.safe_load`` accepts it and keeps the last value.
``contracts/OMN-17427.yaml`` carried one evidence item whose ``description`` and
``command`` were each written twice; this pins every contract to one value per key.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_core.errors.error_duplicate_yaml_mapping_key import (
    DuplicateYamlMappingKeyError,
)
from omnibase_core.utils.util_safe_yaml_loader import load_yaml_mapping_no_duplicates

pytestmark = pytest.mark.unit

CONTRACTS = Path(__file__).resolve().parents[2] / "contracts"


def test_every_contract_loads_without_a_duplicate_mapping_key() -> None:
    paths = sorted(CONTRACTS.glob("*.yaml"))
    assert paths, f"no contract under {CONTRACTS}"
    refused: list[str] = []
    for path in paths:
        try:
            load_yaml_mapping_no_duplicates(
                path.read_text(encoding="utf-8"), source=path.name
            )
        except DuplicateYamlMappingKeyError as error:
            refused.append(str(error))
    assert not refused, "contracts with a duplicate mapping key: " + "; ".join(refused)


def test_strict_loader_refuses_a_duplicate_key() -> None:
    with pytest.raises(DuplicateYamlMappingKeyError):
        load_yaml_mapping_no_duplicates(
            "item:\n  description: first\n  description: second\n",
            source="positive-control.yaml",
        )
