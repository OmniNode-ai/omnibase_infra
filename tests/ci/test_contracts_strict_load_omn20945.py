# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20945: every ticket contract in contracts/ loads through the strict loader.

The dod-verify node reads product-repo contracts through omnibase_core's
``load_yaml_mapping_no_duplicates``, so a contract with a duplicate mapping key is a
FAILED verdict even though ``yaml.safe_load`` accepts it and keeps the last value.
``contracts/OMN-17427.yaml`` carried one evidence item whose ``description`` and
``command`` were each written twice; this pins every contract to one value per key,
and shows the strict loader refuses that item as it stood before the repair.
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
OMN_17427 = CONTRACTS / "OMN-17427.yaml"
REPAIRED_COMMAND = (
    "uv run pytest tests/ci/test_runtime_boot_dev_lane_class_omn17427.py -q"
)


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
    text = OMN_17427.read_text(encoding="utf-8")
    items = load_yaml_mapping_no_duplicates(text, source=OMN_17427.name)[
        "evidence_requirements"
    ]
    assert isinstance(items, list)
    commands = [item["command"] for item in items]
    assert commands.count(REPAIRED_COMMAND) == 1
    repaired = commands.index(REPAIRED_COMMAND)
    for item in items[repaired : repaired + 2]:
        assert set(item) == {"kind", "description", "command"}

    # Join the repaired item to the next one again, as the file stood before the
    # repair: one mapping that writes description and command twice.
    lines = text.splitlines(keepends=True)
    split = 1 + next(
        index
        for index, line in enumerate(lines)
        if line.strip() == f"command: {REPAIRED_COMMAND}"
    )
    assert lines[split].strip() == "- kind: tests"
    rejoined = "".join(lines[:split] + lines[split + 1 :])
    with pytest.raises(
        DuplicateYamlMappingKeyError, match="duplicate key 'description'"
    ):
        load_yaml_mapping_no_duplicates(rejoined, source=OMN_17427.name)
