# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
# Ticket: OMN-7089

"""Parametrized test ensuring every contract.yaml has a non-null, meaningful metadata.description."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

NODES_DIR = Path(__file__).resolve().parents[3] / "src" / "omnibase_infra" / "nodes"

_BAD_PREFIX = re.compile(r"^(Compute|Effect|Reducer|Orchestrator) node", re.IGNORECASE)


def _discover_contracts() -> list[Path]:
    """Return all contract.yaml files under the nodes directory."""
    return sorted(NODES_DIR.rglob("contract.yaml"))


@pytest.mark.unit
@pytest.mark.parametrize(
    "contract_path",
    _discover_contracts(),
    ids=lambda p: str(p.relative_to(NODES_DIR)),
)
def test_metadata_description_is_present_and_meaningful(
    contract_path: Path,
) -> None:
    """Each contract.yaml must have a non-null, non-empty metadata.description
    that does NOT start with a generic node-type prefix."""
    data = yaml.safe_load(contract_path.read_text())
    assert data is not None, f"Empty or unparseable YAML: {contract_path}"

    metadata = data.get("metadata")
    assert metadata is not None, (
        f"Missing 'metadata' block in {contract_path.relative_to(NODES_DIR)}"
    )

    description = metadata.get("description")
    assert description is not None, (
        f"metadata.description is null in {contract_path.relative_to(NODES_DIR)}"
    )
    assert isinstance(description, str) and description.strip(), (
        f"metadata.description is empty in {contract_path.relative_to(NODES_DIR)}"
    )
    assert not _BAD_PREFIX.match(description), (
        f"metadata.description starts with generic node-type prefix in "
        f"{contract_path.relative_to(NODES_DIR)}: {description[:80]!r}"
    )
