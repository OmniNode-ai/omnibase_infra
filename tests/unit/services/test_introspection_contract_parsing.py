# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression test: every contract.yaml must parse through the introspection pipeline.

OMN-6405: ModelContractBase uses extra="forbid" but production contract YAMLs
contain sections not declared in the Pydantic model. This caused
_try_load_contract() to silently return None, leaving metadata.description=None
in all Kafka introspection events. The omnidash node registry showed all nodes
as unnamed COMPUTE types with no description.

This test loads every contract.yaml through ServiceNodeIntrospection.from_contract_dir()
and asserts the description is non-None. If this test fails, it means a contract YAML
has diverged from what the introspection parsing pipeline can handle.
"""

from __future__ import annotations

from pathlib import Path

import pytest

NODES_DIR = Path(__file__).resolve().parents[3] / "src" / "omnibase_infra" / "nodes"


def _discover_contract_dirs() -> list[Path]:
    """Find all node directories containing a contract.yaml."""
    if not NODES_DIR.is_dir():
        return []
    return sorted(
        d for d in NODES_DIR.iterdir() if d.is_dir() and (d / "contract.yaml").exists()
    )


CONTRACT_DIRS = _discover_contract_dirs()


@pytest.mark.unit
@pytest.mark.parametrize(
    "contract_dir",
    CONTRACT_DIRS,
    ids=[d.name for d in CONTRACT_DIRS],
)
def test_introspection_extracts_description(contract_dir: Path) -> None:
    """ServiceNodeIntrospection.from_contract_dir() must produce a non-None description.

    If this fails, either:
    1. The contract.yaml is missing a 'description' field, or
    2. The parsing pipeline silently dropped the description (the OMN-6405 bug class)

    Fix: ensure _try_load_contract + raw YAML fallback both extract description.
    """
    from omnibase_infra.services.service_node_introspection import (
        ServiceNodeIntrospection,
    )

    svc = ServiceNodeIntrospection.from_contract_dir(
        contracts_dir=contract_dir,
        event_bus=None,
        node_name=f"test-{contract_dir.name}",
    )

    assert svc._description_override is not None or (
        svc._introspection_contract is not None
        and getattr(svc._introspection_contract, "description", None) is not None
    ), (
        f"Contract at {contract_dir}/contract.yaml has a description field "
        f"but ServiceNodeIntrospection could not extract it. "
        f"This is the OMN-6405 bug class — introspection events will have "
        f"metadata.description=None on the omnidash registry."
    )
