# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Contract parity gate for the delegation topic constants (OMN-10907).

``topic_constants.py`` carries the ``TOPIC_DELEGATION_*`` Final[str] constants
that the delegation pipeline imports. Every one of them must also appear in the
``node_delegation_orchestrator`` contract.yaml event_bus topic sets — the
contract is the source of truth and these constants are a typed mirror of it.

This gate fails if:
- a constant is added without a matching contract declaration;
- a contract topic is renamed but the constant is not updated;
- the constant count drops below the documented baseline of 13.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

from omnibase_infra.event_bus import topic_constants

_DELEGATION_CONTRACT_RELATIVE = Path("nodes/node_delegation_orchestrator/contract.yaml")
_EXPECTED_MINIMUM_CONSTANTS = 13


def _delegation_contract_path() -> Path:
    spec = importlib.util.find_spec("omnibase_infra")
    assert spec is not None and spec.origin is not None, (
        "omnibase_infra package spec not found"
    )
    package_root = Path(spec.origin).parent
    return package_root / _DELEGATION_CONTRACT_RELATIVE


def _delegation_topic_constants() -> dict[str, str]:
    return {
        name: getattr(topic_constants, name)
        for name in dir(topic_constants)
        if name.startswith("TOPIC_DELEGATION_")
        and isinstance(getattr(topic_constants, name), str)
    }


def _contract_topics() -> set[str]:
    contract_path = _delegation_contract_path()
    assert contract_path.exists(), (
        f"delegation orchestrator contract not found at {contract_path}"
    )
    contract = yaml.safe_load(contract_path.read_text())
    event_bus = contract["event_bus"]
    return set(event_bus["subscribe_topics"]) | set(event_bus["publish_topics"])


@pytest.mark.unit
def test_delegation_constant_count_meets_baseline() -> None:
    constants = _delegation_topic_constants()
    assert len(constants) >= _EXPECTED_MINIMUM_CONSTANTS, (
        f"expected >= {_EXPECTED_MINIMUM_CONSTANTS} TOPIC_DELEGATION_* constants, "
        f"found {len(constants)}: {sorted(constants)}"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "constant_name",
    sorted(_delegation_topic_constants()),
)
def test_delegation_topic_constant_appears_in_contract(constant_name: str) -> None:
    value = getattr(topic_constants, constant_name)
    contract_topics = _contract_topics()
    assert value in contract_topics, (
        f"{constant_name}={value!r} is not declared in the "
        f"node_delegation_orchestrator contract event_bus topics. "
        f"Add it to subscribe_topics or publish_topics, or remove the constant."
    )
