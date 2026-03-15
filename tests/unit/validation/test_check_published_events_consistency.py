# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the published_events consistency checker.

Validates that check_contract_consistency() correctly identifies when
published_events topics are missing from event_bus.publish_topics.

OMN-4885 / OMN-4880
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.validation.check_published_events_consistency import (
    ContractConsistencyError,
    check_contract_consistency,
)


@pytest.mark.unit
def test_clean_contract_passes() -> None:
    """A contract where published_events topics are all in event_bus.publish_topics passes."""
    contract = {
        "published_events": [
            {
                "event_type": "NodeRegistrationAccepted",
                "topic": "onex.evt.platform.node-registration-accepted.v1",
            },
        ],
        "event_bus": {
            "publish_topics": ["onex.evt.platform.node-registration-accepted.v1"],
        },
    }
    # No exception expected
    check_contract_consistency(contract, contract_path="test/contract.yaml")


@pytest.mark.unit
def test_missing_from_event_bus_publish_topics_raises() -> None:
    """A topic in published_events but not in event_bus.publish_topics raises ContractConsistencyError."""
    contract = {
        "published_events": [
            {
                "event_type": "NodeRegistrationAccepted",
                "topic": "onex.evt.platform.node-registration-accepted.v1",
            },
        ],
        "event_bus": {
            "publish_topics": [],  # missing the topic!
        },
    }
    with pytest.raises(ContractConsistencyError, match="node-registration-accepted"):
        check_contract_consistency(contract, contract_path="test/contract.yaml")


@pytest.mark.unit
def test_no_published_events_key_passes() -> None:
    """A contract with no published_events key passes (nothing to check)."""
    contract: dict[str, object] = {
        "event_bus": {"publish_topics": ["onex.evt.platform.other.v1"]}
    }
    check_contract_consistency(contract, contract_path="test/contract.yaml")  # no error


@pytest.mark.unit
def test_malformed_contract_raises_clearly() -> None:
    """Non-dict YAML (e.g. a bare string) raises ContractConsistencyError, not AttributeError."""
    with pytest.raises(ContractConsistencyError, match="not a dict"):
        check_contract_consistency("not a dict", contract_path="bad/contract.yaml")  # type: ignore[arg-type]


@pytest.mark.unit
def test_real_registration_orchestrator_contract_is_consistent() -> None:
    """The actual contract.yaml must have published_events topics in event_bus.publish_topics."""
    contract_path = (
        Path(__file__).parents[3]
        / "src/omnibase_infra/nodes/node_registration_orchestrator/contract.yaml"
    )
    contract_data = yaml.safe_load(contract_path.read_text())
    check_contract_consistency(contract_data, contract_path=str(contract_path))
