# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for contract YAML parser [OMN-7043]."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.verification.contract_parser import (
    ModelParsedContractForVerification,
    parse_contract_for_verification,
)

# Path to the nodes directory in the source tree
NODES_DIR = Path(__file__).resolve().parents[3] / "src" / "omnibase_infra" / "nodes"


@pytest.mark.unit
class TestParseRegistrationOrchestratorContract:
    """Tests for parsing the registration orchestrator contract."""

    @pytest.fixture
    def contract(self) -> ModelParsedContractForVerification:
        contract_path = NODES_DIR / "node_registration_orchestrator" / "contract.yaml"
        return parse_contract_for_verification(contract_path)

    def test_name(self, contract: ModelParsedContractForVerification) -> None:
        assert contract.name == "node_registration_orchestrator"

    def test_node_type(self, contract: ModelParsedContractForVerification) -> None:
        assert contract.node_type == "ORCHESTRATOR_GENERIC"

    def test_subscribe_topics_count(
        self, contract: ModelParsedContractForVerification
    ) -> None:
        assert len(contract.subscribe_topics) == 7

    def test_subscribe_topics_contains_introspection(
        self, contract: ModelParsedContractForVerification
    ) -> None:
        assert "onex.evt.platform.node-introspection.v1" in contract.subscribe_topics

    def test_publish_topics_count(
        self, contract: ModelParsedContractForVerification
    ) -> None:
        assert len(contract.publish_topics) == 20

    def test_publish_topics_contains_registration_result(
        self, contract: ModelParsedContractForVerification
    ) -> None:
        assert (
            "onex.evt.platform.node-registration-result.v1" in contract.publish_topics
        )

    def test_consumed_events_populated(
        self, contract: ModelParsedContractForVerification
    ) -> None:
        assert len(contract.consumed_events) > 0
        assert "NodeIntrospectionEvent" in contract.consumed_events

    def test_published_events_populated(
        self, contract: ModelParsedContractForVerification
    ) -> None:
        assert len(contract.published_events) > 0

    def test_handler_names_populated(
        self, contract: ModelParsedContractForVerification
    ) -> None:
        assert len(contract.handler_names) > 0

    def test_fsm_states_populated(
        self, contract: ModelParsedContractForVerification
    ) -> None:
        # The orchestrator has state_decision_matrix entries
        assert len(contract.fsm_states) > 0

    def test_contract_path_set(
        self, contract: ModelParsedContractForVerification
    ) -> None:
        assert "node_registration_orchestrator" in contract.contract_path
        assert contract.contract_path.endswith("contract.yaml")


@pytest.mark.unit
class TestParseAllContracts:
    """Parameterized test that all 52+ contracts parse without error."""

    @staticmethod
    def _collect_contracts() -> list[Path]:
        """Collect all contract.yaml files under nodes/."""
        if not NODES_DIR.exists():
            return []
        return sorted(NODES_DIR.glob("*/contract.yaml"))

    @pytest.fixture(params=_collect_contracts.__func__())  # type: ignore[attr-defined]
    def contract_path(self, request: pytest.FixtureRequest) -> Path:
        return request.param

    def test_parse_without_error(self, contract_path: Path) -> None:
        result = parse_contract_for_verification(contract_path)
        assert result.name != ""
        assert result.node_type != ""

    def test_returns_structured_type(self, contract_path: Path) -> None:
        result = parse_contract_for_verification(contract_path)
        assert isinstance(result, ModelParsedContractForVerification)


@pytest.mark.unit
class TestComputeOnlyContract:
    """Test that contracts without event_bus are handled gracefully."""

    def test_no_event_bus_returns_empty_topics(self, tmp_path: Path) -> None:
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(
            "name: test_compute_node\nnode_type: COMPUTE_GENERIC\n"
        )
        result = parse_contract_for_verification(contract_file)
        assert result.subscribe_topics == ()
        assert result.publish_topics == ()
        assert result.handler_names == ()
        assert result.consumed_events == ()
        assert result.published_events == ()
        assert result.fsm_states == ()
