# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ServiceNodeIntrospection (OMN-5609).

Validates that:
- from_kernel_config creates a properly initialized service
- from_contract_dir loads contract and populates metadata/event_bus
- Introspection events include metadata.description from contract
- Introspection events include event_bus topics from contract
- Graceful fallback when no contract is available
"""

from pathlib import Path
from uuid import UUID

import pytest
import yaml

from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_infra.services.service_node_introspection import (
    ServiceNodeIntrospection,
    _map_node_type,
    _parse_event_bus_subcontract,
    _try_load_contract,
)

# Deterministic UUID for reproducible tests
TEST_NODE_ID = UUID("00000000-0000-0000-0000-000000005609")

# Minimal valid ONEX contract YAML (as it appears in contract.yaml files)
MINIMAL_CONTRACT_YAML: dict[str, object] = {
    "name": "test_node",
    "contract_version": {"major": 1, "minor": 2, "patch": 3},
    "node_type": "ORCHESTRATOR_GENERIC",
    "description": "Test orchestrator for introspection",
    "input_model": "omnibase_core.models.events.ModelEventEnvelope",
    "output_model": "omnibase_core.models.events.ModelEventEnvelope",
    "performance": {
        "single_operation_max_ms": 5000,
    },
    "event_bus": {
        "version": {"major": 1, "minor": 0, "patch": 0},
        "subscribe_topics": [
            "onex.evt.platform.node-introspection.v1",
        ],
        "publish_topics": [
            "onex.evt.platform.registration-completed.v1",
        ],
    },
}


def _make_contract_and_subcontract() -> tuple:
    """Parse the minimal contract YAML into contract + event_bus subcontract."""
    from omnibase_core.models.contracts.model_contract_orchestrator import (
        ModelContractOrchestrator,
    )
    from omnibase_core.models.contracts.subcontracts.model_event_bus_subcontract import (
        ModelEventBusSubcontract,
    )

    # Strip event_bus before parsing contract (extra="forbid")
    filtered = {k: v for k, v in MINIMAL_CONTRACT_YAML.items() if k != "event_bus"}
    contract = ModelContractOrchestrator(**filtered)

    eb_data = MINIMAL_CONTRACT_YAML["event_bus"]
    assert isinstance(eb_data, dict)
    event_bus_sub = ModelEventBusSubcontract(**eb_data)

    return contract, event_bus_sub


@pytest.mark.unit
class TestServiceNodeIntrospectionInit:
    """Test initialization paths."""

    def test_from_kernel_config_no_contract(self) -> None:
        """Service initializes without contract, metadata will be empty."""
        service = ServiceNodeIntrospection.from_kernel_config(
            event_bus=None,
            node_name="test-service",
            node_id=TEST_NODE_ID,
            description="fallback description",
        )
        assert service is not None
        assert service._introspection_node_id == TEST_NODE_ID
        assert service._introspection_node_name == "test-service"

    def test_from_kernel_config_with_contract(self) -> None:
        """Service initializes with contract for full metadata."""
        contract, event_bus_sub = _make_contract_and_subcontract()

        service = ServiceNodeIntrospection.from_kernel_config(
            event_bus=None,
            node_name="test-service",
            node_id=TEST_NODE_ID,
            contract=contract,
            event_bus_subcontract=event_bus_sub,
        )
        assert service is not None
        assert service._introspection_contract is not None
        # The wrapper should proxy description from the real contract
        assert getattr(service._introspection_contract, "description", None) == (
            "Test orchestrator for introspection"
        )
        # And have event_bus subcontract attached
        eb = getattr(service._introspection_contract, "event_bus", None)
        assert eb is not None
        assert len(eb.subscribe_topics) > 0


@pytest.mark.unit
class TestServiceNodeIntrospectionContractDir:
    """Test from_contract_dir factory."""

    def test_loads_contract_from_dir(self, tmp_path: Path) -> None:
        """Loads contract.yaml from directory."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(yaml.dump(MINIMAL_CONTRACT_YAML))

        service = ServiceNodeIntrospection.from_contract_dir(
            contracts_dir=tmp_path,
            event_bus=None,
            node_name="test-service",
            node_id=TEST_NODE_ID,
        )
        assert service._introspection_contract is not None
        assert getattr(service._introspection_contract, "name", None) == "test_node"

    def test_node_name_from_contract(self, tmp_path: Path) -> None:
        """OMN-6405: node_name is populated from contract name, not caller arg."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(yaml.dump(MINIMAL_CONTRACT_YAML))

        service = ServiceNodeIntrospection.from_contract_dir(
            contracts_dir=tmp_path,
            event_bus=None,
            node_name="runtime_config",  # caller passes generic config name
            node_id=TEST_NODE_ID,
        )
        # Should use contract name, not the caller-provided "runtime_config"
        assert service._introspection_node_name == "test_node"

    @pytest.mark.asyncio
    async def test_introspection_event_node_name_from_contract(
        self, tmp_path: Path
    ) -> None:
        """OMN-6405: introspection event carries contract node_name, not None."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(yaml.dump(MINIMAL_CONTRACT_YAML))

        service = ServiceNodeIntrospection.from_contract_dir(
            contracts_dir=tmp_path,
            event_bus=None,
            node_name="runtime_config",
            node_id=TEST_NODE_ID,
        )
        event = await service.get_introspection_data()
        assert event.node_name is not None
        assert event.node_name == "test_node"

    def test_node_name_from_raw_yaml_fallback(self, tmp_path: Path) -> None:
        """OMN-6405: node_name extracted from raw YAML when Pydantic parsing fails."""
        # Write a contract with extra fields that cause ModelContractBase to reject it
        raw_yaml = {
            "name": "my_custom_node",
            "contract_version": {"major": 1, "minor": 0, "patch": 0},
            "node_type": "EFFECT_GENERIC",
            "description": "Test effect node",
            "input_model": "omnibase_core.models.events.ModelEventEnvelope",
            "output_model": "omnibase_core.models.events.ModelEventEnvelope",
            "performance": {"single_operation_max_ms": 5000},
            "unknown_field_that_breaks_pydantic": True,
        }
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(yaml.dump(raw_yaml))

        service = ServiceNodeIntrospection.from_contract_dir(
            contracts_dir=tmp_path,
            event_bus=None,
            node_name="runtime_config",
            node_id=TEST_NODE_ID,
        )
        # Contract parsing fails, but raw YAML fallback should extract the name
        assert service._introspection_node_name == "my_custom_node"

    def test_graceful_fallback_no_contract(self, tmp_path: Path) -> None:
        """Falls back gracefully when no contract.yaml exists."""
        service = ServiceNodeIntrospection.from_contract_dir(
            contracts_dir=tmp_path,
            event_bus=None,
            node_name="test-service",
            node_id=TEST_NODE_ID,
        )
        assert service is not None
        assert service._introspection_contract is None
        # Without contract, caller-provided node_name is used
        assert service._introspection_node_name == "test-service"


@pytest.mark.unit
class TestIntrospectionDataPopulation:
    """Test that introspection data includes contract fields."""

    @pytest.mark.asyncio
    async def test_metadata_description_from_contract(self) -> None:
        """metadata.description is populated from contract.description."""
        contract, event_bus_sub = _make_contract_and_subcontract()

        service = ServiceNodeIntrospection.from_kernel_config(
            event_bus=None,
            node_name="test-service",
            node_id=TEST_NODE_ID,
            contract=contract,
            event_bus_subcontract=event_bus_sub,
        )

        event = await service.get_introspection_data()

        assert event.metadata is not None
        assert event.metadata.description == "Test orchestrator for introspection"

    @pytest.mark.asyncio
    async def test_event_bus_populated_from_contract(self) -> None:
        """event_bus fields populated from contract event_bus subcontract."""
        contract, event_bus_sub = _make_contract_and_subcontract()

        service = ServiceNodeIntrospection.from_kernel_config(
            event_bus=None,
            node_name="test-service",
            node_id=TEST_NODE_ID,
            contract=contract,
            event_bus_subcontract=event_bus_sub,
        )

        event = await service.get_introspection_data()

        assert event.event_bus is not None
        assert len(event.event_bus.subscribe_topics) > 0
        assert len(event.event_bus.publish_topics) > 0
        sub_topics = event.event_bus.subscribe_topic_strings
        pub_topics = event.event_bus.publish_topic_strings
        assert any("node-introspection" in t for t in sub_topics)
        assert any("registration-completed" in t for t in pub_topics)

    @pytest.mark.asyncio
    async def test_contract_capabilities_populated(self) -> None:
        """contract_capabilities extracted when contract is provided."""
        contract, event_bus_sub = _make_contract_and_subcontract()

        service = ServiceNodeIntrospection.from_kernel_config(
            event_bus=None,
            node_name="test-service",
            node_id=TEST_NODE_ID,
            contract=contract,
            event_bus_subcontract=event_bus_sub,
        )

        event = await service.get_introspection_data()

        assert event.contract_capabilities is not None
        assert event.contract_capabilities.contract_type is not None

    @pytest.mark.asyncio
    async def test_node_type_preserved(self) -> None:
        """node_type is correctly set, not defaulting to COMPUTE."""
        contract, event_bus_sub = _make_contract_and_subcontract()

        service = ServiceNodeIntrospection.from_kernel_config(
            event_bus=None,
            node_name="test-service",
            node_id=TEST_NODE_ID,
            node_type=EnumNodeKind.ORCHESTRATOR,
            contract=contract,
            event_bus_subcontract=event_bus_sub,
        )

        event = await service.get_introspection_data()

        assert event.node_type == EnumNodeKind.ORCHESTRATOR

    @pytest.mark.asyncio
    async def test_no_contract_still_produces_event(self) -> None:
        """Introspection event is produced even without contract."""
        service = ServiceNodeIntrospection.from_kernel_config(
            event_bus=None,
            node_name="test-service",
            node_id=TEST_NODE_ID,
        )

        event = await service.get_introspection_data()

        assert event.node_id == TEST_NODE_ID
        assert event.event_bus is None
        assert event.metadata.description is None


@pytest.mark.unit
class TestMapNodeType:
    """Test _map_node_type helper."""

    def test_effect_generic(self) -> None:
        assert _map_node_type("EFFECT_GENERIC") == EnumNodeKind.EFFECT

    def test_compute_generic(self) -> None:
        assert _map_node_type("COMPUTE_GENERIC") == EnumNodeKind.COMPUTE

    def test_reducer_generic(self) -> None:
        assert _map_node_type("REDUCER_GENERIC") == EnumNodeKind.REDUCER

    def test_orchestrator_generic(self) -> None:
        assert _map_node_type("ORCHESTRATOR_GENERIC") == EnumNodeKind.ORCHESTRATOR

    def test_unknown_defaults_to_effect(self) -> None:
        assert _map_node_type("UNKNOWN_TYPE") == EnumNodeKind.EFFECT


@pytest.mark.unit
class TestTryLoadContract:
    """Test _try_load_contract helper."""

    def test_loads_contract_and_event_bus(self, tmp_path: Path) -> None:
        """Loads contract.yaml and extracts event_bus subcontract."""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(yaml.dump(MINIMAL_CONTRACT_YAML))

        contract, event_bus_sub = _try_load_contract(tmp_path)
        assert contract is not None
        assert contract.name == "test_node"
        assert event_bus_sub is not None
        assert len(event_bus_sub.subscribe_topics) > 0

    def test_returns_none_for_empty_dir(self, tmp_path: Path) -> None:
        """Returns None tuple when no contract exists."""
        contract, event_bus_sub = _try_load_contract(tmp_path)
        assert contract is None
        assert event_bus_sub is None

    def test_searches_subdirectories(self, tmp_path: Path) -> None:
        """Finds contract.yaml in subdirectories."""
        subdir = tmp_path / "my_node"
        subdir.mkdir()
        contract_file = subdir / "contract.yaml"
        contract_file.write_text(yaml.dump(MINIMAL_CONTRACT_YAML))

        contract, _event_bus_sub = _try_load_contract(tmp_path)
        assert contract is not None
        assert contract.name == "test_node"


@pytest.mark.unit
class TestParseEventBusSubcontract:
    """Test _parse_event_bus_subcontract helper."""

    def test_parses_valid_section(self) -> None:
        """Parses valid event_bus dict."""
        raw: dict[str, object] = {
            "event_bus": {
                "version": {"major": 1, "minor": 0, "patch": 0},
                "subscribe_topics": ["onex.evt.platform.test.v1"],
            }
        }
        result = _parse_event_bus_subcontract(raw)
        assert result is not None
        assert result.subscribe_topics == ["onex.evt.platform.test.v1"]

    def test_returns_none_without_section(self) -> None:
        """Returns None when event_bus is absent."""
        result = _parse_event_bus_subcontract({})
        assert result is None
