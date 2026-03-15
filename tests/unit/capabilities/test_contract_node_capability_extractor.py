# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ContractNodeCapabilityExtractor.

Tests the YAML-based node capability extraction that reads the
``node_capabilities`` block from contract YAML files and returns
a ``ModelNodeCapabilities`` instance.

OMN-5054: Contract-driven node capability declaration.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from omnibase_infra.capabilities.contract_node_capability_extractor import (
    ContractNodeCapabilityExtractor,
)
from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)

pytestmark = [pytest.mark.unit]


@pytest.fixture
def extractor() -> ContractNodeCapabilityExtractor:
    """Provide a fresh extractor instance for each test."""
    return ContractNodeCapabilityExtractor()


# =============================================================================
# Tests: extract_from_yaml
# =============================================================================


class TestExtractFromYaml:
    """Tests for extracting node capabilities from contract YAML files."""

    def test_extracts_boolean_capabilities(
        self, extractor: ContractNodeCapabilityExtractor, tmp_path: Path
    ) -> None:
        """Should extract boolean capability flags from node_capabilities block."""
        contract = tmp_path / "contract.yaml"
        contract.write_text(
            dedent("""\
            name: "test_node"
            node_type: "EFFECT_GENERIC"
            node_capabilities:
              postgres: true
              read: true
              write: true
              transactions: true
            """)
        )

        caps = extractor.extract_from_yaml(contract)

        assert caps.postgres is True
        assert caps.read is True
        assert caps.write is True
        assert caps.transactions is True
        # Defaults for unspecified fields
        assert caps.database is False
        assert caps.routing is False
        assert caps.processing is False

    def test_returns_default_when_no_node_capabilities_block(
        self, extractor: ContractNodeCapabilityExtractor, tmp_path: Path
    ) -> None:
        """Should return all-false defaults when node_capabilities is absent."""
        contract = tmp_path / "contract.yaml"
        contract.write_text(
            dedent("""\
            name: "test_node"
            node_type: "EFFECT_GENERIC"
            """)
        )

        caps = extractor.extract_from_yaml(contract)

        assert caps.postgres is False
        assert caps.read is False
        assert caps.write is False

    def test_returns_default_for_empty_file(
        self, extractor: ContractNodeCapabilityExtractor, tmp_path: Path
    ) -> None:
        """Should return all-false defaults for an empty YAML file."""
        contract = tmp_path / "contract.yaml"
        contract.write_text("")

        caps = extractor.extract_from_yaml(contract)

        assert caps == ModelNodeCapabilities()

    def test_returns_default_for_nonexistent_file(
        self, extractor: ContractNodeCapabilityExtractor, tmp_path: Path
    ) -> None:
        """Should return all-false defaults when file does not exist."""
        contract = tmp_path / "does_not_exist.yaml"

        caps = extractor.extract_from_yaml(contract)

        assert caps == ModelNodeCapabilities()

    def test_returns_default_for_invalid_yaml(
        self, extractor: ContractNodeCapabilityExtractor, tmp_path: Path
    ) -> None:
        """Should return all-false defaults when YAML is malformed."""
        contract = tmp_path / "contract.yaml"
        contract.write_text("{{invalid yaml!!")

        caps = extractor.extract_from_yaml(contract)

        assert caps == ModelNodeCapabilities()

    def test_returns_default_when_node_capabilities_not_a_dict(
        self, extractor: ContractNodeCapabilityExtractor, tmp_path: Path
    ) -> None:
        """Should return defaults when node_capabilities is a list or scalar."""
        contract = tmp_path / "contract.yaml"
        contract.write_text(
            dedent("""\
            name: "test_node"
            node_capabilities:
              - "postgres"
              - "read"
            """)
        )

        caps = extractor.extract_from_yaml(contract)

        assert caps == ModelNodeCapabilities()

    def test_extracts_custom_capabilities_via_extra(
        self, extractor: ContractNodeCapabilityExtractor, tmp_path: Path
    ) -> None:
        """Custom capability fields should be stored in model_extra."""
        contract = tmp_path / "contract.yaml"
        contract.write_text(
            dedent("""\
            name: "test_node"
            node_capabilities:
              postgres: true
              custom_ml_inference: true
              max_tokens: 4096
            """)
        )

        caps = extractor.extract_from_yaml(contract)

        assert caps.postgres is True
        assert caps.model_extra is not None
        assert caps.model_extra["custom_ml_inference"] is True
        assert caps.model_extra["max_tokens"] == 4096

    def test_extracts_processing_capabilities(
        self, extractor: ContractNodeCapabilityExtractor, tmp_path: Path
    ) -> None:
        """Should extract processing-related capabilities."""
        contract = tmp_path / "contract.yaml"
        contract.write_text(
            dedent("""\
            name: "test_node"
            node_capabilities:
              processing: true
              batch_size: 100
              max_batch: 500
              supported_types:
                - "json"
                - "yaml"
            """)
        )

        caps = extractor.extract_from_yaml(contract)

        assert caps.processing is True
        assert caps.batch_size == 100
        assert caps.max_batch == 500
        assert caps.supported_types == ["json", "yaml"]

    def test_extracts_from_real_contract(
        self, extractor: ContractNodeCapabilityExtractor
    ) -> None:
        """Should extract from the real registration_storage_effect contract."""
        contract_path = (
            Path(__file__).parents[3]
            / "src"
            / "omnibase_infra"
            / "nodes"
            / "node_registration_storage_effect"
            / "contract.yaml"
        )
        if not contract_path.exists():
            pytest.skip("Real contract not found")

        caps = extractor.extract_from_yaml(contract_path)

        assert caps.postgres is True
        assert caps.read is True
        assert caps.write is True
        assert caps.database is True
        assert caps.transactions is True


# =============================================================================
# Tests: extract_from_dict
# =============================================================================


class TestExtractFromDict:
    """Tests for extracting node capabilities from pre-parsed YAML dicts."""

    def test_extracts_from_dict(
        self, extractor: ContractNodeCapabilityExtractor
    ) -> None:
        """Should extract capabilities from a pre-parsed dict."""
        data: dict[str, object] = {
            "name": "test_node",
            "node_capabilities": {
                "postgres": True,
                "read": True,
                "write": True,
            },
        }

        caps = extractor.extract_from_dict(data)

        assert caps.postgres is True
        assert caps.read is True
        assert caps.write is True

    def test_returns_default_when_no_block_in_dict(
        self, extractor: ContractNodeCapabilityExtractor
    ) -> None:
        """Should return defaults when node_capabilities key is absent."""
        data: dict[str, object] = {"name": "test_node"}

        caps = extractor.extract_from_dict(data)

        assert caps == ModelNodeCapabilities()

    def test_returns_default_when_block_is_not_dict(
        self, extractor: ContractNodeCapabilityExtractor
    ) -> None:
        """Should return defaults when node_capabilities is not a dict."""
        data: dict[str, object] = {
            "name": "test_node",
            "node_capabilities": ["postgres"],
        }

        caps = extractor.extract_from_dict(data)

        assert caps == ModelNodeCapabilities()


# =============================================================================
# Tests: Integration with ModelIntrospectionConfig
# =============================================================================


class TestIntrospectionConfigIntegration:
    """Tests that extracted capabilities can be passed to ModelIntrospectionConfig."""

    def test_extracted_caps_accepted_by_config(
        self, extractor: ContractNodeCapabilityExtractor, tmp_path: Path
    ) -> None:
        """Extracted capabilities should be accepted by ModelIntrospectionConfig."""
        from uuid import uuid4

        from omnibase_core.enums import EnumNodeKind
        from omnibase_infra.models.discovery.model_introspection_config import (
            ModelIntrospectionConfig,
        )

        contract = tmp_path / "contract.yaml"
        contract.write_text(
            dedent("""\
            name: "test_node"
            node_capabilities:
              postgres: true
              read: true
            """)
        )

        caps = extractor.extract_from_yaml(contract)

        config = ModelIntrospectionConfig(
            node_id=uuid4(),
            node_type=EnumNodeKind.EFFECT,
            node_name="test_node",
            declared_capabilities=caps,
        )

        assert config.declared_capabilities is not None
        assert config.declared_capabilities.postgres is True
        assert config.declared_capabilities.read is True
