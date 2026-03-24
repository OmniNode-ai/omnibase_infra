# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for feature flag extraction in ContractNodeCapabilityExtractor.

OMN-5574: Extend the capability extractor to extract feature flags from
contract YAML ``feature_flags:`` blocks.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from omnibase_core.models.core.model_feature_flags import ModelFeatureFlags
from omnibase_infra.capabilities.contract_node_capability_extractor import (
    ContractNodeCapabilityExtractor,
)

pytestmark = [pytest.mark.unit]


@pytest.fixture
def extractor() -> ContractNodeCapabilityExtractor:
    """Provide a fresh extractor instance for each test."""
    return ContractNodeCapabilityExtractor()


class TestExtractFeatureFlagsFromYaml:
    """Tests for extract_feature_flags_from_yaml."""

    def test_extract_feature_flags_from_yaml(
        self, extractor: ContractNodeCapabilityExtractor, tmp_path: Path
    ) -> None:
        """Valid YAML with feature_flags block extracts flags correctly."""
        contract = tmp_path / "contract.yaml"
        contract.write_text(
            dedent("""\
            name: "test_node"
            feature_flags:
              - name: enable_caching
                description: "Enable response caching"
                default_value: true
                env_var: ENABLE_CACHING
                category: infrastructure
                owner: infra-team
              - name: debug_mode
                description: "Enable debug logging"
                default_value: false
                category: observability
            """)
        )

        result = extractor.extract_feature_flags_from_yaml(contract)

        assert isinstance(result, ModelFeatureFlags)
        assert result.is_enabled("enable_caching") is True
        assert result.is_enabled("debug_mode") is False
        assert result.get_flag_count() == 2
        assert extractor.last_validation_errors == []

    def test_missing_block_returns_empty(
        self, extractor: ContractNodeCapabilityExtractor, tmp_path: Path
    ) -> None:
        """YAML without feature_flags block returns empty ModelFeatureFlags."""
        contract = tmp_path / "contract.yaml"
        contract.write_text(
            dedent("""\
            name: "test_node"
            node_type: "EFFECT_GENERIC"
            """)
        )

        result = extractor.extract_feature_flags_from_yaml(contract)

        assert isinstance(result, ModelFeatureFlags)
        assert result.get_flag_count() == 0
        assert extractor.last_validation_errors == []

    def test_invalid_block_returns_empty_with_validation_error(
        self, extractor: ContractNodeCapabilityExtractor, tmp_path: Path
    ) -> None:
        """feature_flags that is not a list returns empty + validation errors."""
        contract = tmp_path / "contract.yaml"
        contract.write_text(
            dedent("""\
            name: "test_node"
            feature_flags: not_a_list
            """)
        )

        result = extractor.extract_feature_flags_from_yaml(contract)

        assert isinstance(result, ModelFeatureFlags)
        assert result.get_flag_count() == 0
        assert len(extractor.last_validation_errors) > 0

    def test_invalid_block_populates_validation_errors(
        self, extractor: ContractNodeCapabilityExtractor, tmp_path: Path
    ) -> None:
        """Malformed entries populate last_validation_errors with structured messages."""
        contract = tmp_path / "contract.yaml"
        contract.write_text(
            dedent("""\
            name: "test_node"
            feature_flags:
              - not_a_mapping
              - name: valid_flag
                default_value: true
            """)
        )

        result = extractor.extract_feature_flags_from_yaml(contract)

        assert isinstance(result, ModelFeatureFlags)
        assert result.get_flag_count() == 0
        assert len(extractor.last_validation_errors) > 0

    def test_nonexistent_file_returns_empty(
        self, extractor: ContractNodeCapabilityExtractor, tmp_path: Path
    ) -> None:
        """Nonexistent file returns empty ModelFeatureFlags."""
        contract = tmp_path / "does_not_exist.yaml"

        result = extractor.extract_feature_flags_from_yaml(contract)

        assert isinstance(result, ModelFeatureFlags)
        assert result.get_flag_count() == 0


class TestExtractFeatureFlagsFromDict:
    """Tests for extract_feature_flags_from_dict."""

    def test_extract_feature_flags_from_dict(
        self, extractor: ContractNodeCapabilityExtractor
    ) -> None:
        """Dict input with valid feature_flags list extracts correctly."""
        data: dict[str, object] = {
            "name": "test_node",
            "feature_flags": [
                {
                    "name": "enable_caching",
                    "description": "Enable response caching",
                    "default_value": True,
                    "env_var": "ENABLE_CACHING",
                    "category": "infrastructure",
                },
                {
                    "name": "debug_mode",
                    "default_value": False,
                },
            ],
        }

        result = extractor.extract_feature_flags_from_dict(data)

        assert isinstance(result, ModelFeatureFlags)
        assert result.is_enabled("enable_caching") is True
        assert result.is_enabled("debug_mode") is False
        assert result.get_flag_count() == 2
        assert extractor.last_validation_errors == []

    def test_missing_block_returns_empty(
        self, extractor: ContractNodeCapabilityExtractor
    ) -> None:
        """Dict without feature_flags key returns empty ModelFeatureFlags."""
        data: dict[str, object] = {"name": "test_node"}

        result = extractor.extract_feature_flags_from_dict(data)

        assert isinstance(result, ModelFeatureFlags)
        assert result.get_flag_count() == 0

    def test_invalid_block_returns_empty_with_errors(
        self, extractor: ContractNodeCapabilityExtractor
    ) -> None:
        """Dict with non-list feature_flags returns empty + errors."""
        data: dict[str, object] = {
            "name": "test_node",
            "feature_flags": "not_a_list",
        }

        result = extractor.extract_feature_flags_from_dict(data)

        assert isinstance(result, ModelFeatureFlags)
        assert result.get_flag_count() == 0
        assert len(extractor.last_validation_errors) > 0
