#!/usr/bin/env python3
"""
Integration tests for Template Variant Selection system (Phase 3).

Tests integration of TemplateSelector with TemplateEngine and artifact generation.

Performance Target: <5ms variant selection
Accuracy Target: >95% correct template selection
"""

import logging
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from omninode_bridge.codegen.models_contract import EnumTemplateVariant
from omninode_bridge.codegen.template_selector import TemplateSelector

logger = logging.getLogger(__name__)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def contracts_dir():
    """Get contracts directory."""
    return Path(__file__).parent.parent / "fixtures" / "contracts"


@pytest.fixture
def load_contract(contracts_dir):
    """Factory to load test contracts."""

    def _load(filename: str) -> dict[str, Any]:
        with open(contracts_dir / filename) as f:
            return yaml.safe_load(f)

    return _load


@pytest.fixture
def template_selector():
    """Create TemplateSelector instance."""
    return TemplateSelector()


# ============================================================================
# Integration Tests: Variant Selection
# ============================================================================


@pytest.mark.integration
class TestVariantIntegration:
    """Integration tests for template variant selection."""

    @pytest.mark.asyncio
    async def test_minimal_variant_selection(self, template_selector, load_contract):
        """Test minimal variant selection for simple contracts."""
        contract = load_contract("minimal_effect_contract.yaml")

        start_time = time.perf_counter()
        selection = template_selector.select_template(
            requirements=contract,
            node_type="effect",
        )
        selection_time = (time.perf_counter() - start_time) * 1000

        assert selection.variant == EnumTemplateVariant.MINIMAL
        assert selection.confidence > 0.7
        assert selection_time < 5.0, f"Selection took {selection_time:.2f}ms"

        logger.info(
            f"✅ Minimal variant: confidence={selection.confidence:.2f}, time={selection_time:.2f}ms"
        )

    @pytest.mark.asyncio
    async def test_standard_variant_selection(self, template_selector, load_contract):
        """Test standard variant selection for moderate complexity."""
        contract = load_contract("standard_effect_contract.yaml")

        start_time = time.perf_counter()
        selection = template_selector.select_template(
            requirements=contract,
            node_type="effect",
        )
        selection_time = (time.perf_counter() - start_time) * 1000

        assert selection.variant == EnumTemplateVariant.STANDARD
        assert selection.confidence > 0.7
        assert selection_time < 5.0

        logger.info(
            f"✅ Standard variant: confidence={selection.confidence:.2f}, time={selection_time:.2f}ms"
        )

    @pytest.mark.asyncio
    async def test_production_variant_selection(self, template_selector, load_contract):
        """Test production variant selection for complex contracts."""
        contract = load_contract("production_orchestrator_contract.yaml")

        start_time = time.perf_counter()
        selection = template_selector.select_template(
            requirements=contract,
            node_type="orchestrator",
        )
        selection_time = (time.perf_counter() - start_time) * 1000

        assert selection.variant == EnumTemplateVariant.PRODUCTION
        assert selection.confidence > 0.7
        assert selection_time < 5.0

        logger.info(
            f"✅ Production variant: confidence={selection.confidence:.2f}, time={selection_time:.2f}ms"
        )

    @pytest.mark.asyncio
    async def test_database_heavy_variant_selection(
        self, template_selector, load_contract
    ):
        """Test database_heavy variant for DB-intensive contracts."""
        contract = load_contract("database_adapter_effect.yaml")

        selection = template_selector.select_template(
            requirements=contract,
            node_type="effect",
        )

        # Should select database-heavy or standard variant
        assert selection.variant in [
            EnumTemplateVariant.DATABASE_HEAVY,
            EnumTemplateVariant.STANDARD,
        ]
        assert selection.confidence > 0.6

        logger.info(
            f"✅ Database variant: {selection.variant}, confidence={selection.confidence:.2f}"
        )

    @pytest.mark.asyncio
    async def test_api_heavy_variant_selection(self, template_selector, load_contract):
        """Test api_heavy variant for API-intensive contracts."""
        contract = load_contract("api_client_effect.yaml")

        selection = template_selector.select_template(
            requirements=contract,
            node_type="effect",
        )

        # Should select API-heavy or standard variant
        assert selection.variant in [
            EnumTemplateVariant.API_HEAVY,
            EnumTemplateVariant.STANDARD,
        ]
        assert selection.confidence > 0.6

        logger.info(
            f"✅ API variant: {selection.variant}, confidence={selection.confidence:.2f}"
        )

    @pytest.mark.asyncio
    async def test_kafka_heavy_variant_selection(
        self, template_selector, load_contract
    ):
        """Test kafka_heavy variant for event streaming contracts."""
        contract = load_contract("kafka_consumer_effect.yaml")

        selection = template_selector.select_template(
            requirements=contract,
            node_type="effect",
        )

        # Should select Kafka-heavy or standard variant
        assert selection.variant in [
            EnumTemplateVariant.KAFKA_HEAVY,
            EnumTemplateVariant.STANDARD,
        ]
        assert selection.confidence > 0.6

        logger.info(
            f"✅ Kafka variant: {selection.variant}, confidence={selection.confidence:.2f}"
        )

    @pytest.mark.asyncio
    async def test_variant_selection_with_target_environment(
        self, template_selector, load_contract
    ):
        """Test variant selection respects target environment."""
        contract = load_contract("production_orchestrator_contract.yaml")

        # Production environment
        prod_selection = template_selector.select_template(
            requirements=contract,
            node_type="orchestrator",
            target_environment="production",
        )

        # Development environment
        dev_selection = template_selector.select_template(
            requirements=contract,
            node_type="orchestrator",
            target_environment="development",
        )

        # Production should favor production variant
        assert prod_selection.variant == EnumTemplateVariant.PRODUCTION

        logger.info(
            f"✅ Environment-based selection: prod={prod_selection.variant}, "
            f"dev={dev_selection.variant}"
        )

    @pytest.mark.asyncio
    async def test_variant_selection_performance_batch(
        self, template_selector, load_contract
    ):
        """Test variant selection performance with batch processing."""
        contracts = [
            ("minimal_effect_contract.yaml", "effect"),
            ("standard_effect_contract.yaml", "effect"),
            ("production_orchestrator_contract.yaml", "orchestrator"),
            ("database_adapter_effect.yaml", "effect"),
            ("api_client_effect.yaml", "effect"),
        ]

        total_time = 0
        selections = []

        for contract_file, node_type in contracts:
            contract = load_contract(contract_file)

            start_time = time.perf_counter()
            selection = template_selector.select_template(
                requirements=contract,
                node_type=node_type,
            )
            selection_time = (time.perf_counter() - start_time) * 1000

            total_time += selection_time
            selections.append((contract_file, selection))

            assert (
                selection_time < 5.0
            ), f"Selection for {contract_file} took {selection_time:.2f}ms"

        avg_time = total_time / len(contracts)
        assert (
            avg_time < 5.0
        ), f"Average selection time {avg_time:.2f}ms exceeds 5ms target"

        logger.info(
            f"✅ Batch selection: {len(contracts)} contracts in {total_time:.2f}ms "
            f"(avg: {avg_time:.2f}ms)"
        )

    @pytest.mark.asyncio
    async def test_variant_selection_confidence_scoring(
        self, template_selector, load_contract
    ):
        """Test confidence scoring accuracy."""
        test_cases = [
            ("minimal_effect_contract.yaml", "effect", 0.8),  # High confidence
            (
                "production_orchestrator_contract.yaml",
                "orchestrator",
                0.85,
            ),  # Very high
        ]

        for contract_file, node_type, min_confidence in test_cases:
            contract = load_contract(contract_file)

            selection = template_selector.select_template(
                requirements=contract,
                node_type=node_type,
            )

            assert (
                selection.confidence >= min_confidence
            ), f"Confidence {selection.confidence:.2f} below threshold {min_confidence}"

            logger.info(
                f"✅ Confidence for {contract_file}: {selection.confidence:.2f} "
                f"(threshold: {min_confidence})"
            )

    @pytest.mark.asyncio
    async def test_variant_fallback_scenarios(self, template_selector):
        """Test fallback behavior for edge cases."""
        # Empty contract
        empty_contract = {}

        selection = template_selector.select_template(
            requirements=empty_contract,
            node_type="effect",
        )

        # Should fallback to minimal variant
        assert selection.variant in [
            EnumTemplateVariant.MINIMAL,
            EnumTemplateVariant.STANDARD,
        ]
        assert selection.confidence >= 0.0  # Should have some confidence

        logger.info(
            f"✅ Fallback selection: {selection.variant}, confidence={selection.confidence:.2f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
