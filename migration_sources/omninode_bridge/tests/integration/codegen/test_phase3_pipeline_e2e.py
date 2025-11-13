#!/usr/bin/env python3
"""
End-to-End Integration Tests for Phase 3 Enhanced Pipeline.

Tests complete pipeline flow with all Phase 3 components:
- Template variant selection
- Pattern matching
- Mixin recommendations
- Contract processing
- Enhanced LLM context

Performance Targets:
- Pipeline execution: <20s (including LLM)
- Template selection: <5ms
- Pattern matching: <10ms
- Mixin recommendation: <200ms
- Context building: <50ms

Quality Targets:
- >95% correct template selection
- >90% pattern match relevance
- >90% mixin recommendation accuracy
- >90% first-pass success rate
"""

import logging
import time

import pytest

from omninode_bridge.codegen.context_builder import EnhancedContextBuilder
from omninode_bridge.codegen.mixins.mixin_recommender import MixinRecommender
from omninode_bridge.codegen.mixins.requirements_analyzer import RequirementsAnalyzer
from omninode_bridge.codegen.models_contract import EnumTemplateVariant
from omninode_bridge.codegen.pattern_library import ProductionPatternLibrary
from omninode_bridge.codegen.pipeline import EnhancedCodeGenerationPipeline
from omninode_bridge.codegen.template_selector import TemplateSelector

logger = logging.getLogger(__name__)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def template_selector():
    """Create TemplateSelector instance."""
    return TemplateSelector()


@pytest.fixture
def pattern_library():
    """Create ProductionPatternLibrary instance."""
    return ProductionPatternLibrary()


@pytest.fixture
def requirements_analyzer():
    """Create RequirementsAnalyzer instance."""
    return RequirementsAnalyzer()


@pytest.fixture
def mixin_recommender():
    """Create MixinRecommender instance."""
    return MixinRecommender()


@pytest.fixture
def context_builder():
    """Create EnhancedContextBuilder instance."""
    return EnhancedContextBuilder()


@pytest.fixture
def enhanced_pipeline(tmp_path):
    """Create EnhancedCodeGenerationPipeline instance."""
    return EnhancedCodeGenerationPipeline(
        template_dir=None,
        enable_llm=False,  # Disable LLM for tests
        enable_validation=False,  # Disable validation for speed
    )


@pytest.fixture
def simple_contract():
    """Create simple contract for testing."""
    from dataclasses import dataclass, field

    @dataclass
    class Metadata:
        name: str = "simple_service"
        description: str = "Simple test service"

    @dataclass
    class Operation:
        name: str = "fetch_data"
        description: str = "Fetch data from source"

    @dataclass
    class Contract:
        metadata: Metadata = field(default_factory=Metadata)
        node_type: str = "effect"
        domain: str = "database"
        operations: list[str] = field(default_factory=lambda: ["read"])
        io_operations: list[Operation] = field(default_factory=lambda: [Operation()])
        dependencies: dict = field(default_factory=lambda: {"asyncpg": "^0.29.0"})
        subcontracts: list = field(default_factory=list)

    return Contract()


@pytest.fixture
def production_contract():
    """Create production-grade contract for testing."""
    from dataclasses import dataclass, field

    @dataclass
    class Metadata:
        name: str = "production_service"
        description: str = "Production-grade service with full observability"

    @dataclass
    class GenerationConfig:
        quality_level: str = "production"
        enable_llm: bool = False

    @dataclass
    class Operation:
        name: str
        description: str

    @dataclass
    class Contract:
        metadata: Metadata = field(default_factory=Metadata)
        node_type: str = "effect"
        domain: str = "database"
        operations: list[str] = field(
            default_factory=lambda: [
                "create",
                "read",
                "update",
                "delete",
                "batch_create",
            ]
        )
        io_operations: list[Operation] = field(
            default_factory=lambda: [
                Operation("create", "Create new record"),
                Operation("read", "Read existing record"),
                Operation("update", "Update record"),
                Operation("delete", "Delete record"),
                Operation("batch_create", "Batch create records"),
            ]
        )
        dependencies: dict = field(
            default_factory=lambda: {
                "asyncpg": "^0.29.0",
                "aiokafka": "^0.10.0",
                "prometheus-client": "^0.19.0",
            }
        )
        generation: GenerationConfig = field(default_factory=GenerationConfig)
        subcontracts: list = field(default_factory=list)

    return Contract()


# ============================================================================
# Component Tests
# ============================================================================


@pytest.mark.integration
class TestTemplateSelection:
    """Test template variant selection."""

    def test_minimal_template_selection(self, template_selector, simple_contract):
        """Test selection of minimal template."""
        # Act
        selection = template_selector.select_template(
            requirements=simple_contract,
            node_type="effect",
            target_environment="development",
        )

        # Assert
        assert selection.variant in (
            EnumTemplateVariant.MINIMAL,
            EnumTemplateVariant.STANDARD,
        )
        assert selection.confidence >= 0.7
        assert selection.selection_time_ms < 5  # <5ms target

    def test_production_template_selection(
        self, template_selector, production_contract
    ):
        """Test selection of production template."""
        # Act
        selection = template_selector.select_template(
            requirements=production_contract,
            node_type="effect",
            target_environment="production",
        )

        # Assert
        assert selection.variant == EnumTemplateVariant.PRODUCTION
        assert selection.confidence >= 0.9
        assert len(selection.patterns) >= 3
        assert "lifecycle" in selection.patterns
        assert "health_checks" in selection.patterns
        assert selection.selection_time_ms < 5  # <5ms target

    def test_standard_template_selection(self, template_selector, simple_contract):
        """Test selection of standard template."""
        # Modify contract to be moderate complexity
        simple_contract.operations = ["read", "write", "update"]
        simple_contract.dependencies = {
            "asyncpg": "^0.29.0",
            "aiokafka": "^0.10.0",
        }

        # Act
        selection = template_selector.select_template(
            requirements=simple_contract,
            node_type="effect",
            target_environment="staging",
        )

        # Assert
        assert selection.variant == EnumTemplateVariant.STANDARD
        assert selection.confidence >= 0.8
        assert selection.selection_time_ms < 5  # <5ms target


@pytest.mark.integration
class TestPatternMatching:
    """Test pattern matching and discovery."""

    def test_database_pattern_matching(self, pattern_library):
        """Test pattern matching for database operations."""
        # Act
        matches = pattern_library.find_matching_patterns(
            operation_type="database",
            features={"health_checks", "metrics"},
            node_type="effect",
            min_relevance=0.5,
        )

        # Assert
        assert len(matches) >= 2
        assert any("health" in m.pattern_info.name for m in matches)
        assert all(m.relevance_score >= 0.5 for m in matches)
        # Check performance (pattern search should be fast)

    def test_kafka_pattern_matching(self, pattern_library):
        """Test pattern matching for Kafka operations."""
        # Act
        matches = pattern_library.find_matching_patterns(
            operation_type="kafka",
            features={"event_publishing", "kafka_health"},
            node_type="effect",
            min_relevance=0.5,
        )

        # Assert
        assert len(matches) >= 1
        assert any("event" in m.pattern_info.name.lower() for m in matches)
        assert any("kafka" in m.pattern_info.name.lower() for m in matches)

    def test_production_pattern_matching(self, pattern_library):
        """Test pattern matching for production requirements."""
        # Act
        matches = pattern_library.find_matching_patterns(
            operation_type="api",
            features={
                "lifecycle",
                "health_checks",
                "metrics",
                "event_publishing",
                "consul_integration",
            },
            node_type="effect",
            min_relevance=0.3,
        )

        # Assert
        assert len(matches) >= 4
        pattern_names = [m.pattern_info.name for m in matches]
        assert "lifecycle" in pattern_names
        assert "health_checks" in pattern_names
        assert "metrics" in pattern_names


@pytest.mark.integration
class TestMixinRecommendation:
    """Test mixin recommendation system."""

    def test_simple_mixin_recommendation(
        self, requirements_analyzer, mixin_recommender, simple_contract
    ):
        """Test mixin recommendation for simple contract."""
        # Analyze requirements
        analysis = requirements_analyzer.analyze(simple_contract)

        # Get recommendations
        start_time = time.perf_counter()
        recommendations = mixin_recommender.recommend_mixins(analysis, top_k=5)
        recommendation_time_ms = (time.perf_counter() - start_time) * 1000

        # Assert
        assert len(recommendations) >= 1
        assert recommendation_time_ms < 200  # <200ms target
        assert all(r.score > 0 for r in recommendations)
        assert all(r.explanation for r in recommendations)

    def test_production_mixin_recommendation(
        self, requirements_analyzer, mixin_recommender, production_contract
    ):
        """Test mixin recommendation for production contract."""
        # Analyze requirements
        analysis = requirements_analyzer.analyze(production_contract)

        # Get recommendations
        recommendations = mixin_recommender.recommend_mixins(analysis, top_k=10)

        # Assert
        assert len(recommendations) >= 5
        # Should recommend database, health, metrics, and event mixins
        mixin_names = [r.mixin_name.lower() for r in recommendations]
        # At least some production-oriented mixins
        assert any("health" in name or "metric" in name for name in mixin_names)


@pytest.mark.integration
class TestContextBuilding:
    """Test enhanced LLM context building."""

    def test_simple_context_building(
        self,
        context_builder,
        template_selector,
        mixin_recommender,
        requirements_analyzer,
        pattern_library,
        simple_contract,
    ):
        """Test context building for simple contract."""
        # Prepare components
        template_selection = template_selector.select_template(
            simple_contract, "effect"
        )
        requirement_analysis = requirements_analyzer.analyze(simple_contract)
        mixin_recommendations = mixin_recommender.recommend_mixins(
            requirement_analysis, top_k=3
        )
        pattern_matches = pattern_library.find_matching_patterns(
            operation_type="database",
            features=set(template_selection.patterns),
            node_type="effect",
        )

        # Build context
        start_time = time.perf_counter()
        context = context_builder.build_context(
            requirements=simple_contract,
            operation=simple_contract.io_operations[0],
            template_selection=template_selection,
            mixin_selection=mixin_recommendations,
            pattern_matches=pattern_matches,
        )
        build_time_ms = (time.perf_counter() - start_time) * 1000

        # Assert
        assert context.operation_name == "fetch_data"
        assert context.node_type == "effect"
        assert context.estimated_tokens > 0
        assert context.estimated_tokens < 8000  # <8K tokens target
        assert build_time_ms < 50  # <50ms target
        assert len(context.patterns) >= 0
        assert len(context.mixins) >= 1

    def test_production_context_building(
        self,
        context_builder,
        template_selector,
        mixin_recommender,
        requirements_analyzer,
        pattern_library,
        production_contract,
    ):
        """Test context building for production contract."""
        # Prepare components
        template_selection = template_selector.select_template(
            production_contract, "effect", "production"
        )
        requirement_analysis = requirements_analyzer.analyze(production_contract)
        mixin_recommendations = mixin_recommender.recommend_mixins(
            requirement_analysis, top_k=10
        )
        pattern_matches = pattern_library.find_matching_patterns(
            operation_type="database",
            features=set(template_selection.patterns),
            node_type="effect",
        )

        # Build context for first operation
        context = context_builder.build_context(
            requirements=production_contract,
            operation=production_contract.io_operations[0],
            template_selection=template_selection,
            mixin_selection=mixin_recommendations,
            pattern_matches=pattern_matches,
        )

        # Assert
        assert context.operation_name == "create"
        assert context.template_variant == "production"
        assert len(context.patterns) >= 3
        assert len(context.mixins) >= 5
        assert len(context.best_practices) >= 4
        assert context.estimated_tokens < 8000  # <8K tokens target


# ============================================================================
# Pipeline End-to-End Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestEnhancedPipeline:
    """Test complete enhanced pipeline end-to-end."""

    def test_pipeline_initialization(self, enhanced_pipeline):
        """Test pipeline initializes all components."""
        assert enhanced_pipeline.template_selector is not None
        assert enhanced_pipeline.pattern_library is not None
        assert enhanced_pipeline.context_builder is not None
        assert enhanced_pipeline.requirements_analyzer is not None
        assert enhanced_pipeline.mixin_recommender is not None
        assert enhanced_pipeline.subcontract_processor is not None

    @pytest.mark.skip(reason="Requires contract YAML files")
    async def test_simple_pipeline_execution(self, enhanced_pipeline, tmp_path):
        """Test pipeline execution with simple contract."""
        # This test requires actual contract YAML files
        # Skipping for now as it's implementation-dependent
        pass

    @pytest.mark.skip(reason="Requires contract YAML files")
    async def test_production_pipeline_execution(self, enhanced_pipeline, tmp_path):
        """Test pipeline execution with production contract."""
        # This test requires actual contract YAML files
        # Skipping for now as it's implementation-dependent
        pass


# ============================================================================
# Performance Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.performance
class TestPhase3Performance:
    """Test performance of Phase 3 components."""

    def test_template_selection_performance(self, template_selector):
        """Test template selection meets <5ms target."""
        from dataclasses import dataclass, field

        @dataclass
        class Contract:
            node_type: str = "effect"
            operations: list = field(default_factory=lambda: ["read", "write"])
            dependencies: dict = field(default_factory=dict)
            io_operations: list = field(default_factory=list)

        contract = Contract()

        # Warm up
        for _ in range(5):
            template_selector.select_template(contract, "effect")

        # Measure
        times = []
        for _ in range(100):
            selection = template_selector.select_template(contract, "effect")
            times.append(selection.selection_time_ms)

        avg_time = sum(times) / len(times)
        p99_time = sorted(times)[98]

        # Assert
        assert avg_time < 3  # Average <3ms
        assert p99_time < 5  # P99 <5ms

    def test_pattern_matching_performance(self, pattern_library):
        """Test pattern matching meets <10ms target."""
        # Measure
        times = []
        for _ in range(100):
            start = time.perf_counter()
            pattern_library.find_matching_patterns(
                operation_type="database",
                features={"health_checks", "metrics", "lifecycle"},
                node_type="effect",
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)

        avg_time = sum(times) / len(times)
        p99_time = sorted(times)[98]

        # Assert
        assert avg_time < 8  # Average <8ms
        assert p99_time < 10  # P99 <10ms

    def test_mixin_recommendation_performance(
        self, requirements_analyzer, mixin_recommender
    ):
        """Test mixin recommendation meets <200ms target."""
        from dataclasses import dataclass, field

        @dataclass
        class Contract:
            node_type: str = "effect"
            operations: list = field(
                default_factory=lambda: ["create", "read", "update", "delete"]
            )
            dependencies: dict = field(
                default_factory=lambda: {
                    "asyncpg": "^0.29.0",
                    "aiokafka": "^0.10.0",
                }
            )
            domain: str = "database"

        contract = Contract()

        # Measure
        times = []
        for _ in range(50):
            start = time.perf_counter()
            analysis = requirements_analyzer.analyze(contract)
            mixin_recommender.recommend_mixins(analysis, top_k=10)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)

        avg_time = sum(times) / len(times)
        p99_time = sorted(times)[int(len(times) * 0.99)]

        # Assert
        assert avg_time < 150  # Average <150ms
        assert p99_time < 200  # P99 <200ms


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
