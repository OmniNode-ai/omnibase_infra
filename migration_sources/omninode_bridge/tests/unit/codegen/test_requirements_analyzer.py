"""
Unit tests for RequirementsAnalyzer.

Tests keyword extraction, dependency analysis, operation pattern recognition,
and requirement categorization.
"""

import pytest

from omninode_bridge.codegen.mixins.requirements_analyzer import RequirementsAnalyzer
from omninode_bridge.codegen.prd_analyzer import ModelPRDRequirements


class TestRequirementsAnalyzer:
    """Test suite for RequirementsAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create RequirementsAnalyzer instance."""
        return RequirementsAnalyzer()

    @pytest.fixture
    def database_requirements(self):
        """Create database-focused requirements."""
        return ModelPRDRequirements(
            node_type="effect",
            service_name="postgres_adapter",
            domain="database",
            operations=[
                "create_record",
                "read_record",
                "update_record",
                "delete_record",
            ],
            features=["connection_pooling", "transaction_management"],
            dependencies={"asyncpg": ">=0.28.0"},
            performance_requirements={"latency_ms": 50, "throughput_rps": 500},
            business_description="PostgreSQL CRUD adapter with connection pooling and transactions",
        )

    @pytest.fixture
    def api_requirements(self):
        """Create API-focused requirements."""
        return ModelPRDRequirements(
            node_type="effect",
            service_name="api_client",
            domain="api",
            operations=["fetch_data", "post_data", "handle_errors"],
            features=["retry_logic", "circuit_breaker", "timeout_handling"],
            dependencies={"httpx": ">=0.24.0"},
            performance_requirements={"availability": 0.999, "timeout_ms": 5000},
            business_description="HTTP client with fault tolerance and retry logic",
        )

    @pytest.fixture
    def kafka_requirements(self):
        """Create Kafka-focused requirements."""
        return ModelPRDRequirements(
            node_type="orchestrator",
            service_name="event_processor",
            domain="messaging",
            operations=["publish_event", "consume_event", "process_message"],
            features=["event_driven", "async_processing"],
            dependencies={"aiokafka": ">=0.8.0"},
            performance_requirements={"throughput_rps": 10000},
            business_description="Event-driven message processor with Kafka",
        )

    def test_keyword_extraction_database(self, analyzer, database_requirements):
        """Test keyword extraction for database requirements."""
        analysis = analyzer.analyze(database_requirements)

        assert (
            "postgresql" in analysis.keywords
            or "postgres" in analysis.keywords
            or "database" in analysis.keywords
        )
        assert (
            "connection" in analysis.keywords
            or "pool" in analysis.keywords
            or "pooling" in analysis.keywords
        )
        assert "transaction" in analysis.keywords or "transactions" in analysis.keywords
        assert len(analysis.keywords) > 5

    def test_dependency_analysis_database(self, analyzer, database_requirements):
        """Test dependency analysis for database requirements."""
        analysis = analyzer.analyze(database_requirements)

        assert "database" in analysis.dependency_packages
        assert "postgres" in analysis.dependency_packages

    def test_dependency_analysis_api(self, analyzer, api_requirements):
        """Test dependency analysis for API requirements."""
        analysis = analyzer.analyze(api_requirements)

        assert "api" in analysis.dependency_packages
        assert "http-client" in analysis.dependency_packages

    def test_categorization_database(self, analyzer, database_requirements):
        """Test requirement categorization for database node."""
        analysis = analyzer.analyze(database_requirements)

        # Database score should be highest
        assert analysis.database_score > 6.0
        assert analysis.database_score > analysis.api_score
        assert analysis.database_score > analysis.kafka_score

        # Other scores should be lower
        assert analysis.api_score < 3.0
        assert analysis.kafka_score < 3.0

    def test_categorization_api(self, analyzer, api_requirements):
        """Test requirement categorization for API node."""
        analysis = analyzer.analyze(api_requirements)

        # API score should be high, resilience should be present
        assert analysis.api_score > 5.0
        assert analysis.resilience_score > 1.5

        # Database and Kafka scores should be low
        assert analysis.database_score < 2.0
        assert analysis.kafka_score < 2.0

    def test_categorization_kafka(self, analyzer, kafka_requirements):
        """Test requirement categorization for Kafka node."""
        analysis = analyzer.analyze(kafka_requirements)

        # Kafka score should be highest
        assert analysis.kafka_score > 4.5
        assert analysis.kafka_score > analysis.database_score
        assert analysis.kafka_score > analysis.api_score

    def test_confidence_calculation(self, analyzer, database_requirements):
        """Test confidence calculation."""
        analysis = analyzer.analyze(database_requirements)

        # Should have reasonable confidence with clear signals
        assert 0.5 <= analysis.confidence <= 1.0

    def test_rationale_generation(self, analyzer, database_requirements):
        """Test rationale generation."""
        analysis = analyzer.analyze(database_requirements)

        assert len(analysis.rationale) > 10
        assert (
            "database" in analysis.rationale.lower()
            or "Detected requirements" in analysis.rationale
        )

    def test_operation_pattern_recognition(self, analyzer):
        """Test operation pattern recognition."""
        requirements = ModelPRDRequirements(
            node_type="effect",
            service_name="crud_adapter",
            domain="database",
            operations=["create", "read", "update", "delete"],
            features=[],
            dependencies={},
            performance_requirements={},
            business_description="CRUD operations",
        )

        analysis = analyzer.analyze(requirements)

        # Should detect database pattern from CRUD operations
        assert analysis.database_score > 3.0

    def test_performance_requirement_analysis(self, analyzer):
        """Test performance requirement analysis."""
        requirements = ModelPRDRequirements(
            node_type="effect",
            service_name="high_perf_adapter",
            domain="database",
            operations=["query"],
            features=[],
            dependencies={"asyncpg": ">=0.28.0"},
            performance_requirements={
                "latency_ms": 50,
                "throughput_rps": 1000,
                "availability": 0.999,
            },
            business_description="High-performance database adapter",
        )

        analysis = analyzer.analyze(requirements)

        # Should have reasonable database score and some performance consideration
        assert analysis.database_score > 3.0
        assert analysis.performance_score >= 0.0

    def test_empty_requirements(self, analyzer):
        """Test with minimal requirements."""
        requirements = ModelPRDRequirements(
            node_type="compute",
            service_name="simple_compute",
            domain="compute",
            operations=[],
            features=[],
            dependencies={},
            performance_requirements={},
            business_description="Simple compute node",
        )

        analysis = analyzer.analyze(requirements)

        # Should still produce valid analysis
        assert analysis.confidence >= 0.0
        assert len(analysis.keywords) >= 0

    def test_mixed_requirements(self, analyzer):
        """Test with mixed domain requirements."""
        requirements = ModelPRDRequirements(
            node_type="orchestrator",
            service_name="mixed_orchestrator",
            domain="orchestration",
            operations=["query_database", "call_api", "publish_event"],
            features=["database", "api", "kafka"],
            dependencies={
                "asyncpg": ">=0.28.0",
                "httpx": ">=0.24.0",
                "aiokafka": ">=0.8.0",
            },
            performance_requirements={},
            business_description="Orchestrator with database, API, and Kafka access",
        )

        analysis = analyzer.analyze(requirements)

        # Should have reasonable scores in all three categories
        assert analysis.database_score > 1.0
        assert analysis.api_score > 3.0
        assert analysis.kafka_score > 2.0

    def test_security_requirements(self, analyzer):
        """Test security requirement detection."""
        requirements = ModelPRDRequirements(
            node_type="effect",
            service_name="secure_api",
            domain="api",
            operations=["authenticate", "validate_token", "authorize"],
            features=["security", "token_validation", "encryption"],
            dependencies={"httpx": ">=0.24.0"},
            performance_requirements={},
            business_description="Secure API with authentication and encryption",
        )

        analysis = analyzer.analyze(requirements)

        # Should detect security requirements
        assert analysis.security_score > 3.0

    def test_observability_requirements(self, analyzer):
        """Test observability requirement detection."""
        requirements = ModelPRDRequirements(
            node_type="effect",
            service_name="monitored_service",
            domain="api",
            operations=["process_request"],
            features=["metrics", "logging", "health_checks", "tracing"],
            dependencies={"prometheus-client": ">=0.16.0"},
            performance_requirements={},
            business_description="Service with comprehensive observability",
        )

        analysis = analyzer.analyze(requirements)

        # Should detect observability requirements
        assert analysis.observability_score > 2.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
