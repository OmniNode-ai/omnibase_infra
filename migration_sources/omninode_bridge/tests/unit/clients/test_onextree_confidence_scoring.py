"""Unit tests for OnexTree client confidence scoring and fallback mechanisms."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest

from omninode_bridge.clients.onextree_client import (
    AsyncOnexTreeClient,
    ConfidenceLevel,
    ConfidenceScore,
    IntelligenceResult,
)


@pytest.fixture
def mock_httpx_client():
    """Mock httpx AsyncClient."""
    mock = AsyncMock(spec=httpx.AsyncClient)
    mock.aclose = AsyncMock()
    return mock


@pytest.fixture
async def onextree_client(mock_httpx_client):
    """Provide AsyncOnexTreeClient instance for testing."""
    client = AsyncOnexTreeClient(
        base_url="http://localhost:8054",
        timeout=5.0,
        max_retries=2,
        enable_cache=True,
        cache_ttl=300,
    )

    # Mock HTTP client
    with patch.object(
        client,
        "_http_client",
        mock_httpx_client,
    ):
        yield client

    # Cleanup
    await client.close()


class TestConfidenceScoring:
    """Test confidence scoring functionality."""

    def test_confidence_score_initialization(self):
        """Test ConfidenceScore initializes with defaults."""
        score = ConfidenceScore()

        assert score.data_quality == 0.0
        assert score.pattern_strength == 0.0
        assert score.coverage == 0.0
        assert score.relationship_density == 0.0
        assert score.overall == 0.0
        assert score.level == ConfidenceLevel.MINIMAL

    def test_confidence_score_overall_calculation(self):
        """Test overall confidence score is calculated correctly."""
        score = ConfidenceScore(
            data_quality=1.0,  # 35% weight
            pattern_strength=0.8,  # 30% weight
            coverage=0.6,  # 20% weight
            relationship_density=0.4,  # 15% weight
        )

        # Expected: 1.0*0.35 + 0.8*0.30 + 0.6*0.20 + 0.4*0.15 = 0.83
        expected = 0.35 + 0.24 + 0.12 + 0.06
        assert abs(score.overall - expected) < 0.01

    def test_confidence_level_high(self):
        """Test HIGH confidence level (0.8+)."""
        score = ConfidenceScore(
            data_quality=1.0,
            pattern_strength=1.0,
            coverage=1.0,
            relationship_density=1.0,
        )

        assert score.overall >= 0.8
        assert score.level == ConfidenceLevel.HIGH

    def test_confidence_level_medium(self):
        """Test MEDIUM confidence level (0.5-0.8)."""
        score = ConfidenceScore(
            data_quality=0.6,
            pattern_strength=0.6,
            coverage=0.6,
            relationship_density=0.6,
        )

        assert 0.5 <= score.overall < 0.8
        assert score.level == ConfidenceLevel.MEDIUM

    def test_confidence_level_low(self):
        """Test LOW confidence level (0.2-0.5)."""
        score = ConfidenceScore(
            data_quality=0.3,
            pattern_strength=0.3,
            coverage=0.3,
            relationship_density=0.3,
        )

        assert 0.2 <= score.overall < 0.5
        assert score.level == ConfidenceLevel.LOW

    def test_confidence_level_minimal(self):
        """Test MINIMAL confidence level (<0.2)."""
        score = ConfidenceScore(
            data_quality=0.0,
            pattern_strength=0.0,
            coverage=0.0,
            relationship_density=0.0,
        )

        assert score.overall < 0.2
        assert score.level == ConfidenceLevel.MINIMAL

    def test_confidence_calculation_with_patterns(self, onextree_client):
        """Test confidence calculation includes pattern strength."""
        patterns = ["pattern1", "pattern2", "pattern3"]
        confidence = onextree_client._calculate_confidence(
            intelligence_data={},
            patterns=patterns,
            relationships=[],
            metadata={},
        )

        # Should have some pattern strength
        assert confidence.pattern_strength > 0.0

    def test_confidence_calculation_with_relationships(self, onextree_client):
        """Test confidence calculation includes relationship density."""
        relationships = [
            {"from": "A", "to": "B"},
            {"from": "B", "to": "C"},
            {"from": "C", "to": "D"},
        ]
        confidence = onextree_client._calculate_confidence(
            intelligence_data={},
            patterns=[],
            relationships=relationships,
            metadata={},
        )

        # Should have some relationship density
        assert confidence.relationship_density > 0.0

    def test_confidence_calculation_with_complete_data(self, onextree_client):
        """Test confidence calculation with complete intelligence data."""
        intelligence_data = {
            "analysis_type": "comprehensive",
            "recommendations": "This is a detailed recommendation with over 100 characters to test the boost logic for detailed content.",
        }
        confidence = onextree_client._calculate_confidence(
            intelligence_data=intelligence_data,
            patterns=[],
            relationships=[],
            metadata={},
        )

        # Should have high data quality
        assert confidence.data_quality > 0.5

    def test_confidence_calculation_with_metadata(self, onextree_client):
        """Test confidence calculation includes coverage from metadata."""
        metadata = {
            "nodes_analyzed": 100,
            "files_scanned": 50,
            "tree_loaded": True,
        }
        confidence = onextree_client._calculate_confidence(
            intelligence_data={},
            patterns=[],
            relationships=[],
            metadata=metadata,
        )

        # Should have high coverage
        assert confidence.coverage > 0.5


class TestFallbackMechanism:
    """Test fallback mechanism for graceful degradation."""

    def test_fallback_result_creation(self, onextree_client):
        """Test fallback result is created correctly."""
        context = "test context"
        fallback = onextree_client._create_fallback_result(context)

        assert isinstance(fallback, IntelligenceResult)
        assert fallback.degraded is True
        assert fallback.confidence.overall == 0.0
        assert fallback.confidence.level == ConfidenceLevel.MINIMAL
        assert fallback.intelligence["analysis_type"] == "fallback"
        assert fallback.intelligence["status"] == "degraded"

    def test_fallback_result_with_error(self, onextree_client):
        """Test fallback result includes error information."""
        context = "test context"
        error = Exception("Service unavailable")
        fallback = onextree_client._create_fallback_result(context, error=error)

        assert "Service unavailable" in fallback.metadata["error"]
        assert fallback.metadata["fallback"] is True

    @pytest.mark.asyncio
    async def test_get_intelligence_fallback_on_error(
        self, onextree_client, mock_httpx_client
    ):
        """Test get_intelligence returns fallback on error."""
        # Mock request to raise exception
        mock_httpx_client.request = AsyncMock(
            side_effect=httpx.TimeoutException("Timeout")
        )

        # Should return fallback result, not raise exception
        result = await onextree_client.get_intelligence(
            context="test",
            enable_fallback=True,
        )

        assert isinstance(result, IntelligenceResult)
        assert result.degraded is True
        assert result.confidence.overall == 0.0

    @pytest.mark.asyncio
    async def test_get_intelligence_raises_without_fallback(
        self, onextree_client, mock_httpx_client
    ):
        """Test get_intelligence raises exception when fallback disabled."""
        from omninode_bridge.clients.base_client import ServiceUnavailableError

        # Mock request to raise exception
        mock_httpx_client.request = AsyncMock(
            side_effect=httpx.TimeoutException("Timeout")
        )

        # Should raise exception
        with pytest.raises((ServiceUnavailableError, Exception)):
            await onextree_client.get_intelligence(
                context="test",
                enable_fallback=False,
            )


class TestIntelligenceResultWithConfidence:
    """Test IntelligenceResult model."""

    def test_intelligence_result_initialization(self):
        """Test IntelligenceResult initializes correctly."""
        result = IntelligenceResult(
            intelligence={"key": "value"},
            patterns=["pattern1"],
            relationships=[{"from": "A", "to": "B"}],
            metadata={"source": "test"},
        )

        assert result.intelligence == {"key": "value"}
        assert result.patterns == ["pattern1"]
        assert result.relationships == [{"from": "A", "to": "B"}]
        assert result.metadata == {"source": "test"}
        assert isinstance(result.confidence, ConfidenceScore)
        assert result.degraded is False

    def test_intelligence_result_with_high_confidence(self):
        """Test IntelligenceResult with high confidence."""
        confidence = ConfidenceScore(
            data_quality=0.9,
            pattern_strength=0.8,
            coverage=0.9,
            relationship_density=0.8,
        )

        result = IntelligenceResult(
            intelligence={},
            confidence=confidence,
        )

        assert result.confidence.level == ConfidenceLevel.HIGH
        assert result.confidence.overall >= 0.8

    def test_intelligence_result_degraded_flag(self):
        """Test IntelligenceResult degraded flag."""
        result = IntelligenceResult(
            intelligence={},
            degraded=True,
        )

        assert result.degraded is True


class TestIntegrationWithConfidence:
    """Test integration of confidence scoring with client methods."""

    @pytest.mark.asyncio
    async def test_get_intelligence_returns_intelligence_result(
        self, onextree_client, mock_httpx_client
    ):
        """Test get_intelligence returns IntelligenceResult with confidence."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "intelligence": {
                    "analysis_type": "comprehensive",
                    "recommendations": "Detailed recommendations with sufficient length to trigger quality boost.",
                },
                "patterns": ["pattern1", "pattern2", "pattern3"],
                "relationships": [
                    {"from": "A", "to": "B"},
                    {"from": "B", "to": "C"},
                ],
                "metadata": {
                    "nodes_analyzed": 50,
                    "tree_loaded": True,
                },
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        result = await onextree_client.get_intelligence(
            context="test",
            correlation_id=uuid4(),
        )

        # Verify result type and structure
        assert isinstance(result, IntelligenceResult)
        assert isinstance(result.confidence, ConfidenceScore)
        assert result.degraded is False

        # Verify confidence was calculated
        assert result.confidence.overall > 0.0
        assert result.confidence.data_quality > 0.0
        assert result.confidence.pattern_strength > 0.0
        assert result.confidence.coverage > 0.0
        assert result.confidence.relationship_density > 0.0

    @pytest.mark.asyncio
    async def test_cached_intelligence_result_compatibility(
        self, onextree_client, mock_httpx_client
    ):
        """Test client handles both legacy dict cache and new IntelligenceResult cache."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "intelligence": {"key": "value"},
                "patterns": ["pattern1"],
                "relationships": [],
                "metadata": {},
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.request = AsyncMock(return_value=mock_response)

        # First call - should hit service and cache result
        result1 = await onextree_client.get_intelligence(context="test", use_cache=True)

        # Second call - should use cache
        result2 = await onextree_client.get_intelligence(context="test", use_cache=True)

        # Both should be IntelligenceResult
        assert isinstance(result1, IntelligenceResult)
        assert isinstance(result2, IntelligenceResult)

        # Should only have made one request (second was cached)
        assert mock_httpx_client.request.call_count == 1


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_confidence_score_boundary_values(self):
        """Test confidence score handles boundary values correctly."""
        # All zeros
        score = ConfidenceScore(
            data_quality=0.0,
            pattern_strength=0.0,
            coverage=0.0,
            relationship_density=0.0,
        )
        assert score.overall == 0.0

        # All ones (use approximate comparison for floating point)
        score = ConfidenceScore(
            data_quality=1.0,
            pattern_strength=1.0,
            coverage=1.0,
            relationship_density=1.0,
        )
        assert abs(score.overall - 1.0) < 0.0001

    def test_confidence_calculation_empty_data(self, onextree_client):
        """Test confidence calculation with empty data."""
        confidence = onextree_client._calculate_confidence(
            intelligence_data={},
            patterns=[],
            relationships=[],
            metadata={},
        )

        assert confidence.overall == 0.0
        assert confidence.level == ConfidenceLevel.MINIMAL

    def test_confidence_calculation_many_patterns(self, onextree_client):
        """Test confidence calculation with many patterns (>10)."""
        patterns = [f"pattern{i}" for i in range(15)]
        confidence = onextree_client._calculate_confidence(
            intelligence_data={},
            patterns=patterns,
            relationships=[],
            metadata={},
        )

        # Pattern strength should be capped at 1.0 even with >10 patterns
        assert confidence.pattern_strength <= 1.0

    def test_confidence_calculation_many_relationships(self, onextree_client):
        """Test confidence calculation with many relationships (>20)."""
        relationships = [{"from": str(i), "to": str(i + 1)} for i in range(25)]
        confidence = onextree_client._calculate_confidence(
            intelligence_data={},
            patterns=[],
            relationships=relationships,
            metadata={},
        )

        # Relationship density should be capped at 1.0 even with >20 relationships
        assert confidence.relationship_density <= 1.0
