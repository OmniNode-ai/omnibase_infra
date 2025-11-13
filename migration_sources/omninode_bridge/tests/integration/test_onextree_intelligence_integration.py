"""Integration tests for OnexTree Intelligence Service integration.

Tests real HTTP integration with OnexTree service including:
- Intelligence endpoint integration
- Timeout configuration (500ms max)
- Graceful degradation when service unavailable
- Error handling and fallback behavior
"""

import time
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio
from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_core.models.contracts.model_contract_orchestrator import (
    ModelContractOrchestrator,
)

from omninode_bridge.clients.onextree_client import AsyncOnexTreeClient
from omninode_bridge.nodes.orchestrator.v1_0_0.node import NodeBridgeOrchestrator


@pytest_asyncio.fixture
async def onextree_service():
    """Provide OnexTree ASGI app with test tree loaded for testing.

    Returns ASGI app configured with test data, avoiding network calls.
    Tests use this via ASGI transport in AsyncOnexTreeClient.
    """
    # Initialize app state
    import asyncio
    import tempfile

    import httpx

    from src.onextree_service.main import app

    app.state.generation_lock = asyncio.Lock()
    app.state.query_engine = None
    app.state.current_tree = None

    # Generate a test tree using ASGI transport
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some test files
        test_dir = Path(tmpdir) / "test_project"
        test_dir.mkdir()
        (test_dir / "main.py").write_text("# Test file")
        (test_dir / "auth.py").write_text("# Authentication")
        (test_dir / "api.py").write_text("# API endpoints")

        # Use ASGI client to generate tree
        async with httpx.AsyncClient(app=app, base_url="http://testserver") as client:
            response = await client.post(
                "/generate", json={"project_root": str(test_dir)}
            )
            assert response.status_code == 200

        # Yield the app for tests to use via ASGI transport
        yield app


@pytest.mark.integration
@pytest.mark.asyncio
async def test_intelligence_endpoint_integration(onextree_service):
    """Test /intelligence endpoint returns expected response."""
    async with AsyncOnexTreeClient(
        timeout=1.0,
        enable_cache=True,
        app=onextree_service,
    ) as client:
        # Test intelligence retrieval
        result = await client.get_intelligence(
            context="authentication patterns",
            include_patterns=True,
            include_relationships=True,
            correlation_id=uuid4(),
        )

        # Verify response structure
        assert "intelligence" in result
        assert "patterns" in result
        assert "relationships" in result
        assert "metadata" in result

        # Verify intelligence fields
        intelligence = result["intelligence"]
        assert "analysis_type" in intelligence
        assert "confidence_score" in intelligence
        assert "recommendations" in intelligence


@pytest.mark.integration
@pytest.mark.asyncio
async def test_intelligence_timeout_configuration(onextree_service):
    """Test intelligence requests respect timeout configuration."""
    # Create client with very short timeout (100ms)
    async with AsyncOnexTreeClient(
        timeout=0.1,  # 100ms timeout
        max_retries=1,
        app=onextree_service,
    ) as client:
        start_time = time.time()

        try:
            # This should timeout quickly
            await client.get_intelligence(
                context="test",
                correlation_id=uuid4(),
            )
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000

            # Should fail fast due to timeout
            assert elapsed_ms < 500, f"Timeout took {elapsed_ms}ms, expected < 500ms"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_intelligence_patterns_detection(onextree_service):
    """Test intelligence endpoint detects patterns based on context."""
    async with AsyncOnexTreeClient(
        timeout=1.0,
        app=onextree_service,
    ) as client:
        # Test different context types
        test_cases = [
            ("authentication patterns", "auth"),
            ("API endpoints design", "api"),
            ("database schema validation", "data"),
            ("test coverage analysis", "test"),
        ]

        for context, expected_pattern in test_cases:
            result = await client.get_intelligence(
                context=context,
                include_patterns=True,
                correlation_id=uuid4(),
            )

            patterns = result.get("patterns", [])
            # Should detect relevant patterns
            assert len(patterns) > 0, f"No patterns detected for: {context}"

            # Verify pattern relevance
            pattern_text = " ".join(patterns).lower()
            assert any(
                keyword in pattern_text
                for keyword in [expected_pattern, "pattern", "structure"]
            ), f"Expected pattern keyword not found for: {context}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_orchestrator_onextree_integration(onextree_service):
    """Test NodeBridgeOrchestrator integrates with OnexTree service."""
    # Create container with OnexTree configuration and ASGI app
    container = ModelONEXContainer()

    # Create orchestrator
    orchestrator = NodeBridgeOrchestrator(container)

    # Create contract with OnexTree intelligence step
    workflow_id = uuid4()
    contract = ModelContractOrchestrator(
        correlation_id=workflow_id,
        input_data={
            "context": "authentication patterns",
            "content": "test content",
        },
    )

    # Execute OnexTree intelligence step
    step = {
        "step_id": "onextree_intelligence",
        "step_type": "onextree_intelligence",
        "required": False,  # Optional step for graceful degradation
    }

    result = await orchestrator._route_to_onextree(step, contract, workflow_id)

    # Verify result structure
    assert result["step_type"] == "onextree_intelligence"
    assert result["status"] in ["success", "degraded"]
    assert "intelligence_data" in result
    assert "intelligence_time_ms" in result

    # Verify intelligence data
    if result["status"] == "success":
        intelligence = result["intelligence_data"]
        assert "analysis_type" in intelligence
        assert "confidence_score" in intelligence
        assert "recommendations" in intelligence


@pytest.mark.integration
@pytest.mark.asyncio
async def test_graceful_degradation_service_unavailable():
    """Test graceful degradation when OnexTree service is unavailable."""
    # Create container with invalid OnexTree URL
    container = ModelONEXContainer()

    orchestrator = NodeBridgeOrchestrator(container)

    workflow_id = uuid4()
    contract = ModelContractOrchestrator(
        correlation_id=workflow_id,
        input_data={"context": "test"},
    )

    step = {
        "step_id": "onextree_intelligence",
        "step_type": "onextree_intelligence",
        "required": False,
    }

    # Should not raise exception - graceful degradation
    result = await orchestrator._route_to_onextree(step, contract, workflow_id)

    # Verify fallback behavior
    assert result["step_type"] == "onextree_intelligence"
    assert result["status"] == "degraded"
    assert "intelligence_data" in result
    assert result.get("degraded") is True

    # Verify fallback intelligence
    fallback_intelligence = result["intelligence_data"]
    assert fallback_intelligence["analysis_type"] == "fallback"
    assert float(fallback_intelligence["confidence_score"]) == 0.0
    assert "unavailable" in fallback_intelligence["recommendations"].lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_timeout_enforcement(onextree_service):
    """Test timeout is enforced at 500ms."""
    container = ModelONEXContainer()

    orchestrator = NodeBridgeOrchestrator(container)

    workflow_id = uuid4()
    contract = ModelContractOrchestrator(
        correlation_id=workflow_id,
        input_data={"context": "test"},
    )

    step = {
        "step_id": "onextree_intelligence",
        "step_type": "onextree_intelligence",
    }

    start_time = time.time()
    result = await orchestrator._route_to_onextree(step, contract, workflow_id)
    elapsed_ms = (time.time() - start_time) * 1000

    # Should complete within reasonable time (timeout + overhead)
    assert elapsed_ms < 1000, f"Request took {elapsed_ms}ms, expected < 1000ms"

    # Result should indicate timeout or success
    assert result["status"] in ["success", "degraded"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_intelligence_caching(onextree_service):
    """Test intelligence responses are cached for performance."""
    async with AsyncOnexTreeClient(
        timeout=1.0,
        enable_cache=True,
        cache_ttl=300,
        app=onextree_service,
    ) as client:
        # First request
        start1 = time.time()
        result1 = await client.get_intelligence(
            context="authentication patterns",
            use_cache=True,
        )
        time1_ms = (time.time() - start1) * 1000

        # Second request (should hit cache)
        start2 = time.time()
        result2 = await client.get_intelligence(
            context="authentication patterns",
            use_cache=True,
        )
        time2_ms = (time.time() - start2) * 1000

        # Results should be identical
        assert result1 == result2

        # Second request should be much faster (cached)
        assert (
            time2_ms < time1_ms
        ), f"Cache hit ({time2_ms}ms) not faster than miss ({time1_ms}ms)"

        # Verify cache stats
        cache_stats = client.get_cache_stats()
        assert cache_stats["cache_hits"] >= 1
        assert cache_stats["hit_rate"] > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_correlation_id_propagation(onextree_service):
    """Test correlation ID is propagated through requests."""
    correlation_id = uuid4()

    async with AsyncOnexTreeClient(
        timeout=1.0,
        app=onextree_service,
    ) as client:
        # Mock to verify headers
        original_request = client._http_client.request

        async def mock_request(*args, **kwargs):
            # Verify correlation ID header
            headers = kwargs.get("headers", {})
            assert "X-Correlation-ID" in headers
            assert headers["X-Correlation-ID"] == str(correlation_id)
            return await original_request(*args, **kwargs)

        client._http_client.request = mock_request

        await client.get_intelligence(
            context="test",
            correlation_id=correlation_id,
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_error_scenarios(onextree_service):
    """Test various error scenarios and recovery."""
    async with AsyncOnexTreeClient(
        timeout=5.0,
        max_retries=2,
        app=onextree_service,
    ) as client:
        # Test with no tree loaded (should return minimal intelligence)
        result = await client.get_intelligence(
            context="test without tree",
            correlation_id=uuid4(),
        )

        # Should still return valid response structure
        assert "intelligence" in result
        assert "metadata" in result

        # Metadata should indicate no tree loaded
        metadata = result.get("metadata", {})
        if "tree_loaded" in metadata:
            assert metadata["tree_loaded"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
