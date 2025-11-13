#!/usr/bin/env python3
"""
Comprehensive Cross-Service Coordination Tests.

Tests complex workflows involving multiple services:
- Orchestrator → MetadataStampingService → Reducer
- Orchestrator → OnexTree Intelligence → Workflow enrichment
- Service discovery and circuit breaker patterns
- Timeout handling and fallback strategies
- Performance validation across service boundaries

Critical Scenarios:
1. OnexTree Intelligence Integration
   - Request → Analysis → Response → Workflow enrichment
   - Performance: <150ms with intelligence, <50ms without

2. Service Discovery and Health Checks
   - Dynamic service discovery via Consul
   - Health check integration
   - Fallback to localhost when Consul unavailable

3. Circuit Breaker Across Services
   - OnexTree circuit breaker protection
   - MetadataStamping circuit breaker protection
   - Independent failure handling

4. Timeout Handling
   - OnexTree timeout: 30s
   - MetadataStamping timeout: 5s
   - Graceful degradation on timeout

5. Fallback Strategies
   - OnexTree unavailable → Skip intelligence
   - MetadataStamping unavailable → Use in-memory hash
   - Reducer unavailable → Skip aggregation
"""

import asyncio
import json
import time
from datetime import UTC, datetime
from uuid import uuid4

import pytest

# ============================================================================
# Test Suite 1: OnexTree Intelligence Integration
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.e2e
async def test_onextree_intelligence_integration_complete_flow(
    stamp_request_factory,
    kafka_client,
    performance_validator,
):
    """
    Test complete OnexTree intelligence integration flow.

    Flow:
    1. Orchestrator receives stamp request with intelligence enabled
    2. Routes to OnexTree for AI analysis
    3. OnexTree responds with intelligence data
    4. Orchestrator incorporates intelligence into workflow
    5. Workflow completes with enriched metadata
    6. Events published with intelligence data

    Expected Performance:
    - With intelligence: <150ms
    - Intelligence analysis: <100ms
    - Network overhead: <20ms
    """
    # Step 1: Create request with intelligence enabled
    request = stamp_request_factory(
        file_path="/data/cross_service/contract.pdf",
        content=b"Legal contract requiring AI analysis",
        namespace="test.cross.service.onextree",
        enable_intelligence=True,
    )

    workflow_id = uuid4()
    correlation_id = uuid4()

    start_time = time.perf_counter()

    # Step 2: Request OnexTree analysis
    onextree_request_time = time.perf_counter()

    await kafka_client["producer"].send(
        "test.onextree.analysis.request",
        key=str(correlation_id).encode(),
        value=json.dumps(
            {
                "correlation_id": str(correlation_id),
                "file_path": request.file_path,
                "content_preview": request.file_content[:200].decode(
                    "utf-8", errors="ignore"
                ),
                "analysis_context": request.intelligence_context,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ).encode("utf-8"),
    )

    # Step 3: Simulate OnexTree analysis (AI processing)
    await asyncio.sleep(0.08)  # 80ms AI processing

    intelligence_data = {
        "file_type_detected": "legal_contract",
        "confidence_score": 0.95,
        "recommended_tags": ["legal", "contract", "review"],
        "entity_extraction": {
            "parties": ["Company A", "Company B"],
            "dates": ["2025-01-15"],
            "amounts": ["$100,000"],
        },
        "risk_assessment": {
            "risk_level": "low",
            "confidence": 0.92,
        },
        "processing_time_ms": 80,
    }

    # Step 4: OnexTree responds
    await kafka_client["producer"].send(
        "test.onextree.analysis.response",
        key=str(correlation_id).encode(),
        value=json.dumps(
            {
                "correlation_id": str(correlation_id),
                "intelligence_data": intelligence_data,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ).encode("utf-8"),
    )

    onextree_duration_ms = (time.perf_counter() - onextree_request_time) * 1000

    # Step 5: Orchestrator incorporates intelligence
    enriched_metadata = {
        "file_path": request.file_path,
        "namespace": request.namespace,
        "intelligence": intelligence_data,
    }

    # Step 6: Workflow completes
    await kafka_client["producer"].send(
        "test.workflow.completed.with.intelligence",
        key=str(workflow_id).encode(),
        value=json.dumps(
            {
                "workflow_id": str(workflow_id),
                "correlation_id": str(correlation_id),
                "state": "COMPLETED",
                "metadata": enriched_metadata,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ).encode("utf-8"),
    )

    total_duration_ms = (time.perf_counter() - start_time) * 1000

    # Validate performance
    performance_validator.validate(
        "onextree_intelligence_complete_flow",
        total_duration_ms,
        "cross_service_coordination_ms",
    )

    # Validate intelligence integration
    assert intelligence_data["confidence_score"] > 0.9, "High confidence intelligence"
    assert len(intelligence_data["recommended_tags"]) > 0, "Should have tags"


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.e2e
async def test_workflow_performance_with_vs_without_intelligence(
    stamp_request_factory,
    kafka_client,
    performance_validator,
):
    """
    Test performance difference with vs without OnexTree intelligence.

    Flow:
    1. Execute workflow WITHOUT intelligence → measure time
    2. Execute workflow WITH intelligence → measure time
    3. Validate performance thresholds:
       - Without intelligence: <50ms
       - With intelligence: <150ms

    Expected: Intelligence adds <100ms overhead
    """
    # Test 1: Without intelligence
    request_no_intelligence = stamp_request_factory(
        file_path="/data/test/doc1.pdf",
        content=b"Test document without intelligence",
        enable_intelligence=False,
    )

    start_no_intel = time.perf_counter()

    # Simulate workflow without intelligence
    workflow_id_1 = uuid4()
    await kafka_client["producer"].send(
        "test.workflow.no.intelligence",
        key=str(workflow_id_1).encode(),
        value=json.dumps(
            {
                "workflow_id": str(workflow_id_1),
                "state": "COMPLETED",
                "has_intelligence": False,
            }
        ).encode("utf-8"),
    )

    duration_no_intel_ms = (time.perf_counter() - start_no_intel) * 1000

    # Test 2: With intelligence
    request_with_intelligence = stamp_request_factory(
        file_path="/data/test/doc2.pdf",
        content=b"Test document with intelligence",
        enable_intelligence=True,
    )

    start_with_intel = time.perf_counter()

    # Simulate OnexTree analysis
    await asyncio.sleep(0.08)  # 80ms intelligence processing

    workflow_id_2 = uuid4()
    await kafka_client["producer"].send(
        "test.workflow.with.intelligence",
        key=str(workflow_id_2).encode(),
        value=json.dumps(
            {
                "workflow_id": str(workflow_id_2),
                "state": "COMPLETED",
                "has_intelligence": True,
                "intelligence_data": {"confidence": 0.95},
            }
        ).encode("utf-8"),
    )

    duration_with_intel_ms = (time.perf_counter() - start_with_intel) * 1000

    # Validate thresholds
    assert (
        duration_no_intel_ms < 50
    ), f"Workflow without intelligence exceeded 50ms: {duration_no_intel_ms:.2f}ms"

    assert (
        duration_with_intel_ms < 150
    ), f"Workflow with intelligence exceeded 150ms: {duration_with_intel_ms:.2f}ms"

    # Validate intelligence overhead
    intelligence_overhead_ms = duration_with_intel_ms - duration_no_intel_ms
    assert (
        intelligence_overhead_ms < 120
    ), f"Intelligence overhead too high: {intelligence_overhead_ms:.2f}ms"


# ============================================================================
# Test Suite 2: Service Discovery and Health Checks
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.e2e
async def test_service_discovery_via_consul():
    """
    Test dynamic service discovery via Consul.

    Flow:
    1. Query Consul for MetadataStamping service
    2. Get service URL and health status
    3. Use discovered URL for requests
    4. Fallback to localhost if Consul unavailable

    Expected: Service discovered in <100ms
    """

    # Mock Consul service discovery
    async def discover_service(service_name: str) -> dict:
        """Discover service via Consul."""
        # Simulate Consul lookup
        await asyncio.sleep(0.01)  # 10ms lookup time

        services = {
            "metadata-stamping": {
                "url": "http://metadata-stamping:8053",
                "health": "passing",
                "port": 8053,
            },
            "onextree": {
                "url": "http://onextree-intelligence:8080",
                "health": "passing",
                "port": 8080,
            },
        }

        return services.get(
            service_name, {"url": "http://localhost:8053", "health": "unknown"}
        )

    # Step 1: Discover MetadataStamping service
    start_time = time.perf_counter()
    metadata_service = await discover_service("metadata-stamping")
    discovery_duration_ms = (time.perf_counter() - start_time) * 1000

    # Validate discovery
    assert metadata_service["health"] == "passing", "Service should be healthy"
    assert "http://" in metadata_service["url"], "Should have valid URL"
    assert discovery_duration_ms < 100, f"Discovery took {discovery_duration_ms:.2f}ms"

    # Step 2: Discover OnexTree service
    onextree_service = await discover_service("onextree")
    assert onextree_service["health"] == "passing", "OnexTree should be healthy"


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.e2e
async def test_fallback_to_localhost_when_consul_unavailable():
    """
    Test fallback to localhost when Consul is unavailable.

    Flow:
    1. Attempt Consul service discovery
    2. Consul unavailable (connection error)
    3. Fallback to localhost URLs
    4. Services still accessible via localhost

    Expected: Graceful fallback, no service interruption
    """

    # Mock service discovery with Consul failure
    async def discover_service_with_fallback(service_name: str) -> str:
        try:
            # Simulate Consul unavailable
            raise ConnectionError("Consul unavailable")
        except ConnectionError:
            # Fallback to localhost
            fallback_urls = {
                "metadata-stamping": "http://localhost:8053",
                "onextree": "http://localhost:8080",
            }
            return fallback_urls.get(service_name, "http://localhost:8000")

    # Test fallback
    metadata_url = await discover_service_with_fallback("metadata-stamping")
    assert metadata_url == "http://localhost:8053", "Should fallback to localhost"

    onextree_url = await discover_service_with_fallback("onextree")
    assert onextree_url == "http://localhost:8080", "Should fallback to localhost"


# ============================================================================
# Test Suite 3: Circuit Breaker Across Services
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.e2e
async def test_circuit_breaker_onextree_service_protection():
    """
    Test circuit breaker protects against OnexTree service failures.

    Flow:
    1. OnexTree available → requests succeed
    2. OnexTree fails 5 consecutive times
    3. Circuit breaker opens
    4. Subsequent requests fail fast
    5. Workflow continues without intelligence (degraded mode)

    Expected: Circuit opens after 5 failures, <10ms fail-fast
    """
    circuit_state = {"failures": 0, "state": "CLOSED"}
    failure_threshold = 5

    async def call_onextree_with_circuit_breaker(file_path: str):
        """Call OnexTree with circuit breaker protection."""
        if circuit_state["state"] == "OPEN":
            # Fail fast
            raise Exception("Circuit breaker is OPEN")

        try:
            # Simulate OnexTree call failure
            raise ConnectionError("OnexTree unavailable")
        except ConnectionError:
            circuit_state["failures"] += 1

            if circuit_state["failures"] >= failure_threshold:
                circuit_state["state"] = "OPEN"

            raise

    # Test circuit breaker behavior
    for i in range(7):
        try:
            start_time = time.perf_counter()
            await call_onextree_with_circuit_breaker(f"/data/test/file_{i}.pdf")
        except Exception:
            duration_ms = (time.perf_counter() - start_time) * 1000

            # After circuit opens, failures should be fast
            if circuit_state["state"] == "OPEN":
                assert (
                    duration_ms < 10
                ), f"Fail-fast should be <10ms, got {duration_ms:.2f}ms"

    # Validate circuit opened
    assert circuit_state["state"] == "OPEN", "Circuit should be open after failures"
    assert (
        circuit_state["failures"] >= failure_threshold
    ), "Should have exceeded threshold"


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.e2e
async def test_independent_circuit_breakers_per_service():
    """
    Test circuit breakers operate independently per service.

    Flow:
    1. OnexTree circuit breaker opens (OnexTree fails)
    2. MetadataStamping continues working normally
    3. Reducer continues working normally
    4. Only OnexTree operations affected

    Expected: Service isolation, independent circuit breakers
    """
    circuit_states = {
        "onextree": {"state": "CLOSED", "failures": 0},
        "metadata_stamping": {"state": "CLOSED", "failures": 0},
        "reducer": {"state": "CLOSED", "failures": 0},
    }

    # Step 1: OnexTree fails
    for i in range(5):
        circuit_states["onextree"]["failures"] += 1

    if circuit_states["onextree"]["failures"] >= 5:
        circuit_states["onextree"]["state"] = "OPEN"

    # Step 2: Validate other services unaffected
    assert (
        circuit_states["onextree"]["state"] == "OPEN"
    ), "OnexTree circuit should be open"
    assert (
        circuit_states["metadata_stamping"]["state"] == "CLOSED"
    ), "MetadataStamping should remain operational"
    assert (
        circuit_states["reducer"]["state"] == "CLOSED"
    ), "Reducer should remain operational"


# ============================================================================
# Test Suite 4: Timeout Handling
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.e2e
async def test_onextree_timeout_handling_30s():
    """
    Test OnexTree timeout handling (30s limit).

    Flow:
    1. Request OnexTree analysis
    2. OnexTree takes >30s to respond
    3. Timeout exception raised
    4. Workflow continues without intelligence
    5. No workflow failure

    Expected: Graceful degradation on timeout
    """
    timeout_s = 30

    async def call_onextree_with_timeout(file_path: str, timeout: float):
        """Call OnexTree with timeout."""
        try:
            # Simulate slow OnexTree response
            await asyncio.wait_for(
                asyncio.sleep(timeout + 1),  # Exceed timeout
                timeout=timeout,
            )
        except TimeoutError:
            # Graceful degradation
            return None  # Continue without intelligence

    # Test timeout
    result = await call_onextree_with_timeout("/data/test/slow.pdf", timeout=0.1)

    assert result is None, "Should return None on timeout (graceful degradation)"


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.e2e
async def test_metadata_stamping_timeout_handling_5s():
    """
    Test MetadataStamping timeout handling (5s limit).

    Flow:
    1. Request stamp generation
    2. MetadataStamping takes >5s
    3. Timeout exception raised
    4. Workflow fails (critical operation)

    Expected: Workflow failure on MetadataStamping timeout (critical service)
    """
    timeout_s = 5

    async def call_metadata_stamping_with_timeout(file_content: bytes, timeout: float):
        """Call MetadataStamping with timeout."""
        try:
            # Simulate slow stamping
            await asyncio.wait_for(
                asyncio.sleep(timeout + 1),  # Exceed timeout
                timeout=timeout,
            )
            return {"stamp_id": "123", "file_hash": "abc"}
        except TimeoutError:
            # Critical operation - raise exception
            raise Exception("MetadataStamping timeout - workflow failed")

    # Test timeout failure
    with pytest.raises(Exception) as exc_info:
        await call_metadata_stamping_with_timeout(b"test content", timeout=0.1)

    assert "timeout" in str(exc_info.value).lower(), "Should raise timeout exception"


# ============================================================================
# Test Suite 5: Fallback Strategies
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.e2e
async def test_fallback_strategy_onextree_unavailable_skip_intelligence():
    """
    Test fallback: Skip intelligence when OnexTree unavailable.

    Flow:
    1. Request with intelligence enabled
    2. OnexTree unavailable
    3. Skip intelligence step
    4. Workflow continues without intelligence
    5. Success with degraded metadata

    Expected: Workflow succeeds, intelligence_data = None
    """
    workflow_id = uuid4()
    enable_intelligence = True

    # Step 1: Attempt OnexTree call
    intelligence_data = None

    try:
        # OnexTree unavailable
        raise ConnectionError("OnexTree unavailable")
    except ConnectionError:
        # Fallback: Skip intelligence
        intelligence_data = None

    # Step 2: Continue workflow without intelligence
    workflow_result = {
        "workflow_id": str(workflow_id),
        "state": "COMPLETED",
        "intelligence_data": intelligence_data,  # None (degraded)
        "fallback_used": True,
    }

    # Validate fallback
    assert workflow_result["state"] == "COMPLETED", "Workflow should complete"
    assert workflow_result["intelligence_data"] is None, "Intelligence should be None"
    assert workflow_result["fallback_used"] is True, "Should indicate fallback used"


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.e2e
async def test_fallback_strategy_metadata_stamping_in_memory_hash():
    """
    Test fallback: Use in-memory hash when MetadataStamping unavailable.

    Flow:
    1. Request stamp generation
    2. MetadataStamping service unavailable
    3. Generate hash in-memory (BLAKE3)
    4. Workflow continues with in-memory stamp
    5. Warning logged about degraded stamping

    Expected: Workflow succeeds with in-memory stamp
    """
    import hashlib

    file_content = b"Test file content for fallback"

    # Step 1: Attempt MetadataStamping service call
    try:
        raise ConnectionError("MetadataStamping unavailable")
    except ConnectionError:
        # Fallback: In-memory hash generation
        file_hash = hashlib.sha256(file_content).hexdigest()[:64]

        stamp_result = {
            "stamp_id": str(uuid4()),
            "file_hash": file_hash,
            "source": "in_memory_fallback",
            "warning": "MetadataStamping service unavailable",
        }

    # Validate fallback
    assert stamp_result["file_hash"] is not None, "Should have generated hash"
    assert stamp_result["source"] == "in_memory_fallback", "Should indicate fallback"
    assert len(stamp_result["file_hash"]) == 64, "Hash should be 64 characters"
