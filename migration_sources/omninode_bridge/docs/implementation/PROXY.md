# Proxy Implementation Guide

## Overview

This guide provides comprehensive implementation of the ToolCapture Proxy service, which provides intelligent service-to-service communication with automatic optimization and intelligence capture. Based on patterns from `omnibase_3`, this proxy transforms simple request forwarding into intelligent communication orchestration.

## ToolCapture Proxy Implementation

### Project Structure
```
services/tool-capture-proxy/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── proxy.py
│   │   ├── routing.py
│   │   ├── caching.py
│   │   └── intelligence.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── proxy_engine.py
│   │   ├── route_optimizer.py
│   │   ├── cache_engine.py
│   │   ├── circuit_breaker.py
│   │   └── intelligence_capture.py
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── correlation.py
│   │   ├── performance.py
│   │   └── security.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── proxy.py
│   │   ├── admin.py
│   │   └── health.py
│   └── config/
│       ├── __init__.py
│       └── settings.py
├── requirements.txt
├── Dockerfile
└── tests/
    ├── __init__.py
    ├── test_proxy.py
    ├── test_routing.py
    ├── test_caching.py
    └── test_integration.py
```

### Core Models

#### Proxy Models
```python
# app/models/proxy.py
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from enum import Enum
import uuid

class ProxyMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"

class ProxyRequest(BaseModel):
    method: ProxyMethod
    path: str
    headers: Dict[str, str] = {}
    query_params: Dict[str, str] = {}
    body: Optional[Union[str, bytes, Dict[str, Any]]] = None
    target_service: Optional[str] = None
    user_context: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ProxyResponse(BaseModel):
    status_code: int
    headers: Dict[str, str] = {}
    body: Optional[Union[str, bytes, Dict[str, Any]]] = None
    performance_metrics: Dict[str, Any] = {}
    cache_metadata: Dict[str, Any] = {}
    routing_metadata: Dict[str, Any] = {}
    intelligence_metadata: Dict[str, Any] = {}

class RequestProfile(BaseModel):
    request_hash: str
    method: str
    path: str
    payload_size: int
    user_context: Optional[str] = None
    estimated_complexity: str = "medium"  # low, medium, high
    cache_eligibility: bool = False
    security_level: str = "standard"  # low, standard, high, critical

class PerformanceMetrics(BaseModel):
    request_id: str
    correlation_id: str
    proxy_overhead_ms: float
    target_response_time_ms: float
    total_response_time_ms: float
    cache_hit: bool = False
    route_optimization_applied: bool = False
    circuit_breaker_triggered: bool = False
    intelligence_processing_ms: float = 0.0
```

#### Routing Models
```python
# app/models/routing.py
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

class ServiceHealth(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class RoutingAlgorithm(str, Enum):
    ROUND_ROBIN = "round_robin"
    PERFORMANCE_BASED = "performance_based"
    LOAD_BALANCING = "load_balancing"
    GEOGRAPHIC_PROXIMITY = "geographic_proximity"
    INTELLIGENT_ADAPTIVE = "intelligent_adaptive"

class ServiceInstance(BaseModel):
    service_id: str
    name: str
    address: str
    port: int
    health: ServiceHealth = ServiceHealth.UNKNOWN
    metadata: Dict[str, Any] = {}
    performance_metrics: Dict[str, float] = {}
    load_metrics: Dict[str, float] = {}
    last_health_check: datetime = Field(default_factory=datetime.utcnow)

class RoutingDecision(BaseModel):
    selected_service: ServiceInstance
    algorithm_used: RoutingAlgorithm
    decision_factors: Dict[str, float] = {}
    confidence_score: float = Field(ge=0.0, le=1.0)
    alternative_services: List[ServiceInstance] = []
    decision_time_ms: float

class RouteOptimization(BaseModel):
    optimization_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_service: str
    target_service_pattern: str
    optimization_type: str
    optimization_data: Dict[str, Any]
    expected_improvement: Dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)
    applied_at: Optional[datetime] = None
    effectiveness_score: Optional[float] = None
```

#### Caching Models
```python
# app/models/caching.py
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from enum import Enum

class CacheStrategy(str, Enum):
    CONTENT_BASED = "content_based"
    USER_CONTEXT = "user_context"
    TEMPORAL_PATTERNS = "temporal_patterns"
    INTELLIGENT_ADAPTIVE = "intelligent_adaptive"

class CacheRule(BaseModel):
    pattern: str
    ttl: int  # seconds
    vary_by: List[str] = []
    strategy: CacheStrategy = CacheStrategy.CONTENT_BASED
    conditions: Dict[str, Any] = {}
    priority: int = 0

class CacheEntry(BaseModel):
    cache_key: str
    content: Any
    content_type: str
    headers: Dict[str, str] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime
    access_count: int = 0
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    cache_score: float = Field(ge=0.0, le=1.0, default=1.0)

class CacheAnalytics(BaseModel):
    cache_key: str
    hit_count: int = 0
    miss_count: int = 0
    hit_ratio: float = 0.0
    average_response_time_saved_ms: float = 0.0
    total_bytes_served: int = 0
    first_cached: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
```

### Core Service Implementation

#### Proxy Engine
```python
# app/services/proxy_engine.py
from fastapi import HTTPException
from typing import Dict, Any, Optional
import asyncio
import time
import hashlib
from datetime import datetime

from ..models.proxy import ProxyRequest, ProxyResponse, RequestProfile, PerformanceMetrics
from .route_optimizer import RouteOptimizer
from .cache_engine import IntelligentCacheEngine
from .circuit_breaker import CircuitBreakerManager
from .intelligence_capture import IntelligenceCaptureService

class ProxyEngine:
    """
    Core proxy engine with intelligent routing, caching, and optimization.
    """

    def __init__(self):
        self.route_optimizer = RouteOptimizer()
        self.cache_engine = IntelligentCacheEngine()
        self.circuit_breaker = CircuitBreakerManager()
        self.intelligence_capture = IntelligenceCaptureService()
        self.http_client = None
        self.performance_tracker = {}

    async def initialize(self):
        """
        Initialize proxy engine and all subsystems.
        """
        await self.route_optimizer.initialize()
        await self.cache_engine.initialize()
        await self.circuit_breaker.initialize()
        await self.intelligence_capture.initialize()

        # Initialize HTTP client with optimized settings
        import aiohttp
        connector = aiohttp.TCPConnector(
            limit=1000,
            limit_per_host=100,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=60,
            enable_cleanup_closed=True
        )

        timeout = aiohttp.ClientTimeout(total=30, connect=5)
        self.http_client = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )

    async def process_request(self, request: ProxyRequest) -> ProxyResponse:
        """
        Main proxy request processing with full intelligence pipeline.
        """
        start_time = time.time()

        try:
            # Generate request profile for intelligence
            request_profile = await self._create_request_profile(request)

            # Capture pre-request intelligence
            await self.intelligence_capture.capture_pre_request_intelligence(
                request, request_profile
            )

            # Check intelligent cache first
            cached_response = await self.cache_engine.check_cache(request, request_profile)
            if cached_response:
                # Cache hit - capture cache intelligence and return
                performance_metrics = PerformanceMetrics(
                    request_id=request.correlation_id,
                    correlation_id=request.correlation_id,
                    proxy_overhead_ms=(time.time() - start_time) * 1000,
                    target_response_time_ms=0.0,
                    total_response_time_ms=(time.time() - start_time) * 1000,
                    cache_hit=True
                )

                await self.intelligence_capture.capture_cache_hit_intelligence(
                    request, cached_response, performance_metrics
                )

                cached_response.performance_metrics = performance_metrics.dict()
                return cached_response

            # Intelligent routing
            target_service = await self.route_optimizer.select_optimal_target(
                request, request_profile
            )

            # Execute request with circuit breaker protection
            response = await self._execute_proxied_request(
                request, target_service, request_profile, start_time
            )

            # Post-request intelligence capture
            await self.intelligence_capture.capture_post_request_intelligence(
                request, response, target_service, request_profile
            )

            # Consider caching response
            await self.cache_engine.consider_caching(
                request, response, request_profile
            )

            return response

        except Exception as e:
            # Error handling with intelligence capture
            error_response = await self._handle_proxy_error(
                e, request, request_profile, start_time
            )
            return error_response

    async def _execute_proxied_request(self, request: ProxyRequest, target_service: Dict[str, Any], request_profile: RequestProfile, start_time: float) -> ProxyResponse:
        """
        Execute the actual proxied request with circuit breaker protection.
        """
        service_id = target_service["service_id"]

        # Circuit breaker execution
        try:
            response = await self.circuit_breaker.execute_with_protection(
                service_id,
                lambda: self._forward_request_to_service(
                    request, target_service, request_profile
                )
            )

            # Calculate performance metrics
            proxy_end_time = time.time()
            performance_metrics = PerformanceMetrics(
                request_id=request.correlation_id,
                correlation_id=request.correlation_id,
                proxy_overhead_ms=response.get("proxy_overhead_ms", 0.0),
                target_response_time_ms=response.get("target_response_time_ms", 0.0),
                total_response_time_ms=(proxy_end_time - start_time) * 1000,
                cache_hit=False,
                route_optimization_applied=True,
                circuit_breaker_triggered=False
            )

            # Create proxy response
            proxy_response = ProxyResponse(
                status_code=response["status_code"],
                headers=response["headers"],
                body=response["body"],
                performance_metrics=performance_metrics.dict(),
                routing_metadata={
                    "target_service": target_service,
                    "routing_algorithm": response.get("routing_algorithm"),
                    "routing_decision_time_ms": response.get("routing_decision_time_ms", 0.0)
                }
            )

            return proxy_response

        except CircuitBreakerOpenException as e:
            # Handle circuit breaker open scenario
            fallback_response = await self._handle_circuit_breaker_fallback(
                request, service_id, request_profile
            )

            performance_metrics = PerformanceMetrics(
                request_id=request.correlation_id,
                correlation_id=request.correlation_id,
                proxy_overhead_ms=(time.time() - start_time) * 1000,
                target_response_time_ms=0.0,
                total_response_time_ms=(time.time() - start_time) * 1000,
                cache_hit=False,
                circuit_breaker_triggered=True
            )

            fallback_response.performance_metrics = performance_metrics.dict()
            return fallback_response

    async def _forward_request_to_service(self, request: ProxyRequest, target_service: Dict[str, Any], request_profile: RequestProfile) -> Dict[str, Any]:
        """
        Forward request to target service and measure performance.
        """
        target_url = f"http://{target_service['address']}:{target_service['port']}{request.path}"

        # Add query parameters
        if request.query_params:
            query_string = "&".join([f"{k}={v}" for k, v in request.query_params.items()])
            target_url += f"?{query_string}"

        # Prepare headers
        headers = request.headers.copy()
        headers["X-Proxy-Correlation-ID"] = request.correlation_id
        headers["X-Proxy-Request-ID"] = request.correlation_id
        headers["X-Forwarded-For"] = "omninode-bridge-proxy"

        # Performance timing
        request_start = time.time()

        try:
            # Execute HTTP request
            async with self.http_client.request(
                method=request.method.value,
                url=target_url,
                headers=headers,
                data=request.body if request.body else None,
                allow_redirects=False
            ) as response:
                response_body = await response.read()
                request_end = time.time()

                # Process response
                response_headers = dict(response.headers)

                # Remove hop-by-hop headers
                hop_by_hop_headers = {
                    'connection', 'keep-alive', 'proxy-authenticate',
                    'proxy-authorization', 'te', 'trailers', 'transfer-encoding', 'upgrade'
                }
                for header in hop_by_hop_headers:
                    response_headers.pop(header, None)

                return {
                    "status_code": response.status,
                    "headers": response_headers,
                    "body": response_body,
                    "target_response_time_ms": (request_end - request_start) * 1000,
                    "proxy_overhead_ms": 0.0,  # Calculated externally
                    "target_service_id": target_service["service_id"]
                }

        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Gateway timeout")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Bad gateway: {str(e)}")

    async def _create_request_profile(self, request: ProxyRequest) -> RequestProfile:
        """
        Create comprehensive request profile for intelligence analysis.
        """
        # Calculate request hash
        request_content = f"{request.method}:{request.path}:{request.body or ''}"
        request_hash = hashlib.sha256(request_content.encode()).hexdigest()

        # Estimate payload size
        payload_size = 0
        if request.body:
            if isinstance(request.body, str):
                payload_size = len(request.body.encode())
            elif isinstance(request.body, bytes):
                payload_size = len(request.body)
            elif isinstance(request.body, dict):
                payload_size = len(str(request.body).encode())

        # Estimate complexity based on request characteristics
        complexity = "low"
        if payload_size > 1024 * 1024:  # 1MB
            complexity = "high"
        elif payload_size > 10 * 1024 or request.method in ["POST", "PUT", "PATCH"]:
            complexity = "medium"

        # Check cache eligibility
        cache_eligible = (
            request.method == "GET" and
            payload_size < 100 * 1024 and  # Less than 100KB
            "no-cache" not in request.headers.get("cache-control", "").lower()
        )

        # Determine security level
        security_level = "standard"
        if "authorization" in request.headers:
            security_level = "high"
        if any(sensitive in request.path.lower() for sensitive in ["admin", "auth", "token", "secret"]):
            security_level = "critical"

        return RequestProfile(
            request_hash=request_hash,
            method=request.method.value,
            path=request.path,
            payload_size=payload_size,
            user_context=request.user_context,
            estimated_complexity=complexity,
            cache_eligibility=cache_eligible,
            security_level=security_level
        )

    async def _handle_circuit_breaker_fallback(self, request: ProxyRequest, service_id: str, request_profile: RequestProfile) -> ProxyResponse:
        """
        Handle circuit breaker fallback scenarios.
        """
        # Try to find alternative service
        alternative_services = await self.route_optimizer.get_alternative_services(
            service_id, request_profile
        )

        if alternative_services:
            # Try alternative service
            try:
                return await self._execute_proxied_request(
                    request, alternative_services[0], request_profile, time.time()
                )
            except Exception:
                pass

        # Check for cached fallback response
        fallback_response = await self.cache_engine.get_fallback_response(
            request, request_profile
        )

        if fallback_response:
            return fallback_response

        # Generate error response
        return ProxyResponse(
            status_code=503,
            headers={"Content-Type": "application/json"},
            body={
                "error": "Service temporarily unavailable",
                "service_id": service_id,
                "correlation_id": request.correlation_id,
                "timestamp": datetime.utcnow().isoformat()
            },
            performance_metrics={},
            routing_metadata={"circuit_breaker_triggered": True}
        )

    async def _handle_proxy_error(self, error: Exception, request: ProxyRequest, request_profile: RequestProfile, start_time: float) -> ProxyResponse:
        """
        Handle proxy errors with intelligence capture.
        """
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "request_path": request.path,
            "request_method": request.method.value,
            "correlation_id": request.correlation_id
        }

        # Capture error intelligence
        await self.intelligence_capture.capture_error_intelligence(
            request, error_data, request_profile
        )

        # Generate error response
        if isinstance(error, HTTPException):
            status_code = error.status_code
            error_detail = error.detail
        else:
            status_code = 500
            error_detail = "Internal proxy error"

        performance_metrics = PerformanceMetrics(
            request_id=request.correlation_id,
            correlation_id=request.correlation_id,
            proxy_overhead_ms=(time.time() - start_time) * 1000,
            target_response_time_ms=0.0,
            total_response_time_ms=(time.time() - start_time) * 1000,
            cache_hit=False
        )

        return ProxyResponse(
            status_code=status_code,
            headers={"Content-Type": "application/json"},
            body={
                "error": error_detail,
                "correlation_id": request.correlation_id,
                "timestamp": datetime.utcnow().isoformat()
            },
            performance_metrics=performance_metrics.dict()
        )
```

#### Route Optimizer
```python
# app/services/route_optimizer.py
from typing import Dict, Any, List, Optional
import asyncio
import time
import statistics
from datetime import datetime, timedelta

from ..models.routing import ServiceInstance, RoutingDecision, RoutingAlgorithm, ServiceHealth
from ..models.proxy import RequestProfile

class RouteOptimizer:
    """
    Intelligent routing optimization with multiple algorithms and machine learning.
    """

    def __init__(self):
        self.consul_client = None
        self.service_registry = {}
        self.performance_history = {}
        self.routing_analytics = {}
        self.algorithm_weights = {
            RoutingAlgorithm.PERFORMANCE_BASED: 0.4,
            RoutingAlgorithm.LOAD_BALANCING: 0.3,
            RoutingAlgorithm.GEOGRAPHIC_PROXIMITY: 0.2,
            RoutingAlgorithm.INTELLIGENT_ADAPTIVE: 0.1
        }

    async def initialize(self):
        """
        Initialize route optimizer with service discovery.
        """
        # Initialize Consul client
        import consul
        self.consul_client = consul.Consul()

        # Start background service discovery
        asyncio.create_task(self._periodic_service_discovery())
        asyncio.create_task(self._periodic_performance_analysis())

    async def select_optimal_target(self, request, request_profile: RequestProfile) -> Dict[str, Any]:
        """
        Select optimal target service using intelligent routing algorithms.
        """
        start_time = time.time()

        # Determine target service type
        target_service_type = self._extract_service_type_from_request(request)

        # Get available services
        available_services = await self._get_available_services(target_service_type)

        if not available_services:
            raise Exception(f"No healthy services available for type: {target_service_type}")

        # Apply intelligent routing algorithms
        routing_scores = {}

        for service in available_services:
            scores = await self._calculate_service_scores(service, request_profile)
            routing_scores[service.service_id] = scores

        # Select best service based on weighted scores
        best_service = await self._select_best_service(routing_scores, available_services)

        # Record routing decision
        decision_time = (time.time() - start_time) * 1000
        routing_decision = RoutingDecision(
            selected_service=best_service,
            algorithm_used=RoutingAlgorithm.INTELLIGENT_ADAPTIVE,
            decision_factors=routing_scores[best_service.service_id],
            confidence_score=routing_scores[best_service.service_id]["total_score"],
            alternative_services=[s for s in available_services if s.service_id != best_service.service_id][:3],
            decision_time_ms=decision_time
        )

        await self._record_routing_decision(routing_decision, request_profile)

        return {
            "service_id": best_service.service_id,
            "address": best_service.address,
            "port": best_service.port,
            "routing_decision": routing_decision.dict()
        }

    async def _calculate_service_scores(self, service: ServiceInstance, request_profile: RequestProfile) -> Dict[str, float]:
        """
        Calculate comprehensive service scores using multiple factors.
        """
        scores = {}

        # Performance-based scoring
        scores["performance"] = await self._calculate_performance_score(service, request_profile)

        # Load balancing scoring
        scores["load_balancing"] = await self._calculate_load_score(service)

        # Health scoring
        scores["health"] = self._calculate_health_score(service)

        # Compatibility scoring
        scores["compatibility"] = await self._calculate_compatibility_score(service, request_profile)

        # Historical success rate scoring
        scores["success_rate"] = await self._calculate_success_rate_score(service, request_profile)

        # Calculate weighted total score
        total_score = (
            scores["performance"] * 0.3 +
            scores["load_balancing"] * 0.25 +
            scores["health"] * 0.2 +
            scores["compatibility"] * 0.15 +
            scores["success_rate"] * 0.1
        )

        scores["total_score"] = total_score
        return scores

    async def _calculate_performance_score(self, service: ServiceInstance, request_profile: RequestProfile) -> float:
        """
        Calculate performance score based on historical data and predicted performance.
        """
        service_id = service.service_id

        # Get historical performance data
        history = self.performance_history.get(service_id, {})

        if not history:
            return 0.7  # Default score for new services

        # Calculate average response time for similar requests
        similar_requests = [
            perf for perf in history.get("requests", [])
            if (perf.get("complexity", "medium") == request_profile.estimated_complexity and
                perf.get("method", "GET") == request_profile.method)
        ]

        if not similar_requests:
            similar_requests = history.get("requests", [])

        if not similar_requests:
            return 0.7

        # Calculate performance metrics
        avg_response_time = statistics.mean([r["response_time_ms"] for r in similar_requests[-20:]])
        p95_response_time = statistics.quantiles([r["response_time_ms"] for r in similar_requests[-20:]], n=20)[18] if len(similar_requests) >= 20 else avg_response_time

        # Normalize performance score (lower response time = higher score)
        # Assume baseline of 100ms for score of 1.0, and 1000ms for score of 0.1
        performance_score = max(0.1, min(1.0, 1.0 - (avg_response_time - 100) / 900))

        # Adjust for consistency (lower P95 relative to average = higher score)
        consistency_factor = min(1.0, avg_response_time / p95_response_time) if p95_response_time > 0 else 1.0
        performance_score *= consistency_factor

        return performance_score

    async def _calculate_load_score(self, service: ServiceInstance) -> float:
        """
        Calculate load balancing score based on current service load.
        """
        load_metrics = service.load_metrics

        # CPU utilization scoring
        cpu_util = load_metrics.get("cpu_utilization", 0.5)
        cpu_score = max(0.0, 1.0 - cpu_util)

        # Memory utilization scoring
        memory_util = load_metrics.get("memory_utilization", 0.5)
        memory_score = max(0.0, 1.0 - memory_util)

        # Active connections scoring
        active_connections = load_metrics.get("active_connections", 0)
        max_connections = load_metrics.get("max_connections", 1000)
        connection_score = max(0.0, 1.0 - (active_connections / max_connections))

        # Request queue depth scoring
        queue_depth = load_metrics.get("request_queue_depth", 0)
        max_queue = load_metrics.get("max_queue_depth", 100)
        queue_score = max(0.0, 1.0 - (queue_depth / max_queue))

        # Weighted load score
        load_score = (
            cpu_score * 0.3 +
            memory_score * 0.25 +
            connection_score * 0.25 +
            queue_score * 0.2
        )

        return load_score

    def _calculate_health_score(self, service: ServiceInstance) -> float:
        """
        Calculate health score based on service health status.
        """
        health_scores = {
            ServiceHealth.HEALTHY: 1.0,
            ServiceHealth.DEGRADED: 0.6,
            ServiceHealth.UNHEALTHY: 0.1,
            ServiceHealth.UNKNOWN: 0.5
        }

        base_score = health_scores.get(service.health, 0.5)

        # Adjust based on health check recency
        now = datetime.utcnow()
        time_since_check = (now - service.last_health_check).total_seconds()

        # Reduce score if health check is stale
        if time_since_check > 300:  # 5 minutes
            freshness_factor = max(0.1, 1.0 - (time_since_check - 300) / 3600)  # Decay over 1 hour
            base_score *= freshness_factor

        return base_score

    async def _calculate_compatibility_score(self, service: ServiceInstance, request_profile: RequestProfile) -> float:
        """
        Calculate compatibility score based on service capabilities and request requirements.
        """
        service_metadata = service.metadata

        # Check for specific capability requirements
        compatibility_score = 1.0

        # Security level compatibility
        service_security_level = service_metadata.get("security_level", "standard")
        if request_profile.security_level == "critical" and service_security_level != "critical":
            compatibility_score *= 0.5
        elif request_profile.security_level == "high" and service_security_level == "low":
            compatibility_score *= 0.7

        # Payload size compatibility
        max_payload_size = service_metadata.get("max_payload_size", float('inf'))
        if request_profile.payload_size > max_payload_size:
            compatibility_score = 0.0  # Cannot handle request

        # Method support
        supported_methods = service_metadata.get("supported_methods", ["GET", "POST", "PUT", "DELETE"])
        if request_profile.method not in supported_methods:
            compatibility_score = 0.0  # Cannot handle method

        # Special feature support
        if request_profile.estimated_complexity == "high":
            supports_complex = service_metadata.get("supports_complex_requests", True)
            if not supports_complex:
                compatibility_score *= 0.3

        return compatibility_score

    async def _calculate_success_rate_score(self, service: ServiceInstance, request_profile: RequestProfile) -> float:
        """
        Calculate success rate score based on historical success rates.
        """
        service_id = service.service_id
        history = self.performance_history.get(service_id, {})

        if not history:
            return 0.8  # Default for new services

        # Get recent requests
        recent_requests = history.get("requests", [])[-100:]  # Last 100 requests

        if not recent_requests:
            return 0.8

        # Calculate overall success rate
        successful_requests = [r for r in recent_requests if r.get("success", False)]
        overall_success_rate = len(successful_requests) / len(recent_requests)

        # Calculate success rate for similar requests
        similar_requests = [
            r for r in recent_requests
            if (r.get("method", "GET") == request_profile.method and
                r.get("complexity", "medium") == request_profile.estimated_complexity)
        ]

        if similar_requests:
            similar_successful = [r for r in similar_requests if r.get("success", False)]
            similar_success_rate = len(similar_successful) / len(similar_requests)
            # Weight similar requests more heavily
            final_score = (similar_success_rate * 0.7) + (overall_success_rate * 0.3)
        else:
            final_score = overall_success_rate

        return final_score

    async def _select_best_service(self, routing_scores: Dict[str, Dict[str, float]], available_services: List[ServiceInstance]) -> ServiceInstance:
        """
        Select the best service based on calculated scores.
        """
        # Find service with highest total score
        best_service_id = max(routing_scores.keys(), key=lambda k: routing_scores[k]["total_score"])

        # Get the service instance
        best_service = next(s for s in available_services if s.service_id == best_service_id)

        return best_service

    async def _get_available_services(self, service_type: str) -> List[ServiceInstance]:
        """
        Get available services from service registry.
        """
        # Get services from Consul
        try:
            _, services = self.consul_client.health.service(service_type, passing=True)

            service_instances = []
            for service_data in services:
                service = service_data["Service"]
                health_status = self._determine_health_status(service_data["Checks"])

                instance = ServiceInstance(
                    service_id=f"{service['Service']}_{service['ID']}",
                    name=service["Service"],
                    address=service["Address"],
                    port=service["Port"],
                    health=health_status,
                    metadata=service.get("Meta", {}),
                    performance_metrics=self.performance_history.get(f"{service['Service']}_{service['ID']}", {}).get("current_metrics", {}),
                    load_metrics=self._get_service_load_metrics(service)
                )

                service_instances.append(instance)

            return service_instances

        except Exception as e:
            # Fallback to cached registry
            return list(self.service_registry.get(service_type, {}).values())

    def _extract_service_type_from_request(self, request) -> str:
        """
        Extract target service type from request path or headers.
        """
        # Extract from path patterns
        path_parts = request.path.strip("/").split("/")

        if len(path_parts) > 0:
            # Common patterns: /api/v1/service_name/...
            if path_parts[0] == "api" and len(path_parts) > 2:
                return path_parts[2]
            elif len(path_parts) > 0:
                return path_parts[0]

        # Extract from headers
        target_service = request.headers.get("X-Target-Service")
        if target_service:
            return target_service

        # Default service type
        return "default"

    def _determine_health_status(self, checks: List[Dict[str, Any]]) -> ServiceHealth:
        """
        Determine overall health status from Consul health checks.
        """
        if not checks:
            return ServiceHealth.UNKNOWN

        statuses = [check["Status"] for check in checks]

        if all(status == "passing" for status in statuses):
            return ServiceHealth.HEALTHY
        elif any(status == "critical" for status in statuses):
            return ServiceHealth.UNHEALTHY
        elif any(status == "warning" for status in statuses):
            return ServiceHealth.DEGRADED
        else:
            return ServiceHealth.UNKNOWN

    async def _periodic_service_discovery(self):
        """
        Periodic service discovery and registry updates.
        """
        while True:
            try:
                # Update service registry from Consul
                await self._update_service_registry()
                await asyncio.sleep(30)  # Update every 30 seconds
            except Exception as e:
                await asyncio.sleep(60)  # Retry after 1 minute on error

    async def _periodic_performance_analysis(self):
        """
        Periodic performance analysis and optimization.
        """
        while True:
            try:
                # Analyze routing performance
                await self._analyze_routing_performance()

                # Update algorithm weights based on effectiveness
                await self._update_algorithm_weights()

                await asyncio.sleep(300)  # Analyze every 5 minutes
            except Exception as e:
                await asyncio.sleep(600)  # Retry after 10 minutes on error
```

### Intelligent Caching Engine
```python
# app/services/cache_engine.py
from typing import Dict, Any, Optional, List
import json
import hashlib
import time
from datetime import datetime, timedelta

from ..models.caching import CacheRule, CacheEntry, CacheAnalytics, CacheStrategy
from ..models.proxy import ProxyRequest, ProxyResponse, RequestProfile

class IntelligentCacheEngine:
    """
    Intelligent caching engine with adaptive TTL and smart cache decisions.
    """

    def __init__(self):
        self.redis_client = None
        self.cache_rules = []
        self.cache_analytics = {}
        self.cache_intelligence = {}

    async def initialize(self):
        """
        Initialize cache engine with Redis and load cache rules.
        """
        import redis.asyncio as redis
        self.redis_client = redis.Redis.from_url("redis://localhost:6379", decode_responses=False)

        # Load default cache rules
        self.cache_rules = self._load_default_cache_rules()

        # Start background cache optimization
        import asyncio
        asyncio.create_task(self._periodic_cache_optimization())

    def _load_default_cache_rules(self) -> List[CacheRule]:
        """
        Load default cache rules for common patterns.
        """
        return [
            CacheRule(
                pattern="GET /api/v1/users/*",
                ttl=300,  # 5 minutes
                vary_by=["Authorization"],
                strategy=CacheStrategy.USER_CONTEXT,
                priority=10
            ),
            CacheRule(
                pattern="GET /api/v1/config/*",
                ttl=3600,  # 1 hour
                vary_by=[],
                strategy=CacheStrategy.CONTENT_BASED,
                priority=20
            ),
            CacheRule(
                pattern="POST /api/v1/query",
                ttl=60,  # 1 minute
                vary_by=["payload_hash"],
                strategy=CacheStrategy.CONTENT_BASED,
                conditions={"payload_size": {"max": 10240}},  # 10KB max
                priority=5
            ),
            CacheRule(
                pattern="GET /api/v1/reports/*",
                ttl=1800,  # 30 minutes
                vary_by=["Authorization", "date_range"],
                strategy=CacheStrategy.TEMPORAL_PATTERNS,
                priority=15
            )
        ]

    async def check_cache(self, request: ProxyRequest, request_profile: RequestProfile) -> Optional[ProxyResponse]:
        """
        Check cache for existing response with intelligent cache key generation.
        """
        if not request_profile.cache_eligibility:
            return None

        # Generate intelligent cache key
        cache_key = await self._generate_cache_key(request, request_profile)

        try:
            # Check Redis cache
            cached_data = await self.redis_client.get(cache_key)

            if cached_data:
                # Parse cached response
                cache_entry = json.loads(cached_data)

                # Validate cache entry freshness
                if await self._is_cache_entry_fresh(cache_entry, request_profile):
                    # Update cache analytics
                    await self._record_cache_hit(cache_key, request_profile)

                    # Convert to ProxyResponse
                    response = ProxyResponse(
                        status_code=cache_entry["status_code"],
                        headers=cache_entry["headers"],
                        body=cache_entry["body"],
                        cache_metadata={
                            "cache_hit": True,
                            "cache_key": cache_key,
                            "cached_at": cache_entry["cached_at"],
                            "ttl": cache_entry["ttl"]
                        }
                    )

                    return response
                else:
                    # Cache entry is stale, remove it
                    await self.redis_client.delete(cache_key)

            # Cache miss
            await self._record_cache_miss(cache_key, request_profile)
            return None

        except Exception as e:
            # Cache error, proceed without cache
            return None

    async def consider_caching(self, request: ProxyRequest, response: ProxyResponse, request_profile: RequestProfile) -> None:
        """
        Intelligently decide whether to cache the response.
        """
        if not await self._should_cache_response(request, response, request_profile):
            return

        # Generate cache key
        cache_key = await self._generate_cache_key(request, request_profile)

        # Determine optimal TTL
        optimal_ttl = await self._calculate_optimal_ttl(request, response, request_profile)

        # Create cache entry
        cache_entry = {
            "status_code": response.status_code,
            "headers": response.headers,
            "body": response.body,
            "cached_at": datetime.utcnow().isoformat(),
            "ttl": optimal_ttl,
            "cache_score": await self._calculate_cache_score(request, response, request_profile),
            "request_profile": request_profile.dict()
        }

        try:
            # Store in Redis with TTL
            await self.redis_client.setex(
                cache_key,
                optimal_ttl,
                json.dumps(cache_entry, default=str)
            )

            # Update cache analytics
            await self._record_cache_store(cache_key, cache_entry, request_profile)

        except Exception as e:
            # Cache storage failed, continue without caching
            pass

    async def _generate_cache_key(self, request: ProxyRequest, request_profile: RequestProfile) -> str:
        """
        Generate intelligent cache key based on request characteristics.
        """
        # Find matching cache rule
        cache_rule = await self._find_matching_cache_rule(request, request_profile)

        # Base key components
        key_components = [
            request.method.value,
            request.path
        ]

        # Add vary_by components from cache rule
        if cache_rule:
            for vary_key in cache_rule.vary_by:
                if vary_key == "Authorization":
                    auth_header = request.headers.get("Authorization", "")
                    # Hash the authorization header for privacy
                    auth_hash = hashlib.sha256(auth_header.encode()).hexdigest()[:16]
                    key_components.append(f"auth:{auth_hash}")

                elif vary_key == "payload_hash":
                    if request.body:
                        payload_hash = hashlib.sha256(str(request.body).encode()).hexdigest()[:16]
                        key_components.append(f"payload:{payload_hash}")

                elif vary_key == "date_range":
                    # Extract date range from query parameters
                    date_range = request.query_params.get("date_range", "default")
                    key_components.append(f"date:{date_range}")

                elif vary_key in request.headers:
                    header_value = request.headers[vary_key]
                    header_hash = hashlib.sha256(header_value.encode()).hexdigest()[:8]
                    key_components.append(f"{vary_key.lower()}:{header_hash}")

        # Add user context if available
        if request.user_context:
            user_hash = hashlib.sha256(request.user_context.encode()).hexdigest()[:12]
            key_components.append(f"user:{user_hash}")

        # Generate final cache key
        cache_key = "proxy:cache:" + ":".join(key_components)

        # Ensure cache key length is reasonable
        if len(cache_key) > 250:
            cache_key_hash = hashlib.sha256(cache_key.encode()).hexdigest()
            cache_key = f"proxy:cache:hash:{cache_key_hash}"

        return cache_key

    async def _find_matching_cache_rule(self, request: ProxyRequest, request_profile: RequestProfile) -> Optional[CacheRule]:
        """
        Find the best matching cache rule for the request.
        """
        matching_rules = []

        for rule in self.cache_rules:
            if await self._rule_matches_request(rule, request, request_profile):
                matching_rules.append(rule)

        if not matching_rules:
            return None

        # Return rule with highest priority
        return max(matching_rules, key=lambda r: r.priority)

    async def _rule_matches_request(self, rule: CacheRule, request: ProxyRequest, request_profile: RequestProfile) -> bool:
        """
        Check if a cache rule matches the current request.
        """
        # Simple pattern matching for now (could be enhanced with regex)
        rule_parts = rule.pattern.split()
        if len(rule_parts) >= 2:
            rule_method = rule_parts[0]
            rule_path = rule_parts[1]

            # Check method match
            if rule_method != "*" and rule_method != request.method.value:
                return False

            # Check path match (simple wildcard support)
            if rule_path.endswith("*"):
                path_prefix = rule_path[:-1]
                if not request.path.startswith(path_prefix):
                    return False
            elif rule_path != request.path:
                return False

        # Check conditions
        for condition_key, condition_value in rule.conditions.items():
            if condition_key == "payload_size":
                max_size = condition_value.get("max", float('inf'))
                if request_profile.payload_size > max_size:
                    return False

        return True

    async def _should_cache_response(self, request: ProxyRequest, response: ProxyResponse, request_profile: RequestProfile) -> bool:
        """
        Determine if response should be cached based on intelligent analysis.
        """
        # Don't cache errors
        if response.status_code >= 400:
            return False

        # Don't cache if explicitly told not to
        cache_control = response.headers.get("cache-control", "").lower()
        if "no-cache" in cache_control or "no-store" in cache_control:
            return False

        # Don't cache if request is not cache eligible
        if not request_profile.cache_eligibility:
            return False

        # Check if response size is reasonable for caching
        response_size = len(str(response.body)) if response.body else 0
        if response_size > 10 * 1024 * 1024:  # 10MB limit
            return False

        # Check content type
        content_type = response.headers.get("content-type", "").lower()
        cacheable_types = [
            "application/json",
            "text/html",
            "text/plain",
            "text/xml",
            "application/xml"
        ]

        if not any(ct in content_type for ct in cacheable_types):
            return False

        # Intelligent caching decision based on patterns
        cache_score = await self._calculate_cache_score(request, response, request_profile)
        return cache_score > 0.6

    async def _calculate_optimal_ttl(self, request: ProxyRequest, response: ProxyResponse, request_profile: RequestProfile) -> int:
        """
        Calculate optimal TTL based on content analysis and patterns.
        """
        # Base TTL from cache rules
        cache_rule = await self._find_matching_cache_rule(request, request_profile)
        base_ttl = cache_rule.ttl if cache_rule else 300  # 5 minutes default

        # Adjust based on response characteristics
        ttl_multiplier = 1.0

        # Content type adjustments
        content_type = response.headers.get("content-type", "").lower()
        if "json" in content_type:
            ttl_multiplier *= 0.8  # JSON typically changes more frequently
        elif "html" in content_type:
            ttl_multiplier *= 1.2  # HTML can be cached longer

        # Response size adjustments
        response_size = len(str(response.body)) if response.body else 0
        if response_size > 100 * 1024:  # 100KB
            ttl_multiplier *= 1.5  # Larger responses benefit from longer caching

        # Performance-based adjustments
        response_time = response.performance_metrics.get("target_response_time_ms", 0)
        if response_time > 1000:  # Slow responses
            ttl_multiplier *= 2.0  # Cache longer to avoid repeating slow operations

        # Historical pattern adjustments
        cache_analytics = self.cache_analytics.get(request.path, {})
        if cache_analytics:
            hit_ratio = cache_analytics.get("hit_ratio", 0.0)
            if hit_ratio > 0.8:
                ttl_multiplier *= 1.3  # High hit ratio suggests longer TTL is beneficial

        # Calculate final TTL
        optimal_ttl = int(base_ttl * ttl_multiplier)

        # Ensure reasonable bounds
        return max(60, min(7200, optimal_ttl))  # Between 1 minute and 2 hours

    async def _calculate_cache_score(self, request: ProxyRequest, response: ProxyResponse, request_profile: RequestProfile) -> float:
        """
        Calculate cache worthiness score (0.0 to 1.0).
        """
        score = 0.5  # Base score

        # Response time factor (slower responses benefit more from caching)
        response_time = response.performance_metrics.get("target_response_time_ms", 0)
        if response_time > 500:
            score += 0.2
        if response_time > 1000:
            score += 0.2

        # Content stability factor
        content_type = response.headers.get("content-type", "").lower()
        if "json" in content_type:
            score += 0.1
        if "image" in content_type or "css" in content_type or "javascript" in content_type:
            score += 0.3

        # Request frequency factor
        path_analytics = self.cache_analytics.get(request.path, {})
        if path_analytics:
            # Higher frequency paths benefit more from caching
            request_frequency = path_analytics.get("request_frequency", 0)
            if request_frequency > 10:  # More than 10 requests in recent period
                score += 0.2

        # Response size factor
        response_size = len(str(response.body)) if response.body else 0
        if response_size > 10 * 1024:  # 10KB
            score += 0.1
        if response_size > 100 * 1024:  # 100KB
            score += 0.1

        return min(1.0, max(0.0, score))

    async def _periodic_cache_optimization(self):
        """
        Periodic cache optimization and analytics update.
        """
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes

                # Analyze cache performance
                await self._analyze_cache_performance()

                # Optimize cache rules
                await self._optimize_cache_rules()

                # Clean up expired analytics
                await self._cleanup_cache_analytics()

            except Exception as e:
                pass  # Continue on error
```

This comprehensive proxy implementation guide provides the foundation for the intelligent ToolCapture Proxy that transforms basic service-to-service communication into optimized, intelligent routing with automatic learning and optimization capabilities.
