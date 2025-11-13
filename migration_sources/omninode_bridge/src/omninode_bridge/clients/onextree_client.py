"""AsyncOnexTreeClient for OnexTree Agent Intelligence integration.

Production-grade async client with resilience patterns:
    - Knowledge graph queries with confidence scoring
    - Intelligence data retrieval with multi-dimensional confidence
    - Tree navigation and pattern analysis
    - Response caching with TTL (default: 5 minutes)
    - Circuit breaker protection (via BaseServiceClient)
    - Retry logic with exponential backoff (via tenacity)
    - Timeout handling (configurable, default: 30s)
    - Graceful degradation with fallback results
    - Correlation ID propagation for distributed tracing
    - Comprehensive metrics collection

Resilience Features:
    - Circuit Breaker: Protects against cascading failures
        - States: CLOSED (normal), OPEN (failing), HALF_OPEN (testing recovery)
        - Configurable failure threshold and recovery timeout
        - Automatic recovery testing after timeout period

    - Retry Logic: Exponential backoff for transient failures
        - Configurable max retries (default: 3)
        - Retries on timeout and network errors
        - Does not retry on client errors (4xx)

    - Fallback Mechanism: Graceful degradation when service unavailable
        - Returns minimal intelligence result with degraded flag
        - Zero confidence score to indicate fallback state
        - Allows workflow to continue despite service outage

    - Confidence Scoring: Multi-dimensional quality assessment
        - Data Quality (35%): Completeness of intelligence data
        - Pattern Strength (30%): Number and quality of patterns
        - Coverage (20%): Context analysis coverage
        - Relationship Density (15%): Relationship graph richness
        - Overall: Weighted average of all dimensions
        - Levels: HIGH (0.8+), MEDIUM (0.5-0.8), LOW (0.2-0.5), MINIMAL (<0.2)

Example Usage:
    # Basic usage with confidence scoring
    async with AsyncOnexTreeClient() as client:
        result = await client.get_intelligence("authentication patterns")

        if result.confidence.overall >= 0.7:
            # High confidence - apply recommendations
            implement_patterns(result.intelligence)
        elif result.degraded:
            # Fallback result - proceed with caution
            logger.warning("Using degraded intelligence")
            proceed_with_basic_workflow()

    # Advanced usage with fallback control
    async with AsyncOnexTreeClient(timeout=5.0, max_retries=5) as client:
        result = await client.get_intelligence(
            context="API security patterns",
            enable_fallback=True,  # Graceful degradation
            use_cache=True,        # Use cached results
            correlation_id=uuid4()  # Request tracing
        )

        print(f"Confidence: {result.confidence.overall:.2%}")
        print(f"Level: {result.confidence.level.value}")
        print(f"Degraded: {result.degraded}")
"""

import asyncio
import hashlib
import logging
import time
from enum import Enum
from typing import Any, Optional
from uuid import UUID

import aiohttp
from pydantic import BaseModel, Field

from .base_client import BaseServiceClient, ClientError

logger = logging.getLogger(__name__)


class ConfidenceLevel(str, Enum):
    """Confidence levels for intelligence results."""

    HIGH = "high"  # 0.8 - 1.0
    MEDIUM = "medium"  # 0.5 - 0.8
    LOW = "low"  # 0.2 - 0.5
    MINIMAL = "minimal"  # 0.0 - 0.2


# Request/Response Models


class KnowledgeQueryRequest(BaseModel):
    """Request model for knowledge graph queries."""

    query: str = Field(..., description="Natural language query")
    max_depth: int = Field(default=3, description="Maximum traversal depth")
    filters: Optional[dict[str, Any]] = Field(None, description="Query filters")


class IntelligenceRequest(BaseModel):
    """Request model for intelligence retrieval."""

    context: str = Field(..., description="Context for intelligence lookup")
    include_patterns: bool = Field(default=True, description="Include pattern analysis")
    include_relationships: bool = Field(
        default=True, description="Include relationship data"
    )


class TreeNavigationRequest(BaseModel):
    """Request model for tree navigation."""

    start_node: str = Field(..., description="Starting node ID")
    direction: str = Field(default="both", description="Navigation direction")
    max_nodes: int = Field(default=100, description="Maximum nodes to retrieve")


class ConfidenceScore(BaseModel):
    """Confidence score for intelligence results.

    Provides multi-dimensional confidence scoring:
    - Data quality: How complete and accurate the source data is
    - Pattern strength: How strong the identified patterns are
    - Coverage: How much of the context was analyzed
    - Relationship density: How many relationships were found
    """

    data_quality: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Data quality score"
    )
    pattern_strength: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Pattern strength score"
    )
    coverage: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Context coverage score"
    )
    relationship_density: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Relationship density score"
    )

    @property
    def overall(self) -> float:
        """Calculate overall confidence as weighted average."""
        weights = {
            "data_quality": 0.35,
            "pattern_strength": 0.30,
            "coverage": 0.20,
            "relationship_density": 0.15,
        }
        return (
            self.data_quality * weights["data_quality"]
            + self.pattern_strength * weights["pattern_strength"]
            + self.coverage * weights["coverage"]
            + self.relationship_density * weights["relationship_density"]
        )

    @property
    def level(self) -> ConfidenceLevel:
        """Get confidence level based on overall score."""
        score = self.overall
        if score >= 0.8:
            return ConfidenceLevel.HIGH
        elif score >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.MINIMAL


class IntelligenceResult(BaseModel):
    """Intelligence result with confidence scoring."""

    intelligence: dict[str, Any] = Field(description="Intelligence data")
    patterns: list[str] = Field(default_factory=list, description="Identified patterns")
    relationships: list[dict[str, Any]] = Field(
        default_factory=list, description="Relationship data"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Result metadata"
    )
    confidence: ConfidenceScore = Field(
        default_factory=ConfidenceScore, description="Confidence scoring"
    )
    degraded: bool = Field(
        default=False, description="Whether this is a fallback/degraded result"
    )


class CacheEntry:
    """Cache entry with TTL and hit tracking.

    Note: Uses monotonic time for TTL calculations to avoid issues with
    system clock adjustments (NTP drift, DST changes, manual adjustments).
    Monotonic clock only moves forward, making it reliable for measuring
    elapsed time and cache expiration.
    """

    def __init__(self, data: Any, ttl: int = 300):
        """Initialize cache entry.

        Args:
            data: Data to cache
            ttl: Time to live in seconds (default: 5 minutes)
        """
        self.data = data
        # Use monotonic time for cache TTL to avoid wall clock adjustments
        self.created_at = time.monotonic()
        self.ttl = ttl
        self.hits = 0
        self.last_access = time.monotonic()

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return (time.monotonic() - self.created_at) > self.ttl

    def access(self) -> Any:
        """Access cached data and update metrics."""
        self.hits += 1
        self.last_access = time.monotonic()
        return self.data


class AsyncOnexTreeClient(BaseServiceClient):
    """Async client for OnexTree Agent Intelligence Service.

    Features:
        - Knowledge graph queries
        - Intelligence data retrieval
        - Tree navigation and pattern analysis
        - In-memory caching with TTL
        - Circuit breaker protection
        - Retry logic with exponential backoff
        - Correlation ID propagation

    Example:
        async with AsyncOnexTreeClient("http://192.168.86.200:8058") as client:
            intelligence = await client.get_intelligence(
                context="authentication patterns",
                correlation_id=correlation_id
            )

            knowledge = await client.query_knowledge(
                query="Find all API endpoints",
                correlation_id=correlation_id
            )
    """

    # Cache configuration constants
    DEFAULT_CACHE_MAX_SIZE = 1000  # Maximum number of cache entries
    DEFAULT_CACHE_EVICTION_RATIO = 0.2  # Evict 20% of oldest entries when limit reached

    def __init__(
        self,
        base_url: str = "http://192.168.86.200:8058",  # Remote OnexTree Service (via hostname resolution)
        timeout: float = 30.0,
        max_retries: int = 3,
        enable_cache: bool = True,
        cache_ttl: int = 300,
        cache_max_size: int = DEFAULT_CACHE_MAX_SIZE,
        cache_eviction_ratio: float = DEFAULT_CACHE_EVICTION_RATIO,
        app: Optional[Any] = None,
    ):
        """Initialize OnexTree client.

        Args:
            base_url: Base URL for OnexTreeService (defaults to remote infrastructure at 192.168.86.200:8058)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            enable_cache: Enable in-memory caching
            cache_ttl: Cache TTL in seconds (default: 5 minutes)
            cache_max_size: Maximum cache entries before eviction (default: 1000)
            cache_eviction_ratio: Ratio of entries to evict when max size reached (default: 0.2)
            app: Optional ASGI app for testing (bypasses network calls)
        """
        super().__init__(
            base_url=base_url,
            service_name="OnexTreeService",
            timeout=timeout,
            max_retries=max_retries,
            app=app,
        )

        # Caching configuration
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl
        self.cache_max_size = cache_max_size
        self.cache_eviction_ratio = cache_eviction_ratio
        self._cache: dict[str, CacheEntry] = {}
        self._cache_lock = asyncio.Lock()

        # Cache metrics
        self._cache_hits = 0
        self._cache_misses = 0

    async def _validate_connection(self) -> bool:
        """Validate connection to OnexTreeService.

        Returns:
            True if service is reachable and healthy
        """
        try:
            health = await self.health_check()
            return health.get("status") == "healthy"
        except ClientError as e:
            # Expected client errors (4xx, 5xx responses)
            logger.error(f"Connection validation failed (client error): {e}")
            return False
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            # Network/timeout errors
            logger.error(f"Connection validation failed (network/timeout): {e}")
            return False
        except Exception as e:
            # Unexpected errors - log with full traceback and re-raise
            logger.exception(f"Unexpected error during connection validation: {e}")
            raise

    def _get_cache_key(self, endpoint: str, params: Optional[dict] = None) -> str:
        """Generate cache key from endpoint and parameters.

        Args:
            endpoint: API endpoint
            params: Optional query parameters

        Returns:
            Cache key string
        """
        key_parts = [endpoint]
        if params:
            # Sort for consistent keys
            sorted_params = sorted(params.items())
            key_parts.append(str(sorted_params))

        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()

    async def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get data from cache if available and not expired.

        Args:
            cache_key: Cache key

        Returns:
            Cached data or None
        """
        if not self.enable_cache:
            return None

        async with self._cache_lock:
            entry = self._cache.get(cache_key)

            if entry is None:
                self._cache_misses += 1
                return None

            if entry.is_expired():
                # Remove expired entry
                del self._cache[cache_key]
                self._cache_misses += 1
                return None

            # Cache hit
            self._cache_hits += 1
            return entry.access()

    async def _set_cache(self, cache_key: str, data: Any) -> None:
        """Store data in cache.

        Args:
            cache_key: Cache key
            data: Data to cache
        """
        if not self.enable_cache:
            return

        async with self._cache_lock:
            self._cache[cache_key] = CacheEntry(data, ttl=self.cache_ttl)

            # Simple cache size management
            if len(self._cache) > self.cache_max_size:
                # Remove oldest entries based on eviction ratio
                sorted_entries = sorted(
                    self._cache.items(),
                    key=lambda x: x[1].last_access,
                )
                # Calculate number of entries to evict
                num_to_evict = int(len(sorted_entries) * self.cache_eviction_ratio)
                for key, _ in sorted_entries[:num_to_evict]:
                    del self._cache[key]

    async def clear_cache(self) -> None:
        """Clear all cached data."""
        async with self._cache_lock:
            self._cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
            logger.info("OnexTree client cache cleared")

    def _calculate_confidence(
        self,
        intelligence_data: dict[str, Any],
        patterns: list[str],
        relationships: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> ConfidenceScore:
        """Calculate confidence score for intelligence results.

        Args:
            intelligence_data: Raw intelligence data
            patterns: Identified patterns
            relationships: Relationship data
            metadata: Result metadata

        Returns:
            ConfidenceScore with detailed breakdown
        """
        # Data quality: Based on completeness of intelligence data
        data_quality = 0.0
        # Type guard: Ensure intelligence_data is actually a dict before dict operations
        if intelligence_data and isinstance(intelligence_data, dict):
            required_fields = ["analysis_type", "recommendations"]
            present_fields = sum(
                1 for field in required_fields if field in intelligence_data
            )
            data_quality = present_fields / len(required_fields)

            # Boost if we have detailed content
            recommendations = intelligence_data.get("recommendations", "")
            if isinstance(recommendations, str) and len(recommendations) > 100:
                data_quality = min(1.0, data_quality + 0.2)

        # Pattern strength: Based on number and quality of patterns
        pattern_strength = 0.0
        if patterns:
            # Base score from pattern count (up to 10 patterns = max score)
            pattern_count_score = min(1.0, len(patterns) / 10.0)

            # Quality score from pattern length (longer patterns = more detailed)
            avg_pattern_length = sum(len(p) for p in patterns) / len(patterns)
            pattern_quality_score = min(1.0, avg_pattern_length / 100.0)

            pattern_strength = (pattern_count_score + pattern_quality_score) / 2

        # Coverage: Based on metadata indicators
        coverage = 0.0
        if metadata:
            # Check for coverage indicators
            coverage_indicators = ["nodes_analyzed", "files_scanned", "tree_loaded"]
            present_indicators = sum(
                1 for indicator in coverage_indicators if indicator in metadata
            )

            if present_indicators > 0:
                coverage = present_indicators / len(coverage_indicators)

                # Boost if tree is loaded
                if metadata.get("tree_loaded", False):
                    coverage = min(1.0, coverage + 0.3)

        # Relationship density: Based on relationships found
        relationship_density = 0.0
        if relationships:
            # Base score from relationship count (up to 20 relationships = max score)
            relationship_density = min(1.0, len(relationships) / 20.0)

        return ConfidenceScore(
            data_quality=data_quality,
            pattern_strength=pattern_strength,
            coverage=coverage,
            relationship_density=relationship_density,
        )

    def _create_fallback_result(
        self, context: str, error: Optional[Exception] = None
    ) -> IntelligenceResult:
        """Create fallback intelligence result when service unavailable.

        Args:
            context: Original query context
            error: Optional error that triggered fallback

        Returns:
            IntelligenceResult with degraded flag and minimal confidence
        """
        logger.warning(
            f"Creating fallback intelligence result for context: {context}",
            extra={"error": str(error) if error else None},
        )

        return IntelligenceResult(
            intelligence={
                "analysis_type": "fallback",
                "context": context,
                "recommendations": "OnexTree intelligence service unavailable. Proceeding with basic workflow.",
                "status": "degraded",
            },
            patterns=[],
            relationships=[],
            metadata={
                "fallback": True,
                "error": str(error) if error else "Service unavailable",
                "timestamp": time.time(),
            },
            confidence=ConfidenceScore(
                data_quality=0.0,
                pattern_strength=0.0,
                coverage=0.0,
                relationship_density=0.0,
            ),
            degraded=True,
        )

    async def get_intelligence(
        self,
        context: str,
        include_patterns: bool = True,
        include_relationships: bool = True,
        correlation_id: Optional[UUID] = None,
        use_cache: bool = True,
        enable_fallback: bool = True,
    ) -> IntelligenceResult:
        """Get intelligence data for given context with confidence scoring.

        Args:
            context: Context for intelligence lookup
            include_patterns: Include pattern analysis in results
            include_relationships: Include relationship data
            correlation_id: Optional correlation ID for request tracing
            use_cache: Use cached results if available
            enable_fallback: Enable fallback result if service unavailable

        Returns:
            IntelligenceResult with:
            - intelligence: Intelligence data
            - patterns: Identified patterns
            - relationships: Relationship data
            - metadata: Result metadata
            - confidence: Multi-dimensional confidence scoring
            - degraded: Whether this is a fallback result

        Example:
            result = await client.get_intelligence("authentication patterns")
            if result.confidence.overall >= 0.7:
                # High confidence result
                apply_recommendations(result.intelligence)
            elif result.degraded:
                # Fallback result, proceed with caution
                log_warning("Using degraded intelligence")
        """
        try:
            # Check cache
            cache_key = self._get_cache_key(
                "/intelligence",
                {
                    "context": context,
                    "patterns": include_patterns,
                    "relationships": include_relationships,
                },
            )

            if use_cache:
                cached_data = await self._get_from_cache(cache_key)
                if cached_data is not None:
                    logger.debug(
                        f"Cache hit for intelligence query: {context}",
                        extra={
                            "correlation_id": (
                                str(correlation_id) if correlation_id else None
                            ),
                        },
                    )
                    # If cached data is dict (legacy), convert to IntelligenceResult
                    if isinstance(cached_data, dict):
                        # Calculate confidence for legacy data
                        confidence = self._calculate_confidence(
                            intelligence_data=cached_data.get("intelligence", {}),
                            patterns=cached_data.get("patterns", []),
                            relationships=cached_data.get("relationships", []),
                            metadata=cached_data.get("metadata", {}),
                        )
                        return IntelligenceResult(
                            intelligence=cached_data.get("intelligence", {}),
                            patterns=cached_data.get("patterns", []),
                            relationships=cached_data.get("relationships", []),
                            metadata=cached_data.get("metadata", {}),
                            confidence=confidence,
                            degraded=False,
                        )
                    return cached_data

            # Make request
            request = IntelligenceRequest(
                context=context,
                include_patterns=include_patterns,
                include_relationships=include_relationships,
            )

            response = await self._make_request_with_retry(
                method="POST",
                endpoint="/intelligence",
                correlation_id=correlation_id,
                json=request.model_dump(mode="json"),
            )

            result = response.json()

            if result.get("status") == "success":
                data = result.get("data", {})

                # Extract components
                intelligence_data = data.get("intelligence", {})
                patterns = data.get("patterns", [])
                relationships = data.get("relationships", [])
                metadata = data.get("metadata", {})

                # Calculate confidence
                confidence = self._calculate_confidence(
                    intelligence_data=intelligence_data,
                    patterns=patterns,
                    relationships=relationships,
                    metadata=metadata,
                )

                # Create result
                intelligence_result = IntelligenceResult(
                    intelligence=intelligence_data,
                    patterns=patterns,
                    relationships=relationships,
                    metadata=metadata,
                    confidence=confidence,
                    degraded=False,
                )

                # Cache result
                await self._set_cache(cache_key, intelligence_result)

                logger.info(
                    f"Intelligence retrieved successfully for context: {context}",
                    extra={
                        "correlation_id": (
                            str(correlation_id) if correlation_id else None
                        ),
                        "confidence_overall": confidence.overall,
                        "confidence_level": confidence.level.value,
                    },
                )

                return intelligence_result
            else:
                raise ClientError(
                    f"Intelligence retrieval failed: {result.get('error', 'Unknown error')}"
                )

        except ClientError as e:
            # Expected client errors (4xx, 5xx responses)
            logger.error(
                f"Intelligence retrieval failed (client error): {e}",
                extra={
                    "correlation_id": str(correlation_id) if correlation_id else None,
                    "context": context,
                    "enable_fallback": enable_fallback,
                },
            )
            # Return fallback result if enabled
            if enable_fallback:
                return self._create_fallback_result(context, error=e)
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            # Network/timeout errors
            logger.error(
                f"Intelligence retrieval failed (network/timeout): {e}",
                extra={
                    "correlation_id": str(correlation_id) if correlation_id else None,
                    "context": context,
                    "enable_fallback": enable_fallback,
                },
            )
            # Return fallback result if enabled
            if enable_fallback:
                return self._create_fallback_result(context, error=e)
            raise
        except (KeyError, ValueError, TypeError) as e:
            # Data structure/parsing errors
            logger.error(
                f"Intelligence retrieval failed (data error): {e}",
                extra={
                    "correlation_id": str(correlation_id) if correlation_id else None,
                    "context": context,
                    "enable_fallback": enable_fallback,
                },
            )
            # Return fallback result if enabled
            if enable_fallback:
                return self._create_fallback_result(context, error=e)
            raise
        except Exception as e:
            # Unexpected errors - log with full traceback and re-raise
            logger.exception(
                f"Unexpected error during intelligence retrieval: {e}",
                extra={
                    "correlation_id": str(correlation_id) if correlation_id else None,
                    "context": context,
                },
            )
            raise

    async def query_knowledge(
        self,
        query: str,
        max_depth: int = 3,
        filters: Optional[dict[str, Any]] = None,
        correlation_id: Optional[UUID] = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """Query knowledge graph.

        Args:
            query: Natural language query
            max_depth: Maximum graph traversal depth
            filters: Optional query filters
            correlation_id: Optional correlation ID for request tracing
            use_cache: Use cached results if available

        Returns:
            Dictionary with query results:
            {
                "results": [...],
                "paths": [...],
                "metadata": {...}
            }

        Raises:
            ClientError: If query fails
        """
        try:
            # Check cache
            cache_key = self._get_cache_key(
                "/knowledge/query",
                {
                    "query": query,
                    "max_depth": max_depth,
                    "filters": str(filters) if filters else None,
                },
            )

            if use_cache:
                cached_data = await self._get_from_cache(cache_key)
                if cached_data is not None:
                    logger.debug(
                        f"Cache hit for knowledge query: {query}",
                        extra={
                            "correlation_id": (
                                str(correlation_id) if correlation_id else None
                            ),
                        },
                    )
                    return cached_data

            # Make request
            request = KnowledgeQueryRequest(
                query=query,
                max_depth=max_depth,
                filters=filters,
            )

            response = await self._make_request_with_retry(
                method="POST",
                endpoint="/knowledge/query",
                correlation_id=correlation_id,
                json=request.model_dump(mode="json"),
            )

            result = response.json()

            if result.get("status") == "success":
                data = result.get("data", {})
                # Cache result
                await self._set_cache(cache_key, data)
                return data
            else:
                raise ClientError(
                    f"Knowledge query failed: {result.get('error', 'Unknown error')}"
                )

        except ClientError as e:
            # Expected client errors (4xx, 5xx responses)
            logger.error(
                f"Knowledge query failed (client error): {e}",
                extra={
                    "correlation_id": str(correlation_id) if correlation_id else None,
                    "query": query,
                },
            )
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            # Network/timeout errors
            logger.error(
                f"Knowledge query failed (network/timeout): {e}",
                extra={
                    "correlation_id": str(correlation_id) if correlation_id else None,
                    "query": query,
                },
            )
            raise
        except (KeyError, ValueError, TypeError) as e:
            # Data structure/parsing errors
            logger.error(
                f"Knowledge query failed (data error): {e}",
                extra={
                    "correlation_id": str(correlation_id) if correlation_id else None,
                    "query": query,
                },
            )
            raise
        except Exception as e:
            # Unexpected errors - log with full traceback and re-raise
            logger.exception(
                f"Unexpected error during knowledge query: {e}",
                extra={
                    "correlation_id": str(correlation_id) if correlation_id else None,
                    "query": query,
                },
            )
            raise

    async def navigate_tree(
        self,
        start_node: str,
        direction: str = "both",
        max_nodes: int = 100,
        correlation_id: Optional[UUID] = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """Navigate tree from starting node.

        Args:
            start_node: Starting node ID
            direction: Navigation direction (up, down, both)
            max_nodes: Maximum nodes to retrieve
            correlation_id: Optional correlation ID for request tracing
            use_cache: Use cached results if available

        Returns:
            Dictionary with navigation results:
            {
                "nodes": [...],
                "edges": [...],
                "metadata": {...}
            }

        Raises:
            ClientError: If navigation fails
        """
        try:
            # Check cache
            cache_key = self._get_cache_key(
                "/tree/navigate",
                {
                    "start_node": start_node,
                    "direction": direction,
                    "max_nodes": max_nodes,
                },
            )

            if use_cache:
                cached_data = await self._get_from_cache(cache_key)
                if cached_data is not None:
                    logger.debug(
                        f"Cache hit for tree navigation: {start_node}",
                        extra={
                            "correlation_id": (
                                str(correlation_id) if correlation_id else None
                            ),
                        },
                    )
                    return cached_data

            # Make request
            request = TreeNavigationRequest(
                start_node=start_node,
                direction=direction,
                max_nodes=max_nodes,
            )

            response = await self._make_request_with_retry(
                method="POST",
                endpoint="/tree/navigate",
                correlation_id=correlation_id,
                json=request.model_dump(mode="json"),
            )

            result = response.json()

            if result.get("status") == "success":
                data = result.get("data", {})
                # Cache result
                await self._set_cache(cache_key, data)
                return data
            else:
                raise ClientError(
                    f"Tree navigation failed: {result.get('error', 'Unknown error')}"
                )

        except ClientError as e:
            # Expected client errors (4xx, 5xx responses)
            logger.error(
                f"Tree navigation failed (client error): {e}",
                extra={
                    "correlation_id": str(correlation_id) if correlation_id else None,
                    "start_node": start_node,
                },
            )
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            # Network/timeout errors
            logger.error(
                f"Tree navigation failed (network/timeout): {e}",
                extra={
                    "correlation_id": str(correlation_id) if correlation_id else None,
                    "start_node": start_node,
                },
            )
            raise
        except (KeyError, ValueError, TypeError) as e:
            # Data structure/parsing errors
            logger.error(
                f"Tree navigation failed (data error): {e}",
                extra={
                    "correlation_id": str(correlation_id) if correlation_id else None,
                    "start_node": start_node,
                },
            )
            raise
        except Exception as e:
            # Unexpected errors - log with full traceback and re-raise
            logger.exception(
                f"Unexpected error during tree navigation: {e}",
                extra={
                    "correlation_id": str(correlation_id) if correlation_id else None,
                    "start_node": start_node,
                },
            )
            raise

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache metrics
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0

        return {
            "enabled": self.enable_cache,
            "cache_size": len(self._cache),
            "cache_max_size": self.cache_max_size,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "ttl_seconds": self.cache_ttl,
            "eviction_ratio": self.cache_eviction_ratio,
        }

    def get_metrics(self) -> dict[str, Any]:
        """Get client metrics including cache statistics.

        Returns:
            Dictionary with comprehensive client metrics
        """
        base_metrics = super().get_metrics()
        base_metrics["cache"] = self.get_cache_stats()
        return base_metrics
