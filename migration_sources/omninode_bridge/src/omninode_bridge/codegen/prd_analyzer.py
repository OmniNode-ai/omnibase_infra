#!/usr/bin/env python3
"""
PRD Analyzer for Code Generation Workflow.

Analyzes Product Requirements Documents (PRDs) and natural language prompts
to extract structured requirements for node generation.

Integrates with Archon MCP for intelligence gathering and context enrichment.

ONEX v2.0 Compliance:
- Pure function design for requirement extraction
- Integration with Archon MCP intelligence services
- Event-driven intelligence gathering via Kafka
- Structured output for downstream stages
"""

import asyncio
import logging
import re
from datetime import UTC, datetime
from typing import Any, Optional
from uuid import UUID, uuid4

import aiohttp
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ModelPRDRequirements(BaseModel):
    """
    Structured requirements extracted from PRD analysis.

    Contains all information needed for node generation workflow.
    """

    # Core identification
    node_type: str = Field(
        ..., description="effect|compute|reducer|orchestrator (inferred or specified)"
    )
    service_name: str = Field(..., description="Service name in snake_case")
    domain: str = Field(
        ..., description="Domain: database|api|ml|messaging|storage|..."
    )

    # Functional requirements
    operations: list[str] = Field(
        default_factory=list, description="List of operations/methods to implement"
    )
    business_description: str = Field(
        ..., description="Human-readable business purpose"
    )
    features: list[str] = Field(
        default_factory=list, description="Key features to implement"
    )

    # Technical requirements
    dependencies: dict[str, str] = Field(
        default_factory=dict, description="External service dependencies"
    )
    performance_requirements: dict[str, Any] = Field(
        default_factory=dict,
        description="Performance targets (latency, throughput, etc.)",
    )
    data_models: list[str] = Field(
        default_factory=list, description="Data models to generate"
    )

    # Intelligence metadata
    best_practices: list[str] = Field(
        default_factory=list, description="Best practices from RAG intelligence"
    )
    similar_patterns: list[str] = Field(
        default_factory=list,
        description="Similar pattern IDs from intelligence service",
    )
    code_examples: list[str] = Field(
        default_factory=list, description="Relevant code examples"
    )

    # Quality gates
    min_test_coverage: float = Field(default=0.85, ge=0.0, le=1.0)
    complexity_threshold: int = Field(
        default=10, description="Max cyclomatic complexity"
    )

    # Metadata
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    extraction_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class PRDAnalyzer:
    """
    Analyzes PRDs and prompts to extract structured requirements.

    Integrates with Archon MCP for intelligence gathering and pattern matching.
    Provides fallback for offline operation without external dependencies.
    """

    def __init__(
        self,
        archon_mcp_url: Optional[str] = None,
        enable_intelligence: bool = True,
        timeout_seconds: int = 30,
    ):
        """
        Initialize PRD Analyzer.

        Args:
            archon_mcp_url: Archon MCP endpoint URL (e.g., http://archon:8060)
            enable_intelligence: Enable RAG intelligence gathering
            timeout_seconds: HTTP request timeout
        """
        self.archon_mcp_url = archon_mcp_url or "http://localhost:8060"
        self.enable_intelligence = enable_intelligence
        self.timeout_seconds = timeout_seconds

    async def analyze_prompt(
        self,
        prompt: str,
        correlation_id: Optional[UUID] = None,
        node_type_hint: Optional[str] = None,
    ) -> ModelPRDRequirements:
        """
        Analyze natural language prompt and extract requirements.

        Args:
            prompt: Natural language description of node to generate
            correlation_id: Optional correlation ID for tracing
            node_type_hint: Optional hint for node type

        Returns:
            ModelPRDRequirements with extracted requirements

        Example:
            >>> analyzer = PRDAnalyzer()
            >>> reqs = await analyzer.analyze_prompt(
            ...     "Create PostgreSQL CRUD Effect node with connection pooling"
            ... )
            >>> assert reqs.node_type == "effect"
            >>> assert "database" in reqs.domain
        """
        correlation_id = correlation_id or uuid4()

        # Step 1: Extract basic requirements from prompt
        basic_reqs = self._extract_basic_requirements(prompt, node_type_hint)

        # Step 2: Gather intelligence if enabled
        if self.enable_intelligence:
            try:
                intelligence_data = await self._query_archon_intelligence(
                    prompt=prompt,
                    node_type=basic_reqs["node_type"],
                    correlation_id=correlation_id,
                )
                basic_reqs.update(intelligence_data)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                # Expected errors during intelligence gathering - graceful degradation
                basic_reqs["intelligence_error"] = f"Network/timeout error: {e}"
                logger.warning(
                    f"Intelligence gathering failed (network/timeout), continuing without: {e}"
                )
            except (KeyError, ValueError, TypeError) as e:
                # Data parsing errors - graceful degradation
                basic_reqs["intelligence_error"] = f"Data error: {e}"
                logger.warning(
                    f"Intelligence gathering failed (data error), continuing without: {e}"
                )
            except Exception as e:
                # Unexpected errors - log with full traceback but continue with degraded mode
                logger.exception(f"Unexpected error during intelligence gathering: {e}")
                basic_reqs["intelligence_error"] = f"Unexpected error: {e}"

        # Step 3: Build structured requirements
        return ModelPRDRequirements(**basic_reqs)

    def _extract_basic_requirements(
        self, prompt: str, node_type_hint: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Extract basic requirements from prompt using pattern matching.

        This is a fallback implementation using regex patterns.
        In production, this would use an LLM for better extraction.

        Args:
            prompt: User prompt
            node_type_hint: Optional node type hint

        Returns:
            Dictionary with basic requirements
        """
        # Infer node type from prompt keywords
        node_type = node_type_hint or self._infer_node_type(prompt)

        # Extract service name (default to generic if not found)
        service_name = self._extract_service_name(prompt)

        # Extract domain
        domain = self._extract_domain(prompt)

        # Extract operations
        operations = self._extract_operations(prompt)

        # Business description (use first sentence or full prompt if short)
        business_description = self._extract_business_description(prompt)

        return {
            "node_type": node_type,
            "service_name": service_name,
            "domain": domain,
            "operations": operations,
            "business_description": business_description,
            "features": self._extract_features(prompt),
            "extraction_confidence": 0.7,  # Medium confidence for pattern matching
        }

    def _infer_node_type(self, prompt: str) -> str:
        """
        Infer node type from prompt keywords.

        Effect: I/O operations, external services, side effects
        Compute: Pure transformations, calculations, algorithms
        Reducer: Aggregation, accumulation, state management
        Orchestrator: Coordination, workflow management, multi-step processes
        """
        prompt_lower = prompt.lower()

        # Orchestrator indicators
        if any(
            word in prompt_lower
            for word in [
                "orchestrate",
                "coordinate",
                "workflow",
                "multi-step",
                "pipeline",
                "stages",
            ]
        ):
            return "orchestrator"

        # Reducer indicators
        if any(
            word in prompt_lower
            for word in [
                "aggregate",
                "reduce",
                "accumulate",
                "collect",
                "group",
                "summarize",
            ]
        ):
            return "reducer"

        # Compute indicators
        if any(
            word in prompt_lower
            for word in [
                "transform",
                "calculate",
                "compute",
                "process",
                "parse",
                "convert",
            ]
        ):
            return "compute"

        # Effect indicators (default for I/O operations)
        return "effect"

    def _extract_service_name(self, prompt: str) -> str:
        """Extract or generate service name from prompt."""
        # Look for capitalized words or quoted names
        match = re.search(r'(?:called?|named?)\s+["\']?([a-zA-Z_]+)["\']?', prompt)
        if match:
            return match.group(1).lower()

        # Extract first meaningful noun phrase
        words = prompt.split()
        for i, word in enumerate(words):
            if word.lower() in ["create", "build", "implement", "generate"]:
                # Look for a valid service name after the action word
                # Skip articles and other single-character words
                for j in range(i + 1, len(words)):
                    service_word = re.sub(r"[^a-z0-9_]", "", words[j].lower())
                    # Must be at least 2 characters and start with a letter
                    if len(service_word) >= 2 and service_word[0].isalpha():
                        return service_word

        return "generated_service"

    def _extract_domain(self, prompt: str) -> str:
        """Extract domain from prompt keywords."""
        prompt_lower = prompt.lower()

        domain_keywords = {
            "database": [
                "postgres",
                "postgresql",
                "mysql",
                "database",
                "crud",
                "sql",
            ],
            "api": ["api", "rest", "graphql", "endpoint", "http", "request"],
            "ml": ["ml", "machine learning", "model", "prediction", "training"],
            "messaging": ["kafka", "queue", "message", "event", "publish", "subscribe"],
            "storage": ["s3", "storage", "file", "object", "blob"],
            "cache": ["redis", "cache", "memcache", "valkey"],
            "monitoring": [
                "metrics",
                "logging",
                "tracing",
                "monitoring",
                "observability",
            ],
        }

        for domain, keywords in domain_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return domain

        return "general"

    def _extract_operations(self, prompt: str) -> list[str]:
        """Extract operations/methods from prompt."""
        operations = []

        # CRUD operations
        crud_map = {
            "create": ["create", "insert", "add", "new"],
            "read": ["read", "get", "fetch", "retrieve", "query", "list"],
            "update": ["update", "modify", "edit", "change"],
            "delete": ["delete", "remove", "destroy"],
        }

        prompt_lower = prompt.lower()
        for op, keywords in crud_map.items():
            if any(keyword in prompt_lower for keyword in keywords):
                operations.append(op)

        # If no specific operations found, default to common operations
        if not operations:
            operations = ["execute", "process"]

        return operations

    def _extract_business_description(self, prompt: str) -> str:
        """Extract business description from prompt."""
        # Use first sentence or full prompt if short
        sentences = re.split(r"[.!?]", prompt)
        if sentences:
            first_sentence = sentences[0].strip()
            if len(first_sentence) > 10:
                return first_sentence

        # Fallback: use full prompt if short enough
        if len(prompt) < 200:
            return prompt

        return prompt[:200] + "..."

    def _extract_features(self, prompt: str) -> list[str]:
        """Extract key features from prompt."""
        features = []

        feature_keywords = {
            "connection_pooling": ["connection pool", "pooling"],
            "caching": ["cache", "caching"],
            "retry_logic": ["retry", "retries", "resilience"],
            "circuit_breaker": ["circuit breaker", "circuit-breaker"],
            "rate_limiting": ["rate limit", "throttle"],
            "authentication": ["auth", "authentication", "authorization"],
            "validation": ["validate", "validation"],
            "logging": ["log", "logging"],
            "metrics": ["metrics", "monitoring"],
        }

        prompt_lower = prompt.lower()
        for feature, keywords in feature_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                features.append(feature)

        return features

    async def _query_archon_intelligence(
        self, prompt: str, node_type: str, correlation_id: UUID
    ) -> dict[str, Any]:
        """
        Query Archon MCP for intelligence gathering.

        Fetches:
        - Similar code patterns
        - Best practices
        - Code examples
        - Performance targets

        Args:
            prompt: User prompt
            node_type: Inferred node type
            correlation_id: Correlation ID for tracing

        Returns:
            Dictionary with intelligence data

        Raises:
            aiohttp.ClientError: If HTTP request fails
        """
        url = f"{self.archon_mcp_url}/api/intelligence/query"

        request_payload = {
            "query": prompt,
            "node_type": node_type,
            "correlation_id": str(correlation_id),
            "top_k": 5,
            "min_quality_score": 0.8,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=request_payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout_seconds),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_intelligence_response(data)
                    else:
                        # Non-200 response: return empty intelligence
                        logger.warning(
                            f"Archon intelligence query returned {response.status}, using empty intelligence"
                        )
                        return {}
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            # Expected errors: network issues, timeouts - graceful degradation
            logger.warning(
                f"Archon intelligence query failed (network/timeout), using empty intelligence: {e}"
            )
            return {}
        except (KeyError, ValueError, TypeError) as e:
            # Data parsing errors - graceful degradation
            logger.warning(
                f"Archon intelligence response parsing failed, using empty intelligence: {e}"
            )
            return {}
        except Exception as e:
            # Unexpected errors - log with full traceback but continue with empty intelligence
            logger.exception(f"Unexpected error querying Archon intelligence: {e}")
            return {}

    def _parse_intelligence_response(self, data: dict[str, Any]) -> dict[str, Any]:
        """Parse Archon MCP intelligence response into requirements format."""
        return {
            "best_practices": data.get("best_practices", []),
            "similar_patterns": [p.get("pattern_id") for p in data.get("patterns", [])],
            "code_examples": data.get("code_examples", []),
            "performance_requirements": data.get("performance_targets", {}),
            "extraction_confidence": min(
                data.get("confidence", 0.7) + 0.2, 1.0
            ),  # Boost confidence with intelligence
        }


# Export
__all__ = ["PRDAnalyzer", "ModelPRDRequirements"]
