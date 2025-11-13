"""
Requirements Analyzer for intelligent mixin selection.

Analyzes ModelPRDRequirements to extract features and categorize requirement
strengths across different domains (database, API, Kafka, security, etc.).

Performance Target: <50ms per analysis
Accuracy Target: >90% categorization correctness
"""

import logging
import re
from collections import defaultdict
from typing import Any

from omninode_bridge.codegen.mixins.models import ModelRequirementAnalysis
from omninode_bridge.codegen.prd_analyzer import ModelPRDRequirements

logger = logging.getLogger(__name__)

# Domain-specific keyword dictionaries
DOMAIN_KEYWORDS = {
    "database": {
        "database",
        "db",
        "postgres",
        "postgresql",
        "sql",
        "query",
        "transaction",
        "connection",
        "pool",
        "pooling",
        "crud",
        "create",
        "read",
        "update",
        "delete",
        "table",
        "schema",
        "migration",
        "orm",
        "sqlalchemy",
        "asyncpg",
        "commit",
        "rollback",
        "isolation",
        "lock",
    },
    "api": {
        "api",
        "rest",
        "http",
        "https",
        "client",
        "request",
        "response",
        "get",
        "post",
        "put",
        "patch",
        "delete",
        "endpoint",
        "json",
        "xml",
        "graphql",
        "soap",
        "httpx",
        "aiohttp",
        "requests",
        "urllib",
        "retry",
        "timeout",
        "backoff",
        "circuit",
        "breaker",
    },
    "kafka": {
        "kafka",
        "redpanda",
        "event",
        "message",
        "publish",
        "consume",
        "topic",
        "partition",
        "offset",
        "broker",
        "producer",
        "consumer",
        "stream",
        "streaming",
        "event-driven",
        "async",
        "aiokafka",
        "confluent",
        "serialization",
        "deserialization",
    },
    "security": {
        "secure",
        "security",
        "auth",
        "authentication",
        "authorization",
        "token",
        "jwt",
        "oauth",
        "api-key",
        "credential",
        "secret",
        "encrypt",
        "decrypt",
        "hash",
        "sign",
        "verify",
        "validate",
        "validation",
        "sanitize",
        "escape",
        "sensitive",
        "pii",
        "redact",
        "mask",
    },
    "observability": {
        "metrics",
        "logging",
        "tracing",
        "monitoring",
        "observability",
        "prometheus",
        "grafana",
        "datadog",
        "opentelemetry",
        "health",
        "healthcheck",
        "liveness",
        "readiness",
        "log",
        "logger",
        "structured-logging",
        "json-log",
        "trace",
        "span",
        "baggage",
        "context",
    },
    "resilience": {
        "retry",
        "circuit-breaker",
        "fallback",
        "timeout",
        "fault-tolerant",
        "resilient",
        "robust",
        "reliable",
        "backoff",
        "exponential-backoff",
        "jitter",
        "degradation",
        "graceful-degradation",
    },
    "caching": {
        "cache",
        "caching",
        "memoize",
        "ttl",
        "expiration",
        "redis",
        "memcached",
        "in-memory",
        "lru",
        "invalidate",
        "evict",
        "warm-up",
    },
    "performance": {
        "performance",
        "optimize",
        "optimization",
        "fast",
        "speed",
        "throughput",
        "latency",
        "qps",
        "rps",
        "concurrent",
        "batch",
        "bulk",
        "parallel",
        "async",
        "asyncio",
        "pool",
        "pooling",
        "connection-reuse",
    },
}

# Dependency package → capability mapping
DEPENDENCY_CAPABILITIES = {
    "asyncpg": ["database", "postgres"],
    "psycopg2": ["database", "postgres"],
    "psycopg3": ["database", "postgres"],
    "sqlalchemy": ["database", "orm"],
    "httpx": ["api", "http-client"],
    "aiohttp": ["api", "http-client"],
    "requests": ["api", "http-client"],
    "aiokafka": ["kafka", "messaging"],
    "confluent-kafka": ["kafka", "messaging"],
    "redis": ["caching"],
    "aioredis": ["caching"],
    "prometheus-client": ["metrics", "observability"],
    "opentelemetry-api": ["metrics", "observability"],
}

# Operation pattern definitions
OPERATION_PATTERNS = {
    "database": {
        "keywords": [
            "create",
            "read",
            "update",
            "delete",
            "insert",
            "select",
            "upsert",
            "get",
            "put",
            "post",
            "patch",
        ],
        "weight": 3.0,
    },
    "api": {
        "keywords": ["fetch", "call", "request", "invoke", "query", "get", "post"],
        "weight": 2.5,
    },
    "kafka": {
        "keywords": [
            "publish",
            "emit",
            "send",
            "notify",
            "broadcast",
            "produce",
            "consume",
            "handle",
        ],
        "weight": 2.5,
    },
    "security": {
        "keywords": ["validate", "check", "verify", "sanitize", "clean"],
        "weight": 2.0,
    },
    "performance": {
        "keywords": [
            "aggregate",
            "sum",
            "count",
            "reduce",
            "collect",
            "accumulate",
            "batch",
            "stream",
        ],
        "weight": 1.5,
    },
}


class RequirementsAnalyzer:
    """
    Analyzes PRD requirements to extract features and calculate category scores.

    This is the first stage of the intelligent mixin selection pipeline.
    """

    def __init__(self):
        """Initialize the requirements analyzer."""
        self._stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "to",
            "of",
            "in",
            "on",
            "at",
            "for",
            "with",
        }

    def analyze(self, requirements: ModelPRDRequirements) -> ModelRequirementAnalysis:
        """
        Analyze requirements and extract features.

        Args:
            requirements: PRD requirements from prd_analyzer

        Returns:
            ModelRequirementAnalysis with category scores and extracted features
        """
        # Step 1: Extract keywords
        keywords = self._extract_keywords(requirements)

        # Step 2: Analyze dependencies
        capabilities = self._analyze_dependencies(requirements.dependencies)

        # Step 3: Identify operation patterns
        operation_patterns = self._identify_operation_patterns(requirements.operations)

        # Step 4: Analyze performance requirements
        performance_optimizations = self._analyze_performance_requirements(
            requirements.performance_requirements
        )

        # Step 5: Calculate category scores
        category_scores = self._categorize_requirements(
            keywords, capabilities, operation_patterns, performance_optimizations
        )

        # Step 6: Calculate confidence
        confidence = self._calculate_confidence(category_scores, keywords, capabilities)

        # Step 7: Generate rationale
        rationale = self._generate_rationale(category_scores, keywords, capabilities)

        return ModelRequirementAnalysis(
            keywords=keywords,
            dependency_packages=capabilities,
            operation_types=set(operation_patterns.keys()),
            database_score=min(10.0, category_scores["database"]),
            api_score=min(10.0, category_scores["api"]),
            kafka_score=min(10.0, category_scores["kafka"]),
            security_score=min(10.0, category_scores["security"]),
            observability_score=min(10.0, category_scores["observability"]),
            resilience_score=min(10.0, category_scores["resilience"]),
            caching_score=min(10.0, category_scores["caching"]),
            performance_score=min(10.0, category_scores["performance"]),
            confidence=confidence,
            rationale=rationale,
        )

    def _extract_keywords(self, requirements: ModelPRDRequirements) -> set[str]:
        """Extract and normalize keywords from text fields."""
        keywords = set()

        # Extract from business_description
        text_fields = [
            requirements.business_description,
            *requirements.operations,
            *requirements.features,
        ]

        for text in text_fields:
            words = self._tokenize(text)
            for word in words:
                if len(word) > 2 and word not in self._stopwords:
                    keywords.add(word)

        return keywords

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into normalized words."""
        # Convert to lowercase
        text_lower = text.lower()
        # Split on non-alphanumeric characters
        words = re.findall(r"\b[a-z][a-z0-9_-]*\b", text_lower)
        return words

    def _analyze_dependencies(self, dependencies: dict[str, str]) -> set[str]:
        """Extract capabilities from dependency packages."""
        capabilities = set()

        for package_name in dependencies:
            # Normalize package name (remove extras, lowercase)
            normalized = package_name.lower().split("[")[0].strip()

            if normalized in DEPENDENCY_CAPABILITIES:
                capabilities.update(DEPENDENCY_CAPABILITIES[normalized])

        return capabilities

    def _identify_operation_patterns(self, operations: list[str]) -> dict[str, float]:
        """Identify patterns in operation names and score categories."""
        category_scores = defaultdict(float)

        for operation in operations:
            operation_lower = operation.lower()

            # Check each pattern
            for category, pattern_config in OPERATION_PATTERNS.items():
                keywords = pattern_config["keywords"]
                weight = pattern_config["weight"]

                # Count keyword matches
                matches = sum(1 for kw in keywords if kw in operation_lower)

                if matches > 0:
                    category_scores[category] += matches * weight

        # Normalize scores to 0-10 range
        if category_scores:
            max_score = max(category_scores.values())
            for category in category_scores:
                category_scores[category] = min(
                    10.0, (category_scores[category] / max_score) * 10
                )

        return dict(category_scores)

    def _analyze_performance_requirements(
        self, perf_reqs: dict[str, Any]
    ) -> dict[str, float]:
        """Analyze performance requirements to determine optimization needs."""
        optimizations = {
            "caching": 0.0,
            "connection_pooling": 0.0,
            "retry_mechanism": 0.0,
            "circuit_breaker": 0.0,
            "batch_processing": 0.0,
        }

        # Check latency requirements
        if "latency_ms" in perf_reqs:
            latency = perf_reqs["latency_ms"]
            if latency < 100:
                optimizations["caching"] = 8.0
                optimizations["connection_pooling"] = 10.0
            elif latency < 500:
                optimizations["caching"] = 6.0
                optimizations["connection_pooling"] = 7.0

        # Check throughput requirements
        if "throughput_rps" in perf_reqs:
            rps = perf_reqs["throughput_rps"]
            if rps > 1000:
                optimizations["connection_pooling"] = 10.0
                optimizations["batch_processing"] = 8.0
            elif rps > 100:
                optimizations["connection_pooling"] = 7.0

        # Check reliability requirements
        if "availability" in perf_reqs:
            availability = perf_reqs["availability"]
            if availability > 0.999:
                optimizations["retry_mechanism"] = 9.0
                optimizations["circuit_breaker"] = 10.0
            elif availability > 0.99:
                optimizations["retry_mechanism"] = 7.0
                optimizations["circuit_breaker"] = 8.0

        return optimizations

    def _categorize_requirements(
        self,
        keywords: set[str],
        capabilities: set[str],
        operation_patterns: dict[str, float],
        performance_optimizations: dict[str, float],
    ) -> dict[str, float]:
        """Calculate category scores based on extracted features."""
        scores = {}

        for category in [
            "database",
            "api",
            "kafka",
            "security",
            "observability",
            "resilience",
            "caching",
            "performance",
        ]:
            # Keyword match score (0-10)
            keyword_score = self._count_keyword_matches(
                keywords, DOMAIN_KEYWORDS.get(category, set())
            )

            # Capability match score (0-3)
            capability_score = 3.0 if category in capabilities else 0.0

            # Operation pattern score (0-10)
            operation_score = operation_patterns.get(category, 0.0)

            # Performance optimization score (0-10)
            perf_score = 0.0
            if category == "database":
                perf_score = performance_optimizations.get("connection_pooling", 0.0)
            elif category == "api" or category == "resilience":
                perf_score = (
                    performance_optimizations.get("retry_mechanism", 0.0)
                    + performance_optimizations.get("circuit_breaker", 0.0)
                ) / 2.0
            elif category == "caching":
                perf_score = performance_optimizations.get("caching", 0.0)
            elif category == "performance":
                perf_score = performance_optimizations.get("batch_processing", 0.0)

            # Weighted average
            total_weight = 2.0 + 3.0 + 2.0 + 1.5
            scores[category] = (
                keyword_score * 2.0
                + capability_score * 3.0
                + operation_score * 2.0
                + perf_score * 1.5
            ) / total_weight

        return scores

    def _count_keyword_matches(
        self, extracted_keywords: set[str], category_keywords: set[str]
    ) -> float:
        """Count keyword matches and return score (0-10)."""
        matches = len(extracted_keywords.intersection(category_keywords))
        return min(10.0, matches * 2.0)

    def _calculate_confidence(
        self, scores: dict[str, float], keywords: set[str], capabilities: set[str]
    ) -> float:
        """Calculate overall confidence in requirement extraction."""
        # Factor 1: Non-zero scores
        non_zero_scores = sum(1 for s in scores.values() if s > 0.5)
        score_factor = min(0.3, non_zero_scores * 0.1)

        # Factor 2: Keyword count
        keyword_factor = min(0.3, len(keywords) * 0.03)

        # Factor 3: Capability count
        capability_factor = min(0.2, len(capabilities) * 0.05)

        # Factor 4: Score clarity
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) >= 2 and sorted_scores[0] > 0:
            clarity = (sorted_scores[0] - sorted_scores[1]) / 10.0
            clarity_factor = min(0.2, clarity * 0.2)
        else:
            clarity_factor = 0.1

        return score_factor + keyword_factor + capability_factor + clarity_factor

    def _generate_rationale(
        self, scores: dict[str, float], keywords: set[str], capabilities: set[str]
    ) -> str:
        """Generate human-readable rationale for category scores."""
        top_categories = sorted(
            [(cat, score) for cat, score in scores.items() if score > 2.0],
            key=lambda x: x[1],
            reverse=True,
        )[:3]

        if not top_categories:
            return "Low confidence: No clear requirement signals detected."

        rationale_parts = ["Detected requirements:"]
        for category, score in top_categories:
            rationale_parts.append(f"{category} ({score:.1f}/10)")

        if capabilities:
            rationale_parts.append(f"Dependencies: {', '.join(sorted(capabilities))}")

        if len(keywords) > 5:
            top_keywords = sorted(keywords)[:5]
            rationale_parts.append(f"Keywords: {', '.join(top_keywords)}")

        return " | ".join(rationale_parts)
