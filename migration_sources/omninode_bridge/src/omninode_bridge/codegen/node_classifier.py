#!/usr/bin/env python3
"""
Node Classifier for ONEX Code Generation.

Classifies requirements into ONEX v2.0 node types and selects appropriate templates.

ONEX v2.0 Node Types:
- Effect: External I/O, APIs, side effects (databases, HTTP, files)
- Compute: Pure transformations, algorithms, calculations
- Reducer: Aggregation, accumulation, state management
- Orchestrator: Workflow coordination, multi-step processes

Classification Strategy:
1. Keyword-based classification (fast, 90% accuracy)
2. Pattern matching against known node characteristics
3. Confidence scoring (0.0-1.0)
4. Template selection based on node type and domain
"""

from enum import Enum
from typing import Any, ClassVar, Optional

from pydantic import BaseModel, Field

from .prd_analyzer import ModelPRDRequirements


class EnumNodeType(str, Enum):
    """ONEX v2.0 Node Types."""

    EFFECT = "effect"
    COMPUTE = "compute"
    REDUCER = "reducer"
    ORCHESTRATOR = "orchestrator"


class ModelClassificationResult(BaseModel):
    """
    Result of node type classification.

    Contains node type, confidence score, and template selection metadata.
    """

    # Classification
    node_type: EnumNodeType = Field(..., description="Classified node type")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Classification confidence"
    )

    # Template selection
    template_name: str = Field(..., description="Selected template name")
    template_variant: Optional[str] = Field(
        None, description="Template variant (e.g., 'async', 'pooled')"
    )

    # Classification reasoning
    primary_indicators: list[str] = Field(
        default_factory=list, description="Primary classification indicators"
    )
    secondary_indicators: list[str] = Field(
        default_factory=list, description="Secondary classification indicators"
    )

    # Alternative classifications
    alternatives: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Alternative node types with confidence scores",
    )


class NodeClassifier:
    """
    Classifies requirements into ONEX node types and selects templates.

    Uses multi-factor classification:
    - Domain analysis (database → Effect, math → Compute)
    - Operation analysis (aggregate → Reducer, coordinate → Orchestrator)
    - Dependency analysis (external services → Effect, pure functions → Compute)
    - Feature analysis (connection pooling → Effect, parallel execution → Orchestrator)
    """

    # Node type characteristics for classification
    NODE_CHARACTERISTICS: ClassVar[dict[EnumNodeType, dict[str, Any]]] = {
        EnumNodeType.EFFECT: {
            "keywords": [
                # I/O operations
                "database",
                "api",
                "http",
                "request",
                "query",
                "crud",
                "insert",
                "update",
                "delete",
                # External services
                "kafka",
                "redis",
                "s3",
                "storage",
                "file",
                "socket",
                # Side effects
                "write",
                "persist",
                "save",
                "fetch",
                "retrieve",
                "send",
                "publish",
            ],
            "domains": ["database", "api", "messaging", "storage", "cache"],
            "operations": [
                "create",
                "read",
                "update",
                "delete",
                "send",
                "receive",
                "publish",
                "subscribe",
            ],
            "features": [
                "connection_pooling",
                "retry_logic",
                "circuit_breaker",
                "rate_limiting",
            ],
        },
        EnumNodeType.COMPUTE: {
            "keywords": [
                # Transformations
                "transform",
                "convert",
                "parse",
                "format",
                "encode",
                "decode",
                "serialize",
                "deserialize",
                # Calculations
                "calculate",
                "compute",
                "process",
                "algorithm",
                "hash",
                "encrypt",
                "decrypt",
                # Pure functions
                "pure",
                "stateless",
                "functional",
                "map",
                "filter",
            ],
            "domains": ["ml", "general"],
            "operations": [
                "transform",
                "calculate",
                "compute",
                "parse",
                "validate",
                "process",
            ],
            "features": ["validation", "caching"],
        },
        EnumNodeType.REDUCER: {
            "keywords": [
                # Aggregation
                "aggregate",
                "reduce",
                "collect",
                "accumulate",
                "group",
                "summarize",
                "combine",
                "merge",
                # State management
                "state",
                "accumulator",
                "counter",
                "metrics",
                "statistics",
                "rollup",
            ],
            "domains": ["monitoring"],
            "operations": [
                "aggregate",
                "reduce",
                "collect",
                "group",
                "summarize",
                "accumulate",
            ],
            "features": ["metrics", "logging"],
        },
        EnumNodeType.ORCHESTRATOR: {
            "keywords": [
                # Coordination
                "orchestrate",
                "coordinate",
                "workflow",
                "pipeline",
                "stages",
                "steps",
                # Multi-process
                "multi-step",
                "sequential",
                "parallel",
                "concurrent",
                "routing",
                "dispatch",
                # Management
                "manage",
                "control",
                "schedule",
                "execute",
            ],
            "domains": [],  # Can apply to any domain
            "operations": [
                "orchestrate",
                "coordinate",
                "execute",
                "route",
                "dispatch",
            ],
            "features": ["circuit_breaker", "retry_logic"],
        },
    }

    def classify(self, requirements: ModelPRDRequirements) -> ModelClassificationResult:
        """
        Classify requirements into ONEX node type.

        Args:
            requirements: Extracted PRD requirements

        Returns:
            ModelClassificationResult with node type and confidence

        Example:
            >>> classifier = NodeClassifier()
            >>> reqs = ModelPRDRequirements(
            ...     node_type="effect",  # May be initial guess
            ...     service_name="postgres_crud",
            ...     domain="database",
            ...     operations=["create", "read", "update", "delete"],
            ...     business_description="PostgreSQL CRUD operations"
            ... )
            >>> result = classifier.classify(reqs)
            >>> assert result.node_type == EnumNodeType.EFFECT
            >>> assert result.confidence > 0.9
        """
        # If requirements already specify node type with high confidence, use it
        if requirements.extraction_confidence > 0.9 and requirements.node_type in [
            nt.value for nt in EnumNodeType
        ]:
            node_type = EnumNodeType(requirements.node_type)
            return self._build_result(
                node_type=node_type,
                confidence=requirements.extraction_confidence,
                requirements=requirements,
                primary_indicators=["Explicit node type specification"],
            )

        # Multi-factor classification
        scores = {node_type: 0.0 for node_type in EnumNodeType}
        indicators = {node_type: [] for node_type in EnumNodeType}

        # Factor 1: Domain analysis (weight: 30%)
        domain_scores, domain_indicators = self._analyze_domain(requirements)
        for node_type, score in domain_scores.items():
            scores[node_type] += score * 0.3
            indicators[node_type].extend(domain_indicators[node_type])

        # Factor 2: Operation analysis (weight: 30%)
        op_scores, op_indicators = self._analyze_operations(requirements)
        for node_type, score in op_scores.items():
            scores[node_type] += score * 0.3
            indicators[node_type].extend(op_indicators[node_type])

        # Factor 3: Keyword analysis (weight: 25%)
        keyword_scores, keyword_indicators = self._analyze_keywords(requirements)
        for node_type, score in keyword_scores.items():
            scores[node_type] += score * 0.25
            indicators[node_type].extend(keyword_indicators[node_type])

        # Factor 4: Feature analysis (weight: 15%)
        feature_scores, feature_indicators = self._analyze_features(requirements)
        for node_type, score in feature_scores.items():
            scores[node_type] += score * 0.15
            indicators[node_type].extend(feature_indicators[node_type])

        # Select node type with highest score
        best_node_type = max(scores, key=scores.get)
        confidence = scores[best_node_type]

        # Build alternatives list
        alternatives = [
            {"node_type": nt.value, "confidence": scores[nt]}
            for nt in EnumNodeType
            if nt != best_node_type and scores[nt] > 0.3
        ]
        alternatives.sort(key=lambda x: x["confidence"], reverse=True)

        return self._build_result(
            node_type=best_node_type,
            confidence=confidence,
            requirements=requirements,
            primary_indicators=indicators[best_node_type][:3],
            alternatives=alternatives,
        )

    def _analyze_domain(
        self, requirements: ModelPRDRequirements
    ) -> tuple[dict[EnumNodeType, float], dict[EnumNodeType, list[str]]]:
        """Analyze domain to determine node type."""
        scores = {node_type: 0.0 for node_type in EnumNodeType}
        indicators = {node_type: [] for node_type in EnumNodeType}

        domain = requirements.domain.lower()

        for node_type, characteristics in self.NODE_CHARACTERISTICS.items():
            if domain in characteristics["domains"]:
                scores[node_type] = 1.0
                indicators[node_type].append(f"Domain match: {domain}")

        return scores, indicators

    def _analyze_operations(
        self, requirements: ModelPRDRequirements
    ) -> tuple[dict[EnumNodeType, float], dict[EnumNodeType, list[str]]]:
        """Analyze operations to determine node type."""
        scores = {node_type: 0.0 for node_type in EnumNodeType}
        indicators = {node_type: [] for node_type in EnumNodeType}

        operations = [op.lower() for op in requirements.operations]

        for node_type, characteristics in self.NODE_CHARACTERISTICS.items():
            matching_ops = [
                op for op in operations if op in characteristics["operations"]
            ]
            if matching_ops:
                score = len(matching_ops) / len(operations) if operations else 0.0
                scores[node_type] = score
                indicators[node_type].append(
                    f"Operations match: {', '.join(matching_ops)}"
                )

        return scores, indicators

    def _analyze_keywords(
        self, requirements: ModelPRDRequirements
    ) -> tuple[dict[EnumNodeType, float], dict[EnumNodeType, list[str]]]:
        """Analyze business description keywords."""
        scores = {node_type: 0.0 for node_type in EnumNodeType}
        indicators = {node_type: [] for node_type in EnumNodeType}

        text = (
            requirements.business_description.lower()
            + " "
            + " ".join(requirements.features).lower()
        )

        for node_type, characteristics in self.NODE_CHARACTERISTICS.items():
            matching_keywords = [kw for kw in characteristics["keywords"] if kw in text]
            if matching_keywords:
                # Score based on number of matches
                score = min(len(matching_keywords) / 3.0, 1.0)
                scores[node_type] = score
                indicators[node_type].append(
                    f"Keywords: {', '.join(matching_keywords[:3])}"
                )

        return scores, indicators

    def _analyze_features(
        self, requirements: ModelPRDRequirements
    ) -> tuple[dict[EnumNodeType, float], dict[EnumNodeType, list[str]]]:
        """Analyze requested features."""
        scores = {node_type: 0.0 for node_type in EnumNodeType}
        indicators = {node_type: [] for node_type in EnumNodeType}

        features = [f.lower() for f in requirements.features]

        for node_type, characteristics in self.NODE_CHARACTERISTICS.items():
            matching_features = [
                f for f in features if f in characteristics["features"]
            ]
            if matching_features:
                score = len(matching_features) / len(features) if features else 0.0
                scores[node_type] = score
                indicators[node_type].append(
                    f"Features: {', '.join(matching_features)}"
                )

        return scores, indicators

    def _build_result(
        self,
        node_type: EnumNodeType,
        confidence: float,
        requirements: ModelPRDRequirements,
        primary_indicators: list[str],
        alternatives: Optional[list[dict[str, Any]]] = None,
    ) -> ModelClassificationResult:
        """Build classification result with template selection."""
        # Select template based on node type and domain
        template_name = self._select_template(node_type, requirements)

        # Select template variant based on features
        template_variant = self._select_template_variant(node_type, requirements)

        return ModelClassificationResult(
            node_type=node_type,
            confidence=confidence,
            template_name=template_name,
            template_variant=template_variant,
            primary_indicators=primary_indicators,
            alternatives=alternatives or [],
        )

    def _select_template(
        self, node_type: EnumNodeType, requirements: ModelPRDRequirements
    ) -> str:
        """Select appropriate template based on node type and domain."""
        domain = requirements.domain.lower()

        # Template naming: {node_type}_{domain} or {node_type}_generic
        template_map: dict[EnumNodeType, dict[str, str]] = {
            EnumNodeType.EFFECT: {
                "database": "effect_database",
                "api": "effect_api",
                "messaging": "effect_messaging",
                "default": "effect_generic",
            },
            EnumNodeType.COMPUTE: {
                "default": "compute_generic",
            },
            EnumNodeType.REDUCER: {
                "monitoring": "reducer_metrics",
                "default": "reducer_generic",
            },
            EnumNodeType.ORCHESTRATOR: {
                "default": "orchestrator_workflow",
            },
        }

        # Get templates for node type
        type_templates = template_map.get(node_type, {})
        return type_templates.get(
            domain, type_templates.get("default", f"{node_type.value}_generic")
        )

    def _select_template_variant(
        self, node_type: EnumNodeType, requirements: ModelPRDRequirements
    ) -> Optional[str]:
        """Select template variant based on features."""
        features = [f.lower() for f in requirements.features]

        # Effect variants
        if node_type == EnumNodeType.EFFECT:
            if "connection_pooling" in features:
                return "pooled"
            elif "circuit_breaker" in features:
                return "resilient"

        # Orchestrator variants
        elif node_type == EnumNodeType.ORCHESTRATOR:
            if "parallel" in " ".join(features).lower():
                return "parallel"
            else:
                return "sequential"

        # Reducer variants
        elif node_type == EnumNodeType.REDUCER:
            if "metrics" in features:
                return "metrics"

        return None


# Export
__all__ = ["NodeClassifier", "EnumNodeType", "ModelClassificationResult"]
