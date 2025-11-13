"""
Pydantic models for production pattern metadata.

This module defines the data models used to represent production patterns,
their metadata, matching results, and examples.

Performance Target: <1ms per model instantiation
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class EnumPatternCategory(str, Enum):
    """
    Pattern categories for organization and filtering.

    Categories align with ONEX v2.0 architectural concerns:
    - Structure: Basic node structure and organization
    - Resilience: Fault tolerance and recovery
    - Observability: Monitoring and debugging
    - Security: Protection and compliance
    - Performance: Optimization and efficiency
    - Integration: External system connectivity
    - Configuration: Configuration loading and management
    """

    STRUCTURE = "structure"
    RESILIENCE = "resilience"
    OBSERVABILITY = "observability"
    SECURITY = "security"
    PERFORMANCE = "performance"
    INTEGRATION = "integration"
    CONFIGURATION = "configuration"


class EnumNodeType(str, Enum):
    """ONEX node types."""

    EFFECT = "effect"
    COMPUTE = "compute"
    REDUCER = "reducer"
    ORCHESTRATOR = "orchestrator"


class ModelPatternExample(BaseModel):
    """
    Example implementation of a pattern from production nodes.

    Attributes:
        node_name: Name of the node this example is from
        node_type: Type of node (effect/compute/reducer/orchestrator)
        code_snippet: Python code demonstrating the pattern
        description: What this example demonstrates
        file_path: Optional path to the source file
        line_range: Optional tuple of (start_line, end_line)
    """

    node_name: str = Field(..., description="Name of the example node")
    node_type: EnumNodeType = Field(..., description="Type of the node")
    code_snippet: str = Field(..., min_length=1, description="Python code example")
    description: str = Field(..., min_length=10, description="What this demonstrates")
    file_path: Optional[Path] = Field(None, description="Source file path")
    line_range: Optional[tuple[int, int]] = Field(
        None, description="Line range (start, end) in source file"
    )

    class Config:
        frozen = False
        str_strip_whitespace = True


class ModelPatternMetadata(BaseModel):
    """
    Metadata describing a production pattern.

    This is the core data structure for the pattern library. Each pattern
    represents a reusable solution extracted from production ONEX nodes.

    Attributes:
        pattern_id: Unique identifier (e.g., "circuit_breaker_v1")
        name: Human-readable name
        version: Pattern version (semver format)
        category: Pattern category
        description: What problem this pattern solves
        applicable_to: Which node types can use this pattern
        prerequisites: Required mixins, imports, or dependencies
        code_template: Jinja2 template for generating code
        configuration: Default configuration values
        examples: List of real implementations
        tags: Searchable tags for filtering
        complexity: Complexity level (1-5, where 1=simple, 5=complex)
        performance_impact: Performance characteristics
        created_at: When this pattern was added
        updated_at: Last modification timestamp
    """

    # Identity
    pattern_id: str = Field(
        ...,
        pattern=r"^[a-z0-9_]+_v\d+$",
        description="Unique pattern identifier with version",
    )
    name: str = Field(..., min_length=3, description="Human-readable pattern name")
    version: str = Field(
        ..., pattern=r"^\d+\.\d+\.\d+$", description="Semantic version (e.g., 1.0.0)"
    )

    # Classification
    category: EnumPatternCategory = Field(..., description="Pattern category")
    applicable_to: list[EnumNodeType] = Field(
        ..., min_length=1, description="Node types that can use this pattern"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Searchable tags (e.g., 'async', 'database', 'retry')",
    )

    # Description
    description: str = Field(
        ..., min_length=20, description="What problem this pattern solves"
    )
    use_cases: list[str] = Field(
        default_factory=list,
        description="Specific scenarios where this pattern applies",
    )

    # Technical Details
    prerequisites: list[str] = Field(
        default_factory=list, description="Required mixins, imports, or dependencies"
    )
    code_template: str = Field(
        ..., min_length=10, description="Jinja2 template or code snippet"
    )
    configuration: dict[str, Any] = Field(
        default_factory=dict, description="Default configuration values"
    )

    # Examples
    examples: list[ModelPatternExample] = Field(
        default_factory=list, description="Real implementations from production nodes"
    )

    # Metrics
    complexity: int = Field(
        default=2, ge=1, le=5, description="Complexity level (1=simple, 5=complex)"
    )
    performance_impact: dict[str, Any] = Field(
        default_factory=dict,
        description="Performance characteristics (latency, memory, CPU)",
    )

    # Metadata
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Pattern creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last modification timestamp"
    )

    @field_validator("tags", mode="before")
    @classmethod
    def normalize_tags(cls, v: list[str]) -> list[str]:
        """Normalize tags to lowercase and remove duplicates."""
        if not v:
            return []
        return sorted(set(tag.lower().strip() for tag in v if tag.strip()))

    @field_validator("prerequisites", mode="before")
    @classmethod
    def normalize_prerequisites(cls, v: list[str]) -> list[str]:
        """Remove duplicates and empty strings from prerequisites."""
        if not v:
            return []
        return sorted(set(prereq.strip() for prereq in v if prereq.strip()))

    def matches_requirements(
        self,
        node_type: EnumNodeType,
        required_features: set[str],
        threshold: float = 0.3,
    ) -> bool:
        """
        Check if this pattern matches the given requirements.

        Args:
            node_type: Target node type
            required_features: Set of required feature tags
            threshold: Minimum overlap ratio to consider a match

        Returns:
            True if pattern matches requirements
        """
        # Check node type compatibility
        if node_type not in self.applicable_to:
            return False

        # Calculate feature overlap
        pattern_features = set(self.tags)
        if not pattern_features or not required_features:
            return False

        overlap = len(pattern_features.intersection(required_features))
        ratio = overlap / len(pattern_features)

        return ratio >= threshold

    def calculate_match_score(
        self, node_type: EnumNodeType, required_features: set[str]
    ) -> float:
        """
        Calculate match score (0.0 to 1.0) for given requirements.

        Scoring algorithm:
        - Node type match: +0.3
        - Feature overlap: up to +0.7 based on ratio

        Args:
            node_type: Target node type
            required_features: Set of required feature tags

        Returns:
            Match score between 0.0 and 1.0
        """
        score = 0.0

        # Node type compatibility
        if node_type in self.applicable_to:
            score += 0.3
        else:
            return 0.0  # No match if node type incompatible

        # Feature overlap
        pattern_features = set(self.tags)
        if pattern_features and required_features:
            overlap = len(pattern_features.intersection(required_features))
            ratio = overlap / len(pattern_features)
            score += 0.7 * ratio

        return min(score, 1.0)

    class Config:
        frozen = False
        str_strip_whitespace = True
        json_schema_extra = {
            "example": {
                "pattern_id": "circuit_breaker_v1",
                "name": "Circuit Breaker Pattern",
                "version": "1.0.0",
                "category": "resilience",
                "applicable_to": ["effect", "orchestrator"],
                "tags": ["async", "fault-tolerance", "resilience"],
                "description": "Prevents cascading failures by tracking failure rates",
                "prerequisites": ["ModelCircuitBreaker from omnibase_core"],
                "code_template": "self._circuit_breaker = ModelCircuitBreaker(...)",
                "configuration": {"failure_threshold": 5, "recovery_timeout_ms": 60000},
            }
        }


class ModelPatternMatch(BaseModel):
    """
    Result of pattern matching operation.

    Represents a pattern that was matched against requirements, including
    the match score and rationale.

    Attributes:
        pattern: The matched pattern metadata
        score: Match confidence (0.0 to 1.0)
        rationale: Explanation of why this pattern matched
        matched_features: Features that contributed to the match
    """

    pattern: ModelPatternMetadata = Field(..., description="Matched pattern")
    score: float = Field(..., ge=0.0, le=1.0, description="Match confidence score")
    rationale: str = Field(..., min_length=10, description="Why this pattern matched")
    matched_features: list[str] = Field(
        default_factory=list, description="Features that matched requirements"
    )

    class Config:
        frozen = True


class ModelPatternQuery(BaseModel):
    """
    Query parameters for pattern search.

    Attributes:
        node_type: Target node type
        required_features: Required feature tags
        categories: Filter by categories
        min_score: Minimum match score threshold
        max_results: Maximum number of results to return
        exclude_complex: Exclude patterns with complexity > threshold
    """

    node_type: EnumNodeType = Field(..., description="Target node type")
    required_features: set[str] = Field(
        default_factory=set, description="Required feature tags"
    )
    categories: Optional[list[EnumPatternCategory]] = Field(
        None, description="Filter by categories"
    )
    min_score: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Minimum match score"
    )
    max_results: int = Field(
        default=10, ge=1, le=50, description="Maximum results to return"
    )
    exclude_complex: Optional[int] = Field(
        None, ge=1, le=5, description="Exclude patterns with complexity > this value"
    )

    class Config:
        frozen = False


class ModelPatternLibraryStats(BaseModel):
    """
    Statistics about the pattern library.

    Attributes:
        total_patterns: Total number of patterns
        patterns_by_category: Count by category
        patterns_by_node_type: Count by node type
        average_complexity: Average complexity score
        last_updated: When library was last updated
    """

    total_patterns: int = Field(..., ge=0, description="Total pattern count")
    patterns_by_category: dict[str, int] = Field(
        default_factory=dict, description="Pattern count by category"
    )
    patterns_by_node_type: dict[str, int] = Field(
        default_factory=dict, description="Pattern count by node type"
    )
    average_complexity: float = Field(
        default=0.0, ge=0.0, le=5.0, description="Average complexity score"
    )
    last_updated: datetime = Field(
        default_factory=datetime.utcnow, description="Last library update"
    )

    class Config:
        frozen = True
