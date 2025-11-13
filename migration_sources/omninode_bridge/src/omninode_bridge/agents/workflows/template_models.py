"""
Template models for code generation workflows.

Defines Template dataclass, enums, and supporting models for template management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class TemplateType(str, Enum):
    """
    Template types for different node categories and components.

    ONEX v2.0 node types:
    - EFFECT: External I/O operations (APIs, DB, files)
    - COMPUTE: Pure business logic and data processing
    - REDUCER: State aggregation and persistence
    - ORCHESTRATOR: Workflow coordination

    Supporting types:
    - MODEL: Pydantic models
    - VALIDATOR: Validation logic
    - TEST: Test cases
    - CONTRACT: Contract YAML files
    """

    # ONEX v2.0 node types
    EFFECT = "effect"
    COMPUTE = "compute"
    REDUCER = "reducer"
    ORCHESTRATOR = "orchestrator"

    # Supporting types
    MODEL = "model"
    VALIDATOR = "validator"
    TEST = "test"
    CONTRACT = "contract"

    def __str__(self) -> str:
        """String representation."""
        return self.value


@dataclass
class TemplateMetadata:
    """
    Template metadata for tracking and organization.

    Attributes:
        version: Template version (semantic versioning)
        author: Template author/creator
        description: Template description
        tags: Tags for categorization and search
        node_type_category: Node type category (effect, compute, reducer, orchestrator)
        supports_batch: Whether template supports batch operations
        min_python_version: Minimum Python version required
        dependencies: Required dependencies for template
    """

    version: str = "1.0.0"
    author: str = "OmniNode Code Generator"
    description: str = ""
    tags: list[str] = field(default_factory=list)
    node_type_category: Optional[str] = None
    supports_batch: bool = False
    min_python_version: str = "3.11"
    dependencies: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "tags": self.tags,
            "node_type_category": self.node_type_category,
            "supports_batch": self.supports_batch,
            "min_python_version": self.min_python_version,
            "dependencies": self.dependencies,
        }


@dataclass
class Template:
    """
    Template representation with content and metadata.

    Performance targets:
    - Memory per template: 5-20KB (typical)
    - Load time (uncached): 10-50ms
    - Load time (cached): <1ms

    Attributes:
        template_id: Unique template identifier (e.g., "node_effect_v1")
        template_type: Template type (effect, compute, reducer, etc.)
        content: Jinja2 template content
        metadata: Template metadata
        loaded_at: Timestamp when template was loaded
        size_bytes: Template size in bytes
        file_path: Optional file path for template source
    """

    template_id: str
    template_type: TemplateType
    content: str
    metadata: TemplateMetadata
    loaded_at: datetime = field(default_factory=datetime.utcnow)
    size_bytes: int = field(init=False)
    file_path: Optional[str] = None

    def __post_init__(self) -> None:
        """Calculate template size after initialization."""
        self.size_bytes = len(self.content.encode("utf-8"))

    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary."""
        return {
            "template_id": self.template_id,
            "template_type": str(self.template_type),
            "content_length": len(self.content),
            "metadata": self.metadata.to_dict(),
            "loaded_at": self.loaded_at.isoformat(),
            "size_bytes": self.size_bytes,
            "file_path": self.file_path,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Template(id={self.template_id}, "
            f"type={self.template_type}, "
            f"size={self.size_bytes}B)"
        )


@dataclass
class TemplateRenderContext:
    """
    Context for template rendering with Jinja2.

    Provides variables and configuration for template rendering.

    Attributes:
        variables: Variables to inject into template
        filters: Custom Jinja2 filters
        globals: Global variables available in all templates
        autoescape: Enable HTML/XML autoescaping
        trim_blocks: Trim trailing newline after block
        lstrip_blocks: Strip leading spaces before block
    """

    variables: Dict[str, Any] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)
    globals: Dict[str, Any] = field(default_factory=dict)
    autoescape: bool = False
    trim_blocks: bool = True
    lstrip_blocks: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "variables": self.variables,
            "filters": list(self.filters.keys()),
            "globals": list(self.globals.keys()),
            "autoescape": self.autoescape,
            "trim_blocks": self.trim_blocks,
            "lstrip_blocks": self.lstrip_blocks,
        }


@dataclass
class TemplateCacheStats:
    """
    Template cache statistics.

    Tracks cache performance and hit rates.

    Attributes:
        total_requests: Total cache requests
        cache_hits: Number of cache hits
        cache_misses: Number of cache misses
        hit_rate: Cache hit rate (0.0-1.0)
        current_size: Current number of cached templates
        max_size: Maximum cache size
        evictions: Number of evictions
        total_size_bytes: Total size of cached templates in bytes
    """

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    hit_rate: float = 0.0
    current_size: int = 0
    max_size: int = 100
    evictions: int = 0
    total_size_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": self.hit_rate,
            "hit_rate_percent": f"{self.hit_rate * 100:.2f}%",
            "current_size": self.current_size,
            "max_size": self.max_size,
            "evictions": self.evictions,
            "total_size_bytes": self.total_size_bytes,
            "avg_template_size_bytes": (
                self.total_size_bytes // self.current_size if self.current_size > 0 else 0
            ),
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TemplateCacheStats(hit_rate={self.hit_rate:.2%}, "
            f"size={self.current_size}/{self.max_size}, "
            f"evictions={self.evictions})"
        )
