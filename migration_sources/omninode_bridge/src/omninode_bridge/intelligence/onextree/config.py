"""
Configuration and constants for OnexTree.

Feature flags, performance tuning, and default settings.
"""

from pydantic_settings import BaseSettings


class OnexTreeConfig(BaseSettings):
    """
    Configuration with feature flags for OnexTree system.

    Core features are always enabled in standalone mode.
    Integration features are disabled until components available.
    """

    # Core features (always enabled in standalone)
    enable_filesystem_watcher: bool = True
    enable_mcp_server: bool = True

    # Integration features (disabled until components available)
    enable_unified_file_processor_integration: bool = False
    enable_kafka_event_publishing: bool = False
    enable_database_persistence: bool = False
    enable_batch_processing_integration: bool = False

    # Performance tuning
    tree_regeneration_debounce_seconds: float = 0.5
    max_index_memory_mb: int = 50
    enable_lru_cache: bool = True
    lru_cache_size: int = 1000

    # Tree generation settings
    hasher_pool_size: int = 100
    max_concurrent_operations: int = 10

    # MCP server settings
    mcp_server_port: int = 8060

    class Config:
        env_prefix = "ONEXTREE_"


# Default exclusion patterns for tree generation
DEFAULT_EXCLUDE_PATTERNS: list[str] = [
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "*.pyc",
    ".DS_Store",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "*.egg-info",
    "dist",
    "build",
    ".tox",
    "coverage",
    ".coverage",
    "htmlcov",
    "*.swp",
    "*.swo",
    "*~",
    ".tmp",
]

# Performance thresholds for validation
PERFORMANCE_THRESHOLDS = {
    "tree_generation_ms": 100,  # < 100ms for 10K files
    "exact_lookup_ms": 1,  # < 1ms for exact path lookup
    "extension_search_ms": 5,  # < 5ms for extension-based search
    "index_rebuild_ms": 100,  # < 100ms for full index rebuild
    "watcher_latency_ms": 1000,  # < 1s for file change detection
    "memory_mb_per_10k_files": 20,  # < 20MB for 10K files
}
