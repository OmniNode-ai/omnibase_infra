#!/usr/bin/env python3
"""
Configuration for Template Engine Loader.

Defines paths, patterns, and defaults for template discovery and loading.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True)
class TemplateEngineConfig:
    """
    Configuration for TemplateEngine.

    All values are immutable after initialization.
    """

    # === TEMPLATE DIRECTORY PATHS ===

    DEFAULT_TEMPLATE_ROOT: ClassVar[Path] = (
        Path(__file__).parent.parent / "templates" / "node_templates"
    )
    """Default root directory for node templates"""

    # === NODE TYPE MAPPING ===

    NODE_TYPE_DIRECTORIES: ClassVar[dict[str, str]] = {
        "effect": "effect",
        "compute": "compute",
        "reducer": "reducer",
        "orchestrator": "orchestrator",
    }
    """Mapping of node types to directory names"""

    # === FILE PATTERNS ===

    TEMPLATE_FILE_PATTERN: ClassVar[str] = "*.py"
    """Glob pattern for template files"""

    TEMPLATE_FILE_EXTENSIONS: ClassVar[tuple[str, ...]] = (".py",)
    """Valid template file extensions"""

    # === VERSION PATTERNS ===

    VERSION_PATTERN: ClassVar[str] = r"v\d+_\d+_\d+"
    """Regex pattern for version strings (e.g., v1_0_0)"""

    # === METADATA EXTRACTION ===

    METADATA_MARKERS: ClassVar[dict[str, str]] = {
        "description": r'"""([^"]+)"""',  # First docstring
        "author": r"#\s*Author:\s*(.+)",
        "version": r"#\s*Version:\s*(.+)",
        "tags": r"#\s*Tags:\s*(.+)",
    }
    """Patterns for extracting metadata from template files"""

    # === STUB DETECTION ===

    STUB_INDICATORS: ClassVar[tuple[str, ...]] = (
        "# IMPLEMENTATION REQUIRED",
        "# TODO:",
        "pass  # Stub",
        "raise NotImplementedError",
    )
    """Patterns that indicate a method stub needs implementation"""

    # === VALIDATION ===

    REQUIRED_METHODS: ClassVar[dict[str, tuple[str, ...]]] = {
        "effect": ("execute_effect",),
        "compute": ("execute_compute",),
        "reducer": ("execute_reduction",),
        "orchestrator": ("execute_orchestration",),
    }
    """Required methods for each node type"""

    # === ERROR HANDLING ===

    STRICT_VALIDATION: ClassVar[bool] = False
    """Whether to raise exceptions on validation failures"""

    WARN_ON_MISSING_METADATA: ClassVar[bool] = True
    """Whether to warn when metadata is missing"""


__all__ = ["TemplateEngineConfig"]
