#!/usr/bin/env python3
"""
Template Engine Loader for ONEX Code Generation.

Loads and manages ONEX node templates from directory structure.
Provides alternative to Jinja2-based generation for hand-written templates.

Features:
- Template discovery by node type and version
- Automatic stub detection using CodeInjector
- Metadata extraction from template files
- Integration with BusinessLogicGenerator

Example Usage:
    >>> from omninode_bridge.codegen.template_engine_loader import TemplateEngine
    >>> engine = TemplateEngine()
    >>>
    >>> # Discover templates
    >>> templates = engine.discover_templates()
    >>>
    >>> # Load specific template
    >>> artifacts = engine.load_template("effect", "v1_0_0")
    >>> print(f"Loaded {artifacts.get_stub_count()} stubs")
"""

from .config import TemplateEngineConfig
from .engine import TemplateEngine, TemplateEngineError
from .models import (
    ModelStubInfo,
    ModelTemplateArtifacts,
    ModelTemplateInfo,
    ModelTemplateMetadata,
)

__all__ = [
    # Engine
    "TemplateEngine",
    "TemplateEngineError",
    # Models
    "ModelTemplateArtifacts",
    "ModelTemplateInfo",
    "ModelTemplateMetadata",
    "ModelStubInfo",
    # Config
    "TemplateEngineConfig",
]
